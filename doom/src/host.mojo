# SPDX-License-Identifier: GPL-2.0-or-later (see ../COPYING); links the doomgeneric engine.
# host.mojo -- X-DOOM-1: the DOOM platform layer and game loop, in Mojo.
#
# The engine (doomgeneric, C, GPL-2.0) is linked in as a static library with no
# platform backend. This file supplies that backend:
#   * DG_Init / DG_DrawFrame / DG_SleepMs / DG_GetTicksMs / DG_GetKey /
#     DG_SetWindowTitle as C-ABI exports (resolved by the linker, no shim),
#   * `main`, which owns the game loop (doomgeneric_Create + doomgeneric_Tick),
#   * the framebuffer (allocated here and installed into the engine),
#   * windowed output via SDL2 (called over FFI from Mojo),
#   * headless mode: deterministic virtual clock, PNG captures, SHA-256 frame
#     hashes, fps.
# The only other C we wrote is shim/dg_glue.c (a context-pointer slot, because
# Mojo has no module-level globals, plus an exit hook registration).

from std.ffi import external_call
from std.memory import Layout, alloc
from std.sys import argv, is_defined
from std.time import perf_counter_ns

import drawers  # X-DOOM-2 kernels (exported; called only when linked with r_draw_mojo.o)

# X-DOOM-3: built with -D DOOM_GPU (MAX toolchain) and linked with r_draw_gpu.o +
# r_main_gpu.o, the frame's column/span draws are replayed on the GPU (gpu_replay.mojo).
# CPU builds put src/nogpu (an empty stub) on the import path instead of src/metal.
import gpu_replay

comptime DOOM_GPU = is_defined["DOOM_GPU"]()

comptime RESX = 320
comptime RESY = 200
comptime FB_BYTES = RESX * RESY * 4

comptime BytePtr = MutPointer[UInt8, MutUntrackedOrigin]
comptime IntPtr = MutPointer[Int32, MutUntrackedOrigin]
comptime ArgvPtr = MutPointer[BytePtr, MutUntrackedOrigin]

# SDL2 constants (from SDL2 headers; checked with a C probe).
comptime SDL_INIT_VIDEO: UInt32 = 0x20
comptime SDL_WINDOWPOS_UNDEFINED: Int32 = 0x1FFF0000
comptime SDL_WINDOW_SHOWN: UInt32 = 0x4
comptime SDL_WINDOW_RESIZABLE: UInt32 = 0x20
comptime SDL_RENDERER_ACCELERATED: UInt32 = 0x2
comptime SDL_PIXELFORMAT_XRGB8888: UInt32 = 0x16161804
comptime SDL_TEXTUREACCESS_STREAMING: Int32 = 1
comptime SDL_QUIT: UInt32 = 0x100
comptime SDL_KEYDOWN: UInt32 = 0x300
comptime SDL_KEYUP: UInt32 = 0x301
comptime SDL_EVENT_SIZE = 56
comptime SDL_KEYSYM_SYM_OFFSET = 20


struct Host(Movable):
    var headless: Bool
    var virtual_clock: Bool
    var vclock_ms: UInt32
    var t0_ns: Int
    var fb: BytePtr
    var frame: Int
    var max_frames: Int
    var captures: List[Int]
    var outdir: String
    var label: String
    var hash_all_path: String
    var hash_lines: List[String]
    var capture_lines: List[String]
    var first_frame_ns: Int
    var last_frame_ns: Int
    var window: Int
    var renderer: Int
    var texture: Int
    var event: BytePtr
    var keys: List[UInt16]
    var injects: List[Int]  # (frame, doom key) pairs, flattened
    var finished: Bool
    var scale: Int

    def __init__(out self):
        self.headless = False
        self.virtual_clock = False
        self.vclock_ms = 0
        self.t0_ns = perf_counter_ns()
        self.fb = external_call["calloc", BytePtr](Int(FB_BYTES), Int(1))
        self.frame = 0
        self.max_frames = 0
        self.captures = []
        self.outdir = "results"
        self.label = "mojo-host"
        self.hash_all_path = ""
        self.hash_lines = []
        self.capture_lines = []
        self.first_frame_ns = 0
        self.last_frame_ns = 0
        self.window = 0
        self.renderer = 0
        self.texture = 0
        self.event = external_call["calloc", BytePtr](Int(SDL_EVENT_SIZE), Int(1))
        self.keys = []
        self.injects = []
        self.finished = False
        self.scale = 3


comptime HostPtr = MutPointer[Host, MutUntrackedOrigin]


def host() -> HostPtr:
    return HostPtr(unsafe_from_address=external_call["dg_glue_get_ctx", Int]())


# ----------------------------------------------------------------------------
# Hashing (CommonCrypto SHA-256) and PNG (zlib) helpers
# ----------------------------------------------------------------------------


def sha256_hex(data: BytePtr, n: Int) -> String:
    var md = List[UInt8](length=32, fill=0)
    _ = external_call["CC_SHA256", Int](data, UInt32(n), md.unsafe_ptr())
    comptime HEX = "0123456789abcdef"
    var s = String()
    for i in range(32):
        var b = Int(md[i])
        s += String(HEX[byte = b >> 4])
        s += String(HEX[byte = b & 15])
    return s^


def _be32(mut out: List[UInt8], v: UInt32):
    out.append(UInt8((v >> 24) & 0xFF))
    out.append(UInt8((v >> 16) & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8(v & 0xFF))


def _chunk(mut png: List[UInt8], tag: StaticString, body: List[UInt8]):
    _be32(png, UInt32(len(body)))
    var start = len(png)
    for b in tag.as_bytes():
        png.append(b)
    for b in body:
        png.append(b)
    var crc = external_call["crc32", UInt](
        UInt(0), png.unsafe_ptr().unsafe_offset(start), UInt32(len(png) - start)
    )
    _be32(png, UInt32(crc))


def write_png(path: String, fb: BytePtr) raises:
    # fb: RESX*RESY little-endian 0x00RRGGBB words -> 8-bit RGB PNG.
    var raw: List[UInt8] = []
    raw.reserve(RESY * (RESX * 3 + 1))
    for y in range(RESY):
        raw.append(0)  # filter: none
        for x in range(RESX):
            var p = fb.unsafe_offset((y * RESX + x) * 4)
            raw.append(p[unsafe_offset=2])
            raw.append(p[unsafe_offset=1])
            raw.append(p[unsafe_offset=0])
    var cap = len(raw) + len(raw) // 100 + 1024
    var z = List[UInt8](length=cap, fill=0)
    var zlen = UInt(cap)
    var rc = external_call["compress2", Int32](
        z.unsafe_ptr(), Pointer(to=zlen), raw.unsafe_ptr(), UInt(len(raw)), Int32(9)
    )
    if rc != 0:
        raise Error("compress2 failed")
    z.resize(Int(zlen), 0)
    var png: List[UInt8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
    var ihdr: List[UInt8] = []
    _be32(ihdr, RESX)
    _be32(ihdr, RESY)
    ihdr.extend([UInt8(8), 2, 0, 0, 0])  # 8-bit, RGB, deflate, no filter, no interlace
    _chunk(png, "IHDR", ihdr)
    _chunk(png, "IDAT", z)
    _chunk(png, "IEND", [])
    with open(path, "w") as f:
        f.write_bytes(Span(png))


# ----------------------------------------------------------------------------
# doomgeneric platform callbacks (C ABI, resolved at link time)
# ----------------------------------------------------------------------------


@export
def DG_Init() abi("C"):
    var h = host()
    # Install the Mojo-owned framebuffer in place of the one the engine malloc'd.
    var slot = external_call["dg_glue_screen_buffer_slot", MutPointer[BytePtr, MutUntrackedOrigin]]()
    external_call["free", NoneType](slot[])
    slot[] = h[].fb
    external_call["dg_glue_register_exit_hook", NoneType]()
    comptime if DOOM_GPU:
        external_call["r_gpu_init", NoneType]()  # Metal device + kernel load, before frame 1
    if h[].headless:
        return
    if external_call["SDL_Init", Int32](SDL_INIT_VIDEO) != 0:
        print("SDL_Init failed")
        return
    h[].window = external_call["SDL_CreateWindow", Int](
        c_string("DOOM (Mojo host)"),
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        Int32(RESX * h[].scale),
        Int32(RESY * h[].scale),
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE,
    )
    h[].renderer = external_call["SDL_CreateRenderer", Int](
        h[].window, Int32(-1), SDL_RENDERER_ACCELERATED
    )
    h[].texture = external_call["SDL_CreateTexture", Int](
        h[].renderer,
        SDL_PIXELFORMAT_XRGB8888,
        SDL_TEXTUREACCESS_STREAMING,
        Int32(RESX),
        Int32(RESY),
    )


def doom_key(sym: Int32) -> UInt8:
    # SDL keycode -> doomkeys.h (same table as upstream doomgeneric_sdl.c).
    if sym == 13:
        return 13  # KEY_ENTER
    if sym == 27:
        return 27  # KEY_ESCAPE
    if sym == 0x40000050:
        return 0xAC  # KEY_LEFTARROW
    if sym == 0x4000004F:
        return 0xAE  # KEY_RIGHTARROW
    if sym == 0x40000052:
        return 0xAD  # KEY_UPARROW
    if sym == 0x40000051:
        return 0xAF  # KEY_DOWNARROW
    if sym == 0x400000E0 or sym == 0x400000E4:
        return 0xA3  # KEY_FIRE (ctrl)
    if sym == 32:
        return 0xA2  # KEY_USE (space)
    if sym == 0x400000E1 or sym == 0x400000E5:
        return 0xB6  # KEY_RSHIFT
    if sym == 0x400000E2 or sym == 0x400000E6:
        return 0xB8  # KEY_RALT
    if sym >= 0x4000003B and sym <= 0x40000043:  # F2..F10
        return UInt8(0x80 + 0x3C + (sym - 0x4000003B))
    if sym == 0x40000044:
        return 0x80 + 0x57  # F11
    if sym == 61 or sym == 43:
        return 61  # KEY_EQUALS
    if sym == 45:
        return 45  # KEY_MINUS
    if sym >= 65 and sym <= 90:
        return UInt8(sym + 32)  # tolower
    return UInt8(sym & 0xFF)


def le32(p: BytePtr, off: Int) -> Int32:
    var v: UInt32 = 0
    for k in range(4):
        v |= UInt32(p[unsafe_offset=off + k]) << UInt32(8 * k)
    return v.cast[.int32]()


def pump_events(h: HostPtr):
    var ev = h[].event
    while external_call["SDL_PollEvent", Int32](ev) != 0:
        var typ = le32(ev, 0).cast[.uint32]()
        if typ == SDL_QUIT:
            print("Quit requested")
            external_call["I_Quit", NoneType]()  # runs exit hooks, exits
        elif typ == SDL_KEYDOWN or typ == SDL_KEYUP:
            var sym = le32(ev, SDL_KEYSYM_SYM_OFFSET)
            var pressed = UInt16(1) if typ == SDL_KEYDOWN else UInt16(0)
            h[].keys.append((pressed << 8) | UInt16(doom_key(sym)))


def finish(h: HostPtr):
    # Called once: at --frames N, or from the engine's exit path.
    if h[].finished:
        return
    h[].finished = True
    var n = h[].frame
    var secs = Float64(h[].last_frame_ns - h[].first_frame_ns) / 1e9
    var fps = Float64(n - 1) / secs if n > 1 and secs > 0 else 0.0
    print(
        t'RESULT {{"label": "{h[].label}", "frames": {n}, "seconds": {secs}, "fps": {fps}, "headless": {"true" if h[].headless else "false"}}}'
    )
    for line in h[].capture_lines:
        print("CAPTURE", line, flush=True)
    if h[].hash_all_path.byte_length() > 0:
        try:
            with open(h[].hash_all_path, "w") as f:
                for line in h[].hash_lines:
                    f.write(line, "\n")
        except e:
            print("hash-all write failed:", e)
    if not h[].headless:
        external_call["SDL_Quit", NoneType]()
    _ = external_call["fflush", Int32](Int(0))


@export
def mojo_on_exit() abi("C"):
    finish(host())


@export
def DG_DrawFrame() abi("C"):
    var h = host()
    var now = perf_counter_ns()
    h[].frame += 1
    if h[].frame == 1:
        h[].first_frame_ns = now
    h[].last_frame_ns = now
    var f = h[].frame
    if not h[].headless:
        _ = external_call["SDL_UpdateTexture", Int32](
            h[].texture, Int(0), h[].fb, Int32(RESX * 4)
        )
        _ = external_call["SDL_RenderClear", Int32](h[].renderer)
        _ = external_call["SDL_RenderCopy", Int32](h[].renderer, h[].texture, Int(0), Int(0))
        external_call["SDL_RenderPresent", NoneType](h[].renderer)
        pump_events(h)
    if h[].hash_all_path.byte_length() > 0:
        h[].hash_lines.append(String(f, " ", sha256_hex(h[].fb, FB_BYTES)))
    for c in h[].captures:
        if c == f:
            var digest = sha256_hex(h[].fb, FB_BYTES)
            var png = String(h[].outdir, "/", h[].label, "-frame", f, ".png")
            try:
                write_png(png, h[].fb)
            except e:
                print("png write failed:", e)
            h[].capture_lines.append(String(f, " ", digest, " ", png))
    var k = 0
    while k + 1 < len(h[].injects):
        if h[].injects[k] == f:
            var key = UInt16(h[].injects[k + 1])
            h[].keys.append((1 << 8) | key)  # press
            h[].keys.append(key)  # release
        k += 2
    if h[].max_frames > 0 and f >= h[].max_frames:
        finish(h)
        external_call["exit", NoneType](Int32(0))


@export
def DG_SleepMs(ms: UInt32) abi("C"):
    var h = host()
    if h[].virtual_clock:
        h[].vclock_ms += ms
    else:
        _ = external_call["usleep", Int32](ms * 1000)


@export
def DG_GetTicksMs() abi("C") -> UInt32:
    var h = host()
    if h[].virtual_clock:
        return h[].vclock_ms
    return UInt32((perf_counter_ns() - h[].t0_ns) // 1_000_000)


@export
def DG_GetKey(pressed: IntPtr, key: BytePtr) abi("C") -> Int32:
    var h = host()
    if len(h[].keys) == 0:
        return 0
    var k = h[].keys.pop(0)
    pressed[] = Int32(k >> 8)
    key[] = UInt8(k & 0xFF)
    return 1


@export
def DG_SetWindowTitle(title: BytePtr) abi("C"):
    var h = host()
    if h[].window != 0:
        external_call["SDL_SetWindowTitle", NoneType](h[].window, title)


# ----------------------------------------------------------------------------
# main: owns the game loop
# ----------------------------------------------------------------------------


def c_string(s: StringSpan) -> BytePtr:
    var n = s.byte_length()
    var p = external_call["malloc", BytePtr](n + 1)
    var b = s.as_bytes()
    for i in range(n):
        p[unsafe_offset=i] = b[i]
    p[unsafe_offset=n] = 0
    return p


def usage():
    print(
        "doom host flags (Mojo): --headless --frames N --capture 1,100,500"
        " --out DIR --label NAME --hash-all FILE --inject F:KEY,..."
        " --scale N --vclock|--realclock --microbench;"
        " everything else goes to DOOM (e.g. -iwad X.wad -timedemo demo1)"
    )


def main() raises:
    var a = alloc(Layout[Host](length=1))
    var hp = a^.unsafe_leak()
    hp.unsafe_write(Host())
    var h = HostPtr(unsafe_from_address=Int(hp))
    external_call["dg_glue_set_ctx", NoneType](Int(hp))

    var args = argv()
    var doom_args: List[String] = [String(args[0])]
    var clock_set = False
    var i = 1
    while i < len(args):
        var s = String(args[i])
        if s == "--headless":
            h[].headless = True
        elif s == "--frames":
            i += 1
            h[].max_frames = Int(String(args[i]))
        elif s == "--capture":
            i += 1
            for part in String(args[i]).split(","):
                h[].captures.append(Int(String(part)))
        elif s == "--out":
            i += 1
            h[].outdir = String(args[i])
        elif s == "--label":
            i += 1
            h[].label = String(args[i])
        elif s == "--hash-all":
            i += 1
            h[].hash_all_path = String(args[i])
        elif s == "--inject":
            # --inject FRAME:KEY,... (doom key codes, e.g. 191=F5 detail toggle,
            # 45='-' smaller view): scripted key presses for headless runs.
            i += 1
            for part in String(args[i]).split(","):
                var fk = String(part).split(":")
                h[].injects.append(Int(String(fk[0])))
                h[].injects.append(Int(String(fk[1])))
        elif s == "--scale":
            i += 1
            h[].scale = Int(String(args[i]))
        elif s == "--vclock":
            h[].virtual_clock = True
            clock_set = True
        elif s == "--realclock":
            h[].virtual_clock = False
            clock_set = True
        elif s == "--microbench":
            # Drawer-only C vs Mojo timing (shim/drawbench.c); no game run.
            external_call["dg_drawer_microbench", NoneType]()
            return
        elif s == "--help":
            usage()
            return
        else:
            doom_args.append(s)
        i += 1
    if not clock_set:
        # Headless runs use a virtual clock so wipes/tics are reproducible.
        h[].virtual_clock = h[].headless

    var argc = len(doom_args)
    var cargv = external_call["calloc", ArgvPtr](Int(argc + 1), Int(8))
    for j in range(argc):
        cargv[unsafe_offset=j] = c_string(doom_args[j])

    if argc < 0:
        drawers.keep_exports()  # never runs; makes the tier-2 @export symbols get emitted
        gpu_replay.keep_exports()  # likewise for the tier-3 exports (no-op stub in CPU builds)
    external_call["doomgeneric_Create", NoneType](Int32(argc), cargv)
    while True:
        external_call["doomgeneric_Tick", NoneType]()

