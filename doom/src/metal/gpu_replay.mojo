# SPDX-License-Identifier: GPL-2.0-or-later (see ../COPYING)
# Translation of doomgeneric r_draw.c drawers: Copyright(C) 1993-1996 Id Software, Inc.;
# Copyright(C) 2005-2014 Simon Howard. Mojo port for this repo.
# gpu_replay.mojo -- X-DOOM-3: replay one frame's column/span draws on the GPU (Metal).
#
# shim/r_draw_gpu.c records every drawer call of a view render as a fixed-size
# record (12 x int32) and snapshots the bytes it reads into an arena. At the end
# of R_RenderPlayerView, mojo_gpu_flush builds per-column / per-row record lists,
# uploads screen + records + lists + arena, runs ONE dispatch of
# `replay_kernel`, and copies the 8-bit screen back.
#
# Ordering (bit-identity with the C drawers run call by call):
#  * every write to a pixel comes from a record covering its column, and the
#    only screen reads (fuzz: dest[+-SCREENWIDTH]) stay in the same column, so
#    columns are independent and only the order within a column matters;
#  * a non-fuzz write depends only on the record and the pixel, so in a column
#    without fuzz each pixel just takes the value of the LAST record (by
#    submission index) covering it: one thread per pixel, no ordering hazard;
#  * a column with fuzz is replayed by one threadgroup in submission order:
#    per-pixel last-writer for the stretch between fuzz records, then the fuzz
#    record serially (it reads the row it may have just written).
# Fixed point is the same 32-bit wraparound arithmetic as r_draw.c.
#
# Built only with -D DOOM_GPU and the MAX toolchain (needs max.gpu).

from std.ffi import external_call
from std.os import getenv
from std.time import perf_counter_ns
from std.memory import Layout, alloc, stack_allocation
from max.gpu import barrier, block_idx, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

comptime SCREENWIDTH = 320
comptime SCREENHEIGHT = 200
comptime NPIX = SCREENWIDTH * SCREENHEIGHT
comptime FRACBITS = 16
comptime FUZZTABLE = 50

# Must match shim/r_draw_gpu.c.
comptime REC = 12
comptime REC_CAP = 65536
comptime ARENA_CAP = 8 << 20
# Staging layout (host and device): [screen | records | lists | arena], one
# contiguous upload; OFF_ARENA bounds where the arena can start.
comptime OFF_REC = 65536
comptime OFF_ARENA = OFF_REC + (REC + 4) * 4 * REC_CAP + 8192
comptime NSTART = SCREENWIDTH + 4  # col_start[321], padded
comptime RSTART = SCREENHEIGHT + 4  # row_start[201], padded
comptime TOTAL = OFF_ARENA + ARENA_CAP

comptime K_COL = 0
comptime K_COL_LOW = 1
comptime K_TCOL = 2
comptime K_TCOL_LOW = 3
comptime K_FUZZ = 4
comptime K_FUZZ_LOW = 5
comptime K_SPAN = 6
comptime K_SPAN_LOW = 7

comptime BlockSize = 256  # = TILE_W * TILE_H; >= SCREENHEIGHT for the fuzz blocks

comptime BytePtr = MutPointer[UInt8, MutUntrackedOrigin]
comptime ImmBytePtr = ImmPointer[UInt8, ImmUntrackedOrigin]
comptime ImmIntPtr = ImmPointer[Int32, ImmUntrackedOrigin]


@always_inline
def _pixel(r: ImmIntPtr, x: Int, y: Int, arena: ImmBytePtr) -> Int32:
    # Value record r writes at (x, y), or -1 if it does not cover the pixel.
    # Non-fuzz records: the value depends only on the record and the pixel
    # (column: frac = frac0 + (y - row0) * step, the same int32 wraparound as
    # the drawer's repeated frac += step; span: position = start + i * step).
    var kind = r[unsafe_offset=0]
    var xlo = Int(r[unsafe_offset=1])
    var row0 = Int(r[unsafe_offset=3])
    var frac = r[unsafe_offset=5]
    var step = r[unsafe_offset=6]
    var src = Int(r[unsafe_offset=7])
    var cmap = Int(r[unsafe_offset=8])
    if kind <= K_TCOL_LOW:
        var j = y - row0
        if j < 0 or j >= Int(r[unsafe_offset=4]):
            return -1
        var f = frac + Int32(j) * step
        var c: UInt8
        if kind >= K_TCOL:  # translated: no &127 upstream
            c = arena[unsafe_offset=src + Int(f >> FRACBITS)]
            c = arena[unsafe_offset=Int(r[unsafe_offset=9]) + Int(c)]
        else:
            c = arena[unsafe_offset=src + Int((f >> FRACBITS) & 127)]
        return Int32(arena[unsafe_offset=cmap + Int(c)])
    # span (callers only pass spans of row y that cover x)
    var i2 = x - xlo
    if kind == K_SPAN_LOW:
        i2 >>= 1
    var pos = frac.cast[.uint32]() + UInt32(i2) * step.cast[.uint32]()
    var spot = ((pos >> 4) & 0x0FC0) | (pos >> 26)
    var c = arena[unsafe_offset=src + Int(spot)]
    return Int32(arena[unsafe_offset=cmap + Int(c)])




@always_inline
def _last_writer(
    x: Int,
    y: Int,
    recs: ImmIntPtr,
    arena: ImmBytePtr,
    col_idx: ImmIntPtr,
    c_lo: Int,
    c_hi: Int,
    row_idx: ImmIntPtr,
    s0: Int,
    s1: Int,
    lo: Int32,
    hi: Int32,
) -> Int32:
    # Value of the last non-fuzz record with index in (lo, hi) covering (x, y),
    # or -1. Candidates: column x's records col_idx[c_lo:c_hi] and row y's
    # spans row_idx[s0:s1]; both ascending, walked backwards and merged.
    var ci = c_hi - 1
    var si = s1 - 1
    while si >= s0 and row_idx[unsafe_offset=si] >= hi:
        si -= 1
    while True:
        var c_rec = col_idx[unsafe_offset=ci] if ci >= c_lo else Int32(-1)
        var s_rec = row_idx[unsafe_offset=si] if si >= s0 else Int32(-1)
        var r: ImmIntPtr
        if c_rec > s_rec:
            r = recs.unsafe_offset(Int(c_rec) * REC)
            ci -= 1
        else:
            if s_rec <= lo:
                return -1
            r = recs.unsafe_offset(Int(s_rec) * REC)
            si -= 1
            if x < Int(r[unsafe_offset=1]) or x > Int(r[unsafe_offset=2]):
                continue
        var v = _pixel(r, x, y, arena)
        if v >= 0:
            return v


comptime TILE_W = 32  # pixel blocks: 32 columns x 8 rows (one SIMD group per row)
comptime TILE_H = 8
comptime GRID_X = SCREENWIDTH // TILE_W
comptime GRID_Y = SCREENHEIGHT // TILE_H


def replay_kernel(
    screen: BytePtr,
    recs: ImmIntPtr,
    lists: ImmIntPtr,
    ncol: Int32,
    nspan: Int32,
    nfz: Int32,
    arena: ImmBytePtr,
    fuzzoffset: ImmIntPtr,
):
    # lists = col_start[NSTART] | col_fuzz[NSTART] | row_start[RSTART]
    #         | col_idx[ncol] | row_idx[nspan] | fz_cols[nfz]
    # (record indices = submission order, ascending within each list).
    #
    # Blocks with block_idx.y < GRID_Y: one thread per pixel of a 32x8 tile.
    # Every non-fuzz drawer write depends only on the record and the pixel,
    # so a pixel in a column without fuzz ends up with the value of the LAST
    # record covering it: found by walking its column's and row's lists
    # backwards. Columns with fuzz are left to the blocks below.
    #
    # Blocks with block_idx.y >= GRID_Y: one threadgroup per fuzz column
    # (thread = row), the column held in threadgroup memory. The column's
    # records are split at its fuzz records; each segment in between is
    # resolved per pixel as above, then thread 0 applies the fuzz record
    # serially (it reads the row above/below, possibly just written).
    var col_start = lists
    var col_fuzz = lists.unsafe_offset(NSTART)
    var row_start = lists.unsafe_offset(2 * NSTART)
    var col_idx = lists.unsafe_offset(2 * NSTART + RSTART)
    var row_idx = col_idx.unsafe_offset(Int(ncol))
    var fz_cols = row_idx.unsafe_offset(Int(nspan))
    var t = thread_idx.x
    if block_idx.y < GRID_Y:
        var x = block_idx.x * TILE_W + t % TILE_W
        var y = block_idx.y * TILE_H + t // TILE_W
        if col_fuzz[unsafe_offset=x] != 0:
            return
        var v = _last_writer(
            x, y, recs, arena,
            col_idx, Int(col_start[unsafe_offset=x]), Int(col_start[unsafe_offset=x + 1]),
            row_idx, Int(row_start[unsafe_offset=y]), Int(row_start[unsafe_offset=y + 1]),
            -1, Int32.MAX,
        )
        if v >= 0:
            screen[unsafe_offset=y * SCREENWIDTH + x] = UInt8(v)
        return

    var k = (block_idx.y - GRID_Y) * GRID_X + block_idx.x
    if k >= Int(nfz):
        return  # uniform for the whole block
    var x = Int(fz_cols[unsafe_offset=k])
    var y = t
    var tile = stack_allocation[256, UInt8, address_space=AddressSpace.SHARED]()
    if y < SCREENHEIGHT:
        tile[unsafe_offset=y] = screen[unsafe_offset=y * SCREENWIDTH + x]
    barrier()
    var c0 = Int(col_start[unsafe_offset=x])
    var c1 = Int(col_start[unsafe_offset=x + 1])
    var s0 = Int(row_start[unsafe_offset=min(y, SCREENHEIGHT)])
    var s1 = Int(row_start[unsafe_offset=min(y + 1, SCREENHEIGHT)])
    var seg = c0
    var lo = Int32(-1)
    while True:
        # next fuzz record of this column (same answer in every thread)
        var f = seg
        while f < c1:
            var kind = recs[unsafe_offset=Int(col_idx[unsafe_offset=f]) * REC]
            if kind == K_FUZZ or kind == K_FUZZ_LOW:
                break
            f += 1
        var hi = col_idx[unsafe_offset=f] if f < c1 else Int32.MAX
        if y < SCREENHEIGHT:
            var v = _last_writer(x, y, recs, arena, col_idx, seg, f, row_idx, s0, s1, lo, hi)
            if v >= 0:
                tile[unsafe_offset=y] = UInt8(v)
        barrier()
        if f >= c1:
            break
        if y == 0:
            var r = recs.unsafe_offset(Int(hi) * REC)
            var row = Int(r[unsafe_offset=3])
            var cmap = Int(r[unsafe_offset=8])
            var pos = Int(r[unsafe_offset=9])
            for _ in range(Int(r[unsafe_offset=4])):
                var o = Int(fuzzoffset[unsafe_offset=pos]) // SCREENWIDTH  # +-1 row
                var px = tile[unsafe_offset=row + o]
                tile[unsafe_offset=row] = arena[unsafe_offset=cmap + Int(px)]
                pos += 1
                if pos == FUZZTABLE:
                    pos = 0
                row += 1
        barrier()
        lo = hi
        seg = f + 1
    if y < SCREENHEIGHT:
        screen[unsafe_offset=y * SCREENWIDTH + x] = tile[unsafe_offset=y]


struct GpuState(Movable):
    var ctx: DeviceContext[]
    var dev: DeviceBuffer[DType.uint8]
    var host: HostBuffer[DType.uint8]
    var profile: Bool  # DOOM_GPU_PROFILE=1: sync after each stage and time it
    var t_ns: List[Int]  # lists, upload, kernel, download, kernel amortized

    def __init__(out self) raises:
        self.profile = getenv("DOOM_GPU_PROFILE") == "1"
        self.t_ns = [0, 0, 0, 0, 0, 0, 0]
        self.ctx = DeviceContext()
        self.dev = self.ctx.enqueue_create_buffer[DType.uint8](TOTAL)
        self.host = self.ctx.enqueue_create_host_buffer[DType.uint8](TOTAL)
        self.ctx.synchronize()


comptime StatePtr = MutPointer[GpuState, MutUntrackedOrigin]


@export
def mojo_gpu_init() abi("C") -> Int:
    # Returns an opaque state pointer (0 on failure).
    try:
        var a = alloc(Layout[GpuState](length=1))
        var p = a^.unsafe_leak()
        p.unsafe_write(GpuState())
        var st = StatePtr(unsafe_from_address=Int(p))
        # Load the kernel now, not inside the first timed frame; then time the
        # fixed cost of one frame's round trip with nothing to draw (upload
        # screen, empty dispatch, download screen, wait), for the report.
        var screen = List[UInt8](length=NPIX, fill=0)
        var z = List[Int32](length=64, fill=0)
        var sp = BytePtr(unsafe_from_address=Int(screen.unsafe_ptr()))
        var zp = ImmIntPtr(unsafe_from_address=Int(z.unsafe_ptr()))
        var zb = ImmBytePtr(unsafe_from_address=Int(z.unsafe_ptr()))
        _ = _flush(st, zp, 0, zb, 0, sp)
        comptime NB = 50
        var t0 = perf_counter_ns()
        for _ in range(NB):
            _ = _flush(st, zp, 0, zb, 0, sp)
        var us = Float64(perf_counter_ns() - t0) / NB / 1000
        print("GPU replay on", st[].ctx.device().get_name(), "api", st[].ctx.api(), "empty_flush_us", us)
        st[].t_ns = [0, 0, 0, 0, 0, 0, 0]
        return Int(p)
    except e:
        print("mojo_gpu_init failed:", e)
        return 0


def _build_lists(recs: ImmIntPtr, base: BytePtr, nrec: Int) -> Tuple[Int, Int, Int, Int]:
    # Written right after the records (see replay_kernel for the layout):
    # per screen column, the column/fuzz records touching it (Low ones touch
    # two columns) and whether any is fuzz; per row, the span records on it;
    # the list of fuzz columns. Counting sort keeps submission order.
    # Returns (bytes, ncol, nspan, nfz).
    # Reads the records from the recorder's (cached) memory; writes the lists
    # into the staging buffer, which may be write-combined: never read back.
    comptime IP = MutPointer[Int32, MutUntrackedOrigin]
    var lists = IP(unsafe_from_address=Int(base) + OFF_REC + nrec * REC * 4)
    var cs = List[Int32](length=NSTART, fill=0)
    var cf = List[Int32](length=NSTART, fill=0)
    var rs = List[Int32](length=RSTART, fill=0)
    var col_start = cs.unsafe_ptr()
    var col_fuzz = cf.unsafe_ptr()
    var row_start = rs.unsafe_ptr()
    var nspan = 0
    for i in range(nrec):
        var r = recs.unsafe_offset(i * REC)
        var kind = r[unsafe_offset=0]
        if kind >= K_SPAN:
            nspan += 1
            row_start[unsafe_offset=Int(r[unsafe_offset=3]) + 1] += 1
        else:
            for c in range(Int(r[unsafe_offset=1]), Int(r[unsafe_offset=2]) + 1):
                col_start[unsafe_offset=c + 1] += 1
                if kind == K_FUZZ or kind == K_FUZZ_LOW:
                    col_fuzz[unsafe_offset=c] = 1
    for c in range(SCREENWIDTH):
        col_start[unsafe_offset=c + 1] += col_start[unsafe_offset=c]
    for y in range(SCREENHEIGHT):
        row_start[unsafe_offset=y + 1] += row_start[unsafe_offset=y]
    var ncol = Int(col_start[unsafe_offset=SCREENWIDTH])
    for i in range(NSTART):
        lists[unsafe_offset=i] = col_start[unsafe_offset=i]
        lists[unsafe_offset=NSTART + i] = col_fuzz[unsafe_offset=i]
    for i in range(RSTART):
        lists[unsafe_offset=2 * NSTART + i] = row_start[unsafe_offset=i]
    var col_idx = lists.unsafe_offset(2 * NSTART + RSTART)
    var row_idx = col_idx.unsafe_offset(ncol)
    var fz_cols = row_idx.unsafe_offset(nspan)
    var nfz = 0
    for c in range(SCREENWIDTH):
        if col_fuzz[unsafe_offset=c] != 0:
            fz_cols[unsafe_offset=nfz] = Int32(c)
            nfz += 1
    # Fill, using the (already copied) start arrays as cursors.
    for i in range(nrec):
        var r = recs.unsafe_offset(i * REC)
        if r[unsafe_offset=0] >= K_SPAN:
            var y = Int(r[unsafe_offset=3])
            row_idx[unsafe_offset=Int(row_start[unsafe_offset=y])] = Int32(i)
            row_start[unsafe_offset=y] += 1
        else:
            for c in range(Int(r[unsafe_offset=1]), Int(r[unsafe_offset=2]) + 1):
                col_idx[unsafe_offset=Int(col_start[unsafe_offset=c])] = Int32(i)
                col_start[unsafe_offset=c] += 1
    return ((2 * NSTART + RSTART + ncol + nspan + nfz) * 4, ncol, nspan, nfz)


def _launch(
    ctx: DeviceContext[], d: Int, off_lists: Int, off_arena: Int, ncol: Int, nspan: Int, nfz: Int
) raises:
    ctx.enqueue_function[replay_kernel](
        BytePtr(unsafe_from_address=d),
        ImmIntPtr(unsafe_from_address=d + OFF_REC),
        ImmIntPtr(unsafe_from_address=d + off_lists),
        Int32(ncol),
        Int32(nspan),
        Int32(nfz),
        ImmBytePtr(unsafe_from_address=d + off_arena),
        ImmIntPtr(unsafe_from_address=d + off_arena),  # arena[0..200) = fuzzoffset[50]
        grid_dim=(GRID_X, GRID_Y + (nfz + GRID_X - 1) // GRID_X),
        block_dim=BlockSize,
    )


def _flush(
    st: StatePtr, recs: ImmIntPtr, nrec: Int, arena: ImmBytePtr, arena_used: Int, screen: BytePtr
) raises -> Int32:
    ref s = st[]
    var hp = s.host.unsafe_ptr()
    var dp = s.dev.unsafe_ptr()
    var t0 = perf_counter_ns()
    _ = external_call["memcpy", Int](hp, screen, NPIX)
    _ = external_call["memcpy", Int](hp.unsafe_offset(OFF_REC), recs, nrec * REC * 4)
    var lists = _build_lists(recs, hp, nrec)
    var off_lists = OFF_REC + nrec * REC * 4
    # The arena goes right after the lists, so one copy uploads everything.
    var off_arena = (off_lists + lists[0] + 15) & ~15
    _ = external_call["memcpy", Int](hp.unsafe_offset(off_arena), arena, arena_used)
    var t1 = perf_counter_ns()
    var total = off_arena + arena_used
    s.ctx.enqueue_copy(
        s.dev.create_sub_buffer[DType.uint8](0, total),
        s.host.create_sub_buffer[DType.uint8](0, total),
    )
    var t2 = t1
    if s.profile:
        s.ctx.synchronize()
        t2 = perf_counter_ns()
    var d = Int(dp)
    _launch(s.ctx, d, off_lists, off_arena, lists[1], lists[2], lists[3])
    var t3 = t2
    if s.profile:
        s.ctx.synchronize()
        t3 = perf_counter_ns()
    s.ctx.enqueue_copy(
        s.host.create_sub_buffer[DType.uint8](0, NPIX),
        s.dev.create_sub_buffer[DType.uint8](0, NPIX),
    )
    s.ctx.synchronize()
    if s.profile:
        var t4 = perf_counter_ns()
        s.t_ns[0] += t1 - t0
        s.t_ns[1] += t2 - t1
        s.t_ns[2] += t3 - t2
        s.t_ns[3] += t4 - t3
        # GPU time of the kernel alone: 16 back-to-back dispatches, one wait
        # (re-running it only disturbs the device copy of the screen, which is
        # re-uploaded by the next flush; the result was already copied back).
        comptime KREP = 16
        for _ in range(KREP):
            _launch(s.ctx, d, off_lists, off_arena, lists[1], lists[2], lists[3])
        s.ctx.synchronize()
        var kt = (perf_counter_ns() - t4) // KREP
        s.t_ns[4] += kt
        var fz = False
        for i in range(nrec):
            var k = recs[unsafe_offset=i * REC]
            if k == K_FUZZ or k == K_FUZZ_LOW:
                fz = True
        if fz:
            s.t_ns[5] += kt
            s.t_ns[6] += 1
    _ = external_call["memcpy", Int](screen, hp, NPIX)
    return 0


@export
def mojo_gpu_report(state: Int, flushes: Int32) abi("C"):
    var st = StatePtr(unsafe_from_address=state)
    if st[].profile and flushes > 0:
        var n = Float64(Int(flushes)) * 1000
        print(
            "GPU profile_us lists", Float64(st[].t_ns[0]) / n, "upload", Float64(st[].t_ns[1]) / n,
            "kernel", Float64(st[].t_ns[2]) / n, "download", Float64(st[].t_ns[3]) / n, "kernel_amortized_x16", Float64(st[].t_ns[4]) / n,
            "fuzz_frames", st[].t_ns[6], "kernel_amortized_fuzz_frames", Float64(st[].t_ns[5]) / Float64(max(st[].t_ns[6], 1)) / 1000,
        )


@export
def mojo_gpu_flush(
    state: Int, recs: ImmIntPtr, nrec: Int32, arena: ImmBytePtr, arena_used: Int32, screen: BytePtr
) abi("C") -> Int32:
    try:
        return _flush(StatePtr(unsafe_from_address=state), recs, Int(nrec), arena, Int(arena_used), screen)
    except e:
        print("mojo_gpu_flush failed:", e)
        return 1


def keep_exports():
    # Called (never at runtime) from host.mojo's main so the exports are emitted.
    _ = mojo_gpu_init()
    var zi = ImmIntPtr(unsafe_from_address=16)  # dummy; never dereferenced
    var zb = ImmBytePtr(unsafe_from_address=16)
    _ = mojo_gpu_flush(0, zi, 0, zb, 0, BytePtr(unsafe_from_address=16))
    mojo_gpu_report(0, 0)
