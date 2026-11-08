# fpga/soc: RISC-V SoC simulation with the matmul tile as a bus peripheral

A LiteX SoC with a VexRiscv soft CPU, simulated cycle-accurately in Verilator on this Mac. The int8 matmul
tiles sit on the Wishbone bus as memory-mapped slaves. This is the step toward (a) a real ECP5 board and
(b) X-DOOM-4: DOOM frames produced by the soft CPU and checked against the host build.

| milestone | status | evidence |
|---|---|---|
| 1. LiteX BIOS boots, runs a C hello world | **pass** | BIOS banner, `Liftoff!`, firmware prints, checksum matches |
| 2. tile on the bus, random int8 matmul vs CPU reference | **pass** for v1 (own wrapper) and **tile2** (lane-tile2 RTL, INTERFACE.md v2.0) | 12/12 runs bit-exact; independent Python host check; negative check caught |
| 3. DOOM in the Verilator SoC, frame vs host | **pass**: frame 1 (title screen) and frame 400 (in-game DEMO1) have the host's SHA-256 and 0/64000 differing pixels; all 17 checkpoint frames on the way to 400 match | `scripts/compare_frame.py`, `results/` |

## The SoC (`soc.py`)

- CPU: VexRiscv "standard" (rv32im, 4 KiB I$/D$; prebuilt Verilog from `pythondata-cpu-vexriscv`, no Scala).
- 64 KiB ROM (LiteX BIOS), 64 KiB SRAM, main RAM (integrated, 16 MiB; 64 MiB for DOOM) at `0x40000000`.
- UART: LiteX sim PHY to stdout (`serial2console`).
- `simctrl` CSRs: 64-bit cycle counter (`latch` then read) and `finish` (Verilog `$finish`).
- Tiles (uncached IO regions):
  - `0x82000000` tile v1: `rtl/tile_v1_wb.v`, a small hand-written Wishbone wrapper around the generated
    `../rtl/systolic_n4_i8_a32.v` (register map in the RTL header). The CPU streams every (A,B) beat over
    the bus; the wrapper captures the drained rows.
  - `0x90000000` tile2: `../tile2/rtl/tile2_n8.v` (N=8, on-chip A/B/C buffers + sequencer), wired per
    `../tile2/INTERFACE.md` §8.
- `sys_clk` is nominally 1 MHz (only matters for BIOS delays). All numbers below are SoC cycles.

Build once, run many: Migen emits `$readmemh` for ROM and RAM, so `run.py` writes the firmware (and any
data blobs) into a sparse `.init` file in a per-run directory and starts the prebuilt `Vsim` there.
Changing firmware never rebuilds the Verilator model (Verilator build: ~5 s; SoC + BIOS + picolibc: ~15 s).

## Toolchain

LLVM 23.1.4 from conda-forge (pixi env) behind GCC-named shims in `tools/bin/` (LiteX's VexRiscv only
supports the GCC path; the shim maps `-march=rv32i2p0_m` to `rv32im_zicsr_zifencei` and links with lld).
LiteX + Migen + pythondata packages are pinned git commits installed into the uv venv `.venv` from local
checkouts in `third_party/src`. Pins, licenses and the reasoning are in [UPSTREAM.md](UPSTREAM.md).

## Reproduce

```sh
cd fpga/soc
pixi install
pixi run fetch       # pinned LiteX/Migen/VexRiscv/picolibc/compiler_rt/tapcfg sources -> .venv
pixi run soc         # build/sim: SoC + BIOS + Verilator model
pixi run hello       # milestone 1
pixi run tiletest    # milestone 2 (+ host check)
pixi run doom-soc    # build/doom: 64 MiB RAM, full picolibc (DOOM needs stdio/malloc)
pixi run doom-run    # milestone 3, frame 1 (~6.5 min wall); MAX_FRAME=400 HASH_EVERY=25 pixi run doom-run
```
DOOM needs `../../doom` fetched and built (`cd doom && pixi run fetch && pixi run build`) for the
doomgeneric sources, the Freedoom WAD and the host reference binary `doom/build/doom-c`.

Negative checks: `make -f fw/fw.mk APP=tiletest APP_COMMON='tile_v1.c tile_v2.c' APP_CFLAGS=-DCORRUPT OUT=$PWD/build/sim/fw/tiletest_corrupt`
then `run.py` on it flips one result bit after each tile run: firmware reports 12 mismatching runs,
`SIM_FINISH=1`, and `check_tiletest.py` prints `HOSTCHECK FAIL`.

## Milestone 1: boot + hello

```
LiteX git sha1: dff2494 ... CPU: VexRiscv @ 1MHz, BUS: wishbone 32-bit, MAIN RAM: 16.0MiB
Executing booted program at 0x40000000
--============== Liftoff! ==============--
hello from VexRiscv (fpga/soc)
checksum 0xee85ed14 (expect 0xee85ed14), 11863 cycles
SIM_FINISH 0
```
(The BIOS prints LiteX's own sha `dff2494` as its version string; the checkout is the `2026.08` tag commit
`b6ae9e0b`.) The BIOS jumps to main RAM via `ROM_BOOT_ADDRESS`; serial boot is disabled so it does not
wait for a timeout. Firmware: `fw/hello`, linked entirely into main RAM by `fw/linker.ld` with LiteX's
VexRiscv `crt0.S`, picolibc and libbase. Simulation speed: ~4-5 M SoC cycles per wall second.

## Milestone 2: tile tests (`fw/tiletest`, `fw/tile_v1.c`, `fw/tile_v2.c`)

Two shapes x 3 trials (2 random xorshift64*, 1 all -128 for the extreme -128·-128 accumulation), each run
on both tiles and compared element by element against the CPU reference on the soft core.
`scripts/check_tiletest.py` regenerates the same matrices in Python and checks the FNV-1a hash of every
reference C, so the soft-CPU reference is itself checked independently.

| shape (M×K×N) | CPU ref (cycles) | v1: total / tile busy / ideal | tile2: total = load + run + read-back; CYCLES reg = formula |
|---|---|---|---|
| 13×64×9 | 103,517 | 53,374 / 50,127 / 900 | 23,240 = 20,687 + 416 + 1,517 (+setup); CYCLES 354 = 354 |
| 8×256×32 | 2,090,499 | 398,603 / 392,983 / 4,272 | 130,685 = 126,011 + 1,181 + 2,996 (+setup); CYCLES 1,122 = 1,122 |

All 12 runs have 0 mismatches; HOSTCHECK PASS.

How to read this:
- **v1 is CPU-feed-bound.** Each beat is two uncached 32-bit stores plus the byte packing loop
  (~50 cycles/beat), so the tile's 3N-1 overhead is irrelevant: 392,983 busy cycles against an
  ideal-host 4,272. That is the expected result of driving a streaming tile from a soft CPU.
- **tile2 removes the CPU from the inner loop.** The run itself is exactly the INTERFACE.md §6 formula
  (`2 + tiles·(K + 24)`), with ~60 cycles of poll overhead. The 8×256×32 matmul is 16x faster
  end-to-end than the soft-CPU reference, and **96% of its time is loading A/B over the bus**
  (word stores, packing on the CPU). The next speedups are DMA from main RAM or reusing operands
  across runs (`ACCUM`, base offsets), not the array.

## Milestone 3: DOOM on the soft CPU (`fw/doom`)

doomgeneric (pinned commit, GPL, compiled **unmodified** from `doom/third_party/doomgeneric`) plus a
no-OS platform layer `fw/doom/dg_soc.c`:
- `DG_*` with the same virtual clock as `doom/src/host.mojo --headless --vclock` (`DG_SleepMs` advances
  it, nothing else does), so SoC frame N is the same game state as host frame N.
- POSIX shims under picolibc's stdio (`open/read/lseek/close`, ...): a read-only in-memory file system
  with one file, `freedoom1.wad`, which `run.py --wad` preloads into simulated RAM at `0x42000000`
  (16-byte header + 27.5 MiB). Writes (config, savegames) fail cleanly, `mkdir` is a no-op.
- `DG_DrawFrame` computes SHA-256 of the 320×200 XRGB8888 `DG_ScreenBuffer` (same bytes the host hashes),
  dumps the target frame as hex over the UART, and `$finish`es.
- Same command line as the host reference (`-iwad freedoom1.wad`, no `-timedemo`, no config file), same
  engine flags as `doom/scripts/build_c.sh`, plus `-fsigned-char` (host arm64 char is signed, RISC-V is not).
- Memory: code+data ~0.65 MiB, heap (zone 6 MiB + engine mallocs) up to `0x41ff0000`, WAD at
  `0x42000000`, stack at the top of the 64 MiB RAM.

The host reference is `doom/build/doom-c --headless --vclock --hash-all` run in a fresh directory with
the same arguments; its frame hashes are identical across three runs for frames 1-400. Without
`-timedemo` the game starts on the title page with no wipe, so frame 1 is deterministic (the X-DOOM-1
frame-1 caveat applies only to the timedemo melt from uninitialised memory). Frames 1-342 are the title
page; DEMO1 starts with a melt at frame 343; frame 400 is in-game.

### Result: frame 1 (title screen) — pass

```
FRAME 1 cycles=1825182473 vclock_ms=0 sha256=6a035fd96074ee458b5291726cb0a03bd8153affd20b1048ddc16af5fc70094f
frame 1: soc 6a035fd96074ee45.. host 6a035fd96074ee45.. MATCH
pixel compare vs host-frame1.png: 0 differing pixels of 64000
COMPARE PASS
[sim] simulated=1825.781s (SoC cycles at 1 MHz), elapsed=370.028s, performance=4.934x realtime
```
1.83 G SoC cycles from reset to the first frame (BIOS, zone/WAD init, `R_Init` texture/flat/sprite
setup, first `D_Display` and palette conversion), 6.2 minutes of Verilator wall time on the M5. No ISS
fast-forward was needed: the pass comes straight from the cycle-level SoC.

### Frame 400 (in-game DEMO1)

`MAX_FRAME=400 HASH_EVERY=25 pixi run doom-run`: SHA-256 printed at frames 1, 25, 50, ..., 400, the full
frame-400 framebuffer dumped over the UART.

```
frame 1..325 (14 checkpoints, title page): MATCH   6a035fd9...
frame 350 (melt into DEMO1):               MATCH   7c6f8f11...
frame 375 (in-game):                       MATCH   0661a25e...
frame 400 (in-game):                       MATCH   7e01a89cad3fead02248593612c784ffbd2b445077c779af0ab413d528ae1c1d
pixel compare vs host-frame400.png: 0 differing pixels of 64000
COMPARE PASS
DOOM_DONE frames=400 cycles=3824955362
[sim] simulated=3825.554s, elapsed=761.865s, performance=5.021x realtime
```
`results/soc-frame400.png` is the frame decoded from the UART dump (DEMO1 on E1M6: room, two monsters, status bar);
`results/doom-f400-frames.txt` has every checkpoint line. Costs on the soft CPU (SoC cycles): ~1.76 G to
the first frame, ~4.6 M per title frame, ~7.4 M per in-game frame (frames 375-400, including one SHA-256
per 25 frames), i.e. ~0.14 fps at a hypothetical 1 MHz, or ~7 fps if this core ran at 50 MHz on an ECP5
with single-cycle RAM (it will not have single-cycle RAM; SDRAM will cost more).

## Files

| path | what |
|---|---|
| `soc.py` | SoC definition + build (`--out`, `--main-ram-mb`, `--libc minimal|full`, `--tile v1,v2`, `--cpu-variant`) |
| `run.py` | run firmware on a built SoC: sparse `$readmemh` image, `--blob`, `--wad`, timeout, UART log, `SIM_FINISH` |
| `rtl/tile_v1_wb.v` | Wishbone wrapper for the v1 tile (stand-in, kept for comparison) |
| `fw/linker.ld`, `fw/fw.mk`, `fw/sim.[ch]` | firmware link script (all in main RAM), build rules, cycle counter / finish helpers |
| `fw/hello`, `fw/tiletest`, `fw/tile_v1.c`, `fw/tile_v2.c` | milestone 1/2 firmware and tile drivers |
| `fw/doom/` | DOOM port: `dg_soc.c` platform layer + POSIX shims, `sha256.c`, `doom.mk` (GPL-2.0-or-later) |
| `scripts/fetch.sh` | pinned source fetch + uv install |
| `scripts/doom_run.sh`, `scripts/compare_frame.py` | host reference, SoC run, hash + pixel comparison, PNG out |
| `scripts/check_tiletest.py` | independent host check of tile results |
| `tools/bin/riscv32-unknown-elf-*` | GCC-named shims over LLVM |

## Mojo on the soft CPU (with lane-rvmojo)

- `pixi run mojolink` (`fw/mojolink`): C firmware linked with lane-rvmojo's Mojo-compiled C-ABI object
  `../rvmojo/build/libtile.o` (build recipe in `fpga/rvmojo`). The Mojo tile v1 / tile2 drivers and Mojo CPU
  reference all match a C reference on 12x40x10 plus the 4x4x4 known answer (C[0][0] = -4): `MOJOLINK PASS`.
- lane-rvmojo's standalone Mojo firmware (Mojo crt0, no libc) also runs on this SoC:
  `bash ../rvmojo/build.sh ../rvmojo/fw/tile2_test.mojo && .venv/bin/python run.py ../rvmojo/build/tile2_test.bin --run-dir build/sim/rvmojo_tile2`
  gives `ALL PASS` for tile2 shapes up to 72x600x80 (K split with ACCUM; 2.80 M driver cycles vs 110 M for the CPU reference).
- Generic hook: `make -f fw/fw.mk APP=<app> EXTRA_OBJS=/abs/path/obj.o` (rv32im, ilp32, bare-metal ELF).

## Known gaps / next

- The DOOM SoC (`build/doom`) is the same SoC with 64 MiB RAM and full picolibc; the frame-400 pass ran on a
  build with tile v1 only (built before tile2 landed); it was then rebuilt with both tiles
  and the frame-400 run repeated on that build: same 17 checkpoint hashes, 0 differing pixels.
  DOOM does not use the tiles.
- For a board: swap integrated main RAM for LiteDRAM, the sim UART for a real one, and the WAD preload for
  SPI flash; VexRiscv standard + tile2 N=8 must also fit the target ECP5 (not checked here).
- No DOOM input path: the SoC run plays the title loop and DEMO1 with no key input, same as the host reference.
