# Does it run DOOM? (X-DOOM-1, X-DOOM-2, X-DOOM-3)

Real DOOM (the GPL source, via doomgeneric) runs on this M5 Mac with **Mojo as the host**:
Mojo's `main` runs the game loop, owns the framebuffer and implements the whole doomgeneric
platform layer, with SDL2 for the window. Tier 2 replaces DOOM's renderer inner loops (the
column and span drawers in `r_draw.c`) with Mojo. Tier 3 moves that whole drawing pass to
the M5 GPU: each frame's draws are recorded and replayed by a Mojo Metal kernel in one
dispatch. Both produce frames bit-identical to the C renderer.

| tier | status | headline (latest run, machine not idle-checked) |
|---|---|---|
| X-DOOM-1 | pass | headless timedemo ~8,600–9,500 fps; SDL path Mojo host 3,412 vs stock C host 3,321 fps (same within noise) |
| X-DOOM-2 | pass | Mojo drawers bit-identical (0 mismatches in 12.5M demo calls + 200k randomised calls; every reproducible frame hash matches); whole-frame fps C 7,697 / Mojo 7,974 / Mojo SIMD 7,816 |
| X-DOOM-3 | pass (bit-identity); GPU is ~29x slower | GPU replay bit-identical on 7,116 + 7,116 view renders (normal + low detail) and 200k randomised overlapping draws; every reproducible frame hash matches. fps C 8,010 / Mojo CPU 8,540 / Mojo GPU 278 (GPU shared with other jobs) |

The numbers are in `results/x-doom-{1,2,3}.json`; the tier 1 and 2 files are described in
`report/README.md`, under the DOOM section, and `x-doom-3.json` uses the same schema (its
extra fields are listed under Tier 3 below). The frames are `results/x-doom-{1,2,3}.png`
(frame 1000 of DEMO1; all three PNGs are byte-identical).

## Sources and licenses

| input | version | license | how |
|---|---|---|---|
| doomgeneric (ozkl) | commit `e257a09527062bfb986b6e5e0780839b6cbd5b71` (2026-10-11) | GPL-2.0-or-later (DOOM source: id Software 1993-1996, Chocolate Doom: Simon Howard 2005-2014) | `scripts/fetch.sh` clones to `third_party/doomgeneric/` (gitignored) |
| Freedoom Phase 1 | 0.13.0, `freedoom1.wad` (zip sha256 `3f9b264f…9688f59`, wad sha256 `7323bcc1…9e703d`) | BSD-3-Clause (`third_party/freedoom/COPYING.txt`) | `scripts/fetch.sh`; the WAD is not committed |
| SDL2 | Homebrew `sdl2-compat` 2.32.74 | Zlib | system install, linked dynamically |
| MAX (`max.gpu`, for tier 3 only) | `max-all` 26.8.0.dev2026110805 (same Mojo nightly as this env) | Modular license | the existing `../mojo-max` pixi env; nothing new fetched |
| zlib, CommonCrypto | macOS system libraries | — | PNG deflate/crc, SHA-256 |

The GPL-2.0 text is in `COPYING`, copied from doomgeneric's `LICENSE`. All of `doom/` is
GPL-2.0-or-later, because it links the engine and `src/drawers.mojo` is a translation of
`r_draw.c`. No commercial id WADs are used. Nothing in the game is reimplemented beyond
the drawer functions.

## What is Mojo and what is C

| piece | language | file |
|---|---|---|
| DOOM engine (game logic, BSP, sprites, planes, status bar, menus, WAD, zone) | C, upstream unmodified | `third_party/doomgeneric/doomgeneric/*.c` → `build/libdoomcore.a` |
| `main` and the game loop (`doomgeneric_Create`, then `doomgeneric_Tick` forever) | **Mojo** | `src/host.mojo` |
| Platform layer: `DG_Init`, `DG_DrawFrame`, `DG_SleepMs`, `DG_GetTicksMs`, `DG_GetKey`, `DG_SetWindowTitle` | **Mojo** `@export ... abi("C")`, resolved by the linker (no callback table) | `src/host.mojo` |
| Framebuffer (allocated by Mojo and installed into `DG_ScreenBuffer`) | **Mojo** | `src/host.mojo` |
| Window, input (SDL2 calls via `external_call`, SDL keycode → DOOM key map) | **Mojo** | `src/host.mojo` |
| Headless mode: virtual clock, PNG writer, SHA-256 frame hashes, fps timer, scripted keys | **Mojo** | `src/host.mojo` |
| Context-pointer slot, `&DG_ScreenBuffer`, exit-hook registration (~10 lines) | C glue | `shim/dg_glue.c` |
| Tier 2 drawers: `R_DrawColumn`, `R_DrawColumnLow`, `R_DrawFuzzColumn`, `R_DrawFuzzColumnLow`, `R_DrawTranslatedColumn`, `R_DrawTranslatedColumnLow`, `R_DrawSpan`, `R_DrawSpanLow` | **Mojo** (all per-pixel work and fixed-point arithmetic) | `src/drawers.mojo` |
| Tier 2 marshalling stubs (read the engine globals, call the Mojo kernel, raise upstream's `I_Error` on RANGECHECK) | C | `shim/r_draw_mojo.c` (`#include`s upstream `r_draw.c` with the 8 drawers renamed `C_*`) |
| Differential verifier and microbenchmark | C (test only) | `shim/r_draw_mojo.c` with `-DDRAW_VERIFY`, `shim/drawbench.c` |
| Tier 3 recorders: the 8 drawers become range-check + side effects + append a record; texture bytes snapshotted per frame | C | `shim/r_draw_gpu.c` (`#include`s upstream `r_draw.c`, drawers renamed `C_*`) |
| Tier 3 frame bracket around `R_RenderPlayerView` | C | `shim/r_main_gpu.c` (`#include`s upstream `r_main.c`) |
| Tier 3 per-frame record lists, GPU buffers, upload/dispatch/download, and the replay kernel (all per-pixel work) | **Mojo** (host + Metal kernel) | `src/metal/gpu_replay.mojo` |
| Tier 3 per-frame GPU-vs-C verifier and randomised self-test | C (test only) | `shim/r_draw_gpu.c` with `-DDRAW_GPU_VERIFY` |

Mojo needs a little C glue because it has no module-level variables ("global variables are
not supported"). The `DG_*` callbacks get their state through a pointer parked in
`dg_glue.c`. For the same reason, the Mojo drawers can't read C globals such as `dc_x` or
`ylookup`, so a C stub passes them as arguments. That costs one extra direct call per
column or span.

The renderer is selected at link time. The Mojo source is the same in all builds, and only
the C object linked in changes. The exception is tier 3: `max.gpu` is not in this env's Mojo
package, so the two GPU binaries are built with the `../mojo-max` env's `mojo` (same nightly)
with `-D DOOM_GPU -I src/metal`; the CPU builds put an empty `src/nogpu/gpu_replay.mojo` on
the import path instead.

| binary | drawers |
|---|---|
| `build/doom-c` | stock upstream C `r_draw.o` (this is tier 1, and the reference for tier 2) |
| `build/doom-mojo` | Mojo scalar (`r_draw_mojo.o` → `drawers.mojo`) |
| `build/doom-mojo-simd` | Mojo, built with `-D DOOM_SIMD`: 8-wide SIMD span addressing with a vector store, and 4-wide index computation for columns |
| `build/doom-mojo{,-simd}-verify` | runs both C and Mojo on every call and compares them (see below) |
| `build/doom-mojo-gpu` | tier 3: drawers recorded, replayed on the GPU once per frame (`r_draw_gpu.o` + `r_main_gpu.o` → `src/metal/gpu_replay.mojo`) |
| `build/doom-mojo-gpu-verify` | tier 3 with upstream C drawing too; the GPU result is compared with C after every view render |
| `build/doom-stock-sdl` | stock upstream `doomgeneric_sdl.c`, pure C, used as the tier-1 fps reference |

## Build and run

```sh
cd doom
pixi run fetch        # pinned doomgeneric commit + Freedoom 0.13.0 (sha256-checked)
pixi run build        # clang -O2 engine libs, then `mojo build -O3` linked 7 ways (GPU ones via ../mojo-max)
pixi run play         # windowed (3x, resizable). Arrows, Ctrl=fire, Space=use, Esc=menu
pixi run play-mojo    # same, but with the Mojo drawers
pixi run bench        # tier 1 -> results/x-doom-1.{json,png}
pixi run bench-tier2  # tier 2 -> results/x-doom-2.{json,png}  (DOOM_RUNS=N to change repeats)
pixi run bench-tier3  # tier 3 -> results/x-doom-3.{json,png}  (DOOM_RUNS=N, default 3; ~15 min on a busy GPU)
```

Flags read by the host (anything else is passed to DOOM): `--headless`, `--frames N` (exit
after N frames), `--capture 1,100,500` (PNG plus SHA-256), `--hash-all FILE` (SHA-256 of
every frame), `--inject FRAME:KEY,...` (scripted key presses), `--scale N`,
`--vclock`/`--realclock`, `--microbench`, `--out DIR`, `--label NAME`.

Example:

```sh
build/doom-mojo --headless --capture 100,500 -iwad third_party/freedoom/freedoom1.wad -timedemo demo1 -nogui
```

- Headless runs use a virtual clock by default: `DG_SleepMs` advances it and nothing else
  does. DOOM's screen-melt loop polls the real clock, so without this the number of melt
  frames would vary from run to run.
- `-timedemo` ends the way upstream ends it, through `I_Error("timed …")` and `exit(-1)`,
  so the exit code is 255. The host prints its `RESULT {...}` line from the engine's
  exit hook.
- DOOM writes `.default.cfg` and `.savegame/` to the current directory. Both are
  gitignored, and the bench scripts run in `build/bench/`.
- I ran `pixi run play` and `play-mojo` windowed for 150-210 frames (title loop) to confirm
  a window opens and renders. Both exited cleanly and nothing was left running.

## Results

### Tier 1: Mojo-hosted DOOM

The demo is Freedoom `DEMO1`, lump version 109, E1M6 on skill 3. Freedoom 1 has no E1M1
demo. With `-timedemo` it is 7,117 gametics and 7,156 frames (DG_DrawFrame calls), the
extra frames being the opening melt.

- **Headless** (no SDL, virtual clock), Mojo frame timer from the first to the last
  `DG_DrawFrame`: median 8,604 fps over 5 runs (8,457–8,704) in the latest run. Earlier
  quieter runs gave 9,300–9,550.
- **SDL path, like for like.** Both hosts use `SDL_VIDEODRIVER=dummy`, 320x200, and the fps
  the engine reports (`gametics*35/realtics`, which has 35 Hz granularity, about ±1.5%).
  The Mojo host gave 3,412 and the stock C `doomgeneric_sdl.c` gave 3,321 (medians of 3).
  An earlier run gave 3,460 vs 3,558. The two are the same within noise. With SDL in the
  loop, the per-frame `SDL_UpdateTexture` + `RenderCopy` dominates.

Reference frame hashes are SHA-256 of the raw 320x200 XRGB8888 `DG_ScreenBuffer` after the
Nth `DG_DrawFrame`. They are the same for all three renderers and across runs:

| frame | sha256 |
|---|---|
| 100 | `9780d579ca7c3374c4b3f3f31c24d720d8afd0f96f3f02bfc4c73458d9ebaaed` |
| 500 | `f97dc62425422b4061a4bb23afbec82037fcd51aa856b28455a0567101d91f02` |
| 1000 | `a26073f407b66d6bcbe336415b350b89447aa9cf13086d0008c0f525a42d85ab` |
| 2000 | `e4fb13d764140c1e3135a451d33b113a976ea5b97f2aa292da9803374ebe8f0e` |
| 3000 | `25d399c7dc5244b8a1380b6fe03e4dcd83c62f537623fff8ce2e9dac202cf0ac` |
| 5000 | `5c605edac4ade02c3cf0d3dc00a3576b9c02507032e57f6056131df936d4ff2b` |
| 7000 | `7bfc38a03b6dfa0f0927b27cd9bf75667d8f309b1e6e3c2789005c8383c3a00e` |

Frame 1 is not a valid reference: see Determinism below.

### Tier 2: Mojo drawers

**Bit-identity is checked three ways.**

1. **Per-call differential check** (`doom-mojo*-verify`). For every drawer call, the stub
   saves the destination pixels and the side-effect state (`dc_yl`, `dc_yh`, `fuzzpos`,
   `ds_x1`, `ds_x2`). It runs upstream C, captures the result, restores the saved state,
   runs Mojo, and compares the two.
   - Over the demo: 10.63M `Column`, 96k `FuzzColumn` and 1.81M `Span` calls, with **0
     mismatches** for both the scalar and SIMD builds.
   - In a low-detail run (F5 injected at frame 1500, view shrunk at frame 3000): 2.48M
     `ColumnLow`, 1.14M `SpanLow` and 40k `FuzzColumnLow` calls, also 0 mismatches.
   - A randomised self-test covers all 8 drawers, 25k calls each, with full-range int32
     `iscale`, `texturemid`, `xfrac` and steps, so fixed-point wraparound is exercised:
     0 mismatches. The translated drawers only run in multiplayer, so this test is their
     only coverage.
   - I checked that the verifier catches bugs: a deliberate one-bit change in the span
     addressing produced 1.74M mismatches.
2. **SHA-256 of every frame** (`--hash-all`, 7,156 frames) was compared across 6 C runs
   and 2 runs of each Mojo variant. Every frame that is reproducible gives identical
   hashes for C, Mojo and Mojo SIMD: 7,125 of 7,156 frames compared, 0 mismatches. The
   other 31 frames are not reproducible even between C runs (see Determinism).
3. All 7 fixed reference frames above are identical.

**fps** comes from the headless timedemo, as the median of 7 interleaved runs. The machine
was shared and busy:

| | C drawers | Mojo scalar | Mojo SIMD |
|---|---|---|---|
| latest run (busy) | 7,697 | 7,974 (+3.6%) | 7,816 (+1.5%) |
| earlier run (quieter) | 9,398 | 9,623 (+2.4%) | 9,425 (+0.3%) |

These are whole-frame numbers, so differences in the drawers are diluted by BSP, game
logic and the 8-bit→32-bit palette conversion.

**Drawer-only microbenchmark** (`--microbench`, `shim/drawbench.c`), in ns/pixel, best of
7. The Mojo times include the C stub and the extra call:

| case | C (`clang -O2`) | Mojo scalar | Mojo SIMD |
|---|---|---|---|
| column, 168 px | 0.39 | **0.33** | 0.39 |
| column, 16 px | 0.54 | 0.54 | 0.55 |
| fuzz column, 100 px | 0.98 | **0.78** | 0.77 |
| span, 320 px | 0.38 | 0.37 | **0.27** |
| span, 16 px | 0.54 | 0.55 | **0.44** |

How to read these numbers:

- Mojo is not slower. Scalar Mojo matches C on short columns and spans. There, the extra
  stub call and argument marshalling (10–11 arguments, some on the stack) costs about as
  much as Mojo gains in the loop. On long columns and fuzz, scalar Mojo is 15–20% faster:
  `-O3` unrolls the loop, while upstream is built at `-O2` here (upstream's Makefile uses
  `-Os`).
- The SIMD variant helps spans by about 25%. It computes 8 texel addresses at once and
  writes them with one 8-byte vector store. The loads stay scalar because ARM64 has no
  byte gather.
- The SIMD column variant is no better than scalar. Columns are strided by 320 bytes, so
  only the address arithmetic vectorizes, and fuzz is inherently sequential because it
  reads the row it just wrote.
- At the whole-frame level, the drawers are a minority of the frame time, so everything
  lands within a few percent of C.

## Determinism (gotcha)

Upstream doomgeneric is not bit-reproducible from run to run on its own:

- The first melt (frames 1–27) starts from `I_VideoBuffer`, which is uninitialised zone
  memory. It shows garbage that differs every run.
- A few later frames (878, 1047, 1402 and 5858 in DEMO1) get one or two garbage pixels.
  Something reads past the data, and zone-block headers there hold pointers that depend
  on ASLR and malloc.
- Two runs of the same C binary differ on these 31 frames. Disabling ASLR at spawn did not
  help, because the 6 MB zone `malloc` lands at a random address regardless.

Two choices follow from this:

- **Whole-frame hashes** are compared only on frames that are reproducible across repeated
  runs.
- **The per-call verifier** is the authoritative bit-identity check, because both drawers
  see byte-identical input in the same process.

## Other gotchas found

- **No Mojo module globals.** Platform-layer state lives in a heap struct, reached through
  `dg_glue_get_ctx()`.
- **`@export` in an imported module is dropped** unless that module is referenced from the
  root module. `drawers.keep_exports()` is called from a branch in `main` that never runs,
  so all 8 kernels are emitted.
- **`Pointer` is non-nullable.** `Pointer(unsafe_from_address=0)` fails a compile-time
  constraint. C pointers cross as `MutPointer[T, MutUntrackedOrigin]`.
- **The upstream drawers have side effects** that the Mojo kernels must reproduce through
  pointers:
  - `R_DrawFuzzColumn` clamps the globals `dc_yl` and `dc_yh`, and advances `fuzzpos`.
  - `R_DrawSpanLow` doubles `ds_x1` and `ds_x2`.
  - `R_DrawTranslatedColumn` indexes `frac>>16` without `&127`, so the index can be
    negative.
  - The RANGECHECK conditions are returned to the C stub, which raises the same `I_Error`.
- **doomgeneric compiles out config loading** (`#if ORIGCODE`), so `-config` cannot select
  low detail. The verifier injects F5 instead.
- **Output ordering.** Mojo `print` is unbuffered and C stdio is buffered, so `RESULT`
  lines appear before the engine's log.

## Tier 3: the drawing pass on the GPU (X-DOOM-3)

The C front end (BSP, clipping, visplanes, sprite sorting) is unchanged. Every call to the
eight `r_draw.c` drawers during a view render is recorded, and at the end of
`R_RenderPlayerView` the whole frame's drawing is replayed by one Mojo Metal kernel dispatch
on the M5 GPU. The 8-bit screen comes back before the status bar, HUD and menus are drawn
over it.

### How it works

- **Recording** (`shim/r_draw_gpu.c`). Each `R_Draw*` stub does what upstream does to the
  engine state: the same RANGECHECK and `I_Error`, the fuzz clamp of `dc_yl`/`dc_yh` and
  advance of `fuzzpos`, and SpanLow's doubling of `ds_x1`/`ds_x2`. It then appends a
  12×int32 record: kind, screen column range, first row, pixel count, `frac`/packed span
  position, step, and arena offsets for source, colormap and translation.
- **Texture snapshots.** The GPU cannot see DOOM's zone memory, so the bytes each call will
  read are copied into a per-frame arena when the call is recorded:
  - the hull of the texel indices the column reads (`(frac>>16)&127`, or unmasked for the
    translated drawers)
  - the 64×64 flat for spans
  - the 256-byte colormap and the translation table
  The copy is taken at call time, so the GPU sees exactly what C would have read then.
  Copies are deduplicated by (pointer, length) and re-validated with `memcmp`. That is
  about 1,760 records and 66 KiB per frame.
- **Frame bracket** (`shim/r_main_gpu.c`). Upstream `r_main.c` is `#include`d with
  `R_RenderPlayerView` renamed, and the wrapper calls `r_gpu_frame_begin`, the original,
  then `r_gpu_frame_end`. The object is linked ahead of `libdoomcore.a`, so the archive's
  `r_main.o` is never pulled in. Drawers called outside a view render (none in the demo)
  are flushed immediately. Geometry the GPU path does not model falls back to the C drawer,
  in order (none in the demo).
- **Mojo host side** (`src/metal/gpu_replay.mojo`, `mojo_gpu_flush`). It does a counting
  sort of the records into per-column lists (column and fuzz records, Low ones in both
  columns), per-row span lists and a list of fuzz columns, all in submission order. Then
  one contiguous upload (screen | records | lists | arena), **one dispatch**, a download
  of the 64,000-byte screen, and a wait. The Metal device and kernel are set up in
  `DG_Init`, before the fps timer starts.
- **Kernel, and why the order is right.**
  - Every drawer write to a pixel comes from a record that covers its column. The only
    reads of the screen are fuzz reading `dest[±SCREENWIDTH]`, which is in the same column.
    So columns are independent, and only the order within a column matters.
  - Non-fuzz writes depend only on the record and the pixel. For a column frac is
    `frac0 + j*step`, the same int32 wraparound as repeated `frac += step`. For a span the
    position is `start + i*step` (uint32). So in a column without fuzz, each pixel takes the
    value of the **last record that covers it**. One thread per pixel (32×8 tiles) walks its
    column's list and its row's span list backwards, merged by record index, and stops at
    the first hit. There are no write races and no ordering hazard.
  - A column that has fuzz gets one threadgroup, with the column in threadgroup memory.
    The column's records are split at its fuzz records. Each stretch in between is
    resolved per pixel as above, then thread 0 applies the fuzz record serially, because
    it reads the row it may have just written. `fuzzpos` at the start of each fuzz record
    is fixed when the call is recorded on the CPU.
- **Not on the GPU:** the 8→32-bit palette conversion (`I_FinishUpdate`) still runs on the
  CPU. It happens after the HUD and status bar are drawn, so moving it would need a second
  round trip per frame.

### Bit-identity (`doom-mojo-gpu-verify`, `doom-mojo-gpu`)

1. **Per frame, in process (authoritative).** In the verify build each stub records the call
   and also runs upstream C on the real screen. At frame end the GPU replays onto a
   snapshot of the pre-render screen, and all 64,000 bytes are compared. Side effects
   (`dc_yl`, `dc_yh`, `fuzzpos`, `ds_x1`, `ds_x2`) are compared per call.
   - Normal run: 7,116 view renders, 0 mismatched pixels.
   - Low-detail run (F5 at frame 1500, smaller view at 3000): 7,116 view renders, 0
     mismatched pixels. It covers `ColumnLow`, `SpanLow` and `FuzzColumnLow`.
   - Side effects: 0 mismatches.
2. **Randomised self-test.** 100 batches of 2,000 random, heavily overlapping calls, one GPU
   dispatch per batch. They cover all 8 drawers, full-range int32 `iscale`, `texturemid`
   and span steps (so the fixed point wraps), translated indices in ±32k, and fuzz with
   random `fuzzpos` and `viewheight`. Result: 0 bad pixels.
3. **SHA-256 of every frame** of `doom-mojo-gpu` against 6 C runs: 0 mismatches over the
   7,125 reproducible frames, and the 7 fixed reference frames above are identical.
4. **The verifier catches bugs.** Two deliberate faults were tried (not in the tree):
   - flipping one bit of the span texel index gave 7,056 bad frames and 46M bad pixels
   - letting column records always win over spans (breaking submission order) gave 184k bad
     pixels in the self-test

### fps (headless timedemo, median of 3 interleaved runs, GPU shared with MAX/MLX jobs)

| | C drawers | Mojo CPU drawers | Mojo GPU replay |
|---|---|---|---|
| fps | 8,010 | 8,540 | **278** (0.035× C) |
| time per frame | ~125 µs | ~117 µs | ~3.6 ms |

- The GPU path is about 29× slower, as predicted.
- The CPU does a whole frame in about 125 µs.
- The GPU path spends about 3.4 ms per frame in `mojo_gpu_flush`:
  - building the lists: 12–22 µs
  - recording: tens of µs
  - the replay kernel: about 580 µs of GPU time, measured as 16 back-to-back dispatches per
    wait, and about the same on frames with fuzz
  - the rest: submission, synchronisation and copy latency, paid once per frame because
    the HUD has to draw on top of the result
- Measured round trips varied from 0.6 ms to 19 ms during this session, depending on what
  else was running on the GPU. Every GPU number here is non-authoritative.
- Batching several frames is not possible without changing the engine, because each frame's
  game logic and HUD wait on its render.

**`x-doom-3.json` extra fields:**
- `fps_mojo_cpu`
- `gpu_flush_runs`: mean flush and the startup empty-round-trip µs per timed run
- `gpu_profile`: per-stage µs with a wait after each stage, and `kernel_amortized_x16`
- `verify`: per-frame and self-test counts, plus per-run GPU stats
- `hash_check`, `c_unstable_frames` and `frame_sha256`, as in tier 2

### Tier 3 gotchas

- **`max.gpu` is not in the plain `mojo` package.** This env's `std` has no `gpu.host`.
  Adding `-I ../mojo-max/.../lib/mojo` compiles, but the link fails on the missing AsyncRT
  symbols. So the GPU binaries are built with `pixi run --manifest-path ../mojo-max/pixi.toml
  mojo build` (rpath into that env), not by adding MAX to this env.
- **An import inside a never-taken `comptime if` is still resolved**, so a conditional
  `import gpu_replay` breaks the CPU builds. The fix is two modules with the same name, in
  `src/metal` (real) and `src/nogpu` (empty stub), chosen with `-I`.
- **Kernel arguments can't be `Int`** ("Int and UInt do not conform to DevicePassable"). Use
  `Int32`.
- **The host staging buffer is slow to read from the CPU.** It is likely write-combined.
  Building the lists from it took about 460 µs per frame; from the recorder's own `malloc`
  memory it takes about 12 µs. The recorder writes to cached memory, and Mojo only *writes*
  into the staging buffer.
- **The first kernel was one thread per column scanning every record** (about 3.3 ms
  replay). One thread per pixel with backward last-writer search needs far more threads to
  hide memory latency on a GPU this shared. A per-column serial path is used only where
  fuzz forces it.
- **Exports in the GPU module** stay alive because the C recorder references them; the host
  also calls `gpu_replay.keep_exports()` from the never-taken branch, the same as tier 2.
