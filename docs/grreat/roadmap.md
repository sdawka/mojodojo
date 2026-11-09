<!-- grreat:record id=roadmap.done kind=roadmap_item status=done parent=goal.north-star -->
# Phase 1 — Toolchains and baseline (done)

- Mojo/MAX nightly on Metal, llama.cpp (Metal), MLX; same Llama 3.2 3B across all three. Evidence: [[journals/2026-11-08]].

<!-- grreat:record id=roadmap.next kind=roadmap_item status=active parent=goal.north-star -->
# Sprint 2 — G2-M3 timed runs ∥ RISC-V SoC path toward hardware (active)

- Supports: [[goals]] (both child goals). First action: [[execution]].
- The single active record is the resume pointer, not a concurrency limit: each lane runs as its own subagent with its own directory; the orchestrator is the only writer of these notes and runs anything timing-sensitive.
- Sprint 1 (G1-A ∥ G2-M1) finished 2026-11-08, followed the same day by G1-B, G1-C, G1-D, G2-M2, G2-M3 quality, X-DOOM-1..3, tile v2, the SoC and X-DOOM-4 in simulation ([[journals/2026-11-08]], review in [[analysis]]).
- Sprint 2 is done when: (a) G2-M3 meets its criterion for two models on idle-machine timed runs, and (b) G1-E either runs on a board or is explicitly deferred by the user.

### Status (2026-11-09)

| Stage | Status |
| --- | --- |
| G1-A, G1-B, G1-C (FFI route), G1-D, G1-T2, G1-SoC, G1-RV | done (verified) |
| G1-C real codegen plugin | blocked: needs full compiler source build (user approval); `./bazelw query` check not yet run |
| G1-E, G1-DMA | not started (G1-E needs a board, see [fpga/BOARDS.md](../../fpga/BOARDS.md)) |
| G2-M1, G2-M2 | done (verified) |
| G2-M3 | quality done (Llama, Gemma, Llama W8A8 in MAX); timed runs pending idle machine |
| X-DOOM-1..3 | pass; X-DOOM-4 pass in cycle-level SoC sim, board run pending |

### Lane G1-A — Mojo cycle-accurate simulator of the int8 tile (Goal 1)
- Owns: `fpga/`.
- Build: parametric N×N output-stationary int8 systolic array (int32 accumulate) as a cycle-level simulator in Mojo, with a reference matmul and a cycle-count model.
- Completion criteria: bit-exact vs a reference int8 matmul on randomized shapes (incl. non-multiples of N); cycle counts match the analytical formula; tile size and data widths are comptime parameters.

### Lane G2-M1 — Processor atlas (Goal 2)
- Owns: `atlas/`.
- Completion criteria: see G2-M1 in the Goal 2 table below. Timed runs only on an idle machine and never concurrently with other GPU/CPU-heavy work; G1-A's builds/tests must pause or the atlas lane must re-run affected measurements.

## Goal 1 — stages (G1-A is in Sprint 1; the rest planned, in order)

| ID | Stage | Depends on | Completion check |
| --- | --- | --- | --- |
| G1-B | Mojo comptime emits Verilog for the same tile | G1-A | Verilator co-sim matches the simulator cycle-for-cycle on randomized tests |
| G1-C | "Phantom NPU": compiler + HAL plugin backed by the simulator | G1-A | Mojo code reports `TargetDeviceKind.other("phantom", True)` and runs an int8 matmul bit-exactly. Fallback if the plugin ABI is impractical: Mojo FFI driver to a simulator `.dylib` |
| G1-D | MAX custom op routes Llama 3.2 3B prefill matmuls to the tile | G1-C, G2-M2 | Greedy output matches the GPU-only run for a fixed prompt set; simulated cycles and projected TTFT reported |
| G1-T2 | Tile v2: buffered N=8 tile with CSRs + Wishbone slave, generated from Mojo | G1-B | Bit- and cycle-exact Verilator co-sim vs a Mojo bus model; ECP5 synthesis recorded |
| G1-SoC | LiteX + VexRiscv SoC in Verilator with the tile(s) on the bus | G1-T2 | C firmware matmuls bit-exact vs host; cycle counts match the formula |
| G1-RV | Bare-metal Mojo firmware (riscv32) drives the tile in the SoC | G1-SoC | Mojo driver bit-exact vs a Mojo CPU reference on the core |
| G1-DMA | Feed the tile by DMA / double buffering (bus-store bound today) | G1-SoC | Same bit-exact tests; operand-load share of time well below today's ~97% |
| G1-E | (Optional) synthesize for ECP5 and run on a board: the tile over USB from Mojo, or the whole SoC | G1-B (G1-SoC for the SoC variant), hardware | On-board results match the simulator; fmax and utilization recorded |

## Goal 2 — processor atlas and model optimization (G2-M1 is in Sprint 1)

| ID | Milestone | Depends on | Completion check |
| --- | --- | --- | --- |
| G2-M1 | Processor atlas: bandwidth (CPU, GPU) and matmul throughput per engine × dtype × shape (CPU NEON, Accelerate/AMX-SME, Metal GPU, M5 Neural Accelerator via MAX kernels/`kbench`, ANE via Core ML) | Phase 1 | Reproducible scripts + data + plots; each number has run count and variance; idle-machine runs only |
| G2-M2 | Roofline model for 1–4B decode and prefill per engine; pick candidate configs | G2-M1 | Predicted vs measured tok/s within 20% for the phase-1 configs |
| G2-M3 | Optimize ≥2 models (e.g. Llama 3.2 1B/3B, Qwen3 1.7B/4B, Gemma 3 1B/4B): quantization, MAX 4-bit/int8, speculative decoding (1B draft), engine split (e.g. ANE/GPU prefill vs GPU decode) | G2-M2 | ≥15% better decode tok/s or TTFT than the best phase-1 baseline at ≤3% perplexity delta vs bf16 (relaxed from 1% by user, 2026-11-08) |

## Cross-cutting (planned)

| ID | Item | Depends on | Completion check |
| --- | --- | --- | --- |
| X-PAGE | Comparison page at `~/Desktop/mojo-vs-apple-silicon.html` (see [[goals]]), generated by a script under `report/` from `bench/results`, `atlas/results`, `fpga/` and `doom/` outputs | Atlas idle-machine run (G2-M1); updates after each later milestone | Opens offline as one file, light/dark, phone-width OK; every number links to its source JSON + date; regenerating reproduces it |
| X-DOOM-1 | DOOM hosted by Mojo: doomgeneric built as a C library, Mojo drives the game loop over FFI and blits frames to a window | — | Freedoom E1M1 demo plays; frame captured; fps recorded |
| X-DOOM-2 | DOOM renderer inner loops (column/span drawing) rewritten in Mojo on CPU, rest stays C | X-DOOM-1 | Bit-identical frames vs C renderer on a recorded demo; fps compared |
| X-DOOM-3 | DOOM frame rendering on the M5 GPU via a Mojo Metal kernel (column/span pass) | X-DOOM-2 | Bit-identical frames on the demo; fps compared |
| X-DOOM-4 | DOOM on our hardware path: soft CPU (e.g. small RISC-V) beside the int8 tile on the ECP5, or on a cycle-level sim of that SoC | G1-B, G1-E (hardware) | Title screen/demo frame from the soft CPU, verified against host frame. Expected to be the hardest; the tile alone is not a CPU and can't run DOOM |

Accepted criteria above are the ones [[analysis]] reviews at each milestone.
