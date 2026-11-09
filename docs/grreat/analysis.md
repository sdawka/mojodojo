<!-- grreat:record id=analysis.current kind=analysis status=active parent=goal.north-star -->
# Milestone review

- Reviews: [[goals]] and [[execution]].

## 2026-11-08 — Phase 1 (toolchains and baseline)

| Success criterion | Evidence and tested state | Remaining gap |
| --- | --- | --- |
| Mojo kernel runs on Apple GPU | `pixi run kernel-smoke`: Apple M5 / metal, vector add + 512² matmul 0 mismatches ([[journals/2026-11-08]]) | Relies on undocumented `MODULAR_METAL_INTREE_METALLIB=1` |
| Same model runs on MAX, llama.cpp, MLX | Llama 3.2 3B on all three; MAX via `max serve` on GPU | MAX 4-bit not measured |
| Fair, reproducible comparison | Sequential idle runs; endpoint client and `max benchmark` agree within 1%; [bench/results/summary.md](../../bench/results/summary.md) | llama-bench measures tg on empty context; F16 vs bf16 |

Decision: aim Goal 1's tile at prefill (MAX's weakest point; decode is memory-bound everywhere).

Goal 1 and Goal 2 criteria: not yet reviewed (see [[roadmap]]).

## 2026-11-09 — Goal 1 and Goal 2 against accepted criteria (tested state: local 5a7d0c0 + w8a8/ results)

| Success criterion | Evidence | Remaining gap |
| --- | --- | --- |
| G1: RTL matches the Mojo sim cycle-for-cycle | 62 co-sim traces (re-verified, 0d9bd7e); tile v2: 62 programs bit- and cycle-exact, 8 negative checks caught (e762621) | — |
| G1: Mojo targets the phantom device, bit-exact int8 | 37/37 tests via FFI route + stdlib target kind + HAL plugin (3df9888) | Real codegen plugin needs the compiler source build (not approved) |
| G1: Llama 3.2 3B in MAX with prefill matmuls on the tile, greedy match, cycles reported | `maxop-test` re-verified: 112 matmuls int32-exact vs W8A8, 8/8 greedy match vs GPU W8A8, TTFT ≈ 94.6 s projected (3788200). W8A8 costs +0.67% ppl vs bf16 ([[journals/2026-11-09]]) | Greedy vs bf16 is 5/8 (W8A8 itself, not the tile) |
| G1 extension: Mojo firmware on a RISC-V SoC drives the tile | 5 shapes bit-exact on VexRiscv RTL in Verilator, 72×600×80 ≈ 39× vs CPU (6ecdf0b, c431cc7) | Simulation only; bus-store bound (~97%) |
| G1 (optional): runs on an ECP5 | Synthesis only (tile ~60 MHz, tile v2 ~97 MHz on 85F) | No board |
| X-DOOM-4: frame from the soft CPU matches host | Frame 1 re-verified 0/64,000 px; frame 400 reported by lane (810963d) | Board run |
| G2: reproducible atlas with variance | 797 authoritative records (1ed3f38) | — |
| G2: ≥15% decode/TTFT at ≤3% ppl for two models | Quality passes for Llama Q4_K_M (+1.72%), Gemma 3 4B (all formats), Llama W8A8 (+0.67%); roofline predicts +53% (Gemma QAT + 270M draft), α≈0.84 (Llama + 1B draft) | **No timed run yet** — criterion not met |

Decision: Goal 1 met through G1-D in simulation, plus the SoC extension; what remains needs hardware or an approved build. Goal 2 is gated only on idle-machine timed runs, which are the next action.
