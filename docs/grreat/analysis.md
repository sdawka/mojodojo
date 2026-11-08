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
