<!-- grreat:record id=execution.next kind=execution_item status=active parent=roadmap.next -->
# Sprint 1 — run lanes G1-A and G2-M1 in parallel

## Now

- Goal: [[goals]]. Milestone: [[roadmap]] `roadmap.next` (Sprint 1).
- Lane G1-A (`fpga/`): DONE — simulator + spec, 24/24 tests re-verified by orchestrator ([[journals/2026-11-08]]). Next for Goal 1: G1-B (Verilog from Mojo comptime, Verilator co-sim vs `pixi run trace` golden vectors) — start after the atlas timed run, or now if CPU-light.
- Lane G2-M1 (`atlas/`): DONE — idle-machine run, 797 authoritative records ([atlas/ATLAS.md](../../atlas/ATLAS.md), [[journals/2026-11-08]]). Sprint 1 complete.
- X-PAGE: built and rebuilt with authoritative atlas data (`~/Desktop/mojo-vs-apple-silicon.html`).
- X-DOOM-1 + X-DOOM-2: PASS (re-verified: 0 mismatches; Mojo drawers +3.6% fps vs C, non-authoritative). X-DOOM-4: PASS in cycle-level SoC sim (VexRiscv + tile, frames 1/400 bit-identical, re-verified frame 1); board run pending. X-DOOM-3 (Metal): PASS — bit-identical (7116 frames, 0 px, re-verified), GPU ~29× slower than CPU (round-trip bound, non-authoritative).
- G1-B: DONE — co-sim 62 traces pass (re-verified), ECP5 synthesis recorded.
- G2-M2: DONE — 7/7 within 20% on holdout (re-verified). Next for Goal 2: G2-M3, starting with idle-machine measurements M4 (speculative α) and M8 (perplexity) once lanes are quiet; perplexity gate relaxed to ≤3% vs bf16 (user, 2026-11-08); +15% measured against the 4-bit 48.8 tok/s / 329 ms baselines.
- G1-C: DONE (re-verified 36/36; now 37/37 with mode flag); `./bazelw query` feasibility check running — device kind + HAL plugin + FFI API; real codegen plugin blocked pending a full compiler source build (user decision).
- G1-D: DONE — prefill projections routed to the phantom tile through the full MAX pipeline; tile int32-exact vs W8A8, all-routing 8/8 greedy match vs GPU W8A8; W8A8 vs bf16 5/8 (all) / 7/8 (L0 MLP); projected TTFT ≈ 94.6 s at N=16 (re-verified `maxop-test`, commit 3788200). Next for Goal 1: G1-E needs an ECP5 board; real codegen plugin needs the deferred source build.
- G2-M3 quality: DONE (re-verified from raw NLL + logs) — MLX 4-bit baseline fails the 3% gate (+7.25%); gate-valid baseline is llama.cpp Q4_K_M (+1.72%). Surviving decode candidate: Q4_K_M 3B + 1B draft (α≈0.84). Next: idle-machine timed runs per [quality/QUALITY.md](../../quality/QUALITY.md) (llama-speculative / llama-server -md, k=2,3); second model: Gemma 3 4B — all formats pass ppl, QAT Q4_0 lowest KLD; best predicted llama.cpp QAT-local Q4_0 + 270M draft (+53% pooled). Timed runs (no Gemma baseline yet) listed in QUALITY.md §5.
- Running now:  lane-w8a8-ppl (Llama 3B W8A8 wikitext ppl in MAX), lane-g1c (bazel query). See [decision log](../decision_log.md).
- Next decision: Sprint 2 candidates — G2-M2 (roofline model) → G2-M3, and G1-C (phantom NPU plugin). DOOM-1 runs alongside.
- Blockers: None. ECP5 board not purchased (only needed for G1-E).
- Latest verification: Phase-1 benchmarks on an idle machine, 2026-11-08 ([[journals/2026-11-08]]). Atlas idle run 2026-11-08 (commit 1ed3f38); co-sim re-verified (0d9bd7e).
- Last updated: 2026-11-08 (G1-D verified).

## Work history

- [[journals/2026-11-08]] — Setup, no-Xcode Metal fix, phase-1 baseline, toolchain research, GRREAT initialized, git init.
