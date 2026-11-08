# Decision log

## 2026-11-08 — Use the in-tree metallib writer instead of Xcode
`MODULAR_METAL_INTREE_METALLIB=1` lets Mojo/MAX build Metal code on a Command-Line-Tools-only Mac. Undocumented; if a nightly drops it, pin the last working version or install Xcode + MetalToolchain.

## 2026-11-08 — Aim the invented int8 tile at prefill
16-bit decode is memory-bound at ~15.3 tok/s on every runtime; MAX's gap is prefill (TTFT 632 ms vs 444 ms). Offloading matmuls helps prefill, not decode.

## 2026-11-08 — Two tracked goals under one lab vault
Goal 1: invented accelerator from Mojo to FPGA. Goal 2: M5 processor atlas, then optimize 1–4B models. User passed over generic follow-ups (GPT-2-in-Mojo, kernel shootouts, MLX LoRA).

## 2026-11-08 — Parallel lanes under one active GRREAT record
The single active Roadmap/Execution record is a sprint holding parallel lanes, each owned by a subagent with its own directory. Orchestrator is the only notes writer and schedules timing-sensitive runs.

## 2026-11-08 — Add comparison page and "Does it run DOOM?" ladder
User request. Comparison page is a regenerated single HTML file on the Desktop (toolchain stages side by side, cross-toolchain performance, Mojo across processors/backends). DOOM uses the GPL source (doomgeneric) and Freedoom data only; tiers climb from Mojo host → Mojo CPU renderer → Mojo Metal renderer → soft CPU on the FPGA path.

## 2026-11-08 — Relax G2-M3 quality gate to ≤3% perplexity
User decision. The best phase-1 decode baseline is 4-bit (48.8 tok/s), which likely fails a 1% gate; 8-bit 3B decodes only ~27–29 tok/s. With ≤3%, configs are compared directly against the 4-bit baselines (48.8 tok/s decode, 329 ms TTFT). Perplexity still has to be measured for every candidate.

## 2026-11-08 — Gemma 3 as the second Goal 2 model; three more parallel lanes

- User suggested Gemma. Rationale: Google's quantization-aware-trained int4 checkpoints are the best chance for a 4-bit config under the 3% gate (plain MLX 4-bit failed for Llama at +7.25%); same-tokenizer 270M/1B drafts; already in the roofline. Risk: 262k vocab makes the LM head costly per decode step; α unmeasured.
- Started in parallel (quality/correctness only, load-insensitive): lane-gemma (quality/), lane-w8a8-ppl (w8a8/: wikitext ppl of Llama 3B W8A8 prefill in MAX), lane-doom3 (X-DOOM-3). Timed G2-M3 runs still wait for an idle machine.

## 2026-11-08 — Toward a RISC-V + tile SoC and DOOM level 4, simulation first

- Verified: shipped Mojo nightly emits RISC-V (`--target-triple riscv32-unknown-none-elf --emit object` → ELF32 RISC-V; an exported add → `add a0, a1, a0; ret`). So Mojo can be the soft core's firmware language without the compiler source build.
- Constraints: N=16 tile fills an 85F → v2 at N=8 with DSPs + on-chip buffers + Wishbone slave; DOOM needs MBs of RAM → SDRAM board; X-DOOM-4 may pass on a cycle-level SoC sim.
- Started in parallel (user-approved, incl. LiteX/VexRiscv/RISC-V toolchain downloads): lane-tile2 (fpga/tile2), lane-soc (fpga/soc: LiteX+VexRiscv Verilator sim, tile peripheral, DOOM port), lane-rvmojo (fpga/rvmojo: bare-metal Mojo firmware driving the tile), lane-boards (fpga/BOARDS.md).
