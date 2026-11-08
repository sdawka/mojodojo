<!-- grreat:record id=research.current kind=research status=active parent=goal.north-star -->
# Current understanding

- Informs: [[goals]] and [[roadmap]].

## Findings (2026-11-08)

- **Toolchain brief:** [notes/toolchain-brief.md](../../notes/toolchain-brief.md). Key points:
  - The compiler loads target plugins (`MODULAR_COMPILER_PLUGINS`, `M_KGEN_getTargetTraits` / `M_KGEN_getTargetBackend`) that can register MLIR dialects and passes and own offload lowering. The stdlib has `TargetDeviceKind.other(name, is_accelerator)`, and there is a C HAL driver ABI. Undocumented, stability unknown.
  - Comptime runs ordinary Mojo; `comptime if/for` replaced `@parameter`. `kbench` autotunes from YAML (Apple matmul config exists).
  - MAX is inference-only (no autograd). Apple GPU kernels: simdgroup MMA, M5 Neural Accelerator matmul, fp4/fp8/int8, FA2 prefill. CPU: AMX via inline asm, Accelerate.
  - Mojo→CIRCT/HLS is research-grade to fantasy; ANE is reachable only through Core ML. ECP5 + yosys/nextpnr/openFPGALoader work natively on macOS; Vivado does not.
- **No Xcode needed:** `MODULAR_METAL_INTREE_METALLIB=1` (undocumented, found in the `mojo` binary) makes the compiler package metallibs itself. Set in [mojo-max/pixi.toml](../../mojo-max/pixi.toml). See [[journals/2026-11-08]].
- **Phase-1 baseline** (Llama 3.2 3B, 512-token prompt, 128 greedy tokens): [bench/results/summary.md](../../bench/results/summary.md).
  - 16-bit decode ≈ 15.3 tok/s on MAX, llama.cpp and MLX alike → memory-bound (~99 GB/s effective).
  - MAX prefill is the weak spot: TTFT 632 ms vs llama-server 444 ms through the same HTTP client.
  - 4-bit: llama.cpp Q4_K_M 48.8 tok/s, MLX 4-bit 48.8 tok/s. MAX 4-bit not yet measured.
- **Official Modular skills** in [.claude/skills/](../../.claude/skills/SOURCE.txt): good for model porting/serving. Weak for Apple: `profile-model` and `benchmark-model` GPU stats are NVIDIA/AMD-only, and `has_apple_gpu_accelerator` import from `mojo-gpu-fundamentals` doesn't exist in this nightly.

## Open questions and assumptions

- Is the compiler plugin ABI buildable out-of-tree against the nightly, or does it need a full `./bazelw` compiler build? (Goal 1, stage C)
- Can MAX use int8/fp4 weights for Llama on Apple GPU (which `--quantization-encoding` values work)? (Goal 2)
- How are the M5 GPU Neural Accelerators exposed — only via MAX's Mojo kernels and Metal 4 tensor ops, or also through MLX? (Goal 2)
- Can Core ML run a 1–4B model's prefill on the ANE fast enough to be worth a hybrid split? (Goal 2)
- Assumption: the ECP5 board is optional and not yet purchased.
