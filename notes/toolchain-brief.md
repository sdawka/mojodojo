# Mojo / MAX open-source toolchain brief (as of 2026-11-08)

Evidence: `github.com/modular/modular@main` (read via GitHub API, nothing installed), Modular blog/forum, and Modular's official agent skills cloned at `.claude/skills/` (`modular/skills @ 148fb58`). Items marked **[inferred]** are my reading, not stated in source.

## 1. Compiler architecture

- **Pipeline:** `Mojo/lib/MojoParser` → KGEN / POP / LIT dialects → parameter elaboration (`Mojo/lib/Elaborator`) with a comptime interpreter (`Mojo/lib/Interpreter`) → `Mojo/lib/KGENToLLVM` (`LowerPOPToLLVM.cpp`, `LowerKGENToLLVM.cpp`, C ABI lowering `CABIAAPCS.cpp`/`CABISystemV.cpp`) → LLVM dialect → `Mojo/lib/Compiler/ObjectCompiler` (`KGENToLLVMPipeline.cpp`, per-target `*Backend.cpp`, bitcode writers for LLVM 17/19/21).
- **Mojo dialects:** `Mojo/include/Mojo/{KGEN,POP,LIT,HLCF,CO,DepGraph,Interpreter}Dialect`. **MAX graph-compiler dialects:** MO, MOGG, MOSH, RMO, MAP, MGP (`GraphCompiler/include/GraphCompiler/*`); runtime: GRT, MEF (`MLRT/include/MLRT/GraphRT`).
- **Inline MLIR:** `__mlir_op`, `__mlir_type`, `__mlir_attr` are used throughout stdlib. Targets are first-class values: `__mlir_attr[#kgen.target<triple=..., arch=..., features=..., data_layout=..., simd_bit_width=..., opaque_plugin=...>]` (`Mojo/test/mojo-integration/compile_offload/target_plugin.mojo`).
- **Per-target plumbing (3 layers):** traits `Mojo/lib/Target/{Host,max_license/{NVPTX,AMDGPU,Metal,Plugin}}`; KGEN→LLVM lowering `Mojo/lib/KGENToLLVM/Target/...`; backends `Mojo/lib/Compiler/ObjectCompiler/Target/...`. Runtime drivers: `Target/lib/{CPU,CUDA,HIP,Metal}`.
- **Apple Metal path:** `MetalLowering.cpp` → LLVM IR → `MetalAIRPass.cpp`/`MetalModuleRewritePass.cpp` → AIR bitcode → `MetallibWriter.cpp`, which "packages an AIR bitcode module into a .metallib container without the Metal toolchain" (no Xcode required). Runtime: `Target/lib/Metal` (metal-cpp) + `MLRT/lib/Driver/DeviceContext/Metal`.
- **Adding a new target — there is a compiler plugin ABI.** The compiler `dlopen`s libraries listed in `MODULAR_COMPILER_PLUGINS` or `modular.cfg` (`Mojo/lib/Support/max_license/PluginUtils.cpp`) and calls `M_KGEN_getTargetTraits` / `M_KGEN_getTargetBackend` (`Mojo/lib/Target/max_license/Plugin/PluginTraits.cpp`, `.../ObjectCompiler/Target/max_license/Plugin/PluginBackend.cpp`). `TargetTraits` (`Mojo/lib/Target/TargetTraits.h`) lets a target `registerDialects`, `registerPasses`, claim a dialect namespace for `__mlir_op` (`implementsDialect`, `inlineOpToGeneric`), and add host link args. `TargetLowering.h` lets it override POP lowering and add late LLVM-dialect passes; `TargetBackend.h` lets it own offload lowering + emission (`ownsOffloadLowering`).
- **Stdlib side:** `Mojo/stdlib/std/_plugin/_trait.mojo` defines `TargetDeviceKind.other(name, is_accelerator)` — "Any other device, such as an NPU" — plus hook types for math, cast, stack allocation, print, sleep, reductions. Existing hook sets: `_plugin/{cuda,hip,metal}`.
- **Runtime side:** `Target/README.md` describes an ABI-safe C HAL (`SDK/include/HAL/M_driver_*.h`) intended so "hardware vendors could independently prototype custom Driver-implementations"; mock at `MLRT/test/Driver/HAL/mock_plugin.c`.
- **Caveats:** dirs are named `max_license` but sampled file headers say Apache 2.0 — check per file (`Licenses/README.md`). Compiler/tooling contributions not yet accepted; target "by the end of this year" (blog). Plugin ABI has no stability promise **[inferred]**. Skills don't cover any of this.

## 2. Compile-time metaprogramming

- Parameters `[...]`, now with `comptime if` / `comptime for` / `comptime assert` / `comptime X = ...`; `@parameter` and `alias` are gone (`mojo-syntax/SKILL.md:40-41,118-134`; e.g. `max/kernels/src/linalg/matmul/gpu/apple/matmul_kernel.mojo` has 0 `@parameter`, 33 `comptime if/for`).
- Ordinary Mojo functions execute at compile time in the interpreter; types, `TypeList`s and compilation targets are values. A constraint solver propagates parameter facts (`Mojo/docs/site/manual/metaprogramming/constraints.mdx`).
- Kernels specialize per vendor/generation via `comptime if` on target info (e.g. M5-only fused conv path in `max/kernels/src/linalg/arch/apple/mma.mojo`).
- Autotuning is external: `kbench` sweeps YAML parameter grids (`max/kernels/benchmarks/autotune/README.md`, e.g. `benchmarks/gpu/linalg/bench_apple_gpu_matmul.yaml`).
- **vs C++ templates:** real compile-time execution in the same language, no SFINAE. **vs Triton:** not a DSL with a fixed tile model; full control of SIMD, memory spaces, inline asm, MLIR; same code targets CPU and GPUs.

## 3. MAX

- **Python APIs:** `max/python/max/{graph,functional,engine,driver,nn}`; ModuleV3 (`max.nn`, `Tensor`, `F.lazy()` + `compile()`) is current, V2 `max.nn.legacy`/explicit `Graph` is legacy (`migrate-max-v2-to-v3/SKILL.md`). `max/python/max/experimental/{torch,cascade}`.
- **Custom ops:** Mojo kernels registered and compiled with the graph; fusion done in MO/MOGG/RMO passes (`GraphCompiler/test/integration/cpu/OptimizationPatterns/*` — rms-norm+quant, matmul+reduce-scatter, layer-norm+rope…).
- **LLM pipeline:** `max serve` / `max generate`, paged KV cache (`max/kernels/src/kv_cache`, `rust_kv/`), OpenAI-compatible endpoints.
- **Training/autograd:** none found — only `torch.autograd` in reference test code. Inference only.
- **Apple coverage:** GPU kernels in `max/kernels/src/linalg/matmul/gpu/apple/` (fp4/fp8/int8 matmul, gemv, grouped matmul, simdgroup 16x16 MMA in `linalg/arch/apple/mma.mojo`), attention in `nn/attention/gpu/apple` (M5 MMA FA2 prefill, split-K decode). CPU: `linalg/arch/cpu/apple_amx_intrinsics.mojo` (AMX via inline asm), `matmul/cpu/apple_accelerate.mojo`. Since 26.4 MAX models run on M1–M5; M5 Neural Accelerator matmul kernels exist; tested mostly M3–M5 (forum).
- **Apple gaps:** no CUDA/HIP-style graph builder on Metal; unified memory needs `--device-memory-utilization`; no benchmarks vs MLX published; earlier reports of missing atomics/print/async_copy and host-buffer write bug #6145 (possibly fixed since); no GPU debugger; no ANE. `profile-model` skill is NVIDIA/AMD-only (nsys/ncu/rocprofv3) and `benchmark-model` GPU stats are NVIDIA-only — on Mac use Instruments/Metal System Trace + `kbench` **[inferred]**. `mojo-gpu-fundamentals/SKILL.md` hardware table (612-618) omits Apple (Apple simdgroup = 32 threads per `mma.mojo`). Metal device is `DeviceSpec("gpu")` (`import-model/references/pitfalls-config.md:103`).
- **Porting new archs:** `import-model` workflow runs on macOS/Apple silicon with no CUDA (`SKILL.md:70-73`); `references/native-arch-check.md` (is the arch already registered?); `references/recognize-walls.md` lists blockers: research attention kernels (NSA, Lightning Indexer), MoE quant schemes other than block-FP8/NVFP4/MXFP4, ALiBi, custom tokenizers, diffusion samplers, multimodal MoE with non-shared routing.

## 4. Interop

- **Python → Mojo:** `@export def PyInit_<mod>() abi("C") -> PythonObject` + `PythonModuleBuilder`, `mojo build --emit shared-lib`, or auto-compile via `import mojo.importer`; Python-visible structs with `py_init(args, kwargs)` (`mojo-python-interop/SKILL.md:151-221`; `Mojo/docs/site/manual/python/mojo-from-python.mdx`, labeled work-in-progress).
- **Mojo → Python:** `Python.import_module`, `PythonObject`; conversions use `Int(py=...)` (skill line 31).
- **C FFI:** `std.ffi.external_call`, `OwnedDLHandle` for dlopen (`Mojo/docs/site/manual/c-ffi.mdx`). C++ header generation for exports (`ObjectCompiler/CXXHeaderGen.cpp`).
- **Embedding:** Mojo `.dylib` with C-ABI exports usable from any host.

## 5. Hardware-adjacent

No official or community Mojo→FPGA/CIRCT/ANE work found (searches turned up only unrelated Alchitry "Mojo" FPGA board material). No skill covers it.

| Path | Verdict on an M5 Mac |
|---|---|
| Mojo as host driver to FPGA over USB/UART/Ethernet (FFI to libusb/libftdi, sockets) | **Feasible.** ECP5/iCE40 + yosys/nextpnr/openFPGALoader run natively on macOS. |
| Mojo comptime generating Verilog/FIRRTL text | **Feasible.** String generation, but typed and parametric. |
| Cycle-level accelerator simulator in Mojo, co-sim vs Verilator | **Feasible.** |
| Out-of-tree compiler + HAL plugin for a fake "npu" target (`TargetDeviceKind.other`) executing via CPU sim or a link | **Medium/hard.** ABI exists, undocumented, unstable; plugin build needs Bazel + matching LLVM headers. |
| Plugin registering CIRCT dialects; `__mlir_op` with `hw/comb/seq`; lower to `handshake`/`calyx` → Verilog | **Research-grade.** Hooks exist but pipeline is LLVM-centric; must own offload lowering and match CIRCT's LLVM version. |
| Automatic HLS of general Mojo to RTL | **Fantasy** without a real KGEN→CIRCT lowering. |
| Targeting Apple Neural Engine | **Fantasy.** Undocumented; only CoreML reaches it. |
| Xilinx/AMD boards | Vivado has no macOS build — Linux VM/remote needed. Prefer ECP5. |

## 6. What's unusual vs llama.cpp / MLX / PyTorch

- One language from SIMD/GPU kernel → graph custom op → server.
- Same kernel source specializes for NVIDIA, AMD, Apple via `comptime if`.
- Fully readable compiler (KGEN → LLVM → metallib) with an out-of-tree plugin ABI for new targets and dialects, plus a vendor-neutral HAL driver ABI.
- Xcode-free metallib emission; AMX intrinsics wrappers on CPU.
- None of the alternatives offer an open, multi-vendor kernel compiler with plugin targets.

## Distinctive project ideas (ranked by uniqueness × feasibility on M5)

1. **"Phantom NPU" compiler + HAL plugin** — invent an accelerator triple, implement `M_KGEN_getTargetTraits`/`Backend` + an HAL driver backed by a CPU simulator, run a MAX custom op on it.
   *Milestone:* a no-op plugin `.dylib` loaded via `MODULAR_COMPILER_PLUGINS` that makes `CompilationTarget` report `other("phantom", True)`.
   *Skills:* minimal (`mojo-syntax`, `mojo-gpu-fundamentals` for is_/has_ dispatch).
2. **Comptime Verilog generator + ECP5 matmul tile, Mojo host driver** — comptime emits a parametric systolic array; Mojo streams tiles over USB and checks vs MAX CPU matmul.
   *Milestone:* comptime-generated 4x4 int8 systolic Verilog passing Verilator.
   *Skills:* `mojo-syntax` (comptime), `mojo-python-interop` (cocotb/pytest harness).
3. **Port a "walled" architecture to MAX on M5** — e.g. an ALiBi model or a missing attention variant as a Mojo custom op.
   *Milestone:* `native-arch-check` + walls audit for a chosen HF model, with a plan-and-veto doc.
   *Skills:* heavily accelerated — `import-model`, `migrate-max-v2-to-v3`, `debug-model`, `eval-model`, `serve-model`.
4. **CIRCT dialect via plugin `__mlir_op`** — write `hw`/`comb` fragments inline in Mojo.
   *Milestone:* plugin that parses a single `comb.add` from Mojo source.
   *Skills:* none.
5. **M5 Neural Accelerator microbenchmark atlas vs MLX** — roofline data for Apple MMA/fp4/fp8 kernels.
   *Milestone:* run `bench_apple_gpu_matmul.yaml` via `kbench` on M5 and plot.
   *Skills:* `mojo-gpu-fundamentals` (bench harness), `benchmark-model` (serve-level only).
6. **AMX / GPU / Neural Accelerator matmul router as a MAX custom op.**
   *Milestone:* AMX and Metal kernels called from one Mojo binary with timings.
   *Skills:* `mojo-gpu-fundamentals`, `migrate-max-v2-to-v3` (V3 custom op wiring).
7. **Accelerator cycle-simulator in Mojo co-simulated with Verilator.**
   *Milestone:* SIMD-vectorized systolic simulator matching generated RTL cycle-for-cycle.
   *Skills:* `mojo-syntax`, `mojo-python-interop`.
8. **Tiny trainer on MAX with hand-written backward ops** (MAX has no autograd).
   *Milestone:* 2-layer MLP gradient check vs numpy.
   *Skills:* `mojo-python-interop`, `mojo-gpu-fundamentals`.
9. **Compiler-from-source IR study** — build KGEN with `./bazelw`, add a toy POP pass.
   *Milestone:* `./bazelw run --config=build-mojo KGEN:mojo` on M5 with `--mlir-print-ir-after-all` dumps.
   *Skills:* `new-modular-project` (setup) only partly.

**Hard/unproven overall:** plugin ABI stability; Bazel/LLVM build time and disk; CIRCT LLVM-version matching; Metal debugging/profiling (skills don't cover Mac); no upstream compiler PRs yet. Suggested progression: start with #3 (skill-supported, builds MAX fluency on M5), then #5/#6, then the unassisted #1/#2/#4.

## Sources

- https://www.modular.com/blog/mojo-open-source
- https://github.com/modular/modular
- https://forum.modular.com/t/max-models-can-now-run-on-apple-silicon-gpus/3283
- https://forum.modular.com/t/apple-silicon-gpu-support-in-mojo/2295
- https://docs.modular.com/max/api/kernels/nn/attention/gpu/apple/
- `/Users/sdawka/Code/llming/.claude/skills/` (modular/skills @ 148fb58)
