# MAX / Mojo on Apple M5 (Metal)

Status (2026-11-08): **working on the Apple GPU without Xcode.**
- The Mojo GPU kernels (vector add and naive matmul) run on "Apple M5" with API `metal`, and their results are correct.
- `max generate` and `max serve` run Llama 3.2 3B Instruct in bf16 on `gpu[0]`.

The key fix was `MODULAR_METAL_INTREE_METALLIB=1`, which is set in `pixi.toml`. See Gotchas.

## Machine
- Apple M5, 32 GB unified memory, macOS 26.5 (25F71), arm64
- Command Line Tools only (`/Library/Developer/CommandLineTools`, Apple clang 21). No Xcode.app and no Metal Toolchain: `xcrun -f metal` fails.

## Install
```bash
brew install pixi                      # pixi 0.84.1 (the curl|sh installer was blocked by the agent sandbox)
cd /Users/sdawka/Code/llming/mojo-max
pixi init . -c https://conda.modular.com/max-nightly/ -c conda-forge
pixi add "python=3.12" max-all
```
Then add this to `pixi.toml` (already done):
```toml
[activation.env]
MODULAR_METAL_INTREE_METALLIB = "1"   # package metallibs in-process, no `xcrun metal` needed
MODULAR_TELEMETRY_ENABLED = "0"
```
Versions (nightly channel `https://conda.modular.com/max-nightly/`):
- `max`, `max-core`, `max-serve`, `max-benchmark`, `max-all`: 26.8.0.dev2026110805
- `mojo`, `mojo-compiler`: 1.3.0.dev2026110805 (11efb73e)

This matches the official `new-modular-project` / `serve-model` skills in `../.claude/skills/`: pixi, the nightly channel, and conda-forge. pixi wrote the channel URL without the trailing slash, and it resolves fine.

pixi tasks (they inherit the activation env): `pixi run kernel-smoke`, `pixi run generate`, `pixi run serve` (port 8000).

Fallback only: to use the official Xcode route instead, install Xcode, `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`, `xcodebuild -downloadComponent MetalToolchain`, and set `MODULAR_METAL_INTREE_METALLIB=0`.

If you hit OOM on unified memory, lower `--device-memory-utilization` on `max serve` / `max generate`. I did not need it at 3B bf16 with `--max-length 4096`.

## 1. Mojo GPU kernel
Source: `kernels/gpu_smoke.mojo`, using `max.gpu`, `max.gpu.host.DeviceContext` and `layout.TileTensor`, as in the `mojo-gpu-fundamentals` skill.
```bash
pixi run mojo kernels/gpu_smoke.mojo
```
Output:
```
GPU device: Apple M5
API: metal
vector_add N= 1048576  mismatches= 0
matmul_naive  512 x 512  mismatches= 0  time(ms, incl. first-launch)= 0.69  GFLOP/s= 391
```

## 2. MAX LLM on GPU
- Model: `unsloth/Llama-3.2-3B-Instruct`, bf16 safetensors, rev 006f5dcd13. It is shared with the llama.cpp/MLX agent at `/Users/sdawka/Code/llming/models/Llama-3.2-3B-Instruct` (6.0 GB).
- MAX architecture: `LlamaForCausalLM_ModuleV3`. The log shows `devices: gpu[0]`.
- `ioreg` reported 81-93% Apple GPU "Device Utilization" during decode.

Generate smoke test:
```bash
pixi run max generate --model /Users/sdawka/Code/llming/models/Llama-3.2-3B-Instruct \
  --devices gpu --quantization-encoding bfloat16 --max-length 1024 \
  --max-new-tokens 128 --top-k 1 --prompt "Write a short story about a robot learning to paint."
```
Results from a single run (smoke test, not a benchmark), 128 output tokens:
- TTFT 130 ms
- prefill 353 tok/s on a short prompt
- decode 15.7 tok/s (TPOT 63.9 ms)
- model compile ~29-34 s on every launch, plus sampler compile 6 s on the first launch
- peak RSS ~12 GB

Serve (OpenAI-compatible) smoke test. It was ready about 44 s after launch, and a chat completion returned the correct answer:
```bash
pixi run max serve --model /Users/sdawka/Code/llming/models/Llama-3.2-3B-Instruct \
  --devices gpu --quantization-encoding bfloat16 --max-length 4096 --port 8000
# optional: --served-model-name llama-3.2-3b  (otherwise the model id is the full path)
curl localhost:8000/v1/models
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model":"/Users/sdawka/Code/llming/models/Llama-3.2-3B-Instruct",
  "messages":[{"role":"user","content":"Name three primary colors."}],
  "max_tokens":40,"temperature":0}'
```
Readiness check: poll `GET /v1/models` until it returns 200.

The fallback models (1B, Gemma 3, Qwen2.5) were not needed and were not tried.

## Gotchas
- **No Xcode / Metal Toolchain:** by default, Mojo and MAX shell out to `xcrun metal` / `xcrun metallib`, and that fails with:
  ```
  error: Metal Compiler failed to compile metallib. Please submit a bug report.
  xcrun: error: unable to find utility "metal", not a developer tool or in PATH
  ```
  For MAX the error is wrapped in `RuntimeError: Graph compilation failed ... The graph compiler could not elaborate the generated KGEN`.
  The fix is `MODULAR_METAL_INTREE_METALLIB=1`, which uses Mojo's in-process metallib packager. I found this env var in the `libMGPRT` / `mojo` binary strings; it is not in the docs. Setting it to `=0` on a host with Xcode switches back to the `xcrun` path. The docs suggest installing Xcode plus `xcodebuild -downloadComponent MetalToolchain`, but this machine does not need that.
- The compile cache (`.pixi/envs/default/share/max/cache`) can hide the problem. A kernel that was compiled once with the flag later runs fine without it. Test with a fresh `MODULAR_CACHE_DIR` to be sure.
- Imports: `std.gpu` / `std.gpu.host` are now `max.gpu` / `max.gpu.host`. `ctx.name()` is deprecated, so use `ctx.device().get_name()`.
- **Contradicts the skill:** the `mojo-gpu-fundamentals` skill says `from std.sys.info import has_apple_gpu_accelerator`, but this nightly reports "module 'info' does not contain 'has_apple_gpu_accelerator'". Use `std.sys.has_accelerator` instead.
- `serve-model/scripts/suggest_serve_command.py` only handles custom architectures (`--custom-architectures`). It does not apply to built-in Llama.
- The MAX CLI prints harmless click warnings: "--top-k / --temperature used more than once".
- Model compilation is not cached across launches (~30 s each time). Benchmarks should exclude startup.
- The KV-cache warning "Jenga KV geometry can strand 40%..." is informational.
- macOS has no `timeout` command.
- Unified memory: MAX peaked at ~12 GB RSS for the 3B bf16 model. Do not run it at the same time as the llama.cpp/MLX runs.
