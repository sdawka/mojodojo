# maxop: Llama 3.2 3B prefill matmuls on the phantom NPU, via a MAX custom op (G1-D)

The invented int8 systolic tile (`../DESIGN.md`) runs Llama 3.2 3B projections inside the real MAX serving pipeline. A MAX custom op (`phantom_matmul_i8`, Mojo, host CPU) takes the int8 operands of a W8A8 linear layer and hands them to a phantom-NPU adapter. The adapter either computes the int32 result functionally and charges the DESIGN.md cycle formula (`stub`), or calls lane G1-C's simulated device `libphantom.dylib`, which steps the cycle-accurate register model (`device`). Results are checked against an exact integer reference and the GPU bf16 model. Simulated cycles become projected TTFT at the synthesized fmax and at several host-link bandwidths.

This is a verification and projection exercise, not a speedup. The tile is 3 to 5 orders of magnitude slower than the M5 GPU for these shapes (G1-A/G1-B).

**Integration level: full pipeline.** `max generate` / `max serve` / the Python pipeline API run the stock `LlamaForCausalLM_ModuleV3` with `--custom-architectures fpga/maxop/phantom_llama`, and the routed projections execute on the phantom during prefill and decode.

## Layout

| Path | What |
| --- | --- |
| `kernels/phantom_op.mojo` | MAX custom op `phantom_matmul_i8` (`@extensibility.register`): `c[M,N] int32 = a[M,K] int8 · w[N,K]ᵀ int8`, plus `site[1] int32` (call-site id for the log). CPU device only. |
| `kernels/phantom_adapter.mojo` | The adapter, the only code that touches the device. `PHANTOM_BACKEND=stub` (default): SIMD and threaded exact int32 matmul plus the formula cycles. `PHANTOM_BACKEND=device`: dlopen `$PHANTOM_LIB` (lane G1-C `libphantom.dylib`, C ABI `phantom_open/alloc/memcpy_h2d/launch_matmul_i8/memcpy_d2h/close`), with tile `PHANTOM_DEVICE_N`. It checks measured cycles == formula. Each launch appends `site M K N_out cyc@4 cyc@8 cyc@16 dev_N measured` to `$PHANTOM_LOG`. |
| `phantom_llama/phantom_ops.py` | `phantom_linear`: in-graph W8A8 quantization on the GPU, transfer to the host, the custom op, transfer back, dequant to bf16. Also `parse_route`. `PHANTOM_MODE=phantom\|w8a8_gpu\|bf16`. |
| `phantom_llama/__init__.py` | Custom architecture. It re-registers `LlamaForCausalLM_ModuleV3` with a `Llama3Decoder` subclass that class-swaps the routed modules (`QKVLinear`→`PhantomQKV`, `o_proj`→`PhantomLinear`, `MLP`→`PhantomMLP`). Weight names do not change. |
| `run_model.py` | Python driver: builds the full pipeline (`PipelineArgs` → `PipelineConfig.from_args` → `PIPELINE_REGISTRY.retrieve`), then `max.pipelines.lib.generate.generate` gives greedy tokens and prefill logits. |
| `demo.py` | 5 configurations, comparison, TTFT projection, and `../results/maxop.json`. |
| `maxop_lib.py` | Checkpoint reader (bf16 safetensors via numpy), numpy W8A8 reference, cycle/IO/TTFT model. |
| `tests/test_maxop.py`, `tests/spot_check.mojo` | `pixi run maxop-test`. |

Routing is set by `PHANTOM_ROUTE`: `0:mlp` (default), `0-3:qkv,o`, `5:gate_up`, `all:all`, `;`-separated. The projections are `qkv` (fused q|k|v, 3072→5120), `o` (3072→3072), `gate_up` (fused, 3072→16384) and `down` (8192→3072). The embedding, attention core (QKᵀ/PV), norms and lm_head stay on the GPU.

## Running

The environment is a new pixi env in this directory, `max-all ==26.8.0.dev2026110805` (the same nightly as `../../mojo-max`, mojo 1.3.0.dev2026110805). It uses the same channels and activation env (`MODULAR_METAL_INTREE_METALLIB=1`).

```bash
cd fpga/maxop
pixi install
pixi run maxop-test     # ~1 min, GPU used only for tiny graphs
pixi run maxop-demo     # ~5 min: 5 model loads + short generations; cached in out/ (--fresh to redo)
# Device backend needs lane G1-C's library:  (cd .. && pixi run phantom-build)

# Through the stock CLI (prefill + decode routed):
PHANTOM_ROUTE=0:mlp PHANTOM_LOG=/tmp/phantom.log pixi run max generate \
  --custom-architectures $PWD/phantom_llama \
  --model /Users/sdawka/Code/llming/models/Llama-3.2-3B-Instruct --devices gpu \
  --quantization-encoding bfloat16 --max-length 512 --max-new-tokens 8 --top-k 1 \
  --prompt "The capital of France is"          # -> " Paris." ; log shows 2 launches per forward
# Same for `max serve` (add the same flag). Don't leave a server running: GPU is shared with the DOOM lane.
```

## Results (2026-11-08, Apple M5; `../results/maxop.json`, authoritative: false)

### Correctness (`pixi run maxop-test`, all pass)

- **int32 bit-exact, real shapes.** Every projection of layers 0 and 27 ran through the MAX op with the real weights at full K × N_out and M = 37 tokens (8 matmuls, up to 37×3072×16384 and 37×8192×3072). The int32 output equals an exact integer matmul of the op's int8 operands.
  - Activation quantization matches an independent numpy quantizer exactly.
  - Weight quantization matches except for 0.002–0.04% of values on .5 rounding ties, which are off by 1 because GPU division is not correctly rounded. This is a quantizer detail, not a tile error.
  - Dequantized bf16 output is within 1 ulp.
- **Cycle-accurate spot checks.** 8 windows of those routed matmuls (full K = 3072/8192, partial edge tiles, N = 4/8/16, 2 with random host stalls and garbage on don't-care cycles) were stepped on `fpga/systolic` (the register model the RTL co-simulates against). The int32 results are bit-exact vs the MAX op output. The stepped cycles equal the formula, plus stall cycles where stalls are on.
- **Device backend (lane G1-C libphantom.dylib).** The same MAX op with `PHANTOM_BACKEND=device` at N = 16/8/4 on real layer-0 weight windows is bit-exact vs the stub, and the measured cycles equal the formula.
- **Routed MLP layer.** Layer 0's SwiGLU MLP (gate_up + down on the phantom) vs a numpy W8A8 reference has rel. err 5.4e-3. The difference comes only from GPU bf16 SiLU rounding feeding down_proj. With synthetic outlier-heavy activations, the error vs the bf16 MLP is 8.6% (cos 0.9963).

### Whole model: greedy tokens and prefill logits vs GPU bf16 (`pixi run maxop-demo`)

8 prompts × 8 greedy tokens (EOS ignored), plus the 512-token bench prompt (`bench/prompt_512.txt`).

| Run | Route | Greedy all-8 match | Matching-prefix tokens | Prefill top-1 | Prefill logit cos (min) | KL (mean) |
| --- | --- | --- | --- | --- | --- | --- |
| ph_l0 (stub) | layer 0 MLP | 7/8 | 63/64 | 8/8 | 0.99908 | 1.4e-3 |
| ph_all (stub) | all 112 projections | 5/8 | 56/64 | 8/8 | 0.97354 | 1.7e-2 |
| fq_all (W8A8 in f32 on GPU) | all 112 projections | 5/8 | 56/64 | 8/8 | 0.97354 | 1.7e-2 |
| **ph_all vs fq_all** | all | **8/8** | **64/64** | 8/8 | 1.00000 | 0 |
| **dev_l0 (libphantom, N=16) vs ph_l0** | layer 0 MLP | **2/2** (3 tokens) | 6/6 | 2/2 | 1.00000 | 0 |
| ph_all, 512-token prompt | all | next token matches | — | 1/1 | 0.99885 | — |

Reading:
- **The routing is exact at model level.** The phantom path (`ph_all`) and the same W8A8 math done on the GPU (`fq_all`) give identical tokens and logits. The cycle-accurate device run reproduces the stub run's tokens.
- **Every gap vs bf16 is the W8A8 quantization**, not the tile:
  - One layer's MLP: 1 of 64 tokens diverges (late in one prompt).
  - Every projection: 5/8 prompts match all 8 tokens, the first token always matches, and 56/64 tokens are on the matching prefix.
  - Naive per-token/per-channel W8A8 with no outlier handling (no SmoothQuant) is the cause.
- So the roadmap criterion "greedy output matches the GPU-only run" holds against the W8A8 GPU reference (8/8). Against bf16 it holds for every first token, but not for all continuations when every layer is quantized.

### Cycles and projected TTFT (512-token prefill, every projection routed: 112 matmuls)

| Shape (×28 each) | cycles/call N=4 | N=8 | N=16 |
| --- | --- | --- | --- |
| qkv 512×3072×5120 | 5.05e8 | 1.27e8 | 3.19e7 |
| o 512×3072×3072 | 3.03e8 | 7.61e7 | 1.92e7 |
| gate_up 512×3072×16384 | 1.62e9 | 4.06e8 | 1.02e8 |
| down 512×8192×3072 | 8.06e8 | 2.02e8 | 5.06e7 |
| **total** | **9.05e10** | **2.27e10** | **5.71e9** |

Projected TTFT is 632 ms (MAX bf16 HTTP TTFT, 512 tok) − (routed MAC share × 538 ms in-process bf16 prefill) + tile time. Tile time is max(cycles/fmax, link bytes/BW), run serially. The fmax values are synthesized ECP5: N=4 84.2, N=8 67.5, N=16 60.4 MHz. Two I/O cases:
- "minimal": each operand crosses the link once (4.65 GB in total), which needs on-board buffers.
- "streamed": the v1 tile as specified, with no operand buffers (2N int8 per beat per pass): 723 / 362 / 182 GB for N = 4/8/16.

| Route | N | compute only | USB 2.0 (40 MB/s) minimal / streamed | USB 3 (400 MB/s) minimal / streamed | PCIe-class (4 GB/s) streamed |
| --- | --- | --- | --- | --- | --- |
| all:all (98.4% of prefill MACs) | 4 | 1074 s | 1074 s / 18079 s | 1074 s / 1808 s | 1074 s |
| | 8 | 336 s | 336 s / 9059 s | 336 s / 906 s | 336 s |
| | 16 | 94.6 s | 116 s / 4550 s | 94.6 s / 455 s | 94.6 s |
| 0:mlp (one layer, 2.6%) | 16 | 3.1 s | 3.6 s / 122 s | 3.1 s / 12.8 s | 3.1 s |

The measured GPU TTFT is **0.632 s** (bf16, HTTP) and 0.538 s in-process. The best phantom case (N=16, all projections) is ~95 s, **~150× slower** than the GPU. Even routing one layer's MLP adds ~2.5 s at N=16, a 5× TTFT regression. As specified, with no operand buffers, the tile is I/O-bound at USB 3 and below. With minimal I/O it is compute-bound everywhere except N=16 over USB 2.0. The host-side functional stub (CPU, ~90 GMAC/s) runs the all-routed 512-token prefill in 16.9 s wall; that is simulation cost, not a projection.

## MAX custom-op notes for this nightly (26.8.0.dev2026110805 / mojo 1.3.0.dev2026110805)

- **Registration.** `import extensibility` and `@extensibility.register("name")` on a struct with `@staticmethod def execute(out: OutputTensor[...], in...: InputTensor[...], ctx: DeviceContext) raises`. Read dims with `t.dim_size[i]()` and pointers with `t.unsafe_ptr()`. Pass a kernel directory (with `__init__.mojo`) as `custom_extensions`, or `F.custom(..., custom_extensions=path)` in ModuleV3. MAX `mojo precompile`s it on first use.
- **Threading.** `parallelize` lives in `max.algorithm` (not `std.algorithm`). Closures use capture lists, `def work(i: Int) {var a, var n}:`, and you call it as `parallelize(work, n)`. `@parameter` closures are rejected.
- **Call-site ids.** A `StaticString` struct parameter creates one kernel specialization per value, so the call-site id is passed as a tiny `int32[1]` tensor instead.
- **FFI.** Mojo ASAP destruction freed a `List` whose `unsafe_ptr()` was passed as an `Int` to an FFI memcpy, before the call read it. That produced silently wrong results on the device backend. Keep it alive with `_ = buf^` after the call.
- **Custom architectures.** `dataclasses.replace(llama_modulev3_arch, modules=...)` copies the derived `pipeline_model` (built from the *stock* decoder). Pass `pipeline_model=None` or the custom decoder is silently ignored. Custom archs register with `allow_override=True`, and lazy built-ins defer to them.
- **Mixed devices and eager mode.** Mixed-device graphs work: GPU quantize, then `.to(CPU())`, then the CPU custom op, then `.to(gpu)`, all inside the pipeline graph. In eager mode, the interpreter refuses a custom op that spans devices and falls back to compiling (a warning, still correct).
- **Quantization rounding.** The GPU quantizer (division + round) differs from numpy on a few .5 ties, so int32 exactness is asserted on the op's own int8 operands.

## What real-hardware integration would need

- The `device` backend already goes through the C ABI that a real FPGA driver would export (lane G1-C keeps the same symbols). Swapping `PHANTOM_LIB` is the integration point.
- For usable throughput: offline int8 weights (today they are re-quantized in-graph on every forward), on-board operand buffers (the "minimal" I/O column), batching the routed calls per layer, and an int8-native host path. Even then, the compute-only projection is ~150× slower than the GPU. G1-E (ECP5 over USB) is a correctness demonstration, not a performance one.
