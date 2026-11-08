### Fitted parameters

| fit | eta_d | c_d (ms/token, 28 layers) | eta_p | c_http (ms) |
|---|---:|---:|---:|---:|
| all 7 rows | 0.814 | 1.52 | 0.837 | 10.4 |
| without MAX bf16 | 0.796 | 1.09 | 0.846 | -22.4 |
| without MLX 4bit | 0.791 | -0.24 | 0.857 | 21.6 |
| without MLX bf16 | 0.825 | 1.77 | 0.851 | 18.8 |
| without llama.cpp F16 | 0.821 | 1.67 | 0.810 | 42.1 |
| without llama.cpp Q4_K_M | 0.832 | 3.04 | 0.826 | 4.5 |
| 16-bit rows only (c_d fixed 0) | 0.794 | 0 | 0.851 | 18.3 |

### Validation: decode tok/s (Llama 3.2 3B, phase-1)

| row | measured | raw roofline | err | fit (all) | err | **leave-group-out** | **err** | 16-bit-only fit | err | check (LOGO, 20%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| llama-bench F16 | 15.2 | 19.5 | +28.1% | 15.5 | +1.8% | 15.6 | +2.4% | 15.5 | +1.6% | pass |
| llama-bench Q4_K_M | 48.8 | 60.9 | +24.9% | 46.1 | -5.5% | 43.9 | -9.9% | 48.4 | -0.9% | pass |
| llama-server F16 (HTTP) | 15.4 | 19.3 | +25.7% | 15.4 | -0.1% | 15.4 | +0.4% | 15.3 | -0.3% | pass |
| MAX bf16 (HTTP) | 15.3 | 18.9 | +23.5% | 15.0 | -1.8% | 14.8 | -3.3% | 15.0 | -2.0% | pass |
| max benchmark bf16 (HTTP) | 15.4 | 18.9 | +22.6% | 15.0 | -2.4% | 14.8 | -3.9% | 15.0 | -2.7% | pass |
| MLX bf16 | 14.9 | 19.5 | +30.9% | 15.5 | +4.0% | 15.6 | +5.0% | 15.5 | +3.9% | pass |
| MLX 4bit | 48.8 | 68.7 | +41.0% | 51.6 | +5.7% | 55.1 | +12.9% | 54.6 | +11.9% | pass |

### Validation: TTFT ms

| row | method | measured | raw roofline | err | fit (all) | err | **leave-group-out** | **err** | check (LOGO, 20%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| llama-bench F16 | llama-bench: TTFT=512/pp512 (one 512-token ubatch, logits for last token); tg128 on empty KV | 362 | 339 | -6.4% | 405 | +11.8% | 418 | +15.5% | pass |
| llama-bench Q4_K_M | llama-bench (as above) | 375 | 326 | -13.0% | 390 | +3.9% | 395 | +5.2% | pass |
| llama-server F16 (HTTP) | HTTP client TTFT; 513 tokens (nonce) > n_ubatch 512 -> second 1-token ubatch = one extra full weight pass | 444 | 391 | -12.1% | 481 | +8.1% | 525 | +18.2% | pass |
| MAX bf16 (HTTP) | HTTP client TTFT, single prefill pass (ReturnLogits.LAST_TOKEN) | 632 | 469 | -25.8% | 571 | -9.7% | 532 | -15.9% | pass |
| max benchmark bf16 (HTTP) | max benchmark random dataset, mean input 534 tok, 6% prefix-cache hits | 633 | 488 | -22.9% | 594 | -6.2% | 555 | -12.4% | pass |
| MLX bf16 | in-process; mlx_lm prefill of 511 tokens (no lm_head) + one M=1 step for the last prompt token | 459 | 362 | -21.0% | 436 | -4.9% | 429 | -6.4% | pass |
| MLX 4bit | in-process; mlx_lm prefill of 511 tokens (no lm_head) + one M=1 step for the last prompt token | 329 | 251 | -23.6% | 302 | -8.0% | 294 | -10.4% | pass |

### Runtime calibration factors (measured / fitted prediction; used only to anchor candidate rankings)

| runtime | decode tok/s factor | TTFT factor | rows |
|---|---:|---:|---|
| mlx | 0.953 | 1.069 | MLX bf16, MLX 4bit |
| llama.cpp | 1.013 | 0.927 | llama-bench F16, llama-bench Q4_K_M, llama-server F16 (HTTP) |
| max | 1.022 | 1.087 | MAX bf16 (HTTP), max benchmark bf16 (HTTP) |
| coreml-ane | 1.000 | 1.000 | none (factor 1) |

### Per-model accounting (text-only; decode GB = weight bytes read per token incl. lm_head)

| model | params (B) | embedding (B) | layers | d / ffn | heads q/kv x hd | vocab | bf16 GB | 8-bit g64 GB | Q4_K_M GB | MLX 4-bit GB | KV KiB/token | KV @8k ctx (MB) | prefill 512 GFLOP (linear / attn) |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| llama3.2-1b | 1.24 | 0.26 tied | 16 | 2048 / 8192 | 32/8 x 64 | 128256 | 2.47 | 1.31 | 0.80 | 0.70 | 32 | 268 | 997 / 17 |
| llama3.2-3b | 3.21 | 0.39 tied | 28 | 3072 / 8192 | 24/8 x 128 | 128256 | 6.43 | 3.41 | 2.01 | 1.81 | 112 | 940 | 2887 / 45 |
| qwen3-0.6b | 0.60 | 0.16 tied | 28 | 1024 / 3072 | 16/8 x 128 | 151936 | 1.19 | 0.63 | 0.39 | 0.34 | 112 | 940 | 451 / 30 |
| qwen3-1.7b | 1.72 | 0.31 tied | 28 | 2048 / 6144 | 16/8 x 128 | 151936 | 3.44 | 1.83 | 1.10 | 0.97 | 112 | 940 | 1444 / 30 |
| qwen3-4b | 4.02 | 0.39 tied | 36 | 2560 / 9728 | 32/8 x 128 | 151936 | 8.04 | 4.27 | 2.49 | 2.26 | 144 | 1208 | 3721 / 77 |
| gemma3-1b | 1.00 | 0.30 tied | 26 | 1152 / 6912 | 4/1 x 256 (SWA 512, 1 global per 6) | 262144 | 2.00 | 1.06 | 0.67 | 0.56 | 26 | 45 | 715 / 14 |
| gemma3-4b | 3.88 | 0.67 tied | 34 | 2560 / 10240 | 8/4 x 256 (SWA 1024, 1 global per 6) | 262208 | 7.76 | 4.12 | 2.48 | 2.18 | 136 | 289 | 3287 / 37 |

### llama3.2-3b: candidates

Reference: decode 48.8 tok/s, in-process TTFT 329 ms (measured phase-1). 15% target: decode >= 56.1 tok/s, TTFT <= 286 ms.

| decode config | weights GB | pred tok/s (raw) | calibrated | gain | >=15%? | ppl risk (prior) | support / proxy |
|---|---:|---:|---:|---:|---|---|---|
| mlx/4bit-g128 | 1.71 | 54.3 | 51.7 | +6.0% | no | high: ~+4-10% | proxy: g128 kernel unmeasured: g64 per-byte rate used |
| mlx/4bit | 1.81 | 51.6 | 49.2 | +0.8% | no | high: ~+3-8% |  |
| llama.cpp/Q4_0 | 1.91 | 46.1 | 46.7 | -4.2% | no | high: ~+4-10% | proxy: Q6_K output: mean of Q4_K/Q8_0 rates |
| llama.cpp/Q4_K_M | 2.01 | 45.0 | 45.6 | -6.6% | no | high: ~+2-5% (small models are more sensitive) | proxy: Q6_K parts: mean of Q4_K/Q8_0 rates |
| llama.cpp/Q6_K | 2.64 | 35.5 | 35.9 | -26.4% | no | low-med: ~+0.3-0.8% | proxy: Q6_K unmeasured: mean of Q4_K/Q8_0 per-byte rates |
| llama.cpp/Q8_0 | 3.32 | 29.0 | 29.4 | -39.8% | no | low: ~+0.1-0.3% | proxy: Q6_K output tensor: mean of Q4_K/Q8_0 rates |
| mlx/8bit | 3.41 | 28.6 | 27.2 | -44.2% | no | low: ~+0.1-0.3% |  |
| max/nvfp4 | 2.37 | 26.5 | 27.1 | -44.5% | no | high: ~+3-6% (PTQ, no calibration) | encoding float4_e2m1fnx2 listed for legacy Llama3 arch; Apple W4A16 path only exercised by FLUX.2 so far; needs NVFP4 (modelopt) checkpoint |
| max/fp8 | 3.61 | 26.3 | 26.9 | -44.9% | no | low-med: ~+0.2-1% (per-tensor scale only on the Apple path) | encoding float8_e4m3fn listed for legacy Llama3 arch (not ModuleV3); needs an FP8 checkpoint with per-tensor scale; proxy: FP8 W8A16 Apple kernel not in atlas: bf16 GEMV per-byte rate (decode) and bf16 MMA rate x1/0.9 (prefill, dequant to transient bf16) assumed |
| max/int8-w8a8 | 3.61 | 19.8 | 20.2 | -58.5% | no | medium: ~+0.5-2% (dynamic per-token int8 activations; Llama-3 activation outliers) | NOT a pipeline encoding: op exists (max.nn quant int8 W8A8, Apple M5 only); needs a custom Llama graph + offline weight quantization |
| llama.cpp/F16 | 6.43 | 15.4 | 15.6 | -68.1% | no | none: ~0% |  |
| max/bf16 | 6.43 | 15.0 | 15.4 | -68.5% | no | none: 0% |  |
| mlx/bf16 | 6.43 | 15.5 | 14.8 | -69.7% | no | none: 0% |  |

| TTFT config (512-token prompt) | pred ms (raw) | calibrated in-process | calibrated HTTP | gain | >=15%? | ppl risk (prior) | support / proxy |
|---|---:|---:|---:|---:|---|---|---|
| max/int8-w8a8 (one pass, last-token logits) | 193 | 210 | 220 | +56.8% | yes | medium | NOT a pipeline encoding: op exists (max.nn quant int8 W8A8, Apple M5 only); needs a custom Llama graph + offline weight quantization |
| ane/w8a8 (one pass, last-token logits) | 213 | 213 | 223 | +54.7% | yes | medium | whole-model Core ML conversion required (coremltools); not tried; proxy: ANE shapes are 3B attn_q_o / mlp_up_down chains of 8 linears; attention on ANE assumed at 50% of ANE fp16 linear rate; KV handoff to GPU not modeled (unified memory) |
| ane/lut4 (one pass, last-token logits) | 267 | 267 | 278 | +23.0% | yes | high | whole-model Core ML conversion required (coremltools); not tried; proxy: ANE shapes are 3B attn_q_o / mlp_up_down chains of 8 linears; attention on ANE assumed at 50% of ANE fp16 linear rate; KV handoff to GPU not modeled (unified memory) |
| mlx/4bit-g128 (one pass, last-token logits) | 285 | 305 | 316 | +7.8% | no | high | proxy: g128 kernel unmeasured: g64 per-byte rate used |
| mlx/4bit (one pass, last-token logits) | 286 | 305 | 316 | +7.7% | no | high |  |
| mlx/4bit (mlx_lm as benchmarked: +1 decode step) | 302 | 323 | - | +1.7% | no | high |  |
| llama.cpp/Q4_0 (one pass, last-token logits) | 361 | 334 | 344 | -1.5% | no | high | proxy: Q6_K output: mean of Q4_K/Q8_0 rates |
| mlx/8bit (one pass, last-token logits) | 318 | 340 | 350 | -3.1% | no | low |  |
| llama.cpp/Q8_0 (one pass, last-token logits) | 381 | 353 | 363 | -6.7% | no | low | proxy: Q6_K output tensor: mean of Q4_K/Q8_0 rates |
| llama.cpp/Q6_K (one pass, last-token logits) | 386 | 357 | 368 | -8.0% | no | low-med | proxy: Q6_K unmeasured: mean of Q4_K/Q8_0 per-byte rates |
| llama.cpp/Q4_K_M (one pass, last-token logits) | 390 | 362 | 372 | -9.0% | no | high | proxy: Q6_K parts: mean of Q4_K/Q8_0 rates |
| mlx/8bit (mlx_lm as benchmarked: +1 decode step) | 348 | 372 | - | -11.6% | no | low |  |
| llama.cpp/F16 (one pass, last-token logits) | 405 | 375 | 386 | -12.3% | no | none |  |
| ane/w8 (one pass, last-token logits) | 398 | 398 | 408 | -17.3% | no | low-med | whole-model Core ML conversion required (coremltools); not tried; proxy: ANE shapes are 3B attn_q_o / mlp_up_down chains of 8 linears; attention on ANE assumed at 50% of ANE fp16 linear rate; KV handoff to GPU not modeled (unified memory) |
| mlx/bf16 (one pass, last-token logits) | 380 | 406 | 417 | -19.0% | no | none |  |
| mlx/bf16 (mlx_lm as benchmarked: +1 decode step) | 436 | 467 | - | -29.5% | no | none |  |
| max/bf16 (one pass, last-token logits) | 559 | 608 | 618 | -45.8% | no | none |  |
| ane/fp16 (one pass, last-token logits) | 609 | 609 | 620 | -46.0% | no | none | whole-model Core ML conversion required (coremltools); not tried; proxy: ANE shapes are 3B attn_q_o / mlp_up_down chains of 8 linears; attention on ANE assumed at 50% of ANE fp16 linear rate; KV handoff to GPU not modeled (unified memory) |
| max/nvfp4 (one pass, last-token logits) | 568 | 617 | 627 | -46.6% | no | high | encoding float4_e2m1fnx2 listed for legacy Llama3 arch; Apple W4A16 path only exercised by FLUX.2 so far; needs NVFP4 (modelopt) checkpoint |
| max/fp8 (one pass, last-token logits) | 619 | 672 | 683 | -51.1% | no | low-med | encoding float8_e4m3fn listed for legacy Llama3 arch (not ModuleV3); needs an FP8 checkpoint with per-tensor scale; proxy: FP8 W8A16 Apple kernel not in atlas: bf16 GEMV per-byte rate (decode) and bf16 MMA rate x1/0.9 (prefill, dequant to transient bf16) assumed |

Speculative decoding (draft llama3.2-1b MLX 4-bit; best k per alpha; calibrated tok/s and gain):

| target | alpha=0.5 | alpha=0.6 | alpha=0.7 | alpha=0.8 | alpha=0.9 |
|---|---:|---:|---:|---:|---:|
| mlx/4bit | 49.7 (k=1, +1.9%) | 53.0 (k=1, +8.7%) | 56.3 (k=1, +15.5%) | 62.6 (k=2, +28.3%) | 72.0 (k=3, +47.5%) |
| mlx/8bit | 32.7 (k=1, -33.0%) | 36.3 (k=2, -25.6%) | 40.8 (k=3, -16.4%) | 47.5 (k=3, -2.6%) | 55.4 (k=3, +13.5%) |
| mlx/bf16 | 20.7 (k=2, -57.6%) | 23.6 (k=3, -51.7%) | 27.4 (k=3, -43.8%) | 32.5 (k=4, -33.4%) | 41.7 (k=7, -14.5%) |

Draft step 7.7 ms; target step (M=1) for reference is in the decode table.

### llama3.2-1b: candidates

Reference: decode 124.2 tok/s, in-process TTFT 114 ms (calibrated prediction of phase-1-style configs (MLX 4-bit / llama.cpp Q4_K_M), not measured). 15% target: decode >= 142.8 tok/s, TTFT <= 99 ms.

| decode config | weights GB | pred tok/s (raw) | calibrated | gain | >=15%? | ppl risk (prior) | support / proxy |
|---|---:|---:|---:|---:|---|---|---|
| mlx/4bit-g128 | 0.66 | 136.8 | 130.4 | +5.0% | no | high: ~+4-10% | proxy: g128 kernel unmeasured: g64 per-byte rate used |
| mlx/4bit | 0.70 | 130.3 | 124.2 | +0.0% | no | high: ~+3-8% |  |
| llama.cpp/Q4_0 | 0.76 | 113.4 | 114.9 | -7.5% | no | high: ~+4-10% | proxy: Q6_K output: mean of Q4_K/Q8_0 rates |
| llama.cpp/Q4_K_M | 0.80 | 110.8 | 112.3 | -9.6% | no | high: ~+2-5% (small models are more sensitive) | proxy: Q6_K parts: mean of Q4_K/Q8_0 rates |
| llama.cpp/Q6_K | 1.01 | 90.3 | 91.5 | -26.3% | no | low-med: ~+0.3-0.8% | proxy: Q6_K unmeasured: mean of Q4_K/Q8_0 per-byte rates |
| llama.cpp/Q8_0 | 1.25 | 75.5 | 76.5 | -38.4% | no | low: ~+0.1-0.3% | proxy: Q6_K output tensor: mean of Q4_K/Q8_0 rates |
| mlx/8bit | 1.31 | 73.1 | 69.7 | -43.9% | no | low: ~+0.1-0.3% |  |
| max/nvfp4 | 1.07 | 64.2 | 65.6 | -47.2% | no | high: ~+3-6% (PTQ, no calibration) | encoding float4_e2m1fnx2 listed for legacy Llama3 arch; Apple W4A16 path only exercised by FLUX.2 so far; needs NVFP4 (modelopt) checkpoint |
| max/fp8 | 1.50 | 63.8 | 65.2 | -47.5% | no | low-med: ~+0.2-1% (per-tensor scale only on the Apple path) | encoding float8_e4m3fn listed for legacy Llama3 arch (not ModuleV3); needs an FP8 checkpoint with per-tensor scale; proxy: FP8 W8A16 Apple kernel not in atlas: bf16 GEMV per-byte rate (decode) and bf16 MMA rate x1/0.9 (prefill, dequant to transient bf16) assumed |
| max/int8-w8a8 | 1.50 | 50.3 | 51.4 | -58.6% | no | medium: ~+0.5-2% (dynamic per-token int8 activations; Llama-3 activation outliers) | NOT a pipeline encoding: op exists (max.nn quant int8 W8A8, Apple M5 only); needs a custom Llama graph + offline weight quantization |
| max/bf16 | 2.47 | 39.4 | 40.3 | -67.6% | no | none: 0% |  |
| llama.cpp/F16 | 2.47 | 39.6 | 40.1 | -67.7% | no | none: ~0% |  |
| mlx/bf16 | 2.47 | 40.0 | 38.2 | -69.3% | no | none: 0% |  |

| TTFT config (512-token prompt) | pred ms (raw) | calibrated in-process | calibrated HTTP | gain | >=15%? | ppl risk (prior) | support / proxy |
|---|---:|---:|---:|---:|---|---|---|
| max/int8-w8a8 (one pass, last-token logits) | 71 | 77 | 88 | +47.2% | yes | medium | NOT a pipeline encoding: op exists (max.nn quant int8 W8A8, Apple M5 only); needs a custom Llama graph + offline weight quantization |
| ane/w8a8 (one pass, last-token logits) | 78 | 78 | 89 | +45.3% | yes | medium | whole-model Core ML conversion required (coremltools); not tried; proxy: ANE shapes are 3B attn_q_o / mlp_up_down chains of 8 linears; attention on ANE assumed at 50% of ANE fp16 linear rate; KV handoff to GPU not modeled (unified memory) |
| ane/lut4 (one pass, last-token logits) | 96 | 96 | 106 | +18.6% | yes | high | whole-model Core ML conversion required (coremltools); not tried; proxy: ANE shapes are 3B attn_q_o / mlp_up_down chains of 8 linears; attention on ANE assumed at 50% of ANE fp16 linear rate; KV handoff to GPU not modeled (unified memory) |
| mlx/4bit-g128 (one pass, last-token logits) | 100 | 107 | 117 | +6.2% | no | high | proxy: g128 kernel unmeasured: g64 per-byte rate used |
| mlx/4bit (one pass, last-token logits) | 100 | 107 | 118 | +6.1% | no | high |  |
| mlx/4bit (mlx_lm as benchmarked: +1 decode step) | 106 | 114 | - | +0.0% | no | high |  |
| llama.cpp/Q4_0 (one pass, last-token logits) | 126 | 117 | 127 | -2.6% | no | high | proxy: Q6_K output: mean of Q4_K/Q8_0 rates |
| mlx/8bit (one pass, last-token logits) | 112 | 120 | 130 | -5.3% | no | low |  |
| llama.cpp/Q8_0 (one pass, last-token logits) | 133 | 123 | 134 | -7.7% | no | low | proxy: Q6_K output tensor: mean of Q4_K/Q8_0 rates |
| llama.cpp/Q6_K (one pass, last-token logits) | 135 | 125 | 135 | -8.9% | no | low-med | proxy: Q6_K unmeasured: mean of Q4_K/Q8_0 per-byte rates |
| llama.cpp/Q4_K_M (one pass, last-token logits) | 136 | 126 | 137 | -9.9% | no | high | proxy: Q6_K parts: mean of Q4_K/Q8_0 rates |
| mlx/8bit (mlx_lm as benchmarked: +1 decode step) | 123 | 132 | - | -13.7% | no | low |  |
| llama.cpp/F16 (one pass, last-token logits) | 143 | 132 | 143 | -14.0% | no | none |  |
| mlx/bf16 (one pass, last-token logits) | 134 | 144 | 154 | -20.9% | no | none |  |
| ane/w8 (one pass, last-token logits) | 144 | 144 | 155 | -21.2% | no | low-med | whole-model Core ML conversion required (coremltools); not tried; proxy: ANE shapes are 3B attn_q_o / mlp_up_down chains of 8 linears; attention on ANE assumed at 50% of ANE fp16 linear rate; KV handoff to GPU not modeled (unified memory) |
| mlx/bf16 (mlx_lm as benchmarked: +1 decode step) | 154 | 165 | - | -31.1% | no | none |  |
| max/nvfp4 (one pass, last-token logits) | 198 | 216 | 226 | -47.3% | no | high | encoding float4_e2m1fnx2 listed for legacy Llama3 arch; Apple W4A16 path only exercised by FLUX.2 so far; needs NVFP4 (modelopt) checkpoint |
| max/bf16 (one pass, last-token logits) | 201 | 218 | 228 | -47.8% | no | none |  |
| ane/fp16 (one pass, last-token logits) | 219 | 219 | 229 | -48.1% | no | none | whole-model Core ML conversion required (coremltools); not tried; proxy: ANE shapes are 3B attn_q_o / mlp_up_down chains of 8 linears; attention on ANE assumed at 50% of ANE fp16 linear rate; KV handoff to GPU not modeled (unified memory) |
| max/fp8 (one pass, last-token logits) | 222 | 241 | 251 | -52.8% | no | low-med | encoding float8_e4m3fn listed for legacy Llama3 arch (not ModuleV3); needs an FP8 checkpoint with per-tensor scale; proxy: FP8 W8A16 Apple kernel not in atlas: bf16 GEMV per-byte rate (decode) and bf16 MMA rate x1/0.9 (prefill, dequant to transient bf16) assumed |

### qwen3-4b: candidates

Reference: decode 39.2 tok/s, in-process TTFT 421 ms (calibrated prediction of phase-1-style configs (MLX 4-bit / llama.cpp Q4_K_M), not measured). 15% target: decode >= 45.0 tok/s, TTFT <= 366 ms.

| decode config | weights GB | pred tok/s (raw) | calibrated | gain | >=15%? | ppl risk (prior) | support / proxy |
|---|---:|---:|---:|---:|---|---|---|
| mlx/4bit-g128 | 2.14 | 43.2 | 41.2 | +5.2% | no | high: ~+4-10% | proxy: g128 kernel unmeasured: g64 per-byte rate used |
| mlx/4bit | 2.26 | 41.1 | 39.2 | +0.0% | no | high: ~+3-8% |  |
| llama.cpp/Q4_0 | 2.36 | 37.1 | 37.6 | -4.1% | no | high: ~+4-10% | proxy: Q6_K output: mean of Q4_K/Q8_0 rates |
| llama.cpp/Q4_K_M | 2.49 | 36.2 | 36.7 | -6.4% | no | high: ~+2-5% (small models are more sensitive) | proxy: Q6_K parts: mean of Q4_K/Q8_0 rates |
| llama.cpp/Q6_K | 3.30 | 28.3 | 28.6 | -26.9% | no | low-med: ~+0.3-0.8% | proxy: Q6_K unmeasured: mean of Q4_K/Q8_0 per-byte rates |
| llama.cpp/Q8_0 | 4.18 | 23.0 | 23.3 | -40.5% | no | low: ~+0.1-0.3% | proxy: Q6_K output tensor: mean of Q4_K/Q8_0 rates |
| max/fp8 | 4.41 | 21.5 | 21.9 | -44.0% | no | low-med: ~+0.2-1% (per-tensor scale only on the Apple path) | encoding float8_e4m3fn listed for legacy Llama3 arch (not ModuleV3); needs an FP8 checkpoint with per-tensor scale; proxy: FP8 W8A16 Apple kernel not in atlas: bf16 GEMV per-byte rate (decode) and bf16 MMA rate x1/0.9 (prefill, dequant to transient bf16) assumed |
| mlx/8bit | 4.27 | 22.8 | 21.7 | -44.6% | no | low: ~+0.1-0.3% |  |
| max/int8-w8a8 | 4.41 | 16.0 | 16.3 | -58.3% | no | medium: ~+0.5-2% (dynamic per-token int8 activations; Llama-3 activation outliers) | NOT a pipeline encoding: op exists (max.nn quant int8 W8A8, Apple M5 only); needs a custom Llama graph + offline weight quantization |
| llama.cpp/F16 | 8.04 | 12.3 | 12.4 | -68.3% | no | none: ~0% |  |
| max/bf16 | 8.04 | 12.0 | 12.3 | -68.6% | no | none: 0% |  |
| mlx/bf16 | 8.04 | 12.4 | 11.8 | -69.9% | no | none: 0% |  |

| TTFT config (512-token prompt) | pred ms (raw) | calibrated in-process | calibrated HTTP | gain | >=15%? | ppl risk (prior) | support / proxy |
|---|---:|---:|---:|---:|---|---|---|
| max/int8-w8a8 (one pass, last-token logits) | 253 | 275 | 285 | +53.2% | yes | medium | NOT a pipeline encoding: op exists (max.nn quant int8 W8A8, Apple M5 only); needs a custom Llama graph + offline weight quantization |
| ane/w8a8 (one pass, last-token logits) | 278 | 278 | 289 | +51.2% | yes | medium | whole-model Core ML conversion required (coremltools); not tried; proxy: ANE shapes are 3B attn_q_o / mlp_up_down chains of 8 linears; attention on ANE assumed at 50% of ANE fp16 linear rate; KV handoff to GPU not modeled (unified memory) |
| ane/lut4 (one pass, last-token logits) | 350 | 350 | 360 | +20.4% | yes | high | whole-model Core ML conversion required (coremltools); not tried; proxy: ANE shapes are 3B attn_q_o / mlp_up_down chains of 8 linears; attention on ANE assumed at 50% of ANE fp16 linear rate; KV handoff to GPU not modeled (unified memory) |
| mlx/4bit-g128 (one pass, last-token logits) | 372 | 398 | 408 | +5.8% | no | high | proxy: g128 kernel unmeasured: g64 per-byte rate used |
| mlx/4bit (one pass, last-token logits) | 372 | 398 | 408 | +5.8% | no | high |  |
| mlx/4bit (mlx_lm as benchmarked: +1 decode step) | 394 | 421 | - | +0.0% | no | high |  |
| llama.cpp/Q4_0 (one pass, last-token logits) | 469 | 434 | 445 | -3.1% | no | high | proxy: Q6_K output: mean of Q4_K/Q8_0 rates |
| mlx/8bit (one pass, last-token logits) | 413 | 441 | 452 | -4.6% | no | low |  |
| llama.cpp/Q8_0 (one pass, last-token logits) | 495 | 459 | 469 | -8.2% | no | low | proxy: Q6_K output tensor: mean of Q4_K/Q8_0 rates |
| llama.cpp/Q6_K (one pass, last-token logits) | 502 | 465 | 475 | -9.4% | no | low-med | proxy: Q6_K unmeasured: mean of Q4_K/Q8_0 per-byte rates |
| llama.cpp/Q4_K_M (one pass, last-token logits) | 507 | 470 | 480 | -10.4% | no | high | proxy: Q6_K parts: mean of Q4_K/Q8_0 rates |
| mlx/8bit (mlx_lm as benchmarked: +1 decode step) | 452 | 483 | - | -12.9% | no | low |  |
| llama.cpp/F16 (one pass, last-token logits) | 525 | 486 | 497 | -13.5% | no | none |  |
| ane/w8 (one pass, last-token logits) | 516 | 516 | 526 | -18.4% | no | low-med | whole-model Core ML conversion required (coremltools); not tried; proxy: ANE shapes are 3B attn_q_o / mlp_up_down chains of 8 linears; attention on ANE assumed at 50% of ANE fp16 linear rate; KV handoff to GPU not modeled (unified memory) |
| mlx/bf16 (one pass, last-token logits) | 491 | 525 | 535 | -19.8% | no | none |  |
| mlx/bf16 (mlx_lm as benchmarked: +1 decode step) | 563 | 603 | - | -30.1% | no | none |  |
| ane/fp16 (one pass, last-token logits) | 788 | 788 | 798 | -46.6% | no | none | whole-model Core ML conversion required (coremltools); not tried; proxy: ANE shapes are 3B attn_q_o / mlp_up_down chains of 8 linears; attention on ANE assumed at 50% of ANE fp16 linear rate; KV handoff to GPU not modeled (unified memory) |
| max/bf16 (one pass, last-token logits) | 726 | 789 | 799 | -46.6% | no | none |  |
| max/fp8 (one pass, last-token logits) | 803 | 872 | 883 | -51.7% | no | low-med | encoding float8_e4m3fn listed for legacy Llama3 arch (not ModuleV3); needs an FP8 checkpoint with per-tensor scale; proxy: FP8 W8A16 Apple kernel not in atlas: bf16 GEMV per-byte rate (decode) and bf16 MMA rate x1/0.9 (prefill, dequant to transient bf16) assumed |

Speculative decoding (draft qwen3-0.6b MLX 4-bit; best k per alpha; calibrated tok/s and gain):

| target | alpha=0.5 | alpha=0.6 | alpha=0.7 | alpha=0.8 | alpha=0.9 |
|---|---:|---:|---:|---:|---:|
| mlx/4bit | 45.3 (k=1, +15.7%) | 49.4 (k=2, +26.1%) | 55.2 (k=2, +41.0%) | 63.9 (k=3, +63.1%) | 74.4 (k=3, +90.0%) |
| mlx/8bit | 29.9 (k=2, -23.6%) | 33.9 (k=3, -13.3%) | 39.5 (k=3, +0.9%) | 46.0 (k=3, +17.6%) | 53.6 (k=3, +37.0%) |
| mlx/bf16 | 18.4 (k=3, -53.1%) | 21.3 (k=3, -45.6%) | 24.9 (k=4, -36.4%) | 30.6 (k=5, -21.9%) | 41.0 (k=7, +4.6%) |

Draft step 5.3 ms; target step (M=1) for reference is in the decode table.

### qwen3-1.7b: candidates

Reference: decode 83.8 tok/s, in-process TTFT 165 ms (calibrated prediction of phase-1-style configs (MLX 4-bit / llama.cpp Q4_K_M), not measured). 15% target: decode >= 96.3 tok/s, TTFT <= 144 ms.

| decode config | weights GB | pred tok/s (raw) | calibrated | gain | >=15%? | ppl risk (prior) | support / proxy |
|---|---:|---:|---:|---:|---|---|---|
| mlx/4bit-g128 | 0.91 | 92.0 | 87.7 | +4.7% | no | high: ~+4-10% | proxy: g128 kernel unmeasured: g64 per-byte rate used |
| mlx/4bit | 0.97 | 87.9 | 83.8 | +0.0% | no | high: ~+3-8% |  |
| llama.cpp/Q4_0 | 1.05 | 77.9 | 79.0 | -5.8% | no | high: ~+4-10% | proxy: Q6_K output: mean of Q4_K/Q8_0 rates |
| llama.cpp/Q4_K_M | 1.10 | 76.2 | 77.2 | -7.9% | no | high: ~+2-5% (small models are more sensitive) | proxy: Q6_K parts: mean of Q4_K/Q8_0 rates |
| llama.cpp/Q6_K | 1.41 | 62.1 | 63.0 | -24.9% | no | low-med: ~+0.3-0.8% | proxy: Q6_K unmeasured: mean of Q4_K/Q8_0 per-byte rates |
| llama.cpp/Q8_0 | 1.75 | 52.0 | 52.6 | -37.2% | no | low: ~+0.1-0.3% | proxy: Q6_K output tensor: mean of Q4_K/Q8_0 rates |
| mlx/8bit | 1.83 | 50.7 | 48.3 | -42.4% | no | low: ~+0.1-0.3% |  |
| max/fp8 | 2.03 | 44.8 | 45.7 | -45.4% | no | low-med: ~+0.2-1% (per-tensor scale only on the Apple path) | encoding float8_e4m3fn listed for legacy Llama3 arch (not ModuleV3); needs an FP8 checkpoint with per-tensor scale; proxy: FP8 W8A16 Apple kernel not in atlas: bf16 GEMV per-byte rate (decode) and bf16 MMA rate x1/0.9 (prefill, dequant to transient bf16) assumed |
| max/int8-w8a8 | 2.03 | 34.9 | 35.7 | -57.4% | no | medium: ~+0.5-2% (dynamic per-token int8 activations; Llama-3 activation outliers) | NOT a pipeline encoding: op exists (max.nn quant int8 W8A8, Apple M5 only); needs a custom Llama graph + offline weight quantization |
| llama.cpp/F16 | 3.44 | 27.9 | 28.2 | -66.3% | no | none: ~0% |  |
| max/bf16 | 3.44 | 27.2 | 27.8 | -66.8% | no | none: 0% |  |
| mlx/bf16 | 3.44 | 28.2 | 26.9 | -67.9% | no | none: 0% |  |

| TTFT config (512-token prompt) | pred ms (raw) | calibrated in-process | calibrated HTTP | gain | >=15%? | ppl risk (prior) | support / proxy |
|---|---:|---:|---:|---:|---|---|---|
| max/int8-w8a8 (one pass, last-token logits) | 102 | 110 | 121 | +49.8% | yes | medium | NOT a pipeline encoding: op exists (max.nn quant int8 W8A8, Apple M5 only); needs a custom Llama graph + offline weight quantization |
| ane/w8a8 (one pass, last-token logits) | 112 | 112 | 122 | +47.8% | yes | medium | whole-model Core ML conversion required (coremltools); not tried; proxy: ANE shapes are 3B attn_q_o / mlp_up_down chains of 8 linears; attention on ANE assumed at 50% of ANE fp16 linear rate; KV handoff to GPU not modeled (unified memory) |
| ane/lut4 (one pass, last-token logits) | 139 | 139 | 150 | +18.7% | yes | high | whole-model Core ML conversion required (coremltools); not tried; proxy: ANE shapes are 3B attn_q_o / mlp_up_down chains of 8 linears; attention on ANE assumed at 50% of ANE fp16 linear rate; KV handoff to GPU not modeled (unified memory) |
| mlx/4bit-g128 (one pass, last-token logits) | 145 | 155 | 166 | +6.6% | no | high | proxy: g128 kernel unmeasured: g64 per-byte rate used |
| mlx/4bit (one pass, last-token logits) | 145 | 155 | 166 | +6.5% | no | high |  |
| mlx/4bit (mlx_lm as benchmarked: +1 decode step) | 155 | 165 | - | +0.0% | no | high |  |
| llama.cpp/Q4_0 (one pass, last-token logits) | 183 | 170 | 180 | -2.5% | no | high | proxy: Q6_K output: mean of Q4_K/Q8_0 rates |
| mlx/8bit (one pass, last-token logits) | 162 | 173 | 183 | -4.4% | no | low |  |
| llama.cpp/Q8_0 (one pass, last-token logits) | 193 | 179 | 189 | -7.6% | no | low | proxy: Q6_K output tensor: mean of Q4_K/Q8_0 rates |
| llama.cpp/Q6_K (one pass, last-token logits) | 196 | 181 | 192 | -8.8% | no | low-med | proxy: Q6_K unmeasured: mean of Q4_K/Q8_0 per-byte rates |
| llama.cpp/Q4_K_M (one pass, last-token logits) | 198 | 183 | 194 | -9.8% | no | high | proxy: Q6_K parts: mean of Q4_K/Q8_0 rates |
| mlx/8bit (mlx_lm as benchmarked: +1 decode step) | 178 | 191 | - | -13.2% | no | low |  |
| llama.cpp/F16 (one pass, last-token logits) | 207 | 192 | 202 | -13.6% | no | none |  |
| ane/w8 (one pass, last-token logits) | 204 | 204 | 215 | -19.1% | no | low-med | whole-model Core ML conversion required (coremltools); not tried; proxy: ANE shapes are 3B attn_q_o / mlp_up_down chains of 8 linears; attention on ANE assumed at 50% of ANE fp16 linear rate; KV handoff to GPU not modeled (unified memory) |
| mlx/bf16 (one pass, last-token logits) | 195 | 208 | 219 | -20.5% | no | none |  |
| mlx/bf16 (mlx_lm as benchmarked: +1 decode step) | 224 | 239 | - | -30.9% | no | none |  |
| max/bf16 (one pass, last-token logits) | 283 | 308 | 318 | -46.2% | no | none |  |
| ane/fp16 (one pass, last-token logits) | 310 | 310 | 321 | -46.7% | no | none | whole-model Core ML conversion required (coremltools); not tried; proxy: ANE shapes are 3B attn_q_o / mlp_up_down chains of 8 linears; attention on ANE assumed at 50% of ANE fp16 linear rate; KV handoff to GPU not modeled (unified memory) |
| max/fp8 (one pass, last-token logits) | 313 | 340 | 350 | -51.3% | no | low-med | encoding float8_e4m3fn listed for legacy Llama3 arch (not ModuleV3); needs an FP8 checkpoint with per-tensor scale; proxy: FP8 W8A16 Apple kernel not in atlas: bf16 GEMV per-byte rate (decode) and bf16 MMA rate x1/0.9 (prefill, dequant to transient bf16) assumed |

Speculative decoding (draft qwen3-0.6b MLX 4-bit; best k per alpha; calibrated tok/s and gain):

| target | alpha=0.5 | alpha=0.6 | alpha=0.7 | alpha=0.8 | alpha=0.9 |
|---|---:|---:|---:|---:|---:|
| mlx/4bit | 79.1 (k=1, -5.6%) | 84.4 (k=1, +0.7%) | 89.6 (k=1, +7.0%) | 97.7 (k=2, +16.7%) | 111.1 (k=3, +32.6%) |
| mlx/8bit | 54.8 (k=1, -34.6%) | 59.4 (k=2, -29.1%) | 66.3 (k=2, -20.8%) | 76.4 (k=3, -8.8%) | 89.0 (k=3, +6.2%) |
| mlx/bf16 | 35.5 (k=2, -57.6%) | 39.7 (k=2, -52.6%) | 46.2 (k=3, -44.9%) | 54.2 (k=4, -35.4%) | 68.0 (k=7, -18.9%) |

Draft step 5.3 ms; target step (M=1) for reference is in the decode table.

### gemma3-4b: candidates

Reference: decode 40.7 tok/s, in-process TTFT 367 ms (calibrated prediction of phase-1-style configs (MLX 4-bit / llama.cpp Q4_K_M), not measured). 15% target: decode >= 46.8 tok/s, TTFT <= 320 ms.

| decode config | weights GB | pred tok/s (raw) | calibrated | gain | >=15%? | ppl risk (prior) | support / proxy |
|---|---:|---:|---:|---:|---|---|---|
| mlx/4bit-g128 | 2.06 | 44.9 | 42.8 | +5.2% | no | high: ~+4-10% | proxy: g128 kernel unmeasured: g64 per-byte rate used |
| mlx/4bit | 2.18 | 42.7 | 40.7 | +0.0% | no | high: ~+3-8% |  |
| llama.cpp/Q4_0 | 2.36 | 37.5 | 38.0 | -6.6% | no | high: ~+4-10% | proxy: Q6_K output: mean of Q4_K/Q8_0 rates |
| llama.cpp/Q4_K_M | 2.48 | 36.5 | 37.0 | -9.1% | no | high: ~+2-5% (small models are more sensitive) | proxy: Q6_K parts: mean of Q4_K/Q8_0 rates |
| llama.cpp/Q6_K | 3.18 | 29.4 | 29.7 | -26.9% | no | low-med: ~+0.3-0.8% | proxy: Q6_K unmeasured: mean of Q4_K/Q8_0 per-byte rates |
| llama.cpp/Q8_0 | 3.96 | 24.2 | 24.6 | -39.7% | no | low: ~+0.1-0.3% | proxy: Q6_K output tensor: mean of Q4_K/Q8_0 rates |
| mlx/8bit | 4.12 | 23.6 | 22.5 | -44.6% | no | low: ~+0.1-0.3% |  |
| max/int8-w8a8 | 4.55 | 16.3 | 16.6 | -59.2% | no | medium: ~+0.5-2% (dynamic per-token int8 activations; Llama-3 activation outliers) | NOT a pipeline encoding: op exists (max.nn quant int8 W8A8, Apple M5 only); needs a custom Llama graph + offline weight quantization |
| llama.cpp/F16 | 7.76 | 12.7 | 12.9 | -68.4% | no | none: ~0% |  |
| max/bf16 | 7.76 | 12.5 | 12.8 | -68.5% | no | none: 0% |  |
| mlx/bf16 | 7.76 | 12.9 | 12.3 | -69.9% | no | none: 0% |  |

| TTFT config (512-token prompt) | pred ms (raw) | calibrated in-process | calibrated HTTP | gain | >=15%? | ppl risk (prior) | support / proxy |
|---|---:|---:|---:|---:|---|---|---|
| max/int8-w8a8 (one pass, last-token logits) | 224 | 244 | 254 | +50.9% | yes | medium | NOT a pipeline encoding: op exists (max.nn quant int8 W8A8, Apple M5 only); needs a custom Llama graph + offline weight quantization |
| ane/w8a8 (one pass, last-token logits) | 244 | 244 | 255 | +50.3% | yes | medium | whole-model Core ML conversion required (coremltools); not tried; proxy: ANE shapes are 3B attn_q_o / mlp_up_down chains of 8 linears; attention on ANE assumed at 50% of ANE fp16 linear rate; KV handoff to GPU not modeled (unified memory) |
| ane/lut4 (one pass, last-token logits) | 302 | 302 | 313 | +21.6% | yes | high | whole-model Core ML conversion required (coremltools); not tried; proxy: ANE shapes are 3B attn_q_o / mlp_up_down chains of 8 linears; attention on ANE assumed at 50% of ANE fp16 linear rate; KV handoff to GPU not modeled (unified memory) |
| mlx/4bit-g128 (one pass, last-token logits) | 324 | 347 | 357 | +6.0% | no | high | proxy: g128 kernel unmeasured: g64 per-byte rate used |
| mlx/4bit (one pass, last-token logits) | 324 | 347 | 357 | +6.0% | no | high |  |
| mlx/4bit (mlx_lm as benchmarked: +1 decode step) | 344 | 367 | - | +0.0% | no | high |  |
| llama.cpp/Q4_0 (one pass, last-token logits) | 408 | 378 | 389 | -2.8% | no | high | proxy: Q6_K output: mean of Q4_K/Q8_0 rates |
| mlx/8bit (one pass, last-token logits) | 363 | 388 | 399 | -5.4% | no | low |  |
| llama.cpp/Q8_0 (one pass, last-token logits) | 431 | 399 | 410 | -8.0% | no | low | proxy: Q6_K output tensor: mean of Q4_K/Q8_0 rates |
| llama.cpp/Q6_K (one pass, last-token logits) | 437 | 405 | 415 | -9.2% | no | low-med | proxy: Q6_K unmeasured: mean of Q4_K/Q8_0 per-byte rates |
| llama.cpp/Q4_K_M (one pass, last-token logits) | 442 | 410 | 420 | -10.3% | no | high | proxy: Q6_K parts: mean of Q4_K/Q8_0 rates |
| mlx/8bit (mlx_lm as benchmarked: +1 decode step) | 398 | 426 | - | -13.7% | no | low |  |
| llama.cpp/F16 (one pass, last-token logits) | 461 | 428 | 438 | -14.0% | no | none |  |
| ane/w8 (one pass, last-token logits) | 463 | 463 | 473 | -20.5% | no | low-med | whole-model Core ML conversion required (coremltools); not tried; proxy: ANE shapes are 3B attn_q_o / mlp_up_down chains of 8 linears; attention on ANE assumed at 50% of ANE fp16 linear rate; KV handoff to GPU not modeled (unified memory) |
| mlx/bf16 (one pass, last-token logits) | 436 | 467 | 477 | -21.2% | no | none |  |
| mlx/bf16 (mlx_lm as benchmarked: +1 decode step) | 501 | 535 | - | -31.3% | no | none |  |
| max/bf16 (one pass, last-token logits) | 648 | 704 | 714 | -47.8% | no | none |  |
| ane/fp16 (one pass, last-token logits) | 709 | 709 | 719 | -48.2% | no | none | whole-model Core ML conversion required (coremltools); not tried; proxy: ANE shapes are 3B attn_q_o / mlp_up_down chains of 8 linears; attention on ANE assumed at 50% of ANE fp16 linear rate; KV handoff to GPU not modeled (unified memory) |

Speculative decoding (draft gemma3-1b MLX 4-bit; best k per alpha; calibrated tok/s and gain):

| target | alpha=0.5 | alpha=0.6 | alpha=0.7 | alpha=0.8 | alpha=0.9 |
|---|---:|---:|---:|---:|---:|
| mlx/4bit | 44.5 (k=1, +9.4%) | 47.5 (k=1, +16.7%) | 52.4 (k=2, +28.7%) | 59.1 (k=3, +45.3%) | 68.9 (k=3, +69.2%) |
| mlx/8bit | 29.1 (k=2, -28.4%) | 32.6 (k=2, -19.8%) | 37.6 (k=3, -7.6%) | 43.8 (k=3, +7.6%) | 51.0 (k=3, +25.4%) |
| mlx/bf16 | 18.1 (k=2, -55.6%) | 20.9 (k=3, -48.6%) | 24.4 (k=3, -40.2%) | 29.2 (k=5, -28.2%) | 38.4 (k=7, -5.7%) |

Draft step 6.9 ms; target step (M=1) for reference is in the decode table.

### gemma3-1b: candidates

Reference: decode 138.0 tok/s, in-process TTFT 84 ms (calibrated prediction of phase-1-style configs (MLX 4-bit / llama.cpp Q4_K_M), not measured). 15% target: decode >= 158.8 tok/s, TTFT <= 73 ms.

| decode config | weights GB | pred tok/s (raw) | calibrated | gain | >=15%? | ppl risk (prior) | support / proxy |
|---|---:|---:|---:|---:|---|---|---|
| mlx/4bit-g128 | 0.53 | 151.3 | 144.3 | +4.5% | no | high: ~+4-10% | proxy: g128 kernel unmeasured: g64 per-byte rate used |
| mlx/4bit | 0.56 | 144.8 | 138.0 | +0.0% | no | high: ~+3-8% |  |
| llama.cpp/Q4_0 | 0.64 | 124.4 | 126.0 | -8.7% | no | high: ~+4-10% | proxy: Q6_K output: mean of Q4_K/Q8_0 rates |
| llama.cpp/Q4_K_M | 0.67 | 122.0 | 123.6 | -10.5% | no | high: ~+2-5% (small models are more sensitive) | proxy: Q6_K parts: mean of Q4_K/Q8_0 rates |
| llama.cpp/Q6_K | 0.82 | 103.6 | 104.9 | -24.0% | no | low-med: ~+0.3-0.8% | proxy: Q6_K unmeasured: mean of Q4_K/Q8_0 per-byte rates |
| llama.cpp/Q8_0 | 0.99 | 89.2 | 90.3 | -34.6% | no | low: ~+0.1-0.3% | proxy: Q6_K output tensor: mean of Q4_K/Q8_0 rates |
| mlx/8bit | 1.06 | 85.0 | 81.1 | -41.3% | no | low: ~+0.1-0.3% |  |
| max/int8-w8a8 | 1.30 | 58.5 | 59.8 | -56.7% | no | medium: ~+0.5-2% (dynamic per-token int8 activations; Llama-3 activation outliers) | NOT a pipeline encoding: op exists (max.nn quant int8 W8A8, Apple M5 only); needs a custom Llama graph + offline weight quantization |
| max/bf16 | 2.00 | 47.6 | 48.6 | -64.8% | no | none: 0% |  |
| llama.cpp/F16 | 2.00 | 47.3 | 47.9 | -65.3% | no | none: ~0% |  |
| mlx/bf16 | 2.00 | 47.9 | 45.7 | -66.9% | no | none: 0% |  |

| TTFT config (512-token prompt) | pred ms (raw) | calibrated in-process | calibrated HTTP | gain | >=15%? | ppl risk (prior) | support / proxy |
|---|---:|---:|---:|---:|---|---|---|
| max/int8-w8a8 (one pass, last-token logits) | 55 | 59 | 70 | +41.3% | yes | medium | NOT a pipeline encoding: op exists (max.nn quant int8 W8A8, Apple M5 only); needs a custom Llama graph + offline weight quantization |
| ane/w8a8 (one pass, last-token logits) | 60 | 60 | 70 | +39.9% | yes | medium | whole-model Core ML conversion required (coremltools); not tried; proxy: ANE shapes are 3B attn_q_o / mlp_up_down chains of 8 linears; attention on ANE assumed at 50% of ANE fp16 linear rate; KV handoff to GPU not modeled (unified memory) |
| ane/lut4 (one pass, last-token logits) | 72 | 72 | 82 | +16.8% | yes | high | whole-model Core ML conversion required (coremltools); not tried; proxy: ANE shapes are 3B attn_q_o / mlp_up_down chains of 8 linears; attention on ANE assumed at 50% of ANE fp16 linear rate; KV handoff to GPU not modeled (unified memory) |
| mlx/4bit-g128 (one pass, last-token logits) | 73 | 78 | 89 | +7.2% | no | high | proxy: g128 kernel unmeasured: g64 per-byte rate used |
| mlx/4bit (one pass, last-token logits) | 73 | 78 | 89 | +7.1% | no | high |  |
| mlx/4bit (mlx_lm as benchmarked: +1 decode step) | 78 | 84 | - | +0.0% | no | high |  |
| llama.cpp/Q4_0 (one pass, last-token logits) | 92 | 85 | 95 | -1.3% | no | high | proxy: Q6_K output: mean of Q4_K/Q8_0 rates |
| mlx/8bit (one pass, last-token logits) | 83 | 88 | 99 | -5.1% | no | low |  |
| llama.cpp/Q8_0 (one pass, last-token logits) | 97 | 89 | 100 | -6.4% | no | low | proxy: Q6_K output tensor: mean of Q4_K/Q8_0 rates |
| llama.cpp/Q6_K (one pass, last-token logits) | 98 | 91 | 101 | -7.6% | no | low-med | proxy: Q6_K unmeasured: mean of Q4_K/Q8_0 per-byte rates |
| llama.cpp/Q4_K_M (one pass, last-token logits) | 99 | 92 | 102 | -8.6% | no | high | proxy: Q6_K parts: mean of Q4_K/Q8_0 rates |
| llama.cpp/F16 (one pass, last-token logits) | 105 | 97 | 108 | -13.8% | no | none |  |
| mlx/8bit (mlx_lm as benchmarked: +1 decode step) | 91 | 97 | - | -14.1% | no | low |  |
| mlx/bf16 (one pass, last-token logits) | 99 | 106 | 117 | -21.1% | no | none |  |
| ane/w8 (one pass, last-token logits) | 108 | 108 | 119 | -22.7% | no | low-med | whole-model Core ML conversion required (coremltools); not tried; proxy: ANE shapes are 3B attn_q_o / mlp_up_down chains of 8 linears; attention on ANE assumed at 50% of ANE fp16 linear rate; KV handoff to GPU not modeled (unified memory) |
| mlx/bf16 (mlx_lm as benchmarked: +1 decode step) | 114 | 122 | - | -31.5% | no | none |  |
| max/bf16 (one pass, last-token logits) | 149 | 162 | 172 | -48.2% | no | none |  |
| ane/fp16 (one pass, last-token logits) | 163 | 163 | 173 | -48.5% | no | none | whole-model Core ML conversion required (coremltools); not tried; proxy: ANE shapes are 3B attn_q_o / mlp_up_down chains of 8 linears; attention on ANE assumed at 50% of ANE fp16 linear rate; KV handoff to GPU not modeled (unified memory) |

