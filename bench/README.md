# Llama 3.2 3B Instruct: llama.cpp vs MLX vs MAX on Apple M5

Machine: Apple M5, 32 GB unified memory, macOS 26.5 (25F71), arm64.
Run benchmarks **one runtime at a time with nothing else running** (no builds,
downloads, or other servers in the background; plug in AC power).

## Versions

| component | version |
|---|---|
| llama.cpp | commit `a691b2345844ce1b430201fc5fd404ce062e05f2` (2026-11-08), ggml 0.29.0, libllama 0.9.0; AppleClang 21.0.0; CMake 4.4.4; Release, `GGML_METAL=ON` (+ Accelerate BLAS) |
| MLX | mlx 0.32.4, mlx-lm 0.32.2 (uv project `/Users/sdawka/Code/llming/mlx`, Python 3.12.10, uv 0.12.23) |
| tokenizer check | transformers 5.22.2 (same uv project) |

Build (already done):

```bash
cd /Users/sdawka/Code/llming/llamacpp/src
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_CURL=OFF
cmake --build build --config Release -j 8
build/bin/llama-bench --list-devices   # -> MTL0: Apple M5 (25559 MiB)
```

## Models (`/Users/sdawka/Code/llming/models/`)

| path | source (HF repo @ rev) | size |
|---|---|---|
| `Llama-3.2-3B-Instruct/` (safetensors bf16, for MAX + tokenizer) | `unsloth/Llama-3.2-3B-Instruct` @ 006f5dcd13 | 6.43 GB |
| `gguf/Llama-3.2-3B-Instruct-F16.gguf` | `unsloth/Llama-3.2-3B-Instruct-GGUF` @ e7d0997e49 | 6.43 GB |
| `gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf` | same | 2.02 GB |
| `mlx-Llama-3.2-3B-Instruct-bf16/` | `mlx-community/Llama-3.2-3B-Instruct-bf16` @ 6d88ba4302 | 6.43 GB |
| `mlx-Llama-3.2-3B-Instruct-4bit/` | `mlx-community/Llama-3.2-3B-Instruct-4bit` @ 7f0dc925e0 (group size 64, affine) | 1.81 GB |

Note: llama.cpp uses F16 (not BF16) as its 16-bit baseline; F16 is the
better-supported fast path on Metal. Q4_K_M (~4.8 bpw, mixed) and MLX 4-bit
(4 bit + bf16 scales/biases per 64, ~4.5 bpw) are not identical quantizations.

## Shared methodology

* Prompt: `bench/prompt_512.txt` = **511 text tokens + BOS = 512 tokens** with
  the Llama 3.2 tokenizer, used raw (no chat template). The endpoint client
  prepends a random 3-digit nonce per request to defeat prefix caching, so it
  sees 513 tokens.
* Generation: 128 tokens, **greedy** (temperature 0 / argmax), **EOS ignored**
  so every run generates exactly 128 tokens.
* Reps: 1 warmup + 3 measured reps (llama-bench: its own warmup + `-r 3`).
* pp tok/s = prompt tokens / prefill time. tg tok/s = decode rate excluding the
  first token: `(n_gen - 1) / (t_last - t_first)` (llama-bench tg128 = 128 decode
  steps on an empty context).
* Differences to keep in mind: llama-bench uses synthetic tokens and measures tg on
  an empty KV cache; MLX and the endpoint client decode after the 512-token prompt
  (slightly slower attention). Endpoint TTFT is measured over HTTP from the client.
* Metric names and definitions follow MAX's `max benchmark` (benchmark-model skill,
  `.claude/skills/benchmark-model/references/metrics.md`; same as vLLM's
  benchmark_serving), measured single-stream (concurrency 1):
  * **TTFT**: request start to first streamed token (prefill latency).
  * **TPOT**: `(e2e - TTFT) / (n_gen - 1)`, the decode time per token after the
    first. **tg tok/s = 1000 / TPOT** in every row of the summary.
  * **ITL**: gaps between consecutive streamed chunks (mean / p50 / p99; endpoint
    client only).
  * **mean_output_throughput**: `n_gen / e2e` per request (endpoint client only;
    at concurrency 1 this is what `max benchmark` reports).
* Peak memory: MLX `mx.get_peak_memory()` (reset after warmup, per format);
  llama.cpp max RSS from `/usr/bin/time -l` (includes mmapped weights wrapped as
  Metal buffers); endpoint: not observable from the client, pass `--peak-mem-gb`
  if you measure the server process yourself.

## Running

All commands are absolute; results land in `bench/results/`.

```bash
# 1. llama.cpp (llama-bench, pp512 + tg128, -r 3, F16 and Q4_K_M) -> llamacpp_<fmt>_<ts>.json + .time.txt
bash /Users/sdawka/Code/llming/bench/run_llamacpp.sh

# 2. MLX (bf16 and 4bit, in-process) -> mlx_<ts>.json
uv run --project /Users/sdawka/Code/llming/mlx python /Users/sdawka/Code/llming/bench/run_mlx.py
#    options: --formats bf16 4bit  --reps 3 --warmup 1 --gen-tokens 128

# 3. Any OpenAI-compatible server (e.g. MAX serve on :8000) -> <label>_<format>_<ts>.json
uv run --project /Users/sdawka/Code/llming/mlx python /Users/sdawka/Code/llming/bench/run_openai_endpoint.py \
    --base-url http://localhost:8000/v1 --model <served model id> --label max --format bf16
#    options: --api chat (use /v1/chat/completions; adds template tokens)
#             --no-cache-bust  --no-ignore-eos  --peak-mem-gb <GB>  --reps 3 --warmup 1
#    Sends temperature=0, top_p=1, seed=0, ignore_eos=true, min_tokens=128,
#    stream_options.include_usage, cache_prompt=false. If the server rejects the
#    request with 400/422, it retries once without the optional fields and records
#    them under settings.dropped_optional_fields; check gen_tokens == 128 in output.
#    Same client can be used against llama-server for an apples-to-apples HTTP baseline:
#      /Users/sdawka/Code/llming/llamacpp/src/build/bin/llama-server \
#          -m /Users/sdawka/Code/llming/models/gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf -ngl 99 --port 8089 -c 4096
#      ... run_openai_endpoint.py --base-url http://localhost:8089/v1 --model llama --label llama-server --format q4_k_m

# 3b. Cross-check with MAX's own load generator (same server, same shape, concurrency 1).
#     Needs the MAX env (run from the mojo-max project; prefix `pixi run` if it is a pixi project).
#     --tokenizer is needed if the served name is an alias rather than an HF id.
curl -s http://localhost:8000/v1/models    # use the served model id below
max benchmark --backend modular --base-url http://localhost:8000 \
    --model <served model id> --tokenizer unsloth/Llama-3.2-3B-Instruct \
    --endpoint /v1/completions --dataset-name random \
    --random-input-len 512 --random-output-len 128 --max-output-len 128 \
    --num-prompts 16 --max-concurrency 1 --temperature 0 \
    --result-filename /Users/sdawka/Code/llming/bench/results/maxbench_bf16.json \
    --metadata runtime=max-benchmark format=bf16
#     Compare its mean/median TTFT and TPOT against run_openai_endpoint.py --label max.
#     Expect close agreement; differences come from random token content and from
#     whether its random dataset ignores EOS (check total_output_tokens == 16*128).
#     summarize.py picks up this JSON too (it detects the mean_ttft_ms key).
#     Optional throughput sweep (not comparable to the single-stream baselines):
#       --max-concurrency 1,2,4,8 --num-prompts 64 (server --max-batch-size >= 8)

# 4. Merge everything into a markdown table (newest file per runtime/format wins)
python3 /Users/sdawka/Code/llming/bench/summarize.py -o /Users/sdawka/Code/llming/bench/results/summary.md
```

Peak memory of a server process (optional, for `--peak-mem-gb`): e.g.
`footprint -p <pid>` or watch "Memory" in Activity Monitor during the run.
