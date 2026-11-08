| runtime | format | pp tok/s | tg tok/s | TPOT (ms) | TTFT (ms) | peak mem (GB) | version | source |
|---|---|---:|---:|---:|---:|---:|---|---|
| llama.cpp | f16 | 1413.8 | 15.2 | 65.77 | 362 | 6.68 | a691b23 (MTL,BLAS) | llamacpp_f16_20261108_063515.json |
| llama.cpp | q4_k_m | 1363.5 | 48.8 | 20.50 | 375 | 2.24 | a691b23 (MTL,BLAS) | llamacpp_q4_k_m_20261108_063515.json |
| llamacpp-server | f16 | 1154.5 | 15.4 | 65.09 | 444 | - | http://localhost:8001/v1/completions | llamacpp-server_f16_20261108_063441.json |
| max | bf16 | 810.1 | 15.3 | 65.32 | 632 | - | http://localhost:8000/v1/completions | max_bf16_20261108_063039.json |
| max-benchmark | bf16 | 843.0 | 15.4 | 64.90 | 633 | - | max benchmark | maxbench_bf16.json |
| mlx | 4bit | 1558.3 | 48.8 | 20.50 | 329 | 2.34 | mlx 0.32.4 mlx-lm 0.32.2 | mlx_20261108_062942.json |
| mlx | bf16 | 1116.8 | 14.9 | 67.09 | 459 | 6.78 | mlx 0.32.4 mlx-lm 0.32.2 | mlx_20261108_062942.json |

Notes: llama.cpp pp/tg are llama-bench pp512/tg128 (synthetic tokens); its TTFT = 512/pp.
MLX TTFT = in-process prefill + first token. OpenAI-endpoint TTFT is client-side over HTTP.
tg tok/s = 1000 / TPOT everywhere (TPOT as defined by `max benchmark`: decode time per token after the first).
`max benchmark` rows: pp tok/s ~ mean input tokens / mean TTFT; stamp them with --metadata runtime=... format=...
All decode is greedy; EOS ignored; 128 generated tokens; prompt = bench/prompt_512.txt.
