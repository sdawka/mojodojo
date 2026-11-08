# roofline/ (G2-M2)

Roofline and performance model for 1-4B LLM inference on the Apple M5. It is built from the processor atlas and validated against the phase-1 benchmarks. The report is [ROOFLINE.md](ROOFLINE.md).

```bash
uv run --project /Users/sdawka/Code/llming/roofline python -m roofline.run     # writes results/model.json + results/*.png
uv run --project /Users/sdawka/Code/llming/roofline python -m roofline.report  # refreshes generated tables in ROOFLINE.md
```

The model makes no network calls and does no GPU work. `configs/` caches the HF `config.json` files: Llama 3.2 1B/3B (unsloth mirrors), Qwen3 0.6B/1.7B/4B, and Gemma 3 1B/4B (unsloth mirrors).

Layout:

| path | contents |
|---|---|
| `roofline/accounting.py` | parameters, bytes per token, KV and FLOPs per model and format |
| `roofline/atlas.py` | atlas JSON reader with shape-specific rates |
| `roofline/predict.py` | runtime configs, predictor, and speculative decoding |
| `roofline/run.py` | fit, validation, ranking, model.json |
| `roofline/plots.py` | PNG plots |
| `roofline/report.py` | markdown tables |

## Data contract: `results/model.json`

A report page can read these top-level keys. Units are in the key names. All times are milliseconds unless the key says `_s`.

| key | type | contents |
|---|---|---|
| `meta` | object | `timestamp` (ISO-8601 UTC), `authoritative: false`, `milestone`, `machine`, `inputs` (absolute paths of every atlas, bench and config file read), `note` |
| `assumptions` | object | `attention_rate_fraction_of_16bit_gemm`, `spec_iter_overhead_ms`, `stream_bw_gbps`, `decode_ctx_default`, `prefill_S_default`, `kv_dtype_bytes`, and `formats` (bpw per format) |
| `fitted_params` | object | `all_rows`, `leave_group_out.{group}` and `fit_16bit_only`, each `{eta_d, c_d_ms, eta_p, c_http_ms}`; `how` (text) |
| `validation` | array | one object per phase-1 row (see below) |
| `runtime_calibration` | object | `{mlx, llama.cpp, max, coreml-ane}` → `{decode_tok_s, ttft, rows}` factors (measured / fit) |
| `models` | object | `{model}` → `repo`, `arch`, `params` (counts), `decode_weight_bytes_per_token.{format}`, `kv_bytes_per_token_fp16`, `kv_bytes_at_ctx_fp16.{ctx}`, `prefill_flops.{S}.{linear, attention}`, `notes` |
| `candidates` | object | `{model}` → `reference`, `draft`, `decode[]`, `prefill[]`, `speculative[]` (see below) |
| `prefill_length_sweep_llama3.2-3b_ms` | object | `{config}` → `{S}` → predicted prefill ms |
| `measurement_requests` | object | predicted values for the measurements listed in ROOFLINE.md §6 |

Each `validation[]` object has:

* `id`, `group`, `rt`, `method`, `src` (bench file), `S`, `ctx`
* `measured.{tok_s, ttft_ms}`
* four prediction blocks, `raw_roofline`, `fit_all`, `leave_group_out` and `fit_16bit_only`, each `{tok_s, ttft_ms, err_tok_s, err_ttft}` (errors are fractions, predicted/measured - 1)
* `decode_breakdown_ms` and `prefill_breakdown_ms`, keyed by role (`attn_q_o`, `attn_kv`, `mlp_up_gate`, `mlp_down`, `lm_head`, `kv_read` | `attention`)

The completion check uses `leave_group_out.err_*` with |err| ≤ 0.20.

`candidates.{model}` contains:

* `reference`: `{decode_tok_s, ttft_inproc_ms, kind}`
* `decode[]`: `{config, fmt, tok_s, tok_s_cal, gain_vs_baseline, reaches_15pct, ppl_risk: [level, text], weight_gb, proxy[], supported}`, sorted by `tok_s_cal`
* `prefill[]`: `{config, fmt, ttft_ms, ttft_cal_ms, ttft_http_cal_ms|null, gain_vs_baseline, reaches_15pct, ppl_risk, proxy[], supported}`, sorted by `ttft_cal_ms`
* `speculative[]`: `{target, draft, k, alpha, E_tokens, t_draft_ms, t_verify_ms, t_iter_ms, tok_s, tok_s_cal, gain_vs_baseline, reaches_15pct, ppl_risk}`

Notes:

* `*_cal` values are raw predictions multiplied by the runtime calibration factor. Gains are computed from the `*_cal` values.
* `ppl_risk` values are priors, not measurements.
* Plots: `results/validation.png`, `breakdown.png`, `ranking.png`, `speculative.png`, `prefill_sweep.png`.
