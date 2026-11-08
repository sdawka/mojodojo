# quality/ (G2-M3 quality measurements: speculative acceptance α and perplexity)

Load-insensitive quality metrics for the G2-M3 candidates ranked in `roofline/ROOFLINE.md` (§6 M4, M8).
Report: [QUALITY.md](QUALITY.md). Any wall-clock numbers in logs/JSON were taken on a shared machine and
are **not** speed results (`timing_authoritative: false`).

All MLX commands use the bench uv project (mlx 0.32.4, mlx-lm 0.32.2); llama.cpp is commit `a691b23`.

```bash
P=/Users/sdawka/Code/llming; UVR="uv run --project $P/mlx python"

# --- alpha (M4), MLX, greedy, 9 prompts x k in {1,2,3,4}, 192 new tokens (stops at EOS)
for pair in llama3b-4bit_1b-4bit llama3b-8bit_1b-4bit qwen4b-8bit_0.6b-8bit qwen4b-8bit_0.6b-4bit qwen4b-4bit_0.6b-4bit; do
  $UVR $P/quality/alpha.py --pair $pair            # -> results/alpha_raw/<pair>.json
done
$UVR $P/quality/greedy_diag.py --pair qwen4b-8bit_0.6b-8bit   # explains spec != plain greedy mismatches (near-tie logits)

# --- alpha (M4), llama.cpp llama-speculative, Q4_K_M 3B target + Q4_K_M 1B draft, same prompts
$UVR $P/quality/dump_prompts.py                    # -> prompts/*.txt (Llama chat template rendered)
$UVR $P/quality/alpha_llamacpp.py                  # -> results/alpha_raw/llamacpp_q4km_3b_1b.json

# --- perplexity (M8): wikitext-2 raw test, ctx 512, all chunks, llama-perplexity method in both runtimes
$P/quality/run_ppl.sh                              # llama.cpp F16/Q8_0/Q4_K_M/Q4_0 + all MLX models -> logs/, results/ppl_raw/
$UVR $P/quality/ppl_mlx.py --model <mlx model dir> --name <family>-mlx-<fmt>   # one MLX model

# extra MLX 4-bit variants tried to get under the 3% gate
$UVR -m mlx_lm convert --hf-path $P/models/mlx-Llama-3.2-3B-Instruct-bf16 --mlx-path $P/models/mlx-Llama-3.2-3B-Instruct-4bit-g32 -q --q-group-size 32
$UVR -m mlx_lm convert --hf-path $P/models/mlx-Llama-3.2-3B-Instruct-bf16 --mlx-path $P/models/mlx-Llama-3.2-3B-Instruct-4bit-mixed_4_6 -q --quant-predicate mixed_4_6
$UVR $P/quality/make_mlx_quant.py --src $P/models/mlx-Llama-3.2-3B-Instruct-bf16 --dst $P/models/mlx-Llama-3.2-3B-Instruct-4bit-mixed_4_6_emb6 --embed-bits 6
$UVR $P/quality/make_mlx_quant.py --src $P/models/mlx-Llama-3.2-3B-Instruct-bf16 --dst $P/models/mlx-Llama-3.2-3B-Instruct-4bit-emb8 --embed-bits 8 --no-mixed
$UVR -m mlx_lm awq --model $P/models/mlx-Llama-3.2-3B-Instruct-bf16 --mlx-path $P/models/mlx-Llama-3.2-3B-Instruct-4bit-awq_emb6 \
     --bits 4 --group-size 64 --embed-bits 6 --embed-group-size 64   # calibration text: ~/.cache/mlx-lm/calibration_v5.txt

# --- collect -> results/alpha.json, results/perplexity.json (+ markdown tables on stdout)
$UVR $P/quality/summarize.py
```

### Gemma 3 (second goal-2 model; QUALITY.md §5)

Models live in `models/gemma3/` (exact repo + revision + provenance in `models/gemma3/DOWNLOADS.json`; Gemma Terms
of Use). Downloads are weights/configs only (`hf download <repo> --revision <sha> --include '*.json' '*.safetensors'
'tokenizer*'`); no code from a model repo is executed (no `trust_remote_code`).

```bash
P=/Users/sdawka/Code/llming; G=$P/models/gemma3; LC=$P/llamacpp/src
# --- conversions (llama.cpp convert script from the llama.cpp checkout, model dirs contain no .py)
python $LC/convert_hf_to_gguf.py $G/gemma-3-4b-it --outtype bf16 --outfile $G/gguf/gemma-3-4b-it-BF16.gguf
python $LC/convert_hf_to_gguf.py $G/gemma-3-4b-it-qat-unquantized --outtype bf16 --outfile $G/gguf/gemma-3-4b-it-qat-BF16.gguf
$LC/build/bin/llama-quantize $G/gguf/gemma-3-4b-it-BF16.gguf $G/gguf/gemma-3-4b-it-Q4_K_M.gguf Q4_K_M    # also Q8_0, Q4_0 -> -plainQ4_0
$LC/build/bin/llama-quantize $G/gguf/gemma-3-4b-it-qat-BF16.gguf $G/gguf/gemma-3-4b-it-qat-localQ4_0.gguf Q4_0
#   (macOS is case-insensitive: never name a local quant gemma-3-4b-it-Q4_0.gguf next to Google's gemma-3-4b-it-q4_0.gguf)
uv run --project $P/mlx python -m mlx_lm convert --hf-path $G/gemma-3-4b-it --mlx-path $G/mlx-gemma-3-4b-it-bf16 --dtype bfloat16
uv run --project $P/mlx python -m mlx_lm convert --hf-path $G/gemma-3-4b-it --mlx-path $G/mlx-gemma-3-4b-it-4bit -q   # / --q-bits 8 -> -8bit
uv run --project $P/mlx python -m mlx_lm convert --hf-path $G/gemma-3-4b-it-qat-unquantized --mlx-path $G/mlx-gemma-3-4b-it-qat-bf16 --dtype bfloat16
uv run --project $P/mlx python $P/quality/make_mlx_quant.py --src $G/mlx-gemma-3-4b-it-qat-bf16 \
       --dst $G/mlx-gemma-3-4b-it-qat-q40grid --q40-grid --embed-bits 6       # exact Q4_0 grid (1B draft: --embed-bits 8)

# --- perplexity (same method), KLD fidelity, alpha
$P/quality/run_ppl.sh gemma                       # or gemma-llamacpp / gemma-mlx; SKIP_DONE=1 skips finished MLX runs
$P/quality/run_kld_gemma.sh                       # llama.cpp --kl-divergence + quality/kld_mlx.py, first 64 chunks
uv run --project $P/mlx python $P/quality/kld_mlx.py --ref $G/mlx-gemma-3-4b-it-bf16 --model $G/mlx-gemma-3-4b-it-4bit --name gemma4b-mlx-4bit
uv run --project $P/mlx python $P/quality/dump_prompts.py --family gemma     # -> prompts/gemma/*.txt (Gemma chat template)
$P/quality/run_alpha_gemma.sh                     # MLX pairs (alpha.py --pair gemma4b-<target>_<draft>) + llama.cpp pairs
uv run --project $P/mlx python $P/quality/alpha.py --pair gemma4b-qatq40grid_270mqat-4bit   # one MLX pair (full-KVCache fix on)
uv run --project $P/mlx python $P/quality/alpha_llamacpp.py --target models/gemma3/gguf/gemma-3-4b-it-qat-localQ4_0.gguf \
       --draft models/gemma3/gguf/gemma-3-270m-it-qat-localQ4_0.gguf --prompts quality/prompts/gemma \
       --name llamacpp_gemma4b-qatlocal_270mqat-q4_0
uv run --project $P/mlx python $P/quality/greedy_diag.py --pair gemma4b-qatq40grid_1b-4bit

# --- predictions + collect (summarize -> gemma_predict -> summarize attaches predictions to alpha.json)
uv run --project $P/mlx python $P/quality/summarize.py
uv run --project $P/roofline python $P/quality/gemma_predict.py              # -> results/gemma_predictions.json
uv run --project $P/mlx python $P/quality/summarize.py
```

| path | contents |
|---|---|
| `alpha.py` | MLX α (Gemma pairs use full `KVCache` per layer, see QUALITY.md §5.4): `mlx_lm.generate.speculative_generate_step` (what `stream_generate(draft_model=...)` wraps), from_draft flags → per-step accounting; greedy-equivalence check vs `generate_step` |
| `greedy_diag.py` | for each spec≠plain mismatch: top-2 logit margin of the target at the first differing token (M=1 vs batched forward) |
| `alpha_llamacpp.py`, `dump_prompts.py`, `prompts/` | llama.cpp α via `llama-speculative` (`--temp 0 --spec-draft-p-min 0 --spec-draft-n-min 0`) |
| `ppl_mlx.py` | MLX perplexity, llama-perplexity method (non-overlapping 512-token chunks, BOS reset, score 2nd half, fp32 NLL); saves per-token NLL for paired Δ SE |
| `run_ppl.sh` | all perplexity runs, sequential |
| `make_mlx_quant.py` | mixed_4_6 / plain 4-bit with a 6- or 8-bit tied embedding (Q4_K_M-like); `--q40-grid`: exact llama.cpp Q4_0 grid as MLX affine g32 (for QAT checkpoints) |
| `kld_mlx.py`, `run_kld_gemma.sh` | KL(ref‖quant) + same-top-1 on the ppl chunks (MLX script; llama.cpp `--kl-divergence`) -> `results/kld_raw/`, `perplexity.json` → `kld` |
| `run_alpha_gemma.sh`, `gemma_predict.py` | Gemma α pairs; Gemma roofline predictions (4B/1B/270M specs, Q4_0-f16emb and q40g32 formats) -> `results/gemma_predictions.json` |
| `summarize.py` | builds `results/alpha.json` + `results/perplexity.json`; adds roofline-predicted tok/s at the measured E (from `roofline/results/model.json` t_iter + calibration) |
| `results/` | `alpha.json`, `perplexity.json` (meta: `authoritative_quality: true`, `timing_authoritative: false`), `alpha_raw/`, `ppl_raw/` (per-model JSON + per-token NLL `.npy`) |
| `logs/` | raw stdout/stderr of every run |

Data: `models/datasets/wiki.test.raw` = rows of `Salesforce/wikitext` `wikitext-2-raw-v1/test` parquet joined
as-is (1,287,656 bytes, sha256 `bbf94c53…964574a`). Models downloaded to `models/` (mlx-community conversions;
`models/gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf` from `unsloth/Llama-3.2-1B-Instruct-GGUF`); Q8_0/Q4_0 GGUFs
made locally with `llama-quantize` from the F16 GGUF.
