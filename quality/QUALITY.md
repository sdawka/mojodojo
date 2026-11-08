# G2-M3 quality measurements: speculative acceptance α (M4) and perplexity (M8)

Date: 2026-11-08. Machine: Apple M5, 32 GB, shared with other lanes during these runs. α and perplexity
do not depend on load, so they are authoritative. **No number in this file is a speed measurement.**
The tok/s figures below are roofline-model predictions (`roofline/results/model.json`) evaluated at the
measured α. Wall-clock values in the logs are non-authoritative.

Versions: mlx 0.32.4, mlx-lm 0.32.2, llama.cpp `a691b23`. Scripts and commands: [README.md](README.md).
Data: [results/alpha.json](results/alpha.json), [results/perplexity.json](results/perplexity.json).
Gemma 3 (second goal-2 model, quality only): §5.

## Summary

1. **The phase-1 MLX 4-bit baseline fails the ≤3% perplexity gate: +7.25% vs bf16.** llama.cpp **Q4_K_M
   passes (+1.72%)**, and its phase-1 decode speed was the same 48.8 tok/s. Every 8-bit format passes
   (MLX 8-bit +0.07%, Q8_0 +0.12%, Qwen3-4B 8-bit −0.62%). Qwen3-4B MLX 4-bit fails (+4.60%), and so
   does llama.cpp Q4_0 (+5.57%). Better MLX 4-bit recipes help but still miss the gate: mixed_4_6 with a
   6-bit tied embedding reaches +3.76%, and AWQ
   (mlx_lm, default calibration) reaches +5.24%.
2. **Speculative decoding does not change perplexity.** Greedy speculative output equals greedy plain
   output, apart from exact or near-exact logit ties (§1.3). So a speculative candidate passes or fails
   the gate with its target model.
3. **Measured α (i.i.d. per-token, pooled over 9 prompts):**
   * Llama 3B MLX 4-bit + 1B 4-bit: about 0.80.
   * Llama 3B MLX 8-bit + 1B 4-bit: about 0.82.
   * llama.cpp Q4_K_M 3B + Q4_K_M 1B: about 0.84.
   * Qwen3-4B 8-bit + 0.6B draft: only 0.65 to 0.69. The roofline assumed 0.8 for this pair, its
     "best low-ppl candidate".
   * The raw bench prompt `prompt_512.txt` is near the bottom of the range: α_iid is 0.63 to 0.72 for
     Llama pairs and 0.49 to 0.57 for Qwen. Code and SQL prompts reach 0.9 to 0.96.
4. **The one surviving G2-M3 decode candidate is llama.cpp Q4_K_M 3B + Q4_K_M 1B draft.** It passes the
   ppl gate. Its predicted decode is about 65 tok/s at k=2-3 (+33%) at the pooled α. On the bench
   prompt alone the prediction drops to about 56 tok/s at k=2 (+15%), which is borderline. The prediction
   is a proxy: it uses the MLX 4-bit step costs, because the roofline has no llama.cpp speculative row.
   * MLX 4-bit + 1B is predicted at +27% pooled, but its target fails the gate.
   * MLX 8-bit + 1B passes the gate but is predicted at −2% (k=3).
   * Qwen3-4B 8-bit + 0.6B passes the gate but is predicted at −7% to −20% against the model's predicted
     4-bit Qwen baseline.

## 1. Speculative acceptance α (M4)

### 1.1 Method

* MLX: `mlx_lm.generate.speculative_generate_step` (the generator `stream_generate(draft_model=…,
  num_draft_tokens=k)` wraps). Settings: greedy (default `greedy_sampler`, temperature 0), k ∈ {1,2,3,4},
  192 new tokens, stop at EOS. No prompt hit EOS before 192 tokens. The installed source yields one tuple
  per token, `(token, logprobs, from_draft)`. Each verify step emits n accepted draft tokens
  (`from_draft=True`) followed by one target token (`from_draft=False`), after
  `num_draft = min(max_tokens − ntoks, k)` proposals. `alpha.py` rebuilds every step from these flags.
* llama.cpp: `llama-speculative --temp 0 --spec-draft-n-max k --spec-draft-n-min 0 --spec-draft-p-min 0`,
  with the same prompts. It reports n_drafted and n_accept. The number of steps is estimated as
  n_drafted/k.
* Prompts (9):
  * `raw_bench512`: `bench/prompt_512.txt` used raw, exactly as the phase-1 bench uses it (512 tokens with
    BOS).
  * 8 chat-templated instruct prompts: prose story, Python code, grouped list, Q&A explanation,
    5-bullet summary of the 512-token text (566 tokens), how-to/technical explanation, email, SQL.
  * Qwen3 prompts use `enable_thinking=False`.
* Metrics. The roofline's speedup model (`roofline/roofline/predict.py:spec_decode`) treats α as an i.i.d.
  **per-token** acceptance probability, with E = (1 − α^(k+1))/(1 − α) tokens per verify step. ROOFLINE
  §6 says "α = accepted/drafted". That ratio equals (E−1)/k, which is smaller than the model's α for
  k > 1, so both are reported:
  * `acc_ratio` = accepted / proposed draft tokens.
  * `alpha_iid` = accepted / (accepted + rejections): the maximum-likelihood estimate of the model's α.
    A rejection is a completed step that accepted fewer tokens than it proposed. For llama.cpp, α is
    solved from E.
  * `E` = tokens emitted per verify step (measured).

  Predictions plug the **measured E** into the roofline's per-k iteration time and runtime calibration.
  Use `alpha_iid` (or E), not `acc_ratio`, when reading ROOFLINE's α-indexed tables.

### 1.2 Results (pooled over 9 prompts × 192 tokens)

| pair | k | acc_ratio pooled | per-prompt range | **α_iid pooled** | per-prompt range | E tokens/verify | greedy == plain | roofline-pred tok/s @ measured E | gain vs baseline |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|
| Llama 3B MLX 4bit + 1B 4bit | 1 | 0.780 | 0.63-0.92 | 0.780 | 0.63-0.92 | 1.78 | 8/9 | 59.0 | +21% |
| | 2 | 0.708 | 0.50-0.91 | 0.798 | 0.63-0.93 | 2.41 | 8/9 | 61.9 | +27% |
| | 3 | 0.642 | 0.44-0.86 | 0.801 | 0.65-0.93 | 2.91 | 8/9 | 61.0 | +25% |
| | 4 | 0.580 | 0.36-0.82 | 0.799 | 0.63-0.94 | 3.30 | 8/9 | 54.4 | +12% |
| Llama 3B MLX 8bit + 1B 4bit | 1 | 0.808 | 0.66-0.95 | 0.808 | 0.66-0.95 | 1.81 | 9/9 | 39.4 | -19% |
| | 2 | 0.744 | 0.58-0.95 | 0.820 | 0.69-0.96 | 2.48 | 9/9 | 46.0 | -6% |
| | 3 | 0.660 | 0.47-0.92 | 0.811 | 0.68-0.96 | 2.97 | 9/9 | 47.9 | -2% |
| | 4 | 0.610 | 0.44-0.87 | 0.818 | 0.71-0.94 | 3.42 | 9/9 | 44.3 | -9% |
| llama.cpp Q4_K_M 3B + Q4_K_M 1B | 1 | 0.841 | 0.67-0.98 | 0.851 (from E) | | 1.85 (est.) | n/a | 61.4 (proxy) | +26% |
| | 2 | 0.766 | 0.58-0.94 | 0.840 (from E) | | 2.55 (est.) | n/a | 65.3 (proxy) | +34% |
| | 3 | 0.699 | 0.47-0.91 | 0.834 (from E) | | 3.11 (est.) | n/a | 65.1 (proxy) | +33% |
| | 4 | 0.654 | 0.45-0.87 | 0.840 (from E) | | 3.63 (est.) | n/a | 60.0 (proxy) | +23% |
| Qwen3-4B MLX 8bit + 0.6B 8bit | 1 | 0.671 | 0.54-0.89 | 0.671 | 0.54-0.89 | 1.67 | 3/9 | 28.9 | -26% |
| | 2 | 0.574 | 0.42-0.87 | 0.683 | 0.54-0.90 | 2.14 | 3/9 | 31.3 | -20% |
| | 3 | 0.482 | 0.33-0.77 | 0.684 | 0.53-0.88 | 2.44 | 3/9 | 30.9 | -21% |
| | 4 | 0.423 | 0.26-0.78 | 0.689 | 0.52-0.90 | 2.68 | 3/9 | 27.2 | -30% |
| Qwen3-4B MLX 8bit + 0.6B 4bit | 1 | 0.636 | 0.46-0.92 | 0.636 | 0.46-0.92 | 1.63 | 3/9 | 31.0 | -21% |
| | 2 | 0.535 | 0.36-0.87 | 0.652 | 0.48-0.92 | 2.07 | 3/9 | 35.3 | -10% |
| | 3 | 0.444 | 0.25-0.83 | 0.650 | 0.45-0.91 | 2.32 | 3/9 | 36.3 | -7% |
| | 4 | 0.376 | 0.21-0.74 | 0.651 | 0.46-0.89 | 2.49 | 3/9 | 31.8 | -19% |
| Qwen3-4B MLX 4bit + 0.6B 4bit | 1 | 0.639 | 0.54-0.83 | 0.639 | 0.54-0.83 | 1.64 | 4/9 | 49.5 | +26% |
| | 2 | 0.536 | 0.41-0.79 | 0.650 | 0.52-0.84 | 2.07 | 4/9 | 52.1 | +33% |
| | 3 | 0.441 | 0.31-0.69 | 0.647 | 0.51-0.83 | 2.32 | 4/9 | 50.1 | +28% |
| | 4 | 0.379 | 0.25-0.68 | 0.654 | 0.51-0.86 | 2.50 | 4/9 | 43.3 | +10% |

Notes on the prediction columns:

* Baselines: Llama = the measured 48.8 tok/s. Qwen = the roofline's *predicted* Qwen3-4B MLX 4-bit
  baseline of 39.2 tok/s; there is no phase-1 Qwen measurement.
* The Qwen 0.6B **8-bit** draft row scales the modeled 4-bit draft step by the byte ratio (×1.89). That
  is why it loses to the 4-bit draft despite a higher α.
* The llama.cpp row uses the MLX 4-bit target/draft step costs as a proxy (both runtimes measured
  48.8 tok/s decode).

Per-prompt α_iid at k=3:

| pair | raw_bench512 | prose | code | list | qa | summarize | howto | email | sql |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama 3B 4bit + 1B 4bit | 0.65 | 0.66 | 0.90 | 0.82 | 0.86 | 0.84 | 0.84 | 0.75 | 0.93 |
| Llama 3B 8bit + 1B 4bit | 0.72 | 0.68 | 0.89 | 0.83 | 0.79 | 0.86 | 0.83 | 0.76 | 0.96 |
| llama.cpp Q4_K_M 3B + 1B | 0.68 | | 0.96 | | | | | | 0.96 |
| Qwen3-4B 4bit + 0.6B 4bit | 0.55 | 0.51 | 0.83 | 0.65 | 0.65 | 0.56 | 0.67 | 0.62 | 0.80 |
| Qwen3-4B 8bit + 0.6B 4bit | 0.50 | 0.45 | 0.91 | 0.61 | 0.67 | 0.60 | 0.70 | 0.63 | 0.85 |
| Qwen3-4B 8bit + 0.6B 8bit | 0.54 | 0.53 | 0.88 | 0.66 | 0.70 | 0.61 | 0.74 | 0.66 | 0.87 |

(Only some llama.cpp per-prompt values are listed here; all of them are in `results/alpha.json`.)

**On the bench prompt alone** (`raw_bench512`, the prompt the phase-1 decode numbers used), at the
measured E:

| pair | best k | α_iid | E | pred tok/s | gain |
|---|---:|---:|---:|---:|---:|
| Llama 3B MLX 4bit + 1B | 1 | 0.63 | 1.63 | 53.9 | +11% |
| Llama 3B MLX 8bit + 1B | 2 | 0.76 | 2.38 | 44.0 | -10% |
| llama.cpp Q4_K_M 3B + 1B (proxy) | 2 | 0.70 | 2.18 | 55.9 | +15% |
| Qwen3-4B 8bit + 0.6B 4bit | 2 | 0.51 | 1.76 | 30.1 | -23% |

### 1.3 Greedy equivalence (spec vs plain)

* **Llama 3B 8-bit pair:** 9/9 prompts are token-identical for every k.
* **Llama 3B 4-bit pair:** 8/9. `raw_bench512` diverges at token 55.
* **Qwen pairs:** 3/9 to 4/9. They diverge at tokens 10 to 161.

The first-divergence position is the same for every k and for both Qwen drafts, so the cause is on the
target side, not in the draft or in the acceptance logic. `greedy_diag.py` measured the target's top-2
logit margin at each divergence point:

* every margin is **0 to 0.375**, which is 0 to 3 bf16 ulps at these logit magnitudes;
* margin 0 is an exact tie in the bf16 logits;
* in several cases the M=1 incremental path and the batched (verify-style, M>1) path rank the top 2 in
  the opposite order.

So the mismatches come from tie-breaking and the different accumulation order of the M=1 and M>1
quantized-matmul kernels. They are not a logic bug, and they do not change the output distribution.
Results: `results/alpha_raw/*.greedy_diag.json`.

## 2. Perplexity (M8)

Method, identical across models within each runtime:

* Data: wikitext-2 raw **test**, all of it.
* Chunks: ctx **512**, non-overlapping, with the BOS token reset at the start of each chunk.
* Scoring: only the second half of each chunk is scored (256 tokens per chunk, each with at least 256
  tokens of context), as `llama-perplexity` does.
* Runtimes:
  * llama.cpp: `llama-perplexity -c 512 -b 2048 -ngl 99` (564 chunks).
  * MLX: `ppl_mlx.py`, which reproduces that method with exact fp32 log-softmax NLL. Llama gives 564
    chunks (144,384 scored tokens); Qwen3 gives 583 chunks (149,248 tokens). Δ SE is paired per token.
* `mlx_lm.perplexity` was not used: it evaluates random tulu-3 chat samples, not wikitext.
* Absolute PPL is comparable only within one tokenizer. As a cross-runtime check, MLX bf16 (10.318) and
  llama.cpp F16 (10.315) agree to 0.04%.

| runtime | model | format | PPL | ± | Δ vs same-runtime 16-bit | gate ≤3% |
|---|---|---|---:|---:|---:|---|
| llama.cpp | Llama 3.2 3B | F16 | 10.315 | 0.074 | ref | |
| llama.cpp | Llama 3.2 3B | Q8_0 | 10.327 | 0.074 | +0.12% | PASS |
| llama.cpp | Llama 3.2 3B | **Q4_K_M** | 10.492 | 0.075 | **+1.72%** | **PASS** |
| llama.cpp | Llama 3.2 3B | Q4_0 | 10.889 | 0.078 | +5.57% | FAIL |
| MLX | Llama 3.2 3B | bf16 | 10.318 | 0.074 | ref | |
| MLX | Llama 3.2 3B | 8bit (g64) | 10.326 | 0.074 | +0.07% ± 0.01 | PASS |
| MLX | Llama 3.2 3B | **4bit (g64, phase-1 baseline)** | 11.067 | 0.080 | **+7.25% ± 0.12** | **FAIL** |
| MLX | Llama 3.2 3B | 4bit g32 | 10.923 | 0.079 | +5.86% ± 0.10 | FAIL |
| MLX | Llama 3.2 3B | 4bit + 8-bit embed | 10.989 | 0.079 | +6.50% ± 0.11 | FAIL |
| MLX | Llama 3.2 3B | mixed_4_6 | 10.781 | 0.077 | +4.48% ± 0.10 | FAIL |
| MLX | Llama 3.2 3B | mixed_4_6 + 6-bit embed | 10.706 | 0.077 | +3.76% ± 0.09 | FAIL |
| MLX | Llama 3.2 3B | AWQ 4bit g64 + 6-bit embed | 10.859 | 0.078 | +5.24% ± 0.09 | FAIL |
| MLX | Llama 3.2 1B | 4bit (draft, info) | 16.246 | 0.121 | (no 1B bf16 ref) | |
| MLX | Qwen3-4B | bf16 | 14.103 | 0.131 | ref | |
| MLX | Qwen3-4B | 8bit | 14.015 | 0.129 | −0.62% ± 0.03 | PASS |
| MLX | Qwen3-4B | 4bit | 14.752 | 0.135 | +4.60% ± 0.21 | FAIL |
| MLX | Qwen3-0.6B | 8bit (draft, info) | 21.532 | 0.188 | | |
| MLX | Qwen3-0.6B | 4bit (draft, info) | 25.419 | 0.221 | | |

Observations:

* The roofline priors were "MLX 4-bit +3-8%, Q4_K_M +2-5%, 8-bit +0.1-0.3%". The measurements land at
  the high end for MLX 4-bit (+7.3%) and the low end for Q4_K_M (+1.7%).
* The gap between MLX 4-bit and Q4_K_M is not mainly the tied embedding/lm_head: keeping it at 8 bits
  only reaches +6.5%. Most of the gap comes from llama.cpp's K-quant scale/min search and Q6_K
  promotions. MLX's plain round-to-nearest min-max affine 4-bit lacks both.
* Qwen3-4B 8-bit scores slightly *below* bf16 (−0.6%, about 20 SE). This is real for this text but
  small. Treat it as "no loss".
* Draft perplexity matters only through α. The draft's own quality does not affect output.

## 3. Implications for the G2-M3 candidates

| candidate (roofline rank) | ppl gate | measured α_iid (pooled / bench prompt) | predicted decode at measured E | verdict |
|---|---|---|---|---|
| Llama 3B MLX 4-bit + 1B draft | FAIL (+7.25%, target) | 0.80 / 0.63 | 62 tok/s k=2 (+27%) / bench prompt 54 (+11%) | out unless a 4-bit MLX recipe gets under 3% |
| Llama 3B **llama.cpp Q4_K_M + 1B Q4_K_M draft** | **PASS (+1.72%)** | 0.84 / 0.70 | 65 tok/s k=2-3 (+33%) / bench prompt 56 (+15%) (proxy costs) | **survives; time it first** |
| Llama 3B MLX 8-bit + 1B draft | PASS (+0.07%) | 0.82 / 0.76 | 48 tok/s k=3 (−2%) | fails the speed target (needs α > 0.9) |
| Qwen3-4B MLX 8-bit + 0.6B draft | PASS (−0.62%) | 0.65-0.69 / 0.50-0.54 | 36 tok/s k=3 (−7% vs predicted Qwen 4-bit) | fails; the roofline's α = 0.8 assumption is not met |
| Qwen3-4B MLX 4-bit + 0.6B draft | FAIL (+4.60%) | 0.65 / 0.55 | 52 tok/s k=2 (+33% vs predicted Qwen 4-bit baseline) | out (ppl) |
| Q4_K_M without speculative decoding | PASS (+1.72%) | n/a | 48.8 tok/s (measured, phase 1) | a gate-valid baseline, but +0% |

Other consequences:

* The gate-valid "best phase-1 baseline" is effectively llama.cpp Q4_K_M at 48.8 tok/s. MLX 4-bit
  decodes at the same speed but is not ppl-valid.
* α for Llama pairs (0.80-0.84) sits at the roofline's α = 0.8 column. The Qwen pair is about 0.65, below
  the α = 0.7 column.
* α depends strongly on the prompt: code and SQL reach about 0.9-0.96; prose and the raw bench text reach
  about 0.65. The timed G2-M3 result depends on the decode prompt, so report both the bench prompt and a
  chat mix.

## 4. Recommended timed runs (idle machine, AC power, one runtime at a time)

1. **llama.cpp Q4_K_M 3B + Q4_K_M 1B draft** (the gate-passing candidate):
   * `llama-speculative -m models/gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf -md
     models/gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf -ngl 99 -ngld 99 --spec-draft-n-max {2,3}
     --spec-draft-p-min 0 -n 128 --temp 0 -f bench/prompt_512.txt`
   * Also `llama-server … -md … --spec-draft-n-max 2/3` with `bench/run_openai_endpoint.py`, on the bench
     prompt **and** on the chat prompt set in `quality/prompts/`.
   * Try the default `--spec-draft-p-min` too: it can skip drafting on low-confidence tokens, which helps
     on the low-α bench prompt.
   * Compare against 48.8 tok/s (decode) and 329 ms (TTFT; the draft adds prefill time).
2. **Llama 3B MLX 4-bit + 1B 4-bit draft, k=1 and k=2**: `mlx_lm.generate --draft-model
   models/mlx-Llama-3.2-1B-Instruct-4bit --num-draft-tokens {1,2}`, bench prompt plus chat prompts. This
   is the speed ceiling and calibrates the roofline's spec model, but the target fails the ppl gate.
3. MLX 4-bit stays out of the gate with the cheap recipes tried here (best is mixed_4_6 + 6-bit embed at +3.76%).
   A DWQ (`mlx_lm dwq`, distilled from bf16) or a 5-bit/mixed_4_6 variant may get under 3%. That is a quality
   run (no timing needed) to try before any MLX timed run.
4. Llama 3B MLX 8-bit + 1B draft, k=3 (predicted −2%): a cheap check of the model at higher target cost.
   Qwen3-4B 8-bit + 0.6B 4-bit at k=3 is low priority (predicted −7%).
5. For the TTFT side of the gate, speculative decoding adds a draft prefill (1B prefill ≈ 1/3 of 3B).
   Check that TTFT stays within the baseline.

## 5. Gemma 3 (second model for goal 2): quality only

Date: 2026-11-08, same shared machine (`timing_authoritative: false`). Target: Gemma 3 4B-it. Drafts:
Gemma 3 1B-it and 270M-it (same 262,144-token vocabulary). Gemma is used under the Gemma Terms of Use.
Downloads, revisions and provenance: `models/gemma3/DOWNLOADS.json`. Scripts and commands: [README.md](README.md).
Every tok/s value in this section is a roofline prediction, not a measurement.

### 5.1 Models and formats

* 16-bit references are converted locally from `unsloth/gemma-3-4b-it` (byte-identical to `google/gemma-3-4b-it`):
  BF16 GGUF (llama.cpp `convert_hf_to_gguf.py`) and MLX bf16 (`mlx_lm convert`). Q8_0, Q4_K_M and plain
  Q4_0 come from `llama-quantize` on the BF16 GGUF. MLX 8bit and 4bit (g64 affine) come from `mlx_lm convert`.
* QAT: Google's QAT Q4_0 GGUF is `gemma-3-4b-it-q4_0.gguf`. Its sha256 matches Google's file, and it is
  from an ungated mirror. Google's GGUF keeps `token_embd` (tied LM head) at **F16**, which is 1.34 GB.
  The QAT unquantized checkpoint is `unsloth/gemma-3-4b-it-qat`. From it:
  * `QATlocal_Q4_0`: a local Q4_0 GGUF with a Q6_K embedding, the llama-quantize default.
  * `qatq40grid` (MLX): `make_mlx_quant.py --q40-grid` puts every linear layer on exactly the Q4_0 grid
    Google trained for, stored as MLX affine g32 (scale = d, bias = −8d). It has a 6-bit embedding.
  * `qat4bit`: `mlx-community/gemma-3-4b-it-qat-4bit`, a plain MLX g64 re-quantization of the QAT weights.
    It is not on the Q4_0 grid.
* QAT 1B and 270M drafts were made the same way.

### 5.2 Perplexity (wikitext-2, ctx 512, 570 chunks, llama-perplexity method)

All QAT rows are compared to the **original** (non-QAT) bf16 reference of the same runtime.

| runtime | model | format | PPL | Δ vs original 16-bit | gate ≤3% |
|---|---|---|---:|---:|---|
| llama.cpp | 4B | BF16 (ref) | 17.551 | ref | - |
| llama.cpp | 4B | Q8_0 | 17.530 | −0.12% | PASS |
| llama.cpp | 4B | Q4_K_M | 17.087 | −2.64% | PASS |
| llama.cpp | 4B | plain Q4_0 | 17.259 | −1.66% | PASS |
| llama.cpp | 4B | Google QAT Q4_0 (F16 embed) | 14.814 | −15.59% | PASS |
| llama.cpp | 4B | QATlocal Q4_0 (Q6_K embed) | 14.597 | −16.83% | PASS |
| MLX | 4B | bf16 (ref) | 17.568 | ref | - |
| MLX | 4B | 8bit | 17.528 | −0.23% | PASS |
| MLX | 4B | 4bit | 17.172 | −2.25% ± 0.31 | PASS |
| MLX | 4B | QAT unquantized bf16 | 15.578 | −11.33% | (not a quant) |
| MLX | 4B | qat4bit (mlx-community, g64) | 15.232 | −13.29% | PASS |
| MLX | 4B | qatq40grid | 14.687 | −16.40% | PASS |
| llama.cpp | 1B | BF16 / Q4_K_M | 30.088 / 30.911 | +2.74% | (draft) |
| MLX | 1B | bf16 / 8bit / 4bit | 30.206 / 30.247 / 38.819 | +0.14% / **+28.5%** | (draft) |
| MLX | 1B | qat4bit / qatq40grid | 35.077 / 28.730 | +16.1% / −4.9% | (draft) |
| llama.cpp | 270M | BF16 / Q8_0 | 61.934 / 61.811 | −0.20% | (draft) |
| MLX | 270M | bf16 / 8bit / 4bit / qat4bit | 62.308 / 62.601 / 100.06 / 95.07 | +0.47% / **+60.6%** / +52.6% | (draft) |

MLX and llama.cpp bf16 references agree within 0.1%. Full table: `results/perplexity.json` (`runs`).

**Answer to the key question.** Yes, QAT 4-bit passes the 3% gate against the original bf16 model, in
both runtimes, by a wide margin (−15.6% to −16.8%). Plain 4-bit failed the gate for Llama 3B.

But for Gemma 3 4B-it **plain 4-bit also passes** (MLX 4bit −2.25%, Q4_0 −1.66%, Q4_K_M −2.64%). That
is because wikitext PPL is not a good fidelity measure for this model: the it-tuned model is far from
wikitext, and quantization noise *lowers* its PPL. The QAT checkpoint is effectively a different model.
Even unquantized it scores 11% lower PPL than the original.

So the ≤3% PPL gate does not tell the Gemma 4-bit formats apart. §5.3 adds a distribution-level check.

### 5.3 Fidelity beyond PPL: KL divergence and top-1 agreement (first 64 chunks, 16,384 tokens)

| runtime | format | vs | mean KLD (nats) | median | p99 | same top-1 |
|---|---|---|---:|---:|---:|---:|
| llama.cpp | Q8_0 | BF16 | 0.0013 | 0.0003 | 0.017 | 98.7% |
| llama.cpp | QAT unquantized (BF16) | BF16 | 0.0606 | 0.0172 | 0.715 | 90.5% |
| llama.cpp | **QATlocal Q4_0** | BF16 | **0.0965** | 0.0239 | 1.29 | 88.7% |
| llama.cpp | Q4_K_M | BF16 | 0.1265 | 0.0284 | 1.67 | 87.0% |
| llama.cpp | plain Q4_0 | BF16 | 0.2014 | 0.0513 | 2.57 | 83.6% |
| llama.cpp | QATlocal Q4_0 | QAT BF16 | 0.1177 | 0.0380 | 1.25 | 86.4% |
| MLX | 8bit | bf16 | 0.0051 | 0.0012 | 0.065 | 97.4% |
| MLX | QAT unquantized bf16 | bf16 | 0.0623 | 0.0181 | 0.713 | 90.5% |
| MLX | **qatq40grid** | bf16 | **0.0991** | 0.0250 | 1.32 | 88.5% |
| MLX | qat4bit (mlx-community g64) | bf16 | 0.1816 | 0.0612 | 1.98 | 83.3% |
| MLX | 4bit | bf16 | 0.2469 | 0.0745 | 3.07 | 81.4% |
| MLX | qatq40grid | QAT bf16 | 0.1208 | 0.0388 | 1.28 | 86.2% |
| MLX | qat4bit | QAT bf16 | 0.1662 | 0.0609 | 1.65 | 83.6% |

Findings:

* **QAT on its Q4_0 grid is the 4-bit format closest to the original bf16 model in both runtimes.**
  QATlocal Q4_0 is at 0.097 and MLX qatq40grid at 0.099. The two runtimes agree, as expected for the same
  grid. Q4_K_M is next at 0.126. Plain Q4_0 (0.201) and MLX 4bit (0.247) are 2-2.5 times further away.
* Re-quantizing the QAT weights onto a different grid gives away most of the QAT benefit. mlx-community
  qat-4bit (g64 affine) is at 0.18. Its distance to its own QAT bf16 is 0.166, versus 0.121 for the exact grid.
* Exact-grid QAT is still not lossless against its own checkpoint (0.12). The tied embedding/LM head was not
  QAT-quantized: it is 6-bit in both local versions, and Google keeps it at F16.
* Google's QAT GGUF could not be put through `--kl-divergence`. Its vocabulary has 262,144 rows, while the
  GGUFs converted here have 262,208, and llama-perplexity rejects the comparison. Its linear weights use the
  same Q4_0 grid as QATlocal, and its full-run PPL is within 1.5% of QATlocal.
* The PPL(base) values printed in the llama.cpp KLD logs come from llama.cpp's compressed stored log-probs
  and are off by up to ~3%. Use the full-run rows in §5.2 for perplexity.

Data: `results/perplexity.json` → `kld.runs`; raw output in `results/kld_raw/` and `logs/kld_*.log`.

### 5.4 Speculative acceptance α (same 9 prompts, Gemma chat template, greedy, 192 tokens)

**mlx_lm 0.32.2 limitation, and the fix used here.** Gemma 3's sliding-window layers use
`RotatingKVCache`. Once a cache holds more than its window (512 tokens for 1B/270M, 1024 for 4B),
`trim_prompt_cache()` silently does nothing (`is_trimmable()` is False), and `speculative_generate_step`
only checks the target cache.

The draft model's rejected tokens therefore stay in its cache whenever prompt + output exceeds 512 tokens,
and α collapses with k. For example, `raw_bench512` gave α_iid 0.09 at k=4 with the stock cache;
`results/alpha_raw_superseded/` keeps that run.

`alpha.py` now passes `prompt_cache` = plain `KVCache` for every target and draft layer. Gemma's attention
builds the exact sliding-window mask from the offset (`create_attention_mask(..., window_size)`), so this
changes only memory, not results. `greedy_diag.py` confirms that plain greedy output is identical with
either cache on every checked prompt (`plain_fullcache_equal: true`).

**A timed MLX speculative run of Gemma 3 must use the same fix.** Stock `mlx_lm generate --draft-model`
understates the speed-up on any prompt longer than 512 tokens, and that includes the bench prompt.

MLX, pooled over 9 prompts (acc = accepted/proposed; α_iid = accepted/(accepted+rejections); E = tokens per verify step):

| target + draft | k=1 α_iid / E | k=2 acc / α_iid / E | k=3 acc / α_iid / E | k=4 acc / α_iid / E | bench α_iid (k=2) | greedy = plain |
|---|---|---|---|---|---:|---|
| qatq40grid + 1B 4bit | 0.743 / 1.74 | 0.656 / 0.757 / 2.31 | 0.587 / 0.763 / 2.75 | 0.522 / 0.761 / 3.07 | 0.54 | 2/9 |
| qatq40grid + 1B-QAT 4bit | 0.772 / 1.77 | 0.681 / 0.776 / 2.36 | 0.616 / 0.779 / 2.84 | 0.545 / 0.773 / 3.15 | 0.55 | 2/9 |
| qatq40grid + 270M-QAT 4bit | 0.611 / 1.61 | 0.503 / 0.621 / 2.00 | 0.405 / 0.612 / 2.20 | 0.337 / 0.615 / 2.33 | 0.47 | 2/9 |
| qat4bit + 1B-QAT 4bit | 0.752 / 1.75 | 0.651 / 0.747 / 2.30 | 0.569 / 0.745 / 2.70 | 0.509 / 0.751 / 3.01 | 0.53 | 4/9 |
| 4bit + 1B 4bit | 0.761 / 1.76 | 0.638 / 0.736 / 2.27 | 0.566 / 0.744 / 2.69 | 0.494 / 0.744 / 2.96 | 0.51 | 3/9 |
| 4bit + 1B 8bit | 0.784 / 1.78 | 0.680 / 0.768 / 2.35 | 0.614 / 0.782 / 2.83 | 0.542 / 0.776 / 3.15 | 0.51 | 3/9 |
| 4bit + 270M 4bit | 0.576 / 1.58 | 0.435 / 0.560 / 1.86 | 0.356 / 0.571 / 2.06 | 0.292 / 0.570 / 2.15 | 0.39 | 3/9 |
| 4bit + 270M bf16 | 0.680 / 1.68 | 0.564 / 0.672 / 2.12 | 0.476 / 0.677 / 2.42 | 0.401 / 0.672 / 2.58 | 0.47 | 3/9 |
| 8bit + 1B 4bit | 0.747 / 1.75 | 0.663 / 0.757 / 2.32 | 0.573 / 0.751 / 2.71 | 0.523 / 0.761 / 3.07 | 0.64 | 1/9 |

llama.cpp `llama-speculative` (`--temp 0 --spec-draft-p-min 0 --spec-draft-n-min 0`), pooled (α_iid from E):

| target + draft | k=1 α / E | k=2 acc / α / E | k=3 acc / α / E | k=4 acc / α / E | bench α (k=2) | same text as k=1 (k=2/3/4) |
|---|---|---|---|---|---:|---|
| QATlocal Q4_0 + 1B Q4_K_M | 0.804 / 1.80 | 0.716 / 0.802 / 2.44 | 0.637 / 0.794 / 2.93 | 0.562 / 0.785 / 3.26 | 0.64 | 5/7/5 of 9 |
| QATlocal Q4_0 + 1B-QAT Q4_0 | 0.794 / 1.79 | 0.717 / 0.802 / 2.45 | 0.626 / 0.787 / 2.89 | 0.558 / 0.783 / 3.25 | 0.64 | 5/6/5 |
| QATlocal Q4_0 + 270M-QAT Q4_0 | 0.670 / 1.67 | 0.563 / 0.678 / 2.14 | 0.473 / 0.673 / 2.43 | 0.390 / 0.660 / 2.57 | 0.60 | 5/6/5 |
| Q4_K_M + 1B Q4_K_M | 0.789 / 1.79 | 0.663 / 0.760 / 2.34 | 0.599 / 0.768 / 2.81 | 0.522 / 0.759 / 3.10 | 0.55 | 2/2/1 |
| Q4_K_M + 270M Q8_0 | 0.710 / 1.71 | 0.578 / 0.691 / 2.17 | 0.483 / 0.682 / 2.46 | 0.413 / 0.679 / 2.66 | 0.50 | 1/3/1 |

Observations:

* α for Gemma 4B with the 1B draft is 0.74-0.80, slightly below Llama 3B + 1B (0.80-0.84).
* The 270M draft costs about 0.1 in α: 0.62-0.68 when it is ≥8-bit or QAT-quantized.
* MLX 4-bit wrecks the small drafts. 270M 4bit loses another 0.1 in α (0.57), consistent with its +61% PPL.
* llama.cpp gets about 0.02-0.05 more α than MLX for the same model pairs.
* The QAT target is easier to draft for: the llama.cpp bench prompt gives 0.64 against 0.55 for Q4_K_M.
* α_iid is flat in k, as it should be with correct cache rewinding.
* α depends on the prompt, ranging from about 0.5 (raw bench text) to 0.95 (SQL/code).

**Greedy equivalence.**

* MLX: spec output equals plain output for only 1/9-4/9 prompts. Divergence points are at the same token
  for every k.
* `greedy_diag.py` was run on `gemma4b-qatq40grid_1b-4bit` and `gemma4b-8bit_1b-4bit` (15 mismatches).
  At every first mismatch the two candidate tokens are the target's top-2. The margin is 0-0.5 logits,
  which is 0-4 bf16 ulps at these magnitudes and often an exact tie. In most cases the M=1 incremental
  path and the batched verify path rank the pair in opposite order.
* The cause is numerical tie-breaking in the 262k-wide bf16 logits, not a drafting or acceptance bug.
  It does not change the output distribution, so a speculative config inherits its target's PPL/KLD.
  Results: `results/alpha_raw/*.greedy_diag.json`.
* llama.cpp: there is no plain-greedy comparison. The cross-k identical-text count (1/9-7/9) shows the same
  near-tie sensitivity.

### 5.5 Roofline predictions (`gemma_predict.py`; ctx 576; calibration mlx 0.953, llama.cpp 1.013)

**The cost of the 262k-vocabulary LM head.** The tied embedding/LM head has 671M parameters, 17% of Gemma 3 4B.

| model | format | per-token weight bytes | of which LM head |
|---|---|---:|---:|
| 4B | MLX 4bit | 2.18 GB | 0.38 GB (17%) |
| 4B | llama.cpp Q4_0 / Q4_K_M (Q6_K embed) | 2.36 / 2.48 GB | 0.55 GB (22-23%) |
| 4B | Google QAT Q4_0 (F16 embed) | 3.15 GB | 1.34 GB (**43%**) |
| 1B draft | MLX 4bit / Q4_K_M | 0.56 / 0.67 GB | 30% / 37% |
| 270M draft | MLX 4bit / Q8_0 / Q4_0 | 0.15 / 0.24 / 0.19 GB | 63% / 56% / 71% |

Consequences:

* Google's QAT GGUF is predicted at only 29.5 tok/s, against 38.0 for the same QAT weights with a Q6_K
  embedding (QATlocal). That is −22% from the LM head alone. **Use QATlocal, not Google's file, for timing.**
* The 270M draft is mostly LM head, so it is only about 3× cheaper per token than the 1B draft, not about 4×.
* The verify pass recomputes the full 262k head for k+1 rows. That is a negligible extra cost while memory-bound.

No-draft predictions (decode tok/s at ctx 576):

| config | GB/token | pred tok/s | PPL gate | KLD vs bf16 |
|---|---:|---:|---|---:|
| MLX 4bit | 2.18 | 40.7 | PASS | 0.247 |
| MLX qat4bit | 2.18 | 40.7 | PASS | 0.182 |
| llama.cpp QATlocal Q4_0 | 2.36 | 38.0 | PASS | 0.097 |
| llama.cpp plain Q4_0 | 2.36 | 38.0 | PASS | 0.201 |
| llama.cpp Q4_K_M | 2.48 | 37.0 | PASS | 0.126 |
| MLX qatq40grid | 2.55 | 35.4 | PASS | 0.099 |
| llama.cpp Google QAT Q4_0 | 3.15 | 29.5 | PASS | n/a |
| llama.cpp Q8_0 / MLX 8bit | 3.96 / 4.12 | 24.6 / 22.5 | PASS | 0.001 / 0.005 |
| MLX bf16 | 7.76 | 12.3 | ref | 0 |

**Best gate-valid no-draft config: MLX 4bit, predicted at 40.7 tok/s.** Every format passes the PPL gate,
so this is simply the fastest one.

Speculative predictions at the measured E. Gains are given against the config's own no-draft baseline and
against the best gate-valid no-draft config (40.7). "pooled" is the 9-prompt E; "bench" is E on
`raw_bench512`.

| pair (best k) | k | pred tok/s pooled / bench | vs own no-draft | vs best gate-valid (40.7) |
|---|---:|---|---|---|
| **llama.cpp QATlocal Q4_0 + 270M-QAT Q4_0** | 2 | 58.3 / 53.8 | **+53% / +41%** | **+43% / +32%** |
| llama.cpp QATlocal Q4_0 + 270M-QAT Q4_0 | 3 | 58.8 / 52.5 | +55% / +38% | +45% / +29% |
| **llama.cpp QATlocal Q4_0 + 1B-QAT Q4_0** | 2 | 52.5 / 44.1 | +38% / +16% | +29% / +8% |
| llama.cpp QATlocal Q4_0 + 1B Q4_K_M | 2 | 52.1 / 43.8 | +37% / +15% | +28% / +8% |
| llama.cpp Q4_K_M + 270M Q8_0 | 2 | 55.8 / 45.0 | +51% / +22% | +37% / +11% |
| llama.cpp Q4_K_M + 1B Q4_K_M | 2 | 48.6 / 38.6 | +31% / +4% | +19% / −5% |
| **MLX qatq40grid + 270M-QAT 4bit** | 2 | 54.9 / 46.6 | +55% / +32% | +35% / +15% |
| MLX qatq40grid + 1B-QAT 4bit | 3 | 52.8 / 38.4 | +49% / +9% | +30% / −6% |
| MLX 4bit + 270M 4bit | 2 | 57.2 / 47.7 | +41% / +17% | +41% / +17% |
| MLX 4bit + 270M bf16 | 2 | 52.5 / 42.2 | +29% / +4% | +29% / +4% |
| MLX 4bit + 1B 4bit | 2 | 54.3 / 42.5 | +33% / +4% | +33% / +4% |
| MLX 4bit + 1B 8bit | 1 | 46.0 / 39.9 | +13% / −2% | +13% / −2% |
| MLX qat4bit + 1B-QAT 4bit | 2 | 54.9 / 43.5 | +35% / +7% | +35% / +7% |
| MLX 8bit + 1B 4bit | 3 | 40.2 / 32.8 | +78% / +45% | −1% / −19% |

All k values: `results/gemma_predictions.json`, also attached to `results/alpha.json` rows as
`gemma_roofline_prediction`.

**Which Gemma configs could beat their own gate-valid no-draft baseline by ≥15%.** All of these pass the
PPL gate.

* Pooled α: every 4-bit-target pair at k=2-3 except MLX 4bit + 1B 8bit, plus MLX 8bit + 1B. The 8bit
  pair only clears the bar because its own baseline is slow.
* Bench prompt as well, the robust set:
  * llama.cpp QATlocal + 270M-QAT, k=2-3: +38% to +41%.
  * MLX qatq40grid + 270M-QAT, k=2: +32%.
  * llama.cpp Q4_K_M + 270M Q8_0, k=2: +22%.
  * MLX 4bit + 270M 4bit, k=2: +17%.
  * llama.cpp QATlocal + 1B(-QAT), k=2: +15% to +16%.
* The 270M draft wins on prediction despite its lower α, because the 1B draft step costs about 7-8 ms
  against 2.5-3 ms.

The 270M prediction is the least certain. Its fixed per-step overhead (dispatch, sampling, the 262k head)
dominates, and the roofline's c_d term was fitted on 1B-3B models. Time it first.

**Recommended Gemma candidate for goal 2:** llama.cpp QATlocal Q4_0 4B + 270M-QAT Q4_0 draft, k=2-3.
* It has the lowest KLD of any 4-bit format.
* PPL is −16.8% vs the original bf16.
* Predicted gain is +53% vs its own baseline (+41% on the bench prompt) and +43% vs the fastest gate-valid
  config.
* The MLX counterpart is qatq40grid + 270M-QAT. Its predicted gain is smaller (+32% on the bench prompt).
  A 270M **q40grid** draft, not yet built, would likely raise its α: the 1B q40grid draft is at −4.9% PPL,
  against +16% for the 1B qat4bit draft.

**TTFT.** Speculative decoding does not reduce TTFT. It adds the draft's prefill (compute-bound, so roughly proportional to
non-embedding parameters: about ¼ of the 4B prefill for the 1B draft, about 1/30 for the 270M). No Gemma config here is predicted to win on TTFT.

### 5.6 MAX nightly (mojo-max/, 26.8.0.dev2026110805): Gemma 3 support (checked, nothing run)

* `Gemma3ForCausalLM` (and `_ModuleV3`) serves the text-only 1B/270M models in **bfloat16 only**.
* `Gemma3ForConditionalGeneration` (and `_ModuleV3`) serves 4B and up (multimodal) in bfloat16 or float8_e4m3fn.
  * It requires prefix caching and chunked prefill to be disabled.
  * It reserves 15 GiB for activations.
  * It has no overlap scheduler.
* The speculative methods are eagle / mtp / dflash / dflash2 / dspark. There is **no separate-draft-model
  method**, and no int4/Q4_0/QAT encoding.
* So MAX can run Gemma 3 4B only at bf16. That is predicted at about 12 tok/s, a third of the 4-bit
  baselines, and it cannot run any of the speculative configs above. MAX is not a goal-2 path for Gemma
  without new work: an int4 encoding plus a draft-model speculative pipeline.

### 5.7 Timed runs needed (idle machine, AC power, one runtime at a time; no Gemma phase-1 baseline exists yet)

1. **No-draft baselines**, decode tok/s and TTFT on `bench/prompt_512.txt` with the phase-1 protocol:
   * llama.cpp (`llama-bench` / `llama-cli`): `gemma-3-4b-it-qat-localQ4_0.gguf`, `-Q4_K_M`, `-plainQ4_0`,
     Google `gguf-qat-google/gemma-3-4b-it-q4_0.gguf` (checks the F16-embedding prediction), `-Q8_0`.
   * MLX (`mlx_lm generate`; the stock cache is fine without a draft): `mlx-gemma-3-4b-it-4bit`, `-qat-4bit`,
     `-qat-q40grid`, `-8bit`.
2. **Speculative, llama.cpp** (`llama-speculative`, plus `llama-server -md` with
   `bench/run_openai_endpoint.py`). Use bench prompt plus the chat prompts in `quality/prompts/gemma/`, and
   both `--spec-draft-p-min 0` and the default:
   * QATlocal Q4_0 + `gemma-3-270m-it-qat-localQ4_0.gguf`, `--spec-draft-n-max 2,3`;
   * QATlocal Q4_0 + `gemma-3-1b-it-qat-localQ4_0.gguf`, n-max 2;
   * Q4_K_M + `gemma-3-270m-it-Q8_0.gguf`, n-max 2.
3. **Speculative, MLX.** These must use the full-KVCache fix (`alpha.spec_cache`), or a patched mlx_lm,
   because the bench prompt exceeds the 512-token draft window:
   * qatq40grid + 270m-qat-4bit, k=2;
   * 4bit + 270m-4bit, k=2;
   * qat4bit + 1b-qat-4bit, k=2.
4. TTFT with each draft loaded, to check the draft-prefill cost.
