#!/bin/zsh
# Quantization fidelity (KL divergence + same-top-1) for Gemma 3 4B on the first CHUNKS wikitext ppl chunks.
# Needed because Gemma-3-it wikitext PPL drops under 4-bit quantization (PPL alone cannot rank the quants).
# llama.cpp: llama-perplexity --kl-divergence-base (saves reference log-probs, ~0.5 MB/token) then --kl-divergence.
# MLX: quality/kld_mlx.py (both models in one process). Logs -> quality/logs/kld_*.log, MLX JSON -> results/kld_raw/
set -u
R=/Users/sdawka/Code/llming; B=$R/llamacpp/src/build/bin; G=$R/models/gemma3; L=$R/quality/logs
CH=${CHUNKS:-64}; W=$R/models/datasets/wiki.test.raw; K=$G/kld; mkdir -p $K
SET=${1:-all}
lc_base() { $B/llama-perplexity -m $1 -f $W -c 512 -b 2048 -ngl 99 --chunks $CH --kl-divergence-base $2 > $3 2>&1; }
lc_kld()  { $B/llama-perplexity -m $1 -f $W -c 512 -b 2048 -ngl 99 --chunks $CH --kl-divergence-base $2 --kl-divergence > $3 2>&1; }
# Google's QAT Q4_0 GGUF is not in the llama.cpp lists: its vocab is 262144 rows, the GGUFs converted here from the
# HF checkpoints have 262208, and llama-perplexity --kl-divergence then logs "inconsistent vocabulary" and prints
# garbage (KLD ~23). QATlocal_Q4_0 (same QAT checkpoint, local Q4_0 of the linears) stands in for it.
if [[ $SET == llamacpp || $SET == all ]]; then
  lc_base $G/gguf/gemma-3-4b-it-BF16.gguf $K/gemma4b-BF16.kld $L/kld_llamacpp_gemma4b-BF16-base.log
  for spec in "Q8_0 gguf/gemma-3-4b-it-Q8_0.gguf" "Q4_K_M gguf/gemma-3-4b-it-Q4_K_M.gguf" "Q4_0 gguf/gemma-3-4b-it-plainQ4_0.gguf" \
              "QATlocal_Q4_0 gguf/gemma-3-4b-it-qat-localQ4_0.gguf" "QAT_BF16 gguf/gemma-3-4b-it-qat-BF16.gguf"; do
    set -- ${=spec}; lc_kld $G/$2 $K/gemma4b-BF16.kld $L/kld_llamacpp_gemma4b-$1_vs_BF16.log
  done
  rm -f $K/gemma4b-BF16.kld
  # QAT quants against the QAT checkpoint's own unquantized weights (pure quantization error)
  lc_base $G/gguf/gemma-3-4b-it-qat-BF16.gguf $K/gemma4b-QAT_BF16.kld $L/kld_llamacpp_gemma4b-QAT_BF16-base.log
  for spec in "QATlocal_Q4_0 gguf/gemma-3-4b-it-qat-localQ4_0.gguf"; do
    set -- ${=spec}; lc_kld $G/$2 $K/gemma4b-QAT_BF16.kld $L/kld_llamacpp_gemma4b-$1_vs_QAT_BF16.log
  done
  rm -f $K/gemma4b-QAT_BF16.kld
fi
if [[ $SET == mlx || $SET == all ]]; then
  for spec in "bf16 8bit gemma4b-mlx-8bit" "bf16 4bit gemma4b-mlx-4bit" "bf16 qat-4bit gemma4b-mlx-qat4bit" \
              "bf16 qat-q40grid gemma4b-mlx-qatq40grid" "bf16 qat-bf16 gemma4b-mlx-qatbf16" \
              "qat-bf16 qat-4bit gemma4b-mlx-qat4bit_vs_qatbf16" "qat-bf16 qat-q40grid gemma4b-mlx-qatq40grid_vs_qatbf16"; do
    set -- ${=spec}
    uv run --project $R/mlx python $R/quality/kld_mlx.py --ref $G/mlx-gemma-3-4b-it-$1 --model $G/mlx-gemma-3-4b-it-$2 \
        --name $3 --chunks $CH > $L/kld_$3.log 2>&1
  done
fi
