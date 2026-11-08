#!/bin/zsh
# Gemma 3 speculative acceptance runs (M4), sequential. Logs -> quality/logs/alpha_*.log
#   run_alpha_gemma.sh [mlx|llamacpp|all]
set -u
R=/Users/sdawka/Code/llming; L=$R/quality/logs; G=models/gemma3/gguf
SET=${1:-all}
if [[ $SET == mlx || $SET == all ]]; then
  # alpha.py uses full (trimmable) KV caches for spec decoding; see FULL_KV_CACHE there
  for pair in gemma4b-qatq40grid_1b-4bit gemma4b-qatq40grid_1bqat-4bit gemma4b-qatq40grid_270mqat-4bit \
              gemma4b-4bit_1b-4bit gemma4b-4bit_270m-4bit gemma4b-4bit_270m-bf16 gemma4b-4bit_1b-8bit \
              gemma4b-8bit_1b-4bit gemma4b-qat4bit_1bqat-4bit; do
    uv run --project $R/mlx python $R/quality/alpha.py --pair $pair > $L/alpha_$pair.log 2>&1
  done
fi
if [[ $SET == llamacpp || $SET == all ]]; then
  for spec in "qatlocal_1b-q4km gemma-3-4b-it-qat-localQ4_0.gguf gemma-3-1b-it-Q4_K_M.gguf" \
              "qatlocal_1bqat-q4_0 gemma-3-4b-it-qat-localQ4_0.gguf gemma-3-1b-it-qat-localQ4_0.gguf" \
              "qatlocal_270mqat-q4_0 gemma-3-4b-it-qat-localQ4_0.gguf gemma-3-270m-it-qat-localQ4_0.gguf" \
              "q4km_1b-q4km gemma-3-4b-it-Q4_K_M.gguf gemma-3-1b-it-Q4_K_M.gguf" \
              "q4km_270m-q8_0 gemma-3-4b-it-Q4_K_M.gguf gemma-3-270m-it-Q8_0.gguf"; do
    set -- ${=spec}
    uv run --project $R/mlx python $R/quality/alpha_llamacpp.py --name llamacpp_gemma4b-$1 \
        --target $G/$2 --draft $G/$3 --prompts quality/prompts/gemma > $L/alpha_llamacpp_gemma4b-$1.log 2>&1
  done
fi
