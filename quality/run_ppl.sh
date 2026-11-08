#!/bin/zsh
# Runs perplexity measurements (M8) sequentially. Logs -> quality/logs/ppl_*.log
#   run_ppl.sh            Llama 3.2 / Qwen3 set (the original G2-M3 runs)
#   run_ppl.sh gemma      Gemma 3 set (4B target formats incl. QAT, 1B / 270M drafts)
#   run_ppl.sh gemma-llamacpp | gemma-mlx   only one runtime of the Gemma set
set -u
R=/Users/sdawka/Code/llming; M=$R/models; B=$R/llamacpp/src/build/bin; L=$R/quality/logs
G=$M/gemma3
CTX=${CTX:-512}
SET=${1:-llama}

mlx_ppl() {  # <model dir> <name: family-mlx-fmt>
  uv run --project $R/mlx python $R/quality/ppl_mlx.py --model $1 --name $2 --ctx $CTX > $L/ppl_$2.log 2>&1
}

if [[ $SET == llama ]]; then
  for q in F16 Q8_0 Q4_K_M Q4_0; do
    $B/llama-perplexity -m $M/gguf/Llama-3.2-3B-Instruct-$q.gguf -f $M/datasets/wiki.test.raw -c $CTX -b 2048 -ngl 99 > $L/ppl_llamacpp_llama3b-$q.log 2>&1
  done
  for spec in "mlx-Llama-3.2-3B-Instruct-bf16 llama3b-mlx-bf16" "mlx-Llama-3.2-3B-Instruct-8bit llama3b-mlx-8bit" \
              "mlx-Llama-3.2-3B-Instruct-4bit llama3b-mlx-4bit" "mlx-Llama-3.2-1B-Instruct-4bit llama1b-mlx-4bit" \
              "mlx-Qwen3-4B-bf16 qwen4b-mlx-bf16" "mlx-Qwen3-4B-8bit qwen4b-mlx-8bit" "mlx-Qwen3-4B-4bit qwen4b-mlx-4bit" \
              "mlx-Qwen3-0.6B-8bit qwen0.6b-mlx-8bit" "mlx-Qwen3-0.6B-4bit qwen0.6b-mlx-4bit"; do
    set -- ${=spec}
    mlx_ppl $M/$1 $2
  done
  echo ALLDONE >> $L/ppl_done.flag
fi

if [[ $SET == gemma || $SET == gemma-llamacpp ]]; then
  # name -> gguf. log: ppl_llamacpp_<family>-<fmt>.log (parsed by summarize.py; ref = <family>-BF16)
  for spec in "gemma4b-BF16 gguf/gemma-3-4b-it-BF16.gguf" "gemma4b-Q8_0 gguf/gemma-3-4b-it-Q8_0.gguf" \
              "gemma4b-Q4_K_M gguf/gemma-3-4b-it-Q4_K_M.gguf" "gemma4b-Q4_0 gguf/gemma-3-4b-it-plainQ4_0.gguf" \
              "gemma4b-QAT_Q4_0 gguf-qat-google/gemma-3-4b-it-q4_0.gguf" \
              "gemma4b-QATlocal_Q4_0 gguf/gemma-3-4b-it-qat-localQ4_0.gguf" \
              "gemma1b-BF16 gguf/gemma-3-1b-it-BF16.gguf" "gemma1b-Q4_K_M gguf/gemma-3-1b-it-Q4_K_M.gguf" \
              "gemma270m-BF16 gguf/gemma-3-270m-it-BF16.gguf" "gemma270m-Q8_0 gguf/gemma-3-270m-it-Q8_0.gguf"; do
    set -- ${=spec}
    [[ -f $G/$2 ]] || { echo "skip $1 (missing $G/$2)"; continue; }
    $B/llama-perplexity -m $G/$2 -f $M/datasets/wiki.test.raw -c $CTX -b 2048 -ngl 99 > $L/ppl_llamacpp_$1.log 2>&1
  done
fi

if [[ $SET == gemma || $SET == gemma-mlx ]]; then
  # SKIP_DONE=1 skips models that already have results/ppl_raw/<name>.json
  for spec in "mlx-gemma-3-4b-it-bf16 gemma4b-mlx-bf16" "mlx-gemma-3-4b-it-8bit gemma4b-mlx-8bit" \
              "mlx-gemma-3-4b-it-4bit gemma4b-mlx-4bit" "mlx-gemma-3-4b-it-qat-4bit gemma4b-mlx-qat4bit" \
              "mlx-gemma-3-4b-it-qat-q40grid gemma4b-mlx-qatq40grid" "mlx-gemma-3-4b-it-qat-bf16 gemma4b-mlx-qatbf16" \
              "mlx-gemma-3-1b-it-bf16 gemma1b-mlx-bf16" "mlx-gemma-3-1b-it-4bit gemma1b-mlx-4bit" \
              "mlx-gemma-3-1b-it-qat-q40grid gemma1b-mlx-qatq40grid" \
              "mlx-gemma-3-270m-it-bf16 gemma270m-mlx-bf16" "mlx-gemma-3-270m-it-4bit gemma270m-mlx-4bit" \
              "mlx-gemma-3-1b-it-8bit gemma1b-mlx-8bit" "mlx-gemma-3-1b-it-qat-4bit gemma1b-mlx-qat4bit" \
              "mlx-gemma-3-270m-it-8bit gemma270m-mlx-8bit" "mlx-gemma-3-270m-it-qat-4bit gemma270m-mlx-qat4bit"; do
    set -- ${=spec}
    [[ -d $G/$1 ]] || { echo "skip $2 (missing $G/$1)"; continue; }
    [[ ${SKIP_DONE:-0} == 1 && -f $R/quality/results/ppl_raw/$2.json ]] && continue
    mlx_ppl $G/$1 $2
  done
  echo GEMMADONE >> $L/ppl_done.flag
fi
