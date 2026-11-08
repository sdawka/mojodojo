#!/usr/bin/env bash
# llama.cpp (Metal) baseline: llama-bench pp512 + tg128, 3 reps, F16 and Q4_K_M.
# Usage: bash /Users/sdawka/Code/llming/bench/run_llamacpp.sh [extra llama-bench args]
#
# llama-bench uses synthetic tokens (no prompt file, no sampling): pp512 = one
# 512-token prompt eval, tg128 = 128 sequential single-token decode steps on an
# empty context. All layers on GPU (-ngl 99), flash-attn auto, default batch
# 2048 / ubatch 512. llama-bench does its own warmup run before the reps.
# Peak memory: /usr/bin/time -l "maximum resident set size". With the default mmap
# loading, Metal wraps the mmapped weights as no-copy shared buffers, so they show
# up in RSS (weights + KV + compute buffers). Do NOT pass -mmp 0, or the weights
# move into Metal-private allocations that RSS doesn't see.
set -euo pipefail

ROOT=/Users/sdawka/Code/llming
BIN=$ROOT/llamacpp/src/build/bin/llama-bench
OUT=$ROOT/bench/results
mkdir -p "$OUT"
TS=$(date +%Y%m%d_%H%M%S)

declare -a FMTS=("f16:$ROOT/models/gguf/Llama-3.2-3B-Instruct-F16.gguf"
                 "q4_k_m:$ROOT/models/gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf")

for entry in "${FMTS[@]}"; do
  fmt=${entry%%:*}; model=${entry#*:}
  json="$OUT/llamacpp_${fmt}_${TS}.json"
  timelog="$OUT/llamacpp_${fmt}_${TS}.time.txt"
  echo "== llama-bench $fmt =="
  /usr/bin/time -l "$BIN" -m "$model" -p 512 -n 128 -r 3 -ngl 99 -o json "$@" \
    > "$json" 2> "$timelog"
  grep -E "maximum resident set size|peak memory footprint" "$timelog" || true
  python3 - "$json" <<'PY'
import json, sys
for t in json.load(open(sys.argv[1])):
    kind = f"pp{t['n_prompt']}" if t['n_gen'] == 0 else f"tg{t['n_gen']}"
    print(f"  {kind}: {t['avg_ts']:.1f} ± {t['stddev_ts']:.1f} tok/s  ({t['backends']}, {t['gpu_info']})")
PY
done
echo "results in $OUT/llamacpp_*_${TS}.json"
