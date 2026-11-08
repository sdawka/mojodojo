#!/usr/bin/env bash
# Full processor-atlas suite for the Apple M5. Runs everything SEQUENTIALLY.
# Authoritative numbers require an idle machine: close other GPU/CPU-heavy work (other agents'
# builds, servers, browsers playing video), stay on AC power, lid open/cool.
#
#   ./run_all.sh            # full timed run (~35-60 min), writes results/*.json, then ATLAS.md
#   ATLAS_SMOKE=1 ./run_all.sh   # quick validation run (~5 min), writes results/*_smoke.json
#   ATLAS_SKIP="ane ggml" ./run_all.sh   # skip sections (inventory build cpu_bw cpu_peak accel
#                                        #   mlx mlx_nonax mlx_notf32 max_gpu ane ggml report)
#   ATLAS_REPS=25 ./run_all.sh            # more timed reps (min 5; default 15, smoke 5)
set -uo pipefail   # no -e: one failing section must not abort the rest (failures are logged)
cd "$(dirname "$0")"
ATLAS="$PWD"
ROOT="$(cd .. && pwd)"
PIXI=(pixi run --manifest-path "$ROOT/mojo-max/pixi.toml")
export MODULAR_METAL_INTREE_METALLIB=1 MODULAR_TELEMETRY_ENABLED=0
SMOKE="${ATLAS_SMOKE:-0}"
SMOKEARG=""; [[ "$SMOKE" == 1 ]] && SMOKEARG="smoke"
SKIP=" ${ATLAS_SKIP:-} "
LOG="$ATLAS/results/run_$(date +%Y%m%d_%H%M%S)$([[ $SMOKE == 1 ]] && echo _smoke).log"
mkdir -p results build
want() { [[ "$SKIP" != *" $1 "* ]]; }
step() { echo; echo "=== [$(date +%H:%M:%S)] $* (load: $(sysctl -n vm.loadavg))" | tee -a "$LOG"; }
chk() { local rc=$?; [[ $rc != 0 ]] && echo "!!! section failed (rc=$rc), see $LOG" | tee -a "$LOG"; return 0; }
cooldown() { sleep "${ATLAS_COOLDOWN_S:-$([[ $SMOKE == 1 ]] && echo 1 || echo 20)}"; }

if [[ "$SMOKE" != 1 ]]; then
  load1=$(sysctl -n vm.loadavg | awk '{print $2}')
  echo "1-min load average: $load1"
  if awk "BEGIN{exit !($load1 > 1.5)}"; then
    echo "WARNING: machine does not look idle (load $load1). Results will be flagged non-authoritative." | tee -a "$LOG"
  else
    export ATLAS_IDLE_CONFIRMED=1
  fi
fi

want inventory && { step inventory; uv run python scripts/inventory.py 2>&1 | tee -a "$LOG" >/dev/null; }

if want build; then
  step "build C + Mojo binaries"
  clang -O3 -mcpu=apple-m4 -o build/stream cpu/stream.c
  clang -O3 -mcpu=apple-m4+sme-f16f16 -o build/peak cpu/peak.c   # NOT -march=armv9: SVE (non-streaming) traps on Apple
  clang -O3 -mcpu=apple-m4 -framework Accelerate -o build/gemm_accel cpu/gemm_accel.c 2>&1 | grep -v deprecated || true
  "${PIXI[@]}" mojo build gpu/max_gpu_bench.mojo -o build/max_gpu_bench 2>&1 | tee -a "$LOG"
fi

W=(uv run python scripts/wrap.py)
want cpu_bw   && { step "CPU bandwidth";  "${W[@]}" cpu_bandwidth -- build/stream 2>>"$LOG"; chk; cooldown; }
want cpu_peak && { step "CPU NEON/SME peak"; "${W[@]}" cpu_peak -- build/peak 2>>"$LOG"; chk; cooldown; }
want accel    && { step "Accelerate GEMM"; "${W[@]}" cpu_accelerate --stdin-shapes -- build/gemm_accel 2>>"$LOG"; chk; cooldown; }
want mlx      && { step "MLX default (NAX)"; uv run python gpu/mlx_bench.py --label default --bandwidth --matmul 2>&1 | tee -a "$LOG"; chk; cooldown; }
want mlx_nonax && { step "MLX NAX disabled"; MLX_METAL_GPU_ARCH=applegpu_g16g uv run python gpu/mlx_bench.py --label nonax_g16g --matmul 2>&1 | tee -a "$LOG"; chk; cooldown; }
want mlx_notf32 && { step "MLX TF32 off"; MLX_ENABLE_TF32=0 uv run python gpu/mlx_bench.py --label notf32 --matmul 2>&1 | tee -a "$LOG"; chk; cooldown; }
want max_gpu  && { step "MAX Mojo GPU kernels"; "${W[@]}" gpu_max -- build/max_gpu_bench bw mm $SMOKEARG 2>>"$LOG"; chk; cooldown; }
want ane      && { step "Core ML / ANE"; uv run python ane/coreml_bench.py 2>&1 | grep -v -E "it/s|Running" | tee -a "$LOG"; chk; cooldown; }
want ggml     && { step "ggml MUL_MAT cross-check"; uv run python gpu/ggml_bench.py 2>&1 | tee -a "$LOG"; }
want report   && { step "report"; uv run python scripts/report.py 2>&1 | tee -a "$LOG"; }
step "done. log: $LOG"
