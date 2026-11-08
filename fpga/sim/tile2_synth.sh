#!/usr/bin/env bash
# Tile v2 ECP5 synthesis + place-and-route on LFE5U-85F (CABGA381).
#
#   pixi run tile2-synth            # tile2_n8 (DSP mapping), seeds 1 2 3
#   SEEDS="1" pixi run tile2-synth
#
# The top has only 90 I/O (Wishbone + irq), so it is placed directly, with no
# serialising wrapper. Results: build/tile2/synth/summary.txt (+ report JSON).
set -euo pipefail

SUITE=$(ls -d .tools/oss-cad-suite-*/oss-cad-suite/bin 2>/dev/null | tail -1 || true)
[[ -n "$SUITE" ]] && export PATH="$(pwd)/$SUITE:$PATH"
command -v yosys >/dev/null || { echo "yosys not found"; exit 2; }
command -v nextpnr-ecp5 >/dev/null || { echo "nextpnr-ecp5 not found"; exit 2; }

read -r -a SEEDS <<<"${SEEDS:-1 2 3}"
TOP=${TOP:-tile2_n8}
SRC=${SRC:-tile2/rtl/$TOP.v}
OUT=build/tile2/synth
mkdir -p "$OUT"
SUM="$OUT/summary.txt"
{
    echo "# $(yosys -V)"
    echo "# $(nextpnr-ecp5 --version 2>&1 | head -1)"
    echo "# top=$TOP src=$SRC device=LFE5U-85F CABGA381"
} >"$SUM"

cell() { awk -v c="$2" '$2 == c {print $1; f=1} END {if (!f) print 0}' "$1"; }

yosys -q -l "$OUT/$TOP.yosys.log" -p "read_verilog $SRC; synth_ecp5 -top $TOP -json $OUT/$TOP.json; tee -q -o $OUT/$TOP.stat stat" >/dev/null
s="$OUT/$TOP.stat"
printf "yosys: LUT4=%s CCU2C=%s FF=%s MULT18X18D=%s DP16KD=%s PFUMX=%s L6MUX21=%s DPR16X4=%s\n" \
    "$(cell "$s" LUT4)" "$(cell "$s" CCU2C)" "$(cell "$s" TRELLIS_FF)" "$(cell "$s" MULT18X18D)" \
    "$(cell "$s" DP16KD)" "$(cell "$s" PFUMX)" "$(cell "$s" L6MUX21)" "$(cell "$s" TRELLIS_DPR16X4)" | tee -a "$SUM"

for seed in "${SEEDS[@]}"; do
    base="$OUT/$TOP.s$seed"
    if nextpnr-ecp5 --85k --package CABGA381 --json "$OUT/$TOP.json" --freq 200 --seed "$seed" \
        --timing-allow-fail --report "$base.report.json" -l "$base.pnr.log" -q >/dev/null 2>&1; then
        fmax=$(grep -E "Max frequency for clock" "$base.pnr.log" | tail -1 | sed -E 's/.*: *([0-9.]+) MHz.*/\1/')
        echo "seed $seed: fmax(clk)=$fmax MHz" | tee -a "$SUM"
        { grep -E "^Info:[[:space:]]+(TRELLIS_COMB|TRELLIS_FF|TRELLIS_RAMW|DP16KD|MULT18X18D|TRELLIS_IO):" "$base.pnr.log" || true; } |
            sed -E 's/^Info:[[:space:]]+/    /' | tee -a "$SUM"
    else
        echo "seed $seed: P&R FAILED (see $base.pnr.log)" | tee -a "$SUM"
    fi
done
echo "summary: $SUM"
