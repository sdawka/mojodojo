#!/usr/bin/env bash
# ECP5 synthesis (yosys synth_ecp5) and place-and-route (nextpnr-ecp5) sanity run.
#
#   pixi run synth                 # N = 4 8 16
#   SIZES="4" pixi run synth
#
# Uses the YosysHQ oss-cad-suite in .tools/ if present (yosys + nextpnr-ecp5),
# else yosys/nextpnr-ecp5 from PATH. Results: build/synth/summary.txt.
set -euo pipefail

SUITE=$(ls -d .tools/oss-cad-suite-*/oss-cad-suite/bin 2>/dev/null | tail -1 || true)
[[ -n "$SUITE" ]] && export PATH="$(pwd)/$SUITE:$PATH"
command -v yosys >/dev/null || { echo "yosys not found"; exit 2; }
HAVE_PNR=1
command -v nextpnr-ecp5 >/dev/null || { echo "nextpnr-ecp5 not found: synthesis only"; HAVE_PNR=0; }

read -r -a SIZES <<<"${SIZES:-4 8 16}"
OUT=build/synth
mkdir -p "$OUT"
SUM="$OUT/summary.txt"
{
    echo "# $(yosys -V)"
    [[ $HAVE_PNR -eq 1 ]] && echo "# $(nextpnr-ecp5 --version 2>&1 | head -1)"
} >"$SUM"

# Device per tile size for P&R: the smallest LFE5U whose MULT18X18D count
# (25F: 28, 45F: 72, 85F: 156) covers the N^2 multipliers. N=16 needs 256,
# more than any ECP5, so it is placed on the 85F with LUT multipliers (-nodsp).
device_for() {
    case $1 in
        1|2|3|4) echo "--25k" ;;
        8) echo "--45k" ;;
        *) echo "--85k" ;;
    esac
}

cell() { awk -v c="$2" '$2 == c {print $1; f=1} END {if (!f) print 0}' "$1"; }

synth_one() {  # synth_one <N> <tag> <extra synth_ecp5 flags>
    local n=$1 tag=$2 flags=$3 top="systolic_n$1_i8_a32"
    local base="$OUT/n${n}${tag}"
    # Bare tile: resource counts.
    yosys -q -l "$base.core.log" -p "read_verilog rtl/$top.v; synth_ecp5 -top $top $flags; tee -q -o $base.core.stat stat" >/dev/null
    # Wrapped tile: input to P&R.
    [[ $4 == pnr ]] && yosys -q -l "$base.wrap.log" -p "read_verilog rtl/$top.v; read_verilog -DTILE=$top -DTILE_N=$n sim/synth_wrap.v; synth_ecp5 -top synth_wrap $flags -json $base.json" >/dev/null
    local s="$base.core.stat"
    local line
    line=$(printf "N=%-2s %-7s LUT4=%-6s CCU2C=%-5s FF=%-6s MULT18X18D=%-4s PFUMX=%-4s L6MUX21=%-4s" \
        "$n" "${tag:-dsp}" "$(cell "$s" LUT4)" "$(cell "$s" CCU2C)" "$(cell "$s" TRELLIS_FF)" \
        "$(cell "$s" MULT18X18D)" "$(cell "$s" PFUMX)" "$(cell "$s" L6MUX21)")
    if [[ $HAVE_PNR -eq 1 && $4 == pnr ]]; then
        local dev
        dev=$(device_for "$n")
        if nextpnr-ecp5 "$dev" --package CABGA381 --json "$base.json" --freq 200 \
            --seed 1 --timing-allow-fail --report "$base.report.json" -l "$base.pnr.log" -q \
            >/dev/null 2>&1; then
            local fmax
            fmax=$(grep -E "Max frequency for clock" "$base.pnr.log" | tail -1 |
                sed -E 's/.*: *([0-9.]+) MHz.*/\1/')
            local util
            util=$(grep -E "Total LUT4s|TRELLIS_COMB|MULT18X18D" "$base.pnr.log" | head -2 |
                sed -E 's/^Info: *//; s/ +/ /g' | tr '\n' ';')
            line="$line  P&R LFE5U${dev#--}: fmax=${fmax} MHz  [$util]"
        else
            line="$line  P&R LFE5U${dev#--}: FAILED (see $base.pnr.log)"
        fi
    fi
    echo "$line" | tee -a "$SUM"
}

for n in "${SIZES[@]}"; do
    if [[ $n -ge 16 ]]; then
        synth_one "$n" "" "" nopnr       # DSP mapping: counts only (does not fit)
        synth_one "$n" "_nodsp" "-nodsp" pnr
    else
        synth_one "$n" "" "" pnr
    fi
done
echo "summary: $SUM"
