#!/usr/bin/env bash
# Co-simulate the generated RTL (rtl/*.v) against fresh golden traces from the
# Mojo cycle-accurate simulator, then run negative checks.
#
#   pixi run cosim            # full matrix (all tile sizes, incl. a K=65535 pass)
#   QUICK=1 pixi run cosim    # skip the long K=65535 trace
#
# Must be run from fpga/ (pixi tasks do that). Needs verilator on PATH.
set -euo pipefail

SIZES=(1 2 3 4 8 16)
BUILD=build
TR=traces/cosim
mkdir -p "$BUILD" "$TR"

command -v verilator >/dev/null || { echo "verilator not found (brew install verilator)"; exit 2; }

echo "== building trace dumper"
mojo build -I . tools/dump_trace.mojo -o "$BUILD/dump_trace"

# verilate <N> <rtl.v> <objdir>: build a harness binary for one module.
verilate() {
    local n=$1 src=$2 obj=$3
    local top
    top=$(basename "$src" .v)
    mkdir -p "$obj"
    cat >"$obj/tb_config.h" <<EOF
#define TOP V${top}
#define TOPHDR "V${top}.h"
#define TOPROOT "V${top}___024root.h"
#define DBG_FSM ${top}__DOT__fsm
#define DBG_CNT ${top}__DOT__cnt
#define TILE_N ${n}
#define IN_BITS 8
#define ACC_BITS 32
EOF
    verilator --cc --exe --build -j 0 -Wall -O3 \
        --top-module "$top" -Mdir "$obj" -CFLAGS "-O2 -I$(pwd)/$obj" \
        -o tb "$src" sim/tb.cpp >"$obj/build.log" 2>&1 ||
        { cat "$obj/build.log"; exit 2; }
}

echo "== verilating"
for n in "${SIZES[@]}"; do
    verilate "$n" "rtl/systolic_n${n}_i8_a32.v" "$BUILD/obj_n$n" &
done
wait
echo "   ok: ${SIZES[*]}"

echo "== generating traces"
TRACES=()
gen() {  # gen <N> <M> <K> <N_out> <seed> [--stalls]
    local n=$1 m=$2 k=$3 no=$4 seed=$5 st=${6:-}
    local f="$TR/n${n}_m${m}_k${k}_n${no}_s${seed}${st:+_stalls}.jsonl"
    "$BUILD/dump_trace" --tile "$n" --m "$m" --k "$k" --n "$no" --seed "$seed" \
        --format jsonl $st --out "$f" >/dev/null
    TRACES+=("$n:$f")
}
for n in "${SIZES[@]}"; do
    s=$((100 * n))
    for st in "" "--stalls"; do
        gen "$n" "$n" 1 "$n" $((s + 1)) $st                          # single pass, K=1
        gen "$n" 5 7 6 $((s + 2)) $st                                # small, non-multiple
        gen "$n" $((2 * n + 1)) $((2 * n + 3)) $((n + 3)) $((s + 3)) $st  # ragged edge tiles
        gen "$n" 1 33 $((2 * n + 1)) $((s + 4)) $st                  # decode-like M=1
        gen "$n" 13 20 9 $((s + 5)) $st                              # tall, several passes
    done
done
gen 16 16 300 16 1601 --stalls
if [[ -z "${QUICK:-}" ]]; then
    gen 4 4 65535 4 401   # longest legal pass: 16-bit k_len / cnt end of range
fi

echo "== co-simulating ${#TRACES[@]} traces"
total_cycles=0
fail=0
for entry in "${TRACES[@]}"; do
    n=${entry%%:*}
    f=${entry#*:}
    if "$BUILD/obj_n$n/tb" "$f" --quiet; then
        c=$(wc -l <"$f")
        total_cycles=$((total_cycles + c))
    else
        fail=$((fail + 1))
    fi
done
for n in "${SIZES[@]}"; do
    cnt=0
    for entry in "${TRACES[@]}"; do [[ ${entry%%:*} == "$n" ]] && cnt=$((cnt + 1)); done
    echo "   N=$n: $cnt traces"
done
if [[ $fail -ne 0 ]]; then
    echo "FAIL: $fail of ${#TRACES[@]} traces mismatched"
    exit 1
fi
echo "PASS: ${#TRACES[@]} traces, $total_cycles cycles, every output field every cycle"

echo "== negative checks (each must be reported as a mismatch)"
neg_fail=0
expect_mismatch() {  # expect_mismatch <label> <tb> <trace>
    local out rc=0
    out=$("$2" "$3" --quiet 2>&1) || rc=$?
    if [[ $rc -eq 1 ]]; then
        echo "   caught ($1): $(echo "$out" | head -1)"
    else
        echo "   NOT CAUGHT ($1): rc=$rc"
        neg_fail=1
    fi
}
src="$TR/n4_m13_k20_n9_s405.jsonl"
# 1. One accumulator value off by one in the first DRAIN cycle.
python3 -I sim/corrupt_trace.py "$src" "$TR/neg_out_row.jsonl" out_row
expect_mismatch "trace out_row[2] +1" "$BUILD/obj_n4/tb" "$TR/neg_out_row.jsonl"
# 2. out_last flag dropped on its first assertion.
python3 -I sim/corrupt_trace.py "$src" "$TR/neg_out_last.jsonl" out_last
expect_mismatch "trace out_last 1->0" "$BUILD/obj_n4/tb" "$TR/neg_out_last.jsonl"
# 3. RTL mutant: PE(1,1) zero-extends its product instead of sign-extending.
sed 's/acc_1_1 + {{16{prod_1_1\[15\]}}/acc_1_1 + {{16{1'"'"'b0}}/' \
    rtl/systolic_n4_i8_a32.v >"$BUILD/mutant_n4.v"
if cmp -s rtl/systolic_n4_i8_a32.v "$BUILD/mutant_n4.v"; then
    echo "   mutant sed did not apply"; neg_fail=1
else
    mkdir -p "$BUILD/mutant_src"
    cp "$BUILD/mutant_n4.v" "$BUILD/mutant_src/systolic_n4_i8_a32.v"
    verilate 4 "$BUILD/mutant_src/systolic_n4_i8_a32.v" "$BUILD/obj_mutant_n4"
    expect_mismatch "RTL mutant: zero-extended product in PE(1,1)" \
        "$BUILD/obj_mutant_n4/tb" "$src"
fi
if [[ $neg_fail -ne 0 ]]; then
    echo "FAIL: a negative check was not caught"
    exit 1
fi
echo "PASS: all negative checks caught"
