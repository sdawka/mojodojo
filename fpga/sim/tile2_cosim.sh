#!/usr/bin/env bash
# Co-simulate the generated tile v2 RTL (tile2/rtl/tile2_n{8,4}.v) against
# randomized register-level programs from the Mojo bus-level model
# (systolic/tile2.mojo via tile2/prog.mojo), then run negative checks.
#
#   pixi run tile2-cosim            # 2 configs x SEEDS programs + long-K programs
#   SEEDS=5 pixi run tile2-cosim
#
# Every read (CSRs incl. STATUS polls and CYCLES while busy, A/B/C buffers),
# the irq level in every request cycle, ack timing and the total cycle count
# must match the model exactly. Run from fpga/ (pixi tasks do that).
set -euo pipefail

SEEDS=${SEEDS:-30}
B=build/tile2
P=$B/progs
mkdir -p "$B" "$P"
command -v verilator >/dev/null || { echo "verilator not found"; exit 2; }

echo "== building program generator (Mojo model)"
mojo build -I . tile2/prog.mojo -o "$B/prog"

verilate() {  # verilate <N> <src.v> <objdir>
    local n=$1 src=$2 obj=$3
    mkdir -p "$obj"
    printf '#define TOP Vtile2_n%s\n#define TOPHDR "Vtile2_n%s.h"\n' "$n" "$n" >"$obj/tb_config.h"
    verilator --cc --exe --build -j 0 -Wall -O3 --top-module "tile2_n$n" -Mdir "$obj" \
        -CFLAGS "-O2 -I$(pwd)/$obj" -o tb "$src" sim/tile2_tb.cpp >"$obj/build.log" 2>&1 ||
        { cat "$obj/build.log"; exit 2; }
}

echo "== verilating tile2_n8, tile2_n4"
verilate 8 tile2/rtl/tile2_n8.v "$B/obj_n8" &
verilate 4 tile2/rtl/tile2_n4.v "$B/obj_n4" &
wait

echo "== generating programs"
PROGS=()
for n in 8 4; do
    for s in $(seq 1 "$SEEDS"); do
        f="$P/n${n}_s${s}.txt"
        "$B/prog" --tile "$n" --seed "$s" --jobs 8 --out "$f" >/dev/null
        PROGS+=("$n:$f")
    done
    f="$P/n${n}_big.txt"
    "$B/prog" --tile "$n" --seed 999 --jobs 3 --big --out "$f" >/dev/null  # first job K=1000
    PROGS+=("$n:$f")
done

echo "== co-simulating ${#PROGS[@]} programs"
tot_c=0; tot_w=0; tot_r=0; tot_runs=0; fail=0
for e in "${PROGS[@]}"; do
    n=${e%%:*}; f=${e#*:}
    if out=$("$B/obj_n$n/tb" "$f" --quiet); then
        read -r c w r runs <<<"$out"
        tot_c=$((tot_c + c)); tot_w=$((tot_w + w)); tot_r=$((tot_r + r)); tot_runs=$((tot_runs + runs))
    else
        echo "$out"; fail=$((fail + 1))
    fi
done
if [[ $fail -ne 0 ]]; then echo "FAIL: $fail of ${#PROGS[@]} programs"; exit 1; fi
echo "PASS: ${#PROGS[@]} programs, $tot_runs matmul runs, $tot_c cycles, $tot_w writes, $tot_r reads checked"

echo "== negative checks (each must be reported as a mismatch)"
neg_fail=0
expect_mismatch() {  # expect_mismatch <label> <tb> <prog>
    local out rc=0
    out=$("$2" "$3" --quiet 2>&1) || rc=$?
    if [[ $rc -eq 1 ]]; then echo "   caught ($1): $(echo "$out" | head -1)"
    else echo "   NOT CAUGHT ($1): rc=$rc"; neg_fail=1; fi
}
src="$P/n8_s1.txt"
# 1. Program: one C read-back value off by one.  2. END cycle count off by one.
python3 -I sim/tile2_corrupt.py "$src" "$P/neg_c.txt" cread
expect_mismatch "program: C read-back value +1" "$B/obj_n8/tb" "$P/neg_c.txt"
python3 -I sim/tile2_corrupt.py "$src" "$P/neg_end.txt" end
expect_mismatch "program: END cycles +1" "$B/obj_n8/tb" "$P/neg_end.txt"
python3 -I sim/tile2_corrupt.py "$src" "$P/neg_poll.txt" poll
expect_mismatch "program: one STATUS poll result flipped busy<->done" "$B/obj_n8/tb" "$P/neg_poll.txt"

mutant() {  # mutant <label> <sed-expr>
    local label=$1 expr=$2 dir="$B/mut_$3"
    mkdir -p "$dir/src"
    sed "$expr" tile2/rtl/tile2_n8.v >"$dir/src/tile2_n8.v"
    if cmp -s tile2/rtl/tile2_n8.v "$dir/src/tile2_n8.v"; then
        echo "   mutant sed did not apply ($label)"; neg_fail=1; return
    fi
    verilate 8 "$dir/src/tile2_n8.v" "$dir/obj"
    local caught=0 out rc
    for s in 1 2 3; do
        rc=0; out=$("$dir/obj/tb" "$P/n8_s$s.txt" --quiet 2>&1) || rc=$?
        if [[ $rc -eq 1 ]]; then caught=1; break; fi
    done
    if [[ $caught -eq 1 ]]; then echo "   caught (RTL mutant: $label): $(echo "$out" | head -1)"
    else echo "   NOT CAUGHT (RTL mutant: $label)"; neg_fail=1; fi
}
mutant "PE(1,1) zero-extends its registered product" \
    's/acc_1_1 + {{16{prodq_1_1\[15\]}}/acc_1_1 + {{16{1'"'"'b0}}/' sext
mutant "flush one cycle short" "s/if (cnt == 16'd14) begin/if (cnt == 16'd13) begin/" flush
mutant "C lane-7 write mask ignores N" "s/assign d_lane\[7\]  = .*/assign d_lane[7] = 1'b1;/" lane
mutant "ACCUM ignored in lane 0" "s/crd_q_0 <= mode_q\[0\] ? c_rd_0 : 32'd0/crd_q_0 <= 32'd0/" accum
mutant "CSR write while busy not blocked" "s/wire csr_wr = wr \&\& !busy \&\&/wire csr_wr = wr \&\&/" busywr
if [[ $neg_fail -ne 0 ]]; then echo "FAIL: a negative check was not caught"; exit 1; fi
echo "PASS: all negative checks caught"
