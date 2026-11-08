#!/usr/bin/env bash
# Build a SoC (gateware + BIOS) with soc.py and compile its Verilator model.
#   scripts/build_soc.sh OUT_DIR [soc.py args...]
set -euo pipefail
cd "$(dirname "$0")/.."
out=$1; shift
python soc.py --out "$out" "$@" > "$out.log" 2>&1 || { tail -30 "$out.log"; exit 1; }
( cd "$out/gateware" && bash build_sim.sh > build.log 2>&1 ) || { tail -30 "$out/gateware/build.log"; exit 1; }
echo "built $out/gateware/obj_dir/Vsim"
