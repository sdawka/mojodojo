#!/usr/bin/env bash
# Build the Mojo ISS + all firmware, run everything on rv32sim, then (if the
# fpga/soc Verilator model is built) on the LiteX/VexRiscv SoC. Asserts exit
# codes and that diffcheck's checksums equal the host-Mojo ones.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
B="$HERE/build"; mkdir -p "$B"
FWS=(hello diffcheck matmul_test tile2_test stdlib_demo softfloat_demo)
(cd "$ROOT/mojo-max" && pixi run mojo build -I "$HERE/sim" -I "$ROOT/fpga" -o "$B/rv32sim" "$HERE/sim/rv32sim.mojo")
for fw in "${FWS[@]}"; do "$HERE/build.sh" "$HERE/fw/$fw.mojo" >/dev/null; done

HOST="$(cd "$ROOT/mojo-max" && pixi run mojo run -I "$HERE/fw" "$HERE/fw/diffcheck_host.mojo")"
echo "== host Mojo"; echo "$HOST"
same_checksums() { [[ "$(grep checksum <<<"$1" | tr -d "\r")" == "$(grep checksum <<<"$HOST")" ]]; }

for fw in "${FWS[@]}"; do
  echo "== rv32sim $fw"
  out="$("$B/rv32sim" "$B/$fw.elf")"; echo "$out"
  [[ $fw != diffcheck ]] || same_checksums "$out" || { echo "CHECKSUM MISMATCH"; exit 1; }
done
echo "== rv32sim matmul_test --tile-fault (must FAIL)"
if "$B/rv32sim" "$B/matmul_test.elf" --tile-fault | tail -4; then
  echo "fault injection was not detected"; exit 1
fi

SOC="$ROOT/fpga/soc"
if [[ -x "$SOC/build/sim/gateware/obj_dir/Vsim" && -x "$SOC/.venv/bin/python" ]]; then
  for fw in "${FWS[@]}"; do
    if [[ $fw == tile2_test ]] && ! grep -qi tile2 "$SOC/build/sim/csr.csv"; then
      echo "== Verilator: SoC has no tile2 yet, skipping tile2_test"; continue
    fi
    echo "== Verilator LiteX/VexRiscv $fw"
    out="$(cd "$SOC" && .venv/bin/python run.py "$B/$fw.bin" --run-dir "$B/vrun_$fw" --timeout 600 \
           | sed -n '/Liftoff/,$p')"
    echo "$out"
    [[ $fw != diffcheck ]] || same_checksums "$out" || { echo "CHECKSUM MISMATCH"; exit 1; }
  done
else
  echo "(fpga/soc Verilator model not built; skipping RTL runs)"
fi
echo "ALL CHECKS PASSED"
