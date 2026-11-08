#!/usr/bin/env bash
# Build a bare-metal rv32im Mojo firmware image.
#   fpga/rvmojo/build.sh <app.mojo> [out_dir]
# Env overrides (decimal or 0x hex): ROM_BASE ROM_SIZE RAM_BASE RAM_SIZE,
# and Mojo -D board defines: UART_BASE SIMCTRL_BASE TILE_BASE (decimal or hex).
# Output: <out>/<app>.elf, <app>.bin (ROM image from ROM_BASE), <app>.lst.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
APP="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
NAME="$(basename "$APP" .mojo)"
OUT="${2:-$HERE/build}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

ROM_BASE="${ROM_BASE:-0x40000000}"; ROM_SIZE="${ROM_SIZE:-0x100000}"
RAM_BASE="${RAM_BASE:-0x40100000}"; RAM_SIZE="${RAM_SIZE:-0x100000}"

# Linker: lld from the mojo-max pixi env (multi-flavor binary).
LLD="$ROOT/mojo-max/.pixi/envs/default/bin/lld"
# Optional: llvm-objcopy/objdump with RISC-V support from the fpga/soc env.
SOCBIN="$ROOT/fpga/soc/.pixi/envs/default/bin"

DEFS=()
for v in UART_BASE SIMCTRL_BASE TILE_BASE; do
  if [[ -n "${!v:-}" ]]; then DEFS+=(-D "$v=$(( ${!v} ))"); fi
done

TARGET=(--target-triple riscv32-unknown-none-elf --target-cpu generic-rv32
        --target-features "+m,+zicsr")

cd "$ROOT/mojo-max"
pixi run mojo build "${TARGET[@]}" ${DEFS[@]+"${DEFS[@]}"} --emit object \
  -o "$OUT/rt.o" "$HERE/fw/rt.mojo"
pixi run mojo build "${TARGET[@]}" ${DEFS[@]+"${DEFS[@]}"} -I "$HERE/fw" \
  --emit object -o "$OUT/$NAME.o" "$APP"

# Link fw/shim.mojo (print/String/List/raise support) only when the app
# needs one of its symbols.
OBJS=("$OUT/rt.o" "$OUT/$NAME.o")
NM="$SOCBIN/llvm-nm"; [[ -x "$NM" ]] || NM=/Library/Developer/CommandLineTools/usr/bin/llvm-nm
if "$NM" -u "$OUT/$NAME.o" | grep -qE ' (KGEN_CompilerRT_|write$|fdopen$|__atomic_)'; then
  pixi run mojo build "${TARGET[@]}" ${DEFS[@]+"${DEFS[@]}"} -I "$HERE/fw" \
    --emit object -o "$OUT/shim.o" "$HERE/fw/shim.mojo"
  OBJS+=("$OUT/shim.o")
  echo "linking fw/shim.mojo (libc/runtime shim)"
fi

# compiler-rt builtins for rv32 (soft-float, 64-bit div): the archive LiteX
# builds for its BIOS. Only referenced members get linked.
CRT_LIB="${COMPILER_RT:-$ROOT/fpga/soc/build/sim/software/libcompiler_rt/libcompiler_rt.a}"
if [[ -f "$CRT_LIB" ]]; then OBJS+=("$CRT_LIB"); fi

sed -e "s/@ROM_BASE@/$ROM_BASE/" -e "s/@ROM_SIZE@/$ROM_SIZE/" \
    -e "s/@RAM_BASE@/$RAM_BASE/" -e "s/@RAM_SIZE@/$RAM_SIZE/" \
    "$HERE/link.ld.in" > "$OUT/link.ld"

"$LLD" -flavor gnu -m elf32lriscv -nostdlib -static --no-relax \
  -T "$OUT/link.ld" -Map "$OUT/$NAME.map" \
  -o "$OUT/$NAME.elf" "${OBJS[@]}"

if [[ -x "$SOCBIN/llvm-objcopy" ]]; then
  "$SOCBIN/llvm-objcopy" -O binary "$OUT/$NAME.elf" "$OUT/$NAME.bin"
  "$SOCBIN/llvm-objdump" -d --no-show-raw-insn "$OUT/$NAME.elf" > "$OUT/$NAME.lst"
  "$SOCBIN/llvm-size" "$OUT/$NAME.elf" 2>/dev/null || true
fi
echo "built $OUT/$NAME.elf"
