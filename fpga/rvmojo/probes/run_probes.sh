#!/usr/bin/env bash
# Compile each capability probe for bare-metal rv32im and list the external
# symbols it needs (anything listed must be provided at link time).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
NM="$ROOT/fpga/soc/.pixi/envs/default/bin/llvm-nm"
[[ -x "$NM" ]] || NM=/Library/Developer/CommandLineTools/usr/bin/llvm-nm  # ELF-capable too
OUT="$HERE/out"; mkdir -p "$OUT"
cd "$ROOT/mojo-max"
for f in "$HERE"/p_*.mojo; do
  n="$(basename "$f" .mojo)"
  pixi run mojo build --target-triple riscv32-unknown-none-elf --target-cpu generic-rv32 \
    --target-features "+m,+zicsr" --emit object -o "$OUT/$n.o" "$f" >/dev/null 2>"$OUT/$n.err" \
    || { echo "$n: COMPILE ERROR (see $OUT/$n.err)"; continue; }
  pixi run mojo build --target-triple riscv32-unknown-none-elf --target-cpu generic-rv32 \
    --target-features "+m,+zicsr" --emit asm -o "$OUT/$n.s" "$f" >/dev/null 2>&1 || true
  echo "$n: undefined = [$("$NM" -u "$OUT/$n.o" | awk '{print $2}' | tr '\n' ' ')]"
done
