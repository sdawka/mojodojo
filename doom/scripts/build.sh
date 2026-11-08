#!/usr/bin/env bash
# Build: C engine libs, then the Mojo host linked three ways.
#   doom-c         Mojo host + stock C drawers            (tier 1; tier-2 reference)
#   doom-mojo      Mojo host + Mojo drawers (scalar)      (tier 2)
#   doom-mojo-simd Mojo host + Mojo drawers (-D DOOM_SIMD)
#   doom-mojo*-verify  per-call C-vs-Mojo differential check builds
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
B="$HERE/build"
bash "$HERE/scripts/build_c.sh"
SDL_PREFIX="$(sdl2-config --prefix 2>/dev/null || echo /opt/homebrew)"
LINK=(-Xlinker "$B/dg_glue.o" -Xlinker "$B/libdoomcore.a"
      -Xlinker -lz -Xlinker -L"$SDL_PREFIX/lib" -Xlinker -lSDL2)
mojo build -O3 -I "$HERE/src" -I "$HERE/src/nogpu" "$HERE/src/host.mojo" -o "$B/doom-c" -Xlinker "$B/r_draw_c.o" -Xlinker "$B/drawbench_c.o" "${LINK[@]}"
mojo build -O3 -I "$HERE/src" -I "$HERE/src/nogpu" "$HERE/src/host.mojo" -o "$B/doom-mojo" -Xlinker "$B/r_draw_mojo.o" -Xlinker "$B/drawbench_mojo.o" "${LINK[@]}"
mojo build -O3 -D DOOM_SIMD -I "$HERE/src" -I "$HERE/src/nogpu" "$HERE/src/host.mojo" -o "$B/doom-mojo-simd" -Xlinker "$B/r_draw_mojo.o" -Xlinker "$B/drawbench_mojo.o" "${LINK[@]}"
# Differential-verify builds (C and Mojo drawers both run on every call; see shim/r_draw_mojo.c)
mojo build -O3 -I "$HERE/src" -I "$HERE/src/nogpu" "$HERE/src/host.mojo" -o "$B/doom-mojo-verify" -Xlinker "$B/r_draw_verify.o" -Xlinker "$B/drawbench_c.o" "${LINK[@]}"
mojo build -O3 -D DOOM_SIMD -I "$HERE/src" -I "$HERE/src/nogpu" "$HERE/src/host.mojo" -o "$B/doom-mojo-simd-verify" -Xlinker "$B/r_draw_verify.o" -Xlinker "$B/drawbench_c.o" "${LINK[@]}"
# Tier 3 (X-DOOM-3): GPU replay. Needs max.gpu, which lives in the MAX toolchain
# (../mojo-max pixi env, same Mojo nightly), not in this env's mojo package.
MAXENV="${DOOM_MAX_PIXI:-$HERE/../mojo-max/pixi.toml}"
if [ -f "$MAXENV" ]; then
  MAXMOJO=(pixi run --manifest-path "$MAXENV" mojo)
  GPU_LINK=(-Xlinker "$B/r_main_gpu.o" -Xlinker "$B/drawbench_c.o" "${LINK[@]}")
  "${MAXMOJO[@]}" build -O3 -D DOOM_GPU -I "$HERE/src" -I "$HERE/src/metal" "$HERE/src/host.mojo" -o "$B/doom-mojo-gpu" -Xlinker "$B/r_draw_gpu.o" "${GPU_LINK[@]}"
  "${MAXMOJO[@]}" build -O3 -D DOOM_GPU -I "$HERE/src" -I "$HERE/src/metal" "$HERE/src/host.mojo" -o "$B/doom-mojo-gpu-verify" -Xlinker "$B/r_draw_gpu_verify.o" "${GPU_LINK[@]}"
else
  echo "skipping doom-mojo-gpu: no MAX pixi env at $MAXENV (set DOOM_MAX_PIXI)"
fi
echo "built: $(ls "$B" | grep '^doom-' | tr '\n' ' ')"
