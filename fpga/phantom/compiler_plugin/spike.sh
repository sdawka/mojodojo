#!/bin/bash
# Feasibility evidence for the compiler-plugin route (see ../README.md).
#  1. stdlib plugin (pure Mojo): device_kind() == other("phantom", True). Works.
#  2. codegen plugin: the shipped mojo dlopens MODULAR_COMPILER_PLUGINS and
#     calls our M_KGEN_getTargetTraits, but codegen for "phantom-unknown-none"
#     still fails because TargetLowering/TargetBackend are not provided.
#     That step is EXPECTED to fail; this script exits 0 if it fails that way.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PH="$(cd "$HERE/.." && pwd)"
cd "$PH/.."
bash "$HERE/build.sh" || { echo "plugin build failed (run phantom-fetch-src first?)"; exit 1; }
echo; echo "== 1. stdlib plugin on a host-triple target"
mojo run "$PH/spike/stdlib_plugin.mojo" || exit 1
echo; echo "== 2. custom triple with the TargetTraits plugin loaded"
out=$(PHANTOM_PLUGIN_LOG=1 MODULAR_COMPILER_PLUGINS="$PH/build/libphantom_compiler_plugin.dylib" \
  mojo run "$PH/spike/custom_triple.mojo" 2>&1)
echo "$out" | grep -v '^ *[~^]' | head -20
if echo "$out" | grep -q "M_KGEN_getTargetTraits() called" &&
   echo "$out" | grep -q "is not supported by this build"; then
  echo; echo "verdict: plugin loaded and consulted; codegen backend missing (as documented)"
else
  echo; echo "verdict: UNEXPECTED output, re-check the spike"; exit 1
fi
