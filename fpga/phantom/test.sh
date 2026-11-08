#!/bin/bash
# Runs every phantom test. Expects phantom/build.sh to have run (pixi does it).
#   1. C ABI from plain C           (test_capi)
#   2. Mojo DeviceContext-style API (tests/test_phantom.mojo)
#   3. Python binding               (tests/test_python.py)
#   4. HAL driver plugin through MAX's NeoDriver client, in the ../mojo-max
#      env (needs the `max` package); skipped if that env or the plugin is absent.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FPGA="$(cd "$HERE/.." && pwd)"
cd "$FPGA"
export PHANTOM_LIB="$HERE/build/libphantom.dylib"
fail=0
echo "== C ABI";        "$HERE/build/test_capi" || fail=1
echo "== Mojo API";     mojo run -I . -I phantom phantom/tests/test_phantom.mojo || fail=1
echo "== Python";       python "$HERE/tests/test_python.py" || fail=1
MAXENV="$FPGA/../mojo-max"
if [ -x "$MAXENV/.pixi/envs/default/bin/mojo" ] && [ -f "$HERE/build/libphantom_hal.dylib" ]; then
  echo "== HAL plugin (mojo-max env)"
  MODULAR_DRIVER_PLUGINS="phantom@$HERE/build/libphantom_hal.dylib" \
    pixi run --manifest-path "$MAXENV/pixi.toml" mojo run "$HERE/tests/test_hal.mojo" || fail=1
else
  echo "== HAL plugin: skipped (needs ../mojo-max env and build/libphantom_hal.dylib)"
fi
[ $fail = 0 ] && echo "phantom-test: all suites passed" || echo "phantom-test: FAILURES"
exit $fail
