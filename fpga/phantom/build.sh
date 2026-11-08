#!/bin/bash
# Builds the phantom NPU artifacts into phantom/build/. Run from anywhere.
#   libphantom.dylib      C-ABI driver (simulator), from capi.mojo
#   phantom_py.so         Python extension module, from phantom_py.mojo
#   libphantom_hal.dylib  Modular HAL driver plugin (C), links libphantom
#   test_capi             plain-C test of the C ABI
# The compiler-plugin spike needs third_party/ and is built by
# compiler_plugin/build.sh (pixi run phantom-plugin-spike).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FPGA="$(cd "$HERE/.." && pwd)"
OUT="$HERE/build"
mkdir -p "$OUT"
cd "$FPGA"

mojo build -I . -I phantom --emit shared-lib phantom/capi.mojo -o "$OUT/libphantom.dylib"
install_name_tool -id @rpath/libphantom.dylib "$OUT/libphantom.dylib"
echo "built $OUT/libphantom.dylib"

mojo build -I . -I phantom --emit shared-lib phantom/phantom_py.mojo -o "$OUT/phantom_py.so"
echo "built $OUT/phantom_py.so"

clang -O1 -Wall -I"$HERE/include" -L"$OUT" -lphantom -Wl,-rpath,@loader_path \
  "$HERE/tests/test_capi.c" -o "$OUT/test_capi"
echo "built $OUT/test_capi"

# The HAL plugin compiles against Modular's public HAL header (SDK/include,
# MAX License), taken from the gitignored third_party checkout. Skip if absent.
HAL_INC="$HERE/third_party/modular/SDK/include"
if [ -f "$HAL_INC/HAL/M_driver_api.h" ]; then
  clang -O1 -fPIC -shared -fvisibility=hidden -Wall -Wno-unused-parameter \
    -I"$HAL_INC" -I"$HERE/include" -L"$OUT" -lphantom -Wl,-rpath,@loader_path \
    -Wl,-install_name,@rpath/libphantom_hal.dylib \
    "$HERE/hal_plugin/phantom_hal.c" -o "$OUT/libphantom_hal.dylib"
  echo "built $OUT/libphantom_hal.dylib ($(nm -gU "$OUT/libphantom_hal.dylib" | grep -c M_driver_) M_driver_* exports)"
else
  echo "skipped libphantom_hal.dylib: run phantom/fetch_toolchain_src.sh first"
fi
