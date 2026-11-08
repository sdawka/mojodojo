#!/bin/bash
# Builds libphantom_compiler_plugin.dylib against the headers of the exact
# toolchain commit (modular@2841b316 / llvm-project@fa8c6738).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TP="$HERE/../third_party"
OUT="$HERE/../build"
mkdir -p "$OUT"
"$HERE/gen_llvm_config.sh" "$TP/llvm-project/llvm/include/llvm/Config" "$OUT/gen/llvm/Config"
clang++ -std=c++17 -O1 -fPIC -shared -fno-rtti -fno-exceptions \
  -fvisibility=hidden -fvisibility-inlines-hidden -DNDEBUG -DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1 \
  -I"$HERE/shim" -I"$OUT/gen" \
  -I"$TP/modular/Mojo/lib" \
  -I"$TP/llvm-project/llvm/include" \
  -include unistd.h \
  "$HERE/phantom_traits.cpp" -o "$OUT/libphantom_compiler_plugin.dylib"
echo "built $OUT/libphantom_compiler_plugin.dylib"
nm -gU "$OUT/libphantom_compiler_plugin.dylib"
echo "undefined:"; nm -u "$OUT/libphantom_compiler_plugin.dylib" | c++filt
