#!/bin/bash
# Generates the two CMake-configured LLVM headers the plugin's includes need
# (llvm-config.h, abi-breaking.h) for the LLVM the nightly was built from.
# The installed libLLVM.dylib exports llvm::DisableABIBreakingChecks, so
# ABI-breaking checks are OFF.
set -euo pipefail
SRC="$1"   # third_party/llvm-project/llvm/include/llvm/Config
OUT="$2"   # build/gen/llvm/Config
mkdir -p "$OUT"
conf() {
  sed -E \
    -e 's/^#cmakedefine01 (LLVM_ENABLE_THREADS|LLVM_HAS_ATOMICS)$/#define \1 1/' \
    -e 's/^#cmakedefine01 ([A-Z_0-9]+)$/#define \1 0/' \
    -e 's/^#cmakedefine LLVM_ON_UNIX .*/#define LLVM_ON_UNIX 1/' \
    -e 's/^#cmakedefine .*$//' \
    -e 's/\$\{LLVM_VERSION_MAJOR\}/24/; s/\$\{LLVM_VERSION_MINOR\}/0/; s/\$\{LLVM_VERSION_PATCH\}/0/' \
    -e 's/\$\{PACKAGE_VERSION\}/24.0.0git/' \
    -e 's/\$\{LLVM_DEFAULT_TARGET_TRIPLE\}/arm64-apple-darwin/' \
    -e 's/\$\{[A-Z_]+\}//g' "$1"
}
conf "$SRC/llvm-config.h.cmake" > "$OUT/llvm-config.h"
conf "$SRC/abi-breaking.h.cmake" > "$OUT/abi-breaking.h"
