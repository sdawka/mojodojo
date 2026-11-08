#!/bin/bash
# Fetches the toolchain sources the phantom plugins compile against into
# third_party/ (gitignored, marked .metadata_never_index). Sparse, blob-less
# checkouts of the exact commits that match the pinned nightly:
#
#   mojo 1.3.0.dev2026110805 reports internal hash 11efb73e (not on GitHub).
#   The public commit for that nightly is the release-pin commit
#     modular@2841b31631e7d31ae74f0ca20335ed1541cddec1
#     "[Release] Pin lockfiles to Mojo 1.3.0.dev2026110805, MAX 26.8.0.dev2026110805"
#   Its bazel/public-patches/llvm_source.bzl pins
#     llvm-project@fa8c67382e4fa68010fce98bbd267f91b397f80e (binary: LLVM 24.0.0git).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TP="$HERE/third_party"
mkdir -p "$TP"

fetch() { # dir url sha paths...
  local dir=$1 url=$2 sha=$3; shift 3
  if [ "$(git -C "$TP/$dir" rev-parse HEAD 2>/dev/null)" = "$sha" ]; then
    echo "$dir already at $sha"; return
  fi
  rm -rf "${TP:?}/$dir"
  git init -q "$TP/$dir"
  git -C "$TP/$dir" remote add origin "$url"
  git -C "$TP/$dir" config remote.origin.partialclonefilter blob:none
  git -C "$TP/$dir" sparse-checkout set --no-cone "$@"
  git -C "$TP/$dir" fetch -q --depth 1 --filter=blob:none origin "$sha"
  git -C "$TP/$dir" checkout -q FETCH_HEAD
  touch "$TP/$dir/.metadata_never_index"
  echo "$dir at $(git -C "$TP/$dir" rev-parse HEAD)"
}

fetch modular https://github.com/modular/modular.git \
  2841b31631e7d31ae74f0ca20335ed1541cddec1 \
  /Mojo/lib/Target/ /Mojo/include/ /Mojo/lib/Support/ \
  '/Mojo/lib/Compiler/ObjectCompiler/Target/' '/Mojo/lib/Compiler/ObjectCompiler/*.h' \
  /Mojo/lib/KGENToLLVM/Target/ '/Mojo/lib/KGENToLLVM/*.h' \
  /Mojo/stdlib/std/_plugin/ /Mojo/stdlib/std/sys/ /Mojo/stdlib/std/gpu/host/ \
  /Mojo/stdlib/std/memory/ /Mojo/stdlib/std/ffi/ /Mojo/stdlib/std/python/ \
  /Mojo/test/mojo-integration/compile_offload/ \
  /SDK/include/ /MLRT/test/Driver/ /MLRT/lib/Driver/ /Target/ \
  /max/mojo/max/_driver/ /max/mojo/max/_plugin/ \
  /Licenses/ '/LICENSE*' /bazel/ /MODULE.bazel '/WORKSPACE*'

fetch llvm-project https://github.com/llvm/llvm-project.git \
  fa8c67382e4fa68010fce98bbd267f91b397f80e \
  /llvm/include/llvm/ADT/ /llvm/include/llvm/Support/ /llvm/include/llvm/TargetParser/ \
  /llvm/include/llvm/Config/ /llvm/include/llvm/Demangle/ /llvm/include/llvm-c/ /llvm/LICENSE.TXT
touch "$TP/.metadata_never_index"
