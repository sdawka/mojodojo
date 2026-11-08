#!/usr/bin/env bash
# Fetch pinned upstream sources (see ../UPSTREAM.md) into third_party/src
# (gitignored), then install them into the isolated uv venv (.venv).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party/src
pins=(
  "m-labs/migen 4c2ae8dfeea37f235b52acb8166f12acaaae4f7c"
  "enjoy-digital/litex b6ae9e0b227354aecffef5339d3e946f2395ac09"
  "litex-hub/pythondata-cpu-vexriscv 642ecfed1c84460555d6d803d660cc60cfc1ecb6"
  "litex-hub/pythondata-software-picolibc 6a13ccce7c575b32c102dd9dc52178505b81fe39"
  "litex-hub/pythondata-software-compiler_rt 6eb76609c9627bf26635e57c63fb22cda7115887"
  "litex-hub/pythondata-misc-tapcfg a12c3f592c99f9c082fdc68c065b81cbd6e6b238"
)
for p in "${pins[@]}"; do
  repo=${p% *}; sha=${p#* }; d=third_party/src/$(basename "$repo")
  if [ "$(git -C "$d" rev-parse HEAD 2>/dev/null || true)" != "$sha" ]; then
    rm -rf "$d"; git init -q "$d"
    git -C "$d" fetch -q --depth 1 "https://github.com/$repo" "$sha"
    git -C "$d" checkout -q FETCH_HEAD
  fi
  echo "$(basename "$repo") $(git -C "$d" rev-parse HEAD)"
done
# picolibc is a submodule of the pythondata wrapper (pinned by the wrapper commit)
git -C third_party/src/pythondata-software-picolibc submodule update --init --depth 1 -q
git -C third_party/src/pythondata-software-picolibc submodule status
UV_CACHE_DIR="$PWD/third_party/uv-cache" uv sync --frozen --python 3.12
