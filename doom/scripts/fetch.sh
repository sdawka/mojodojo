#!/usr/bin/env bash
# Fetch pinned third-party inputs (untrusted; each in its own dir).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
TP="$HERE/third_party"
DG_COMMIT=e257a09527062bfb986b6e5e0780839b6cbd5b71
FD_VER=0.13.0
FD_URL="https://github.com/freedoom/freedoom/releases/download/v${FD_VER}/freedoom-${FD_VER}.zip"
FD_SHA256=3f9b264f3e3ce503b4fb7f6bdcb1f419d93c7b546f4df3e874dd878db9688f59   # from freedoom-0.13.0-CHECKSUM
WAD_SHA256=7323bcc168c5a45ff10749b339960e98314740a734c30d4b9f3337001f9e703d
mkdir -p "$TP"; touch "$TP/.metadata_never_index"
if [ ! -d "$TP/doomgeneric/.git" ]; then
  git clone --quiet https://github.com/ozkl/doomgeneric "$TP/doomgeneric"
fi
git -C "$TP/doomgeneric" checkout --quiet "$DG_COMMIT"
mkdir -p "$TP/freedoom"
if [ ! -f "$TP/freedoom/freedoom1.wad" ]; then
  curl -fsSL -o "$TP/freedoom/freedoom.zip" "$FD_URL"
  echo "$FD_SHA256  $TP/freedoom/freedoom.zip" | shasum -a 256 -c -
  (cd "$TP/freedoom" && unzip -q -o -j freedoom.zip '*/freedoom1.wad' '*/COPYING.txt' '*/CREDITS.txt' && rm freedoom.zip)
fi
echo "$WAD_SHA256  $TP/freedoom/freedoom1.wad" | shasum -a 256 -c -
echo "doomgeneric @ $(git -C "$TP/doomgeneric" rev-parse HEAD)"
