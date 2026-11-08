#!/usr/bin/env bash
# Run DOOM on the Verilator SoC and compare with the host doomgeneric build.
#   pixi run doom-run            # frame 1 (title screen), MAX_FRAME/DUMP_FRAME env to change
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(cd ../.. && pwd)
WAD=$REPO/doom/third_party/freedoom/freedoom1.wad
MAX_FRAME=${MAX_FRAME:-1}; DUMP_FRAME=${DUMP_FRAME:-$MAX_FRAME}; HASH_EVERY=${HASH_EVERY:-1}
make -j8 -f fw/doom/doom.mk MAX_FRAME=$MAX_FRAME DUMP_FRAME=$DUMP_FRAME HASH_EVERY=$HASH_EVERY >/dev/null
# host reference: same args, fresh dir (no .default.cfg), virtual clock
HREF=build/hostref/f$MAX_FRAME; rm -rf "$HREF"; mkdir -p "$HREF"
ln -sf "$WAD" "$HREF/freedoom1.wad"
( cd "$HREF" && "$REPO/doom/build/doom-c" --headless --vclock --frames "$MAX_FRAME" --hash-all hashes.txt \
    --capture "$DUMP_FRAME" --out . --label host -iwad freedoom1.wad >host.log 2>&1 )
LOG=build/doom/fw/doom/uart-f$MAX_FRAME.log
python run.py build/doom/fw/doom/doom.bin --soc build/doom --wad 0x42000000:"$WAD" \
    --timeout "${TIMEOUT:-14400}" --log "$LOG" | grep -v '^FB \|loaded' || true
python scripts/compare_frame.py "$LOG" "$HREF/hashes.txt" --host-png "$HREF/host-frame$DUMP_FRAME.png" \
    --png build/doom/fw/doom/soc-frame$DUMP_FRAME.png
