"""X-DOOM-1 bench: Mojo-hosted DOOM, headless timedemo + SDL-path comparison.

Run via `pixi run bench` (after `pixi run build`). Writes results/x-doom-1.json
and results/x-doom-1.png.
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from doomrun import (  # noqa: E402
    DEMO, FRAMES, OUT, RESULTS, SHOW_FRAME, median, now_iso, run, write_json,
)

RUNS = int(os.environ.get("DOOM_RUNS", "5"))


def main():
    frames = ",".join(str(f) for f in FRAMES)
    # 1) Reference hashes: two headless capture runs must agree.
    caps = []
    for i in range(2):
        r = run("doom-c", "--headless", "--label", f"t1-{i}", "--out", OUT, "--capture", frames, *DEMO)
        nframes = r["result"]["frames"]
        caps.append({f: h for f, (h, _) in r["captures"].items()})
    deterministic = caps[0] == caps[1] and len(caps[0]) == len(FRAMES)
    shutil.copy(os.path.join(OUT, f"t1-0-frame{SHOW_FRAME}.png"), os.path.join(RESULTS, "x-doom-1.png"))

    # 2) Headless timedemo fps (Mojo frame timer, first->last DG_DrawFrame).
    headless = [run("doom-c", "--headless", *DEMO)["result"]["fps"] for _ in range(RUNS)]

    # 3) SDL path, apples to apples: Mojo host vs stock upstream C host
    #    (doomgeneric_sdl.c), both 320x200 with SDL's offscreen-less dummy video
    #    driver; fps as reported by the engine itself (gametics*35/realtics).
    env = {"SDL_VIDEODRIVER": "dummy"}
    mojo_sdl, stock_sdl, mojo_sdl_timer = [], [], []
    for _ in range(3):
        r = run("doom-c", "--scale", "1", *DEMO, env=env)
        mojo_sdl.append(r["timed"]["fps"])
        mojo_sdl_timer.append(r["result"]["fps"])
        s = run("doom-stock-sdl", *DEMO, env=env)
        stock_sdl.append(s["timed"]["fps"])

    out = {
        "tier": "X-DOOM-1",
        "status": "pass" if deterministic else "fail",
        "fps": round(median(mojo_sdl), 1),
        "fps_reference": round(median(stock_sdl), 1),
        "frame": "x-doom-1.png",
        "date": now_iso(),
        "backend": "Mojo host (game loop, DG_* platform layer, framebuffer, SDL2 via FFI) + doomgeneric C engine",
        "authoritative": False,
        "notes": (
            f"Freedoom 0.13.0 freedoom1.wad DEMO1 (E1M6), -timedemo, {nframes} frames. "
            f"fps/fps_reference = engine-reported timedemo fps through the SDL2 path with "
            f"SDL_VIDEODRIVER=dummy: Mojo host vs stock upstream doomgeneric_sdl.c (pure C), medians of 3. "
            f"Headless Mojo host (no SDL, virtual clock): median {median(headless):.0f} fps over {RUNS} runs. "
            f"Frame hashes for frames {FRAMES} reproducible across runs: {deterministic}. Machine not idle-checked."
        ),
        "fps_headless": round(median(headless), 1),
        "fps_headless_runs": [round(x, 1) for x in headless],
        "fps_sdl_dummy_runs": {"mojo_host": mojo_sdl, "stock_c_host": stock_sdl,
                               "mojo_host_frame_timer": [round(x, 1) for x in mojo_sdl_timer]},
        "frame_sha256": {str(f): caps[0][f] for f in sorted(caps[0])},
        "hash_of": "raw 320x200 XRGB8888 DG_ScreenBuffer bytes after DG_DrawFrame N",
    }
    write_json("x-doom-1.json", out)
    print(f"X-DOOM-1: headless {out['fps_headless']} fps; SDL-dummy Mojo host {out['fps']} vs stock C {out['fps_reference']}; deterministic={deterministic}")
    return 0 if deterministic else 1


if __name__ == "__main__":
    sys.exit(main())
