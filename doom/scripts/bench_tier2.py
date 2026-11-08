"""X-DOOM-2 bench: C drawers vs Mojo drawers (scalar, SIMD). Hash + per-call check + fps.

Run via `pixi run bench-tier2` (after `pixi run build`). Writes
results/x-doom-2.json and results/x-doom-2.png.
"""

import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from doomrun import (  # noqa: E402
    DEMO, FRAMES, OUT, RESULTS, SHOW_FRAME, load_hashes, median, now_iso, run, write_json,
)

RUNS = int(os.environ.get("DOOM_RUNS", "7"))
HASH_RUNS_C, HASH_RUNS_V = 6, 2
MICRO_RE = re.compile(r"^MICRO (\w+) c_ns_per_px=([\d.]+) mojo_ns_per_px=([\d.]+)", re.M)
VARIANTS = {"c": "doom-c", "mojo": "doom-mojo", "mojo-simd": "doom-mojo-simd"}


def main():
    ok = True
    # 1) Whole-run frame hashes (SHA-256 of every frame). Upstream reads
    #    uninitialised memory: the startup melt starts from an uninitialised
    #    screen, and a few frames get a transient garbage pixel (zone-block
    #    headers hold ASLR/malloc-dependent pointers). So even two *C* runs
    #    differ on ~30 frames. A frame is compared only if it is reproducible
    #    across all runs of C *and* across all runs of the variant; a real
    #    drawer bug shows up as a reproducible difference.
    runs = {"c": ("doom-c", HASH_RUNS_C), "mojo": ("doom-mojo", HASH_RUNS_V),
            "mojo-simd": ("doom-mojo-simd", HASH_RUNS_V)}
    h = {}
    for name, (binary, k) in runs.items():
        h[name] = []
        for i in range(k):
            path = os.path.join(OUT, f"hashall-{name}-{i}.txt")
            run(binary, "--headless", "--hash-all", path, "--label", f"t2-{name}", "--out", OUT,
                "--capture", str(SHOW_FRAME), *DEMO)
            h[name].append(load_hashes(path))

    def unstable_frames(hs):
        return {f for f in hs[0] if any(x.get(f) != hs[0][f] for x in hs[1:])}

    n = len(h["c"][0])
    unstable = {name: unstable_frames(hs) for name, hs in h.items()}
    hash_check = {}
    for v in ("mojo", "mojo-simd"):
        skip = unstable["c"] | unstable[v]
        compared = [f for f in h["c"][0] if f not in skip]
        bad = [f for f in compared if h[v][0].get(f) != h["c"][0][f]]
        raw = [f for f in h["c"][0] if h[v][0].get(f) != h["c"][0][f]]
        fixed_ok = all(h[v][0][f] == h["c"][0][f] for f in FRAMES)
        hash_check[v] = {"frames": len(h[v][0]), "runs_c": HASH_RUNS_C, "runs_variant": HASH_RUNS_V,
                         "frames_compared": len(compared), "frames_skipped_unstable": sorted(skip),
                         "mismatches": len(bad), "first_mismatches": bad[:10],
                         "raw_diff_vs_c_run0": len(raw), "fixed_frames_identical": fixed_ok}
        ok &= not bad and fixed_ok and len(h[v][0]) == n
    stable = [f for f in h["c"][0] if f not in unstable["c"]]
    shutil.copy(os.path.join(OUT, f"t2-mojo-frame{SHOW_FRAME}.png"), os.path.join(RESULTS, "x-doom-2.png"))

    # 2) Per-call differential check (C and Mojo drawer run on identical input
    #    for every call; includes a 200k-call randomised self-test of all eight
    #    drawers) and a low-detail run (F5 at frame 1500, smaller view at 3000).
    verify = {}
    for name, binary, extra in [
        ("mojo", "doom-mojo-verify", []),
        ("mojo-simd", "doom-mojo-simd-verify", []),
        ("mojo-lowdetail", "doom-mojo-verify", ["--inject", "1500:191,3000:45,3001:45"]),
        ("mojo-simd-lowdetail", "doom-mojo-simd-verify", ["--inject", "1500:191,3000:45,3001:45"]),
    ]:
        r = run(binary, "--headless", *extra, *DEMO)
        totals = [(c, b) for k, c, b in r["verify"] if k == "total"]
        per = {}
        for k, c, b in r["verify"]:
            if k != "total" and k != "selftest":
                per.setdefault(k, [0, 0])
                per[k][0] += c
                per[k][1] += b
        verify[name] = {"selftest_calls": totals[0][0], "selftest_mismatches": totals[0][1],
                        "demo_calls": totals[1][0], "demo_mismatches": totals[1][1],
                        "per_drawer_calls_mismatches": per}
        ok &= all(b == 0 for _, b in totals) and len(totals) == 2

    # 3) Timing, interleaved to spread machine noise across variants.
    fps = {k: [] for k in VARIANTS}
    for _ in range(RUNS):
        for k, binary in VARIANTS.items():
            fps[k].append(run(binary, "--headless", *DEMO)["result"]["fps"])
    med = {k: median(v) for k, v in fps.items()}

    # 4) Drawer-only microbenchmark (shim/drawbench.c): ns/pixel, C vs Mojo, incl. stub/FFI cost.
    micro = {}
    for k in ("mojo", "mojo-simd"):
        r = run(VARIANTS[k], "--microbench", require_result=False)
        for m in MICRO_RE.finditer(r["text"]):
            micro.setdefault(m.group(1), {})["c_ns_per_px" if k == "mojo" else "c_ns_per_px_2"] = float(m.group(2))
            micro[m.group(1)][f"{k}_ns_per_px"] = float(m.group(3))

    out = {
        "tier": "X-DOOM-2",
        "status": "pass" if ok else "fail",
        "fps": round(med["mojo"], 1),
        "fps_reference": round(med["c"], 1),
        "frame": "x-doom-2.png",
        "date": now_iso(),
        "backend": "Mojo drawers (R_DrawColumn/Span/Fuzz/Translated + Low variants) via C marshalling stubs; rest of renderer C",
        "authoritative": False,
        "notes": (
            f"Headless timedemo, Freedoom DEMO1, {n} frames, median of {RUNS} interleaved runs: "
            f"C drawers {med['c']:.0f} fps, Mojo scalar {med['mojo']:.0f} fps "
            f"({100 * (med['mojo'] / med['c'] - 1):+.1f}%), Mojo SIMD {med['mojo-simd']:.0f} fps "
            f"({100 * (med['mojo-simd'] / med['c'] - 1):+.1f}%). Bit-identical: SHA-256 of every frame; frames "
            f"reproducible across {HASH_RUNS_C} C runs and {HASH_RUNS_V} runs of each Mojo variant all match C "
            f"(compared {min(v['frames_compared'] for v in hash_check.values())}/{n}; ~30 skipped because upstream "
            f"itself is nondeterministic there: startup melt from uninitialised memory, transient garbage pixels); "
            f"fixed frames {FRAMES} identical. Per-call check C vs Mojo on all demo calls (normal + low detail) plus "
            f"200k randomised calls: {'0 mismatches' if ok else 'MISMATCHES'}. Whole-frame fps, so drawer "
            f"differences are diluted by BSP/game logic/palette conversion. Drawer-only microbench, ns/pixel "
            f"C / Mojo / Mojo SIMD (Mojo incl. C stub + call): "
            + "; ".join(f"{k} {v.get('c_ns_per_px', 0):.3f}/{v.get('mojo_ns_per_px', 0):.3f}/{v.get('mojo-simd_ns_per_px', 0):.3f}"
                        for k, v in micro.items())
            + ". Machine not idle-checked."
        ),
        "fps_mojo_simd": round(med["mojo-simd"], 1),
        "fps_runs": {k: [round(x, 1) for x in v] for k, v in fps.items()},
        "hash_check": hash_check,
        "c_unstable_frames": sorted(unstable["c"]),
        "verify": verify,
        "frame_sha256": {str(f): h["mojo"][0][f] for f in FRAMES},
        "drawer_microbench": micro,
    }
    write_json("x-doom-2.json", out)
    print(f"X-DOOM-2 {out['status']}: C {med['c']:.0f} fps, Mojo {med['mojo']:.0f}, Mojo SIMD {med['mojo-simd']:.0f}; "
          f"hash mismatches {[hash_check[v]['mismatches'] for v in hash_check]}; "
          f"verify {[(k, v['demo_mismatches'], v['selftest_mismatches']) for k, v in verify.items()]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
