"""X-DOOM-3 bench: the column/span drawing pass on the GPU (Mojo Metal kernel).

Run via `pixi run bench-tier3` (after `pixi run build`). Writes
results/x-doom-3.json and results/x-doom-3.png.

Bit-identity is checked two ways:
  1. Per frame, in process (doom-mojo-gpu-verify): upstream C draws every call
     on the real screen while the same calls are recorded; at the end of each
     view render the GPU replays the recording onto a snapshot of the
     pre-render screen and all 64,000 bytes are compared. Plus 100 batches of
     2,000 randomised overlapping draws of all eight drawers. Normal and low
     detail runs.
  2. SHA-256 of every frame of doom-mojo-gpu vs the C renderer, over the frames
     that are reproducible across runs (same method as tier 2).
fps: whole-frame headless timedemo, C vs Mojo CPU drawers vs Mojo GPU, interleaved.
"""

import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from doomrun import (  # noqa: E402
    DEMO, FRAMES, OUT, RESULTS, SHOW_FRAME, load_hashes, median, now_iso, run, write_json,
)

RUNS = int(os.environ.get("DOOM_RUNS", "3"))
HASH_RUNS_C, HASH_RUNS_V = 6, 2
VARIANTS = {"c": "doom-c", "mojo": "doom-mojo", "mojo-gpu": "doom-mojo-gpu"}
LOWDETAIL = ["--inject", "1500:191,3000:45,3001:45"]  # F5 low detail, then a smaller view

GPU_RE = re.compile(r"^GPU frames=(\d+) flushes=(\d+) records=(\d+) arena_bytes=(\d+) dedup_hits=(\d+) "
                    r"overflow_flushes=(\d+) outside_calls=(\d+) fallbacks=(\d+) flush_us_avg=([\d.]+)$", re.M)
EMPTY_RE = re.compile(r"^GPU replay on (.+?) api (\w+) empty_flush_us ([\d.]+)$", re.M)
CALLS_RE = re.compile(r"^GPU calls (\w+)=(\d+)$", re.M)
PROFILE_RE = re.compile(r"^GPU profile_us (.*)$", re.M)
V_ST_RE = re.compile(r"^GPUVERIFY selftest batches=(\d+) calls=(\d+) bad_batches=(\d+) bad_pixels=(\d+)$", re.M)
V_FR_RE = re.compile(r"^GPUVERIFY frames=(\d+) mismatched_frames=(\d+) mismatched_pixels=(\d+) "
                     r"sidefx_mismatches=(\d+) fallbacks=(\d+)$", re.M)


def gpu_stats(text):
    m = GPU_RE.findall(text)[-1]
    keys = ["frames", "flushes", "records", "arena_bytes", "dedup_hits", "overflow_flushes",
            "outside_calls", "fallbacks"]
    st = {k: int(v) for k, v in zip(keys, m[:8])}
    st["flush_us_avg"] = float(m[8])
    e = EMPTY_RE.search(text)
    st["device"], st["api"], st["empty_flush_us"] = e.group(1), e.group(2), float(e.group(3))
    st["calls"] = {k: int(v) for k, v in CALLS_RE.findall(text)}
    return st


def main():
    ok = True
    # 1) Per-frame in-process check + randomised self-test (authoritative).
    verify = {}
    for name, extra in [("normal", []), ("lowdetail", LOWDETAIL)]:
        r = run("doom-mojo-gpu-verify", "--headless", *extra, *DEMO)
        st = [tuple(map(int, x)) for x in V_ST_RE.findall(r["text"])]
        fr = [tuple(map(int, x)) for x in V_FR_RE.findall(r["text"])]
        b, c, bb, bp = st[-1]
        f, mf, mp, fx, fb = fr[-1]
        verify[name] = {"selftest_batches": b, "selftest_calls": c, "selftest_bad_batches": bb,
                        "selftest_bad_pixels": bp, "frames": f, "mismatched_frames": mf,
                        "mismatched_pixels": mp, "sidefx_mismatches": fx, "fallbacks": fb,
                        "gpu": gpu_stats(r["text"])}
        ok &= bb == 0 and bp == 0 and mf == 0 and mp == 0 and fx == 0 and f > 7000

    # 2) Whole-run frame hashes vs the C renderer (reproducible frames only).
    h = {}
    for name, binary, k in [("c", "doom-c", HASH_RUNS_C), ("mojo-gpu", "doom-mojo-gpu", HASH_RUNS_V)]:
        h[name] = []
        for i in range(k):
            path = os.path.join(OUT, f"t3-hashall-{name}-{i}.txt")
            run(binary, "--headless", "--hash-all", path, "--label", f"t3-{name}", "--out", OUT,
                "--capture", str(SHOW_FRAME), *DEMO)
            h[name].append(load_hashes(path))

    def unstable_frames(hs):
        return {f for f in hs[0] if any(x.get(f) != hs[0][f] for x in hs[1:])}

    n = len(h["c"][0])
    unstable = {name: unstable_frames(hs) for name, hs in h.items()}
    skip = unstable["c"] | unstable["mojo-gpu"]
    compared = [f for f in h["c"][0] if f not in skip]
    g = h["mojo-gpu"][0]
    bad = [f for f in compared if g.get(f) != h["c"][0][f]]
    raw = [f for f in h["c"][0] if g.get(f) != h["c"][0][f]]
    fixed_ok = all(g[f] == h["c"][0][f] for f in FRAMES)
    hash_check = {"frames": len(g), "runs_c": HASH_RUNS_C, "runs_variant": HASH_RUNS_V,
                  "frames_compared": len(compared), "frames_skipped_unstable": sorted(skip),
                  "mismatches": len(bad), "first_mismatches": bad[:10],
                  "raw_diff_vs_c_run0": len(raw), "fixed_frames_identical": fixed_ok}
    ok &= not bad and fixed_ok and len(g) == n
    shutil.copy(os.path.join(OUT, f"t3-mojo-gpu-frame{SHOW_FRAME}.png"), os.path.join(RESULTS, "x-doom-3.png"))

    # 3) Timing, interleaved to spread machine (and GPU) contention across variants.
    fps = {k: [] for k in VARIANTS}
    gpu_runs = []
    for _ in range(RUNS):
        for k, binary in VARIANTS.items():
            r = run(binary, "--headless", *DEMO)
            fps[k].append(r["result"]["fps"])
            if k == "mojo-gpu":
                s = gpu_stats(r["text"])
                gpu_runs.append({"flush_us_avg": s["flush_us_avg"], "empty_flush_us": s["empty_flush_us"]})
    med = {k: median(v) for k, v in fps.items()}

    # 4) One profiled run: per-stage time (each stage followed by a wait) and the
    #    GPU time of the replay kernel alone (16 back-to-back dispatches, one wait).
    r = run("doom-mojo-gpu", "--headless", *DEMO, env={"DOOM_GPU_PROFILE": "1"})
    p = PROFILE_RE.search(r["text"]).group(1).split()
    profile = {p[i]: float(p[i + 1]) for i in range(0, len(p) - 1, 2)}
    profile["fps"] = r["result"]["fps"]
    profile["empty_flush_us"] = gpu_stats(r["text"])["empty_flush_us"]

    gs = verify["normal"]["gpu"]
    per_frame_us = {k: 1e6 / v for k, v in med.items()}
    out = {
        "tier": "X-DOOM-3",
        "status": "pass" if ok else "fail",
        "fps": round(med["mojo-gpu"], 1),
        "fps_reference": round(med["c"], 1),
        "frame": "x-doom-3.png",
        "date": now_iso(),
        "backend": ("Column/span drawing pass (all 8 r_draw.c drawers) recorded per frame by C stubs, replayed "
                    "in one Mojo Metal kernel dispatch per frame on the M5 GPU; rest of renderer C"),
        "authoritative": False,
        "notes": (
            f"Headless timedemo, Freedoom DEMO1, {n} frames, median of {RUNS} interleaved runs: "
            f"C drawers {med['c']:.0f} fps, Mojo CPU drawers {med['mojo']:.0f} fps, Mojo GPU {med['mojo-gpu']:.0f} fps "
            f"({med['mojo-gpu'] / med['c']:.3f}x C). Bit-identical: per-frame in-process check (GPU replay vs "
            f"upstream C on the same pre-render screen, all 64,000 bytes) on {verify['normal']['frames']} + "
            f"{verify['lowdetail']['frames']} view renders (normal + low detail): "
            f"{verify['normal']['mismatched_frames'] + verify['lowdetail']['mismatched_frames']} mismatched frames; "
            f"randomised self-test {verify['normal']['selftest_calls']} overlapping calls of all 8 drawers: "
            f"{verify['normal']['selftest_bad_pixels']} bad pixels; SHA-256 of every frame vs C: "
            f"{hash_check['mismatches']} mismatches over {hash_check['frames_compared']}/{n} reproducible frames; "
            f"fixed frames {FRAMES} identical: {fixed_ok}. One GPU dispatch per frame "
            f"({gs['records'] / max(gs['flushes'], 1):.0f} records, {gs['arena_bytes'] / max(gs['flushes'], 1) / 1024:.0f} KiB "
            f"of texture snapshots per frame on average). Mean flush (record lists, upload, dispatch, wait, "
            f"download) {median([x['flush_us_avg'] for x in gpu_runs]):.0f} us per frame against "
            f"~{per_frame_us['c']:.0f} us for a whole CPU frame; of that the replay kernel itself is "
            f"~{profile['kernel_amortized_x16']:.0f} us of GPU time (16 back-to-back dispatches per wait), the rest is "
            f"submission/synchronisation/copy latency. The GPU was shared with other jobs "
            f"(MAX/MLX runs) during these measurements, which inflates every GPU number. Machine not idle-checked."
        ),
        "fps_mojo_cpu": round(med["mojo"], 1),
        "fps_runs": {k: [round(x, 1) for x in v] for k, v in fps.items()},
        "gpu_flush_runs": gpu_runs,
        "gpu_profile": profile,
        "hash_check": hash_check,
        "c_unstable_frames": sorted(unstable["c"]),
        "verify": verify,
        "frame_sha256": {str(f): g[f] for f in FRAMES},
    }
    write_json("x-doom-3.json", out)
    print(f"X-DOOM-3 {out['status']}: C {med['c']:.0f} fps, Mojo CPU {med['mojo']:.0f}, Mojo GPU {med['mojo-gpu']:.0f}; "
          f"hash mismatches {hash_check['mismatches']}/{hash_check['frames_compared']}; "
          f"verify {[(k, v['mismatched_frames'], v['frames'], v['selftest_bad_pixels']) for k, v in verify.items()]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
