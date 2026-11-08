"""Shared helpers for atlas Python benchmarks: stats, metadata, JSON output."""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ATLAS = Path(__file__).resolve().parent
RESULTS = ATLAS / "results"
SMOKE = os.environ.get("ATLAS_SMOKE", "0") == "1"
REPS = max(5, int(os.environ.get("ATLAS_REPS", "5" if SMOKE else "15")))
WARMUP = int(os.environ.get("ATLAS_WARMUP", "1" if SMOKE else "3"))
MIN_REP_S = 0.02 if SMOKE else 0.1
COLD_BYTES = int(os.environ.get("ATLAS_COLD_MIB", "256")) << 20


def stats(values: list[float]) -> dict:
    v = sorted(values)
    return {
        "median": statistics.median(v),
        "min": v[0],
        "max": v[-1],
        "mean": statistics.fmean(v),
        "stddev": statistics.stdev(v) if len(v) > 1 else 0.0,
        "reps": len(v),
    }


def ncopies_for(weight_bytes: int, cap: int = 64) -> int:
    """Copies of the weight needed so cycling through them exceeds COLD_BYTES (cache-cold)."""
    return max(1, min(cap, -(-COLD_BYTES // max(1, weight_bytes))))


def _cmd(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def meta(extra: dict | None = None) -> dict:
    m = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "smoke": SMOKE,
        "authoritative": not SMOKE and os.environ.get("ATLAS_IDLE_CONFIRMED", "0") == "1",
        "reps": REPS,
        "warmup": WARMUP,
        "python": sys.version.split()[0],
        "macos": platform.mac_ver()[0],
        "uname": platform.uname()._asdict(),
        "load_avg_at_start": os.getloadavg(),
        "env": {k: v for k, v in os.environ.items() if k.startswith(("ATLAS_", "MLX_", "MODULAR_"))},
    }
    if extra:
        m.update(extra)
    return m


def write_results(name: str, records: list[dict], extra_meta: dict | None = None) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if SMOKE else ""
    out = RESULTS / f"{name}{suffix}.json"
    m = meta(extra_meta)
    m["load_avg_at_end"] = os.getloadavg()
    out.write_text(json.dumps({"meta": m, "records": records}, indent=1))
    print(f"wrote {out} ({len(records)} records)", file=sys.stderr)
    return out


def calibrate_inner(fn, min_rep_s: float = MIN_REP_S, cap: int = 10000) -> int:
    fn(1)  # first call may include JIT / pipeline compile: never calibrate on it
    fn(1)
    probe = 1
    while True:
        t0 = time.perf_counter()
        fn(probe)
        dt = time.perf_counter() - t0
        if dt > min_rep_s / 10 or probe >= cap:
            break
        probe *= 4
    one = dt / probe
    return max(1, min(cap, int(min_rep_s / max(one, 1e-7)) + 1))
