"""Shared helpers for the DOOM bench scripts (run with `python3 -I`)."""

import json
import os
import re
import statistics
import subprocess
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(HERE, "build")
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(BUILD, "bench")
WAD = os.path.join(HERE, "third_party", "freedoom", "freedoom1.wad")
DEMO = ["-iwad", WAD, "-timedemo", "demo1", "-nogui"]
# Fixed frames used for reference hashes (frame N = Nth DG_DrawFrame call).
FRAMES = [100, 500, 1000, 2000, 3000, 5000, 7000]
SHOW_FRAME = 1000

RESULT_RE = re.compile(r"^RESULT (\{.*\})$", re.M)
TIMED_RE = re.compile(r"timed (\d+) gametics in (\d+) realtics \(([\d.]+) fps\)")
CAPTURE_RE = re.compile(r"^CAPTURE (\d+) ([0-9a-f]{64}) (\S+)$", re.M)
VERIFY_RE = re.compile(r"^VERIFY (\w+) calls=(\d+) mismatches=(\d+)$", re.M)


def run(binary, *args, env=None, require_result=True):
    """Run a doom binary from build/ (cwd = OUT so .default.cfg/.savegame land there)."""
    mojo_host = binary != "doom-stock-sdl"
    os.makedirs(OUT, exist_ok=True)
    cmd = [os.path.join(BUILD, binary), *args]
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run(cmd, cwd=OUT, env=e, capture_output=True, text=True, errors="replace")
    text = p.stdout + "\n" + p.stderr
    r = {"rc": p.returncode, "text": text}
    m = RESULT_RE.search(p.stdout)
    r["result"] = json.loads(m.group(1)) if m else None
    m = TIMED_RE.search(text)
    r["timed"] = (
        {"gametics": int(m.group(1)), "realtics": int(m.group(2)), "fps": float(m.group(3))}
        if m
        else None
    )
    r["captures"] = {int(f): (h, path) for f, h, path in CAPTURE_RE.findall(p.stdout)}
    r["verify"] = [(k, int(c), int(b)) for k, c, b in VERIFY_RE.findall(p.stdout)]
    # The engine ends -timedemo via I_Error -> exit(-1) (rc 255); anything else is a failure.
    if p.returncode not in (0, 255) or (mojo_host and require_result and r["result"] is None) or (not mojo_host and r["timed"] is None):
        raise RuntimeError(f"{binary} failed rc={p.returncode}\n{text[-2000:]}")
    return r


def load_hashes(path):
    with open(path) as f:
        return {int(a): b for a, b in (line.split() for line in f if line.strip())}


def median(xs):
    return statistics.median(xs) if xs else None


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
