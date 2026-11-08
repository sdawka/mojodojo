"""Cross-check: llama.cpp/ggml MUL_MAT kernels via `test-backend-ops perf` (Metal + CPU backends).

ggml picks its own shapes (m=4096, k=14336, n in {1..512}); we keep its plain 2-D cases for
the weight types used by quantized LLMs. test-backend-ops reports one number per case, so the
whole sweep is repeated ATLAS_REPS times and stats are taken across sweeps.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import atlas_common as ac  # noqa: E402

BIN = Path(os.environ.get("LLAMACPP_BIN", ac.ATLAS.parent / "llamacpp/src/build/bin"))
TYPES = "f16|bf16|q8_0|q4_0|q4_K" if not ac.SMOKE else "f16|q4_0"
NS = "1|4|16|512" if not ac.SMOKE else "1|512"
GGML_REPS = max(5, int(os.environ.get("ATLAS_GGML_REPS", "5")))  # each sweep is slow
PARAMS = f"type_a=({TYPES}),type_b=f32,m=4096,n=({NS}),k=14336,bs=\\[1,1\\],nr=\\[1,1\\],per=\\[0,1,2,3\\],k_v=0,o=1"
LINE = re.compile(r"MUL_MAT\((?P<p>[^)]*)\):\s+(?P<runs>\d+) runs -\s+(?P<us>[\d.]+) us/run -\s+"
                  r"(?P<work>[\d.]+) (?P<wu>[KMGT]?FLOP)/run - \S*?\s*(?P<rate>[\d.]+) (?P<ru>[KMGT]?FLOPS)")
SCALE = {"": 1, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}


def sweep(backend: str) -> dict:
    p = subprocess.run([str(BIN / "test-backend-ops"), "perf", "-o", "MUL_MAT", "-b", backend,
                        "-p", PARAMS], capture_output=True, text=True)
    out = {}
    for line in re.sub(r"\x1b\[[0-9;]*m", "", p.stdout).splitlines():
        m = LINE.search(line)
        if m:
            gflops = float(m["rate"]) * SCALE[m["ru"][0] if m["ru"][0] in "KMGT" else ""] / 1e9
            out[m["p"]] = (gflops, float(m["us"]), int(m["runs"]))
    return out


def main():
    recs = []
    for backend in ("MTL0", "CPU"):
        acc = defaultdict(list)
        runs = {}
        for _ in range(GGML_REPS):
            for k, (g, us, r) in sweep(backend).items():
                acc[k].append(g)
                runs[k] = r
        for params, vals in acc.items():
            kv = dict(x.split("=", 1) for x in re.findall(r"(\w+=[^,\[]+|\w+=\[[^\]]*\])", params))
            m, n, k = int(kv["m"]), int(kv["n"]), int(kv["k"])
            st = ac.stats(vals)
            recs.append({"bench": "gemm", "engine": f"ggml_{backend.lower()}", "op": "MUL_MAT",
                         "dtype": f"{kv['type_a']}w x f32", "unit": "GFLOP/s", **st,
                         "shape": f"ggml_M{n}", "M": n, "N": m, "K": k,
                         "ggml_params": params, "ggml_runs_per_case": runs[params]})
            print(f"  {backend} {kv['type_a']:5s} n={n:4d} {st['median']:9.1f} GF/s (sd {st['stddev']:.1f})",
                  file=sys.stderr)
    commit = subprocess.run(["git", "-C", str(BIN.parents[1]), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    ac.write_results("ggml_mulmat", recs, {"llamacpp_commit": commit, "note":
                     "ggml convention: weight [m=N rows, k=K], activations n=M tokens"})


if __name__ == "__main__":
    main()
