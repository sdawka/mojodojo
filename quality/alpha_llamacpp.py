"""Speculative acceptance with llama.cpp `llama-speculative` (greedy, --temp 0, p_min 0, n_min 0):
Llama 3.2 3B Q4_K_M target + Llama 3.2 1B Q4_K_M draft, same prompt set as alpha.py (quality/prompts/,
written by dump_prompts.py). llama-speculative reports n_drafted / n_accept; acc_ratio = n_accept/n_drafted.
Verify steps are estimated as n_drafted / k (every step drafts k when p_min = 0, except near the end),
E = n_predict / steps, and alpha_iid is solved from E = (1 - a^(k+1)) / (1 - a).

usage: uv run --project /Users/sdawka/Code/llming/mlx python /Users/sdawka/Code/llming/quality/alpha_llamacpp.py
       (defaults = the Llama run above). Gemma 3, e.g.:
       ... alpha_llamacpp.py --name llamacpp_gemma4b-qatlocalq40_1b-q4km \
           --target models/gemma3/gguf/gemma-3-4b-it-qat-localQ4_0.gguf --draft models/gemma3/gguf/gemma-3-1b-it-Q4_K_M.gguf \
           --prompts quality/prompts/gemma        (relative paths are under the repo root)
"""

import argparse

import glob
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone

ROOT = "/Users/sdawka/Code/llming"
BIN = f"{ROOT}/llamacpp/src/build/bin/llama-speculative"
TGT = f"{ROOT}/models/gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
DRF = f"{ROOT}/models/gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
N = 192


def solve_alpha(E, k):
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        e = (1 - mid ** (k + 1)) / (1 - mid) if mid < 1 else k + 1
        lo, hi = (mid, hi) if e < E else (lo, mid)
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=TGT)
    ap.add_argument("--draft", default=DRF)
    ap.add_argument("--prompts", default=f"{ROOT}/quality/prompts")
    ap.add_argument("--name", default="llamacpp_q4km_3b_1b")
    a = ap.parse_args()
    ab = lambda p: p if os.path.isabs(p) else f"{ROOT}/{p}"
    tgt, drf = ab(a.target), ab(a.draft)
    res = {"name": a.name, "target": tgt, "draft": drf, "n_predict": N, "prompts_dir": ab(a.prompts),
           "prompts": {}, "by_k": {}}
    prompts = sorted(glob.glob(f"{ab(a.prompts)}/*.txt"))
    for k in [1, 2, 3, 4]:
        tot = {"n_predict": 0, "n_drafted": 0, "n_accept": 0, "steps": 0}
        for p in prompts:
            name = os.path.basename(p)[:-4]
            proc = subprocess.run(
                [BIN, "-m", tgt, "-md", drf, "-ngl", "99", "-ngld", "99", "--spec-draft-n-max", str(k),
                 "--spec-draft-n-min", "0", "--spec-draft-p-min", "0", "-n", str(N), "--temp", "0",
                 "-c", "4096", "-f", p], capture_output=True, text=True)
            out = proc.stderr
            g = lambda key: int(re.search(rf"{key}\s*=\s*(\d+)", out).group(1))
            r = {"n_predict": g("n_predict"), "n_drafted": g("n_drafted"), "n_accept": g("n_accept")}
            r["steps_est"] = r["n_drafted"] / k
            r["acc_ratio"] = r["n_accept"] / r["n_drafted"]
            r["E_est"] = r["n_predict"] / r["steps_est"]
            r["alpha_iid_from_E"] = solve_alpha(r["E_est"], k)
            # greedy output must not depend on k (proxy for spec == plain greedy; stdout = prompt + generation)
            r["stdout_sha256"] = hashlib.sha256(proc.stdout.encode()).hexdigest()
            res["prompts"].setdefault(name, {})[k] = r
            for f in ["n_predict", "n_drafted", "n_accept"]:
                tot[f] += r[f]
            tot["steps"] += r["steps_est"]
            print(f"{a.name} {name:13s} k={k} acc={r['acc_ratio']:.3f} E~{r['E_est']:.2f} a_iid~{r['alpha_iid_from_E']:.3f}", flush=True)
        tot["acc_ratio"] = tot["n_accept"] / tot["n_drafted"]
        tot["E_est"] = tot["n_predict"] / tot["steps"]
        tot["alpha_iid_from_E"] = solve_alpha(tot["E_est"], k)
        per = [res["prompts"][n][k]["acc_ratio"] for n in res["prompts"]]
        tot["acc_ratio_range"] = [min(per), max(per)]
        tot["same_text_as_k1_prompts"] = sum(res["prompts"][n][k]["stdout_sha256"] == res["prompts"][n][1]["stdout_sha256"]
                                             for n in res["prompts"])
        res["by_k"][k] = tot
        print(f"POOLED {a.name} k={k}: acc={tot['acc_ratio']:.3f} E~{tot['E_est']:.2f} a_iid~{tot['alpha_iid_from_E']:.3f}")
    res["meta"] = {"timestamp": datetime.now(timezone.utc).isoformat(), "authoritative_quality": True,
                   "timing_authoritative": False,
                   "versions": {"llama.cpp": "a691b2345844ce1b430201fc5fd404ce062e05f2"}}
    os.makedirs(f"{ROOT}/quality/results/alpha_raw", exist_ok=True)
    json.dump(res, open(f"{ROOT}/quality/results/alpha_raw/{a.name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
