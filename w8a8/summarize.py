"""Collects w8a8/results/ppl_raw/*.json -> w8a8/results/perplexity_w8a8.json (+ markdown table).

Δ% is vs the MAX bf16 run (same runtime, same tokenization); the paired standard error of
the mean-NLL difference uses the per-token NLL arrays. References: MLX bf16 and llama.cpp F16
from quality/results/perplexity.json.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "results/ppl_raw"
Q = Path("/Users/sdawka/Code/llming/quality/results")
BASE = "llama3b-max-bf16"
GATE = 3.0


def main() -> None:
    runs = {p.stem: json.loads(p.read_text()) for p in sorted(RAW.glob("*.json"))}
    nll = {k: np.load(RAW / f"{k}.nll.npy").astype(np.float64) for k in runs}
    ref_q = json.loads((Q / "perplexity.json").read_text())["runs"]
    mlx_nll = np.load(Q / "ppl_raw/llama3b-mlx-bf16.nll.npy").astype(np.float64)
    refs = {
        "llama3b-mlx-bf16": ref_q["llama3b-mlx-bf16"]["ppl"],
        "llama3b-llamacpp-F16": ref_q["llama3b-llamacpp-F16"]["ppl"],
        "llama3b-mlx-8bit": ref_q["llama3b-mlx-8bit"]["ppl"],
        "llama3b-llamacpp-Q8_0": ref_q["llama3b-llamacpp-Q8_0"]["ppl"],
    }
    base = runs[BASE]["ppl"]
    out_runs = {}
    rows = []
    for k, r in runs.items():
        d = nll[k] - nll[BASE]
        dmean = d.mean()
        dse = d.std(ddof=1) / math.sqrt(d.size)
        delta = 100 * (r["ppl"] / base - 1)
        # 95% CI of the ppl ratio from the paired mean-NLL difference
        lo, hi = 100 * (math.exp(dmean - 1.96 * dse) - 1), 100 * (math.exp(dmean + 1.96 * dse) - 1)
        e = dict(r)
        e.update({
            "delta_pct_vs_max_bf16": delta,
            "delta_pct_ci95_paired": [lo, hi],
            "delta_pct_vs_mlx_bf16": 100 * (r["ppl"] / refs["llama3b-mlx-bf16"] - 1),
            "delta_pct_vs_llamacpp_f16": 100 * (r["ppl"] / refs["llama3b-llamacpp-F16"] - 1),
            "gate_3pct_pass": None if k == BASE else bool(delta <= GATE),
        })
        if k == BASE:
            e["max_abs_token_nll_diff_vs_mlx_bf16"] = float(np.abs(nll[k] - mlx_nll).max())
            e["mean_nll_diff_vs_mlx_bf16"] = float((nll[k] - mlx_nll).mean())
        out_runs[k] = e
        rows.append((k, r.get("w8a8_mode"), r.get("w8a8_route"), r.get("w8a8_smooth"), r["ppl"], r["ppl_se"], delta, lo, hi))

    res = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "authoritative_quality": True,
            "timing_authoritative": False,
            "method": "wikitext-2 raw test (models/datasets/wiki.test.raw), ctx 512, non-overlapping chunks, "
                      "BOS reset, score 2nd half, fp32 logits -> float64 log-softmax NLL (quality/ppl_mlx.py method)",
            "gate": f"ppl <= +{GATE}% vs bf16 reference (MAX bf16, same runtime/tokenization)",
            "note": "Every scored token is a prefill token, so this is the quality of the W8A8 prefill path. "
                    "Per-token activation scales make W8A8 batch-independent, so this is also the quality of "
                    "W8A8 everywhere (prefill+decode); prefill-only W8A8 with bf16 decode can only be closer to bf16.",
            "versions": {"max": "26.8.0.dev2026110805 (fpga/maxop pixi env)", "mojo": "1.3.0.dev2026110805"},
            "references": refs,
        },
        "runs": out_runs,
    }
    (HERE / "results/perplexity_w8a8.json").write_text(json.dumps(res, indent=1))
    print("| run | mode | route | smooth | ppl ± se | Δ vs MAX bf16 | paired 95% CI |")
    print("|---|---|---|---|---|---|---|")
    for k, m, rt, sm, p, se, d, lo, hi in rows:
        print(f"| {k} | {m} | {rt or '-'} | {Path(sm).name if sm else '-'} | {p:.4f} ± {se:.4f} | {d:+.2f}% | [{lo:+.2f}, {hi:+.2f}] |")


if __name__ == "__main__":
    main()
