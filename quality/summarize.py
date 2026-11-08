"""Collect alpha_raw/*.json and ppl logs into results/alpha.json and results/perplexity.json,
and print markdown tables for QUALITY.md. Adds roofline-predicted spec tok/s at the measured E/alpha.

usage: uv run --project /Users/sdawka/Code/llming/mlx python /Users/sdawka/Code/llming/quality/summarize.py
"""

import glob
import json
import math
import os
import re
from datetime import datetime, timezone

import mlx.core as mx
import mlx_lm
import numpy as np

ROOT = "/Users/sdawka/Code/llming"
Q = f"{ROOT}/quality"
GATE = 0.03
ROOF = json.load(open(f"{ROOT}/roofline/results/model.json"))

# pair -> (roofline model, roofline target fmt, roofline draft name, baseline tok/s, draft byte ratio vs modelled draft)
def draft_ratio(model, fmt_a, fmt_b):
    b = ROOF["models"][model]["decode_weight_bytes_per_token"]
    return b[fmt_a] / b[fmt_b]

PAIR_MAP = {
    "llama3b-4bit_1b-4bit": ("llama3.2-3b", "mlx/4bit", 1.0),
    "llama3b-8bit_1b-4bit": ("llama3.2-3b", "mlx/8bit", 1.0),
    "qwen4b-8bit_0.6b-4bit": ("qwen3-4b", "mlx/8bit", 1.0),
    "qwen4b-8bit_0.6b-8bit": ("qwen3-4b", "mlx/8bit", draft_ratio("qwen3-0.6b", "mlx_8bit_g64", "mlx_4bit_g64")),
    "qwen4b-4bit_0.6b-4bit": ("qwen3-4b", "mlx/4bit", 1.0),
    # proxy: llama.cpp Q4_K_M + 1B Q4_K_M has no roofline spec row; both 4-bit runtimes decode at the
    # measured 48.8 tok/s, so the MLX 4-bit target + 1B 4-bit draft step costs stand in.
    "llamacpp_q4km_3b_1b": ("llama3.2-3b", "mlx/4bit", 1.0),
}


def roof_rows(model, target):
    rows = {}
    for r in ROOF["candidates"][model]["speculative"]:
        if r["target"] == target:
            rows.setdefault(r["k"], r)
    return rows


def predict(pair, k, E):
    model, target, dr = PAIR_MAP[pair]
    r = roof_rows(model, target).get(k)
    if r is None:
        return None
    cal = r["tok_s_cal"] / r["tok_s"]
    t_iter = r["t_iter_ms"] + k * r["t_draft_ms"] * (dr - 1)  # scale draft cost for an 8-bit draft
    base = ROOF["candidates"][model]["reference"]["decode_tok_s"]
    tok_s = cal * 1e3 * E / t_iter
    return {"pred_tok_s_cal": tok_s, "gain_vs_baseline": tok_s / base - 1, "baseline_tok_s": base,
            "t_iter_ms_model": t_iter, "draft_cost_scaled_by": dr}


def E_iid(a, k):
    return (1 - a ** (k + 1)) / (1 - a) if a < 1 else k + 1


def alpha_section():
    out, md = {}, []
    md.append("| pair | k | acc_ratio (accepted/proposed) pooled | per-prompt range | alpha_iid pooled | per-prompt range | E tokens/verify | greedy == plain | roofline-pred tok/s @ measured E | gain vs baseline |")
    md.append("|---|---:|---:|---|---:|---|---:|---:|---:|---:|")
    for f in sorted(glob.glob(f"{Q}/results/alpha_raw/*.json")):
        if f.endswith("greedy_diag.json") or os.path.basename(f).startswith("llamacpp_"):
            continue
        d = json.load(open(f))
        pair = d["pair"]
        diag_f = f.replace(".json", ".greedy_diag.json")
        d["greedy_diag"] = json.load(open(diag_f)) if os.path.exists(diag_f) else None
        for k, t in d["by_k"].items():
            k = int(k)
            p = predict(pair, k, t["E_tokens_per_step"]) if pair in PAIR_MAP else None
            t["roofline_prediction_at_measured_E"] = p
            t["roofline_prediction_at_measured_alpha_iid"] = (
                predict(pair, k, E_iid(t["alpha_iid"], k)) if pair in PAIR_MAP else None)
            md.append(f"| {pair} | {k} | {t['acc_ratio']:.3f} | {t['acc_ratio_range'][0]:.2f}-{t['acc_ratio_range'][1]:.2f} | "
                      f"{t['alpha_iid']:.3f} | {t['alpha_iid_range'][0]:.2f}-{t['alpha_iid_range'][1]:.2f} | "
                      f"{t['E_tokens_per_step']:.2f} | {t['greedy_equal_prompts']}/{t['n_prompts']} | "
                      + (f"{p['pred_tok_s_cal']:.1f} | {p['gain_vs_baseline']*100:+.0f}% |" if p else "- | - |"))
        out[pair] = d
    lf = f"{Q}/results/alpha_raw/llamacpp_q4km_3b_1b.json"
    if os.path.exists(lf):
        d = json.load(open(lf))
        for k, t in d["by_k"].items():
            k = int(k)
            p = predict("llamacpp_q4km_3b_1b", k, t["E_est"])
            t["roofline_prediction_at_measured_E_proxy_mlx4bit_costs"] = p
            md.append(f"| llama.cpp Q4_K_M 3B + Q4_K_M 1B | {k} | {t['acc_ratio']:.3f} | "
                      f"{t['acc_ratio_range'][0]:.2f}-{t['acc_ratio_range'][1]:.2f} | {t['alpha_iid_from_E']:.3f} (from E) | - | "
                      f"{t['E_est']:.2f} (est.) | n/a | {p['pred_tok_s_cal']:.1f} (proxy) | {p['gain_vs_baseline']*100:+.0f}% |")
        out["llamacpp_q4km_3b_1b"] = d
    # other llama.cpp pairs (Gemma 3): alpha only here; predictions come from gemma_predict.py
    for lf in sorted(glob.glob(f"{Q}/results/alpha_raw/llamacpp_*.json")):
        name = os.path.basename(lf)[:-5]
        if name == "llamacpp_q4km_3b_1b":
            continue
        d = json.load(open(lf))
        for k, t in d["by_k"].items():
            md.append(f"| {name} | {k} | {t['acc_ratio']:.3f} | {t['acc_ratio_range'][0]:.2f}-{t['acc_ratio_range'][1]:.2f} | "
                      f"{t['alpha_iid_from_E']:.3f} (from E) | - | {t['E_est']:.2f} (est.) | "
                      f"same text as k=1: {t.get('same_text_as_k1_prompts', '-')}/{len(d['prompts'])} | see gemma_predict | - |")
        out[name] = d
    # Gemma 3 roofline predictions at the measured E (quality/gemma_predict.py)
    gp = f"{Q}/results/gemma_predictions.json"
    if os.path.exists(gp):
        for r in json.load(open(gp))["speculative"]:
            if r["pair"] in out:
                out[r["pair"]]["by_k"][str(r["k"])]["gemma_roofline_prediction"] = r
    md.append("")
    md.append("Per-prompt alpha_iid at k=3:")
    md.append("")
    names = None
    for pair, d in out.items():
        if pair.startswith("llamacpp"):
            continue
        if names is None:
            names = list(d["prompts"])
            md.append("| pair | " + " | ".join(names) + " |")
            md.append("|---|" + "---:|" * len(names))
        md.append(f"| {pair} | " + " | ".join(f"{d['prompts'][n]['k']['3']['alpha_iid']:.2f}" for n in names) + " |")
    return out, "\n".join(md)


def parse_llamacpp(path):
    s = open(path).read()
    m = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", s)
    ch = re.search(r"calculating perplexity over (\d+) chunks, n_ctx=(\d+)", s)
    return {"ppl": float(m.group(1)), "ppl_se": float(m.group(2)), "chunks": int(ch.group(1)), "ctx": int(ch.group(2))}


LLAMACPP_REF = {"llama3b": "F16"}  # every other family: BF16 GGUF reference
_G3 = f"{ROOT}/models/gemma3"
GGUF = {  # family-fmt -> model file (for the record)
    "llama3b": lambda q: f"{ROOT}/models/gguf/Llama-3.2-3B-Instruct-{q}.gguf",
    "gemma4b": lambda q: {"QAT_Q4_0": f"{_G3}/gguf-qat-google/gemma-3-4b-it-q4_0.gguf",
                          "QATlocal_Q4_0": f"{_G3}/gguf/gemma-3-4b-it-qat-localQ4_0.gguf",
                          "Q4_0": f"{_G3}/gguf/gemma-3-4b-it-plainQ4_0.gguf"}.get(q, f"{_G3}/gguf/gemma-3-4b-it-{q}.gguf"),
    "gemma1b": lambda q: f"{_G3}/gguf/gemma-3-1b-it-{q}.gguf",
    "gemma270m": lambda q: f"{_G3}/gguf/gemma-3-270m-it-{q}.gguf",
}


def ppl_section():
    rows = {}
    for p in sorted(glob.glob(f"{Q}/logs/ppl_llamacpp_*.log")):
        if "Final estimate" not in open(p).read():
            continue
        fam, fmt = os.path.basename(p)[len("ppl_llamacpp_"):-4].split("-", 1)
        r = parse_llamacpp(p)
        ref = LLAMACPP_REF.get(fam, "BF16")
        r.update({"runtime": "llama.cpp", "model": GGUF[fam](fmt) if fam in GGUF else None, "log": p,
                  "family": fam, "fmt": fmt, "ref": ref})
        rows[f"{fam}-llamacpp-{fmt}"] = r
    for f in glob.glob(f"{Q}/results/ppl_raw/*.json"):
        r = json.load(open(f))
        name = r["name"]
        fam, _, fmt = name.split("-")
        r.update({"family": fam, "fmt": fmt, "ref": "bf16"})
        rows[name] = r
    # deltas vs same-runtime 16-bit reference
    for name, r in rows.items():
        if r["runtime"] == "llama.cpp":
            ref = rows.get(f"{r['family']}-llamacpp-{r['ref']}")
        else:
            ref = rows.get(f"{r['family']}-mlx-bf16")
        if ref is None:
            r["delta_pct"] = None
            continue
        r["ref_name"] = [k for k, v in rows.items() if v is ref][0]
        r["delta_pct"] = 100 * (r["ppl"] / ref["ppl"] - 1)
        if r["runtime"] == "mlx":  # paired SE on the log-ratio from per-token NLLs
            a = np.load(f"{Q}/results/ppl_raw/{name}.nll.npy").astype(np.float64)
            b = np.load(f"{Q}/results/ppl_raw/{r['ref_name']}.nll.npy").astype(np.float64)
            dd = a - b
            r["delta_pct_se"] = 100 * math.exp(dd.mean()) * dd.std(ddof=1) / math.sqrt(dd.size)
        r["gate_le_3pct"] = None if ref is r else bool(r["delta_pct"] <= 100 * GATE)
        if "qat" in r["fmt"].lower() and r["runtime"] == "mlx" and r["fmt"] != "qatbf16" and f"{r['family']}-mlx-qatbf16" in rows:
            q = rows[f"{r['family']}-mlx-qatbf16"]
            r["delta_pct_vs_qat_unquantized_bf16"] = 100 * (r["ppl"] / q["ppl"] - 1)
    md = ["| runtime | model | format | PPL | ± | Δ vs 16-bit ref | gate (≤3%) |", "|---|---|---|---:|---:|---:|---|"]
    order = sorted(rows, key=lambda n: (rows[n]["runtime"], rows[n]["family"], -rows[n]["ppl"] if False else rows[n]["ppl"]))
    for n in order:
        r = rows[n]
        d = r.get("delta_pct")
        dtxt = "ref" if r.get("gate_le_3pct") is None and d == 0 else (
            "-" if d is None else f"{d:+.2f}%" + (f" ± {r['delta_pct_se']:.2f}" if "delta_pct_se" in r else ""))
        g = r.get("gate_le_3pct")
        gtxt = "-" if g is None else ("PASS" if g else "FAIL")
        md.append(f"| {r['runtime']} | {r['family']} | {r['fmt']} | {r['ppl']:.3f} | {r['ppl_se']:.3f} | {dtxt} | {gtxt} |")
    return rows, "\n".join(md)


def kld_section():
    """Gemma fidelity check: KL(ref || quant) + same top-1 on the first 64 ppl chunks (run_kld_gemma.sh)."""
    rows = {}
    for p in sorted(glob.glob(f"{Q}/logs/kld_llamacpp_*_vs_*.log")):
        s = open(p).read()
        m = re.search(r"Mean    KLD:\s+([\d.]+) ±\s+([\d.]+)", s)
        if not m or "inconsistent vocabulary" in s:
            continue
        fam_fmt, ref = os.path.basename(p)[len("kld_llamacpp_"):-4].split("_vs_")
        fam, fmt = fam_fmt.split("-", 1)
        g = lambda pat: float(re.search(pat, s).group(1))
        rows[f"{fam}-llamacpp-{fmt}_vs_{ref}"] = {
            "runtime": "llama.cpp", "family": fam, "fmt": fmt, "ref": ref, "log": p,
            "mean_kld": float(m.group(1)), "mean_kld_se": float(m.group(2)),
            "median_kld": g(r"Median  KLD:\s+([\d.]+)"), "p99_kld": g(r"99\.0%   KLD:\s+([\d.]+)"),
            "same_top1_pct": g(r"Same top p:\s+([\d.]+)"), "chunks": int(re.search(r"over (\d+) chunks", s).group(1)),
            "note": "PPL(base) printed in this log comes from llama.cpp's compressed stored log-probs; use the "
                    "full-run ppl rows for perplexity"}
    for f in sorted(glob.glob(f"{Q}/results/kld_raw/*.json")):
        r = json.load(open(f))
        name = r["name"]
        ref = "qatbf16" if name.endswith("_vs_qatbf16") else "bf16"
        fam, _, fmt = name.replace("_vs_qatbf16", "").split("-")
        r.update({"family": fam, "fmt": fmt, "ref": ref})
        rows[f"{name}" if ref != "bf16" else f"{name}_vs_bf16"] = r
    md = ["| runtime | format | vs | mean KLD | median | p99 | same top-1 |", "|---|---|---|---:|---:|---:|---:|"]
    for n in sorted(rows, key=lambda n: (rows[n]["runtime"], rows[n]["ref"], rows[n]["mean_kld"])):
        r = rows[n]
        md.append(f"| {r['runtime']} | {r['fmt']} | {r['ref']} | {r['mean_kld']:.4f} ± {r['mean_kld_se']:.4f} | "
                  f"{r['median_kld']:.4f} | {r['p99_kld']:.3f} | {r['same_top1_pct']:.2f}% |")
    return rows, "\n".join(md)


def main():
    meta = {"timestamp": datetime.now(timezone.utc).isoformat(), "authoritative_quality": True,
            "timing_authoritative": False,
            "versions": {"mlx": mx.__version__, "mlx_lm": mlx_lm.__version__,
                         "llama.cpp": "a691b2345844ce1b430201fc5fd404ce062e05f2"},
            "machine": "Apple M5, 32 GB (shared with other lanes during measurement)"}
    a, amd = alpha_section()
    json.dump({"meta": {**meta, "method": "quality/alpha.py; greedy, mlx_lm speculative_generate_step; "
                        "acc_ratio = accepted/proposed; alpha_iid = accepted/(accepted+rejections); "
                        "roofline predictions use roofline/results/model.json t_iter and calibration at measured E"},
               "pairs": a}, open(f"{Q}/results/alpha.json", "w"), indent=1)
    p, pmd = ppl_section()
    kr, kmd = kld_section()
    json.dump({"meta": {**meta, "dataset": "wikitext-2-raw-v1 test (Salesforce/wikitext parquet rows joined), "
                        f"{ROOT}/models/datasets/wiki.test.raw", "ctx": 512,
                        "method": "llama-perplexity method: non-overlapping 512-token chunks, BOS reset, score 2nd half",
                        "gate": "delta_pct <= 3 vs same-runtime 16-bit reference"},
               "runs": p,
               "kld": {"method": "first 64 wikitext ppl chunks (ctx 512, score 2nd half): KL(p_ref || p_quant) in "
                                 "nats per token, same top-1 %; llama.cpp --kl-divergence / quality/kld_mlx.py",
                       "runs": kr}}, open(f"{Q}/results/perplexity.json", "w"), indent=1)
    print(amd)
    print()
    print(pmd)
    print()
    print(kmd)


if __name__ == "__main__":
    main()
