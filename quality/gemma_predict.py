"""Roofline predictions for the Gemma 3 4B speculative-decoding candidates at the *measured* acceptance.

Uses the G2-M2 roofline (roofline/roofline: atlas rates, fitted eta_d/c_d, per-runtime calibration from
roofline/results/model.json) and adds the formats the Gemma quality runs introduced:
  * llama.cpp Q4_0 with an F16 tied embedding/lm_head  (Google's QAT GGUF; 1.34 GB lm_head per token)
  * llama.cpp Q4_0 with the llama-quantize default Q6_K output (QAT re-quantized locally; = roofline q4_0)
  * MLX 4-bit g32 on the Q4_0 grid + 6-bit g64 embedding (make_mlx_quant.py --q40-grid)
  * Gemma 3 270M as a draft (not in the roofline's model list)
For every measured pair (results/alpha_raw/*gemma*.json) and k it predicts decode tok/s at the measured E
(pooled over the 9 prompts, and the raw bench prompt alone) and the gain against the same target without a
draft (its own no-draft baseline) and against the best gate-passing no-draft Gemma config.
All numbers are model predictions, not measurements.

usage: uv run --project /Users/sdawka/Code/llming/roofline python /Users/sdawka/Code/llming/quality/gemma_predict.py
"""

import glob
import json
import os
import sys
from datetime import datetime, timezone

ROOT = "/Users/sdawka/Code/llming"
sys.path.insert(0, f"{ROOT}/roofline")
from roofline.accounting import FORMATS, ModelSpec  # noqa: E402
from roofline.atlas import Atlas  # noqa: E402
from roofline.predict import GG, MLX_I4, MLX_I8, Q6K_PROXY, RUNTIMES, SPEC_ITER_OVERHEAD_MS, Params, Predictor, Runtime, parts  # noqa: E402

Q = f"{ROOT}/quality"
CTX = 576  # roofline decode context (512-token prompt + generated tokens)
GATE = 3.0

# ---- extra formats / runtimes ------------------------------------------------------------------
FORMATS["q4_0_f16emb"] = {"bpw": 4.5, "lm_head_bpw": 16.0, "desc": "GGUF Q4_0 with F16 token_embd/output (Google QAT GGUF)"}
FORMATS["mlx_q40g32_emb6"] = {"bpw": 5.0, "lm_head_bpw": 6.5,
                              "desc": "MLX affine int4 g32 on the Q4_0 grid (bf16 scale+bias) + 6-bit g64 embedding"}
FORMATS["mlx_4bit_g64_emb8"] = {"bpw": 4.5, "lm_head_bpw": 8.5, "desc": "MLX int4 g64 + 8-bit embedding"}
RUNTIMES["llama.cpp/Q4_0-f16emb"] = Runtime(
    "llama.cpp/Q4_0-f16emb", "llama.cpp", "q4_0_f16emb",
    {"main": [GG("q4_0w x f32")], "q6k": Q6K_PROXY, "lm_head": [GG("f16w x f32")]},
    {"main": [GG("q4_0w x f32")], "q6k": Q6K_PROXY, "lm_head": [GG("f16w x f32")]}, GG("f16w x f32"))
RUNTIMES["mlx/q40g32-emb6"] = Runtime(
    "mlx/q40g32-emb6", "mlx", "mlx_q40g32_emb6",
    {"main": [MLX_I4], "q6k": [MLX_I4, MLX_I8], "lm_head": [MLX_I4, MLX_I8]},
    {"main": [MLX_I4], "q6k": [MLX_I4, MLX_I8], "lm_head": [MLX_I4, MLX_I8]}, ("mlx_nax", "bf16"),
    proxy=["g32 kernel unmeasured: g64 int4 per-byte rate; 6-bit lm_head: mean of int4/int8 rates"])

# ---- models ---------------------------------------------------------------------------------------
S4, S1 = ModelSpec.load("gemma3-4b"), ModelSpec.load("gemma3-1b")
S270 = ModelSpec("gemma3-270m", "unsloth/gemma-3-270m-it", d=640, f=2048, L=18, n_h=4, n_kv=1, hd=256, V=262144,
                 tied=True, window=512, global_every=6, notes=["config: models/gemma3/gemma-3-270m-it/config.json"])
DRAFT = {"1b": S1, "270m": S270}

# measured pair -> (target runtime key, draft key, draft runtime key, ppl-row name of the target)
MLX_T = {"8bit": ("mlx/8bit", "gemma4b-mlx-8bit"), "4bit": ("mlx/4bit", "gemma4b-mlx-4bit"),
         "qat4bit": ("mlx/4bit", "gemma4b-mlx-qat4bit"), "qatq40grid": ("mlx/q40g32-emb6", "gemma4b-mlx-qatq40grid"),
         "bf16": ("mlx/bf16", "gemma4b-mlx-bf16")}
LC_T = {"qatgoogle": ("llama.cpp/Q4_0-f16emb", "gemma4b-llamacpp-QAT_Q4_0"),
        "qatlocal": ("llama.cpp/Q4_0", "gemma4b-llamacpp-QATlocal_Q4_0"),
        "q4km": ("llama.cpp/Q4_K_M", "gemma4b-llamacpp-Q4_K_M"), "q8_0": ("llama.cpp/Q8_0", "gemma4b-llamacpp-Q8_0"),
        "q4_0": ("llama.cpp/Q4_0", "gemma4b-llamacpp-Q4_0")}
LC_D = {"1b-q4km": ("1b", "llama.cpp/Q4_K_M"), "270m-q8_0": ("270m", "llama.cpp/Q8_0"),
        "1bqat-q4_0": ("1b", "llama.cpp/Q4_0"), "270mqat-q4_0": ("270m", "llama.cpp/Q4_0")}


def load_ctx():
    m = json.load(open(f"{ROOT}/roofline/results/model.json"))
    fp = m["fitted_params"]["all_rows"]
    P = Params(eta_d=fp["eta_d"], c_d_ms=fp["c_d_ms"], eta_p=fp["eta_p"], c_http_ms=fp["c_http_ms"])
    cal = {k: v["decode_tok_s"] for k, v in m["runtime_calibration"].items()}
    return Predictor(Atlas()), P, cal


def main():
    pred, P, cal = load_ctx()
    ppl = json.load(open(f"{Q}/results/perplexity.json"))["runs"] if os.path.exists(f"{Q}/results/perplexity.json") else {}

    def nodraft(rk):
        rt = RUNTIMES[rk]
        return 1e3 / pred.decode_ms(S4, rt, CTX, P) * cal[rt.runtime]

    def gate(name):
        r = ppl.get(name)
        return None if r is None or r.get("delta_pct") is None else r["delta_pct"] <= GATE

    base = {}
    for rk, name in list(MLX_T.values()) + list(LC_T.values()):
        base[name] = {"runtime_key": rk, "pred_tok_s_cal": nodraft(rk), "gate_pass": gate(name),
                      "delta_pct": (ppl.get(name) or {}).get("delta_pct"),
                      "weight_gb": sum(b for _, _, b, _ in parts(S4, RUNTIMES[rk].fmt)) / 1e9}
    valid = {n: b for n, b in base.items() if b["gate_pass"]}
    best_valid = max(valid.items(), key=lambda kv: kv[1]["pred_tok_s_cal"]) if valid else None

    rows = []
    for f in sorted(glob.glob(f"{Q}/results/alpha_raw/*gemma*.json")):
        if f.endswith("greedy_diag.json"):
            continue
        d = json.load(open(f))
        name = os.path.basename(f)[:-5]
        if name.startswith("llamacpp_"):
            t, dr = name[len("llamacpp_gemma4b-"):].split("_", 1)
            rk, tname = LC_T[t]
            dkey, drk = LC_D[dr]
            Ek = {int(k): (v["E_est"], v["alpha_iid_from_E"], v["acc_ratio"]) for k, v in d["by_k"].items()}
            Eb = {int(k): d["prompts"]["raw_bench512"][k]["E_est"] for k in d["by_k"]}
        else:
            t, dr = name[len("gemma4b-"):].split("_", 1)
            rk, tname = MLX_T[t]
            dkey, dfmt = dr.split("-", 1)
            dkey = dkey.replace("qat", "")
            drk = {"4bit": "mlx/4bit", "8bit": "mlx/8bit", "bf16": "mlx/bf16", "q40grid": "mlx/q40g32-emb6"}[dfmt]
            Ek = {int(k): (v["E_tokens_per_step"], v["alpha_iid"], v["acc_ratio"]) for k, v in d["by_k"].items()}
            Eb = {int(k): d["prompts"]["raw_bench512"]["k"][k]["E_tokens_per_step"] for k in d["by_k"]}
        trt, drt = RUNTIMES[rk], RUNTIMES[drk]
        for k, (E, a_iid, acc) in sorted(Ek.items()):
            t_d = pred.decode_ms(DRAFT[dkey], drt, CTX, P, 1)
            t_v = pred.decode_ms(S4, trt, CTX, P, k + 1)
            t_iter = k * t_d + t_v + SPEC_ITER_OVERHEAD_MS
            c = cal[trt.runtime]
            tok, tok_b = 1e3 * E / t_iter * c, 1e3 * Eb[k] / t_iter * c
            own = base[tname]["pred_tok_s_cal"]
            r = {"pair": name, "target": rk, "target_ppl_row": tname, "target_gate_pass": base[tname]["gate_pass"],
                 "draft": f"gemma3-{dkey} {drk}", "k": k, "alpha_iid": a_iid, "acc_ratio": acc, "E_pooled": E,
                 "E_bench": Eb[k], "t_draft_ms": t_d, "t_verify_ms": t_v, "t_iter_ms": t_iter,
                 "pred_tok_s_pooled": tok, "pred_tok_s_bench": tok_b, "own_nodraft_tok_s": own,
                 "gain_vs_own_pooled": tok / own - 1, "gain_vs_own_bench": tok_b / own - 1}
            if best_valid:
                r["gain_vs_best_valid_nodraft_pooled"] = tok / best_valid[1]["pred_tok_s_cal"] - 1
                r["gain_vs_best_valid_nodraft_bench"] = tok_b / best_valid[1]["pred_tok_s_cal"] - 1
            rows.append(r)

    out = {"meta": {"timestamp": datetime.now(timezone.utc).isoformat(), "authoritative_quality": False,
                    "timing_authoritative": False, "kind": "roofline model predictions at measured E (not measurements)",
                    "ctx": CTX, "params": P.__dict__, "calibration_decode": cal,
                    "draft_270m_note": "270M step cost is byte-scaled through the atlas; small-model per-dispatch "
                                       "overhead is only in the per-layer c_d term, so it is probably optimistic"},
           "nodraft": base, "best_gate_valid_nodraft": best_valid[0] if best_valid else None, "speculative": rows}
    json.dump(out, open(f"{Q}/results/gemma_predictions.json", "w"), indent=1)

    print("| no-draft config | runtime key | weights GB/token | pred tok/s | ppl Δ | gate |")
    print("|---|---|---:|---:|---:|---|")
    for n, b in sorted(base.items(), key=lambda kv: -kv[1]["pred_tok_s_cal"]):
        dl = "-" if b["delta_pct"] is None else f"{b['delta_pct']:+.2f}%"
        g = "-" if b["gate_pass"] is None else ("PASS" if b["gate_pass"] else "FAIL")
        print(f"| {n} | {b['runtime_key']} | {b['weight_gb']:.2f} | {b['pred_tok_s_cal']:.1f} | {dl} | {g} |")
    print()
    print("| pair | k | α_iid | E pooled | E bench | t_draft ms | t_iter ms | pred tok/s pooled | vs own no-draft | pred tok/s bench | vs own | vs best gate-valid no-draft (pooled / bench) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        bv = (f"{r['gain_vs_best_valid_nodraft_pooled']*100:+.0f}% / {r['gain_vs_best_valid_nodraft_bench']*100:+.0f}%"
              if "gain_vs_best_valid_nodraft_pooled" in r else "-")
        print(f"| {r['pair']} | {r['k']} | {r['alpha_iid']:.3f} | {r['E_pooled']:.2f} | {r['E_bench']:.2f} | "
              f"{r['t_draft_ms']:.1f} | {r['t_iter_ms']:.1f} | {r['pred_tok_s_pooled']:.1f} | {r['gain_vs_own_pooled']*100:+.0f}% | "
              f"{r['pred_tok_s_bench']:.1f} | {r['gain_vs_own_bench']*100:+.0f}% | {bv} |")


if __name__ == "__main__":
    main()
