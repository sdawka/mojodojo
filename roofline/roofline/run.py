"""Fit, validate and rank. Writes results/model.json and results/*.png.

    uv run --project /Users/sdawka/Code/llming/roofline python -m roofline.run
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .accounting import FORMATS, MODELS, ModelSpec, summary
from .atlas import Atlas
from .predict import ATTN_FRAC, SPEC_ITER_OVERHEAD_MS, RUNTIMES, Params, Predictor, parts, wls_rel

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "roofline" / "results"
BENCH = ROOT / "bench" / "results"

# --------------------------------------------------------------------------------------------
# Phase-1 measurements (Llama 3.2 3B). Values are read from the bench JSON files.
BENCH_FILES = {
    "llamacpp_f16": "llamacpp_f16_20261108_063515.json",
    "llamacpp_q4": "llamacpp_q4_k_m_20261108_063515.json",
    "server_f16": "llamacpp-server_f16_20261108_063441.json",
    "max_ep": "max_bf16_20261108_063039.json",
    "maxbench": "maxbench_bf16.json",
    "mlx": "mlx_20261108_062942.json",
}


def load_phase1():
    j = {k: json.loads((BENCH / f).read_text()) for k, f in BENCH_FILES.items()}

    def lb(d):
        pp = next(t for t in d if t["n_gen"] == 0)
        tg = next(t for t in d if t["n_prompt"] == 0)
        return pp["avg_ts"], tg["avg_ts"]

    rows = []
    pp, tg = lb(j["llamacpp_f16"])
    rows.append(dict(id="llama-bench F16", group="llama.cpp F16", rt="llama.cpp/F16", tok_s=tg, ttft_ms=512 / pp * 1e3,
                     S=512, ctx=64, method="llama-bench: TTFT=512/pp512 (one 512-token ubatch, logits for last token); tg128 on empty KV",
                     http=False, extra_steps=0, lm_head_tokens=1, src=BENCH_FILES["llamacpp_f16"]))
    pp, tg = lb(j["llamacpp_q4"])
    rows.append(dict(id="llama-bench Q4_K_M", group="llama.cpp Q4_K_M", rt="llama.cpp/Q4_K_M", tok_s=tg, ttft_ms=512 / pp * 1e3,
                     S=512, ctx=64, method="llama-bench (as above)", http=False, extra_steps=0, lm_head_tokens=1,
                     src=BENCH_FILES["llamacpp_q4"]))
    r = j["server_f16"]["results"][0]
    rows.append(dict(id="llama-server F16 (HTTP)", group="llama.cpp F16", rt="llama.cpp/F16", tok_s=r["tg_tps_mean"],
                     ttft_ms=r["ttft_s_mean"] * 1e3, S=512, ctx=513 + 64,
                     method="HTTP client TTFT; 513 tokens (nonce) > n_ubatch 512 -> second 1-token ubatch = one extra full weight pass",
                     http=True, extra_steps=1, lm_head_tokens=1, src=BENCH_FILES["server_f16"]))
    r = j["max_ep"]["results"][0]
    rows.append(dict(id="MAX bf16 (HTTP)", group="MAX bf16", rt="max/bf16", tok_s=r["tg_tps_mean"], ttft_ms=r["ttft_s_mean"] * 1e3,
                     S=513, ctx=513 + 64, method="HTTP client TTFT, single prefill pass (ReturnLogits.LAST_TOKEN)",
                     http=True, extra_steps=0, lm_head_tokens=1, src=BENCH_FILES["max_ep"]))
    m = j["maxbench"]
    S_mb = round(m["total_input_tokens"] / m["num_prompts"])
    rows.append(dict(id="max benchmark bf16 (HTTP)", group="MAX bf16", rt="max/bf16", tok_s=1e3 / m["mean_tpot_ms"],
                     ttft_ms=m["mean_ttft_ms"], S=S_mb, ctx=S_mb + 64,
                     method=f"max benchmark random dataset, mean input {S_mb} tok, {m['global_cached_token_rate']*100:.0f}% prefix-cache hits",
                     http=True, extra_steps=0, lm_head_tokens=1, src=BENCH_FILES["maxbench"]))
    for res in j["mlx"]["results"]:
        fmt = res["format"]
        rows.append(dict(id=f"MLX {fmt}", group=f"MLX {fmt}", rt="mlx/bf16" if fmt == "bf16" else "mlx/4bit",
                         tok_s=res["tg_tps_mean"], ttft_ms=res["ttft_s_mean"] * 1e3, S=511, ctx=512 + 64,
                         method="in-process; mlx_lm prefill of 511 tokens (no lm_head) + one M=1 step for the last prompt token",
                         http=False, extra_steps=1, lm_head_tokens=0, src=BENCH_FILES["mlx"]))
    return rows


# --------------------------------------------------------------------------------------------
def fit(pred: Predictor, spec, rows, decode_intercept: bool = True) -> Params:
    # decode: t = a * ideal + c   (a = 1/eta_d)
    X = np.array([[sum(pred.decode_ideal(spec, RUNTIMES[r["rt"]], r["ctx"]).values()) * 1e3, 1.0] for r in rows])
    y = np.array([1e3 / r["tok_s"] for r in rows])
    if decode_intercept:
        a, c = wls_rel(X, y)
    else:  # single parameter (eta_d only, c_d = 0)
        a, c = float(wls_rel(X[:, :1], y)[0]), 0.0
    P = Params(eta_d=1 / a, c_d_ms=c)
    # prefill: TTFT - extra decode steps = b * ideal_pf + c_http * http
    Xp, yp = [], []
    for r in rows:
        rt = RUNTIMES[r["rt"]]
        ideal = sum(pred.prefill_ideal(spec, rt, r["S"], r["lm_head_tokens"]).values()) * 1e3
        extra = r["extra_steps"] * pred.decode_ms(spec, rt, r["S"], P)
        Xp.append([ideal, 1.0 if r["http"] else 0.0])
        yp.append(r["ttft_ms"] - extra)
    Xp, yp = np.array(Xp), np.array(yp)
    if Xp[:, 1].sum() == 0:
        b = float(np.sum(Xp[:, 0] * yp / yp ** 2) / np.sum(Xp[:, 0] ** 2 / yp ** 2))
        ch = 0.0
    else:
        b, ch = wls_rel(Xp, yp)
    P.eta_p, P.c_http_ms = 1 / b, ch
    return P


def predict_row(pred, spec, r, P: Params):
    rt = RUNTIMES[r["rt"]]
    t_dec = pred.decode_ms(spec, rt, r["ctx"], P)
    ttft = (pred.prefill_ms(spec, rt, r["S"], P, r["lm_head_tokens"]) + r["extra_steps"] * pred.decode_ms(spec, rt, r["S"], P)
            + (P.c_http_ms if r["http"] else 0.0))
    return 1e3 / t_dec, ttft


def validate(pred, spec, rows):
    P_all = fit(pred, spec, rows)
    P_raw = Params()
    groups = sorted({r["group"] for r in rows})
    out = []
    logo_params = {}
    for g in groups:
        logo_params[g] = fit(pred, spec, [r for r in rows if r["group"] != g])
    # cross-format holdout: fit on 16-bit rows, predict 4-bit rows (and the reverse for decode)
    # (16-bit rows all have the same ideal time, so only eta_d is identifiable: c_d fixed to 0)
    p16 = fit(pred, spec, [r for r in rows if "4" not in r["rt"].split("/")[1]], decode_intercept=False)
    for r in rows:
        raw = predict_row(pred, spec, r, P_raw)
        allp = predict_row(pred, spec, r, P_all)
        lo = predict_row(pred, spec, r, logo_params[r["group"]])
        xf = predict_row(pred, spec, r, p16)
        rec = {k: r[k] for k in ("id", "group", "rt", "method", "src", "S", "ctx")}
        rec.update(measured={"tok_s": r["tok_s"], "ttft_ms": r["ttft_ms"]})
        for name, (ts, tt) in (("raw_roofline", raw), ("fit_all", allp), ("leave_group_out", lo), ("fit_16bit_only", xf)):
            rec[name] = {"tok_s": ts, "ttft_ms": tt, "err_tok_s": ts / r["tok_s"] - 1, "err_ttft": tt / r["ttft_ms"] - 1}
        rt = RUNTIMES[r["rt"]]
        rec["decode_breakdown_ms"] = {k: v * 1e3 for k, v in pred.decode_ideal(spec, rt, r["ctx"]).items()}
        rec["prefill_breakdown_ms"] = {k: v * 1e3 for k, v in pred.prefill_ideal(spec, rt, r["S"], r["lm_head_tokens"]).items()}
        out.append(rec)
    return P_all, logo_params, p16, out


# --------------------------------------------------------------------------------------------
PPL_RISK = {  # prior expectations for 1-4B models (to be measured in G2-M3); delta PPL vs bf16
    "bf16": ("none", "0%"), "f16": ("none", "~0%"),
    "mlx_8bit_g64": ("low", "~+0.1-0.3%"), "q8_0": ("low", "~+0.1-0.3%"),
    "fp8_w8a16": ("low-med", "~+0.2-1% (per-tensor scale only on the Apple path)"),
    "q6_k": ("low-med", "~+0.3-0.8%"),
    "int8_w8a8": ("medium", "~+0.5-2% (dynamic per-token int8 activations; Llama-3 activation outliers)"),
    "q4_k_m": ("high", "~+2-5% (small models are more sensitive)"),
    "mlx_4bit_g64": ("high", "~+3-8%"), "mlx_4bit_g128": ("high", "~+4-10%"),
    "nvfp4_w4a16": ("high", "~+3-6% (PTQ, no calibration)"), "q4_0": ("high", "~+4-10%"),
}

CANDIDATE_MODELS = ["llama3.2-3b", "llama3.2-1b", "qwen3-4b", "qwen3-1.7b", "gemma3-4b", "gemma3-1b"]
DRAFTS = {"llama3.2-3b": "llama3.2-1b", "qwen3-4b": "qwen3-0.6b", "gemma3-4b": "gemma3-1b", "qwen3-1.7b": "qwen3-0.6b"}
MAX_ARCH = {"llama": {"bf16", "fp8_w8a16", "nvfp4_w4a16", "int8_w8a8"}, "qwen3": {"bf16", "fp8_w8a16", "int8_w8a8"},
            "gemma3": {"bf16", "int8_w8a8"}}
DECODE_RTS = ["mlx/bf16", "mlx/8bit", "mlx/4bit", "mlx/4bit-g128", "llama.cpp/F16", "llama.cpp/Q8_0", "llama.cpp/Q6_K",
              "llama.cpp/Q4_K_M", "llama.cpp/Q4_0", "max/bf16", "max/fp8", "max/nvfp4", "max/int8-w8a8"]
PREFILL_RTS = DECODE_RTS + ["ane/w8a8", "ane/w8", "ane/fp16", "ane/lut4"]
BASE_DECODE_TOK_S = 48.8   # best phase-1 decode (llama.cpp Q4_K_M and MLX 4-bit, Llama 3.2 3B)
BASE_TTFT_INPROC = 329.0   # MLX 4-bit, in-process
BASE_TTFT_HTTP = 444.0     # llama-server F16 over HTTP


def max_ok(model, fmt):
    fam = "llama" if model.startswith("llama") else model.split("-")[0]
    return fmt in MAX_ARCH.get(fam, set())


def runtime_calibration(val):
    """Per-runtime residual factors from the all-rows fit: measured / predicted (geometric mean).
    Used only to *anchor* candidate predictions to the runtime's measured behaviour; never in validation."""
    cal = {}
    for rtname in ("mlx", "llama.cpp", "max"):
        rs = [v for v in val if v["rt"].startswith(rtname + "/")]
        cal[rtname] = {
            "decode_tok_s": float(np.exp(np.mean([np.log(v["measured"]["tok_s"] / v["fit_all"]["tok_s"]) for v in rs]))),
            "ttft": float(np.exp(np.mean([np.log(v["measured"]["ttft_ms"] / v["fit_all"]["ttft_ms"]) for v in rs]))),
            "rows": [v["id"] for v in rs]}
    cal["coreml-ane"] = {"decode_tok_s": 1.0, "ttft": 1.0, "rows": []}
    return cal


def candidates(pred: Predictor, P: Params, cal, ctx_decode=576, S=512):
    """Rank configs per model. tok_s/ttft are raw model predictions; *_cal are anchored by the
    runtime calibration factor. Gains use calibrated values: for Llama 3.2 3B against the measured
    phase-1 bests; for other models against the calibrated prediction of the same phase-1-style
    configs (MLX 4-bit / llama.cpp Q4_K_M), which were not measured."""
    res = {}
    for m in CANDIDATE_MODELS:
        spec = ModelSpec.load(m)
        dec, pre = [], []
        for k in DECODE_RTS:
            rt = RUNTIMES[k]
            if rt.runtime == "max" and not max_ok(m, rt.fmt):
                continue
            t = pred.decode_ms(spec, rt, ctx_decode, P)
            dec.append(dict(config=k, fmt=rt.fmt, tok_s=1e3 / t, tok_s_cal=1e3 / t * cal[rt.runtime]["decode_tok_s"],
                            ppl_risk=PPL_RISK[rt.fmt], weight_gb=sum(b for _, _, b, _ in parts(spec, rt.fmt)) / 1e9,
                            proxy=rt.proxy, supported=rt.supported))
        for k in PREFILL_RTS:
            rt = RUNTIMES[k]
            if rt.runtime == "max" and not max_ok(m, rt.fmt):
                continue
            pf = pred.prefill_ms(spec, rt, S, P, lm_head_tokens=1 if rt.runtime != "coreml-ane" else 0)
            if rt.runtime == "coreml-ane":  # last-token lm_head on GPU (bf16 MLX GEMV) + ~1 ms Core ML call/handoff (assumed)
                pf += pred.decode_ideal(spec, RUNTIMES["mlx/bf16"], 0)["lm_head"] * 1e3 / P.eta_d + 1.0
            c = cal[rt.runtime]["ttft"]
            pre.append(dict(config=k + " (one pass, last-token logits)", fmt=rt.fmt, ttft_ms=pf, ttft_cal_ms=pf * c,
                            ttft_http_cal_ms=pf * c + P.c_http_ms, ppl_risk=PPL_RISK[rt.fmt], proxy=rt.proxy,
                            supported=rt.supported))
        for k in ("mlx/4bit", "mlx/bf16", "mlx/8bit"):  # MLX as benchmarked (extra M=1 step)
            rt = RUNTIMES[k]
            pf = pred.prefill_ms(spec, rt, S - 1, P, 0) + pred.decode_ms(spec, rt, S, P)
            c = cal["mlx"]["ttft"]
            pre.append(dict(config=k + " (mlx_lm as benchmarked: +1 decode step)", fmt=rt.fmt, ttft_ms=pf, ttft_cal_ms=pf * c,
                            ttft_http_cal_ms=None, ppl_risk=PPL_RISK[rt.fmt], proxy=rt.proxy, supported="yes"))
        if m == "llama3.2-3b":
            ref_dec, ref_ttft, kind = BASE_DECODE_TOK_S, BASE_TTFT_INPROC, "measured phase-1"
        else:
            ref_dec = max(d["tok_s_cal"] for d in dec if d["config"] in ("mlx/4bit", "llama.cpp/Q4_K_M"))
            ref_ttft = min(p["ttft_cal_ms"] for p in pre if p["config"].startswith("mlx/4bit (mlx_lm"))
            kind = "calibrated prediction of phase-1-style configs (MLX 4-bit / llama.cpp Q4_K_M), not measured"
        for d in dec:
            d["gain_vs_baseline"] = d["tok_s_cal"] / ref_dec - 1
            d["reaches_15pct"] = d["gain_vs_baseline"] >= 0.15
        for p in pre:
            p["gain_vs_baseline"] = ref_ttft / p["ttft_cal_ms"] - 1
            p["reaches_15pct"] = p["gain_vs_baseline"] >= 0.15
        spec_rows = []
        if m in DRAFTS:
            dspec = ModelSpec.load(DRAFTS[m])
            for tk in ("mlx/4bit", "mlx/8bit", "mlx/bf16"):
                dk = "mlx/4bit"
                for kdraft in (1, 2, 3, 4, 5, 7):
                    for alpha in (0.5, 0.6, 0.7, 0.8, 0.9):
                        r = pred.spec_decode(spec, RUNTIMES[tk], dspec, RUNTIMES[dk], kdraft, alpha, ctx_decode, P)
                        r["tok_s_cal"] = r["tok_s"] * cal["mlx"]["decode_tok_s"]
                        r.update(target=tk, draft=f"{DRAFTS[m]} {dk}", gain_vs_baseline=r["tok_s_cal"] / ref_dec - 1,
                                 ppl_risk=PPL_RISK[RUNTIMES[tk].fmt])
                        r["reaches_15pct"] = r["gain_vs_baseline"] >= 0.15
                        spec_rows.append(r)
        res[m] = dict(reference={"decode_tok_s": ref_dec, "ttft_inproc_ms": ref_ttft, "kind": kind},
                      draft=DRAFTS.get(m),
                      decode=sorted(dec, key=lambda d: -d["tok_s_cal"]), prefill=sorted(pre, key=lambda d: d["ttft_cal_ms"]),
                      speculative=spec_rows)
    return res



def measurement_requests(pred, P, cal):
    """Predicted values for the validation measurements we ask the orchestrator to run."""
    s3, s1 = ModelSpec.load("llama3.2-3b"), ModelSpec.load("llama3.2-1b")
    R = RUNTIMES
    dec = lambda spec, k, ctx: 1e3 / pred.decode_ms(spec, R[k], ctx, P) * cal[R[k].runtime]["decode_tok_s"]  # noqa: E731
    ttft = lambda spec, k, S, lm=1: pred.prefill_ms(spec, R[k], S, P, lm) * cal[R[k].runtime]["ttft"]  # noqa: E731
    mlx_ttft = lambda spec, k: (pred.prefill_ms(spec, R[k], 511, P, 0) + pred.decode_ms(spec, R[k], 512, P)) * cal["mlx"]["ttft"]  # noqa: E731
    return {
        "llama_bench_depth512": {"F16_tg_d0": dec(s3, "llama.cpp/F16", 64), "F16_tg_d512": dec(s3, "llama.cpp/F16", 576),
                                 "Q4_K_M_tg_d0": dec(s3, "llama.cpp/Q4_K_M", 64), "Q4_K_M_tg_d512": dec(s3, "llama.cpp/Q4_K_M", 576)},
        "llama_bench_Q8_0_Q4_0": {"Q8_0_tg": dec(s3, "llama.cpp/Q8_0", 64), "Q8_0_ttft_ms": ttft(s3, "llama.cpp/Q8_0", 512),
                                  "Q4_0_tg": dec(s3, "llama.cpp/Q4_0", 64), "Q4_0_ttft_ms": ttft(s3, "llama.cpp/Q4_0", 512),
                                  "Q6_K_tg": dec(s3, "llama.cpp/Q6_K", 64)},
        "llama_server_ub1024_ttft_ms": ttft(s3, "llama.cpp/F16", 513) + P.c_http_ms,
        "mlx_8bit_3b": {"tg": dec(s3, "mlx/8bit", 576), "ttft_ms": mlx_ttft(s3, "mlx/8bit")},
        "mlx_4bit_g128_3b": {"tg": dec(s3, "mlx/4bit-g128", 576)},
        "mlx_1b": {"4bit_tg": dec(s1, "mlx/4bit", 576), "4bit_ttft_ms": mlx_ttft(s1, "mlx/4bit"),
                   "bf16_tg": dec(s1, "mlx/bf16", 576), "bf16_ttft_ms": mlx_ttft(s1, "mlx/bf16")},
    }

# --------------------------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    atlas = Atlas()
    pred = Predictor(atlas)
    spec3b = ModelSpec.load("llama3.2-3b")
    rows = load_phase1()
    P, logo, p16, val = validate(pred, spec3b, rows)
    cal = runtime_calibration(val)
    cands = candidates(pred, P, cal)
    # prefill length sweep for 3B, a few configs
    sweep = {}
    for k in ("mlx/4bit", "mlx/bf16", "llama.cpp/F16", "max/bf16", "max/int8-w8a8", "ane/w8a8"):
        rt = RUNTIMES[k]
        sweep[k] = {S: pred.prefill_ms(spec3b, rt, S, P, 1 if rt.runtime != "coreml-ane" else 0) for S in (128, 256, 512, 1024, 2048)}
    reqs = measurement_requests(pred, P, cal)
    model = {
        "meta": {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "authoritative": False,
                 "milestone": "G2-M2", "machine": "Apple M5 32 GB",
                 "inputs": atlas.inputs + [str(BENCH / f) for f in BENCH_FILES.values()]
                 + [str(ROOT / "roofline" / "configs" / MODELS[m][0]) for m in MODELS],
                 "note": "Predictions only. Atlas rates are idle-machine medians; phase-1 bench files are the validation targets."},
        "assumptions": {"attention_rate_fraction_of_16bit_gemm": ATTN_FRAC, "spec_iter_overhead_ms": SPEC_ITER_OVERHEAD_MS, "stream_bw_gbps": pred.bw,
                        "decode_ctx_default": 576, "prefill_S_default": 512, "kv_dtype_bytes": 2,
                        "formats": FORMATS},
        "fitted_params": {"all_rows": asdict(P), "leave_group_out": {g: asdict(p) for g, p in logo.items()},
                          "fit_16bit_only": asdict(p16),
                          "how": "decode: relative-error WLS of measured ms/token = ideal/eta_d + c_d over the 7 phase-1 rows; "
                                 "prefill: WLS of (TTFT - method extra decode steps) = ideal_pf/eta_p + c_http*[HTTP]"},
        "validation": val,
        "runtime_calibration": cal,
        "models": {m: summary(ModelSpec.load(m)) for m in MODELS},
        "candidates": cands,
        "prefill_length_sweep_llama3.2-3b_ms": sweep,
        "measurement_requests": reqs,
    }
    (OUT / "model.json").write_text(json.dumps(model, indent=1, default=float))
    from .plots import make_plots
    make_plots(model, OUT)
    # console summary
    print("fitted (all):", asdict(P))
    for g, p in logo.items():
        print("  LOGO without", g, {k: round(v, 3) for k, v in asdict(p).items()})
    print("  16-bit-only", {k: round(v, 3) for k, v in asdict(p16).items()})
    hdr = f"{'row':30s} {'meas tok/s':>10s} {'raw':>7s} {'fit':>7s} {'LOGO':>7s} {'err':>7s} | {'meas TTFT':>9s} {'raw':>7s} {'fit':>7s} {'LOGO':>7s} {'err':>7s}"
    print(hdr)
    for v in val:
        print(f"{v['id']:30s} {v['measured']['tok_s']:10.1f} {v['raw_roofline']['tok_s']:7.1f} {v['fit_all']['tok_s']:7.1f} "
              f"{v['leave_group_out']['tok_s']:7.1f} {v['leave_group_out']['err_tok_s']*100:+6.1f}% | {v['measured']['ttft_ms']:9.0f} "
              f"{v['raw_roofline']['ttft_ms']:7.0f} {v['fit_all']['ttft_ms']:7.0f} {v['leave_group_out']['ttft_ms']:7.0f} "
              f"{v['leave_group_out']['err_ttft']*100:+6.1f}%")


if __name__ == "__main__":
    main()
