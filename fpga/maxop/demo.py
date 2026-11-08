"""maxop demo (`pixi run maxop-demo`): Llama 3.2 3B prefill with matmuls routed to the phantom NPU.

Runs the full MAX pipeline (custom architecture `phantom_llama`, Python driver
`run_model.py`) once per configuration, each in its own process:

  bf16     stock bf16 on the GPU (baseline; 8 prompts x 8 greedy tokens + the 512-token bench prompt)
  ph_l0    layer 0 MLP (gate_up, down) on the phantom (stub backend)
  ph_all   every q|k|v, o, gate|up, down projection of all 28 layers on the phantom (stub backend)
           + the 512-token bench prompt (its launch log gives the TTFT-shaped matmul list)
  fq_all   same W8A8 math as ph_all but the int8 matmul done in f32 on the GPU (fake quant)
  dev_l0   layer 0 MLP on lane G1-C's libphantom.dylib (cycle-accurate, N=16), 2 prompts x 3 tokens

then compares greedy tokens / prefill logits, projects TTFT from the launch log,
and writes ../results/maxop.json. Cached runs in out/ are reused unless --fresh.
GPU use: five model loads (~30 s compile each) and short generations; nothing is left running.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import maxop_lib as L  # noqa: E402
from phantom_llama.phantom_ops import parse_route  # noqa: E402

OUT = HERE / "out"
PROMPT_512 = L.REPO / "bench" / "prompt_512.txt"
GPU_TTFT_MS = 632.0  # MAX bf16 TTFT over HTTP, 512-token prompt (bench/, Sprint 1 baseline)

CONFIGS = {
    "bf16": ({"PHANTOM_MODE": "bf16"}, ["--prompt-file", str(PROMPT_512)]),
    "ph_l0": ({"PHANTOM_MODE": "phantom", "PHANTOM_ROUTE": "0:mlp"}, []),
    "ph_all": ({"PHANTOM_MODE": "phantom", "PHANTOM_ROUTE": "all:all"}, ["--prompt-file", str(PROMPT_512)]),
    "fq_all": ({"PHANTOM_MODE": "w8a8_gpu", "PHANTOM_ROUTE": "all:all"}, []),
    "dev_l0": ({"PHANTOM_MODE": "phantom", "PHANTOM_ROUTE": "0:mlp", "PHANTOM_BACKEND": "device",
                "PHANTOM_DEVICE_N": "16"}, ["--n-prompts", "2", "--new", "3"]),
}


def run(name: str, fresh: bool) -> dict:
    npz, log = OUT / f"{name}.npz", OUT / f"{name}.log"
    if fresh or not npz.exists():
        env, extra = CONFIGS[name]
        log.unlink(missing_ok=True)
        e = os.environ | env | {"PHANTOM_LOG": str(log)}
        print(f"=== {name}: {env}", flush=True)
        p = subprocess.run([sys.executable, str(HERE / "run_model.py"), str(npz), *extra], env=e,
                           cwd=HERE, capture_output=True, text=True)
        (OUT / f"{name}.stdout").write_text(p.stdout + p.stderr)
        if p.returncode != 0:
            print(p.stderr[-3000:])
            raise SystemExit(f"{name} failed")
        print("\n".join(ln for ln in p.stdout.splitlines() if ln.startswith(("prompt", "wrote"))))
    d = np.load(npz)
    meta = json.loads(str(d["meta"]))
    launches = []
    if log.exists():
        for ln in log.read_text().splitlines():
            v = list(map(int, ln.split()))
            launches.append(dict(zip(("site", "M", "K", "N_out", "c4", "c8", "c16", "dev_n", "measured"), v)))
    return {"meta": meta, "first_logits": d["first_logits"], "plens": d["prompt_lengths"].tolist(),
            "gen_s": d["gen_seconds"].tolist(), "launches": launches}


def compare(a: dict, b: dict, n: int | None = None) -> dict:
    """Greedy-token and prefill-logit agreement of run a vs reference b over the shared prompts."""
    ta, tb = a["meta"]["new_tokens"], b["meta"]["new_tokens"]
    n = n or min(len(ta), len(tb))
    full, tok_agree, tok_total, first_top1, cos, kl = 0, 0, 0, 0, [], []
    per = []
    for i in range(n):
        x, y = ta[i], tb[i][: len(ta[i])]
        k = len(x)
        agree = next((j for j in range(k) if x[j] != y[j]), k)  # matching prefix
        full += agree == k
        tok_agree += agree
        tok_total += k
        la, lb = a["first_logits"][i].astype(np.float64), b["first_logits"][i].astype(np.float64)
        first_top1 += int(la.argmax() == lb.argmax())
        cos.append(float(la @ lb / np.linalg.norm(la) / np.linalg.norm(lb)))
        pa, pb = np.exp(la - la.max()), np.exp(lb - lb.max())
        pa, pb = pa / pa.sum(), pb / pb.sum()
        kl.append(float(np.sum(pb * (np.log(pb + 1e-30) - np.log(pa + 1e-30)))))
        per.append({"prompt": i, "prompt_tokens": a["plens"][i], "matching_prefix": agree, "of": k})
    return {"prompts": n, "greedy_full_match": f"{full}/{n}", "greedy_prefix_tokens": f"{tok_agree}/{tok_total}",
            "prefill_top1_match": f"{first_top1}/{n}", "prefill_logit_cosine_min": min(cos),
            "prefill_logit_cosine_mean": float(np.mean(cos)), "prefill_kl_mean": float(np.mean(kl)), "per_prompt": per}


def prefill_macs(m: int) -> dict:
    """MACs of a Llama 3.2 3B prefill of m tokens: routable projections, attention, lm_head (last token)."""
    proj = sum(m * k * n for k, n in L.PROJ_SHAPES.values()) * L.LAYERS
    head_dim, heads = 128, 24
    attn = L.LAYERS * heads * head_dim * m * (m + 1)  # QK^T + PV, causal (~half of 2*m^2 each)
    lm_head = L.HIDDEN * 128256
    return {"projections": proj, "attention": attn, "lm_head": lm_head}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    R = {n: run(n, a.fresh) for n in CONFIGS}

    bf16, ph_l0, ph_all, fq_all, dev = (R[k] for k in ("bf16", "ph_l0", "ph_all", "fq_all", "dev_l0"))
    n_short = 8  # the short prompts; the 512-token prompt is compared separately below
    quality = {
        "ph_l0_vs_bf16": compare(ph_l0, bf16, n_short),
        "ph_all_vs_bf16": compare(ph_all, bf16, n_short),
        "fq_all_vs_bf16": compare(fq_all, bf16, n_short),
        "ph_all_vs_fq_all": compare(ph_all, fq_all, n_short),
        "dev_l0_vs_ph_l0": compare(dev, ph_l0, 2),
    }
    # 512-token prompt (last row of bf16 / ph_all): prefill logits only.
    i512 = len(bf16["plens"]) - 1
    la, lb = ph_all["first_logits"][i512].astype(np.float64), bf16["first_logits"][i512].astype(np.float64)
    quality["ph_all_vs_bf16_512tok"] = {
        "prompt_tokens": bf16["plens"][i512],
        "next_token_match": bool(la.argmax() == lb.argmax()),
        "logit_cosine": float(la @ lb / np.linalg.norm(la) / np.linalg.norm(lb)),
    }

    # Routed matmuls of the 512-token prefill, from the adapter's launch log.
    m512 = ph_all["plens"][i512]
    pre = [r for r in ph_all["launches"] if r["M"] == m512]
    assert len(pre) == 4 * L.LAYERS, len(pre)
    shapes = {}
    for r in pre:
        key = (r["M"], r["K"], r["N_out"])
        s = shapes.setdefault(key, {"M": r["M"], "K": r["K"], "N_out": r["N_out"], "count": 0,
                                    "cycles_per_call": {"N4": r["c4"], "N8": r["c8"], "N16": r["c16"]}})
        s["count"] += 1
    for s in shapes.values():
        for nk, c in s["cycles_per_call"].items():
            assert c == L.tile_cycles(s["M"], s["K"], s["N_out"], int(nk[1:]))
    sim_cycles = {f"N{n}": sum(r[f"c{n}"] for r in pre) for n in (4, 8, 16)}

    macs = prefill_macs(m512)
    total = sum(macs.values())
    bf16_prefill_ms = bf16["gen_s"][i512] * 1e3  # in-process prefill + 1 sample, 512 tokens
    ph_prefill_ms = ph_all["gen_s"][i512] * 1e3
    routes = {}
    for name, route in (("all:all", parse_route("all:all", L.LAYERS)), ("0:mlp", parse_route("0:mlp", L.LAYERS))):
        mm = L.route_matmuls(route, m512)
        share = sum(r["M"] * r["K"] * r["N_out"] for r in mm) / total
        routes[name] = {
            "routed_matmuls": len(mm),
            "routed_mac_share_of_prefill": share,
            "gpu_ms_removed_estimate": share * bf16_prefill_ms,
            "projection": L.project(mm, GPU_TTFT_MS, share * bf16_prefill_ms),
        }
    assert routes["all:all"]["projection"]["N16"]["cycles"] == sim_cycles["N16"]

    res = {
        "status": "full-pipeline",
        "authoritative": False,
        "date": dt.date.today().isoformat(),
        "model": "Llama-3.2-3B-Instruct bf16 (MAX 26.8.0.dev2026110805, Apple M5 GPU)",
        "integration": "MAX custom op phantom_matmul_i8 (Mojo, host CPU) inside the stock LlamaForCausalLM_ModuleV3 "
                       "pipeline via --custom-architectures fpga/maxop/phantom_llama (class-swapped projections)",
        "route_config": {"demo_routes": {k: v[0] for k, v in CONFIGS.items()},
                         "quantization": "W8A8 symmetric: per-token int8 activations, per-output-channel int8 weights, "
                                         "int32 accumulate, dequant to bf16"},
        "greedy_match": {k: {kk: v[kk] for kk in ("prompts", "greedy_full_match", "greedy_prefix_tokens", "prefill_top1_match")}
                         for k, v in quality.items() if "per_prompt" in v},
        "w8a8_quality": quality,
        "tests": json.loads((HERE / "tests" / "last_test_results.json").read_text())
                 if (HERE / "tests" / "last_test_results.json").exists() else None,
        "device_backend_run": {"launches": len(dev["launches"]),
                               "all_measured_eq_formula": all(r["measured"] == L.tile_cycles(r["M"], r["K"], r["N_out"], r["dev_n"]) for r in dev["launches"]),
                               "tile_n": 16, "sim_cycles_total": sum(r["measured"] for r in dev["launches"]),
                               "wall_s": sum(dev["gen_s"])},
        "routed_matmuls": {"prompt_tokens": m512, "calls": len(pre), "shapes": list(shapes.values())},
        "sim_cycles": sim_cycles,
        "fmax_mhz": L.FMAX_MHZ,
        "io_bw_bytes_per_s": L.IO_BW,
        "projected_ttft_ms": {k: {nk: v["ttft_ms"] for nk, v in r["projection"].items()} for k, r in routes.items()},
        "projection_detail": routes,
        "gpu_ttft_ms": GPU_TTFT_MS,
        "measured_in_process_ms": {"bf16_prefill_512": bf16_prefill_ms, "phantom_stub_all_prefill_512": ph_prefill_ms},
        "prefill_macs": macs,
        "notes": [
            "Projected TTFT = 632 ms (MAX bf16, HTTP, 512 tok) - (routed MAC share x in-process bf16 prefill time) + tile time, "
            "tile time = max(cycles/fmax, link bytes/link BW) summed serially over the routed matmuls.",
            "io 'streamed' = v1 tile as specified (no operand buffers: 2N int8 per beat per pass + int32 results); "
            "'minimal' = each operand crosses the link once (would need on-board A/B buffers).",
            "Cycles are the DESIGN.md formula with an ideal host; tests/spot_check.mojo and the dev_l0 run (libphantom, "
            "cycle-accurate) show the stepped model equals the formula on real routed tiles.",
            "phantom 'stub' backend computes the int32 result functionally on the host CPU (bit-exact, ~90 GMAC/s); "
            "'device' backend calls lane G1-C's libphantom.dylib C ABI and steps the register model per launch.",
            "Weights are re-quantized in-graph on every forward (no offline int8 checkpoint); fine for verification, not for speed.",
        ],
    }
    out = L.REPO / "fpga" / "results" / "maxop.json"
    out.write_text(json.dumps(res, indent=1))

    print("\n=== quality (vs bf16 GPU) ===")
    for k, v in quality.items():
        if "per_prompt" in v:
            print(f"{k:18s} greedy full {v['greedy_full_match']}, prefix tokens {v['greedy_prefix_tokens']}, "
                  f"prefill top1 {v['prefill_top1_match']}, cos min {v['prefill_logit_cosine_min']:.5f}, KL {v['prefill_kl_mean']:.2e}")
    print("512-tok:", quality["ph_all_vs_bf16_512tok"])
    print(f"\n=== 512-token prefill, all projections routed: {len(pre)} matmuls ===")
    for s in shapes.values():
        print(f"  {s['count']:3d} x M={s['M']} K={s['K']} N_out={s['N_out']}: cycles/call {s['cycles_per_call']}")
    for k, r in routes.items():
        print(f"route {k}: {r['routed_matmuls']} matmuls, MAC share {r['routed_mac_share_of_prefill']:.3f}")
        for nk, p in r["projection"].items():
            t = p["ttft_ms"]
            print(f"  {nk} @ {p['fmax_mhz']} MHz: {p['cycles']:.3e} cyc -> compute-only TTFT {t['compute_only'] / 1e3:9.1f} s | "
                  f"usb2 {t['usb2_minimal'] / 1e3:8.1f} s | usb3 {t['usb3_minimal'] / 1e3:8.1f} s | pcie {t['pcie_minimal'] / 1e3:8.1f} s (minimal I/O); "
                  f"streamed usb3 {t['usb3_streamed'] / 1e3:9.1f} s")
    print(f"GPU TTFT (bf16, HTTP): {GPU_TTFT_MS} ms; in-process bf16 512-tok prefill {bf16_prefill_ms:.0f} ms; "
          f"phantom-stub (host CPU functional) {ph_prefill_ms:.0f} ms")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
