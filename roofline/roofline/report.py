"""Render ROOFLINE.md tables from results/model.json.

    uv run --project /Users/sdawka/Code/llming/roofline python -m roofline.report

Writes results/tables.md (generated tables); ROOFLINE.md embeds the same tables between
<!-- BEGIN GENERATED --> / <!-- END GENERATED --> markers and is refreshed in place.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def pct(x):
    return f"{x * 100:+.1f}%"


def flag(x, lim=0.20):
    return "pass" if abs(x) <= lim else "**FAIL**"


def tables(d) -> str:
    L = []
    P = d["fitted_params"]["all_rows"]
    L.append("### Fitted parameters\n")
    L.append("| fit | eta_d | c_d (ms/token, 28 layers) | eta_p | c_http (ms) |")
    L.append("|---|---:|---:|---:|---:|")
    L.append(f"| all 7 rows | {P['eta_d']:.3f} | {P['c_d_ms']:.2f} | {P['eta_p']:.3f} | {P['c_http_ms']:.1f} |")
    for g, p in d["fitted_params"]["leave_group_out"].items():
        L.append(f"| without {g} | {p['eta_d']:.3f} | {p['c_d_ms']:.2f} | {p['eta_p']:.3f} | {p['c_http_ms']:.1f} |")
    p = d["fitted_params"]["fit_16bit_only"]
    L.append(f"| 16-bit rows only (c_d fixed 0) | {p['eta_d']:.3f} | 0 | {p['eta_p']:.3f} | {p['c_http_ms']:.1f} |")
    L.append("")
    L.append("### Validation: decode tok/s (Llama 3.2 3B, phase-1)\n")
    L.append("| row | measured | raw roofline | err | fit (all) | err | **leave-group-out** | **err** | 16-bit-only fit | err | check (LOGO, 20%) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for v in d["validation"]:
        m = v["measured"]["tok_s"]
        L.append(f"| {v['id']} | {m:.1f} | {v['raw_roofline']['tok_s']:.1f} | {pct(v['raw_roofline']['err_tok_s'])} | "
                 f"{v['fit_all']['tok_s']:.1f} | {pct(v['fit_all']['err_tok_s'])} | {v['leave_group_out']['tok_s']:.1f} | "
                 f"{pct(v['leave_group_out']['err_tok_s'])} | {v['fit_16bit_only']['tok_s']:.1f} | {pct(v['fit_16bit_only']['err_tok_s'])} | "
                 f"{flag(v['leave_group_out']['err_tok_s'])} |")
    L.append("")
    L.append("### Validation: TTFT ms\n")
    L.append("| row | method | measured | raw roofline | err | fit (all) | err | **leave-group-out** | **err** | check (LOGO, 20%) |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for v in d["validation"]:
        m = v["measured"]["ttft_ms"]
        L.append(f"| {v['id']} | {v['method']} | {m:.0f} | {v['raw_roofline']['ttft_ms']:.0f} | {pct(v['raw_roofline']['err_ttft'])} | "
                 f"{v['fit_all']['ttft_ms']:.0f} | {pct(v['fit_all']['err_ttft'])} | {v['leave_group_out']['ttft_ms']:.0f} | "
                 f"{pct(v['leave_group_out']['err_ttft'])} | {flag(v['leave_group_out']['err_ttft'])} |")
    L.append("")
    L.append("### Runtime calibration factors (measured / fitted prediction; used only to anchor candidate rankings)\n")
    L.append("| runtime | decode tok/s factor | TTFT factor | rows |")
    L.append("|---|---:|---:|---|")
    for k, c in d["runtime_calibration"].items():
        L.append(f"| {k} | {c['decode_tok_s']:.3f} | {c['ttft']:.3f} | {', '.join(c['rows']) or 'none (factor 1)'} |")
    L.append("")
    L.append("### Per-model accounting (text-only; decode GB = weight bytes read per token incl. lm_head)\n")
    L.append("| model | params (B) | embedding (B) | layers | d / ffn | heads q/kv x hd | vocab | bf16 GB | 8-bit g64 GB | Q4_K_M GB | MLX 4-bit GB | KV KiB/token | KV @8k ctx (MB) | prefill 512 GFLOP (linear / attn) |")
    L.append("|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for m, s in d["models"].items():
        a, p, b = s["arch"], s["params"], s["decode_weight_bytes_per_token"]
        pf = s["prefill_flops"]["512"]
        win = f" (SWA {a['window']}, 1 global per {a['global_every']})" if a["window"] else ""
        L.append(f"| {m} | {p['total_unique'] / 1e9:.2f} | {p['embedding'] / 1e9:.2f}{' tied' if a['tied'] else ''} | {a['L']} | "
                 f"{a['d']} / {a['f']} | {a['n_h']}/{a['n_kv']} x {a['hd']}{win} | {a['V']} | {b['bf16'] / 1e9:.2f} | "
                 f"{b['mlx_8bit_g64'] / 1e9:.2f} | {b['q4_k_m'] / 1e9:.2f} | {b['mlx_4bit_g64'] / 1e9:.2f} | "
                 f"{s['kv_bytes_per_token_fp16'] / 1024:.0f} | {s['kv_bytes_at_ctx_fp16']['8192'] / 1e6:.0f} | "
                 f"{pf['linear'] / 1e9:.0f} / {pf['attention'] / 1e9:.0f} |")
    L.append("")
    for m, c in d["candidates"].items():
        ref = c["reference"]
        L.append(f"### {m}: candidates\n")
        L.append(f"Reference: decode {ref['decode_tok_s']:.1f} tok/s, in-process TTFT {ref['ttft_inproc_ms']:.0f} ms ({ref['kind']}). "
                 f"15% target: decode >= {ref['decode_tok_s'] * 1.15:.1f} tok/s, TTFT <= {ref['ttft_inproc_ms'] / 1.15:.0f} ms.\n")
        L.append("| decode config | weights GB | pred tok/s (raw) | calibrated | gain | >=15%? | ppl risk (prior) | support / proxy |")
        L.append("|---|---:|---:|---:|---:|---|---|---|")
        for x in c["decode"]:
            note = x["supported"] if x["supported"] != "yes" else ""
            if x["proxy"]:
                note = (note + "; " if note else "") + "proxy: " + "; ".join(x["proxy"])
            L.append(f"| {x['config']} | {x['weight_gb']:.2f} | {x['tok_s']:.1f} | {x['tok_s_cal']:.1f} | {pct(x['gain_vs_baseline'])} | "
                     f"{'yes' if x['reaches_15pct'] else 'no'} | {x['ppl_risk'][0]}: {x['ppl_risk'][1]} | {note} |")
        L.append("")
        L.append("| TTFT config (512-token prompt) | pred ms (raw) | calibrated in-process | calibrated HTTP | gain | >=15%? | ppl risk (prior) | support / proxy |")
        L.append("|---|---:|---:|---:|---:|---|---|---|")
        for x in c["prefill"]:
            note = x["supported"] if x["supported"] != "yes" else ""
            if x["proxy"]:
                note = (note + "; " if note else "") + "proxy: " + "; ".join(x["proxy"])
            http = f"{x['ttft_http_cal_ms']:.0f}" if x["ttft_http_cal_ms"] else "-"
            L.append(f"| {x['config']} | {x['ttft_ms']:.0f} | {x['ttft_cal_ms']:.0f} | {http} | {pct(x['gain_vs_baseline'])} | "
                     f"{'yes' if x['reaches_15pct'] else 'no'} | {x['ppl_risk'][0]} | {note} |")
        L.append("")
        if c["speculative"]:
            L.append(f"Speculative decoding (draft {c['draft']} MLX 4-bit; best k per alpha; calibrated tok/s and gain):\n")
            alphas = sorted({r["alpha"] for r in c["speculative"]})
            L.append("| target | " + " | ".join(f"alpha={a}" for a in alphas) + " |")
            L.append("|---|" + "---:|" * len(alphas))
            for t in ("mlx/4bit", "mlx/8bit", "mlx/bf16"):
                cells = []
                for a in alphas:
                    b = max((r for r in c["speculative"] if r["target"] == t and r["alpha"] == a), key=lambda r: r["tok_s_cal"])
                    cells.append(f"{b['tok_s_cal']:.1f} (k={b['k']}, {pct(b['gain_vs_baseline'])})")
                L.append(f"| {t} | " + " | ".join(cells) + " |")
            r0 = c["speculative"][0]
            L.append(f"\nDraft step {r0['t_draft_ms']:.1f} ms; target step (M=1) for reference is in the decode table.\n")
    return "\n".join(L)


def main():
    d = json.loads((RES / "model.json").read_text())
    t = tables(d)
    (RES / "tables.md").write_text(t + "\n")
    md = ROOT / "ROOFLINE.md"
    if md.exists():
        s = md.read_text()
        a, b = "<!-- BEGIN GENERATED -->", "<!-- END GENERATED -->"
        if a in s and b in s:
            s = s[: s.index(a) + len(a)] + "\n" + t + "\n" + s[s.index(b):]
            md.write_text(s)


if __name__ == "__main__":
    main()
