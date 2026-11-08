"""PNG plots for ROOFLINE.md. Palette: dataviz default categorical order (light surface)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def _fig(w, h, n=1):
    fig, axs = plt.subplots(1, n, figsize=(w, h), facecolor=SURF)
    axs = np.atleast_1d(axs)
    for a in axs:
        _style(a)
    return fig, axs


def validation(model, out):
    val = model["validation"]
    fig, (a1, a2) = _fig(10, 4.4, 2)
    groups = sorted({v["group"] for v in val})
    for ax, key, unit, title in ((a1, "tok_s", "tok/s", "Decode (leave-group-out)"), (a2, "ttft_ms", "ms", "TTFT (leave-group-out)")):
        xs = [v["measured"][key] for v in val]
        lo, hi = min(xs) * 0.7, max(xs) * 1.3
        xx = np.linspace(lo, hi, 50)
        ax.fill_between(xx, xx * 0.8, xx * 1.2, color=GRID, alpha=0.7, lw=0, label="±20%")
        ax.plot(xx, xx, color=INK2, lw=1)
        for v in val:
            gi = groups.index(v["group"])
            ax.scatter(v["measured"][key], v["leave_group_out"][key], s=46, color=C[gi], edgecolor=SURF, lw=1.5, zorder=3,
                       label=v["group"])
            ax.scatter(v["measured"][key], v["raw_roofline"][key], s=30, facecolor="none", edgecolor=C[gi], lw=1.2, zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"measured ({unit})", color=INK2); ax.set_ylabel(f"predicted ({unit})", color=INK2)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    h, l = a1.get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, l):
        seen.setdefault(ll, hh)
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=6, frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.text(0.01, 0.01, "filled: fitted model, group held out; hollow: raw roofline (no fitted overheads)", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "validation.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def breakdown(model, out):
    """Decode ms/token by component (ideal, before eta/c) for the phase-1 rows."""
    val = [v for v in model["validation"] if not v["id"].startswith(("max benchmark", "llama-server"))]
    comps = ["mlp_up_gate", "mlp_down", "attn_q_o", "attn_kv", "lm_head", "kv_read"]
    fig, (a1, a2) = _fig(10, 3.8, 2)
    y = np.arange(len(val))
    for ax, key, title in ((a1, "decode_breakdown_ms", "Decode: ideal ms/token by component"),
                           (a2, "prefill_breakdown_ms", "Prefill: ideal ms by component")):
        left = np.zeros(len(val))
        cs = comps if key.startswith("decode") else comps[:5] + ["attention"]
        for i, c in enumerate(cs):
            w = np.array([v[key].get(c, 0.0) for v in val])
            ax.barh(y, w, left=left, color=C[i], edgecolor=SURF, lw=2, height=0.6, label=c, zorder=2)
            left += w
        ax.set_yticks(y, [v["id"] for v in val], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("ms", color=INK2)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "breakdown.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def ranking(model, out):
    ms = list(model["candidates"])
    fig, axs = _fig(12, 2.2 * len(ms), 2)
    plt.close(fig)
    fig, axs = plt.subplots(len(ms), 2, figsize=(12, 2.3 * len(ms)), facecolor=SURF)
    risk_col = {"none": C[0], "low": C[2], "low-med": C[3], "medium": C[1], "high": C[7]}
    for i, m in enumerate(ms):
        c = model["candidates"][m]
        for j, (key, val_key, unit) in enumerate((("decode", "tok_s_cal", "tok/s"), ("prefill", "ttft_cal_ms", "TTFT ms"))):
            ax = axs[i, j]; _style(ax)
            rows = [r for r in c[key] if "as benchmarked" not in r["config"]][:9]
            vals = [r[val_key] for r in rows]
            yy = np.arange(len(rows))
            ax.barh(yy, vals, color=[risk_col[r["ppl_risk"][0]] for r in rows], height=0.65, edgecolor=SURF, lw=2, zorder=2)
            ref = c["reference"]["decode_tok_s" if key == "decode" else "ttft_inproc_ms"]
            tgt = ref * 1.15 if key == "decode" else ref / 1.15
            ax.axvline(ref, color=INK2, lw=1, ls="--", zorder=3)
            ax.axvline(tgt, color=INK, lw=1.2, zorder=3)
            ax.set_yticks(yy, [r["config"].replace(" (one pass, last-token logits)", "") for r in rows], fontsize=8)
            ax.invert_yaxis()
            ax.set_title(f"{m}: {key} ({unit}; dashed = baseline, solid = 15% target)", color=INK, fontsize=9.5, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in risk_col.values()]
    fig.legend(handles, [f"ppl risk: {k}" for k in risk_col], loc="lower center", ncol=5, frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "ranking.png", dpi=130, facecolor=SURF)
    plt.close(fig)


def speculative(model, out):
    ms = [m for m in model["candidates"] if model["candidates"][m]["speculative"]]
    fig, axs = _fig(4.2 * len(ms), 3.8, len(ms))
    for ax, m in zip(axs, ms):
        c = model["candidates"][m]
        sp = c["speculative"]
        for i, tgt in enumerate(("mlx/4bit", "mlx/8bit", "mlx/bf16")):
            alphas = sorted({r["alpha"] for r in sp})
            best = [max((r for r in sp if r["target"] == tgt and r["alpha"] == a), key=lambda r: r["tok_s_cal"]) for a in alphas]
            ax.plot(alphas, [b["tok_s_cal"] for b in best], color=C[i], lw=2, marker="o", ms=6, label=f"target {tgt}", zorder=3)
            for b in best:
                ax.annotate(f"k={b['k']}", (b["alpha"], b["tok_s_cal"]), fontsize=7, color=INK2, xytext=(3, 4), textcoords="offset points")
        ref = c["reference"]["decode_tok_s"]
        ax.axhline(ref, color=INK2, ls="--", lw=1)
        ax.axhline(ref * 1.15, color=INK, lw=1.2)
        ax.set_xlabel("acceptance rate α", color=INK2); ax.set_ylabel("tok/s (calibrated)", color=INK2)
        ax.set_title(f"{m} + {c['draft']} 4-bit draft", color=INK, fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out / "speculative.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def prefill_sweep(model, out):
    sw = model["prefill_length_sweep_llama3.2-3b_ms"]
    fig, (ax,) = _fig(6, 3.8)
    for i, (k, d) in enumerate(sw.items()):
        S = [int(s) for s in d]
        ax.plot(S, [512 * 0 + int(s) / (v / 1e3) for s, v in d.items()], color=C[i], lw=2, marker="o", ms=5, label=k)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("prompt tokens", color=INK2); ax.set_ylabel("predicted prefill tok/s", color=INK2)
    ax.set_title("Llama 3.2 3B prefill throughput vs prompt length", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out / "prefill_sweep.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def make_plots(model, out):
    validation(model, out)
    breakdown(model, out)
    ranking(model, out)
    speculative(model, out)
    prefill_sweep(model, out)
