"""Charts as HTML (bar charts) and inline SVG (roofline). No JS libraries; colours come from
CSS custom properties through `t-<tool>` classes so light/dark both work."""

from __future__ import annotations

import math

from .sources import Reg, esc, fmt_val

TOOLS = {  # toolchain key -> legend label. Fixed order = categorical slot order (validated).
    "mlx": "MLX",
    "max": "Mojo / MAX",
    "coreml": "Core ML",
    "accel": "Accelerate / C on CPU",
    "llama": "llama.cpp (ggml)",
    "npu": "Our int8 tile (sim / FPGA)",
}


def legend(keys) -> str:
    items = "".join(f'<li><span class="sw t-{k}"></span>{esc(TOOLS[k])}</li>' for k in TOOLS if k in keys)
    return f'<ul class="legend" aria-label="Colour key">{items}</ul>'


def hbars(reg: Reg, rows: list[dict], unit: str, refs=(), nd=None, max_v=None, aria="") -> str:
    """Horizontal bars, one per row. rows: {label, sub, value, src, tool, detail, hollow}.
    refs: [(value, label)] vertical reference lines (e.g. Apple spec bandwidth)."""
    vals = [r["value"] for r in rows if r.get("value") is not None]
    if not rows:
        return '<p class="empty">No data in the result files yet.</p>'
    top = max_v or max(vals + [v for v, _ in refs] + [1e-9]) * 1.0
    span = top / 0.84  # leave room for the value label after the longest bar

    def pct(v):
        return max(0.0, min(100.0, 100.0 * v / span))

    refs = sorted(refs, reverse=True)
    out = [f'<div class="hb{" has-refs" if refs else ""}" role="img" aria-label="{esc(aria)}"'
           f' style="--nrefs:{len(refs)}">']
    for i, r in enumerate(rows):
        v = r.get("value")
        cls = f't-{r.get("tool", "other")}' + (" hollow" if r.get("hollow") else "")
        sub = f'<small>{esc(r["sub"])}</small>' if r.get("sub") else ""
        bar = (f'<span class="hb-bar {cls}" style="width:{pct(v):.2f}%"></span>' if v is not None
               else '<span class="hb-none">no data</span>')
        num = reg.num(v, r.get("src"), nd=nd, unit=unit, detail=r.get("detail", "")) if v is not None else ""
        lines = ""
        for j, (rv, lab) in enumerate(refs):
            side = " lft" if pct(rv) > 55 else ""
            em = f'<em style="--i:{j}">{esc(lab)}</em>' if i == 0 else ""
            lines += f'<span class="hb-ref{side}" style="left:{pct(rv):.2f}%">{em}</span>'
        lab = esc(r["label"]) if r.get("label") else ""
        out.append(f'<div class="hb-lab{" cont" if not lab else ""}">{lab}{sub}</div>'
                   f'<div class="hb-track">{lines}{bar}<span class="hb-val">{num}</span></div>')
    out.append("</div>")
    if refs:
        out.insert(0, '<p class="hb-refkey">Dashed lines: ' + "; ".join(esc(l) for _, l in refs) + "</p>")
    return "".join(out)


# ------------------------------------------------------------------------------------- roofline


def roofline(reg: Reg, bw_lines, ceilings, points) -> str:
    """bw_lines: [(GB/s, label, cls, src)]; ceilings: [(GFLOP/s, label, tool, src, dashed)];
    points: [(intensity, GFLOP/s, tool, label, src)]."""
    W, H = 720, 460
    L, R, T, B = 62, 150, 18, 50
    x0, x1 = 0.25, 4096.0
    y0, y1 = 10.0, 60000.0

    def X(v):
        return L + (W - L - R) * (math.log2(v) - math.log2(x0)) / (math.log2(x1) - math.log2(x0))

    def Y(v):
        return T + (H - T - B) * (1 - (math.log10(v) - math.log10(y0)) / (math.log10(y1) - math.log10(y0)))

    s = [f'<svg class="roof" viewBox="0 0 {W} {H}" role="img" aria-labelledby="roof-t roof-d">'
         '<title id="roof-t">Roofline: attainable GFLOP/s vs arithmetic intensity</title>'
         '<desc id="roof-d">Diagonal lines are measured memory bandwidth; horizontal lines are the best '
         'measured matmul rate per engine; dots are Llama 3.2 3B projection shapes.</desc>']
    # grid
    for e in range(-2, 13, 2):
        v = 2.0 ** e
        if x0 <= v <= x1:
            s.append(f'<line class="grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{T}" y2="{H - B}"/>'
                     f'<text class="ax" x="{X(v):.1f}" y="{H - B + 16}" text-anchor="middle">'
                     f'{("1/" + str(int(1 / v))) if v < 1 else int(v)}</text>')
    for e in range(1, 5):
        v = 10.0 ** e
        lab = {10: "10", 100: "100", 1000: "1 T", 10000: "10 T"}[int(v)]
        s.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                 f'<text class="ax" x="{L - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{lab}</text>')
    s.append(f'<text class="ax-t" x="{(L + W - R) / 2:.0f}" y="{H - 10}" text-anchor="middle">'
             'arithmetic intensity (FLOP per byte moved, log₂)</text>'
             f'<text class="ax-t" transform="translate(14 {(T + H - B) / 2:.0f}) rotate(-90)" '
             'text-anchor="middle">GFLOP/s (log)</text>')
    s.append(f'<clipPath id="plot"><rect x="{L}" y="{T}" width="{W - L - R}" height="{H - T - B}"/></clipPath>')
    s.append('<g clip-path="url(#plot)">')
    for bw, lab, cls, src in bw_lines:
        xe = min(x1, y1 / bw)
        s.append(f'<line class="bw {cls}" x1="{X(x0):.1f}" y1="{Y(max(y0, bw * x0)):.1f}" '
                 f'x2="{X(xe):.1f}" y2="{Y(bw * xe):.1f}" data-s="{src.id if src else ""}" '
                 f'data-d="{esc(lab)}"/>')
    gpu_bw = max((b for b, _, c, _ in bw_lines if c == "gpu"), default=None)
    cpu_bw = max((b for b, _, c, _ in bw_lines if c == "cpu"), default=None)
    for v, lab, tool, src, dashed in ceilings:
        ridge_bw = cpu_bw if tool == "accel" else gpu_bw
        xr = v / ridge_bw if ridge_bw else x0
        s.append(f'<line class="ceil t-{tool}{" dash" if dashed else ""}" x1="{X(max(x0, xr)):.1f}" '
                 f'x2="{X(x1):.1f}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" data-s="{src.id if src else ""}" '
                 f'data-d="{esc(lab)}: {fmt_val(v)} GFLOP/s"/>')
    for ai, v, tool, lab, src in points:
        if x0 <= ai <= x1 and y0 <= v <= y1:
            s.append(f'<circle class="pt t-{tool}" cx="{X(ai):.1f}" cy="{Y(v):.1f}" r="5" tabindex="0" '
                     f'data-s="{src.id}" data-d="{esc(lab)}: {fmt_val(v)} GFLOP/s at {ai:.1f} FLOP/B"/>')
    s.append("</g>")
    # bandwidth key, top-left inside the plot
    kx, ky = L + 10, T + 10
    s.append(f'<rect class="lk" x="{kx - 6}" y="{ky - 4}" width="190" height="{len(bw_lines) * 17 + 6}" rx="4"/>')
    for i, (bw, lab, cls, src) in enumerate(bw_lines):
        yy = ky + 8 + i * 17
        s.append(f'<line class="bw lkl {cls}" x1="{kx}" x2="{kx + 22}" y1="{yy}" y2="{yy}"/>'
                 f'<text class="lab" x="{kx + 28}" y="{yy + 4}">{esc(lab)}</text>')
    # ceiling labels in the right gutter, pushed apart
    labs = sorted(((Y(v), lab, v, tool) for v, lab, tool, _, _ in ceilings), key=lambda t: t[0])
    placed = []
    for y, lab, v, tool in labs:
        yy = max(y + 4, (placed[-1] + 13) if placed else T + 8)
        placed.append(yy)
        s.append(f'<rect class="sw t-{tool}" x="{W - R + 6}" y="{yy - 8:.1f}" width="8" height="8" rx="2"/>'
                 f'<text class="lab" x="{W - R + 18}" y="{yy:.1f}">{esc(lab)} {v / 1000:.1f} T</text>')
    s.append("</svg>")
    return "".join(s)
