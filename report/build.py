"""Build ~/Desktop/mojo-vs-apple-silicon.html from the project's result files.

    uv run --project report python report/build.py [-o OUT.html]

Stdlib only. Reads bench/results, atlas/results, fpga/, doom/results (see report/README.md for
the data contract). Missing inputs render as placeholders; nothing here should crash on absent
or partial data.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rlib import charts, loaders, pipeline  # noqa: E402
from rlib.charts import TOOLS, hbars, legend, roofline  # noqa: E402
from rlib.sources import ROOT, STATUS, Reg, esc, fmt_val  # noqa: E402

DEFAULT_OUT = Path.home() / "Desktop" / "mojo-vs-apple-silicon.html"
NOW = datetime.now()


# ===================================================================================== helpers


class Atlas:
    def __init__(self, a: dict):
        self.a = a
        self.ok = [r for r in a["records"] if "error" not in r and r.get("median") is not None]

    def best(self, pred, key="median"):
        rr = [r for r in self.ok if pred(r) and r.get(key) is not None]
        return max(rr, key=lambda r: r[key]) if rr else None

    def gemm(self):
        return [r for r in self.ok if r.get("bench") == "gemm"]


def rdetail(r: dict, key="median") -> str:
    k = r.get("kernel") or r.get("mode") or ""
    shape = f'{r.get("shape", "")} (M={r.get("M")}, N={r.get("N")}, K={r.get("K")})' if r.get("M") else r.get("shape", "")
    sd = r.get("stddev")
    stat = f"median of {r.get('reps', '?')} reps" + (f", sd {fmt_val(sd)} {r.get('unit', '')}" if sd is not None else "")
    if key.startswith("eff"):
        stat = "effective GB/s = bytes per call / median time"
    return f"{r.get('engine')} {k} {r.get('dtype', '')}, {shape}; {stat}".replace("  ", " ")


def derived(reg: Reg, label: str, note: str):
    return reg.add(f"derived: {label}", NOW.timestamp(), "derived", note)


def section(id_, title, body, lead="") -> str:
    return (f'<section id="{id_}" class="part"><h2>{title}</h2>'
            f'{f"<p class=lead>{lead}</p>" if lead else ""}{body}</section>')


def fig(title, body, srcs, reg: Reg, note="", wide=False) -> str:
    srcs = [s for s in srcs if s is not None]
    return (f'<figure class="fig{" wide" if wide else ""}"><figcaption><span class="ft">{title}</span>'
            f'<span class="fb">{reg.badges_for(srcs)}{reg.cite(srcs)}</span></figcaption>{body}'
            f'{f"<p class=note>{note}</p>" if note else ""}</figure>')


def table(head, rows, cls="") -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def placeholder(text) -> str:
    return f'<div class="ph">{text}</div>'


# ====================================================================================== header


def header(reg: Reg, at: Atlas, bench) -> str:
    inv = at.a.get("inventory") or {}
    isrc = at.a.get("inv_src")
    s = inv.get("summary", {})
    v = inv.get("versions", {})
    spec = inv.get("apple_spec", {})
    hw = (inv.get("system_profiler_hardware", {}).get("SPHardwareDataType") or [{}])[0]
    if not s:
        machine = placeholder("No <code>atlas/results/inventory.json</code> yet: run "
                              "<code>atlas/scripts/inventory.py</code>.")
    else:
        mem = s.get("memory_bytes", 0) / 2**30
        gpu = str(s.get("gpu_core_count", "")).split("=")[-1].strip()
        items = [
            ("Machine", f'{esc(hw.get("machine_name", ""))} ({esc(hw.get("machine_model", s.get("model", "")))})'),
            ("Chip", esc(s.get("chip", "?"))),
            ("CPU", f'{s.get("p_cores")} performance + {s.get("e_cores")} efficiency cores; SME2, '
                    f'SVL {s.get("sme_svl_bytes", 0) * 8} bits'),
            ("GPU", f'{esc(gpu)} cores, <code>{esc(s.get("gpu_arch_mlx", ""))}</code>, a Neural Accelerator '
                    'in each core'),
            ("Neural Engine", "16 cores, no published TOPS"),
            ("Memory", f'{mem:.0f} GiB unified, Apple spec '
                       f'{reg.num(spec.get("memory_bandwidth_GBps"), isrc, unit=" GB/s", detail="Apple newsroom figure recorded in inventory.json")}'),
        ]
        machine = "<dl>" + "".join(f"<dt>{k}</dt><dd>{val}</dd>" for k, val in items) + "</dl>"
    mlx_ver = next((r["version"] for r in bench if r["tool"] == "mlx"), v.get("mlx", "?"))
    llama_ver = next((r["version"] for r in bench if r["runtime"] == "llama.cpp"), v.get("llamacpp_commit", "?"))
    tools = [
        ("macOS", esc(v.get("macos", "?"))),
        ("Mojo", esc(v.get("mojo", "?").removeprefix("Mojo "))),
        ("MAX", esc(_max_ver(v.get("max_pkgs", "")))),
        ("MLX", esc(mlx_ver)),
        ("coremltools", esc(v.get("coremltools", "?"))),
        ("llama.cpp", esc(llama_ver)),
        ("Apple compiler", esc(v.get("clang", "?"))),
        ("Xcode", "Command Line Tools only; no <code>metal</code> compiler" if "unable to find" in
         v.get("metal_compiler", "") else esc(v.get("xcode", "?"))),
    ]
    tv = "<dl>" + "".join(f"<dt>{k}</dt><dd>{val}</dd>" for k, val in tools) + "</dl>"
    return (
        '<header class="mast"><div class="mast-in">'
        '<h1>Mojo/MAX against Apple’s own toolchain, on one M5</h1>'
        '<p class="lede">How Modular’s open Mojo compiler and MAX serving stack reach the same silicon as '
        'Metal, MLX, Core ML and Accelerate: what each pipeline does stage by stage, how fast each runs the '
        'same models and matmul shapes, and how far Mojo reaches across the chip’s processors.</p>'
        f'<div class="plate"><div><h3>Machine</h3>{machine}</div><div><h3>Toolchains</h3>{tv}</div></div>'
        f'{freshness_strip(reg)}'
        '<nav class="toc" aria-label="Sections"><a href="#p1">1. Pipelines</a><a href="#p2">2. Performance</a>'
        '<a href="#p3">3. Mojo across processors</a><a href="#sources">Sources and caveats</a></nav>'
        '</div></header>')


def _max_ver(pk: str) -> str:
    for line in pk.splitlines():
        parts = line.split()
        if parts and parts[0] == "max":
            return parts[1]
    return "?"


def freshness_strip(reg: Reg) -> str:
    # filled after all sections registered their sources (placeholder replaced in main)
    return "<!--FRESHNESS-->"


def freshness_html(reg: Reg) -> str:
    counts = {}
    for s in reg.sources.values():
        counts[s.status] = counts.get(s.status, 0) + 1
    parts = " ".join(f'<span class="fc">{reg.badge(k)}<b>{n}</b></span>' for k, n in counts.items())
    return (f'<div class="fresh"><p>Generated {NOW.strftime("%Y-%m-%d %H:%M")} from {len(reg.sources)} '
            f'sources: {parts}</p><p class="fresh-how">Hover, focus or tap any number to see its file, date '
            'and status. Numbers with a dotted underline come from smoke runs and will be replaced by the '
            'idle-machine atlas run.</p></div>')


# ====================================================================================== part 1


def part1(reg: Reg) -> str:
    body = (pipeline.key_html() + '<div class="pipe-wrap">' + pipeline.render(reg) + "</div>" +
            '<div class="prose"><h3>What lines up, and what doesn’t</h3><ul>'
            '<li><strong>AIR is the meeting point.</strong> Both stacks hand the GPU the same LLVM-based AIR '
            'bitcode. Apple produces it with the closed <code>metal</code> front end; Mojo produces it from its '
            'own MLIR pipeline and packs the <code>.metallib</code> itself, so it needs no Xcode.</li>'
            '<li><strong>Openness ends at different depths.</strong> Mojo/MAX is readable from parser to '
            'metallib writer and offers plugin ABIs for new targets and devices. On the Apple side only MLX and '
            'coremltools are open, and both hand off to closed frameworks.</li>'
            '<li><strong>Only Core ML reaches the Neural Engine.</strong> Mojo/MAX covers CPU, GPU and the '
            'GPU Neural Accelerators, but has no ANE path.</li>'
            '<li><strong>One language from kernel to server.</strong> MLX and llama.cpp also serve models, but '
            'their kernels are written in a different language (Metal, C++) from the framework around them. '
            'With MAX, the matmul kernel, the graph custom op and the server path are all Mojo or driven by '
            'it.</li></ul></div>')
    return section("p1", "1. The two pipelines, stage by stage", body,
                   "Each row is an equivalent stage. Read across a row to compare what each toolchain does at "
                   "that point; read down a column to follow one toolchain from source to hardware.")


# ====================================================================================== part 2


def gpu_bw_best(at: Atlas):
    return at.best(lambda r: r.get("bench") == "gpu_bandwidth")


def part2(reg: Reg, at: Atlas, bench) -> str:
    out = []
    # ---------------------------------------------------------------- 2a LLM phase 1
    if not bench:
        out.append(fig("Llama 3.2 3B, single stream", placeholder("No files in <code>bench/results/</code> yet."),
                       [], reg))
    else:
        bw = gpu_bw_best(at)
        llama16 = next((r for r in bench if r["runtime"] == "llama.cpp" and r["fclass"] == "16-bit"), None)
        llama4 = next((r for r in bench if r["runtime"] == "llama.cpp" and r["fclass"] == "4-bit"), None)
        sizes16 = llama16.get("model_bytes") if llama16 else 6.43e9
        size_q4 = llama4.get("model_bytes") if llama4 else None
        size_mlx4 = 1.81e9  # bench/README.md models table (mlx-community 4-bit, 1.81 GB)
        dsrc = None
        refs = {"16-bit": [], "4-bit": []}
        if bw:
            dsrc = derived(reg, "decode bandwidth bound",
                           f"best GPU streaming bandwidth ({fmt_val(bw['median'])} GB/s, {bw['_src'].path}) ÷ "
                           "weight bytes per token (llama-bench model_size; MLX 4-bit 1.81 GB from bench/README.md)")
            refs["16-bit"].append((bw["median"] * 1e9 / sizes16, f"bound {bw['median'] * 1e9 / sizes16:.1f}"))
            if size_q4:
                refs["4-bit"].append((bw["median"] * 1e9 / size_q4, f"Q4_K_M bound {bw['median'] * 1e9 / size_q4:.0f}"))
            refs["4-bit"].append((bw["median"] * 1e9 / size_mlx4, f"MLX 4-bit bound {bw['median'] * 1e9 / size_mlx4:.0f}"))
        top_tg = max([r["tg"] or 0 for r in bench] + [v for g in refs.values() for v, _ in g])
        groups = []
        for fc in ("16-bit", "8-bit", "4-bit"):
            rows = [r for r in bench if r["fclass"] == fc]
            if not rows:
                continue
            hb = [{"label": r["label"], "sub": f'{r["format"]}, {r["method"]}', "value": r["tg"], "src": r["src"],
                   "tool": r["tool"],
                   "detail": f'decode tok/s = 1000 / TPOT; {r["reps"] or "?"} reps'
                             + (f', sd {r["tg_std"]:.2f}' if r.get("tg_std") else "")} for r in rows]
            groups.append(f'<h4 class="gh">{fc} weights</h4>' +
                          hbars(reg, hb, " tok/s", nd=1, max_v=top_tg, refs=refs.get(fc, ()),
                                aria=f"Decode tokens per second, {fc}"))
        top_ttft = max([r["ttft"] or 0 for r in bench])
        ttft_parts = []
        for fc in ("16-bit", "8-bit", "4-bit"):
            sel = [r for r in bench if r["fclass"] == fc]
            if not sel:
                continue
            tr = [{"label": r["label"], "sub": f'{r["format"]}, {r["method"]}', "value": r["ttft"],
                   "src": r["src"], "tool": r["tool"],
                   "detail": f'time to first token, 512-token prompt; {r["ttft_note"]}' +
                             (f', sd {r["ttft_std"]:.1f} ms' if r.get("ttft_std") else "")} for r in sel]
            ttft_parts.append(f'<h4 class="gh">{fc} weights</h4>' +
                              hbars(reg, tr, " ms", nd=0, max_v=top_ttft, aria=f"Time to first token, {fc}"))
        tools_in = {r["tool"] for r in bench}
        srcs = [r["src"] for r in bench] + [dsrc]
        out.append(fig("Llama 3.2 3B decode speed (higher is better)", legend(tools_in) + "".join(groups),
                       srcs, reg, note="Decode is memory-bound: each token streams every weight once, so the "
                       "dashed line is the best measured GPU bandwidth divided by the weight bytes. All three "
                       "runtimes sit on the same wall at 16-bit; 4-bit lifts it about threefold for everyone."))
        out.append(fig("Time to first token, 512-token prompt (lower is better)",
                       legend(tools_in) + "".join(ttft_parts), srcs, reg,
                       note="Methods differ: llama.cpp TTFT is 512 ÷ pp from llama-bench (no HTTP), MLX is "
                       "in-process, MAX and llama.cpp server are measured by an HTTP client. Compare within a "
                       "method first."))
        rows = []
        for r in bench:
            rows.append([f'<span class="rt"><span class="sw t-{r["tool"]}"></span>{esc(r["label"])}</span>', esc(r["format"]),
                         esc(r["method"]), reg.num(r["pp"], r["src"], 0), reg.num(r["tg"], r["src"], 1),
                         reg.num(r["ttft"], r["src"], 0), reg.num(r["peak"], r["src"], 2, detail="peak memory GB")
                         if r["peak"] else '<span class="muted">–</span>',
                         esc(r["version"]) if not str(r["version"]).startswith("http") else
                         '<span class="muted">HTTP endpoint</span>'])
        out.append(fig("Phase-1 table", table(["runtime", "format", "method", "prefill tok/s", "decode tok/s",
                                               "TTFT ms", "peak GB", "version"], rows, "num"),
                       [r["src"] for r in bench], reg,
                       note="Same prompt (bench/prompt_512.txt, 512 tokens), 128 greedy tokens, EOS ignored, "
                            "1 warm-up and 3 measured reps. Formats are not identical quantizations: "
                            "llama.cpp uses F16 and Q4_K_M (~4.8 bpw), MLX bf16 and 4-bit g64 (~4.5 bpw)."))

    # ---------------------------------------------------------------- 2b bandwidth
    bwr = [r for r in at.ok if r.get("bench") in ("cpu_bandwidth", "gpu_bandwidth")]
    inv = at.a.get("inventory") or {}
    spec = (inv.get("apple_spec") or {}).get("memory_bandwidth_GBps")
    if bwr:
        groups = {}
        for r in bwr:
            if r["engine"] == "cpu":
                key = ("cpu_e" if "background" in str(r.get("qos")) else "cpu_p")
            else:
                key = r["engine"]
            if key not in groups or r["median"] > groups[key]["median"]:
                groups[key] = r
        labels = {"cpu_p": ("CPU, P-preferred threads", "accel"), "cpu_e": ("CPU, E-cores only", "accel"),
                  "gpu_mojo": ("GPU, Mojo kernel", "max"), "gpu_mlx": ("GPU, MLX", "mlx")}
        rows = []
        for k in ("gpu_mlx", "gpu_mojo", "cpu_p", "cpu_e"):
            if k in groups:
                r = groups[k]
                th = f'{r.get("threads")} threads, ' if r.get("threads") else ""
                rows.append({"label": labels[k][0], "sub": f'{th}best op: {r["op"]}', "value": r["median"],
                             "src": r["_src"], "tool": labels[k][1], "detail": rdetail(r)})
        refs = [(spec, f"Apple spec {spec}")] if spec else []
        out.append(fig("Streaming memory bandwidth, best configuration (GB/s)",
                       legend({r["tool"] for r in rows}) + hbars(reg, rows, " GB/s", refs=refs, aria="Bandwidth"),
                       [r["src"] for r in rows] + [at.a.get("inv_src")], reg,
                       note="CPU numbers are a C STREAM-style benchmark, not a toolchain. The Mojo GPU kernel "
                            "is a plain hand-written copy/reduce; MLX’s is its built-in ops."))
    else:
        out.append(fig("Memory bandwidth", placeholder("No bandwidth records in <code>atlas/results/</code>."), [], reg))

    # ---------------------------------------------------------------- 2c same-shape matmul
    out.append(matmul_compare(reg, at))
    # ---------------------------------------------------------------- 2d roofline
    out.append(roofline_fig(reg, at))
    return section("p2", "2. Performance across toolchains", "".join(out),
                   "Same model, same prompt and the same matmul shapes, run through each toolchain on this "
                   "machine. Every bar names its kernel and data type; hover for the exact record.")


PATH_LABEL = {
    "max_mma": "MAX AppleM5MatMul (Neural Accel.)", "max_8x8": "MAX matmul_8x8 (simdgroup)",
    "max_apple_int8_w8a8": "MAX int8 W8A8", "max_apple_fp4_w4a16": "MAX NVFP4 W4A16",
    "max_apple_gemv": "MAX apple_gemv", "mlx_nax": "MLX (Neural Accel.)", "mlx_nonax": "MLX (simdgroup, NA off)",
    "mlx_notf32": "MLX (TF32 off)", "sme": "Accelerate (SME)", "coreml_CPU_AND_NE": "Core ML, CPU+ANE",
    "coreml_CPU_AND_GPU": "Core ML, CPU+GPU", "coreml_ALL": "Core ML, all units", "coreml_CPU_ONLY": "Core ML, CPU only",
}
SHAPE_KEYS = {"attn_q_o": (3072, 3072), "attn_kv": (3072, 1024), "mlp_up_gate": (3072, 8192),
              "mlp_down": (8192, 3072), "lm_head": (3072, 128256)}


def canonical_mnk(shape: str):
    if shape.startswith("square"):
        n = int(shape.removeprefix("square"))
        return (n, n, n)
    base, _, m = shape.rpartition("_M")
    if base in SHAPE_KEYS and m.isdigit():
        k, n = SHAPE_KEYS[base]
        return (int(m), n, k)
    return None


def shape_rows(reg: Reg, at: Atlas, shape: str, key: str):
    rows = {}
    for r in at.gemm():
        if r.get("shape") != shape or r.get(key) is None:
            continue
        p = r["_path"]
        if p not in PATH_LABEL or p in ("coreml_CPU_ONLY", "coreml_ALL", "mlx_notf32"):
            continue
        if r.get("mode") == "single_threaded":
            continue
        k = p  # best record per kernel path; its dtype is shown
        if k not in rows or r[key] > rows[k][key]:
            rows[k] = r
    out = []
    can = canonical_mnk(shape)
    for p, r in sorted(rows.items(), key=lambda kv: -kv[1][key]):
        sub = str(r["dtype"])
        if can and (r.get("M"), r.get("N"), r.get("K")) != can:
            sub += f' · ran M={r.get("M")}, not {can[0]}'
        out.append({"label": PATH_LABEL[p], "sub": sub, "value": r[key], "src": r["_src"], "tool": r["_tool"],
                    "detail": rdetail(r, key)})
    return out


def ggml_rows(reg: Reg, at: Atlas, M: int, as_gbps: bool):
    rows = []
    for r in at.gemm():
        if not r["engine"].startswith("ggml") or r.get("M") != M:
            continue
        v = r["median"]
        detail = rdetail(r)
        if as_gbps:  # weights dominate at M=1: bytes ≈ N*K*bytes_per_weight; GFLOP/s = 2*N*K/t
            bpw = 2.0 if "f16" in r["dtype"] else (18 / 32 if "q4_0" in r["dtype"] else None)
            if bpw is None:
                continue
            v = r["median"] * bpw / 2.0
            detail += f"; GB/s derived as GFLOP/s × {bpw:g} B/weight ÷ 2 FLOP"
        be = "Metal" if "mtl" in r["engine"] else "CPU"
        rows.append({"label": f"llama.cpp ggml ({be})", "sub": f'{r["dtype"]} · other shape, K={r["K"]} N={r["N"]}',
                     "value": v, "src": r["_src"], "tool": "llama", "detail": detail, "hollow": True})
    return rows


def matmul_compare(reg: Reg, at: Atlas) -> str:
    if not at.gemm():
        return fig("Matmul across toolchains", placeholder("No GEMM records in <code>atlas/results/</code>."), [], reg)
    parts, srcs = [], []
    specs = [
        ("attn_q_o_M1", "eff_gbps_median", " GB/s", "Decode-shaped: one token × 3072×3072 weights (effective GB/s)", 1, True),
        ("attn_q_o_M512", "median", " GFLOP/s", "Prefill-shaped: 512 tokens × 3072×3072 weights (GFLOP/s)", 512, False),
        ("square1024", "median", " GFLOP/s", "Square 1024³ (GFLOP/s)", None, False),
    ]
    tools = set()
    for shape, key, unit, title, ggm, gb in specs:
        rows = shape_rows(reg, at, shape, key)
        if ggm:
            rows += ggml_rows(reg, at, ggm, gb)
        if not rows:
            parts.append(f'<h4 class="gh">{esc(title)}</h4>' + placeholder(f"No records for <code>{shape}</code>."))
            continue
        tools |= {r["tool"] for r in rows}
        srcs += [r["src"] for r in rows]
        parts.append(f'<h4 class="gh">{esc(title)}</h4>' + hbars(reg, rows, unit, aria=title))
    return fig("The same matmul shapes through every toolchain", legend(tools) + "".join(parts), srcs, reg,
               note="One bar per kernel path: its best record over data types, with that data type underneath. "
                    "Shapes are Llama 3.2 3B projections, C[M,N] = A[M,K]·W[N,K]ᵀ. Hollow bars are llama.cpp’s "
                    "own MUL_MAT test at a different shape (K=14336, N=4096), shown as a cross-check only. "
                    "Core ML timings include Python predict() overhead, which dominates at M=1.")


def roofline_fig(reg: Reg, at: Atlas) -> str:
    gemm = at.gemm()
    if not gemm:
        return fig("Roofline", placeholder("Needs atlas GEMM and bandwidth records."), [], reg)
    inv = at.a.get("inventory") or {}
    spec = (inv.get("apple_spec") or {}).get("memory_bandwidth_GBps")
    bw_lines, srcs = [], []
    if spec:
        bw_lines.append((spec, f"Apple spec {spec} GB/s", "spec", at.a.get("inv_src")))
    g = gpu_bw_best(at)
    c = at.best(lambda r: r.get("bench") == "cpu_bandwidth")
    if g:
        bw_lines.append((g["median"], f"GPU measured {g['median']:.0f} GB/s", "gpu", g["_src"]))
    if c:
        bw_lines.append((c["median"], f"CPU measured {c['median']:.0f} GB/s", "cpu", c["_src"]))
    ceil = []

    def add(pred, label, tool, dashed=False):
        r = at.best(pred)
        if r:
            ceil.append((r["median"], label, tool, r["_src"], dashed))
            srcs.append(r["_src"])

    big = lambda r: (r.get("M") or 0) >= 256 and r.get("bench") == "gemm"
    add(lambda r: big(r) and r["_path"] == "max_apple_int8_w8a8", "MAX int8", "max", True)
    add(lambda r: big(r) and r["_path"] == "max_mma" and r["dtype"] in ("float16", "bfloat16"), "MAX MMA 16b", "max")
    add(lambda r: big(r) and r["_path"] == "mlx_nax" and r["dtype"] in ("f16", "bf16"), "MLX NAX 16b", "mlx")
    add(lambda r: big(r) and r["_path"] == "coreml_CPU_AND_NE", "Core ML ANE", "coreml")
    add(lambda r: big(r) and r["_path"] == "max_8x8" and r["dtype"] == "float16", "MAX simdgrp", "max", True)
    add(lambda r: r.get("bench") == "cpu_peak" and r.get("engine") == "cpu_sme" and r.get("dtype") == "f16->f32",
        "SME f16→32 (C)", "accel")
    add(lambda r: big(r) and r["_path"] == "sme", "Accelerate", "accel", True)
    pts = []
    for r in gemm:
        if r.get("bytes_per_call") and r.get("M") in (1, 4, 16, 512) and not str(r.get("shape", "")).startswith("square") \
                and r["_path"] in ("max_mma", "max_8x8", "mlx_nax", "sme", "max_apple_int8_w8a8", "max_apple_fp4_w4a16") \
                and r.get("mode") != "single_threaded":
            mult = 1 if "int8" in r["_path"] else 1
            ai = 2.0 * r["M"] * r["N"] * r["K"] / r["bytes_per_call"] * mult
            pts.append((ai, r["median"], r["_tool"], f'{PATH_LABEL.get(r["_path"], r["_path"])} {r["dtype"]} {r["shape"]}',
                        r["_src"]))
            srcs.append(r["_src"])
    srcs += [s for *_x, s in bw_lines]
    body = legend({t for *_x, t, _s, _d in ceil} | {p[2] for p in pts}) + roofline(reg, bw_lines, ceil, pts)
    return fig("Roofline: where each shape sits against bandwidth and compute", body, srcs, reg, wide=True,
               note="Decode (M=1) shapes have intensity near 1 FLOP/byte and sit on the bandwidth line for "
                    "every toolchain; only prefill (M=512) reaches the compute ceilings. Ceilings are the best "
                    "measured rate per path at M ≥ 256, not vendor peaks. Dashed ceilings: int8, simdgroup "
                    "(no Neural Accelerator) and Accelerate. int8 is counted in ops, not FLOPs.")


# ====================================================================================== part 3


def part3(reg: Reg, at: Atlas, fpga: dict, doom: list) -> str:
    out = [reach_matrix(reg, at, fpga), na_fig(reg, at), lowp_fig(reg, at), tile_fig(reg, at, fpga),
           synth_fig(reg, fpga), doom_fig(reg, doom)]
    return section("p3", "3. Mojo across processors and backends", "".join(out),
                   "One language, many targets: what Mojo reaches on this chip, what it measures there, and the "
                   "invented hardware the project is building below the GPU.")


def reach_matrix(reg: Reg, at: Atlas, fpga: dict) -> str:
    b = at.best
    big = lambda r: (r.get("M") or 0) >= 256 and r.get("bench") == "gemm"
    neon = b(lambda r: r.get("engine") == "cpu_neon" and r.get("dtype") == "f16" and r.get("threads") == 4)
    sme_c = b(lambda r: r.get("engine") == "cpu_sme" and r.get("dtype") == "f16->f32")
    accel = b(lambda r: big(r) and r["_path"] == "sme")
    mojo_cpu = b(lambda r: r.get("engine") in ("cpu_mojo", "mojo_cpu"))
    m8 = b(lambda r: big(r) and r["_path"] == "max_8x8" and r["dtype"] in ("float16", "bfloat16"))
    xn = b(lambda r: big(r) and r["_path"] == "mlx_nonax" and r["dtype"] in ("f16", "bf16"))
    mm = b(lambda r: big(r) and r["_path"] == "max_mma" and r["dtype"] in ("float16", "bfloat16"))
    xa = b(lambda r: big(r) and r["_path"] == "mlx_nax" and r["dtype"] in ("f16", "bf16"))
    i8 = b(lambda r: big(r) and r["_path"] == "max_apple_int8_w8a8")
    xi8 = b(lambda r: big(r) and r["_path"] == "mlx_nax" and "int8" in str(r["dtype"]))
    f4 = b(lambda r: big(r) and r["_path"] == "max_apple_fp4_w4a16")
    xi4 = b(lambda r: big(r) and r["_path"] == "mlx_nax" and "int4" in str(r["dtype"]))
    ane = b(lambda r: big(r) and r["_path"] == "coreml_CPU_AND_NE")
    gbw_m = b(lambda r: r.get("engine") == "gpu_mojo")
    gbw_x = b(lambda r: r.get("engine") == "gpu_mlx" and r.get("bench") == "gpu_bandwidth")

    def n(r, unit=" GFLOP/s", what=""):
        if not r:
            return '<span class="muted">not measured</span>'
        return reg.num(r["median"], r["_src"], unit=unit, detail=rdetail(r)) + (
            f'<small>{esc(what)}</small>' if what else "")

    def st(word, cls):
        return f'<span class="stt s-{cls}">{word}</span>'

    tile = next((p for p in fpga.get("perf", []) if p.get("N") == 16), None)
    tile_cell = '<span class="muted">not built</span>'
    if tile and fpga.get("perf_src"):
        gops = tile["macs"] * 2 / (tile["cycles"] / 100e6) / 1e9
        tile_cell = reg.num(gops, fpga["perf_src"], unit=" GOP/s",
                            detail="N=16 tile at an assumed 100 MHz, M=512 K=3072 N=8192; cycles from the "
                                   "DESIGN.md §5 formula") + "<small>simulated, 100 MHz assumed</small>"
    synth = [s for s in fpga.get("synth", []) if s.get("fmax_mhz")]
    fp_cell = '<span class="muted">not synthesized</span>'
    if synth:
        s0 = max(synth, key=lambda s: s["N"])
        gops = s0["N"] ** 2 * 2 * s0["fmax_mhz"] * 1e6 / 1e9
        fp_cell = reg.num(gops, s0["_src"], unit=" GOP/s",
                          detail=f'peak N²·2·fmax for N={s0["N"]} at {s0["fmax_mhz"]:.1f} MHz '
                                 f'(nextpnr, {s0["device"]}); not run on a board') + \
            f'<small>N={s0["N"]} peak at {s0["fmax_mhz"]:.0f} MHz</small>'
    ph = fpga.get("phantom")
    rows = [
        ["CPU NEON", "yes (AArch64 backend)", n(mojo_cpu, what="Mojo"), n(neon, what="C microkernel, f16, 4 P-cores"),
         st("no Mojo CPU run yet", "todo")],
        ["CPU SME", "via Accelerate call or AMX asm wrappers", '<span class="muted">not measured</span>',
         n(sme_c, what="C microkernel f16→f32") + "<br>" + n(accel, what="Accelerate"), st("C reference only", "todo")],
        ["GPU simdgroup", "yes", n(m8, what="MAX matmul_8x8"), n(xn, what="MLX, NA off"), st("measured", "ok")],
        ["GPU Neural Accelerators", "yes (Metal 4 MMA)", n(mm, what="MAX AppleM5MatMul"), n(xa, what="MLX NAX"),
         st("measured", "ok")],
        ["GPU int8", "yes", n(i8, " GOP/s", "MAX W8A8"), n(xi8, what="MLX int8 weights, bf16 math"), st("measured", "ok")],
        ["GPU 4-bit", "yes", n(f4, what="MAX NVFP4 W4A16"), n(xi4, what="MLX int4 g64"), st("measured", "ok")],
        ["GPU memory", "yes", n(gbw_m, " GB/s", "Mojo kernel"), n(gbw_x, " GB/s", "MLX"), st("measured", "ok")],
        ["Neural Engine", "no", '<span class="muted">no path</span>', n(ane, what="Core ML CPU+ANE"),
         st("Core ML only", "no")],
        ["Phantom NPU", "yes: stdlib target + HAL driver plugin; FFI route used (codegen plugin needs a source build)",
         (reg.num(ph.get("gops"), ph["_src"], unit=" GOP/s") if ph and ph.get("gops") else
          '<span class="muted">not started</span>'), tile_cell,
         st(ph.get("status", "in progress"), "ok" if ph and ph.get("status") == "pass" else "todo") if ph
         else st("not started", "todo")],
        ["RISC-V soft core (sim)", "yes: <code>--target-triple riscv32-unknown-none-elf</code>, bare-metal firmware",
         '<span class="muted">drives tile v2 in a VexRiscv SoC (Verilator)</span>', '<span class="muted">C firmware</span>',
         st("simulation", "ok")],
        ["ECP5 FPGA", "planned: same SoC on a ULX3S-class board (G1-E)", '<span class="muted">no board run</span>', fp_cell,
         st("synthesis only", "todo")],
    ]
    return fig("Which processors Mojo reaches, and what it measures there",
               table(["engine", "Mojo reaches it?", "Mojo / MAX", "best other path", "status"], rows, "reachtbl"),
               [x["_src"] for x in (neon, sme_c, accel, m8, xn, mm, xa, i8, xi8, f4, xi4, ane, gbw_m, gbw_x) if x]
               + [fpga.get("perf_src")] + [s["_src"] for s in synth], reg,
               note="Matmul rates are the best record at M ≥ 256 for each path. CPU rows are C reference numbers "
                    "until a Mojo CPU benchmark lands (records with engine <code>cpu_mojo</code> fill the Mojo "
                    "column automatically). MLX int8/int4 dequantize to bf16, so they are not W8A8.")


def na_fig(reg: Reg, at: Atlas) -> str:
    pairs = []
    for r in at.gemm():
        if r["_path"] != "max_mma" or r["dtype"] != "float16":
            continue
        twin = [x for x in at.gemm() if x["_path"] == "max_8x8" and x.get("shape") == r.get("shape")
                and x["dtype"] == r["dtype"]]
        if twin:
            pairs.append((r, twin[0]))
    if not pairs:
        return fig("GPU Neural Accelerators vs simdgroup", placeholder("Needs MAX <code>apple_m5_mma</code> and "
                   "<code>simdgroup_8x8</code> records for the same shape."), [], reg)

    def order(p):
        sh = p[0]["shape"]
        return (0 if sh.startswith("square") else 1, p[0].get("M", 0), sh)
    pairs.sort(key=order)
    rows, srcs = [], []
    for a, b in pairs:
        key = "eff_gbps_median" if (a.get("M") or 0) <= 16 and a.get("eff_gbps_median") else "median"
        unit = "GB/s" if key.startswith("eff") else "GFLOP/s"
        rows.append({"label": f'{a["shape"]}', "sub": f"with Neural Accelerators, {unit}", "value": a[key],
                     "src": a["_src"], "tool": "max", "detail": rdetail(a, key)})
        rows.append({"label": "", "sub": f"simdgroup only, ×{a[key] / b[key]:.2f} speed-up", "value": b[key],
                     "src": b["_src"], "tool": "max", "hollow": True, "detail": rdetail(b, key)})
        srcs += [a["_src"], b["_src"]]
    big = [p for p in pairs if (p[0].get("M") or 0) > 16]
    small = [p for p in pairs if (p[0].get("M") or 0) <= 16]
    body = ('<p class="pair-key"><span class="sw t-max"></span>AppleM5MatMul (Neural Accelerators) '
            '<span class="sw t-max hollow"></span>matmul_8x8 (simdgroup)</p>')
    rb = [r for r in rows if True]
    big_rows = [r for i, r in enumerate(rb) if (pairs[i // 2][0].get("M") or 0) > 16]
    small_rows = [r for i, r in enumerate(rb) if (pairs[i // 2][0].get("M") or 0) <= 16]
    if big_rows:
        body += '<h4 class="gh">Compute-bound shapes (GFLOP/s)</h4>' + hbars(reg, big_rows, "", aria="NA vs simdgroup")
    if small_rows:
        body += '<h4 class="gh">Decode shapes, M = 1 (effective GB/s)</h4>' + hbars(reg, small_rows, "", aria="NA vs simdgroup decode")
    sp = [a["median"] / b["median"] for a, b in big]
    note = (f"MAX f16. On compute-bound shapes the Neural Accelerators give ×{min(sp):.1f} to ×{max(sp):.1f}. "
            "At M = 1 both are bandwidth-bound, so the accelerators barely matter." if sp else "")
    return fig("Mojo on the GPU: Neural Accelerators against simdgroup matrices", body, srcs, reg, note=note)


def lowp_fig(reg: Reg, at: Atlas) -> str:
    shape = "attn_q_o_M512"
    want = [("max_mma", ("bfloat16",)), ("max_apple_int8_w8a8", None), ("max_apple_fp4_w4a16", None),
            ("mlx_nax", ("bf16",)), ("mlx_nax", ("int8w(g64)xbf16",)), ("mlx_nax", ("int4w(g64)xbf16",))]
    rows, srcs = [], []
    for p, dts in want:
        r = at.best(lambda r: r.get("shape") == shape and r["_path"] == p and (dts is None or r["dtype"] in dts))
        if r:
            rows.append({"label": PATH_LABEL[p], "sub": str(r["dtype"]), "value": r["median"], "src": r["_src"],
                         "tool": r["_tool"], "detail": rdetail(r)})
            srcs.append(r["_src"])
    if not rows:
        return fig("Low precision", placeholder(f"No <code>{shape}</code> records yet."), [], reg)
    return fig("int8 and 4-bit on the GPU, prefill shape 512×3072×3072 (GFLOP/s or GOP/s)",
               legend({r["tool"] for r in rows}) + hbars(reg, rows, "", aria="Low precision"), srcs, reg,
               note="MAX’s int8 kernel is true W8A8 (int8 × int8 → int32), about twice the 16-bit rate. MAX "
                    "NVFP4 and MLX int4/int8 store low-bit weights but compute in bf16, so their gain is in "
                    "decode bandwidth, not prefill compute.")


def tile_fig(reg: Reg, at: Atlas, fpga: dict) -> str:
    perf = fpga.get("perf") or []
    psrc = fpga.get("perf_src")
    parts = []
    # stage ladder for Goal 1
    cos, tests = fpga.get("cosim"), fpga.get("tests")
    ph = fpga.get("phantom")
    mo = fpga.get("maxop")
    t2 = fpga.get("tile2")
    synth = fpga.get("synth") or []
    stages = [
        ("G1-A", "Cycle-accurate simulator in Mojo", "pass" if tests else "not-started",
         (f'{reg.num(tests["pass"], tests["_src"], 0)}/{tests["total"]} tests bit-exact' if tests else "")),
        ("G1-B", "Verilog from Mojo comptime, Verilator co-sim",
         "pass" if cos else "not-started",
         (f'{reg.num(cos["traces"], cos["_src"], 0)} traces, {reg.num(cos["cycles"], cos["_src"], 0)} cycles, '
          'no mismatch' if cos else "")),
        ("G1-C", "Phantom NPU: compiler + HAL plugin", (ph or {}).get("status", "not-started"),
         esc((ph or {}).get("summary", ""))),
        ("G1-D", "MAX custom op routes Llama prefill to the tile", "pass" if mo else "not-started",
         (f'all 112 projections routed: greedy {esc(mo["greedy_match"]["ph_all_vs_fq_all"]["greedy_full_match"])} '
          f'identical to GPU W8A8 (int32 bit-exact), '
          f'{esc(mo["greedy_match"]["ph_all_vs_bf16"]["greedy_full_match"])} vs bf16; projected TTFT '
          f'{reg.num(mo["projected_ttft_ms"]["all:all"]["N16"]["compute_only"] / 1000, mo["_src"], 1, unit=" s")} '
          f'at N=16 vs {reg.num(mo["gpu_ttft_ms"], mo["_src"], 0, unit=" ms")} on the GPU' if mo else "")),
        ("SoC", "RISC-V soft core + tile v2 + Mojo firmware (simulation)", "pass" if t2 else "not-started",
         (f'tile v2 N=8 with buffers + Wishbone, bit- and cycle-exact co-sim; in a LiteX/VexRiscv SoC, '
          f'Mojo bare-metal firmware drives it ({esc(t2["soc_integration"]["end_to_end_8x256x32_cycles"]["speedup"])} '
          f'vs the soft CPU at 8×256×32); DOOM runs beside it (X-DOOM-4)' if t2 else "")),
        ("G1-E", "ECP5 on a board, driven from Mojo", "in-progress" if synth else "not-started",
         "synthesis and place-and-route only, see below" if synth else ""),
    ]
    lad = "".join(f'<li class="s-{esc(s)}"><span class="lid">{i}</span><span class="lname">{esc(nm)}</span>'
                  f'<span class="stt s-{esc(s)}">{esc(s.replace("-", " "))}</span>'
                  f'<span class="lev">{ev}</span></li>' for i, nm, s, ev in stages)
    parts.append(f'<ol class="ladder g1">{lad}</ol>')
    if perf and psrc:
        gpu = at.best(lambda r: r.get("_path") == "max_apple_int8_w8a8" and (r.get("M") or 0) >= 256)
        fm = {s["N"]: s for s in synth if s.get("fmax_mhz") and s.get("variant") in ("dsp", "nodsp")}
        rows = []
        for p in perf:
            ops = 2 * p.get("macs", p["M"] * p["K"] * p["N_out"])
            t100 = p["cycles"] / 100e6
            g100 = ops / t100 / 1e9
            s = fm.get(p["N"])
            tf = p["cycles"] / (s["fmax_mhz"] * 1e6) if s else None
            gf = ops / tf / 1e9 if tf else None
            vs = gpu["median"] / g100 if gpu else None
            rows.append([f'N = {p["N"]} ({p["N"] ** 2} MAC/cycle)', reg.num(p["passes"], psrc, 0),
                         reg.num(p["cycles"], psrc, 0), reg.num(100 * p["pe_util"], psrc, 1, unit="%"),
                         reg.num(t100, psrc, 3, unit=" s"), reg.num(g100, psrc, 1),
                         (reg.num(gf, s["_src"], 1, detail=f'at synthesized fmax {s["fmax_mhz"]:.1f} MHz ({s["device"]})')
                          if gf else '<span class="muted">no fmax yet</span>'),
                         (reg.num(vs, derived(reg, "tile vs GPU int8", "best MAX int8 W8A8 GOP/s ÷ tile GOP/s at 100 MHz"),
                                  0, unit="×") if vs else "–")])
        parts.append(table(["tile", "passes", "cycles", "PE use", "time @ 100 MHz", "GOP/s @ 100 MHz",
                            "GOP/s @ fmax", "GPU int8 is faster by"], rows, "num"))
        srcs = [psrc] + [s["_src"] for s in synth if s.get("fmax_mhz")] + ([gpu["_src"]] if gpu else [])
    else:
        parts.append(placeholder("No simulator performance data yet."))
        srcs = []
    srcs += [x["_src"] for x in (cos, tests, ph) if x]
    return fig("Our int8 systolic tile: from Mojo simulator toward silicon", "".join(parts), srcs, reg,
               note="One Llama 3.2 3B prefill matmul (M=512, K=3072, N=8192) on an output-stationary tile. "
                    "Cycle counts follow DESIGN.md §5, which the Mojo simulator matches exactly. The tile is a "
                    "verification vehicle, not a speed-up: it is fed 2N bytes per cycle with no on-chip buffers.")


def synth_fig(reg: Reg, fpga: dict) -> str:
    synth = fpga.get("synth") or []
    if not synth:
        return fig("ECP5 synthesis", placeholder("No synthesis results yet (lane G1-B writes "
                   "<code>fpga/build/synth/</code> or <code>fpga/results/synth_ecp5.json</code>)."), [], reg)
    rows = []
    for s in synth:
        src = s["_src"]
        ssrc = s.get("_stat_src", src)
        dsp = s.get("dsp_util") or {}
        comb = s.get("comb_util") or {}
        rows.append([f'N = {s["N"]}', esc("DSP multipliers" if s["variant"] == "dsp" else "LUT multipliers"),
                     reg.num(s.get("lut4"), ssrc, 0), reg.num(s.get("ff"), ssrc, 0), reg.num(s.get("mult18"), ssrc, 0),
                     esc(s.get("device") or "–"),
                     (reg.num(s["fmax_mhz"], src, 1, unit=" MHz", detail="nextpnr achieved fmax (200 MHz target)")
                      if s.get("fmax_mhz") else '<span class="muted">–</span>'),
                     (f'{reg.num(100 * comb["used"] / comb["available"], src, 0, unit="%")} logic, '
                      f'{reg.num(100 * dsp["used"] / dsp["available"], src, 0, unit="%")} DSP'
                      if comb.get("available") and dsp.get("available") else "–"),
                     esc(s.get("pnr", ""))])
    srcs = [s["_src"] for s in synth] + [s.get("_stat_src") for s in synth]
    return fig("Mojo-generated Verilog on a Lattice ECP5 (yosys + nextpnr)",
               table(["tile", "multipliers", "LUT4", "FF", "MULT18", "device", "fmax", "utilization", "place & route"],
                     rows, "num"), srcs, reg,
               note=("Tools: " + esc(fpga.get("synth_tools", "")) + ". ") +
                    "Lane G1-B is still running; these are intermediate build outputs. Resource counts are for "
                    "the bare tile; fmax is for the tile inside a small I/O wrapper.")


def doom_fig(reg: Reg, doom: list) -> str:
    items, srcs = [], []
    for t in doom:
        src = t["_src"]
        srcs.append(src)
        if t["frame"]:
            frame = f'<img src="{t["frame"]}" alt="Frame captured at tier {esc(t["id"])}" loading="lazy">'
        else:
            frame = f'<div class="frame-ph"><span>{esc(t["frame_note"] or "no frame yet")}</span></div>'
        fps = (reg.num(t["fps"], src, 1, unit=" fps") if t["fps"] is not None and src else
               '<span class="muted">fps: –</span>')
        if t.get("fps_ref") is not None and src:
            fps += f' <small>vs {reg.num(t["fps_ref"], src, 1, unit=" fps")} reference</small>'
        notes = f'<p class="note">{esc(t["notes"])}</p>' if t["notes"] else ""
        items.append(
            f'<li class="rung s-{esc(t["status"])}"><div class="rung-txt"><span class="lid">{esc(t["id"])}</span>'
            f'<h4>{esc(t["name"])}</h4><span class="stt s-{esc(t["status"])}">{esc(t["status"].replace("-", " "))}</span>'
            f'<p>{esc(t["what"])}</p><p class="check">Pass when: {esc(t["check"])}</p>'
            f'<p class="fps">{fps}{(" · " + esc(t["backend"])) if t["backend"] else ""}</p>'
            f'{notes}'
            f'</div>{frame}</li>')
    return fig("Does it run DOOM? How far down our stack the released source runs",
               f'<ol class="doom">{"".join(items)}</ol>', srcs, reg,
               note="Real DOOM via doomgeneric (GPL source) and the Freedoom IWAD; no re-creation. Each tier is "
                    "pass/fail with a captured frame. Results come from <code>doom/results/*.json</code>.")


# ====================================================================================== caveats


def caveats(reg: Reg, at: Atlas, bench, fpga) -> list[str]:
    out = []
    smoke = [m for m, s in at.a["files"].values() if m.get("smoke")]
    if smoke:
        loads = [m.get("load_avg_at_start", [0])[0] for m in smoke]
        out.append(f"All {len(smoke)} atlas files are smoke runs (5 reps, 64 MiB cold set) taken with 1-minute "
                   f"load {min(loads):.1f}–{max(loads):.1f}; the atlas README requires ≤ 1.5 for an "
                   "authoritative run. Rankings may change after the idle-machine run.")
    for r in at.gemm():
        can = canonical_mnk(str(r.get("shape", "")))
        if r["_path"].startswith("coreml") and can and (r.get("M"), r.get("N"), r.get("K")) != can:
            out.append(f"Core ML’s <code>{esc(r['shape'])}</code> record runs M={r.get('M')} (not {can[0]}) as a "
                       f"chain of {r.get('layers', '?')} layers, so it is not the same shape as the other "
                       "toolchains’ rows of that name. It is labelled where it appears.")
            break
    for r in bench:
        p = ROOT / r["src"].path
        if r["runtime"] == "max":
            import json
            try:
                d = json.loads(p.read_text())
                dropped = d.get("settings", {}).get("dropped_optional_fields")
                if dropped:
                    out.append(f"The MAX endpoint run dropped request fields {', '.join(dropped)} after the server "
                               "rejected them, including <code>ignore_eos</code>; the file still reports 128 "
                               "generated tokens per rep, so decode lengths match.")
            except (OSError, ValueError):
                pass
    synth = [s for s in fpga.get("synth", []) if s.get("fmax_mhz")]
    if synth:
        fm = ", ".join(f'N={s["N"]}: {s["fmax_mhz"]:.0f} MHz' for s in synth)
        out.append(f"fpga/README.md projects the tile at 100 MHz; place-and-route so far reaches {fm}, so the "
                   "100 MHz throughput figures are optimistic. The README also says the largest ECP5 has "
                   "“on the order of 150” DSPs and that N=16 roughly fits; synth.sh records 156, and N=16 needs "
                   "256 MULT18X18D, so N=16 is being placed with LUT multipliers instead.")
    out.append("Phase-1 TTFT is measured three ways (llama-bench arithmetic, in-process MLX, HTTP client for MAX "
               "and llama-server), so cross-method TTFT gaps include HTTP and tokenization overhead.")
    out.append("The pipeline diagram is curated from notes/toolchain-brief.md (read from GitHub source, nothing "
               "built from source here); cells tagged “inferred” are not stated in the project notes.")
    return out


def sources_html(reg: Reg, cav: list[str]) -> str:
    rows = "".join(
        f'<tr id="src-{s.id}"><td>[{s.id}]</td><td><code>{esc(s.path)}</code>'
        f'{f"<small>{esc(s.note)}</small>" if s.note else ""}</td><td>{esc(s.date)}</td>'
        f'<td>{reg.badge(s.status)}</td></tr>' for s in sorted(reg.sources.values(), key=lambda s: s.id))
    key = "".join(f"<li>{reg.badge(k)} {esc(v[1])}</li>" for k, v in STATUS.items())
    cv = "".join(f"<li>{c}</li>" for c in cav)
    return (f'<section id="sources" class="part"><h2>Sources and caveats</h2>'
            f'<div class="prose"><h3>Things in the inputs that don’t line up</h3><ul class="cav">{cv}</ul></div>'
            f'<h3>Status key</h3><ul class="skey">{key}</ul>'
            f'<h3>Files</h3><div class="tw"><table class="srcs"><thead><tr><th>#</th><th>file (repo-relative)</th>'
            f'<th>date</th><th>status</th></tr></thead><tbody>{rows}</tbody></table></div>'
            f'<p class="regen">Regenerate: <code>uv run --project report python report/build.py</code> from the '
            f'repo root. Data contract: <code>report/README.md</code>.</p></section>')


# ====================================================================================== main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    reg = Reg()
    bench = loaders.load_bench(reg)
    at = Atlas(loaders.load_atlas(reg))
    fpga = loaders.load_fpga(reg)
    doom = loaders.load_doom(reg)

    head = header(reg, at, bench)
    p1 = part1(reg)
    p2 = part2(reg, at, bench)
    p3 = part3(reg, at, fpga, doom)
    cav = caveats(reg, at, bench, fpga)
    src = sources_html(reg, cav)
    head = head.replace("<!--FRESHNESS-->", freshness_html(reg))

    css = (HERE / "assets" / "style.css").read_text()
    js = (HERE / "assets" / "page.js").read_text()
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<meta name="generator" content="report/build.py">
<title>Mojo/MAX vs Apple Silicon</title>
<style>{css}</style></head>
<body>{head}<main>{p1}{p2}{p3}{src}</main>
<footer class="foot"><p>Generated by <code>report/build.py</code> on {NOW.strftime("%Y-%m-%d %H:%M")}. Works offline; all data
is embedded.</p></footer>
<div id="tip" role="tooltip" hidden></div>
<script id="srcdata" type="application/json">{reg.to_json()}</script>
<script>{js}</script></body></html>"""
    out = Path(args.out).expanduser()
    out.write_text(page)
    print(f"wrote {out} ({len(page.encode()) / 1024:.0f} KB, {len(reg.sources)} sources)")


if __name__ == "__main__":
    main()
