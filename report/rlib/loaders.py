"""Read every result source the page uses. Each loader tolerates missing / partial data and
returns plain dicts whose numbers carry a `Source` (see sources.py). The data contract is
documented in report/README.md - keep the two in sync."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

from .sources import ROOT, Reg

# ----------------------------------------------------------------------------------------- helpers


def _json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _status(meta: dict, default="measured") -> str:
    if not isinstance(meta, dict):
        return default
    if meta.get("smoke"):
        return "smoke"
    if meta.get("authoritative") is True:
        return "authoritative"
    if meta.get("authoritative") is False:
        return "nonidle"
    return default


# -------------------------------------------------------------------------------- phase-1 LLM bench

RUNTIME_INFO = {  # runtime label in JSON -> (toolchain key, display label, method)
    "llama.cpp": ("llama", "llama.cpp", "llama-bench, in-process"),
    "llamacpp-server": ("llama", "llama.cpp server", "HTTP, bench client"),
    "max": ("max", "MAX serve", "HTTP, bench client"),
    "max-benchmark": ("max", "MAX serve", "HTTP, max benchmark"),
    "mlx": ("mlx", "MLX (mlx-lm)", "in-process"),
}


def fmt_class(f: str) -> str:
    f = f.lower()
    if any(t in f for t in ("bf16", "f16", "fp16")):
        return "16-bit"
    if any(t in f for t in ("q8", "int8", "8bit", "w8")):
        return "8-bit"
    if any(t in f for t in ("q4", "4bit", "int4", "nvfp4", "w4")):
        return "4-bit"
    return f


def load_bench(reg: Reg) -> list[dict]:
    rows: dict = {}
    d = ROOT / "bench" / "results"
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        data = _json(p)
        if data is None:
            continue
        parsed = []
        if isinstance(data, list) and data and isinstance(data[0], dict) and "avg_ts" in data[0]:
            m = re.match(r"llamacpp_(.+?)_\d{8}_\d{6}", p.stem)
            fmt = m.group(1) if m else p.stem
            pp = next((t for t in data if t.get("n_gen") == 0), None)
            tg = next((t for t in data if t.get("n_prompt") == 0), None)
            ref = pp or tg or {}
            src = reg.add(p, ref.get("test_time"), "measured")
            peak = None
            tfile = p.with_suffix(".time.txt")
            if tfile.exists():
                mm = re.search(r"(\d+)\s+maximum resident set size", tfile.read_text())
                peak = int(mm.group(1)) / 1e9 if mm else None
            parsed.append({
                "runtime": "llama.cpp", "format": fmt,
                "pp": pp["avg_ts"] if pp else None, "pp_std": pp.get("stddev_ts") if pp else None,
                "tg": tg["avg_ts"] if tg else None, "tg_std": tg.get("stddev_ts") if tg else None,
                "ttft": 1000 * pp["n_prompt"] / pp["avg_ts"] if pp else None, "ttft_std": None,
                "peak": peak, "version": f"{ref.get('build_commit')} ({ref.get('backends')})",
                "model_bytes": ref.get("model_size"), "reps": len(pp.get("samples_ts", [])) if pp else None,
                "src": src, "ttft_note": "TTFT = 512 / pp (no HTTP)",
            })
        elif isinstance(data, dict) and "mean_ttft_ms" in data:
            meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else data
            src = reg.add(p, data.get("date"), _status(data))
            tpot, ttft = data.get("mean_tpot_ms"), data.get("mean_ttft_ms")
            n_in, n_req = data.get("total_input_tokens"), data.get("completed") or data.get("num_prompts")
            parsed.append({
                "runtime": meta.get("runtime", "max-benchmark"), "format": meta.get("format", p.stem),
                "pp": (n_in / n_req) / (ttft / 1000) if n_in and n_req and ttft else None, "pp_std": None,
                "tg": 1000 / tpot if tpot else None, "tg_std": None, "ttft": ttft, "ttft_std": None,
                "peak": None, "version": "max benchmark", "reps": n_req, "src": src,
                "ttft_note": "client-side over HTTP",
            })
        elif isinstance(data, dict) and isinstance(data.get("results"), list):
            src = reg.add(p, data.get("timestamp"), _status(data))
            ver = data.get("versions") or {}
            for r in data["results"]:
                tg = r.get("tg_tps_mean")
                parsed.append({
                    "runtime": data.get("runtime", p.stem), "format": r.get("format", "?"),
                    "pp": r.get("pp_tps_mean"), "pp_std": r.get("pp_tps_std"),
                    "tg": tg, "tg_std": r.get("tg_tps_std"),
                    "ttft": 1000 * r["ttft_s_mean"] if r.get("ttft_s_mean") is not None else None,
                    "ttft_std": 1000 * r["ttft_s_std"] if r.get("ttft_s_std") is not None else None,
                    "peak": r.get("peak_mem_gb"),
                    "version": " ".join(f"{k} {v}" for k, v in ver.items()) or data.get("endpoint", ""),
                    "reps": len(r.get("reps", [])) or None, "src": src,
                    "ttft_note": "in-process" if data.get("runtime") == "mlx" else "client-side over HTTP",
                })
        for r in parsed:
            info = RUNTIME_INFO.get(r["runtime"], ("other", r["runtime"], ""))
            r["tool"], r["label"], r["method"] = info
            r["fclass"] = fmt_class(r["format"])
            r["_t"] = r["src"].date
            key = (r["runtime"], r["format"])
            if key not in rows or r["_t"] > rows[key]["_t"]:
                rows[key] = r
    order = {"mlx": 0, "max": 1, "llama": 4}
    return sorted(rows.values(), key=lambda r: (r["fclass"], order.get(r["tool"], 9), r["runtime"]))


# ----------------------------------------------------------------------------------------- atlas


def atlas_path(r: dict) -> str:
    e, k = r.get("engine", ""), r.get("kernel", "")
    if e == "gpu_mlx":
        return {"default": "mlx_nax", "nonax_g16g": "mlx_nonax", "notf32": "mlx_notf32"}.get(r.get("mode"), "mlx")
    if e == "gpu_max":
        return {"apple_m5_mma": "max_mma", "simdgroup_8x8": "max_8x8"}.get(k, "max_" + k)
    if e.startswith("accelerate") or e.startswith("bnns"):
        return "sme"
    if e.startswith("coreml_"):
        return "coreml_" + e.removeprefix("coreml_")
    return e


def path_tool(path: str) -> str:
    if path.startswith("mlx"):
        return "mlx"
    if path.startswith("max") or path.startswith("gpu_mojo") or path.startswith("cpu_mojo") or path == "mojo":
        return "max"
    if path.startswith("coreml"):
        return "coreml"
    if path in ("sme",) or path.startswith("cpu"):
        return "accel"
    if path.startswith("ggml"):
        return "llama"
    return "other"


def load_atlas(reg: Reg) -> dict:
    res = ROOT / "atlas" / "results"
    out = {"records": [], "files": {}, "inventory": None, "inv_src": None}
    if not res.exists():
        return out
    files = {}
    for p in sorted(res.glob("*.json")):
        if p.name == "inventory.json":
            continue
        base = p.stem.removesuffix("_smoke")
        if p.stem.endswith("_smoke") and base in files and not files[base].stem.endswith("_smoke"):
            continue  # a full run exists: it wins over the smoke file
        files[base] = p
    for base, p in files.items():
        d = _json(p)
        if not isinstance(d, dict) or "records" not in d:
            continue
        m = d.get("meta", {})
        la = m.get("load_avg_at_start")
        note = f"1-min load at start {la[0]:.1f}" if isinstance(la, list) and la else ""
        src = reg.add(p, m.get("timestamp"), _status(m, "nonidle"), note)
        out["files"][base] = (m, src)
        for r in d["records"]:
            if not isinstance(r, dict):
                continue
            r["_src"] = src
            r["_path"] = atlas_path(r)
            r["_tool"] = path_tool(r["_path"])
            out["records"].append(r)
    inv = res / "inventory.json"
    if inv.exists():
        out["inventory"] = _json(inv)
        out["inv_src"] = reg.add(inv, None, "measured")
    return out


# ------------------------------------------------------------------------------------------ fpga

STAT_CELLS = ("LUT4", "CCU2C", "TRELLIS_FF", "MULT18X18D", "PFUMX", "L6MUX21")
ECP5_BY_DSP = {28: "LFE5U-25F", 72: "LFE5U-45F", 156: "LFE5U-85F"}


def _parse_stat(p: Path) -> dict:
    cells = {}
    for line in p.read_text().splitlines():
        m = re.match(r"\s*(\d+)\s+(\S+)\s*$", line)
        if m and m.group(2) in STAT_CELLS:
            cells[m.group(2)] = int(m.group(1))
    return {c: cells.get(c, 0) for c in STAT_CELLS} if cells else {}


def load_fpga(reg: Reg) -> dict:
    fp = ROOT / "fpga"
    out = {"synth": [], "synth_tools": "", "perf": [], "cosim": None, "tests": None, "phantom": None, "maxop": None,
           "perf_src": None, "design_src": None}
    if not fp.exists():
        return out
    rdir = fp / "results"
    # ---- preferred: structured JSON written by the fpga lanes (see README data contract)
    sj = _json(rdir / "synth_ecp5.json") if rdir.exists() else None
    if isinstance(sj, dict) and isinstance(sj.get("records"), list):
        src = reg.add(rdir / "synth_ecp5.json", sj.get("meta", {}).get("timestamp"), _status(sj.get("meta", {})))
        out["synth_tools"] = sj.get("meta", {}).get("tools", "")
        for r in sj["records"]:
            r["_src"] = src
            out["synth"].append(r)
    else:
        sd = fp / "build" / "synth"
        if sd.exists():
            summ = sd / "summary.txt"
            if summ.exists():
                tools = []
                for l in summ.read_text().splitlines():
                    m = re.search(r"(Yosys [\w.+-]+)|(nextpnr-\d[\w.-]*)", l)
                    if l.startswith("#") and m:
                        tools.append(m.group(0))
                out["synth_tools"] = ", ".join(tools)
            for st in sorted(sd.glob("n*.core.stat")):
                tag = st.name.removesuffix(".core.stat")  # n4, n16_nodsp
                m = re.match(r"n(\d+)(_\w+)?", tag)
                if not m:
                    continue
                n, variant = int(m.group(1)), (m.group(2) or "_dsp")[1:]
                cells = _parse_stat(st)
                rep = sd / f"{tag}.report.json"
                pnr_log = sd / f"{tag}.pnr.log"
                rec = {"N": n, "variant": variant, "lut4": cells.get("LUT4"), "ccu2c": cells.get("CCU2C"),
                       "ff": cells.get("TRELLIS_FF"), "mult18": cells.get("MULT18X18D"),
                       "fmax_mhz": None, "device": None, "pnr": "not run"}
                if rep.exists():
                    rj = _json(rep) or {}
                    fm = [v.get("achieved") for v in (rj.get("fmax") or {}).values() if isinstance(v, dict)]
                    rec["fmax_mhz"] = min(fm) if fm else None
                    util = rj.get("utilization", {})
                    dsp_avail = util.get("MULT18X18D", {}).get("available")
                    rec["device"] = ECP5_BY_DSP.get(dsp_avail, f"ECP5 ({dsp_avail} DSP)")
                    rec["comb_util"] = util.get("TRELLIS_COMB", {})
                    rec["dsp_util"] = util.get("MULT18X18D", {})
                    rec["pnr"] = "placed and routed"
                    rec["_src"] = reg.add(rep, None, "pending",
                                          "nextpnr-ecp5 --report; written by fpga/sim/synth.sh (lane G1-B)")
                else:
                    rec["pnr"] = "running or failed (no report yet)" if pnr_log.exists() else "not placed"
                    if n >= 16 and variant == "dsp":
                        rec["pnr"] = "does not fit: needs N² = 256 DSP, largest ECP5 has 156"
                    rec["_src"] = reg.add(st, None, "pending", "yosys synth_ecp5 stat; lane G1-B")
                rec["_stat_src"] = reg.add(st, None, "pending", "yosys synth_ecp5 stat; lane G1-B")
                out["synth"].append(rec)
            out["synth"].sort(key=lambda r: (r["N"], r["variant"]))
    # ---- simulator perf: JSON if present, else the DESIGN.md §5 formula (verified by the sim per README)
    pj = _json(rdir / "perf.json") if rdir.exists() else None
    if isinstance(pj, dict) and isinstance(pj.get("records"), list):
        src = reg.add(rdir / "perf.json", pj.get("meta", {}).get("timestamp"), _status(pj.get("meta", {})))
        out["perf_src"] = src
        out["perf"] = pj["records"]
    else:
        design = fp / "DESIGN.md"
        if design.exists():
            out["perf_src"] = reg.add(design, None, "derived",
                                      "cycles(M,K,N_out;N) = ceil(M/N)·ceil(N_out/N)·(K+3N−1), DESIGN.md §5; "
                                      "fpga/README.md reports the simulator matches it exactly")
            for n in (8, 16):
                M, K, NO = 512, 3072, 8192
                passes = -(-M // n) * -(-NO // n)
                cyc = passes * (K + 3 * n - 1)
                out["perf"].append({"N": n, "M": M, "K": K, "N_out": NO, "passes": passes, "cycles": cyc,
                                    "pe_util": K / (K + 3 * n - 1), "macs": M * K * NO})
    out["design_src"] = out["perf_src"]
    readme = fp / "README.md"
    if readme.exists():
        txt = readme.read_text()
        rsrc = reg.add(readme, None, "doc")
        m = re.search(r"(\d+) traces and ([\d,]+) cycles pass with no mismatch", txt)
        if m:
            out["cosim"] = {"traces": int(m.group(1)), "cycles": int(m.group(2).replace(",", "")), "_src": rsrc}
        m = re.search(r"\((\d+)/(\d+) pass\)", txt) or re.search(r"(\d+)/(\d+) pass", txt)
        if m:
            out["tests"] = {"pass": int(m.group(1)), "total": int(m.group(2)), "_src": rsrc}
    cj = _json(rdir / "cosim.json") if rdir.exists() else None
    if isinstance(cj, dict):
        src = reg.add(rdir / "cosim.json", cj.get("meta", {}).get("timestamp"), _status(cj.get("meta", {})))
        out["cosim"] = {"traces": cj.get("traces"), "cycles": cj.get("cycles"),
                        "mismatches": cj.get("mismatches"), "_src": src}
    ph = _json(rdir / "phantom.json") if rdir.exists() else None
    if isinstance(ph, dict):
        ph["_src"] = reg.add(rdir / "phantom.json", ph.get("meta", {}).get("timestamp"),
                             _status(ph.get("meta", {})))
        out["phantom"] = ph
    mo = _json(rdir / "maxop.json") if rdir.exists() else None
    if isinstance(mo, dict):
        mo["_src"] = reg.add(rdir / "maxop.json", mo.get("date"), _status(mo))
        out["maxop"] = mo
    return out


# ------------------------------------------------------------------------------------------ doom

DOOM_TIERS = [
    ("X-DOOM-1", "Hosted by Mojo",
     "doomgeneric built as a C library; Mojo drives the game loop over FFI and blits frames to a window.",
     "Freedoom E1M1 demo plays; frame captured; fps recorded."),
    ("X-DOOM-2", "Renderer inner loops in Mojo (CPU)",
     "Column and span drawing rewritten in Mojo on the CPU; the rest stays C.",
     "Bit-identical frames vs the C renderer on a recorded demo; fps compared."),
    ("X-DOOM-3", "Frame rendering on the M5 GPU",
     "The column/span pass runs as a Mojo Metal kernel.",
     "Bit-identical frames on the demo; fps compared."),
    ("X-DOOM-4", "On our own hardware path",
     "A soft CPU (e.g. small RISC-V) beside the int8 tile on the ECP5, or a cycle-level sim of that SoC. "
     "The tile alone is not a CPU and cannot run DOOM.",
     "Title screen or demo frame from the soft CPU, verified against the host frame."),
]

MAX_FRAME_BYTES = 400_000


def load_doom(reg: Reg) -> list[dict]:
    rdir = ROOT / "doom" / "results"
    found = {}
    if rdir.exists():
        for p in sorted(rdir.glob("*.json")):
            d = _json(p)
            for r in (d if isinstance(d, list) else [d]):
                if not isinstance(r, dict) or "tier" not in r:
                    continue
                r["_src"] = reg.add(p, r.get("date") or r.get("timestamp"), _status(r))
                prev = found.get(r["tier"])
                if prev is None or r["_src"].date >= prev["_src"].date:
                    found[r["tier"]] = r
    tiers = []
    for tid, name, what, check in DOOM_TIERS:
        r = found.get(tid, {})
        frame_uri, frame_note = None, ""
        if r.get("frame"):
            fp = (rdir / r["frame"]).resolve()
            if fp.exists() and fp.is_relative_to(ROOT):
                b = fp.read_bytes()
                mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                        ".webp": "image/webp", ".gif": "image/gif"}.get(fp.suffix.lower())
                if mime and len(b) <= MAX_FRAME_BYTES:
                    frame_uri = f"data:{mime};base64," + base64.b64encode(b).decode()
                else:
                    frame_note = f"frame {fp.name} skipped (type or size > {MAX_FRAME_BYTES // 1000} KB)"
            else:
                frame_note = f"frame {r['frame']} not found"
        tiers.append({"id": tid, "name": name, "what": what, "check": check,
                      "status": r.get("status", "not-started"), "fps": r.get("fps"),
                      "fps_ref": r.get("fps_reference"), "notes": r.get("notes", ""),
                      "backend": r.get("backend", ""), "frame": frame_uri, "frame_note": frame_note,
                      "_src": r.get("_src")})
    return tiers
