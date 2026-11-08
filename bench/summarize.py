"""Merge bench/results/*.json into one markdown table.

    python3 /Users/sdawka/Code/llming/bench/summarize.py [results_dir] [-o summary.md]

Handles three result shapes:
  * llama-bench JSON (a list of tests) + sibling *.time.txt from /usr/bin/time -l
  * run_mlx.py JSON          ({"runtime": "mlx", "results": [...]})
  * run_openai_endpoint.py   ({"runtime": <label>, "results": [...]})
  * `max benchmark --result-filename` JSON (keys like mean_ttft_ms, mean_tpot_ms)
If several files exist for the same (runtime, format), the newest wins.
TTFT for llama.cpp is derived as 512 / pp tok/s (pure prefill, no HTTP).
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def fmt(v, spec):
    return "-" if v is None else format(v, spec)


def parse_llamacpp(path, data):
    fmt_name = re.match(r"llamacpp_(.+?)_\d{8}_\d{6}", path.stem)
    fmt_name = fmt_name.group(1) if fmt_name else path.stem
    pp = next((t for t in data if t["n_gen"] == 0), None)
    tg = next((t for t in data if t["n_prompt"] == 0), None)
    peak = None
    tfile = path.with_suffix(".time.txt")
    if tfile.exists():
        txt = tfile.read_text()
        # max RSS includes the mmapped weights that Metal wraps as no-copy shared
        # buffers; "peak memory footprint" excludes them, so it is not used.
        m = re.search(r"(\d+)\s+maximum resident set size", txt)
        if m:
            peak = int(m.group(1)) / 1e9
    ref = pp or tg
    return {
        "runtime": "llama.cpp",
        "format": fmt_name,
        "pp_tps": pp["avg_ts"] if pp else None,
        "tg_tps": tg["avg_ts"] if tg else None,
        "ttft_ms": 1000 * pp["n_prompt"] / pp["avg_ts"] if pp else None,
        "tpot_ms": 1000 / tg["avg_ts"] if tg else None,
        "peak_mem_gb": peak,
        "version": f"{ref.get('build_commit')} ({ref.get('backends')})" if ref else "",
        "file": path.name,
        "mtime": path.stat().st_mtime,
    }


def parse_max_benchmark(path, data):
    """Result JSON from `max benchmark --result-filename ...` (vLLM-style keys)."""
    meta = data.get("metadata") or data  # --metadata may be nested or merged top-level
    tpot = data.get("mean_tpot_ms")
    ttft = data.get("mean_ttft_ms")
    n_in, n_req = data.get("total_input_tokens"), data.get("completed") or data.get("num_prompts")
    pp = (n_in / n_req) / (ttft / 1000) if n_in and n_req and ttft else None
    return {
        "runtime": meta.get("runtime", "max benchmark"),
        "format": meta.get("format", path.stem),
        "pp_tps": pp,
        "tg_tps": 1000 / tpot if tpot else None,
        "tpot_ms": tpot,
        "ttft_ms": ttft,
        "peak_mem_gb": None,
        "version": meta.get("build", "max benchmark"),
        "file": path.name,
        "mtime": path.stat().st_mtime,
    }


def parse_generic(path, data):
    rows = []
    ver = data.get("versions") or {}
    for r in data["results"]:
        rows.append(
            {
                "runtime": data["runtime"],
                "format": r["format"],
                "pp_tps": r.get("pp_tps_mean"),
                "tg_tps": r.get("tg_tps_mean"),
                "tpot_ms": r.get("tpot_ms_mean") or (1000 / r["tg_tps_mean"] if r.get("tg_tps_mean") else None),
                "ttft_ms": 1000 * r["ttft_s_mean"] if r.get("ttft_s_mean") is not None else None,
                "peak_mem_gb": r.get("peak_mem_gb"),
                "version": " ".join(f"{k} {v}" for k, v in ver.items()) or data.get("endpoint", ""),
                "file": path.name,
                "mtime": path.stat().st_mtime,
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default=str(ROOT / "results"))
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    rows = {}
    for p in sorted(Path(args.results_dir).glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            print(f"skip (bad json): {p.name}")
            continue
        if isinstance(data, list):
            parsed = parse_llamacpp(p, data)
        elif "mean_ttft_ms" in data:
            parsed = parse_max_benchmark(p, data)
        elif "results" in data:
            parsed = parse_generic(p, data)
        else:
            print(f"skip (unknown shape): {p.name}")
            continue
        for r in parsed if isinstance(parsed, list) else [parsed]:
            key = (r["runtime"], r["format"])
            if key not in rows or r["mtime"] > rows[key]["mtime"]:
                rows[key] = r

    lines = [
        "| runtime | format | pp tok/s | tg tok/s | TPOT (ms) | TTFT (ms) | peak mem (GB) | version | source |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in sorted(rows.values(), key=lambda r: (r["runtime"], r["format"])):
        lines.append(
            f"| {r['runtime']} | {r['format']} | {fmt(r['pp_tps'], '.1f')} | {fmt(r['tg_tps'], '.1f')}"
            f" | {fmt(r['tpot_ms'], '.2f')} | {fmt(r['ttft_ms'], '.0f')} | {fmt(r['peak_mem_gb'], '.2f')} | {r['version']} | {r['file']} |"
        )
    lines += [
        "",
        "Notes: llama.cpp pp/tg are llama-bench pp512/tg128 (synthetic tokens); its TTFT = 512/pp.",
        "MLX TTFT = in-process prefill + first token. OpenAI-endpoint TTFT is client-side over HTTP.",
        "tg tok/s = 1000 / TPOT everywhere (TPOT as defined by `max benchmark`: decode time per token after the first).",
        "`max benchmark` rows: pp tok/s ~ mean input tokens / mean TTFT; stamp them with --metadata runtime=... format=...",
        "All decode is greedy; EOS ignored; 128 generated tokens; prompt = bench/prompt_512.txt.",
    ]
    md = "\n".join(lines)
    print(md)
    if args.out:
        Path(args.out).write_text(md + "\n")


if __name__ == "__main__":
    main()
