"""MLX baseline benchmark for Llama 3.2 3B Instruct.

Run from the mlx uv project:
    uv run --project /Users/sdawka/Code/llming/mlx python /Users/sdawka/Code/llming/bench/run_mlx.py

Methodology (kept consistent with run_openai_endpoint.py):
  * Prompt: bench/prompt_512.txt, tokenized raw (BOS + text, no chat template),
    so the model sees exactly the same ~512 tokens in every runtime.
  * Greedy decoding (argmax, temperature 0). EOS is ignored so every rep
    generates exactly --gen-tokens tokens.
  * pp tok/s  = prompt_tokens / time-to-first-token (prefill + first sample).
  * tg tok/s  = (gen_tokens - 1) / (t_last_token - t_first_token)
    i.e. pure decode rate, excluding the first token produced by prefill.
  * Peak memory = mx.get_peak_memory() after the format's runs (reset per format).
  * 1 warmup + --reps measured reps per format, fresh KV cache each rep.
"""

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS = {
    "bf16": ROOT / "models/mlx-Llama-3.2-3B-Instruct-bf16",
    "4bit": ROOT / "models/mlx-Llama-3.2-3B-Instruct-4bit",
}


def run_once(model, prompt_ids, gen_tokens, sampler):
    prompt = mx.array(prompt_ids)
    times = []
    t0 = time.perf_counter()
    for n, (tok, _) in enumerate(
        generate_step(prompt, model, max_tokens=gen_tokens, sampler=sampler)
    ):
        # generate_step yields after mx.async_eval; .item() forces the sync
        _ = tok if isinstance(tok, int) else tok.item()
        times.append(time.perf_counter())
        if n + 1 >= gen_tokens:
            break
    ttft = times[0] - t0
    decode_time = times[-1] - times[0]
    return {
        "prompt_tokens": len(prompt_ids),
        "gen_tokens": len(times),
        "ttft_s": ttft,
        "pp_tps": len(prompt_ids) / ttft,
        "tg_tps": (len(times) - 1) / decode_time if decode_time > 0 else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--prompt-file", default=str(ROOT / "bench/prompt_512.txt"))
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    text = Path(args.prompt_file).read_text()
    sampler = make_sampler(temp=0.0)  # greedy / argmax
    out = {
        "runtime": "mlx",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "versions": {"mlx": version("mlx"), "mlx-lm": version("mlx-lm")},
        "machine": platform.platform(),
        "device": str(mx.default_device()),
        "settings": {
            "sampling": "greedy (temp=0), EOS ignored",
            "gen_tokens": args.gen_tokens,
            "reps": args.reps,
            "warmup": args.warmup,
            "prompt_file": args.prompt_file,
        },
        "results": [],
    }

    for fmt in args.formats:
        path = DEFAULT_MODELS.get(fmt, fmt)
        model, tokenizer = load(str(path))
        prompt_ids = tokenizer.encode(text)  # adds BOS (<|begin_of_text|>)
        for _ in range(args.warmup):
            run_once(model, prompt_ids, args.gen_tokens, sampler)
        mx.clear_cache()
        mx.reset_peak_memory()
        reps = [run_once(model, prompt_ids, args.gen_tokens, sampler) for _ in range(args.reps)]
        peak = mx.get_peak_memory() / 1e9
        r = {
            "format": fmt,
            "model_path": str(path),
            "prompt_tokens": reps[0]["prompt_tokens"],
            "gen_tokens": reps[0]["gen_tokens"],
            "pp_tps_mean": statistics.mean(x["pp_tps"] for x in reps),
            "tg_tps_mean": statistics.mean(x["tg_tps"] for x in reps),
            "ttft_s_mean": statistics.mean(x["ttft_s"] for x in reps),
            "pp_tps_std": statistics.pstdev(x["pp_tps"] for x in reps),
            "tg_tps_std": statistics.pstdev(x["tg_tps"] for x in reps),
            "peak_mem_gb": peak,
            "reps": reps,
        }
        print(
            f"[mlx {fmt}] pp {r['pp_tps_mean']:.1f} tok/s | tg {r['tg_tps_mean']:.1f} tok/s"
            f" | ttft {r['ttft_s_mean']*1000:.0f} ms | peak {peak:.2f} GB"
        )
        out["results"].append(r)
        del model
        mx.clear_cache()

    dest = Path(args.out) if args.out else ROOT / "bench/results" / f"mlx_{datetime.now():%Y%m%d_%H%M%S}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
