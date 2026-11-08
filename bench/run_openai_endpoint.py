"""Generic streaming benchmark client for any OpenAI-compatible server
(MAX serve, llama-server, mlx_lm.server, vLLM, ...).

Run from the mlx uv project (it has requests + transformers):
    uv run --project /Users/sdawka/Code/llming/mlx python \
        /Users/sdawka/Code/llming/bench/run_openai_endpoint.py \
        --base-url http://localhost:8000/v1 --model <served-model-name> \
        --label max --format bf16

Methodology (kept consistent with run_mlx.py):
  * Prompt: bench/prompt_512.txt sent RAW to /v1/completions (no chat template),
    so the server tokenizes it to the same ~512 tokens (+BOS) as the other
    runtimes. Use --api chat to go through /v1/chat/completions instead (adds the
    chat-template tokens; reported prompt_tokens will be higher).
  * Greedy: temperature=0, top_p=1, seed=0. Asks the server to ignore EOS
    (ignore_eos=true, min_tokens=max_tokens) so exactly max_tokens are generated;
    servers that do not honour these may stop early -> check gen_tokens in output.
  * Prefix caching is defeated by prepending a random 3-digit number + space to
    each request (+1 token, so prompt_tokens = 513 incl. BOS) and sending
    cache_prompt=false (llama-server). Disable with --no-cache-bust.
  * Metric definitions follow `max benchmark` (benchmark-model skill, same as
    vLLM's benchmark_serving), so numbers line up with MAX's own tool:
      TTFT  = request send -> first non-empty streamed chunk (client-side,
              includes HTTP + tokenization + prefill).
      TPOT  = (e2e_latency - TTFT) / (completion_tokens - 1).
      ITL   = gaps between consecutive streamed chunks (mean / p50 / p99).
      per-request output throughput = completion_tokens / e2e_latency
              (`mean_output_throughput` at concurrency 1).
    Plus, for comparison with llama-bench / run_mlx.py:
      tg tok/s = 1 / TPOT  (pure decode rate, first token excluded)
      pp tok/s (approx) = prompt_tokens / TTFT.
    Single stream only (concurrency 1). Run sequentially; --reps is the
    equivalent of --num-prompts with --max-concurrency 1.
  * completion_tokens comes from the server's usage block when it sends one
    (stream_options.include_usage), else by re-tokenizing the streamed text.
  * 1 warmup + --reps measured reps. Peak memory is not observable from the
    client; pass --peak-mem-gb if measured externally (e.g. via `footprint`/
    Activity Monitor) to have it included in the summary.
"""

import argparse
import json
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OPTIONAL_KEYS = ["cache_prompt", "ignore_eos", "min_tokens", "seed", "stream_options"]
TOKENIZER_DIR = ROOT / "models/Llama-3.2-3B-Instruct"


def pct(xs, q):
    """Linear-interpolated percentile (numpy's default method)."""
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def stream_request(url, payload, timeout):
    t_send = time.perf_counter()
    token_times, chunks, usage = [], [], None
    with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                piece = ch.get("text")
                if piece is None:
                    piece = (ch.get("delta") or {}).get("content")
                if piece:
                    token_times.append(time.perf_counter())
                    chunks.append(piece)
    return t_send, token_times, "".join(chunks), usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="e.g. http://localhost:8000/v1")
    ap.add_argument("--model", required=True, help="model name the server expects")
    ap.add_argument("--label", default="openai-endpoint", help="runtime label, e.g. max")
    ap.add_argument("--format", default="unknown", help="weight format label, e.g. bf16 / q4")
    ap.add_argument("--api", choices=["completions", "chat"], default="completions")
    ap.add_argument("--prompt-file", default=str(ROOT / "bench/prompt_512.txt"))
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--no-ignore-eos", action="store_true", help="don't send ignore_eos/min_tokens")
    ap.add_argument("--no-cache-bust", action="store_true",
                    help="don't prepend a per-request nonce (prefix caching would hide prefill cost)")
    ap.add_argument("--peak-mem-gb", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    text = Path(args.prompt_file).read_text()
    base = args.base_url.rstrip("/")
    payload = {
        "cache_prompt": False,  # llama-server; ignored by others
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if not args.no_ignore_eos:
        payload["ignore_eos"] = True
        payload["min_tokens"] = args.max_tokens
    if args.api == "completions":
        url = f"{base}/completions"
        payload["prompt"] = text
    else:
        url = f"{base}/chat/completions"
        payload["messages"] = [{"role": "user", "content": text}]

    tok = None
    dropped = []

    def count_tokens(s):
        nonlocal tok
        if tok is None:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
        return len(tok.encode(s, add_special_tokens=False))

    def one():
        if not args.no_cache_bust:
            # unique leading token per request so server prefix/KV caches can't skip prefill
            nonce = f"{random.randint(100, 999)} "
            if args.api == "completions":
                payload["prompt"] = nonce + text
            else:
                payload["messages"] = [{"role": "user", "content": nonce + text}]
        try:
            t_send, times, out_text, usage = stream_request(url, payload, args.timeout)
        except requests.HTTPError as e:
            extras = [k for k in OPTIONAL_KEYS if k in payload]
            if e.response is None or e.response.status_code not in (400, 422) or not extras:
                raise
            print(f"server rejected request ({e.response.status_code}: {e.response.text[:200]}); "
                  f"retrying without optional fields {extras}")
            for k in extras:
                payload.pop(k)
            dropped.extend(extras)
            t_send, times, out_text, usage = stream_request(url, payload, args.timeout)
        if not times:
            raise RuntimeError("no tokens streamed back")
        n_gen = (usage or {}).get("completion_tokens") or count_tokens(out_text)
        n_prompt = (usage or {}).get("prompt_tokens") or count_tokens(text) + 1
        ttft = times[0] - t_send
        dec = times[-1] - times[0]
        e2e = times[-1] - t_send
        itl = [b - a for a, b in zip(times, times[1:])]
        tpot = dec / (n_gen - 1) if n_gen > 1 else None
        return {
            "prompt_tokens": n_prompt,
            "gen_tokens": n_gen,
            "stream_chunks": len(times),
            "usage_from_server": usage is not None,
            "ttft_s": ttft,
            "pp_tps": n_prompt / ttft,
            "tg_tps": 1 / tpot if tpot else None,
            "tpot_ms": 1000 * tpot if tpot else None,
            "itl_ms": [1000 * x for x in itl],
            "output_tps": n_gen / e2e,
            "e2e_s": e2e,
            "text_head": out_text[:80],
        }

    for _ in range(args.warmup):
        one()
    reps = [one() for _ in range(args.reps)]
    tg = [x["tg_tps"] for x in reps if x["tg_tps"]]
    all_itl = [v for x in reps for v in x["itl_ms"]]
    r = {
        "format": args.format,
        "model_path": args.model,
        "prompt_tokens": reps[0]["prompt_tokens"],
        "gen_tokens": reps[0]["gen_tokens"],
        "pp_tps_mean": statistics.mean(x["pp_tps"] for x in reps),
        "tg_tps_mean": statistics.mean(tg) if tg else None,
        "ttft_s_mean": statistics.mean(x["ttft_s"] for x in reps),
        "ttft_s_std": statistics.pstdev(x["ttft_s"] for x in reps),
        "tg_tps_std": statistics.pstdev(tg) if tg else None,
        "ttft_ms_p50": pct([x["ttft_s"] * 1000 for x in reps], 50),
        "ttft_ms_p99": pct([x["ttft_s"] * 1000 for x in reps], 99),
        "tpot_ms_mean": statistics.mean(x["tpot_ms"] for x in reps if x["tpot_ms"]),
        "itl_ms_mean": statistics.mean(all_itl) if all_itl else None,
        "itl_ms_p50": pct(all_itl, 50),
        "itl_ms_p99": pct(all_itl, 99),
        "mean_output_throughput": statistics.mean(x["output_tps"] for x in reps),
        "peak_mem_gb": args.peak_mem_gb,
        "reps": reps,
    }
    out = {
        "runtime": args.label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": url,
        "settings": {
            "api": args.api,
            "sampling": "greedy (temperature=0, top_p=1, seed=0)",
            "ignore_eos": not args.no_ignore_eos,
            "cache_bust_nonce": not args.no_cache_bust,
            "dropped_optional_fields": dropped,
            "max_tokens": args.max_tokens,
            "reps": args.reps,
            "warmup": args.warmup,
            "prompt_file": args.prompt_file,
        },
        "results": [r],
    }
    print(
        f"[{args.label} {args.format}] ttft {r['ttft_s_mean']*1000:.0f} ms (pp~{r['pp_tps_mean']:.0f} tok/s)"
        f" | TPOT {r['tpot_ms_mean']:.2f} ms (tg {r['tg_tps_mean'] or float('nan'):.1f} tok/s)"
        f" | ITL p50/p99 {r['itl_ms_p50']:.2f}/{r['itl_ms_p99']:.2f} ms"
        f" | out {r['mean_output_throughput']:.1f} tok/s | gen_tokens {r['gen_tokens']}"
    )
    if r["gen_tokens"] < args.max_tokens:
        print(f"WARNING: server generated {r['gen_tokens']} < {args.max_tokens} tokens (EOS not ignored?)")
    dest = Path(args.out) if args.out else ROOT / "bench/results" / f"{args.label}_{args.format}_{datetime.now():%Y%m%d_%H%M%S}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
