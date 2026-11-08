"""G2-M3 / ROOFLINE M4: speculative-decoding acceptance (alpha) with mlx_lm, greedy (temp 0).

Uses mlx_lm.generate.speculative_generate_step (the generator stream_generate wraps when a
draft_model is given). It yields (token, logprobs, from_draft); per verify step it yields the
n accepted draft tokens (from_draft=True) and then one target token (from_draft=False).
Each step proposes num_draft = min(max_tokens - ntoks, k) draft tokens.

Metrics (pooled = summed over prompts, then divided):
  acc_ratio   = accepted / proposed draft tokens          (ROOFLINE §6 M4: "alpha = accepted/drafted")
  alpha_iid   = accepted / (accepted + rejections)        (MLE of the per-token i.i.d. acceptance the
                roofline speedup model uses, E = (1 - a^(k+1)) / (1 - a); a rejection = a completed
                step that accepted fewer than it proposed)
  E           = tokens emitted per completed verify step  (mean tokens per target forward)
Also checks greedy speculative output == greedy non-speculative output (token by token).

Timing fields are wall-clock on a shared machine: NOT authoritative.

usage: uv run --project /Users/sdawka/Code/llming/mlx python /Users/sdawka/Code/llming/quality/alpha.py \
          --pair llama3b-4bit_1b-4bit [--ks 1 2 3 4] [--max-tokens 192]
"""

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import generate_step, speculative_generate_step
from mlx_lm.models.cache import KVCache
from mlx_lm.utils import load

ROOT = "/Users/sdawka/Code/llming"
M = f"{ROOT}/models"
PAIRS = {
    "llama3b-4bit_1b-4bit": (f"{M}/mlx-Llama-3.2-3B-Instruct-4bit", f"{M}/mlx-Llama-3.2-1B-Instruct-4bit"),
    "llama3b-8bit_1b-4bit": (f"{M}/mlx-Llama-3.2-3B-Instruct-8bit", f"{M}/mlx-Llama-3.2-1B-Instruct-4bit"),
    "qwen4b-8bit_0.6b-8bit": (f"{M}/mlx-Qwen3-4B-8bit", f"{M}/mlx-Qwen3-0.6B-8bit"),
    "qwen4b-8bit_0.6b-4bit": (f"{M}/mlx-Qwen3-4B-8bit", f"{M}/mlx-Qwen3-0.6B-4bit"),
    "qwen4b-4bit_0.6b-4bit": (f"{M}/mlx-Qwen3-4B-4bit", f"{M}/mlx-Qwen3-0.6B-4bit"),
}
# Gemma 3 (models/gemma3/, see quality/README.md). Targets: 8bit, plain 4bit (g64), QAT 4-bit as
# mlx-community converted it (qat4bit, affine g64) and on the exact Q4_0 grid (qatq40grid, g32 + 6-bit embed).
G3 = f"{M}/gemma3"
_G3T = {"8bit": "mlx-gemma-3-4b-it-8bit", "4bit": "mlx-gemma-3-4b-it-4bit", "bf16": "mlx-gemma-3-4b-it-bf16",
        "qat4bit": "mlx-gemma-3-4b-it-qat-4bit", "qatq40grid": "mlx-gemma-3-4b-it-qat-q40grid"}
_G3D = {"1b-4bit": "mlx-gemma-3-1b-it-4bit", "270m-4bit": "mlx-gemma-3-270m-it-4bit",
        "1bqat-4bit": "mlx-gemma-3-1b-it-qat-4bit", "270mqat-4bit": "mlx-gemma-3-270m-it-qat-4bit",
        "1bqat-q40grid": "mlx-gemma-3-1b-it-qat-q40grid", "270m-bf16": "mlx-gemma-3-270m-it-bf16",
        "1b-8bit": "mlx-gemma-3-1b-it-8bit", "270m-8bit": "mlx-gemma-3-270m-it-8bit"}
for _t, _tp in _G3T.items():
    for _d, _dp in _G3D.items():
        PAIRS[f"gemma4b-{_t}_{_d}"] = (f"{G3}/{_tp}", f"{G3}/{_dp}")

BENCH = open(f"{ROOT}/bench/prompt_512.txt").read()

CHAT = {
    "prose": "Write a short story (about 300 words) about a lighthouse keeper who discovers that the light has started signalling back to someone out at sea.",
    "code": "Write a Python function that parses an Apache combined log file and returns the top 10 IP addresses by request count and the top 10 requested paths. Include type hints, a docstring, and a small usage example.",
    "list": "List 15 practical tips for reducing household energy use, one line each, grouped under headings for heating, lighting, appliances, and habits.",
    "qa": "Explain why the sky is blue and why sunsets are red. Then answer: why isn't the sky violet, given that violet light scatters even more than blue?",
    "summarize": "Summarize the following text in 5 bullet points, then give a one-sentence takeaway.\n\n" + BENCH,
    "howto": "How does a hash map work internally? Cover hashing, buckets, collision resolution (chaining vs open addressing), load factor and resizing, and the time complexity of each operation.",
    "email": "Draft a polite but firm email to a contractor who is three weeks late finishing a kitchen renovation. Ask for a revised schedule and mention the penalty clause in the contract.",
    "sql": "Given tables customers(id, name, country) and orders(id, customer_id, amount, created_at), write SQL queries for: (1) total revenue per country in 2024, (2) the top 5 customers by lifetime spend, (3) customers with no orders. Explain each query briefly.",
}


def build_prompts(tok, family):
    prompts = {}
    # raw bench prompt exactly as the phase-1 bench uses it (no chat template; BOS added if the tokenizer has one)
    prompts["raw_bench512"] = tok.encode(BENCH)
    for name, msg in CHAT.items():
        kw = {"enable_thinking": False} if family == "qwen" else {}
        prompts[name] = tok.apply_chat_template(
            [{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=True, **kw
        )
        if hasattr(prompts[name], "input_ids"):
            prompts[name] = prompts[name]["input_ids"]
    return {k: list(v) for k, v in prompts.items()}


def run_plain(model, prompt, max_tokens, eos):
    toks = []
    t0 = time.perf_counter()
    for tok, _ in generate_step(mx.array(prompt), model, max_tokens=max_tokens):
        tok = int(tok)
        toks.append(tok)
        if tok in eos:
            break
    return toks, time.perf_counter() - t0


# mlx_lm 0.32.2 limitation (Gemma 3): sliding-window layers use RotatingKVCache, and trim_prompt_cache() is a
# silent no-op once any layer of a cache has more than `sliding_window` tokens (is_trimmable() is False). The
# target cache is checked only once, at the start; the draft cache never is. Gemma 3 1B/270M have a 512-token
# window, so on prompts longer than ~512 tokens rejected draft tokens are never rewound out of the draft's
# cache and acceptance collapses as k grows (target output stays correct). With FULL_KV_CACHE, every layer gets
# a plain KVCache: create_attention_mask(..., window_size) then builds the exact sliding-window causal mask
# from the cache offset, so attention is identical and the cache is always trimmable.
FULL_KV_CACHE = True


def spec_cache(model, draft):
    if not FULL_KV_CACHE:
        return None
    return [KVCache() for _ in model.layers] + [KVCache() for _ in draft.layers]


def run_spec(model, draft, prompt, k, max_tokens, eos):
    seq = []
    t0 = time.perf_counter()
    for tok, _, fd in speculative_generate_step(
        mx.array(prompt), model, draft, num_draft_tokens=k, max_tokens=max_tokens,
        prompt_cache=spec_cache(model, draft),
    ):
        tok = int(tok)
        seq.append((tok, bool(fd)))
        if tok in eos:
            break
    return seq, time.perf_counter() - t0


def account(seq, k, max_tokens):
    """Rebuild verify steps from the from_draft flags (mirrors speculative_generate_step)."""
    i = ntoks = 0
    steps = []  # (proposed, accepted, completed)
    while i < len(seq):
        num_draft = min(max_tokens - ntoks, k)
        n = 0
        while n < num_draft and i < len(seq) and seq[i][1]:
            n += 1
            i += 1
        ntoks += n
        if i < len(seq) and not seq[i][1]:
            i += 1
            ntoks += 1
            steps.append((num_draft, n, True))
        else:
            steps.append((num_draft, n, False))  # stream ended (EOS / max_tokens) inside the step
            break
    proposed = sum(s[0] for s in steps)
    accepted = sum(s[1] for s in steps)
    complete = [s for s in steps if s[2]]
    rejections = sum(1 for s in complete if s[1] < s[0])
    return {
        "tokens": len(seq),
        "steps": len(steps),
        "completed_steps": len(complete),
        "proposed": proposed,
        "accepted": accepted,
        "rejections": rejections,
        "tokens_in_completed_steps": sum(s[1] + 1 for s in complete),
        "accept_hist": [sum(1 for s in complete if s[1] == j and s[0] == k) for j in range(k + 1)],
    }


def summarize(c):
    return {
        "acc_ratio": c["accepted"] / c["proposed"] if c["proposed"] else None,
        "alpha_iid": c["accepted"] / (c["accepted"] + c["rejections"]) if c["accepted"] + c["rejections"] else None,
        "E_tokens_per_step": c["tokens_in_completed_steps"] / c["completed_steps"] if c["completed_steps"] else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=list(PAIRS))
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--out", default=f"{ROOT}/quality/results/alpha_raw")
    ap.add_argument("--stock-cache", action="store_true",
                    help="use mlx_lm's default (rotating) caches for spec decoding, i.e. reproduce the limitation")
    a = ap.parse_args()
    global FULL_KV_CACHE
    FULL_KV_CACHE = not a.stock_cache
    tpath, dpath = PAIRS[a.pair]
    family = "qwen" if "qwen" in a.pair else ("gemma" if "gemma" in a.pair else "llama")
    model, tok = load(tpath)
    draft, dtok = load(dpath)
    assert tok.get_vocab() == dtok.get_vocab(), "draft/target vocab mismatch"
    eos = set(tok.eos_token_ids)
    if family == "gemma":
        eos |= {1}  # Gemma 3 generation_config: eos_token_id [1 <eos>, 106 <end_of_turn>]
    prompts = build_prompts(tok, family)

    # warmup (compile kernels) — not recorded
    run_spec(model, draft, prompts["qa"], 2, 16, eos)
    run_plain(model, prompts["qa"], 16, eos)

    res = {"pair": a.pair, "target": tpath, "draft": dpath, "max_tokens": a.max_tokens,
           "spec_cache": "full KVCache per layer (exact sliding-window masks)" if FULL_KV_CACHE else "mlx_lm default",
           "prompts": {}, "by_k": {}}
    for name, p in prompts.items():
        ref, t_ref = run_plain(model, p, a.max_tokens, eos)
        entry = {"prompt_tokens": len(p), "greedy_tokens": len(ref), "plain_wall_s": t_ref,
                 "greedy_text_head": tok.decode(ref[:40]), "k": {}}
        for k in a.ks:
            seq, t = run_spec(model, draft, p, k, a.max_tokens, eos)
            toks = [s[0] for s in seq]
            mism = next((j for j in range(min(len(toks), len(ref))) if toks[j] != ref[j]), None)
            if mism is None and len(toks) != len(ref):
                mism = min(len(toks), len(ref))
            c = account(seq, k, a.max_tokens)
            c.update(summarize(c))
            c.update({"greedy_equal": mism is None, "first_mismatch_index": mism, "spec_wall_s": t})
            entry["k"][k] = c
            print(f"{a.pair} {name:13s} k={k} acc={c['acc_ratio']:.3f} a_iid={c['alpha_iid'] or 0:.3f} "
                  f"E={c['E_tokens_per_step'] or 0:.2f} eq={mism is None} ({mism}) "
                  f"[non-auth: {len(toks)/t:.1f} vs {len(ref)/t_ref:.1f} tok/s]", flush=True)
        res["prompts"][name] = entry

    for k in a.ks:
        tot = {f: sum(res["prompts"][n]["k"][k][f] for n in prompts)
               for f in ["tokens", "steps", "completed_steps", "proposed", "accepted", "rejections",
                         "tokens_in_completed_steps"]}
        tot.update(summarize(tot))
        per = [res["prompts"][n]["k"][k]["acc_ratio"] for n in prompts]
        tot["acc_ratio_range"] = [min(per), max(per)]
        per_a = [res["prompts"][n]["k"][k]["alpha_iid"] for n in prompts if res["prompts"][n]["k"][k]["alpha_iid"] is not None]
        tot["alpha_iid_range"] = [min(per_a), max(per_a)]
        tot["greedy_equal_prompts"] = sum(res["prompts"][n]["k"][k]["greedy_equal"] for n in prompts)
        tot["n_prompts"] = len(prompts)
        res["by_k"][k] = tot
        print(f"POOLED {a.pair} k={k}: acc={tot['acc_ratio']:.3f} a_iid={tot['alpha_iid']:.3f} "
              f"E={tot['E_tokens_per_step']:.2f} greedy_equal={tot['greedy_equal_prompts']}/{len(prompts)}")

    res["meta"] = {"timestamp": datetime.now(timezone.utc).isoformat(), "authoritative_quality": True,
                   "timing_authoritative": False, "machine": platform.platform(),
                   "versions": {"mlx": mx.__version__, "mlx_lm": mlx_lm.__version__}}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/{a.pair}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
