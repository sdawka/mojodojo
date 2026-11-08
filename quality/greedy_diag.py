"""Diagnose greedy speculative vs non-speculative mismatches found by alpha.py.

For each mismatching (pair, prompt) at k=2: rerun both paths, find the first differing position j,
and report the target model's top-2 logit margin at j computed three ways:
  (a) incremental M=1 decode (what plain greedy does),
  (b) one batched forward of prompt+prefix (M = many tokens, the verify-style matmul path).
A tiny margin (near-tie) with (a) and (b) picking different tokens means the mismatch is
numerical (M=1 vs M>1 quantized-matmul kernels accumulate differently), not a logic bug.

usage: uv run --project /Users/sdawka/Code/llming/mlx python /Users/sdawka/Code/llming/quality/greedy_diag.py --pair qwen4b-8bit_0.6b-8bit
"""

import argparse
import json
import os

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.utils import load

from alpha import PAIRS, ROOT, build_prompts, run_plain, run_spec


def top2(logits):
    lg = logits.astype(mx.float32)
    idx = mx.argsort(-lg)[:2]
    mx.eval(idx)
    i0, i1 = idx.tolist()
    return i0, i1, (lg[i0] - lg[i1]).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=192)
    a = ap.parse_args()
    raw = json.load(open(f"{ROOT}/quality/results/alpha_raw/{a.pair}.json"))
    bad = [n for n, e in raw["prompts"].items() if not e["k"][str(a.k)]["greedy_equal"]]
    tpath, dpath = PAIRS[a.pair]
    model, tok = load(tpath)
    draft, _ = load(dpath)
    eos = set(tok.eos_token_ids)
    prompts = build_prompts(tok, "qwen" if "qwen" in a.pair else ("gemma" if "gemma" in a.pair else "llama"))
    out = {}
    for name in bad:
        p = prompts[name]
        ref, _ = run_plain(model, p, a.max_tokens, eos)
        spec = [t for t, _ in run_spec(model, draft, p, a.k, a.max_tokens, eos)[0]]
        j = next((i for i in range(min(len(ref), len(spec))) if ref[i] != spec[i]), None)
        # spec runs use full KVCache for every layer (alpha.spec_cache); check that plain greedy is unchanged by
        # that cache choice, so a mismatch cannot come from RotatingKVCache vs KVCache (Gemma sliding layers)
        ref_full = []
        for t, _ in generate_step(mx.array(p), model, max_tokens=a.max_tokens,
                                  prompt_cache=[KVCache() for _ in model.layers]):
            ref_full.append(int(t))
            if int(t) in eos:
                break
        plain_cache_equal = ref_full == ref
        if j is None:
            out[name] = {"reproduced": False, "plain_fullcache_equal": plain_cache_equal}
            continue
        # (a) incremental M=1 path
        cache = make_prompt_cache(model)
        model(mx.array(p[:-1])[None], cache=cache)
        seq = [p[-1]] + ref[:j]
        for t in seq[:-1]:
            model(mx.array([t])[None], cache=cache)
        la = model(mx.array([seq[-1]])[None], cache=cache)[0, -1]
        # (b) one batched forward over prompt + prefix
        lb = model(mx.array(p + ref[:j])[None])[0, -1]
        a0, a1, am = top2(la)
        b0, b1, bm = top2(lb)
        out[name] = {"first_mismatch": j, "plain_token": ref[j], "spec_token": spec[j],
                     "M1_top2": [a0, a1], "M1_margin": am, "batched_top2": [b0, b1], "batched_margin": bm,
                     "plain_tok_str": tok.decode([ref[j]]), "spec_tok_str": tok.decode([spec[j]]),
                     "plain_fullcache_equal": plain_cache_equal}
        print(name, out[name], flush=True)
    os.makedirs(f"{ROOT}/quality/results/alpha_raw", exist_ok=True)
    json.dump(out, open(f"{ROOT}/quality/results/alpha_raw/{a.pair}.greedy_diag.json", "w"), indent=1)


if __name__ == "__main__":
    main()
