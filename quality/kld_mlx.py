"""Quantization fidelity beyond perplexity: KL(ref || quant) and top-1 agreement on the wikitext-2 ppl chunks.

Why: Gemma 3 *-it* models score PPL ~17.5 on wikitext (text far from their chat distribution), and 4-bit
quants can score *lower* PPL than bf16, so a PPL delta alone can hide a large change in the output
distribution. This script evaluates the reference and the quantized model on the same chunks as ppl_mlx.py
(ctx 512, BOS reset, score the second half) and reports, per scored token,
  KL(p_ref || p_q) in nats (mean, median, p99), the fraction where argmax agrees, and the mean |Δ NLL|.
Same statistics as llama-perplexity --kl-divergence ("Mean KLD", "Same top p").

usage: uv run --project /Users/sdawka/Code/llming/mlx python /Users/sdawka/Code/llming/quality/kld_mlx.py \
          --ref models/gemma3/mlx-gemma-3-4b-it-bf16 --model models/gemma3/mlx-gemma-3-4b-it-4bit \
          --name gemma4b-mlx-4bit [--chunks 64]
"""

import argparse
import json
import math
import os
from datetime import datetime, timezone

import mlx.core as mx
import mlx_lm
import numpy as np
from mlx_lm.utils import load

ROOT = "/Users/sdawka/Code/llming"
WIKI = f"{ROOT}/models/datasets/wiki.test.raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", default=f"{ROOT}/quality/results/kld_raw")
    a = ap.parse_args()
    ab = lambda p: p if os.path.isabs(p) else f"{ROOT}/{p}"
    ref, tok = load(ab(a.ref))
    mdl, tok2 = load(ab(a.model))
    assert tok.get_vocab() == tok2.get_vocab()
    ids = tok.encode(open(WIKI, encoding="utf-8").read())
    bos = tok.bos_token_id
    adds_bos = bos is not None and ids[0] == bos
    n = min(len(ids) // a.ctx, a.chunks)
    data = np.array(ids[: n * a.ctx], dtype=np.int32).reshape(n, a.ctx)
    if adds_bos:
        data[:, 0] = bos
    first = a.ctx // 2
    kl, same, dnll = [], [], []
    for s in range(0, n, a.batch):
        b = mx.array(data[s : s + a.batch])
        lp = ref(b[:, :-1])[:, first - 1 :, :].astype(mx.float32)
        lq = mdl(b[:, :-1])[:, first - 1 :, :].astype(mx.float32)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        lq = lq - mx.logsumexp(lq, axis=-1, keepdims=True)
        k = (mx.exp(lp) * (lp - lq)).sum(-1)
        sm = (lp.argmax(-1) == lq.argmax(-1))
        tgt = b[:, first:][..., None]
        dn = mx.abs(mx.take_along_axis(lp, tgt, -1) - mx.take_along_axis(lq, tgt, -1))[..., 0]
        mx.eval(k, sm, dn)
        kl.append(np.array(k).ravel()); same.append(np.array(sm).ravel()); dnll.append(np.array(dn).ravel())
    kl, same, dnll = np.concatenate(kl).astype(np.float64), np.concatenate(same), np.concatenate(dnll)
    res = {"name": a.name, "ref": ab(a.ref), "model": ab(a.model), "runtime": "mlx", "ctx": a.ctx, "chunks": n,
           "scored_tokens": int(kl.size), "mean_kld": float(kl.mean()), "mean_kld_se": float(kl.std(ddof=1) / math.sqrt(kl.size)),
           "median_kld": float(np.median(kl)), "p99_kld": float(np.quantile(kl, 0.99)),
           "same_top1_pct": float(100 * same.mean()), "mean_abs_dnll": float(dnll.mean()),
           "meta": {"timestamp": datetime.now(timezone.utc).isoformat(), "authoritative_quality": True,
                    "timing_authoritative": False, "versions": {"mlx": mx.__version__, "mlx_lm": mlx_lm.__version__}}}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/{a.name}.json", "w"), indent=1)
    print(f"RESULT {a.name}: mean KLD {res['mean_kld']:.5f} ± {res['mean_kld_se']:.5f}, median {res['median_kld']:.5f}, "
          f"p99 {res['p99_kld']:.4f}, same top-1 {res['same_top1_pct']:.2f}% over {kl.size} tokens ({n} chunks)")


if __name__ == "__main__":
    main()
