"""G2-M3 / ROOFLINE M8: wikitext-2 raw test perplexity for MLX models, llama-perplexity method.

Mirrors llama.cpp `llama-perplexity -c CTX`: tokenize the whole test file once (BOS first if the
tokenizer adds one), split into non-overlapping CTX-token chunks, set each chunk's first token to BOS
(when the tokenizer has a BOS that it adds), and score only the second half of each chunk (tokens
CTX/2 .. CTX-1, each predicted with >= CTX/2 tokens of context). Exact fp32 log-softmax NLL.
mlx_lm's own `mlx_lm.perplexity` is not used: it samples chat data (tulu-3) rather than wikitext.

Per-token NLLs are saved so within-tokenizer deltas get a paired standard error.

usage: uv run --project /Users/sdawka/Code/llming/mlx python /Users/sdawka/Code/llming/quality/ppl_mlx.py \
          --model /Users/sdawka/Code/llming/models/mlx-Llama-3.2-3B-Instruct-4bit --name llama3b-mlx-4bit
"""

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
import numpy as np
from mlx_lm.utils import load

ROOT = "/Users/sdawka/Code/llming"
WIKI = f"{ROOT}/models/datasets/wiki.test.raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=-1, help="-1 = all chunks")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default=f"{ROOT}/quality/results/ppl_raw")
    a = ap.parse_args()

    model, tok = load(a.model)
    text = open(WIKI, encoding="utf-8").read()
    ids = tok.encode(text)  # adds BOS if the tokenizer is configured to
    bos = tok.bos_token_id
    adds_bos = bos is not None and len(ids) > 0 and ids[0] == bos
    n_chunks = len(ids) // a.ctx
    if a.chunks > 0:
        n_chunks = min(n_chunks, a.chunks)
    data = np.array(ids[: n_chunks * a.ctx], dtype=np.int32).reshape(n_chunks, a.ctx)
    if adds_bos:
        data[:, 0] = bos
    first = a.ctx // 2
    nll = []
    t0 = time.perf_counter()
    for s in range(0, n_chunks, a.batch):
        b = mx.array(data[s : s + a.batch])
        logits = model(b[:, :-1])[:, first - 1 :, :].astype(mx.float32)
        loss = nn.losses.cross_entropy(logits, b[:, first:], reduction="none")
        mx.eval(loss)
        nll.append(np.array(loss).reshape(-1))
        if (s // a.batch) % 20 == 0:
            cur = np.concatenate(nll)
            print(f"  [{s + len(b)}/{n_chunks}] ppl so far {math.exp(cur.mean()):.4f}", flush=True)
    wall = time.perf_counter() - t0
    nll = np.concatenate(nll).astype(np.float64)
    ppl = math.exp(nll.mean())
    se = ppl * nll.std(ddof=1) / math.sqrt(nll.size)
    os.makedirs(a.out, exist_ok=True)
    np.save(f"{a.out}/{a.name}.nll.npy", nll.astype(np.float32))
    res = {
        "name": a.name, "model": a.model, "runtime": "mlx", "ctx": a.ctx, "chunks": n_chunks,
        "scored_tokens": int(nll.size), "total_test_tokens": len(ids), "bos_reset_per_chunk": adds_bos,
        "ppl": ppl, "ppl_se": se, "mean_nll": nll.mean(), "wall_s_non_authoritative": wall,
        "meta": {"timestamp": datetime.now(timezone.utc).isoformat(), "authoritative_quality": True,
                 "timing_authoritative": False,
                 "versions": {"mlx": mx.__version__, "mlx_lm": mlx_lm.__version__}},
    }
    json.dump(res, open(f"{a.out}/{a.name}.json", "w"), indent=1)
    print(f"RESULT {a.name}: ppl={ppl:.4f} ± {se:.4f} over {nll.size} tokens ({n_chunks} chunks of {a.ctx})")


if __name__ == "__main__":
    main()
