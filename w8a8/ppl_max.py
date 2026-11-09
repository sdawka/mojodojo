"""wikitext-2 perplexity of Llama 3.2 3B in MAX, llama-perplexity method (same as quality/ppl_mlx.py).

Tokenize models/datasets/wiki.test.raw once (HF tokenizer, BOS first), split into
non-overlapping CTX-token chunks, set each chunk's first token to BOS, score the
second half of each chunk (tokens CTX/2 .. CTX-1) with exact fp32 log-softmax NLL.
Each chunk is one MAX prefill (fresh request => fresh KV), run through the full
pipeline (`custom_architectures=[w8a8_llama]`, `enable_echo=True` so the graph
returns logits for every position). The model's float32 logits are captured from
the pipeline model's execute(); the NLL is computed in numpy (float64 logsumexp).

Every scored token is a prefill token, so this measures the prefill path directly.

usage (from /Users/sdawka/Code/llming/fpga/maxop, whose pixi env this uses):
  W8A8_MODE=int8 W8A8_ROUTE=all:all pixi run python ../../w8a8/ppl_max.py --name llama3b-max-w8a8-int8-all
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path("/Users/sdawka/Code/llming")
WIKI = ROOT / "models/datasets/wiki.test.raw"
MODEL = ROOT / "models/Llama-3.2-3B-Instruct"
HERE = Path(__file__).resolve().parent


def load_pipeline(batch: int, max_length: int):
    from max.driver import DeviceSpec
    from max.pipelines.lib.config import PipelineConfig
    from max.pipelines.lib.pipeline_args import PipelineArgs
    from max.pipelines.lib.registry import PIPELINE_REGISTRY

    args = PipelineArgs(
        model_path=str(MODEL),
        quantization_encoding="bfloat16",
        device_specs=[DeviceSpec.accelerator()],
        max_length=max_length,
        enable_echo=True,
        runtime={"custom_architectures": [str(HERE / "w8a8_llama")], "max_batch_size": batch},
    )
    cfg = PipelineConfig.from_args(args)
    return PIPELINE_REGISTRY.retrieve(cfg)


def to_np(x) -> np.ndarray:
    if hasattr(x, "to_numpy"):
        from max.driver import CPU
        try:
            return x.to(CPU()).to_numpy()
        except Exception:
            return x.to_numpy()
    return np.from_dlpack(x.to(__import__("max.driver", fromlist=["CPU"]).CPU()))


def nll_rows(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    lg = logits.astype(np.float64)
    m = lg.max(axis=-1, keepdims=True)
    lse = (m[:, 0] + np.log(np.exp(lg - m).sum(axis=-1)))
    return lse - lg[np.arange(len(targets)), targets]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=-1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", default=str(HERE / "results/ppl_raw"))
    a = ap.parse_args()

    from max.pipelines.lib.generate import generate

    t0 = time.time()
    tokenizer, pipeline = load_pipeline(a.batch, 2 * a.ctx)
    load_s = time.time() - t0
    hf = tokenizer.delegate
    pm = pipeline._pipeline_model
    captured: list = []
    orig = pm.execute

    def hooked(*args, **kw):
        out = orig(*args, **kw)
        if not captured:  # first execute of a generate() call = the prefill
            captured.append((to_np(out.logits), to_np(out.logit_offsets)))
        return out

    pm.execute = hooked

    text = WIKI.read_text(encoding="utf-8")
    ids = hf(text, add_special_tokens=True)["input_ids"]
    bos = hf.bos_token_id
    adds_bos = bos is not None and ids[0] == bos
    n_chunks = len(ids) // a.ctx
    if a.chunks > 0:
        n_chunks = min(n_chunks, a.chunks)
    data = np.array(ids[: n_chunks * a.ctx], dtype=np.int64).reshape(n_chunks, a.ctx)
    if adds_bos:
        data[:, 0] = bos
    first = a.ctx // 2
    nll = []
    t1 = time.time()
    for s in range(0, n_chunks, a.batch):
        rows = data[s : s + a.batch]
        captured.clear()
        generate(pipeline, input_ids=[r.tolist() for r in rows], max_new_tokens=1, top_k=1, ignore_eos=True)
        logits, offs = captured[0]
        assert logits.shape[0] == len(rows) * a.ctx, (logits.shape, offs)
        for j, r in enumerate(rows):
            o = int(offs[j])
            lg = logits[o + first - 1 : o + a.ctx - 1]
            nll.append(nll_rows(lg, r[first:]))
        if (s // a.batch) % 20 == 0:
            cur = np.concatenate(nll)
            print(f"  [{s + len(rows)}/{n_chunks}] ppl so far {math.exp(cur.mean()):.4f}", flush=True)
    wall = time.time() - t1
    nll = np.concatenate(nll)
    ppl = math.exp(nll.mean())
    se = ppl * nll.std(ddof=1) / math.sqrt(nll.size)
    os.makedirs(a.out, exist_ok=True)
    np.save(f"{a.out}/{a.name}.nll.npy", nll.astype(np.float32))
    import max as _max

    res = {
        "name": a.name, "model": str(MODEL), "runtime": "max",
        "w8a8_mode": os.environ.get("W8A8_MODE", "int8"),
        "w8a8_route": "" if os.environ.get("W8A8_MODE") == "bf16" else os.environ.get("W8A8_ROUTE", "all:all"),
        "w8a8_smooth": os.environ.get("W8A8_SMOOTH"),
        "ctx": a.ctx, "chunks": n_chunks, "scored_tokens": int(nll.size), "total_test_tokens": len(ids),
        "bos_reset_per_chunk": adds_bos, "ppl": ppl, "ppl_se": se, "mean_nll": float(nll.mean()),
        "load_s_non_authoritative": load_s, "wall_s_non_authoritative": wall,
        "meta": {"timestamp": datetime.now(timezone.utc).isoformat(), "authoritative_quality": True,
                 "timing_authoritative": False,
                 "versions": {"max": getattr(_max, "__version__", "26.8.0.dev2026110805")}},
    }
    json.dump(res, open(f"{a.out}/{a.name}.json", "w"), indent=1)
    print(f"RESULT {a.name}: ppl={ppl:.4f} ± {se:.4f} over {nll.size} tokens ({n_chunks} chunks of {a.ctx})")


if __name__ == "__main__":
    main()
