"""Python driver: Llama 3.2 3B through the full MAX pipeline with phantom-routed layers.

Loads the pipeline once per (PHANTOM_MODE, PHANTOM_ROUTE) in this process via the
custom architecture `phantom_llama` (the same one `max generate
--custom-architectures` uses), then runs greedy generation over a prompt set with
next-token logits recorded, and writes an .npz.

    PHANTOM_MODE=phantom PHANTOM_ROUTE=0:mlp python run_model.py out.npz [--new 8]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = str(HERE.parent.parent / "models" / "Llama-3.2-3B-Instruct")

PROMPTS = [
    "The capital of France is",
    "Explain in one sentence why the sky is blue.",
    "Write a haiku about a systolic array.",
    "List three prime numbers greater than 100:",
    "def fibonacci(n):",
    "Translate 'good morning' into Spanish:",
    "The main difference between a CPU and a GPU is",
    "Once upon a time, a robot learned to paint. One day",
]


def load_pipeline(max_length: int = 1024):
    from max.driver import DeviceSpec
    from max.pipelines.lib.config import PipelineConfig
    from max.pipelines.lib.pipeline_args import PipelineArgs
    from max.pipelines.lib.registry import PIPELINE_REGISTRY

    args = PipelineArgs(
        model_path=MODEL,
        quantization_encoding="bfloat16",
        device_specs=[DeviceSpec.accelerator()],
        max_length=max_length,
        runtime={"custom_architectures": [str(HERE / "phantom_llama")], "max_batch_size": 1},
    )
    cfg = PipelineConfig.from_args(args)
    return PIPELINE_REGISTRY.retrieve(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--new", type=int, default=8, help="greedy tokens per prompt")
    ap.add_argument("--prompt-file", help="one extra prompt from a text file (e.g. bench/prompt_512.txt)")
    ap.add_argument("--only-file", action="store_true", help="run only --prompt-file")
    ap.add_argument("--file-new", type=int, default=1, help="greedy tokens for --prompt-file")
    ap.add_argument("--n-prompts", type=int, default=len(PROMPTS))
    a = ap.parse_args()

    from max.pipelines.lib.generate import generate

    t0 = time.time()
    tokenizer, pipeline = load_pipeline()
    load_s = time.time() - t0
    hf_tok = tokenizer.delegate

    prompts = [] if a.only_file else list(PROMPTS[: a.n_prompts])
    news = [a.new] * len(prompts)
    if a.prompt_file:
        prompts.append(Path(a.prompt_file).read_text())
        news.append(a.file_new)

    seqs, plens, first_logits, times = [], [], [], []
    for p, new in zip(prompts, news):
        ids = hf_tok(p)["input_ids"]  # BOS + text, as `max generate` encodes a raw prompt
        t = time.time()
        r = generate(pipeline, input_ids=[ids], max_new_tokens=new, top_k=1, ignore_eos=True, output_logits=True)
        times.append(time.time() - t)
        seqs.append(np.array(r.sequences[0][len(ids):], dtype=np.int64))
        plens.append(len(ids))
        first_logits.append(np.asarray(r.logits[0][0], dtype=np.float32))  # prefill logits [vocab]
        print(f"prompt {len(seqs)}/{len(prompts)}: {len(ids)} tok, {time.time() - t:.2f}s", flush=True)
    np.savez_compressed(
        a.out,
        prompt_lengths=np.array(plens),
        first_logits=np.stack(first_logits),
        gen_seconds=np.array(times),
        meta=json.dumps({
            "mode": os.environ.get("PHANTOM_MODE", "phantom"),
            "route": os.environ.get("PHANTOM_ROUTE", "0:mlp"),
            "backend": os.environ.get("PHANTOM_BACKEND", "stub"),
            "new_tokens": [s.tolist() for s in seqs],
            "load_seconds": load_s,
            "prompts": prompts,
            "texts": [hf_tok.decode(s) for s in seqs],
        }),
    )
    print(f"wrote {a.out}: load {load_s:.1f}s, gen {sum(times):.1f}s")


if __name__ == "__main__":
    main()
