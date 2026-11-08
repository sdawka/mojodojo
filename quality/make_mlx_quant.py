"""Make non-default MLX 4-bit quants.

Default mode: a Q4_K_M-like MLX quant: mlx_lm's mixed_4_6 recipe plus a 6-bit (or 8-bit) embed_tokens.
Llama 3.2 3B ties lm_head to embed_tokens, so mlx_lm's "lm_head -> 6 bit" rule in mixed_4_6 never
fires and the output projection stays 4-bit. llama.cpp Q4_K_M keeps token_embd/output at Q6_K.

--q40-grid mode (Gemma 3 QAT): every linear layer is quantized onto exactly the llama.cpp Q4_0 grid that
Google's quantization-aware training targeted (blocks of 32, symmetric: d = (signed max-abs value) / -8,
q = min(15, trunc(x/d + 8.5)), w = (q - 8) * d). It is stored as MLX affine 4-bit g32 with
scale = d and bias = -8 d, so MLX's own quantized_matmul kernels run it unchanged. The only deviation
from Q4_0 is that d is stored in bf16 (MLX scales follow the model dtype) instead of fp16.
The tied embedding / lm_head gets --embed-bits (affine g64), since Google's QAT GGUF keeps it at F16.

usage: uv run --project /Users/sdawka/Code/llming/mlx python /Users/sdawka/Code/llming/quality/make_mlx_quant.py \
          --src /Users/sdawka/Code/llming/models/mlx-Llama-3.2-3B-Instruct-bf16 \
          --dst /Users/sdawka/Code/llming/models/mlx-Llama-3.2-3B-Instruct-4bit-mixed_4_6-emb6 --embed-bits 6 [--no-mixed]
       uv run --project /Users/sdawka/Code/llming/mlx python /Users/sdawka/Code/llming/quality/make_mlx_quant.py \
          --src /Users/sdawka/Code/llming/models/gemma3/mlx-gemma-3-4b-it-qat-bf16 \
          --dst /Users/sdawka/Code/llming/models/gemma3/mlx-gemma-3-4b-it-qat-q40grid --q40-grid --embed-bits 8
"""

import argparse
import glob
import json
import os

import mlx.core as mx
import numpy as np
from mlx_lm.convert import convert, mixed_quant_predicate_builder
from mlx_lm.utils import load


def q40_grid(w):
    """w: (out, in) float32 numpy -> (packed uint32 (out, in/8), scale (out, in/32), bias (out, in/32), deq)."""
    out, n = w.shape
    b = w.reshape(out, n // 32, 32)
    idx = np.abs(b).argmax(-1)[..., None]
    mx_ = np.take_along_axis(b, idx, -1)[..., 0]  # signed value with the largest magnitude
    d = mx_ / -8.0
    inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1.0), 0.0)
    q = np.minimum(15, np.trunc(b * inv[..., None] + 8.5)).astype(np.uint32)
    # bf16 scale (round-to-nearest-even via mlx); bias = -8 * scale is exact in bf16
    s = np.array(mx.array(d.astype(np.float32)).astype(mx.bfloat16).astype(mx.float32))
    qq = q.reshape(out, n // 8, 8)
    packed = np.zeros((out, n // 8), dtype=np.uint32)
    for j in range(8):
        packed |= qq[..., j] << np.uint32(4 * j)
    return packed, s, -8.0 * s, q


def make_q40_grid(src, dst, embed_bits):
    def pred(path, module):
        if "embed_tokens" in path or "lm_head" in path:
            return {"group_size": 64, "bits": embed_bits, "mode": "affine"}
        return {"group_size": 32, "bits": 4, "mode": "affine"}

    convert(src, dst, quantize=True, q_bits=4, q_group_size=32, quant_predicate=pred)
    srcw = {}
    for f in glob.glob(f"{src}/*.safetensors"):
        srcw.update(mx.load(f))
    n_rep = 0
    for f in sorted(glob.glob(f"{dst}/*.safetensors")):
        dw = mx.load(f)
        for k in [k for k in dw if k.endswith(".scales")]:
            base = k[: -len(".scales")]
            if "embed_tokens" in base or "lm_head" in base:
                continue
            w = np.array(srcw[base + ".weight"].astype(mx.float32))
            assert dw[k].shape[-1] * 32 == w.shape[-1], (k, dw[k].shape, w.shape)
            packed, s, bias, q = q40_grid(w)
            sb = mx.array(s).astype(mx.bfloat16)
            bb = mx.array(bias).astype(mx.bfloat16)
            pk = mx.array(packed)
            # round-trip check: MLX dequantize (bf16 output) must equal bf16((q - 8) * d)
            deq = mx.dequantize(pk, sb, bb, group_size=32, bits=4).astype(mx.float32)
            ref = mx.array(((q.astype(np.float32) - 8.0) * s[..., None]).reshape(w.shape))
            assert mx.array_equal(deq, ref.astype(mx.bfloat16).astype(mx.float32)).item(), base
            dw[base + ".weight"], dw[k], dw[base + ".biases"] = pk, sb, bb
            n_rep += 1
        mx.save_safetensors(f, dw, metadata={"format": "mlx"})
    cfg = json.load(open(f"{dst}/config.json"))
    cfg["quantization_note"] = ("linear layers on the exact llama.cpp Q4_0 grid (g32 symmetric, bf16 d), "
                                f"embed {embed_bits}-bit affine g64; made by quality/make_mlx_quant.py --q40-grid")
    json.dump(cfg, open(f"{dst}/config.json", "w"), indent=4)
    print(f"replaced {n_rep} linear weights with Q4_0-grid values")


ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--embed-bits", type=int, default=6)
ap.add_argument("--no-mixed", action="store_true", help="plain 4-bit g64 everywhere except embed")
ap.add_argument("--q40-grid", action="store_true", help="exact Q4_0 grid for linear layers (QAT checkpoints)")
a = ap.parse_args()

if a.q40_grid:
    make_q40_grid(a.src, a.dst, a.embed_bits)
else:
    model, _ = load(a.src, lazy=True)
    base = None if a.no_mixed else mixed_quant_predicate_builder("mixed_4_6", model, 64)
    del model

    def pred(path, module):
        if "embed_tokens" in path or "lm_head" in path:
            return {"group_size": 64, "bits": a.embed_bits, "mode": "affine"}
        if base is not None:
            return base(path, module)
        return {"group_size": 64, "bits": 4, "mode": "affine"}

    convert(a.src, a.dst, quantize=True, q_bits=4, q_group_size=64, quant_predicate=pred)
