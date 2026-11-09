"""Offline (load-time) RTN int8 weights for W8A8_WEIGHTS=offline.

The weight adapter keeps every bf16 checkpoint tensor (decode still uses them when
W8A8_PREFILL_MIN_M is set) and adds, per routed projection, an int8 weight and an
fp32 per-output-channel scale computed once in numpy from the bf16 safetensors:
  sw = max(max|w_row|, 1e-8) / 127,  qw = clip(round_half_even(w / sw), -127, 127)
(the same formula as ops.quantize_rows; fused q|k|v and gate|up are quantized per
row, so fusing before or after quantization is identical).
"""

from __future__ import annotations

import json
import os
import struct
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL_DIR = Path("/Users/sdawka/Code/llming/models/Llama-3.2-3B-Instruct")


@lru_cache(maxsize=None)
def _index() -> dict[str, str]:
    p = MODEL_DIR / "model.safetensors.index.json"
    if p.exists():
        return json.loads(p.read_text())["weight_map"]
    out = {}
    for f in MODEL_DIR.glob("*.safetensors"):
        for k in _header(f.name)[1]:
            if k != "__metadata__":
                out[k] = f.name
    return out


@lru_cache(maxsize=None)
def _header(fname: str) -> tuple[int, dict]:
    with open(MODEL_DIR / fname, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def load_f32(name: str) -> np.ndarray:
    fname = _index()[name]
    off, hdr = _header(fname)
    meta = hdr[name]
    assert meta["dtype"] == "BF16", meta
    s, e = meta["data_offsets"]
    raw = np.fromfile(MODEL_DIR / fname, dtype=np.uint16, count=(e - s) // 2, offset=off + s)
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])


def quantize_rows_np(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    amax = np.abs(w).max(axis=1, keepdims=True).astype(np.float32)
    scale = (np.maximum(amax, np.float32(1e-8)) / np.float32(127.0)).astype(np.float32)
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, scale


def offline_tensors(route: dict[int, set[str]]) -> dict[str, np.ndarray]:
    """MAX weight name -> numpy array for every routed projection."""
    out: dict[str, np.ndarray] = {}
    for i, projs in sorted(route.items()):
        hp = f"model.layers.{i}."
        mp = f"language_model.layers.{i}."
        if "qkv" in projs:
            w = np.concatenate([load_f32(hp + f"self_attn.{p}_proj.weight") for p in "qkv"])
            out[mp + "self_attn.qkv_w8"], out[mp + "self_attn.qkv_s8"] = quantize_rows_np(w)
        if "o" in projs:
            w = load_f32(hp + "self_attn.o_proj.weight")
            out[mp + "self_attn.o_proj.w8"], out[mp + "self_attn.o_proj.s8"] = quantize_rows_np(w)
        if "gate_up" in projs:
            w = np.concatenate([load_f32(hp + "mlp.gate_proj.weight"), load_f32(hp + "mlp.up_proj.weight")])
            out[mp + "mlp.gate_up_w8"], out[mp + "mlp.gate_up_s8"] = quantize_rows_np(w)
        if "down" in projs:
            w = load_f32(hp + "mlp.down_proj.weight")
            out[mp + "mlp.down_w8"], out[mp + "mlp.down_s8"] = quantize_rows_np(w)
    return out
