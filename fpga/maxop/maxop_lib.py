"""Shared helpers: checkpoint access, numpy W8A8 reference, standalone routed graphs, cost model."""

from __future__ import annotations

import json
import math
import os
import struct
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
MODEL_DIR = REPO / "models" / "Llama-3.2-3B-Instruct"
SYSTOLIC = ROOT.parent  # fpga/ (contains systolic/)

HIDDEN, FFN, Q_DIM, KV_DIM, LAYERS = 3072, 8192, 3072, 1024, 28

# (K, N_out) of each routable projection; M = number of prompt tokens.
PROJ_SHAPES = {
    "qkv": (HIDDEN, Q_DIM + 2 * KV_DIM),
    "o": (Q_DIM, HIDDEN),
    "gate_up": (HIDDEN, 2 * FFN),
    "down": (FFN, HIDDEN),
}

# Synthesized fmax (fpga/README.md, ECP5 nextpnr). N=16 is the no-DSP mapping.
FMAX_MHZ = {4: 84.2, 8: 67.5, 16: 60.4}
# Host link bandwidths in bytes/s.
IO_BW = {"usb2": 40e6, "usb3": 400e6, "pcie": 4e9}


# ---------------------------------------------------------------- checkpoint
@lru_cache(maxsize=None)
def _index() -> dict[str, str]:
    return json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]


@lru_cache(maxsize=None)
def _header(fname: str) -> tuple[int, dict]:
    with open(MODEL_DIR / fname, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def load_bf16_as_f32(name: str) -> np.ndarray:
    """Reads a bf16 checkpoint tensor and widens it exactly to float32."""
    fname = _index()[name]
    off, hdr = _header(fname)
    meta = hdr[name]
    assert meta["dtype"] == "BF16", meta
    s, e = meta["data_offsets"]
    raw = np.fromfile(MODEL_DIR / fname, dtype=np.uint16, count=(e - s) // 2, offset=off + s)
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])


def to_bf16_f32(x: np.ndarray) -> np.ndarray:
    """Rounds float32 to bf16 (RNE) and returns it as float32."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(np.float32)


def layer_weights(layer: int) -> dict[str, np.ndarray]:
    p = f"model.layers.{layer}."
    g = lambda n: load_bf16_as_f32(p + n)  # noqa: E731
    return {
        "qkv": np.concatenate([g("self_attn.q_proj.weight"), g("self_attn.k_proj.weight"), g("self_attn.v_proj.weight")]),
        "o": g("self_attn.o_proj.weight"),
        "gate_up": np.concatenate([g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")]),
        "down": g("mlp.down_proj.weight"),
        "post_attention_layernorm": g("post_attention_layernorm.weight"),
        "input_layernorm": g("input_layernorm.weight"),
    }


# ---------------------------------------------------------------- reference
def quantize_rows_np(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = t.astype(np.float32)
    amax = np.max(np.abs(t), axis=-1, keepdims=True)
    scale = np.maximum(amax, np.float32(1e-8)) / np.float32(127.0)
    q = np.clip(np.round(t / scale), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def int32_matmul_exact(qa: np.ndarray, qw: np.ndarray) -> np.ndarray:
    """Exact int32 qa[M,K] @ qw[N,K]^T (float64 BLAS; |sum| < 2^53, then int32 wrap)."""
    c = qa.astype(np.float64) @ qw.astype(np.float64).T
    return c.astype(np.int64).astype(np.int32)


# ---------------------------------------------------------------- cost model
def tile_cycles(m: int, k: int, n_out: int, n: int) -> int:
    return math.ceil(m / n) * math.ceil(n_out / n) * (k + 3 * n - 1)


def io_bytes(m: int, k: int, n_out: int, n: int) -> dict[str, int]:
    """Host-link bytes for one routed matmul.

    streamed: the v1 tile as specified (no operand buffers): every pass streams
              one A column and one B row (2N int8) per beat, then N rows of N int32.
    minimal : each operand crosses the link once (needs on-board A/B buffers).
    """
    passes = math.ceil(m / n) * math.ceil(n_out / n)
    out = 4 * m * n_out
    return {"streamed": passes * k * 2 * n + out, "minimal": m * k + k * n_out + out}


def route_matmuls(route: dict[int, set[str]], m: int) -> list[dict]:
    rows = []
    for layer in sorted(route):
        for proj in ("qkv", "o", "gate_up", "down"):
            if proj in route[layer]:
                k, n_out = PROJ_SHAPES[proj]
                rows.append({"layer": layer, "proj": proj, "M": m, "K": k, "N_out": n_out})
    return rows


def project(matmuls: list[dict], gpu_ttft_ms: float, gpu_routed_ms: float = 0.0) -> dict:
    """Projected TTFT = GPU TTFT - GPU time of the routed matmuls + tile time.

    Tile time per N: compute-bound (cycles / fmax) and I/O-bound
    (bytes / link BW) per link; the projection takes max(compute, io).
    """
    out: dict = {}
    for n, f in FMAX_MHZ.items():
        cyc = sum(tile_cycles(r["M"], r["K"], r["N_out"], n) for r in matmuls)
        t_compute = cyc / (f * 1e6)
        entry = {"cycles": cyc, "fmax_mhz": f, "tile_compute_ms": t_compute * 1e3, "ttft_ms": {}}
        for io_mode in ("streamed", "minimal"):
            b = sum(io_bytes(r["M"], r["K"], r["N_out"], n)[io_mode] for r in matmuls)
            entry[f"io_bytes_{io_mode}"] = b
            for link, bw in IO_BW.items():
                t = max(t_compute, b / bw)
                entry["ttft_ms"][f"{link}_{io_mode}"] = gpu_ttft_ms - gpu_routed_ms + t * 1e3
        entry["ttft_ms"]["compute_only"] = gpu_ttft_ms - gpu_routed_ms + t_compute * 1e3
        out[f"N{n}"] = entry
    return out
