"""W8A8 linear layers whose int8 matmul runs on the phantom NPU via a MAX custom op.

`phantom_linear(x, w, site)` computes y = x @ w.T with
  * per-token symmetric int8 activations  qa = round(x / sa), sa = max|x_row| / 127
  * per-output-channel symmetric int8 weights qw = round(w / sw), sw = max|w_row| / 127
  * int32 accumulation  c = qa @ qw.T  -> the custom op `phantom_matmul_i8` (CPU host)
  * dequant  y = c * sa * sw.T, cast back to x.dtype.

Backends (``PHANTOM_MODE``):
  phantom   - int32 matmul on the phantom tile (custom op, host CPU)      [default]
  w8a8_gpu  - same quantization, matmul of the int8 values in float32 on the
              model device (fake-quant reference; exact for |c| < 2^24, see README)
  bf16      - no routing (plain x @ w.T); used for the bf16 baseline

Phantom backend (``PHANTOM_BACKEND``, read by the Mojo kernel per launch):
  stub (functional int32 + formula cycles, default) or device (lane G1-C's
  libphantom.dylib, cycle-accurate; tile size ``PHANTOM_DEVICE_N``).

Routing (``PHANTOM_ROUTE``), e.g. ``0:mlp`` (default), ``0-3:qkv,o``, ``all:all``.
Projections: qkv (fused q|k|v), o, gate_up (fused gate|up), down; ``mlp`` =
gate_up,down; ``attn`` = qkv,o; ``all`` = all four.
"""

from __future__ import annotations

import os
from pathlib import Path

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.tensor import Tensor

KERNELS = Path(__file__).resolve().parent.parent / "kernels"
# Lane G1-C's phantom driver (C ABI), used by the kernel when PHANTOM_BACKEND=device.
os.environ.setdefault(
    "PHANTOM_LIB",
    str(Path(__file__).resolve().parents[2] / "phantom" / "build" / "libphantom.dylib"),
)

PROJS = ("qkv", "o", "gate_up", "down")
PROJ_ID = {p: i for i, p in enumerate(PROJS)}
_ALIASES = {"mlp": ("gate_up", "down"), "attn": ("qkv", "o"), "all": PROJS}


def parse_route(spec: str, num_layers: int) -> dict[int, set[str]]:
    """'0-3:qkv,o;5:mlp' -> {0: {'qkv','o'}, ..., 5: {'gate_up','down'}}."""
    route: dict[int, set[str]] = {}
    for part in filter(None, (p.strip() for p in spec.split(";"))):
        layers_s, _, projs_s = part.partition(":")
        projs: set[str] = set()
        for p in (projs_s or "mlp").split(","):
            p = p.strip()
            projs.update(_ALIASES.get(p, (p,)))
        bad = projs - set(PROJS)
        if bad:
            raise ValueError(f"unknown projections {bad} in PHANTOM_ROUTE={spec!r}")
        if layers_s == "all":
            layers = range(num_layers)
        elif "-" in layers_s:
            lo, hi = layers_s.split("-")
            layers = range(int(lo), int(hi) + 1)
        else:
            layers = [int(x) for x in layers_s.split(",")]
        for layer in layers:
            route.setdefault(layer, set()).update(projs)
    return route


def mode() -> str:
    return os.environ.get("PHANTOM_MODE", "phantom")


def quantize_rows(t: Tensor) -> tuple[Tensor, Tensor]:
    """Symmetric per-row int8: returns (q int8 [R, K], scale f32 [R, 1])."""
    tf = F.cast(t, DType.float32)
    amax = F.max(F.abs(tf), axis=-1)
    scale = F.elementwise_max(amax, F.constant(1e-8, DType.float32, device=tf.device)) / 127.0
    q = F.cast(F.clip(F.round(tf / scale), -127.0, 127.0), DType.int8)
    return q, scale


def phantom_matmul_i8(qa: Tensor, qw: Tensor, site: int) -> Tensor:
    """int32 [M, N] = qa[M, K] @ qw[N, K]^T on the phantom tile (host CPU op)."""
    cpu = DeviceRef.CPU()
    a = qa.to(CPU())
    w = qw.to(CPU())
    site_t = F.constant([site], DType.int32, device=CPU())
    (c,) = F.custom(
        "phantom_matmul_i8",
        device=cpu,
        values=[a, w, site_t],
        out_types=[TensorType(DType.int32, [a.shape[0], w.shape[0]], cpu)],
        custom_extensions=KERNELS,
    )
    return c


def phantom_linear(x: Tensor, w: Tensor, site: int) -> Tensor:
    m = mode()
    if m == "bf16":
        return x @ w.T
    qa, sa = quantize_rows(x)
    qw, sw = quantize_rows(w)
    if m == "phantom":
        c = phantom_matmul_i8(qa, qw, site).to(x.device)
        cf = F.cast(c, DType.float32)
    elif m == "w8a8_gpu":
        cf = F.cast(qa, DType.float32) @ F.cast(qw, DType.float32).T
    else:
        raise ValueError(f"PHANTOM_MODE={m!r}")
    y = cf * sa * F.transpose(sw, 0, 1)
    return F.cast(y, x.dtype)


def site_id(layer: int, proj: str) -> int:
    return layer * len(PROJS) + PROJ_ID[proj]
