"""W8A8 linear for Llama 3.2 3B on MAX (Apple M5).

y = x @ w.T with symmetric int8 W8A8:
  * per-token activations     qa = round(x / sa), sa = max|x_row| / 127
  * per-output-channel weights qw = round(w / sw), sw = max|w_row| / 127
  * int32 accumulate, dequant c * sa * sw, cast to bf16.

Modes (``W8A8_MODE``):
  bf16 - no quantization (plain x @ w.T); reference.
  fq   - fake quant: the same int8 values matmul'd in float32 on the GPU (exact for
         |c| < 2^24; identical math to fpga/maxop PHANTOM_MODE=w8a8_gpu).
  int8 - the real MAX Apple M5 int8 widening-MMA GEMM, graph op
         ``matmul.int8.w8a8.apple`` (max.nn._graph_ops._apple_int8_w8a8_matmul; it
         quantizes the bf16 activation per token *inside* the op). The weight is
         quantized in-graph here (RTN per output channel), which is numerically what
         an offline int8 checkpoint would hold but costs time on every forward.

Optional SmoothQuant-style smoothing (``W8A8_SMOOTH=<path to .npz>``): per input
channel factors s (key ``<layer>.<proj>``) are folded as x/s and w*s before
quantization (exact in bf16 arithmetic up to rounding); see calibrate_smooth.py.

Routing (``W8A8_ROUTE``), same syntax as fpga/maxop: ``all:all``, ``1-26:all``,
``all:qkv,o,gate_up``, ``;``-separated. Projections: qkv, o, gate_up, down.
"""

from __future__ import annotations

import os

from max import functional as F
from max.dtype import DType
from max.graph import TensorType
from max.tensor import Tensor

PROJS = ("qkv", "o", "gate_up", "down")
_ALIASES = {"mlp": ("gate_up", "down"), "attn": ("qkv", "o"), "all": PROJS}


def parse_route(spec: str, num_layers: int) -> dict[int, set[str]]:
    route: dict[int, set[str]] = {}
    for part in filter(None, (p.strip() for p in spec.split(";"))):
        layers_s, _, projs_s = part.partition(":")
        projs: set[str] = set()
        for p in (projs_s or "all").split(","):
            p = p.strip()
            projs.update(_ALIASES.get(p, (p,)))
        bad = projs - set(PROJS)
        if bad:
            raise ValueError(f"unknown projections {bad} in W8A8_ROUTE={spec!r}")
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
    return os.environ.get("W8A8_MODE", "int8")


def quantize_rows(t: Tensor) -> tuple[Tensor, Tensor]:
    """Symmetric per-row int8: (q int8 [R, K], scale f32 [R, 1])."""
    tf = F.cast(t, DType.float32)
    amax = F.max(F.abs(tf), axis=-1)
    scale = F.elementwise_max(amax, F.constant(1e-8, DType.float32, device=tf.device)) / 127.0
    q = F.cast(F.clip(F.round(tf / scale), -127.0, 127.0), DType.int8)
    return q, scale


def int8_matmul_apple(x: Tensor, qw: Tensor, sw: Tensor) -> Tensor:
    """bf16 [M, N] = dequant(quant_per_token(x) @ qw.T) on the M5 int8 MMA path."""
    n = qw.shape[0]
    sw_flat = F.reshape(sw, [n])
    (y,) = F.custom(
        "matmul.int8.w8a8.apple",
        device=x.device,
        values=[x, qw, sw_flat],
        out_types=[TensorType(DType.bfloat16, [x.shape[0], n], x.device)],
    )
    return y


def w8a8_linear(x: Tensor, w: Tensor, smooth: Tensor | None = None) -> Tensor:
    m = mode()
    if m == "bf16":
        return x @ w.T
    if smooth is not None:
        # x' = x / s, w' = w * s (s per input channel, [K]); x' @ w'.T == x @ w.T.
        s = F.cast(smooth, DType.float32)
        x = F.cast(F.cast(x, DType.float32) / s, x.dtype)
        w = F.cast(F.cast(w, DType.float32) * s, w.dtype)
    qw, sw = quantize_rows(w)
    if m == "int8":
        return F.cast(int8_matmul_apple(F.cast(x, DType.bfloat16), qw, sw), x.dtype)
    if m == "fq":
        qa, sa = quantize_rows(x)
        cf = F.cast(qa, DType.float32) @ F.cast(qw, DType.float32).T
        return F.cast(cf * sa * F.transpose(sw, 0, 1), x.dtype)
    raise ValueError(f"W8A8_MODE={m!r}")


# --------------------------------------------------------------------------------------------
# Deployment path: offline int8 weights + prefill-only dispatch.


def prefill_min_m() -> int | None:
    """W8A8_PREFILL_MIN_M=T: use int8 only when the batch has >= T tokens (prefill),
    else the plain bf16 matmul (decode). Unset: always int8."""
    v = os.environ.get("W8A8_PREFILL_MIN_M")
    return int(v) if v else None


def offline_weights() -> bool:
    """W8A8_WEIGHTS=offline: int8 weights + scales are loaded with the checkpoint
    (RTN-quantized once by the weight adapter) instead of re-quantized per forward."""
    return os.environ.get("W8A8_WEIGHTS", "graph") == "offline"


def w8a8_linear_q(x: Tensor, w: Tensor, qw: Tensor, sw: Tensor) -> Tensor:
    """Real int8 path with pre-quantized weights; optional prefill-only dispatch."""
    t = prefill_min_m()
    if t is None:
        return F.cast(int8_matmul_apple(F.cast(x, DType.bfloat16), qw, sw), x.dtype)
    m_t = F.shape_to_tensor([x.shape[0]])  # int64 [1] on the host
    pred = F.reshape(F.greater_equal(m_t, F.constant(t, m_t.dtype, device=m_t.device)), [])
    out_t = TensorType(x.dtype, [x.shape[0], w.shape[0]], x.device)
    (y,) = F.cond(
        pred,
        [out_t],
        lambda: F.cast(int8_matmul_apple(F.cast(x, DType.bfloat16), qw, sw), x.dtype),
        lambda: x @ w.T,
    )
    return y
