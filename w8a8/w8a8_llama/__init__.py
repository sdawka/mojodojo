"""Custom MAX architecture: Llama 3 (ModuleV3) with W8A8 projections.

``--custom-architectures /Users/sdawka/Code/llming/w8a8/w8a8_llama`` re-registers
``LlamaForCausalLM_ModuleV3`` (allow_override=True) with a ``Llama3Decoder``
subclass whose routed modules have their class swapped (same technique as
fpga/maxop/phantom_llama), so weight names are unchanged. Env: W8A8_MODE
(bf16|fq|int8), W8A8_ROUTE, W8A8_SMOOTH (see ops.py).
"""

from __future__ import annotations

import dataclasses
import logging
import os

import numpy as np
from max.dtype import DType
from max.nn.common_layers.linear import QKVLinear
from max.nn.common_layers.mlp import MLP
from max.nn.linear import Linear
from max.pipelines.architectures.llama3_modulev3 import arch as _arch
from max.pipelines.architectures.llama3_modulev3.llama3 import Llama3Decoder
from max.pipelines.lib import GraphRole
from max.tensor import Tensor

from max.graph.weights import WeightData, WeightsFormat
from max.pipelines.architectures.llama3_modulev3 import weight_adapters as _wa

from .ops import F, mode, offline_weights, parse_route, w8a8_linear, w8a8_linear_q

logger = logging.getLogger("max.pipelines")

_SMOOTH: dict[str, np.ndarray] = {}


def _smooth(key: str | None, device) -> Tensor | None:
    if key is None or key not in _SMOOTH:
        return None
    return F.constant(_SMOOTH[key], DType.float32, device=device)


class W8A8Linear(Linear):
    _w8a8_key: str

    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self, "w8"):
            return w8a8_linear_q(x, self.weight, self.w8, self.s8)
        return w8a8_linear(x, self.weight, _smooth(self._w8a8_key, x.device))


class W8A8QKV(QKVLinear):
    _w8a8_key: str

    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self, "qkv_w8"):
            return w8a8_linear_q(x, self.fused_weight, self.qkv_w8, self.qkv_s8)
        return w8a8_linear(x, self.fused_weight, _smooth(self._w8a8_key, x.device))


class W8A8MLP(MLP):
    _w8a8_layer: int
    _w8a8_projs: frozenset

    def forward(self, x: Tensor) -> Tensor:
        ffl = int(self.gate_proj.weight.shape[0])
        w_gu = F.concat((self.gate_proj.weight, self.up_proj.weight))
        if "gate_up" in self._w8a8_projs and hasattr(self, "gate_up_w8"):
            gu = w8a8_linear_q(x, w_gu, self.gate_up_w8, self.gate_up_s8)
        elif "gate_up" in self._w8a8_projs:
            gu = w8a8_linear(x, w_gu, _smooth(f"{self._w8a8_layer}.gate_up", x.device))
        else:
            gu = x @ w_gu.T
        gate, up = F.split(gu, [ffl, ffl], axis=1)
        h = F.mul(self.activation_function(gate), up)
        if "down" in self._w8a8_projs and hasattr(self, "down_w8"):
            return w8a8_linear_q(h, self.down_proj.weight, self.down_w8, self.down_s8)
        if "down" in self._w8a8_projs:
            return w8a8_linear(h, self.down_proj.weight, _smooth(f"{self._w8a8_layer}.down", h.device))
        return self.down_proj(h)


def apply_route(layers, spec: str) -> dict[int, set[str]]:
    route = parse_route(spec, len(layers))
    for i, projs in route.items():
        blk = layers[i]
        attn = blk.self_attn
        if "qkv" in projs:
            assert type(attn.qkv_proj) is QKVLinear and not attn.clip_qkv and not attn.has_bias
            attn.qkv_proj.__class__ = W8A8QKV
            object.__setattr__(attn.qkv_proj, "_w8a8_key", f"{i}.qkv")
        if "o" in projs:
            assert type(attn.o_proj) is Linear
            attn.o_proj.__class__ = W8A8Linear
            object.__setattr__(attn.o_proj, "_w8a8_key", f"{i}.o")
        mlp_projs = frozenset(projs & {"gate_up", "down"})
        if mlp_projs:
            assert type(blk.mlp) is MLP and not blk.mlp.has_bias, type(blk.mlp)
            blk.mlp.__class__ = W8A8MLP
            object.__setattr__(blk.mlp, "_w8a8_layer", i)
            object.__setattr__(blk.mlp, "_w8a8_projs", mlp_projs)
    return route


def _placeholders(mod, prefix: str, n: int, k: int) -> None:
    """int8 [n, k] weight + f32 [n, 1] scale, filled from the checkpoint by the adapter."""
    setattr(mod, f"{prefix}w8", Tensor.zeros([n, k], dtype=DType.int8))
    setattr(mod, f"{prefix}s8", Tensor.zeros([n, 1], dtype=DType.float32))


def add_offline_weights(layers, route: dict[int, set[str]]) -> None:
    for i, projs in route.items():
        blk = layers[i]
        attn = blk.self_attn
        if "qkv" in projs:
            w = attn.qkv_proj.fused_weight
            _placeholders(attn.qkv_proj, "qkv_", int(w.shape[0]), int(w.shape[1]))
        if "o" in projs:
            w = attn.o_proj.weight
            _placeholders(attn.o_proj, "", int(w.shape[0]), int(w.shape[1]))
        if "gate_up" in projs:
            g = blk.mlp.gate_proj.weight
            _placeholders(blk.mlp, "gate_up_", 2 * int(g.shape[0]), int(g.shape[1]))
        if "down" in projs:
            w = blk.mlp.down_proj.weight
            _placeholders(blk.mlp, "down_", int(w.shape[0]), int(w.shape[1]))


def _route_spec() -> str:
    return "" if mode() == "bf16" else os.environ.get("W8A8_ROUTE", "all:all")


def convert_safetensor_state_dict(state_dict, **kwargs) -> dict[str, WeightData]:
    """Stock Llama3 adapter + (W8A8_WEIGHTS=offline) load-time RTN int8 weights."""
    out = _wa.convert_safetensor_state_dict(state_dict, **kwargs)
    if offline_weights() and mode() == "int8" and _route_spec():
        from .offline import offline_tensors

        n_layers = 1 + max(int(k.split(".")[2]) for k in out if k.startswith("language_model.layers."))
        for name, arr in offline_tensors(parse_route(_route_spec(), n_layers)).items():
            out[name] = WeightData.from_numpy(arr, name)
    return out


class W8A8Llama3Decoder(Llama3Decoder):
    def __init__(self, config) -> None:
        super().__init__(config)
        spec = _route_spec()
        smooth_path = os.environ.get("W8A8_SMOOTH")
        _SMOOTH.clear()
        if smooth_path and spec:
            with np.load(smooth_path) as z:
                _SMOOTH.update({k: z[k].astype(np.float32) for k in z.files})
        route = apply_route(self.language_model.layers, spec)
        offline = offline_weights() and mode() == "int8" and bool(route)
        if offline:
            assert not smooth_path, "W8A8_SMOOTH is not supported with W8A8_WEIGHTS=offline"
            add_offline_weights(self.language_model.layers, route)
        logger.info(
            "w8a8: mode=%s route=%s smooth=%s offline_weights=%s prefill_min_m=%s (%d layers routed)",
            mode(), spec, smooth_path, offline, os.environ.get("W8A8_PREFILL_MIN_M"), len(route),
        )


ARCHITECTURES = [
    dataclasses.replace(
        _arch.llama_modulev3_arch,
        modules={GraphRole.DECODER: W8A8Llama3Decoder},
        weight_adapters={
            **_arch.llama_modulev3_arch.weight_adapters,
            WeightsFormat.safetensors: convert_safetensor_state_dict,
        },
        pipeline_model=None,  # else the stock decoder is silently used (see fpga/maxop README)
    )
]
