"""Custom MAX architecture: Llama 3 (ModuleV3) with selected projections routed to the phantom NPU.

Use with ``max generate/serve --custom-architectures <abs path to this dir>``.
It re-registers ``LlamaForCausalLM_ModuleV3`` (custom archs register with
allow_override=True), reusing the built-in Llama3 decoder unchanged except that
the routed modules have their class swapped to phantom subclasses. Weight names
are therefore identical to the stock checkpoint.
"""

from __future__ import annotations

import dataclasses
import logging
import os

from max.nn.common_layers.linear import QKVLinear
from max.nn.common_layers.mlp import MLP
from max.nn.linear import Linear
from max.pipelines.architectures.llama3_modulev3 import arch as _arch
from max.pipelines.architectures.llama3_modulev3.llama3 import Llama3Decoder
from max.pipelines.lib import GraphRole
from max.tensor import Tensor

from .phantom_ops import F, mode, parse_route, phantom_linear, site_id

logger = logging.getLogger("max.pipelines")


class PhantomLinear(Linear):
    _phantom_site: int

    def forward(self, x: Tensor) -> Tensor:
        return phantom_linear(x, self.weight, self._phantom_site)


class PhantomQKV(QKVLinear):
    _phantom_site: int

    def forward(self, x: Tensor) -> Tensor:
        return phantom_linear(x, self.fused_weight, self._phantom_site)


class PhantomMLP(MLP):
    """MLP.forward with the fused gate|up and/or the down projection routed."""

    _phantom_layer: int
    _phantom_projs: frozenset

    def forward(self, x: Tensor) -> Tensor:
        ffl = int(self.gate_proj.weight.shape[0])
        w_gu = F.concat((self.gate_proj.weight, self.up_proj.weight))
        if "gate_up" in self._phantom_projs:
            gu = phantom_linear(x, w_gu, site_id(self._phantom_layer, "gate_up"))
        else:
            gu = x @ w_gu.T
        gate, up = F.split(gu, [ffl, ffl], axis=1)
        h = F.mul(self.activation_function(gate), up)
        if "down" in self._phantom_projs:
            return phantom_linear(h, self.down_proj.weight, site_id(self._phantom_layer, "down"))
        return self.down_proj(h)


def apply_route(layers, spec: str) -> dict[int, set[str]]:
    """Swaps the classes of the routed modules in a list of LlamaTransformerBlocks."""
    route = parse_route(spec, len(layers))
    for i, projs in route.items():
        blk = layers[i]
        attn = blk.self_attn
        if "qkv" in projs:
            assert type(attn.qkv_proj) is QKVLinear and not attn.clip_qkv and not attn.has_bias
            attn.qkv_proj.__class__ = PhantomQKV
            object.__setattr__(attn.qkv_proj, "_phantom_site", site_id(i, "qkv"))
        if "o" in projs:
            assert type(attn.o_proj) is Linear
            attn.o_proj.__class__ = PhantomLinear
            object.__setattr__(attn.o_proj, "_phantom_site", site_id(i, "o"))
        mlp_projs = frozenset(projs & {"gate_up", "down"})
        if mlp_projs:
            assert type(blk.mlp) is MLP and not blk.mlp.has_bias, type(blk.mlp)
            blk.mlp.__class__ = PhantomMLP
            object.__setattr__(blk.mlp, "_phantom_layer", i)
            object.__setattr__(blk.mlp, "_phantom_projs", mlp_projs)
    return route


class PhantomLlama3Decoder(Llama3Decoder):
    def __init__(self, config) -> None:
        super().__init__(config)
        spec = os.environ.get("PHANTOM_ROUTE", "0:mlp")
        if mode() == "bf16":
            spec = ""
        route = apply_route(self.language_model.layers, spec)
        logger.info(
            "phantom: mode=%s route=%s (%d layers routed)",
            mode(), spec, len(route),
        )


ARCHITECTURES = [
    dataclasses.replace(
        _arch.llama_modulev3_arch,
        modules={GraphRole.DECODER: PhantomLlama3Decoder},
        # replace() copies the derived pipeline model; clear it so MAX rebuilds
        # it from `modules` (else the stock Llama3Decoder is used).
        pipeline_model=None,
    )
]
