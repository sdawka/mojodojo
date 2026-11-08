"""Roofline predictor: decode time/token and TTFT for (model, runtime, weight format).

decode  t_tok = ( sum_parts bytes_p * s_p(M) + KV_read / BW_stream ) / eta_d + c_d
prefill t_pf  = ( sum_parts flops_p / R_p(M=S) + attn_flops / R_attn ) / eta_p
TTFT          = t_pf + method terms (+ c_http for HTTP endpoints)

s_p(M) is the atlas seconds-per-weight-byte for that engine/dtype/shape at M tokens, R_p the
atlas GFLOP/s at M=512. eta_d, c_d, eta_p, c_http are the only fitted parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .accounting import FORMATS, ROLE, ModelSpec

# --------------------------------------------------------------------------------------------
# Runtime configurations: which atlas path/dtype each weight part runs on.
# part kinds: "main" (bulk), "q6k" (Q6_K portions of GGUF mixes), "lm_head".
# Each value is a list of (path, dtype); the per-byte time is averaged over the list
# (used only where the atlas lacks the exact kernel; such proxies are flagged in `proxy`).


@dataclass
class Runtime:
    key: str
    runtime: str
    fmt: str
    decode: dict[str, list[tuple[str, str]]]
    prefill: dict[str, list[tuple[str, str]]]
    attn_ref: tuple[str, str]           # 16-bit GEMM path whose M=512 rate * ATTN_FRAC is used for attention
    proxy: list[str] = field(default_factory=list)
    supported: str = "yes"
    prefill_scale: float = 1.0          # explicit extra multiplier (e.g. dequant pass), documented


SPEC_ITER_OVERHEAD_MS = 1.0  # assumed: extra sync/rollback/bookkeeping per draft-verify iteration
ATTN_FRAC = 0.5  # assumed: fused attention reaches 50% of the engine's dense 16-bit GEMM rate


def _same(p):
    return {"main": [p], "q6k": [p], "lm_head": [p]}


RUNTIMES: dict[str, Runtime] = {}


def _add(rt: Runtime):
    RUNTIMES[rt.key] = rt


MLX_BF16 = ("mlx_nax", "bf16")
MLX_I4 = ("mlx_nax", "int4w(g64)xbf16")
MLX_I8 = ("mlx_nax", "int8w(g64)xbf16")
GG = lambda dt: ("ggml_mtl0", dt)  # noqa: E731
Q6K_PROXY = [GG("q4_Kw x f32"), GG("q8_0w x f32")]
MAX_GEMV, MAX_MMA = ("max_gemv", "bfloat16"), ("max_mma", "bfloat16")

_add(Runtime("mlx/bf16", "mlx", "bf16", _same(MLX_BF16), _same(MLX_BF16), MLX_BF16))
_add(Runtime("mlx/8bit", "mlx", "mlx_8bit_g64", _same(MLX_I8), _same(MLX_I8), MLX_BF16))
_add(Runtime("mlx/4bit", "mlx", "mlx_4bit_g64", _same(MLX_I4), _same(MLX_I4), MLX_BF16))
_add(Runtime("mlx/4bit-g128", "mlx", "mlx_4bit_g128", _same(MLX_I4), _same(MLX_I4), MLX_BF16,
             proxy=["g128 kernel unmeasured: g64 per-byte rate used"]))
_add(Runtime("llama.cpp/F16", "llama.cpp", "f16", _same(GG("f16w x f32")), _same(GG("f16w x f32")), GG("f16w x f32")))
_add(Runtime("llama.cpp/Q8_0", "llama.cpp", "q8_0",
             {"main": [GG("q8_0w x f32")], "q6k": Q6K_PROXY, "lm_head": Q6K_PROXY},
             {"main": [GG("q8_0w x f32")], "q6k": Q6K_PROXY, "lm_head": Q6K_PROXY}, GG("f16w x f32"),
             proxy=["Q6_K output tensor: mean of Q4_K/Q8_0 rates"]))
_add(Runtime("llama.cpp/Q6_K", "llama.cpp", "q6_k", {k: Q6K_PROXY for k in ("main", "q6k", "lm_head")},
             {k: Q6K_PROXY for k in ("main", "q6k", "lm_head")}, GG("f16w x f32"),
             proxy=["Q6_K unmeasured: mean of Q4_K/Q8_0 per-byte rates"]))
_add(Runtime("llama.cpp/Q4_K_M", "llama.cpp", "q4_k_m",
             {"main": [GG("q4_Kw x f32")], "q6k": Q6K_PROXY, "lm_head": Q6K_PROXY},
             {"main": [GG("q4_Kw x f32")], "q6k": Q6K_PROXY, "lm_head": Q6K_PROXY}, GG("f16w x f32"),
             proxy=["Q6_K parts: mean of Q4_K/Q8_0 rates"]))
_add(Runtime("llama.cpp/Q4_0", "llama.cpp", "q4_0",
             {"main": [GG("q4_0w x f32")], "q6k": Q6K_PROXY, "lm_head": Q6K_PROXY},
             {"main": [GG("q4_0w x f32")], "q6k": Q6K_PROXY, "lm_head": Q6K_PROXY}, GG("f16w x f32"),
             proxy=["Q6_K output: mean of Q4_K/Q8_0 rates"]))
_add(Runtime("max/bf16", "max", "bf16", _same(MAX_GEMV), _same(MAX_MMA), MAX_MMA))
_add(Runtime("max/fp8", "max", "fp8_w8a16",
             {"main": [MAX_GEMV], "q6k": [MAX_GEMV], "lm_head": [MAX_GEMV]},
             {"main": [MAX_MMA], "q6k": [MAX_MMA], "lm_head": [MAX_MMA]}, MAX_MMA,
             proxy=["FP8 W8A16 Apple kernel not in atlas: bf16 GEMV per-byte rate (decode) and bf16 MMA rate x1/0.9 (prefill, dequant to transient bf16) assumed"],
             supported="encoding float8_e4m3fn listed for legacy Llama3 arch (not ModuleV3); needs an FP8 checkpoint with per-tensor scale", prefill_scale=1 / 0.9))
_add(Runtime("max/nvfp4", "max", "nvfp4_w4a16",
             {"main": [("max_fp4", "nvfp4w x bf16")], "q6k": [("max_fp4", "nvfp4w x bf16")], "lm_head": [MAX_GEMV]},
             {"main": [("max_fp4", "nvfp4w x bf16")], "q6k": [("max_fp4", "nvfp4w x bf16")], "lm_head": [MAX_MMA]}, MAX_MMA,
             supported="encoding float4_e2m1fnx2 listed for legacy Llama3 arch; Apple W4A16 path only exercised by FLUX.2 so far; needs NVFP4 (modelopt) checkpoint"))
_add(Runtime("max/int8-w8a8", "max", "int8_w8a8",
             {"main": [("max_int8", "int8")], "q6k": [("max_int8", "int8")], "lm_head": [MAX_GEMV]},
             {"main": [("max_int8", "int8")], "q6k": [("max_int8", "int8")], "lm_head": [MAX_MMA]}, MAX_MMA,
             supported="NOT a pipeline encoding: op exists (max.nn quant int8 W8A8, Apple M5 only); needs a custom Llama graph + offline weight quantization"))
# Core ML / ANE prefill only (decode stays on GPU). Attention reference = ANE fp16 linear rate.
for dt, fmt in (("w8a8", "int8_w8a8"), ("w8", "fp8_w8a16"), ("fp16", "bf16"), ("lut4", "mlx_4bit_g64")):
    _add(Runtime(f"ane/{dt}", "coreml-ane", fmt, {}, _same(("ane", dt)), ("ane", "fp16"),
                 proxy=["ANE shapes are 3B attn_q_o / mlp_up_down chains of 8 linears; attention on ANE assumed at 50% of ANE fp16 linear rate; KV handoff to GPU not modeled (unified memory)"],
                 supported="whole-model Core ML conversion required (coremltools); not tried"))


# --------------------------------------------------------------------------------------------
def parts(spec: ModelSpec, fmt: str):
    """Yield (part_kind, role, weight_bytes, n_weights) per decode token."""
    F = FORMATS[fmt]
    for name, (K, N) in spec.linears().items():
        n = K * N * spec.L
        if "mixed" in F and name in F["mixed"]:
            frac, bpw2 = F["mixed"][name]
            yield ("q6k", ROLE[name], n * frac * bpw2 / 8, n * frac)
            yield ("main", ROLE[name], n * (1 - frac) * F["bpw"] / 8, n * (1 - frac))
        else:
            yield ("main", ROLE[name], n * F["bpw"] / 8, n)
    yield ("lm_head", "lm_head", spec.V * spec.d * F["lm_head_bpw"] / 8, spec.V * spec.d)


@dataclass
class Params:
    eta_d: float = 1.0
    c_d_ms: float = 0.0
    eta_p: float = 1.0
    c_http_ms: float = 0.0


class Predictor:
    def __init__(self, atlas):
        self.a = atlas
        self.bw = atlas.stream_bw()

    def _spb(self, pairs, role, M):
        return float(np.mean([self.a.time_per_weight_byte(p, dt, role, M) for p, dt in pairs]))

    def _rate(self, pairs, role, M):
        """Effective GFLOP/s at M (via time per atlas-weight-byte)."""
        from .atlas import DTYPE_BPW
        ts = []
        for p, dt in pairs:
            spb = self.a.time_per_weight_byte(p, dt, role, M)
            ts.append(spb * DTYPE_BPW[dt] / 8 / (2.0 * M))  # seconds per FLOP
        return 1e-9 / float(np.mean(ts))

    # ---- decode -------------------------------------------------------------------------
    def decode_ideal(self, spec, rt: Runtime, ctx: int, M: int = 1) -> dict:
        """Ideal (unfitted) time for one forward pass over M tokens at context ctx, seconds."""
        br = {}
        for kind, role, b, n in parts(spec, rt.fmt):
            pairs = rt.decode[kind]
            if M <= 16:
                t = b * self._spb(pairs, role, M)
            else:
                t = 2.0 * M * n / (self._rate(pairs, role, M) * 1e9)
            br[role] = br.get(role, 0.0) + t
        br["kv_read"] = spec.kv_read_bytes(ctx) / (self.bw * 1e9)  # KV is read once per pass
        return br

    def decode_ms(self, spec, rt, ctx, P: Params, M: int = 1) -> float:
        # c_d is fitted on the 28-layer 3B model; scaled per layer for other depths (assumption)
        return sum(self.decode_ideal(spec, rt, ctx, M).values()) * 1e3 / P.eta_d + P.c_d_ms * spec.L / 28

    # ---- prefill ------------------------------------------------------------------------
    def prefill_ideal(self, spec, rt: Runtime, S: int, lm_head_tokens: int = 1) -> dict:
        br = {}
        Mq = 512 if S >= 512 else S
        for kind, role, b, n in parts(spec, rt.fmt):
            pairs = rt.prefill[kind]
            if role == "lm_head":
                if lm_head_tokens == 0:
                    continue
                if lm_head_tokens == 1:  # one GEMV over lm_head: use the decode (M=1) path
                    dpairs = rt.decode.get("lm_head") or pairs
                    t = b * self._spb(dpairs, role, 1)
                else:
                    t = 2.0 * lm_head_tokens * n / (self._rate(pairs, role, 512) * 1e9)
            else:
                t = 2.0 * S * n / (self._rate(pairs, role, Mq) * 1e9) * rt.prefill_scale
            br[role] = br.get(role, 0.0) + t
        ref = [rt.attn_ref]
        r_attn = ATTN_FRAC * np.mean([self._rate(ref, r, 512) for r in ("attn_q_o", "mlp_up_gate", "mlp_down")])
        br["attention"] = spec.attn_prefill_flops(S) / (r_attn * 1e9)
        return br

    def prefill_ms(self, spec, rt, S, P: Params, lm_head_tokens: int = 1) -> float:
        return sum(self.prefill_ideal(spec, rt, S, lm_head_tokens).values()) * 1e3 / P.eta_p

    # ---- speculative decoding ----------------------------------------------------------
    def spec_decode(self, tspec, trt, dspec, drt, k: int, alpha: float, ctx: int, P: Params) -> dict:
        """Greedy speculative decoding, k draft tokens, i.i.d. per-token acceptance alpha.
        Per iteration: k draft steps (M=1) + 1 target verify over M=k+1 tokens.
        Expected tokens per iteration E = (1 - alpha^(k+1)) / (1 - alpha)."""
        E = (1 - alpha ** (k + 1)) / (1 - alpha) if alpha < 1 else k + 1
        t_d = self.decode_ms(dspec, drt, ctx, P, 1)
        t_v = self.decode_ms(tspec, trt, ctx, P, k + 1)
        t_iter = k * t_d + t_v + SPEC_ITER_OVERHEAD_MS
        return {"k": k, "alpha": alpha, "E_tokens": E, "t_draft_ms": t_d, "t_verify_ms": t_v,
                "t_iter_ms": t_iter, "tok_s": 1e3 * E / t_iter}


# --------------------------------------------------------------------------------------------
# Fitting helpers: weighted (relative-error) linear least squares y ~ X @ beta.
def wls_rel(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    W = 1.0 / y
    beta, *_ = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)
    return beta
