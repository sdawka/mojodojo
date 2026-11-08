"""Per-model accounting from HF config.json: parameters, bytes per decode token per weight
format, KV-cache bytes, prefill FLOPs. Pure arithmetic, no I/O beyond reading configs.

Convention: a linear layer is (K -> N), weights W[N, K]; FLOPs = 2*M*N*K.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

# Atlas shape "role" each linear maps to (atlas/shapes.py, Llama 3.2 3B shapes).
ROLE = {"q": "attn_q_o", "o": "attn_q_o", "k": "attn_kv", "v": "attn_kv",
        "gate": "mlp_up_gate", "up": "mlp_up_gate", "down": "mlp_down"}

MODELS = {  # short name -> (config file, HF repo)
    "llama3.2-1b": ("unsloth_Llama-3.2-1B-Instruct.config.json", "unsloth/Llama-3.2-1B-Instruct"),
    "llama3.2-3b": ("unsloth_Llama-3.2-3B-Instruct.config.json", "unsloth/Llama-3.2-3B-Instruct"),
    "qwen3-0.6b": ("Qwen_Qwen3-0.6B.config.json", "Qwen/Qwen3-0.6B"),
    "qwen3-1.7b": ("Qwen_Qwen3-1.7B.config.json", "Qwen/Qwen3-1.7B"),
    "qwen3-4b": ("Qwen_Qwen3-4B.config.json", "Qwen/Qwen3-4B"),
    "gemma3-1b": ("unsloth_gemma-3-1b-it.config.json", "unsloth/gemma-3-1b-it"),
    "gemma3-4b": ("unsloth_gemma-3-4b-it.config.json", "unsloth/gemma-3-4b-it"),
}


@dataclass
class ModelSpec:
    name: str
    repo: str
    d: int
    f: int
    L: int
    n_h: int
    n_kv: int
    hd: int
    V: int
    tied: bool
    window: int | None = None          # sliding window (Gemma 3 local layers)
    global_every: int | None = None    # Gemma 3: every 6th layer is global
    notes: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, name: str) -> "ModelSpec":
        fn, repo = MODELS[name]
        raw = json.loads((CONFIG_DIR / fn).read_text())
        c = raw.get("text_config", raw)
        notes = []
        hd = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
        tied = c.get("tie_word_embeddings", raw.get("tie_word_embeddings"))
        if tied is None:  # Gemma3 config default is tied
            tied = True
            notes.append("tie_word_embeddings absent; Gemma3 default (tied) assumed")
        if "vision_config" in raw:
            notes.append("multimodal checkpoint: vision tower excluded (text-only inference)")
        win = c.get("sliding_window")
        pat = c.get("sliding_window_pattern")
        return cls(name, repo, c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"],
                   c["num_attention_heads"], c["num_key_value_heads"], hd, c["vocab_size"], bool(tied),
                   window=win if pat else None, global_every=pat, notes=notes)

    # ---- shapes ---------------------------------------------------------------------------
    def linears(self) -> dict[str, tuple[int, int]]:
        """Per-layer linears: name -> (K, N)."""
        qd, kvd = self.n_h * self.hd, self.n_kv * self.hd
        return {"q": (self.d, qd), "k": (self.d, kvd), "v": (self.d, kvd), "o": (qd, self.d),
                "gate": (self.d, self.f), "up": (self.d, self.f), "down": (self.f, self.d)}

    def params(self) -> dict[str, int]:
        lin = {k: K * N for k, (K, N) in self.linears().items()}
        attn = (lin["q"] + lin["k"] + lin["v"] + lin["o"]) * self.L
        mlp = (lin["gate"] + lin["up"] + lin["down"]) * self.L
        emb = self.V * self.d
        norms = self.L * 2 * self.d + self.d + (self.L * 2 * self.hd if self.name.startswith(("qwen", "gemma")) else 0)
        return {"embedding": emb, "lm_head_separate": 0 if self.tied else emb, "attn_qkvo": attn,
                "mlp": mlp, "norms": norms, "total_unique": emb * (1 if self.tied else 2) + attn + mlp + norms,
                "decode_read_linear": attn + mlp + emb}  # lm_head is read once per token (tied or not)

    def is_global(self, layer: int) -> bool:
        return self.global_every is None or (layer + 1) % self.global_every == 0

    # ---- KV cache ---------------------------------------------------------------------------
    def kv_bytes_per_token(self, kv_bytes: float = 2.0) -> float:
        """Bytes appended to the KV cache per token (all layers)."""
        return 2 * self.L * self.n_kv * self.hd * kv_bytes

    def kv_read_bytes(self, ctx: int, kv_bytes: float = 2.0) -> float:
        """KV bytes read by one decode step at context length ctx (sliding layers capped)."""
        tot = 0.0
        for i in range(self.L):
            span = ctx if (self.is_global(i) or self.window is None) else min(ctx, self.window)
            tot += 2 * self.n_kv * self.hd * kv_bytes * span
        return tot

    # ---- FLOPs ---------------------------------------------------------------------------
    def attn_prefill_flops(self, S: int, past: int = 0) -> float:
        """QK^T + PV FLOPs for S new tokens after `past` cached, causal."""
        tot = 0.0
        for i in range(self.L):
            w = None if (self.is_global(i) or self.window is None) else self.window
            # sum over new positions p of keys attended
            keys = 0
            for p in range(past, past + S):
                keys += (p + 1) if w is None else min(p + 1, w)
            tot += 2 * 2 * self.n_h * self.hd * keys
        return tot

    def linear_prefill_flops(self, S: int, lm_head_tokens: int = 1) -> float:
        p = self.params()
        return 2 * S * (p["attn_qkvo"] + p["mlp"]) + 2 * lm_head_tokens * self.V * self.d


# ---- weight formats -----------------------------------------------------------------------
# bits per weight for linear layers, incl. scales. Per-role overrides handle mixed formats.
FORMATS = {
    "bf16": {"bpw": 16.0, "lm_head_bpw": 16.0, "desc": "bf16/f16 dense"},
    "f16": {"bpw": 16.0, "lm_head_bpw": 16.0, "desc": "llama.cpp F16"},
    "mlx_8bit_g64": {"bpw": 8.5, "lm_head_bpw": 8.5, "desc": "MLX affine int8, group 64, bf16 scale+bias"},
    "mlx_4bit_g64": {"bpw": 4.5, "lm_head_bpw": 4.5, "desc": "MLX affine int4, group 64, bf16 scale+bias (embedding quantized too)"},
    "mlx_4bit_g128": {"bpw": 4.25, "lm_head_bpw": 4.25, "desc": "MLX affine int4, group 128"},
    "q8_0": {"bpw": 8.5, "lm_head_bpw": 6.5625, "desc": "GGUF Q8_0 (output/tied embd Q6_K by llama-quantize default)"},
    "q6_k": {"bpw": 6.5625, "lm_head_bpw": 6.5625, "desc": "GGUF Q6_K"},
    # Q4_K_M rule (llama-quantize): Q4_K everywhere; Q6_K for half of attn_v and ffn_down layers
    # and for output/tied token_embd. Verified against the local 3B GGUF header (2.0115 GB).
    "q4_k_m": {"bpw": 4.5, "lm_head_bpw": 6.5625, "mixed": {"v": (0.5, 6.5625), "down": (0.5, 6.5625)},
               "desc": "GGUF Q4_K_M (Q4_K + Q6_K for 1/2 of attn_v, ffn_down, and output)"},
    "q4_0": {"bpw": 4.5, "lm_head_bpw": 6.5625, "desc": "GGUF Q4_0 (output Q6_K)"},
    "fp8_w8a16": {"bpw": 8.0, "lm_head_bpw": 16.0, "desc": "MAX float8_e4m3fn weight-only; lm_head kept bf16 (modelopt default)"},
    "nvfp4_w4a16": {"bpw": 4.5, "lm_head_bpw": 16.0, "desc": "MAX float4_e2m1fnx2 (NVFP4, fp8 scale per 16); lm_head kept bf16 (modelopt default)"},
    "int8_w8a8": {"bpw": 8.0, "lm_head_bpw": 16.0, "desc": "MAX Apple int8 W8A8 (per-channel weight scale, dynamic per-token act); lm_head bf16"},
}


def role_bytes(spec: ModelSpec, fmt: str) -> dict[str, float]:
    """Weight bytes read per decode token, by atlas role (summed over layers) + lm_head."""
    F = FORMATS[fmt]
    out: dict[str, float] = {}
    for name, (K, N) in spec.linears().items():
        n = K * N * spec.L
        if "mixed" in F and name in F["mixed"]:
            frac, bpw2 = F["mixed"][name]
            b = n * (frac * bpw2 + (1 - frac) * F["bpw"]) / 8
        else:
            b = n * F["bpw"] / 8
        out[ROLE[name]] = out.get(ROLE[name], 0.0) + b
    out["lm_head"] = spec.V * spec.d * F["lm_head_bpw"] / 8
    return out


def role_flops_prefill(spec: ModelSpec, S: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, (K, N) in spec.linears().items():
        out[ROLE[name]] = out.get(ROLE[name], 0.0) + 2.0 * S * K * N * spec.L
    return out


def summary(spec: ModelSpec) -> dict:
    p = spec.params()
    dec = {fmt: sum(role_bytes(spec, fmt).values()) for fmt in FORMATS}
    return {
        "repo": spec.repo,
        "arch": {k: getattr(spec, k) for k in ("d", "f", "L", "n_h", "n_kv", "hd", "V", "tied", "window", "global_every")},
        "params": p,
        "decode_weight_bytes_per_token": dec,
        "kv_bytes_per_token_fp16": spec.kv_bytes_per_token(),
        "kv_bytes_at_ctx_fp16": {c: spec.kv_read_bytes(c) for c in (512, 2048, 8192, 32768)},
        "prefill_flops": {S: {"linear": spec.linear_prefill_flops(S), "attention": spec.attn_prefill_flops(S)}
                          for S in (128, 512, 2048)},
        "notes": spec.notes,
    }
