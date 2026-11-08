"""Single source of truth for benchmark shapes. C[M,N] = A[M,K] @ W[N,K]^T (weights stored
[out, in] like nn.Linear). Llama 3.2 3B: hidden 3072, intermediate 8192, vocab 128256,
24 q heads / 8 kv heads x 128 = 1024 for K/V projections."""

SQUARE = [256, 512, 1024, 2048, 4096, 8192]
LLM_KN = {  # name: (K, N)
    "attn_q_o": (3072, 3072),
    "attn_kv": (3072, 1024),
    "mlp_up_gate": (3072, 8192),
    "mlp_down": (8192, 3072),
}
LLM_M = [1, 4, 16, 512]
LM_HEAD = ("lm_head", 3072, 128256)  # optional: M in (1, 512)


def shapes(smoke: bool = False, include_lm_head: bool = True):
    """Yields (tag, M, N, K)."""
    sq = [256, 1024, 2048] if smoke else SQUARE
    for s in sq:
        yield (f"square{s}", s, s, s)
    ms = [1, 512] if smoke else LLM_M
    kn = dict(list(LLM_KN.items())[:2]) if smoke else LLM_KN
    for name, (k, n) in kn.items():
        for m in ms:
            yield (f"{name}_M{m}", m, n, k)
    if include_lm_head and not smoke:
        name, k, n = LM_HEAD
        for m in (1, 512):
            yield (f"{name}_M{m}", m, n, k)


if __name__ == "__main__":
    import os, sys
    smoke = os.environ.get("ATLAS_SMOKE", "0") == "1"
    lm = "--no-lm-head" not in sys.argv
    for t, m, n, k in shapes(smoke, lm):
        print(t, m, n, k)
