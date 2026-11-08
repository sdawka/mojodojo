"""Read the authoritative processor-atlas JSON and expose shape-specific achieved rates.

A "path" is (engine kernel family, dtype) as measured in atlas/results:
  mlx_nax / mlx_nonax  : gpu_mlx_default.json / gpu_mlx_nonax_g16g.json (dtype bf16, int4w(g64)xbf16, ...)
  max_gemv / max_mma / max_int8 / max_fp4 / max_8x8 : gpu_max.json kernels
  ggml_mtl0            : ggml_mulmat.json (one shape, 4096x14336, used for every role)
  ane_<dtype>          : ane_coreml.json, compute units CPU_AND_NE
For M <= 16 we use weight-bytes throughput (GB/s of *weight* bytes in that dtype); for
M = 512 we use GFLOP/s. Between measured M we interpolate time-per-call linearly in M.
"""
from __future__ import annotations

import json
from pathlib import Path

ATLAS = Path(__file__).resolve().parents[2] / "atlas" / "results"
FILES = ["gpu_mlx_default.json", "gpu_mlx_nonax_g16g.json", "gpu_max.json", "ggml_mulmat.json",
         "ane_coreml.json", "cpu_bandwidth.json", "cpu_accelerate.json"]

KERNEL_PATH = {"apple_gemv": "max_gemv", "apple_m5_mma": "max_mma", "apple_int8_w8a8": "max_int8",
               "apple_fp4_w4a16": "max_fp4", "simdgroup_8x8": "max_8x8"}

# weight bits per element as stored by each benchmarked dtype (for converting to weight GB/s)
DTYPE_BPW = {"bf16": 16, "f16": 16, "f32": 32, "bfloat16": 16, "float16": 16, "float32": 32,
             "int4w(g64)xbf16": 4.5, "int8w(g64)xbf16": 8.5, "int8": 8, "nvfp4w x bf16": 4.5,
             "f16w x f32": 16, "bf16w x f32": 16, "q4_0w x f32": 4.5, "q4_Kw x f32": 4.5,
             "q8_0w x f32": 8.5, "fp16": 16, "w8": 8, "w8a8": 8, "lut4": 4, "w4": 4.5}

ROLE_SHAPES = ("attn_q_o", "attn_kv", "mlp_up_gate", "mlp_down", "lm_head")


class Atlas:
    def __init__(self, root: Path = ATLAS):
        self.root = root
        self.inputs = [str(root / f) for f in FILES]
        self.rec: dict[tuple[str, str, str, int], dict] = {}
        self.bw: dict[str, float] = {}
        for fn in FILES:
            d = json.loads((root / fn).read_text())
            assert d["meta"].get("authoritative", False), fn
            for r in d["records"]:
                if r["bench"].endswith("bandwidth"):
                    self.bw[f"{r['engine']}_{r['op']}"] = max(self.bw.get(f"{r['engine']}_{r['op']}", 0), r["median"])
                    continue
                if r["bench"] != "gemm":
                    continue
                eng = r["engine"]
                if eng == "gpu_mlx":
                    path = "mlx_nax" if r["mode"] == "default" else "mlx_nonax"
                elif eng == "gpu_max":
                    path = KERNEL_PATH[r["kernel"]]
                elif eng == "ggml_mtl0":
                    path = "ggml_mtl0"
                elif eng == "coreml_CPU_AND_NE":
                    path = "ane"
                elif eng == "coreml_CPU_AND_GPU":
                    path = "coreml_gpu"
                else:
                    continue
                shape = r["shape"].rsplit("_M", 1)[0] if "_M" in r["shape"] else r["shape"]
                self.rec[(path, r["dtype"], shape, int(r["M"]))] = r

    # ------------------------------------------------------------------------------------------
    def _lookup(self, path: str, dtype: str, role: str) -> dict[int, dict]:
        if path == "ggml_mtl0":
            role = "ggml"
        if path in ("ane", "coreml_gpu"):
            role = "attn_q_o" if role.startswith("attn") else "mlp_up_down"
        out = {M: r for (p, dt, s, M), r in self.rec.items() if p == path and dt == dtype and s == role}
        if not out and role == "lm_head":  # e.g. max_gemv has lm_head; others fall back to mlp_up_gate
            return self._lookup(path, dtype, "mlp_up_gate")
        return out

    def has(self, path: str, dtype: str, role: str, M: int) -> bool:
        return M in self._lookup(path, dtype, role)

    def gflops(self, path: str, dtype: str, role: str, M: int) -> float:
        r = self._lookup(path, dtype, role)[M]
        return r["median"]

    def time_per_weight_byte(self, path: str, dtype: str, role: str, M: int) -> float:
        """Seconds per weight byte at token count M (M small: memory-bound regime).
        Interpolates time/call linearly between measured M (1, 4, 16); beyond the largest
        measured small M, interpolates toward the M=512 compute rate."""
        recs = self._lookup(path, dtype, role)
        if not recs:
            raise KeyError((path, dtype, role))
        if role == "lm_head" and M not in recs and M < 512:
            return self.time_per_weight_byte(path, dtype, "mlp_up_gate", M)
        bpw = DTYPE_BPW[dtype]

        def spb(Mx):  # seconds per weight byte at measured Mx
            r = recs[Mx]
            flops = 2.0 * r["M"] * r["N"] * r["K"]
            t = flops / (r["median"] * 1e9)
            wbytes = r["N"] * r["K"] * bpw / 8
            return t / wbytes

        Ms = sorted(recs)
        if M in recs:
            return spb(M)
        lo = max([m for m in Ms if m < M], default=None)
        hi = min([m for m in Ms if m > M], default=None)
        if lo is None:
            return spb(hi)
        if hi is None:
            # extrapolate: per-byte time grows ~linearly in M once compute-bound
            return spb(lo) * M / lo
        w = (M - lo) / (hi - lo)
        return spb(lo) * (1 - w) + spb(hi) * w

    def weight_gbps(self, path: str, dtype: str, role: str, M: int = 1) -> float:
        return 1e-9 / self.time_per_weight_byte(path, dtype, role, M)

    def stream_bw(self) -> float:
        """Best GPU streaming read bandwidth (GB/s)."""
        return max(v for k, v in self.bw.items() if k.startswith("gpu_mlx") and "read" in k)
