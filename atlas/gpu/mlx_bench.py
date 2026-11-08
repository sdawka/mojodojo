"""MLX (Metal GPU) bandwidth + matmul throughput.

Modes (one per process, because MLX reads these env vars once at startup):
  default            : MLX's own dispatch. On M5 (applegpu_g17*) + macOS >= 26.2 this uses the
                       "NAX" kernels (Metal 4 tensor ops on the GPU Neural Accelerators).
                       fp32 matmul uses TF32 on NAX unless MLX_ENABLE_TF32=0.
  MLX_METAL_GPU_ARCH=applegpu_g16g : pretend to be an M4-class GPU -> NAX disabled, classic
                       simdgroup_matrix kernels. Results are checked against the default path.
  MLX_ENABLE_TF32=0  : full-precision fp32.

Timing: each rep enqueues `inner` independent matmuls (cycling weight copies so memory-bound
shapes are cache-cold) and evaluates them together -> pipelined throughput per op, which is
how a model graph executes. Launch overhead is therefore amortized, not excluded.

usage: mlx_bench.py --label LABEL [--bandwidth] [--matmul]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import atlas_common as ac  # noqa: E402
import shapes  # noqa: E402

DT = {"f32": mx.float32, "f16": mx.float16, "bf16": mx.bfloat16}


def timed(run_inner, inner: int) -> list[float]:
    """Returns per-op seconds for each rep."""
    for _ in range(ac.WARMUP):
        run_inner(inner)
    out = []
    for _ in range(ac.REPS):
        t0 = time.perf_counter()
        run_inner(inner)
        out.append((time.perf_counter() - t0) / inner)
    return out


def bandwidth() -> list[dict]:
    recs = []
    mib = 64 if ac.SMOKE else 512
    n = mib * (1 << 20) // 4
    x = mx.random.normal((n,), dtype=mx.float32)
    mx.eval(x)
    y = mx.random.normal((n,), dtype=mx.float32)
    mx.eval(y)
    triad = mx.compile(lambda b, c: b + 3.0 * c)  # compiled -> one fused kernel
    tests = {
        # name: (fn, bytes moved)
        "copy": (lambda: x * 1.0, 2 * n * 4),  # read + write
        "read_reduce": (lambda: mx.sum(x), n * 4),
        "triad": (lambda: triad(x, y), 3 * n * 4),
    }
    for name, (fn, nbytes) in tests.items():
        def run(k, fn=fn):
            mx.eval([fn() for _ in range(k)])

        inner = ac.calibrate_inner(run)
        t = timed(run, inner)
        st = ac.stats([nbytes / s / 1e9 for s in t])
        recs.append({"bench": "gpu_bandwidth", "engine": "gpu_mlx", "op": name, "dtype": "f32",
                     "unit": "GB/s", **st, "array_mib": mib, "bytes_per_op": nbytes, "inner_iters": inner})
        print(f"  bw {name:12s} {st['median']:8.1f} GB/s (sd {st['stddev']:.1f})", file=sys.stderr)
    return recs


def matmul_records(label: str) -> list[dict]:
    recs = []
    lm = os.environ.get("ATLAS_LM_HEAD", "1") == "1"
    variants = [("f32", None), ("f16", None), ("bf16", None), ("q8", 8), ("q4", 4)]
    if os.environ.get("MLX_ENABLE_TF32") == "0":
        variants = [v for v in variants if v[0] == "f32"]  # only fp32 is affected by TF32
    elif label != "default":
        variants = [v for v in variants if v[0] in ("f32", "f16", "bf16", "q4")]
    for tag, m, n, k in shapes.shapes(ac.SMOKE, lm):
        for dname, bits in variants:
            in_dt = mx.bfloat16 if bits else DT[dname]
            a = mx.random.normal((m, k), dtype=mx.float32).astype(in_dt)
            if bits:
                w0 = mx.random.normal((n, k), dtype=mx.float32).astype(mx.bfloat16)
                wq, sc, bi = mx.quantize(w0, group_size=64, bits=bits)
                wbytes = wq.nbytes + sc.nbytes + bi.nbytes
            else:
                w0 = mx.random.normal((n, k), dtype=mx.float32).astype(in_dt)
                wbytes = w0.nbytes
            nc = ac.ncopies_for(wbytes)
            if bits:
                # `x + 0` materializes a distinct buffer (MLX does no CSE in eager mode).
                ws = [(wq, sc, bi)] + [(wq + 0, sc + 0, bi + 0) for _ in range(nc - 1)]
                mx.eval(ws)

                def op(i, a=a, ws=ws, bits=bits):
                    q, s_, b_ = ws[i % len(ws)]
                    return mx.quantized_matmul(a, q, s_, b_, transpose=True, group_size=64, bits=bits)
            else:
                ws = [w0] + [w0 * 1 for _ in range(nc - 1)]
                mx.eval(ws)

                def op(i, a=a, ws=ws):
                    return a @ ws[i % len(ws)].T
            mx.eval(a)

            def run(cnt, op=op):
                mx.eval([op(i) for i in range(cnt)])

            inner = ac.calibrate_inner(run, cap=4096)
            t = timed(run, inner)
            flops = 2.0 * m * n * k
            nbytes = wbytes + a.nbytes + m * n * a.itemsize
            st = ac.stats([flops / s / 1e9 for s in t])
            bw = ac.stats([nbytes / s / 1e9 for s in t])
            rec = {"bench": "gemm", "engine": "gpu_mlx", "op": "matmul" if not bits else "quantized_matmul",
                   "dtype": dname if not bits else f"int{bits}w(g64)xbf16", "unit": "GFLOP/s", **st,
                   "shape": tag, "M": m, "N": n, "K": k, "mode": label, "weight_copies": nc,
                   "inner_iters": inner, "eff_gbps_median": bw["median"], "eff_gbps_max": bw["max"],
                   "bytes_per_call": nbytes}
            recs.append(rec)
            print(f"  {label:8s} {rec['dtype']:16s} {tag:18s} {st['median']:9.1f} GF/s  "
                  f"{bw['median']:7.1f} GB/s eff (sd {st['stddev']:.1f}, inner {inner})", file=sys.stderr)
            del ws
    return recs


def correctness() -> dict:
    """Max relative error vs float64 numpy for a 512^3 product, per dtype (records which path ran)."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((512, 512)).astype(np.float32)
    b = rng.standard_normal((512, 512)).astype(np.float32)
    ref = a.astype(np.float64) @ b.astype(np.float64).T
    out = {}
    for d, dt in DT.items():
        r = np.array((mx.array(a).astype(dt) @ mx.array(b).astype(dt).T).astype(mx.float32))
        out[d] = float(np.max(np.abs(r - ref)) / np.max(np.abs(ref)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="default")
    ap.add_argument("--bandwidth", action="store_true")
    ap.add_argument("--matmul", action="store_true")
    args = ap.parse_args()
    info = mx.device_info()
    recs = []
    if args.bandwidth:
        recs += bandwidth()
    if args.matmul:
        recs += matmul_records(args.label)
    err = correctness()
    print(f"  max rel err vs fp64 (512^3): {err}", file=sys.stderr)
    import importlib.metadata as md
    ac.write_results(f"gpu_mlx_{args.label}" + ("_bw" if args.bandwidth and not args.matmul else ""),
                     recs, {"mlx_version": md.version("mlx"), "device_info": info,
                            "label": args.label, "max_rel_err_512": err})


if __name__ == "__main__":
    main()
