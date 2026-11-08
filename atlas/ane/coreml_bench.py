"""Apple Neural Engine (and Core ML's CPU / GPU backends) matmul throughput via coremltools.

Each model is a chain of L `linear` layers alternating K->N and N->K (distinct weights, so the
working set is L x weight bytes and per-predict Python overhead is amortized over L layers).
FLOPs counted = sum over layers of 2*M*N*K. Variants:
  fp16      : fp16 weights + fp16 compute
  w8        : int8 per-channel linear-quantized weights (fp16 compute; weights decompressed)
  w4        : 4-bit per-block (32) linear-quantized weights (iOS18 / macOS15 op set)
  lut4      : 4-bit palettized weights (16-entry LUT, uniform levels)
  w8a8      : int8 weights + int8 activations (linear_quantize_activations, calibrated on
              random data) -- the dtype Apple quotes ANE TOPS in.
Compute units: CPU_ONLY, CPU_AND_GPU, CPU_AND_NE, ALL.

Placement: Core ML does not report where an op *ran*. We record (a) MLComputePlan's planned
device per linear op (coremltools.models.compute_plan, available without Xcode) and (b) timing
differences between compute-unit settings. Neither is proof of execution on the ANE; the plan is
Core ML's own placement decision for that compute-unit setting, which is the best evidence
available without Instruments.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

import atlas_common as ac  # noqa: E402

import coremltools as ct  # noqa: E402
import coremltools.optimize.coreml as cto  # noqa: E402
from coremltools.converters.mil import Builder as mb  # noqa: E402
from coremltools.converters.mil.mil import types  # noqa: E402
from coremltools.models.compute_plan import MLComputePlan  # noqa: E402

UNITS = {
    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    "ALL": ct.ComputeUnit.ALL,
}


def ane_shapes(smoke: bool):
    """(tag, M, N, K, L)."""
    if smoke:
        return [("square1024", 64, 1024, 1024, 4), ("attn_q_o_M1", 1, 3072, 3072, 4),
                ("attn_q_o_M512", 512, 3072, 3072, 4)]
    out = []
    for s in (512, 1024, 2048, 4096):
        out.append((f"square{s}", s, s, s, 8 if s <= 2048 else 4))
    for m in (1, 16, 512):
        out.append((f"attn_q_o_M{m}", m, 3072, 3072, 8))
        out.append((f"mlp_up_down_M{m}", m, 8192, 3072, 8))  # alternating 3072->8192->3072
    return out


def build(m: int, n: int, k: int, layers: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    ws = []
    for i in range(layers):
        o, inp = (n, k) if i % 2 == 0 else (k, n)
        ws.append((rng.standard_normal((o, inp)) / np.sqrt(inp)).astype(np.float16))

    @mb.program(input_specs=[mb.TensorSpec(shape=(m, k), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def prog(x):
        y = x
        for i, w in enumerate(ws):
            y = mb.linear(x=y, weight=w, name=f"lin{i}")
        return y

    model = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                       compute_precision=ct.precision.FLOAT16)
    flops = sum(2.0 * m * w.shape[0] * w.shape[1] for w in ws)
    wbytes_fp16 = sum(w.nbytes for w in ws)
    return model, flops, wbytes_fp16


def variant(model, name: str, m: int, k: int):
    if name == "fp16":
        return model
    if name == "w8":
        cfg = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_channel"))
        return cto.linear_quantize_weights(model, cfg)
    if name == "w4":
        cfg = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int4", granularity="per_block", block_size=32))
        return cto.linear_quantize_weights(model, cfg)
    if name == "lut4":  # 4-bit palettization (LUT), uniform levels: the ANE-friendly 4-bit format
        cfg = cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(
            mode="uniform", nbits=4, granularity="per_tensor"))
        return cto.palettize_weights(model, cfg)
    if name == "w8a8":
        wq = variant(model, "w8", m, k)
        cfg = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8"))
        sample = [{"x": np.random.default_rng(i).standard_normal((m, k)).astype(np.float16)}
                  for i in range(4)]
        return cto.linear_quantize_activations(wq, cfg, sample_data=sample)
    raise ValueError(name)


def planned_devices(compiled_path: str, unit) -> dict:
    """Counts of MLComputePlan preferred devices for the linear ops."""
    plan = MLComputePlan.load_from_path(compiled_path, compute_units=unit)
    fn = plan.model_structure.program.functions["main"]
    counts: dict[str, int] = {}
    for op in fn.block.operations:
        if op.operator_name not in ("linear", "ios18.linear", "matmul", "ios18.matmul"):
            continue
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        dev = type(usage.preferred_compute_device).__name__ if usage else "None"
        counts[dev] = counts.get(dev, 0) + 1
    return counts


def bench_one(pkg_path: str, unit_name: str, m: int, k: int):
    t0 = time.perf_counter()
    mdl = ct.models.MLModel(pkg_path, compute_units=UNITS[unit_name])
    load_s = time.perf_counter() - t0
    x = {"x": np.random.default_rng(1).standard_normal((m, k)).astype(np.float16)}

    def run(cnt):
        for _ in range(cnt):
            mdl.predict(x)

    inner = ac.calibrate_inner(run, cap=2000)
    for _ in range(ac.WARMUP):
        run(inner)
    t = []
    for _ in range(ac.REPS):
        s = time.perf_counter()
        run(inner)
        t.append((time.perf_counter() - s) / inner)
    compiled = mdl.get_compiled_model_path()
    try:
        plan = planned_devices(compiled, UNITS[unit_name])
    except Exception as e:  # noqa: BLE001
        plan = {"error": repr(e)[:200]}
    return t, inner, load_s, plan


def main():
    variants = ["fp16", "w8", "w4", "lut4", "w8a8"]
    if ac.SMOKE:
        variants = ["fp16", "w8"]
    units = list(UNITS)
    recs = []
    tmp = Path(tempfile.mkdtemp(prefix="atlas_coreml_"))
    try:
        for tag, m, n, k, layers in ane_shapes(ac.SMOKE):
            base, flops, wb16 = build(m, n, k, layers)
            for vname in variants:
                try:
                    mdl = variant(base, vname, m, k)
                except Exception as e:  # noqa: BLE001
                    recs.append({"bench": "gemm", "engine": "coreml", "op": "linear_chain",
                                 "dtype": vname, "shape": tag, "M": m, "N": n, "K": k,
                                 "error": f"quantization failed: {e!r}"[:300]})
                    continue
                pkg = tmp / f"{tag}_{vname}.mlpackage"
                mdl.save(str(pkg))
                wbytes = {"fp16": wb16, "w8": wb16 / 2, "w4": wb16 / 4, "lut4": wb16 / 4, "w8a8": wb16 / 2}[vname]
                for u in units:
                    try:
                        t, inner, load_s, plan = bench_one(str(pkg), u, m, k)
                    except Exception as e:  # noqa: BLE001
                        recs.append({"bench": "gemm", "engine": f"coreml_{u}", "op": "linear_chain",
                                     "dtype": vname, "shape": tag, "M": m, "N": n, "K": k,
                                     "error": repr(e)[:300]})
                        traceback.print_exc()
                        continue
                    st = ac.stats([flops / s / 1e9 for s in t])
                    bw = ac.stats([wbytes / s / 1e9 for s in t])
                    rec = {"bench": "gemm", "engine": f"coreml_{u}", "op": "linear_chain",
                           "dtype": vname, "unit": "GFLOP/s", **st, "shape": tag, "M": m, "N": n,
                           "K": k, "layers": layers, "flops_per_predict": flops,
                           "weight_bytes": wbytes, "eff_gbps_median": bw["median"],
                           "eff_gbps_max": bw["max"], "inner_iters": inner,
                           "predict_ms_median": 1e3 * float(np.median(t)),
                           "load_compile_s": load_s, "planned_device_counts": plan}
                    recs.append(rec)
                    print(f"  {u:12s} {vname:5s} {tag:18s} L={layers} {st['median']:9.1f} GF/s "
                          f"{1e3 * np.median(t):8.3f} ms/predict  plan={plan}", file=sys.stderr)
            del base
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    import importlib.metadata as md
    ac.write_results("ane_coreml", recs, {"coremltools_version": md.version("coremltools"),
                                          "note": "ANE placement inferred from MLComputePlan + timing; "
                                                  "not observed (no Xcode/Instruments)."})


if __name__ == "__main__":
    main()
