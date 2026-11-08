"""maxop tests (`pixi run maxop-test`).

1. op_exact   - the MAX custom op `phantom_matmul_i8`, fed by the in-graph W8A8
                quantizer, on every routable projection of layers 0 and 27 with the
                real Llama 3.2 3B weights (full K x N_out, M=37 tokens): int8
                operands equal an independent numpy quantizer and the int32 result is
                bit-exact vs an exact integer matmul; the dequantized bf16 output
                matches the numpy dequant.
2. spot_check - windows of those same routed matmuls (full K, partial edge tiles)
                re-run on the cycle-accurate tile model (fpga/systolic) for N=4/8/16,
                with and without host stalls: int32 bit-exact vs the MAX op output and
                cycles == the formula the adapter reports.
3. mlp        - layer 0's routed SwiGLU MLP (gate_up + down on the phantom) vs a numpy
                W8A8 reference; also reports the W8A8-vs-bf16 error.
4. device     - the same op with PHANTOM_BACKEND=device (lane G1-C's libphantom.dylib,
                cycle-accurate per launch) on windows of real layer-0 weights:
                bit-exact vs the stub backend, measured cycles == formula.
5. launch_log - the adapter's $PHANTOM_LOG lines carry the formula cycle counts.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import maxop_lib as L  # noqa: E402

M = 37  # not a multiple of 4/8/16: exercises edge-tile padding
LAYERS_UNDER_TEST = (0, 27)


def _module_cls():
    from max import functional as F
    from max.dtype import DType
    from max.nn import Module
    from phantom_llama.phantom_ops import phantom_matmul_i8, quantize_rows

    class LinearProbe(Module):
        """x, w (f32, bf16-representable) -> qa, sa, qw, sw, c(int32), y(bf16 as f32)."""

        def forward(self, x, w):
            xb = F.cast(x, DType.bfloat16)
            wb = F.cast(w, DType.bfloat16)
            qa, sa = quantize_rows(xb)
            qw, sw = quantize_rows(wb)
            c = phantom_matmul_i8(qa, qw, 999)
            y = F.cast(F.cast(c.to(x.device), DType.float32) * sa * F.transpose(sw, 0, 1), DType.bfloat16)
            return qa.to(c.device), sa, qw.to(c.device), sw, c, F.cast(y, DType.float32)

    class MLPProbe(Module):
        """Routed SwiGLU MLP and the plain bf16 MLP on the same input."""

        def forward(self, x, w_gu, w_down):
            from phantom_llama.phantom_ops import phantom_linear

            xb = F.cast(x, DType.bfloat16)
            gub = F.cast(w_gu, DType.bfloat16)
            db = F.cast(w_down, DType.bfloat16)
            ffl = int(w_down.shape[1])

            def mlp(lin):
                gu = lin(xb, gub, 1)
                g, u = F.split(gu, [ffl, ffl], axis=1)
                return lin(F.mul(F.silu(g), u), db, 2)

            routed = mlp(phantom_linear)
            plain = mlp(lambda a, w, s: a @ w.T)
            return F.cast(routed, DType.float32), F.cast(plain, DType.float32)

    return LinearProbe, MLPProbe


def _compile(mod, *shapes):
    from max.driver import Accelerator
    from max.dtype import DType
    from max.graph import DeviceRef, TensorType

    gpu = DeviceRef.from_device(Accelerator())
    types = [TensorType(DType.float32, s, device=gpu) for s in shapes]
    return mod.compile(*types)


def _t(a: np.ndarray):
    from max.driver import Accelerator
    from max.tensor import Tensor

    return Tensor.from_dlpack(np.ascontiguousarray(a, dtype=np.float32)).to(Accelerator())


def _np(t) -> np.ndarray:
    from max.driver import CPU

    return np.from_dlpack(t.to(CPU())).copy()


def activations(m: int, k: int, seed: int) -> np.ndarray:
    """Heavy-tailed bf16-representable activations (a few outlier channels, like real LLM)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((m, k)).astype(np.float32)
    x[:, rng.choice(k, 8, replace=False)] *= 20.0
    return L.to_bf16_f32(x)


def test_op_exact_and_spot(results: dict, workdir: Path) -> None:
    LinearProbe, _ = _module_cls()
    os.environ["PHANTOM_MODE"] = "phantom"
    t0 = time.time()
    probe = _compile(LinearProbe(), ["m", "k"], ["n", "k"])
    print(f"[op_exact] compiled probe in {time.time() - t0:.1f}s")
    rng = np.random.default_rng(1)
    cases = []
    tiles = [4, 8, 16]
    ci = 0
    for layer in LAYERS_UNDER_TEST:
        ws = L.layer_weights(layer)
        for proj in L.PROJ_SHAPES:
            w = ws[proj]
            k, n_out = L.PROJ_SHAPES[proj]
            assert w.shape == (n_out, k), (proj, w.shape)
            x = activations(M, k, seed=layer * 10 + len(cases))
            t = time.time()
            qa, sa, qw, sw, c, y = (_np(o) for o in probe(_t(x), _t(w)))
            dt = time.time() - t
            qa_r, sa_r = L.quantize_rows_np(x)
            qw_r, sw_r = L.quantize_rows_np(w)
            c_r = L.int32_matmul_exact(qa, qw)
            # The quantizer runs on the GPU (Metal division is not correctly rounded), so a
            # handful of values sitting on a .5 rounding tie land one step away from numpy.
            # That is a quantizer detail, not a tile error: the int32 check below uses the
            # graph's own int8 operands.
            qdiff = {}
            for nm, q, qr in (("qa", qa, qa_r), ("qw", qw, qw_r)):
                dq = np.abs(q.astype(np.int16) - qr.astype(np.int16))
                assert dq.max() <= 1 and dq.mean() < 1e-4, f"L{layer}.{proj}: {nm} quantization off"
                qdiff[nm] = int((dq != 0).sum())
            assert c.dtype == np.int32 and np.array_equal(c, c_r), f"L{layer}.{proj}: int32 mismatch"
            assert np.allclose(sa, sa_r, rtol=1e-6) and np.allclose(sw, sw_r, rtol=1e-6)
            y_r = L.to_bf16_f32(c_r.astype(np.float32) * sa * sw.T)
            ulp = np.abs(y - y_r) / np.maximum(np.abs(y_r), 1e-30)
            assert np.all((ulp <= 2 ** -7) | (np.abs(y - y_r) < 1e-6)), f"L{layer}.{proj}: dequant differs by >1 bf16 ulp"
            print(f"[op_exact] L{layer:02d}.{proj:8s} M={M} K={k} N_out={n_out}: int32 bit-exact "
                  f"({M * n_out} outputs, |c|max={np.abs(c).max()}), quantizer ties off-by-1 vs numpy: {qdiff}, {dt:.2f}s")
            results.setdefault("op_exact", []).append(
                {"layer": layer, "proj": proj, "M": M, "K": k, "N_out": n_out, "int32_exact": True, "quant_tie_diffs": qdiff, "seconds": round(dt, 3)})

            # Spot-check window for the cycle-accurate tile model.
            n = tiles[ci % 3]
            stall = ci in (3, 6)
            ci += 1
            rows = min(M, 2 * n + 3)
            cols = 2 * n + 5
            j0 = int(rng.integers(0, n_out - cols))
            d = workdir / f"L{layer}_{proj}_N{n}{'_stall' if stall else ''}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "meta.txt").write_text(f"{n} {rows} {k} {cols} {int(stall)}\n")
            qa[:rows].astype(np.int8).tofile(d / "a.bin")
            np.ascontiguousarray(qw[j0:j0 + cols].T).astype(np.int8).tofile(d / "b.bin")
            np.ascontiguousarray(c[:rows, j0:j0 + cols]).astype("<i4").tofile(d / "c.bin")
            cases.append(str(d))

    # Run the cycle-accurate checks in one Mojo process.
    t = time.time()
    p = subprocess.run(
        ["mojo", "run", "-I", str(L.SYSTOLIC), str(HERE / "spot_check.mojo"), *cases],
        capture_output=True, text=True, cwd=ROOT,
    )
    print(p.stdout.strip())
    if p.returncode != 0:
        print(p.stderr)
        raise AssertionError("cycle-accurate spot check failed")
    spots = []
    for line in p.stdout.splitlines():
        if line.startswith("SPOT ") and "cycles=" in line:
            kv = dict(tok.split("=", 1) for tok in line.split()[2:])
            spots.append({k2: (int(v) if v.lstrip("-").isdigit() else v) for k2, v in kv.items()} | {"case": Path(line.split()[1]).name})
    assert len(spots) == len(cases)
    results["spot_check"] = spots
    print(f"[spot_check] {len(spots)} windows cycle-accurate, bit-exact, cycles==formula(+stalls) in {time.time() - t:.1f}s")


def test_mlp(results: dict) -> None:
    _, MLPProbe = _module_cls()
    os.environ["PHANTOM_MODE"] = "phantom"
    ws = L.layer_weights(0)
    probe = _compile(MLPProbe(), ["m", L.HIDDEN], [2 * L.FFN, L.HIDDEN], [L.HIDDEN, L.FFN])
    x = activations(M, L.HIDDEN, seed=7) * L.layer_weights(0)["post_attention_layernorm"][None, :]
    x = L.to_bf16_f32(x)
    routed, plain = (_np(o) for o in probe(_t(x), _t(ws["gate_up"]), _t(ws["down"])))

    # numpy W8A8 reference (same quantizer, exact int32, f32 dequant, bf16 rounding points as the graph)
    def lin(a, w):
        qa, sa = L.quantize_rows_np(a)
        qw, sw = L.quantize_rows_np(w)
        return L.to_bf16_f32(L.int32_matmul_exact(qa, qw).astype(np.float32) * sa * sw.T)

    gu = lin(x, ws["gate_up"])
    g, u = gu[:, :L.FFN], gu[:, L.FFN:]
    h = L.to_bf16_f32(L.to_bf16_f32(g / (1 + np.exp(-g))) * u)
    ref = lin(h, ws["down"])
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))  # noqa: E731
    cos = lambda a, b: float((a * b).sum() / np.linalg.norm(a) / np.linalg.norm(b))  # noqa: E731
    r_ref, r_plain = rel(routed, ref), rel(routed, plain)
    print(f"[mlp] L00 routed MLP vs numpy W8A8 ref: rel_err={r_ref:.2e}  vs bf16 MLP: rel_err={r_plain:.3e} cos={cos(routed, plain):.6f}")
    # Remaining differences vs the numpy reference come only from the GPU's bf16 SiLU/mul
    # (different rounding of the int8 inputs to down_proj at a few ties).
    assert r_ref < 2e-2, r_ref
    results["mlp"] = {"layer": 0, "M": M, "rel_err_vs_w8a8_numpy": r_ref,
                      "rel_err_vs_bf16": r_plain, "cosine_vs_bf16": cos(routed, plain)}


def test_device_backend(results: dict, log: Path) -> None:
    """Same MAX op, PHANTOM_BACKEND=device: lane G1-C's libphantom (cycle-accurate) in the loop."""
    lib = Path(os.environ["PHANTOM_LIB"])
    if not lib.exists():
        raise AssertionError(f"{lib} missing: run `pixi run phantom-build` in fpga/ (lane G1-C)")
    LinearProbe, _ = _module_cls()
    probe = _compile(LinearProbe(), ["m", "k"], ["n", "k"])
    ws = L.layer_weights(0)
    out = []
    for proj, n, m, cols in (("down", 16, 13, 40), ("qkv", 8, 9, 24), ("o", 4, 6, 10)):
        w = ws[proj][:cols]
        x = activations(m, w.shape[1], seed=n)
        os.environ["PHANTOM_BACKEND"] = "stub"
        c_stub = _np(probe(_t(x), _t(w))[4])
        os.environ["PHANTOM_BACKEND"] = "device"
        os.environ["PHANTOM_DEVICE_N"] = str(n)
        t = time.time()
        c_dev = _np(probe(_t(x), _t(w))[4])
        dt = time.time() - t
        os.environ["PHANTOM_BACKEND"] = "stub"
        last = log.read_text().splitlines()[-1].split()
        measured = int(last[-1])
        assert int(last[-2]) == n and measured == L.tile_cycles(m, w.shape[1], cols, n)
        assert np.array_equal(c_dev, c_stub), f"device vs stub mismatch on L0.{proj}"
        print(f"[device] libphantom N={n} L00.{proj} M={m} K={w.shape[1]} N_out={cols}: "
              f"bit-exact vs stub, measured cycles={measured} == formula, {dt:.2f}s")
        out.append({"proj": proj, "N": n, "M": m, "K": int(w.shape[1]), "N_out": cols,
                    "measured_cycles": measured, "bit_exact_vs_stub": True, "seconds": round(dt, 3)})
    results["device_backend"] = out


def test_launch_log(results: dict, log: Path) -> None:
    lines = [ln.split() for ln in log.read_text().splitlines() if ln.strip()]
    assert lines, "adapter wrote no launch log"
    for site, m, k, n_out, c4, c8, c16, dev_n, measured in (map(int, ln) for ln in lines):
        for n, c in ((4, c4), (8, c8), (16, c16)):
            assert c == L.tile_cycles(m, k, n_out, n), (site, m, k, n_out, n, c)
        if dev_n:
            assert measured == L.tile_cycles(m, k, n_out, dev_n)
    print(f"[launch_log] {len(lines)} launches logged; cycles match the formula")
    results["launch_log_entries"] = len(lines)


def main() -> None:
    import json

    results: dict = {}
    with tempfile.TemporaryDirectory(prefix="maxop_test_") as td:
        log = Path(td) / "phantom.log"
        os.environ["PHANTOM_LOG"] = str(log)
        test_op_exact_and_spot(results, Path(td))
        test_mlp(results)
        test_device_backend(results, log)
        test_launch_log(results, log)
    out = ROOT / "tests" / "last_test_results.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"ALL PASS ({len(results['op_exact'])} routed matmuls exact, {len(results['spot_check'])} cycle-accurate spot checks). Wrote {out}")


if __name__ == "__main__":
    main()
