# Apple M5 GPU via the MAX Mojo kernel library (linalg.mojoc shipped in the max nightly).
#
#   bandwidth : hand-written Metal kernels (SIMD-4 fp32 copy / read-reduce / triad).
#   matmul    : MAX's own Apple kernels, called directly (same entry points as
#               max/kernels/benchmarks/gpu/linalg/bench_apple_gpu_matmul.mojo, no kbench needed):
#     - enqueue_apple_matmul      "AppleM5MatMul": Metal 4 hardware MMA (GPU Neural Accelerators)
#     - enqueue_apple_matmul_8x8  M1-M4 simdgroup_matrix 8x8 kernel (does NOT use the accelerators)
#     - enqueue_apple_gemv        decode GEMV (M <= 8), plain SIMD FMA
#     - enqueue_apple_int8_matmul W8A8 int8 x int8 -> int32 -> bf16 (M5 MMA)
#
# Timing: per rep, `inner` launches are enqueued back-to-back then one synchronize -> pipelined
# per-op time (how a model graph runs). Weight buffers are cycled so the working set exceeds
# ATLAS_COLD_MIB (cache-cold). Output: one JSON record per line with raw per-rep seconds;
# scripts/wrap.py turns `samples_s` into median/min/max/stddev.
#
# usage: mojo run max_gpu_bench.mojo [bw] [mm] [smoke]

from std.collections import Optional
from std.math import ceildiv
from std.os import getenv
from std.sys import argv, has_accelerator, size_of
from std.time import perf_counter

from max.gpu import global_idx, block_dim, grid_dim
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor
from layout.layout import row_major

from linalg.matmul.gpu.apple.matmul_kernel import enqueue_apple_matmul
from linalg.matmul.gpu.apple.matmul_8x8 import enqueue_apple_matmul_8x8
from linalg.matmul.gpu.apple.gemv import enqueue_apple_gemv
from linalg.matmul.gpu.apple.int8_matmul import enqueue_apple_int8_matmul
from linalg.matmul.gpu.apple.fp4_matmul import enqueue_apple_fp4_matmul
from linalg.fp4_utils import NVFP4_SF_VECTOR_SIZE


# ----------------------------------------------------------------- bandwidth kernels
comptime VW = 4  # fp32 lanes per thread access (16 B)


def copy_kernel(
    dst: MutPointer[Float32, MutUntrackedOrigin],
    src: MutPointer[Float32, MutUntrackedOrigin],
    nvec64: Int64,
):
    var nvec = Int(nvec64)
    var i = global_idx.x
    if i < nvec:
        dst.unsafe_offset(i * VW).unsafe_store(src.unsafe_load[width=VW](i * VW))


def triad_kernel(
    a: MutPointer[Float32, MutUntrackedOrigin],
    b: MutPointer[Float32, MutUntrackedOrigin],
    c: MutPointer[Float32, MutUntrackedOrigin],
    nvec64: Int64,
):
    var nvec = Int(nvec64)
    var i = global_idx.x
    if i < nvec:
        var v = b.unsafe_load[width=VW](i * VW) + 3.0 * c.unsafe_load[width=VW](i * VW)
        a.unsafe_offset(i * VW).unsafe_store(v)


comptime RD_PER_THREAD = 16


def read_kernel(
    dst: MutPointer[Float32, MutUntrackedOrigin],
    src: MutPointer[Float32, MutUntrackedOrigin],
    nvec64: Int64,
):
    var nvec = Int(nvec64)
    # Each thread sums RD_PER_THREAD strided vectors; writes one float (1/64 of traffic).
    var tid = global_idx.x
    var nthreads = grid_dim.x * block_dim.x
    var acc = SIMD[DType.float32, VW](0)
    for j in range(RD_PER_THREAD):
        var v = tid + j * nthreads
        if v < nvec:
            acc += src.unsafe_load[width=VW](v * VW)
    dst.unsafe_offset(tid).unsafe_store(acc.reduce_add())


def emit_samples(
    bench: String, engine: String, op: String, dtype: String, unit: String,
    work: Float64, samples: List[Float64], extra: String,
):
    """Print a JSON record; `work` = FLOPs (or bytes) per op, samples = per-op seconds."""
    var s = String("[")
    for i in range(len(samples)):
        if i:
            s += ","
        s += String(samples[i])
    s += "]"
    print(
        t'{{"bench":"{bench}","engine":"{engine}","op":"{op}","dtype":"{dtype}",'
        t'"unit":"{unit}","work_per_op":{work},"samples_s":{s}{extra}}}'
    )


def bandwidth(ctx: DeviceContext, smoke: Bool, reps: Int) raises:
    var mib = 64 if smoke else 512
    var n = mib * 1024 * 1024 // 4
    var nvec = n // VW
    var a = ctx.enqueue_create_buffer[DType.float32](n)
    var b = ctx.enqueue_create_buffer[DType.float32](n)
    var c = ctx.enqueue_create_buffer[DType.float32](n)
    a.enqueue_fill(1.0)
    b.enqueue_fill(2.0)
    c.enqueue_fill(0.5)
    var block = 256
    var grid = ceildiv(nvec, block)
    var rgrid = ceildiv(ceildiv(nvec, RD_PER_THREAD), block)
    var part = ctx.enqueue_create_buffer[DType.float32](rgrid * block)
    ctx.synchronize()
    var ap = a.unsafe_ptr()
    var bp = b.unsafe_ptr()
    var cp = c.unsafe_ptr()
    var pp = part.unsafe_ptr()
    comptime inner = 10
    for which in range(3):
        var samples = List[Float64]()
        for r in range(reps + 2):
            var t0 = perf_counter()
            for _ in range(inner):
                if which == 0:
                    ctx.enqueue_function[copy_kernel](ap, bp, Int64(nvec), grid_dim=grid, block_dim=block)
                elif which == 1:
                    ctx.enqueue_function[read_kernel](pp, bp, Int64(nvec), grid_dim=rgrid, block_dim=block)
                else:
                    ctx.enqueue_function[triad_kernel](ap, bp, cp, Int64(nvec), grid_dim=grid, block_dim=block)
            ctx.synchronize()
            if r >= 2:  # first two reps = warmup
                samples.append((perf_counter() - t0) / Float64(inner))
        var names: List[String] = ["copy", "read_reduce", "triad"]
        var mult: List[Int] = [2, 1, 3]
        emit_samples("gpu_bandwidth", "gpu_mojo", names[which], "f32", "GB/s",
                     Float64(mult[which] * n * 4), samples, String(t',"array_mib":{mib}'))
    _ = a^
    _ = b^
    _ = c^
    _ = part^


# ----------------------------------------------------------------- matmul
def bench_mm[
    in_type: DType, kernel: Int
](
    ctx: DeviceContext, tag: String, m: Int, n: Int, k: Int, reps: Int, cold_bytes: Int,
    name: String,
) raises:
    """kernel: 0 = AppleM5MatMul (NA), 1 = 8x8 simdgroup, 2 = gemv, 3 = int8 W8A8."""
    comptime out_type = DType.bfloat16 if kernel == 3 else (
        DType.float32 if in_type == DType.float32 else in_type
    )
    var wbytes = n * k * size_of[in_type]()
    var nc = max(1, min(64, ceildiv(cold_bytes, wbytes)))
    var a = ctx.enqueue_create_buffer[in_type](m * k)
    a.enqueue_fill(Scalar[in_type](1))
    var ws = List[DeviceBuffer[in_type]]()
    for _ in range(nc):
        var w = ctx.enqueue_create_buffer[in_type](n * k)
        w.enqueue_fill(Scalar[in_type](1))
        ws.append(w^)
    var c = ctx.enqueue_create_buffer[out_type](m * n)
    var asc = ctx.enqueue_create_buffer[DType.float32](m)
    var bsc = ctx.enqueue_create_buffer[DType.float32](n)
    var bias = ctx.enqueue_create_buffer[DType.bfloat16](1)
    asc.enqueue_fill(1.0)
    bsc.enqueue_fill(1.0)
    ctx.synchronize()
    var a_tt = TileTensor(a.unsafe_ptr(), row_major(m, k)).as_imm()
    var c_tt = TileTensor(c.unsafe_ptr(), row_major(m, n))
    var c8 = TileTensor(c.unsafe_ptr().unsafe_bitcast[BFloat16](), row_major(m, n))

    def launch(i: Int) raises {imm}:
        var w_tt = TileTensor(ws[i % nc].unsafe_ptr(), row_major(n, k)).as_imm()
        comptime if kernel == 0:
            enqueue_apple_matmul[in_type=in_type, c_type=out_type, transpose_b=True](c_tt, a_tt, w_tt, ctx)
        elif kernel == 1:
            enqueue_apple_matmul_8x8[transpose_b=True](c_tt, a_tt, w_tt, m, n, k, ctx)
        elif kernel == 2:
            enqueue_apple_gemv(c_tt, a_tt, w_tt, ctx)
        else:
            comptime if in_type == DType.int8:
                var as_tt = TileTensor(asc.unsafe_ptr(), row_major(m)).as_imm()
                var bs_tt = TileTensor(bsc.unsafe_ptr(), row_major(n)).as_imm()
                var b_tt = TileTensor(bias.unsafe_ptr(), row_major(1)).as_imm()
                enqueue_apple_int8_matmul[c_type=DType.bfloat16](
                    c8,
                    TileTensor(a.unsafe_ptr().unsafe_bitcast[Int8](), row_major(m, k)).as_imm(),
                    TileTensor(ws[i % nc].unsafe_ptr().unsafe_bitcast[Int8](), row_major(n, k)).as_imm(),
                    as_tt, bs_tt, b_tt, ctx)

    # calibrate inner so a rep lasts ~0.1 s (0.02 s smoke)
    launch(0)
    ctx.synchronize()
    var t0 = perf_counter()
    for i in range(4):
        launch(i)
    ctx.synchronize()
    var one = (perf_counter() - t0) / 4.0
    var target = 0.02 if getenv("ATLAS_SMOKE", "0") == "1" else 0.1
    var inner = max(1, min(4096, Int(target / max(one, 1e-7)) + 1))
    var samples = List[Float64]()
    var idx = 0
    for r in range(reps + 2):
        var s = perf_counter()
        for _ in range(inner):
            launch(idx)
            idx += 1
        ctx.synchronize()
        if r >= 2:
            samples.append((perf_counter() - s) / Float64(inner))
    # Sanity check: A = W = 1 (int8 scales = 1) -> every C entry == K (exact for these K).
    var h = ctx.enqueue_create_host_buffer[out_type](m * n)
    ctx.enqueue_copy(h, c)
    ctx.synchronize()
    var ok = True
    for idx2 in [0, (m * n) // 2, m * n - 1]:
        if Float64(h[idx2].cast[DType.float32]()) != Float64(k):
            ok = False
    var verified = "true" if ok else "false"
    _ = h^
    var flops = 2.0 * Float64(m) * Float64(n) * Float64(k)
    var bytes = Float64(wbytes + m * k * size_of[in_type]() + m * n * size_of[out_type]())
    var dname = String(in_type)
    emit_samples("gemm", "gpu_max", "matmul", dname, "GFLOP/s", flops, samples,
        String(t',"kernel":"{name}","shape":"{tag}","M":{m},"N":{n},"K":{k},"weight_copies":{nc},"inner_iters":{inner},"bytes_per_call":{bytes},"out_dtype":"{String(out_type)}","verified_c_eq_K":{verified}'))
    _ = a^
    _ = ws^
    _ = c^
    _ = asc^
    _ = bsc^
    _ = bias^


def bench_fp4(
    ctx: DeviceContext, tag: String, m: Int, n: Int, k: Int, reps: Int, cold_bytes: Int
) raises:
    """W4A16: NVFP4 (e2m1 nibbles + fp8-e4m3 block scales per 16) weights x bf16 activations.
    Production dispatch: M >= 256 materializes bf16 weights then dense MMA; M < 256 fused."""
    var packed_k = k // 2
    var scale_k = ceildiv(k, NVFP4_SF_VECTOR_SIZE)
    var wbytes = n * packed_k + n * scale_k
    var nc = max(1, min(64, ceildiv(cold_bytes, wbytes)))
    var act = ctx.enqueue_create_buffer[DType.bfloat16](m * k)
    act.enqueue_fill(1.0)
    var packs = List[DeviceBuffer[DType.uint8]]()
    var scales = List[DeviceBuffer[DType.float8_e4m3fn]]()
    for _ in range(nc):
        var pk = ctx.enqueue_create_buffer[DType.uint8](n * packed_k)
        pk.enqueue_fill(0x22)  # e2m1 0b0010 = 1.0 in both nibbles
        packs.append(pk^)
        var sc = ctx.enqueue_create_buffer[DType.float8_e4m3fn](n * scale_k)
        sc.enqueue_fill(Float8_e4m3fn(1.0))
        scales.append(sc^)
    var out = ctx.enqueue_create_buffer[DType.float32](m * n)
    ctx.synchronize()
    var act_tt = TileTensor(act.unsafe_ptr(), row_major(m, k)).as_imm()
    var out_tt = TileTensor(out.unsafe_ptr(), row_major(m, n))

    def launch(i: Int) raises {imm}:
        var p_tt = TileTensor(packs[i % nc].unsafe_ptr(), row_major(n, packed_k)).as_imm()
        var s_tt = TileTensor(scales[i % nc].unsafe_ptr(), row_major(n, scale_k)).as_imm()
        enqueue_apple_fp4_matmul[c_type=DType.float32](out_tt, act_tt, p_tt, s_tt, ctx)

    launch(0)
    ctx.synchronize()
    var t0 = perf_counter()
    for i in range(4):
        launch(i)
    ctx.synchronize()
    var one = (perf_counter() - t0) / 4.0
    var target = 0.02 if getenv("ATLAS_SMOKE", "0") == "1" else 0.1
    var inner = max(1, min(4096, Int(target / max(one, 1e-7)) + 1))
    var samples = List[Float64]()
    var idx = 0
    for r in range(reps + 2):
        var s = perf_counter()
        for _ in range(inner):
            launch(idx)
            idx += 1
        ctx.synchronize()
        if r >= 2:
            samples.append((perf_counter() - s) / Float64(inner))
    var h = ctx.enqueue_create_host_buffer[DType.float32](m * n)
    ctx.enqueue_copy(h, out)
    ctx.synchronize()
    var v0 = Float64(h[0])
    var verified = "true" if v0 == Float64(k) else String("false(c0=") + String(v0) + ")"
    _ = h^
    var flops = 2.0 * Float64(m) * Float64(n) * Float64(k)
    var bytes = Float64(wbytes + m * k * 2 + m * n * 4)
    emit_samples("gemm", "gpu_max", "matmul", "nvfp4w x bf16", "GFLOP/s", flops, samples,
        String(t',"kernel":"apple_fp4_w4a16","shape":"{tag}","M":{m},"N":{n},"K":{k},"weight_copies":{nc},"inner_iters":{inner},"bytes_per_call":{bytes},"out_dtype":"float32","verified_c_eq_K":"{verified}"'))
    _ = act^
    _ = packs^
    _ = scales^
    _ = out^


def main() raises:
    comptime if not has_accelerator():
        print('{"error":"no GPU accelerator"}')
        return
    var do_bw = False
    var do_mm = False
    var smoke = False
    var args = argv()
    for i in range(len(args)):
        if args[i] == "bw":
            do_bw = True
        elif args[i] == "mm":
            do_mm = True
        elif args[i] == "smoke":
            smoke = True
    var reps = max(5, Int(getenv("ATLAS_REPS", "5" if smoke else "15")))
    var ctx = DeviceContext()
    print(t"# device {ctx.device().get_name()} api {ctx.api()} compute_capability {ctx.compute_capability()}")
    if do_bw:
        bandwidth(ctx, smoke, reps)
    if do_mm:
        var cold = Int(getenv("ATLAS_COLD_MIB", "256")) * 1024 * 1024
        var tags = List[String]()
        var ms = List[Int]()
        var ns = List[Int]()
        var ks = List[Int]()
        var sq: List[Int] = [256, 1024, 2048] if smoke else [256, 512, 1024, 2048, 4096, 8192]
        for s in sq:
            tags.append(String("square") + String(s)); ms.append(s); ns.append(s); ks.append(s)
        var kn_names: List[String] = ["attn_q_o", "attn_kv", "mlp_up_gate", "mlp_down"]
        var kk: List[Int] = [3072, 3072, 3072, 8192]
        var nn: List[Int] = [3072, 1024, 8192, 3072]
        var mlist: List[Int] = [1, 512] if smoke else [1, 4, 16, 512]
        var nkn = 2 if smoke else 4
        for j in range(nkn):
            for mm in mlist:
                tags.append(kn_names[j] + "_M" + String(mm)); ms.append(mm); ns.append(nn[j]); ks.append(kk[j])
        if not smoke:
            for mm in [1, 512]:
                tags.append(String("lm_head_M") + String(mm)); ms.append(mm); ns.append(128256); ks.append(3072)
        for i in range(len(tags)):
            var t = tags[i]
            var m = ms[i]
            var n = ns[i]
            var k = ks[i]
            bench_mm[DType.float16, 0](ctx, t, m, n, k, reps, cold, "apple_m5_mma")
            bench_mm[DType.bfloat16, 0](ctx, t, m, n, k, reps, cold, "apple_m5_mma")
            bench_mm[DType.float32, 0](ctx, t, m, n, k, reps, cold, "apple_m5_mma")
            bench_mm[DType.float16, 1](ctx, t, m, n, k, reps, cold, "simdgroup_8x8")
            bench_mm[DType.float32, 1](ctx, t, m, n, k, reps, cold, "simdgroup_8x8")
            bench_mm[DType.int8, 3](ctx, t, m, n, k, reps, cold, "apple_int8_w8a8")
            bench_fp4(ctx, t, m, n, k, reps, cold)
            if m <= 8:
                bench_mm[DType.bfloat16, 2](ctx, t, m, n, k, reps, cold, "apple_gemv")
