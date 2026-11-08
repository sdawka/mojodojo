# GPU smoke test: vector add + naive matmul on the default accelerator (Apple GPU via Metal).
from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from layout import TileTensor, row_major

comptime dt = DType.float32

# ---- vector add ----
comptime N = 1 << 20
comptime vlayout = row_major[N]()
comptime BS = 256


def vector_add(
    a: TileTensor[dt, type_of(vlayout), ImmutAnyOrigin],
    b: TileTensor[dt, type_of(vlayout), ImmutAnyOrigin],
    c: TileTensor[dt, type_of(vlayout), MutAnyOrigin],
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < N:
        c[i] = a[i] + b[i]


# ---- naive matmul C[M,K] = A[M,K] @ B[K,K] ----
comptime M = 512
comptime mlayout = row_major[M, M]()
comptime TILE = 16


def matmul_naive(
    a: TileTensor[dt, type_of(mlayout), ImmutAnyOrigin],
    b: TileTensor[dt, type_of(mlayout), ImmutAnyOrigin],
    c: TileTensor[dt, type_of(mlayout), MutAnyOrigin],
):
    var row = block_idx.y * block_dim.y + thread_idx.y
    var col = block_idx.x * block_dim.x + thread_idx.x
    if row < M and col < M:
        var acc: Float32 = 0
        for k in range(M):
            acc += rebind[Float32](a[row, k]) * rebind[Float32](b[k, col])
        c[row, col] = acc


def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
        return
    var ctx = DeviceContext()
    print("GPU device:", ctx.device().get_name())
    print("API:", ctx.api())

    # vector add
    var ha = ctx.enqueue_create_host_buffer[dt](N)
    var hb = ctx.enqueue_create_host_buffer[dt](N)
    var hc = ctx.enqueue_create_host_buffer[dt](N)
    ctx.synchronize()
    for i in range(N):
        ha[i] = Float32(i)
        hb[i] = Float32(2 * i)
    var da = ctx.enqueue_create_buffer[dt](N)
    var db = ctx.enqueue_create_buffer[dt](N)
    var dc = ctx.enqueue_create_buffer[dt](N)
    ctx.enqueue_copy(dst_buf=da, src_buf=ha)
    ctx.enqueue_copy(dst_buf=db, src_buf=hb)
    ctx.enqueue_function[vector_add](
        TileTensor(da, vlayout), TileTensor(db, vlayout), TileTensor(dc, vlayout),
        grid_dim=ceildiv(N, BS), block_dim=BS,
    )
    ctx.enqueue_copy(dst_buf=hc, src_buf=dc)
    ctx.synchronize()
    var verr = 0
    for i in range(N):
        if hc[i] != Float32(3 * i):
            verr += 1
    print("vector_add N=", N, " mismatches=", verr)

    # matmul: A = all 1, B[k,j] = j  ->  C[i,j] = M * j
    var hA = ctx.enqueue_create_host_buffer[dt](M * M)
    var hB = ctx.enqueue_create_host_buffer[dt](M * M)
    var hC = ctx.enqueue_create_host_buffer[dt](M * M)
    ctx.synchronize()
    for i in range(M):
        for j in range(M):
            hA[i * M + j] = 1.0
            hB[i * M + j] = Float32(j % 7)
    var dA = ctx.enqueue_create_buffer[dt](M * M)
    var dB = ctx.enqueue_create_buffer[dt](M * M)
    var dC = ctx.enqueue_create_buffer[dt](M * M)
    ctx.enqueue_copy(dst_buf=dA, src_buf=hA)
    ctx.enqueue_copy(dst_buf=dB, src_buf=hB)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    ctx.enqueue_function[matmul_naive](
        TileTensor(dA, mlayout), TileTensor(dB, mlayout), TileTensor(dC, mlayout),
        grid_dim=(ceildiv(M, TILE), ceildiv(M, TILE)), block_dim=(TILE, TILE),
    )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    ctx.enqueue_copy(dst_buf=hC, src_buf=dC)
    ctx.synchronize()
    var merr = 0
    for i in range(M):
        for j in range(M):
            if hC[i * M + j] != Float32(M * (j % 7)):
                merr += 1
    var secs = Float64(t1 - t0) / 1e9
    print("matmul_naive ", M, "x", M, " mismatches=", merr,
          " time(ms, incl. first-launch)=", secs * 1e3,
          " GFLOP/s=", 2.0 * Float64(M) ** 3 / secs / 1e9)
