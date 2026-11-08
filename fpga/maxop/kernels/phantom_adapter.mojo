"""Thin adapter between the MAX custom op and the phantom NPU (int8 systolic tile).

The op talks only to `phantom_matmul_i8` and `pass_cycles`; when lane G1-C's
`fpga/phantom` device lands, this is the one file that changes (its launch call
replaces `_functional_matmul`).

Backends, chosen per launch by $PHANTOM_BACKEND:

  stub   (default) functional, bit-exact int8 x int8 -> int32 matmul on the host
         (SIMD + threads), plus the analytical cycle model of fpga/DESIGN.md 5,
             cycles = ceil(M/N) * ceil(Nout/N) * (K + 3N - 1)   (ideal host).
         Int32 wraparound addition is associative and commutative, so any
         summation order gives the tile's result bit for bit. Fast enough for
         whole-model prefill.
  device lane G1-C's phantom NPU driver, `libphantom.dylib` (C ABI,
         fpga/phantom/include/phantom.h; path from $PHANTOM_LIB), opened with an
         N x N tile (N = $PHANTOM_DEVICE_N, default 8). Every launch steps the
         cycle-accurate register model, so cycles are measured, not computed;
         the adapter checks them against the formula. Slow (~10^8 PE-steps/s):
         use for small routes / spot checks.
"""

from max.algorithm import parallelize
from std.ffi import OwnedDLHandle
from std.math import ceildiv
from std.os import getenv

comptime TILE_SIZES = (4, 8, 16)


def pass_cycles(m: Int, k: Int, n_out: Int, tile_n: Int) -> Int:
    """Ideal-host cycle count for M x K x N_out on an N x N tile (DESIGN.md 5)."""
    return ceildiv(m, tile_n) * ceildiv(n_out, tile_n) * (k + 3 * tile_n - 1)


@always_inline
def _dot_i8(
    a: ImmPointer[Int8, _], w: ImmPointer[Int8, _], k: Int
) -> Int32:
    comptime W = 16
    var acc = SIMD[.int32, W](0)
    var kk = 0
    while kk + W <= k:
        var av = a.unsafe_offset(kk).unsafe_load[width=W]().cast[.int32]()
        var wv = w.unsafe_offset(kk).unsafe_load[width=W]().cast[.int32]()
        acc += av * wv
        kk += W
    var s = acc.reduce_add()
    while kk < k:
        s += a[unsafe_offset=kk].cast[.int32]() * w[unsafe_offset=kk].cast[
            .int32
        ]()
        kk += 1
    return s


def phantom_matmul_i8(
    a: ImmPointer[Int8, _],
    w: ImmPointer[Int8, _],
    c: MutPointer[Int32, _],
    m: Int,
    k: Int,
    n_out: Int,
    site: Int,
) raises:
    """C[M, N_out] = A[M, K] @ W[N_out, K]^T on the phantom tile.

    A is row-major [M, K] (activations), W is row-major [N_out, K] (the HF
    Linear weight layout, i.e. B = W^T streamed one row of B per beat).
    """
    var backend = getenv("PHANTOM_BACKEND", "stub")
    var measured = -1
    var dev_n = 0
    if backend == "stub":
        _functional_matmul(a, w, c, m, k, n_out)
    elif backend == "device":
        dev_n = Int(getenv("PHANTOM_DEVICE_N", "8"))
        measured = _device_matmul(a, w, c, m, k, n_out, dev_n)
        var want = pass_cycles(m, k, n_out, dev_n)
        if measured != want:
            raise Error(t"phantom device cycles {measured} != formula {want}")
    else:
        raise Error(t"PHANTOM_BACKEND={backend}: expected stub or device")
    _log_launch(m, k, n_out, site, dev_n, measured)


def _device_matmul(
    a: ImmPointer[Int8, _],
    w: ImmPointer[Int8, _],
    c: MutPointer[Int32, _],
    m: Int,
    k: Int,
    n_out: Int,
    tile_n: Int,
) raises -> Int:
    """One launch on libphantom (C ABI). The device wants B = W^T as [K, N_out]."""
    var path = getenv("PHANTOM_LIB")
    if path == "":
        raise Error("PHANTOM_BACKEND=device needs PHANTOM_LIB (path to libphantom.dylib)")
    var lib = OwnedDLHandle(path)
    var h = lib.call["phantom_open", Int64](Int64(tile_n))
    if h <= 0:
        raise Error(t"phantom_open({tile_n}) failed")
    var bt = List[Int8](length=k * n_out, fill=0)
    for j in range(n_out):
        for kk in range(k):
            bt[kk * n_out + j] = w[unsafe_offset=j * k + kk]
    var da = lib.call["phantom_alloc", UInt64](h, Int64(m * k))
    var db = lib.call["phantom_alloc", UInt64](h, Int64(k * n_out))
    var dc = lib.call["phantom_alloc", UInt64](h, Int64(4 * m * n_out))
    if da == 0 or db == 0 or dc == 0:
        raise Error("phantom_alloc failed")
    var rc = lib.call["phantom_memcpy_h2d", Int64](h, da, Int(a), Int64(m * k))
    rc |= lib.call["phantom_memcpy_h2d", Int64](h, db, Int(bt.unsafe_ptr()), Int64(k * n_out))
    _ = bt^  # keep the transposed copy alive until the copy above has read it
    var cycles = lib.call["phantom_launch_matmul_i8", Int64](
        h, da, db, dc, Int64(m), Int64(k), Int64(n_out)
    )
    rc |= lib.call["phantom_memcpy_d2h", Int64](h, Int(c), dc, Int64(4 * m * n_out))
    _ = lib.call["phantom_close", Int64](h)
    if rc < 0 or cycles < 0:
        raise Error(t"phantom device call failed (rc={rc}, cycles={cycles})")
    return Int(cycles)


def _functional_matmul(
    a: ImmPointer[Int8, _],
    w: ImmPointer[Int8, _],
    c: MutPointer[Int32, _],
    m: Int,
    k: Int,
    n_out: Int,
):
    # One work item per output column block; each reads one W row-block once.
    comptime NB = 64
    var nblocks = ceildiv(n_out, NB)

    def work(blk: Int) {var a, var w, var c, var m, var k, var n_out}:
        var n0 = blk * NB
        var n1 = min(n0 + NB, n_out)
        for i in range(m):
            var arow = a.unsafe_offset(i * k)
            for j in range(n0, n1):
                c.unsafe_offset(i * n_out + j).unsafe_store(
                    _dot_i8(arow, w.unsafe_offset(j * k), k)
                )

    parallelize(work, nblocks)


def _log_launch(
    m: Int, k: Int, n_out: Int, site: Int, dev_n: Int, measured: Int
) raises:
    """Appends one line per launch to $PHANTOM_LOG (the device's launch trace):
    `site M K N_out cycles@N4 cycles@N8 cycles@N16 device_N measured_cycles`
    (formula cycles; device_N/measured are 0/-1 for the stub backend)."""
    var path = getenv("PHANTOM_LOG")
    if path == "":
        return
    var line = String(site, " ", m, " ", k, " ", n_out)
    comptime for i in range(len(TILE_SIZES)):
        line += String(" ", pass_cycles(m, k, n_out, TILE_SIZES[i]))
    line += String(" ", dev_n, " ", measured, "\n")
    with open(path, "a") as f:
        f.write(line)
