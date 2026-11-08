"""Bare-metal Mojo driver for tile v2 (fpga/tile2/INTERFACE.md, contract v2.0).

The tile owns A (int8, stored transposed: k rows of M bytes), B (int8, k rows
of N bytes) and C (int32, row-major) buffers behind one Wishbone window and
runs C (=|+=) A.B from them. `tile2_matmul_i8` blocks an arbitrary
C[M][N] = A[M][K] @ B[K][N] into runs that fit the buffers (sizes read from
INFO), splitting K with MODE.ACCUM, packing operands into 32-bit stores.
"""

from std.sys import get_defined_int
from rt import mmio_read32, mmio_write32

comptime T2_BASE = get_defined_int["TILE2_BASE", 0x90000000]()
comptime T2_ID = T2_BASE + 0x00
comptime T2_INFO = T2_BASE + 0x04
comptime T2_CTRL = T2_BASE + 0x08
comptime T2_STATUS = T2_BASE + 0x0C
comptime T2_MODE = T2_BASE + 0x10
comptime T2_M = T2_BASE + 0x14
comptime T2_K = T2_BASE + 0x18
comptime T2_N = T2_BASE + 0x1C
comptime T2_A_BASE = T2_BASE + 0x20
comptime T2_A_LD = T2_BASE + 0x24
comptime T2_B_BASE = T2_BASE + 0x28
comptime T2_B_LD = T2_BASE + 0x2C
comptime T2_C_BASE = T2_BASE + 0x30
comptime T2_C_LD = T2_BASE + 0x34
comptime T2_CYCLES = T2_BASE + 0x38
comptime T2_A_WIN = T2_BASE + 0x08000
comptime T2_B_WIN = T2_BASE + 0x10000
comptime T2_C_WIN = T2_BASE + 0x18000
comptime T2_MAGIC: UInt32 = 0x54494C32  # "TIL2"
comptime CTRL_START: UInt32 = 1
comptime CTRL_CLEAR: UInt32 = 2
comptime ST_BUSY: UInt32 = 1
comptime ST_DONE: UInt32 = 2
comptime ST_ERR: UInt32 = 0xC  # ERR_CFG | ERR_BUSY

comptime I8Ptr = ImmPointer[Int8, MutUntrackedOrigin]
comptime I32Ptr = MutPointer[Int32, MutUntrackedOrigin]


def tile2_present() -> Bool:
    return mmio_read32(T2_ID) == T2_MAGIC


@always_inline
def _udiv(a: Int, b: Int) -> Int:
    """Non-negative a // b as plain `divu` (signed Int `//` on rv32 lowers
    through soft-float f64 libcalls; see README)."""
    return Int(UInt32(a) // UInt32(b))


@always_inline
def _round_up(x: Int, m: Int) -> Int:
    return (x + m - 1) & ~(m - 1)


@always_inline
def _pack4(p: I8Ptr, stride: Int, count: Int) -> UInt32:
    var w: UInt32 = 0
    for i in range(min(count, 4)):
        w |= UInt32(UInt8(p[unsafe_offset=i * stride])) << UInt32(8 * i)
    return w


def tile2_matmul_i8(
    a: I8Ptr, b: I8Ptr, c: I32Ptr, M: Int, K: Int, N: Int
) -> Int:
    """C[M][N] = A[M][K] @ B[K][N], row-major int8 in / int32 out.

    Returns the total tile BUSY cycles (sum of CYCLES), or -1 on a tile error
    (STATUS.ERR_* set or DONE missing).
    """
    var info = mmio_read32(T2_INFO)
    var nt = Int(info & 0xFF)
    var a_bytes = 1 << Int((info >> 8) & 0xFF)
    var b_bytes = 1 << Int((info >> 16) & 0xFF)
    var c_bytes = 1 << Int((info >> 24) & 0xFF)

    # Block sizes: up to 64 C columns per run, as many rows as C holds,
    # then the longest K both A and B hold (K split => MODE.ACCUM).
    var nb_max = min(_round_up(N, nt), 64)
    var mb_max = min(M, _udiv(c_bytes, 4 * nb_max), 256)
    var total = 0
    mmio_write32(T2_CTRL, CTRL_CLEAR)
    var m0 = 0
    while m0 < M:
        var mb = min(mb_max, M - m0)
        var mt = _round_up(mb, nt)
        var n0 = 0
        while n0 < N:
            var nb = min(nb_max, N - n0)
            var nr = _round_up(nb, nt)
            var kb_max = min(_udiv(a_bytes, mt), _udiv(b_bytes, nr))
            var k0 = 0
            while k0 < K:
                var kb = min(kb_max, K - k0)
                while (mmio_read32(T2_STATUS) & ST_BUSY) != 0:
                    pass
                # A^T rows: byte k*mt + m = A[m0+m][k0+k], 4 m's per store.
                for k in range(kb):
                    var src = a.unsafe_offset(m0 * K + k0 + k)
                    var dst = T2_A_WIN + k * mt
                    var m = 0
                    while m < mt:
                        mmio_write32(
                            dst + m, _pack4(src.unsafe_offset(m * K), K, mb - m)
                        )
                        m += 4
                # B rows: byte k*nr + n = B[k0+k][n0+n].
                for k in range(kb):
                    var src = b.unsafe_offset((k0 + k) * N + n0)
                    var dst = T2_B_WIN + k * nr
                    var n = 0
                    while n < nr:
                        mmio_write32(dst + n, _pack4(src.unsafe_offset(n), 1, nb - n))
                        n += 4
                mmio_write32(T2_M, UInt32(mb))
                mmio_write32(T2_K, UInt32(kb))
                mmio_write32(T2_N, UInt32(nb))
                mmio_write32(T2_A_BASE, 0)
                mmio_write32(T2_A_LD, UInt32(mt))
                mmio_write32(T2_B_BASE, 0)
                mmio_write32(T2_B_LD, UInt32(nr))
                mmio_write32(T2_C_BASE, 0)
                mmio_write32(T2_C_LD, UInt32(4 * nr))
                mmio_write32(T2_MODE, UInt32(1 if k0 > 0 else 0))  # ACCUM
                mmio_write32(T2_CTRL, CTRL_CLEAR | CTRL_START)
                var st = mmio_read32(T2_STATUS)
                while (st & ST_BUSY) != 0:
                    st = mmio_read32(T2_STATUS)
                if (st & ST_DONE) == 0 or (st & ST_ERR) != 0:
                    return -1
                total += Int(mmio_read32(T2_CYCLES))
                k0 += kb
            for r in range(mb):
                var crow = c.unsafe_offset((m0 + r) * N + n0)
                var src = T2_C_WIN + r * 4 * nr
                for j in range(nb):
                    crow[unsafe_offset=j] = Int32(mmio_read32(src + 4 * j))
            mmio_write32(T2_CTRL, CTRL_CLEAR)
            n0 += nb
        m0 += mb
    return total
