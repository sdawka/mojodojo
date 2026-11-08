"""Bare-metal Mojo driver for the v1 int8 matmul tile (fpga/soc/rtl/tile_v1_wb.v).

The tile computes one 4x4 int32 output block per pass: after KLEN and a CTRL
write it accepts K beats, each a packed column of A (4 x int8, register A) and
a packed row of B (4 x int8, register B; the write pushes the beat and the
bus stalls until the tile takes it), then exposes C[4][4] at 0x40.
"""

from std.sys import get_defined_int
from rt import mmio_read32, mmio_write32

comptime TILE_BASE = get_defined_int["TILE_BASE", 0x82000000]()
comptime TN = 4
comptime REG_CTRL = TILE_BASE + 0x00  # W start / R status
comptime REG_KLEN = TILE_BASE + 0x04
comptime REG_A = TILE_BASE + 0x08
comptime REG_B = TILE_BASE + 0x0C
comptime REG_CYCLES = TILE_BASE + 0x10
comptime REG_ID = TILE_BASE + 0x14
comptime REG_C = TILE_BASE + 0x40
comptime ST_IDLE: UInt32 = 1
comptime ST_DONE: UInt32 = 4
comptime TILE_ID: UInt32 = 0x71110004

comptime I8Ptr = ImmPointer[Int8, MutUntrackedOrigin]
comptime I32Ptr = MutPointer[Int32, MutUntrackedOrigin]


def tile_present() -> Bool:
    return mmio_read32(REG_ID) == TILE_ID


@always_inline
def pack4(p: I8Ptr, stride: Int, count: Int) -> UInt32:
    """Pack p[0], p[stride], ... (count <= 4, rest zero) into int8 lanes."""
    var w: UInt32 = 0
    for i in range(count):
        w |= UInt32(UInt8(p[unsafe_offset=i * stride])) << UInt32(8 * i)
    return w


def tile_matmul_i8(
    a: I8Ptr, b: I8Ptr, c: I32Ptr, M: Int, K: Int, N: Int
) -> UInt32:
    """C[M][N] = A[M][K] @ B[K][N] (row-major int8 in, int32 out) on the tile.

    Ragged edges are zero-padded. K must be 1..65535. Returns the sum of the
    tile's per-pass CYCLES counters.
    """
    var tile_cycles: UInt32 = 0
    mmio_write32(REG_KLEN, UInt32(K))
    var m0 = 0
    while m0 < M:
        var mr = min(TN, M - m0)
        var n0 = 0
        while n0 < N:
            var nr = min(TN, N - n0)
            while (mmio_read32(REG_CTRL) & ST_IDLE) == 0:
                pass
            mmio_write32(REG_CTRL, 1)
            var arow = a.unsafe_offset(m0 * K)
            var brow = b.unsafe_offset(n0)
            for k in range(K):
                # A column k of rows m0..m0+3, B row k of cols n0..n0+3
                mmio_write32(REG_A, pack4(arow.unsafe_offset(k), K, mr))
                mmio_write32(REG_B, pack4(brow.unsafe_offset(k * N), 1, nr))
            while (mmio_read32(REG_CTRL) & ST_DONE) == 0:
                pass
            tile_cycles += mmio_read32(REG_CYCLES)
            for r in range(mr):
                var crow = c.unsafe_offset((m0 + r) * N + n0)
                for j in range(nr):
                    crow[unsafe_offset=j] = Int32(
                        mmio_read32(REG_C + 4 * (r * TN + j))
                    )
            n0 += TN
        m0 += TN
    return tile_cycles


def cpu_matmul_i8(a: I8Ptr, b: I8Ptr, c: I32Ptr, M: Int, K: Int, N: Int):
    """Reference: the same product on the rv32 core (mul + add)."""
    for i in range(M):
        for j in range(N):
            var acc: Int32 = 0
            for k in range(K):
                acc += Int32(a[unsafe_offset=i * K + k]) * Int32(
                    b[unsafe_offset=k * N + j]
                )
            c[unsafe_offset=i * N + j] = acc
