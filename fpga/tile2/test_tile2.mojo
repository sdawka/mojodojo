"""Unit tests for the tile v2 bus-level model (pixi run tile2-test).

The model (systolic/tile2.mojo) is the reference the RTL is co-simulated
against, so these tests pin it to the plain reference matmul of the v1
simulator and to the v1 tile's cycle model, through the bus, like firmware.
"""

from std.math import ceildiv
from std.testing import assert_equal, assert_true, assert_false, TestSuite

from systolic.reference import reference_matmul
from systolic.rng import Rng
from systolic.tile import SystolicTile
from systolic.tile2 import (
    Tile2Model,
    t2_busy_cycles,
    T2_ID,
    T2_A_WIN,
    T2_B_WIN,
    T2_C_WIN,
    T2_CTRL,
    T2_STATUS,
    T2_MODE,
    T2_M,
    T2_K,
    T2_N,
    T2_A_BASE,
    T2_A_LD,
    T2_B_BASE,
    T2_B_LD,
    T2_C_BASE,
    T2_C_LD,
    T2_CYCLES,
    ST_BUSY,
    ST_DONE,
    ST_ERR_CFG,
    ST_ERR_BUSY,
)

comptime Model8 = Tile2Model[8, 16384, 32768, 32768]


def wr(mut t: Model8, addr: Int, v: Int, sel: Int = 15):
    _ = t.access(0, True, addr, v, sel)


def rd(mut t: Model8, addr: Int) -> Int:
    return t.access(0, False, addr, 0, 15).rdata


def run_firmware(
    mut t: Model8,
    a: List[Int8],
    b: List[Int8],
    m: Int,
    k: Int,
    n: Int,
    a_base: Int,
    b_base: Int,
    c_base: Int,
    accum: Bool,
) raises -> List[Int32]:
    """INTERFACE.md section 7, byte stores for operands. Returns C (m x n)."""
    var mp = ceildiv(m, 8) * 8
    var np = ceildiv(n, 8) * 8
    for kk in range(k):
        for mi in range(m):
            var byte = Int(a[mi * k + kk]) & 0xFF
            var off = (a_base + kk * mp + mi) % 16384
            wr(t, T2_A_WIN + off, byte << (8 * (off % 4)), 1 << (off % 4))
        for nj in range(n):
            var byte = Int(b[kk * n + nj]) & 0xFF
            var off = (b_base + kk * np + nj) % 32768
            wr(t, T2_B_WIN + off, byte << (8 * (off % 4)), 1 << (off % 4))
    wr(t, T2_M, m)
    wr(t, T2_K, k)
    wr(t, T2_N, n)
    wr(t, T2_A_BASE, a_base)
    wr(t, T2_A_LD, mp)
    wr(t, T2_B_BASE, b_base)
    wr(t, T2_B_LD, np)
    wr(t, T2_C_BASE, c_base)
    wr(t, T2_C_LD, 4 * np)
    wr(t, T2_MODE, 1 if accum else 0)
    wr(t, T2_CTRL, 3)
    var polls = 0
    while rd(t, T2_STATUS) & ST_BUSY:
        polls += 1
    assert_true(polls > 0)
    assert_equal(rd(t, T2_STATUS), ST_DONE)
    assert_equal(rd(t, T2_CYCLES), t2_busy_cycles(8, m, k, n))
    var c = List[Int32](length=m * n, fill=0)
    for mi in range(m):
        for nj in range(n):
            var v = rd(t, T2_C_WIN + (c_base + mi * 4 * np + 4 * nj) % 32768)
            c[mi * n + nj] = Int32(v & 0xFFFFFFFF) if v < 0x80000000 else Int32(v - 0x100000000)
    return c^


def rand_i8(mut rng: Rng, count: Int) -> List[Int8]:
    var out = List[Int8](capacity=count)
    for _ in range(count):
        out.append(rng.scalar[DType.int8]())
    return out^


def test_id_info() raises:
    var t = Model8()
    assert_equal(rd(t, 0x00), T2_ID)
    assert_equal(rd(t, 0x04), 0x0F0F0E08)
    assert_equal(rd(t, 0x40), T2_ID)  # CSR aliasing every 64 bytes


def test_cycles_relate_to_v1_core() raises:
    # Per pass: v1 pass (K + 3N - 1) + 1 registered-product cycle; +1 setup +1 tail per run.
    for k in [1, 2, 7, 64, 1000]:
        var v1 = SystolicTile[8].pass_cycles(k)
        assert_equal(t2_busy_cycles(8, 8, k, 8), 2 + v1 + 1)
        assert_equal(t2_busy_cycles(8, 17, k, 9), 2 + 3 * 2 * (v1 + 1))


def test_random_matmuls_vs_reference() raises:
    var rng = Rng(42)
    var t = Model8()
    for _ in range(25):
        var m = rng.range(1, 30)
        var k = rng.range(1, 90)
        var n = rng.range(1, 30)
        var a = rand_i8(rng, m * k)
        var b = rand_i8(rng, k * n)
        var c = run_firmware(
            t, a, b, m, k, n, 8 * rng.range(0, 2047), 8 * rng.range(0, 4095),
            32 * rng.range(0, 1023), False,
        )
        var ref_c = reference_matmul[DType.int8, DType.int32](a, b, m, k, n)
        for i in range(m * n):
            assert_equal(c[i], ref_c[i])


def test_accum_splits_k() raises:
    # C = A[:, :k1] B[:k1] then C += A[:, k1:] B[k1:]  ==  A B
    var rng = Rng(7)
    var t = Model8()
    var m = 11
    var k = 50
    var n = 13
    var k1 = 23
    var a = rand_i8(rng, m * k)
    var b = rand_i8(rng, k * n)
    var a1 = List[Int8]()
    var a2 = List[Int8]()
    for i in range(m):
        for kk in range(k):
            if kk < k1:
                a1.append(a[i * k + kk])
            else:
                a2.append(a[i * k + kk])
    var b1 = List[Int8]()
    var b2 = List[Int8]()
    for kk in range(k):
        for j in range(n):
            if kk < k1:
                b1.append(b[kk * n + j])
            else:
                b2.append(b[kk * n + j])
    _ = run_firmware(t, a1, b1, m, k1, n, 0, 0, 0, False)
    var c = run_firmware(t, a2, b2, m, k - k1, n, 4096, 8192, 0, True)
    var ref_c = reference_matmul[DType.int8, DType.int32](a, b, m, k, n)
    for i in range(m * n):
        assert_equal(c[i], ref_c[i])


def test_extreme_long_k() raises:
    # K = 2048 of (-128)*(-128): 33,554,432 per element, no wrap; in 16 KiB of A.
    var t = Model8()
    var m = 8
    var k = 2048
    var n = 8
    var a = List[Int8](length=m * k, fill=-128)
    var b = List[Int8](length=k * n, fill=-128)
    var c = run_firmware(t, a, b, m, k, n, 0, 0, 0, False)
    for i in range(m * n):
        assert_equal(c[i], 2048 * 16384)


def test_err_cfg_and_busy_rules() raises:
    var t = Model8()
    wr(t, T2_M, 8)
    wr(t, T2_K, 0)
    wr(t, T2_N, 8)
    wr(t, T2_CTRL, 1)
    assert_equal(rd(t, T2_STATUS), ST_ERR_CFG)
    wr(t, T2_CTRL, 2)
    assert_equal(rd(t, T2_STATUS), 0)
    wr(t, T2_K, 100)
    wr(t, T2_CTRL, 1)
    assert_equal(rd(t, T2_STATUS), ST_BUSY)
    wr(t, T2_M, 1)  # dropped
    assert_equal(rd(t, T2_A_WIN), 0)  # zero read
    assert_equal(rd(t, T2_STATUS), ST_BUSY | ST_ERR_BUSY)
    assert_equal(rd(t, T2_M), 8)
    while rd(t, T2_STATUS) & ST_BUSY:
        pass
    assert_equal(rd(t, T2_STATUS), ST_DONE | ST_ERR_BUSY)
    assert_equal(rd(t, T2_CYCLES), 2 + 100 + 24)


def test_csr_masking() raises:
    var t = Model8()
    wr(t, T2_A_BASE, 0xFFFFFFFF)
    assert_equal(rd(t, T2_A_BASE), 0x3FF8)  # bits [13:3]
    wr(t, T2_B_LD, 0xFFFFFFFF)
    assert_equal(rd(t, T2_B_LD), 0x7FF8)  # bits [14:3]
    wr(t, T2_C_BASE, 0xFFFFFFFF)
    assert_equal(rd(t, T2_C_BASE), 0x7FE0)  # bits [14:5]
    wr(t, T2_M, 0x12345678)
    assert_equal(rd(t, T2_M), 0x5678)
    wr(t, T2_MODE, 0xFF)
    assert_equal(rd(t, T2_MODE), 3)
    assert_equal(rd(t, T2_CTRL), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
