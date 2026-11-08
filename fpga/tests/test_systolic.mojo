from std.testing import assert_equal, assert_true, TestSuite

from systolic.driver import HostConfig, RunStats, predicted_cycles, run_matmul
from systolic.reference import reference_matmul
from systolic.rng import Rng
from systolic.tile import (
    SystolicTile,
    ST_IDLE,
    ST_FEED,
    K_LEN_MAX,
)
from systolic.trace import TraceWriter, TRACE_OFF, TRACE_TEXT, TRACE_JSONL


def rand_matrix[
    dtype: DType
](mut rng: Rng, rows: Int, cols: Int) -> List[Scalar[dtype]]:
    var out = List[Scalar[dtype]](capacity=rows * cols)
    for _ in range(rows * cols):
        out.append(rng.scalar[dtype]())
    return out^


def fill_matrix[
    dtype: DType
](rows: Int, cols: Int, value: Scalar[dtype]) -> List[Scalar[dtype]]:
    return List[Scalar[dtype]](length=rows * cols, fill=value)


def check[
    N: Int, in_dtype: DType = .int8, acc_dtype: DType = .int32
](
    a: List[Scalar[in_dtype]],
    b: List[Scalar[in_dtype]],
    m: Int,
    k: Int,
    n_out: Int,
    cfg: HostConfig = HostConfig.ideal(),
) raises -> RunStats:
    """Bit-exact vs reference; cycles == formula + observed stalls."""
    var trace = TraceWriter(TRACE_OFF)
    var c = List[Scalar[acc_dtype]]()
    var stats = run_matmul[N, in_dtype, acc_dtype](
        a, b, m, k, n_out, cfg, trace, c
    )
    var ref_c = reference_matmul[in_dtype, acc_dtype](a, b, m, k, n_out)
    for idx in range(m * n_out):
        if c[idx] != ref_c[idx]:
            raise Error(
                t"N={N} M={m} K={k} Nout={n_out}: C[{idx // n_out}][{idx % n_out}]"
                t" = {c[idx]}, expected {ref_c[idx]}"
            )
    assert_equal(
        stats.cycles,
        predicted_cycles(N, m, k, n_out) + stats.in_stalls + stats.out_stalls,
    )
    return stats^


def random_shapes[
    N: Int
](seed: UInt64, trials: Int, max_dim: Int, max_k: Int) raises -> Int:
    var rng = Rng(seed)
    for _ in range(trials):
        var m = rng.range(1, max_dim)
        var k = rng.range(1, max_k)
        var n_out = rng.range(1, max_dim)
        var a = rand_matrix[DType.int8](rng, m, k)
        var b = rand_matrix[DType.int8](rng, k, n_out)
        _ = check[N](a, b, m, k, n_out)
    return trials


# --- reference sanity -------------------------------------------------------


def test_reference_hand_example() raises:
    var a: List[Int8] = [1, 2, 3, 4, 5, 6]  # 2x3
    var b: List[Int8] = [7, 8, 9, 10, 11, 12]  # 3x2
    var c = reference_matmul[DType.int8, DType.int32](a, b, 2, 3, 2)
    var want: List[Int32] = [58, 64, 139, 154]
    for i in range(4):
        assert_equal(c[i], want[i])


def test_rng_covers_int8_range() raises:
    var rng = Rng(7)
    var lo = 0
    var hi = 0
    for _ in range(4096):
        var v = Int(rng.scalar[DType.int8]())
        lo = min(lo, v)
        hi = max(hi, v)
    assert_equal(lo, -128)
    assert_equal(hi, 127)


# --- single pass, exact cycle count ----------------------------------------


def single_pass[N: Int]() raises:
    var rng = Rng(UInt64(N))
    for k in [1, 2, 3, N, N + 1, 2 * N + 3, 33]:
        var a = rand_matrix[DType.int8](rng, N, k)
        var b = rand_matrix[DType.int8](rng, k, N)
        var s = check[N](a, b, N, k, N)
        assert_equal(s.cycles, SystolicTile[N].pass_cycles(k))
        assert_equal(s.cycles, k + 3 * N - 1)


def test_single_pass_n1() raises:
    single_pass[1]()


def test_single_pass_n2() raises:
    single_pass[2]()


def test_single_pass_n3() raises:
    single_pass[3]()


def test_single_pass_n4() raises:
    single_pass[4]()


def test_single_pass_n8() raises:
    single_pass[8]()


def test_single_pass_n16() raises:
    single_pass[16]()


# --- randomized shapes incl. non-multiples of N -----------------------------


def test_random_shapes_n2() raises:
    _ = random_shapes[2](101, 60, 9, 20)


def test_random_shapes_n3() raises:
    _ = random_shapes[3](102, 60, 11, 20)


def test_random_shapes_n4() raises:
    _ = random_shapes[4](103, 60, 17, 40)


def test_random_shapes_n8() raises:
    _ = random_shapes[8](104, 40, 30, 70)


def test_random_shapes_n16() raises:
    _ = random_shapes[16](105, 8, 40, 70)


def test_decode_like_1xKxN() raises:
    var rng = Rng(11)
    for n_out in [1, 7, 8, 9, 31]:
        for k in [1, 5, 64, 129]:
            var a = rand_matrix[DType.int8](rng, 1, k)
            var b = rand_matrix[DType.int8](rng, k, n_out)
            _ = check[8](a, b, 1, k, n_out)
            _ = check[4](a, b, 1, k, n_out)


def test_k_equals_1() raises:
    var rng = Rng(12)
    for m in [1, 3, 8, 13]:
        for n_out in [1, 5, 8, 17]:
            var a = rand_matrix[DType.int8](rng, m, 1)
            var b = rand_matrix[DType.int8](rng, 1, n_out)
            _ = check[8](a, b, m, 1, n_out)
            _ = check[3](a, b, m, 1, n_out)


# --- extreme values ---------------------------------------------------------


def test_extremes_min_times_min() raises:
    # (-128)*(-128) = 16384 per term: the largest positive product.
    for k in [1, 17, 1000]:
        var a = fill_matrix[DType.int8](5, k, -128)
        var b = fill_matrix[DType.int8](k, 6, -128)
        _ = check[4](a, b, 5, k, 6)


def test_extremes_max_times_min() raises:
    for k in [1, 17, 1000]:
        var a = fill_matrix[DType.int8](5, k, 127)
        var b = fill_matrix[DType.int8](k, 6, -128)
        _ = check[4](a, b, 5, k, 6)


def test_extremes_mixed_sign_pattern() raises:
    var m = 9
    var k = 37
    var n_out = 10
    var a = List[Int8](capacity=m * k)
    var b = List[Int8](capacity=k * n_out)
    var vals: List[Int8] = [-128, 127, -1, 0, 1, -127]
    for i in range(m * k):
        a.append(vals[i % len(vals)])
    for i in range(k * n_out):
        b.append(vals[(i * 5 + 1) % len(vals)])
    _ = check[8](a, b, m, k, n_out)
    _ = check[4](a, b, m, k, n_out)


def test_max_k_no_overflow() raises:
    # K = 65535 (k_len max) of 16384 stays below 2**31: exact, no wrap.
    var k = K_LEN_MAX
    var a = fill_matrix[DType.int8](2, k, -128)
    var b = fill_matrix[DType.int8](k, 2, -128)
    _ = check[2](a, b, 2, k, 2)
    var c = reference_matmul[DType.int8, DType.int32](a, b, 2, k, 2)
    assert_equal(Int(c[0]), k * 16384)


# --- protocol: stalls, garbage on don't-care cycles, reset, k_len=0 ---------


def test_backpressure_and_garbage() raises:
    var rng = Rng(21)
    var total_stalls = 0
    for t in range(30):
        var m = rng.range(1, 19)
        var k = rng.range(1, 30)
        var n_out = rng.range(1, 19)
        var a = rand_matrix[DType.int8](rng, m, k)
        var b = rand_matrix[DType.int8](rng, k, n_out)
        var s4 = check[4](a, b, m, k, n_out, HostConfig.stalling(UInt64(t)))
        var s8 = check[8](a, b, m, k, n_out, HostConfig.stalling(UInt64(t + 99)))
        total_stalls += s4.in_stalls + s4.out_stalls + s8.in_stalls + s8.out_stalls
    assert_true(total_stalls > 0)


def test_k_len_zero_start_is_ignored() raises:
    var tile = SystolicTile[4]()
    var inp = tile.Inputs()
    inp.start = True
    inp.k_len = 0
    tile.step(inp)
    assert_equal(tile.cur.fsm, ST_IDLE)
    inp.k_len = 1 << 16  # truncated to 16 bits -> 0
    tile.step(inp)
    assert_equal(tile.cur.fsm, ST_IDLE)


def test_reset_mid_pass_recovers() raises:
    var tile = SystolicTile[4]()
    var inp = tile.Inputs()
    inp.start = True
    inp.k_len = 10
    tile.step(inp)
    assert_equal(tile.cur.fsm, ST_FEED)
    var go = tile.Inputs()
    go.in_valid = True
    for i in range(4):
        go.a_col[i] = 100
        go.b_row[i] = -100
    for _ in range(5):
        tile.step(go)
    var rst = tile.Inputs()
    rst.rst = True
    tile.step(rst)
    assert_equal(tile.cur.fsm, ST_IDLE)
    for idx in range(16):
        assert_equal(tile.cur.acc[idx], 0)
        assert_equal(tile.cur.a_reg[idx], 0)
    # A full matmul afterwards (driver resets again, then runs) is still exact.
    var rng = Rng(5)
    var a = rand_matrix[DType.int8](rng, 6, 9)
    var b = rand_matrix[DType.int8](rng, 9, 7)
    _ = check[4](a, b, 6, 9, 7)


# --- comptime width parameters ----------------------------------------------


def test_int16_operands_int64_accumulator() raises:
    var rng = Rng(31)
    var a = rand_matrix[DType.int16](rng, 7, 50)
    var b = rand_matrix[DType.int16](rng, 50, 5)
    _ = check[4, DType.int16, DType.int64](a, b, 7, 50, 5)


# --- trace ------------------------------------------------------------------


def test_trace_one_line_per_cycle() raises:
    var rng = Rng(41)
    var a = rand_matrix[DType.int8](rng, 5, 6)
    var b = rand_matrix[DType.int8](rng, 6, 3)
    var c = List[Int32]()
    for mode in [TRACE_TEXT, TRACE_JSONL]:
        var trace = TraceWriter(mode)
        var s = run_matmul[4](a, b, 5, 6, 3, HostConfig.ideal(), trace, c)
        # reset cycle + every counted cycle
        assert_equal(trace.lines, s.cycles + 1)
        assert_equal(s.cycles, 2 * (6 + 3 * 4 - 1))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
