"""Phantom NPU tests (fallback route: Mojo context -> C ABI -> simulator).

Run from fpga/: `pixi run phantom-test` (builds libphantom.dylib first).
"""

from std.testing import assert_equal, assert_true, assert_false
from phantom.context import PhantomContext, PhantomBuffer
from phantom.target import Phantom, PhantomTarget, device_kind_string
from systolic.driver import predicted_cycles
from systolic.reference import reference_matmul
from systolic.rng import Rng


def test_target_device_kind() raises:
    assert_equal(device_kind_string[PhantomTarget](), "other(phantom, accelerator=True)")
    assert_true(Phantom().is_accelerator())
    assert_false(Phantom().is_gpu())
    assert_equal(device_kind_string(), "cpu")  # the host is unaffected


def check_matmul(
    ctx: PhantomContext, mut rng: Rng, m: Int, k: Int, n: Int, extreme: Bool = False
) raises:
    var a = List[Int8](capacity=m * k)
    var b = List[Int8](capacity=k * n)
    for _ in range(m * k):
        a.append(Int8(-128) if extreme else rng.scalar[DType.int8]())
    for _ in range(k * n):
        b.append(Int8(-128) if extreme else rng.scalar[DType.int8]())
    var want = reference_matmul[DType.int8, DType.int32](a, b, m, k, n)

    var da = ctx.enqueue_create_buffer[DType.int8](m * k)
    var db = ctx.enqueue_create_buffer[DType.int8](k * n)
    var dc = ctx.enqueue_create_buffer[DType.int32](m * n)
    ctx.enqueue_copy(da, a)
    ctx.enqueue_copy(db, b)
    var cycles = ctx.enqueue_matmul_i8(da, db, dc, m, k, n)
    var got = List[Int32]()
    ctx.enqueue_copy(got, dc)
    ctx.synchronize()

    var tn = ctx.tile_n()
    assert_equal(cycles, predicted_cycles(tn, m, k, n), String(t"cycles N={tn} M={m} K={k} N_out={n}"))
    assert_equal(ctx.last_cycles(), cycles)
    assert_equal(len(got), m * n)
    for i in range(m * n):
        if got[i] != want[i]:
            raise Error(t"N={tn} M={m} K={k} N_out={n}: C[{i}] = {got[i]}, want {want[i]}")
    ctx.enqueue_free(da)
    ctx.enqueue_free(db)
    ctx.enqueue_free(dc)


def test_random_bit_exact() raises:
    var rng = Rng(0x9E3779B97F4A7C15)
    var tiles: List[Int] = [1, 2, 4, 8, 16]
    var checked = 0
    for t in tiles:
        var ctx = PhantomContext(t)
        assert_equal(ctx.tile_n(), t)
        var total = 0
        for _ in range(6):
            var m = rng.range(1, 3 * t + 2)
            var k = rng.range(1, 64)
            var n = rng.range(1, 3 * t + 2)
            check_matmul(ctx, rng, m, k, n)
            total += predicted_cycles(t, m, k, n)
            checked += 1
        assert_equal(ctx.launches(), 6)
        assert_equal(ctx.total_cycles(), total)
    print("    ", checked, "random shapes bit-exact, cycles == formula")


def test_edges() raises:
    var rng = Rng(7)
    var ctx = PhantomContext(4)
    check_matmul(ctx, rng, 1, 1, 1)  # smallest problem
    check_matmul(ctx, rng, 4, 17, 4)  # exactly one tile
    check_matmul(ctx, rng, 5, 3, 9)  # ragged edges in both dims
    check_matmul(ctx, rng, 3, 2000, 2, extreme=True)  # -128*-128*2000, no overflow


def test_independent_devices() raises:
    var rng = Rng(11)
    var d0 = PhantomContext(2)
    var d1 = PhantomContext(8)
    check_matmul(d0, rng, 3, 5, 3)
    check_matmul(d1, rng, 9, 5, 9)
    assert_equal(d0.launches(), 1)
    assert_equal(d1.launches(), 1)
    assert_equal(d0.last_cycles(), predicted_cycles(2, 3, 5, 3))


def test_functional_mode() raises:
    var rng = Rng(99)
    var ctx = PhantomContext(8)
    ctx.set_mode(functional=True)
    assert_true(ctx.is_functional())
    for _ in range(5):
        check_matmul(ctx, rng, rng.range(1, 40), rng.range(1, 300), rng.range(1, 40))
    check_matmul(ctx, rng, 2, 3000, 3, extreme=True)
    ctx.set_mode(functional=False)
    assert_false(ctx.is_functional())
    check_matmul(ctx, rng, 9, 17, 10)


def expect_error(msg_part: String, err: Error) raises:
    var s = String(err)
    assert_true(msg_part in s, String(t"error [{s}] lacks [{msg_part}]"))


def test_errors() raises:
    try:
        _ = PhantomContext(3)
        raise Error("tile_n=3 accepted")
    except e:
        expect_error("phantom_open", e)
    var ctx = PhantomContext(4)
    var a = ctx.enqueue_create_buffer[DType.int8](4)
    var c = ctx.enqueue_create_buffer[DType.int32](4)
    try:  # A is 2x4 = 8 bytes but the buffer holds 4
        _ = ctx.enqueue_matmul_i8(a, a, c, 2, 4, 1)
        raise Error("out-of-bounds launch accepted")
    except e:
        expect_error("outside allocation", e)
    try:
        _ = ctx.enqueue_matmul_i8(a, a, c, 1, 0, 1)
        raise Error("K=0 accepted")
    except e:
        expect_error("matmul_i8", e)
    ctx.enqueue_free(a)
    try:
        ctx.enqueue_copy(a, [Int8(1), 2, 3, 4])
        raise Error("use after free accepted")
    except e:
        expect_error("copy host->device", e)
    assert_equal(ctx.launches(), 0)


def main() raises:
    var names: List[String] = [
        "target_device_kind", "random_bit_exact", "edges",
        "independent_devices", "errors", "functional_mode",
    ]
    var passed = 0
    for i in range(len(names)):
        try:
            if i == 0:
                test_target_device_kind()
            elif i == 1:
                test_random_bit_exact()
            elif i == 2:
                test_edges()
            elif i == 3:
                test_independent_devices()
            elif i == 4:
                test_errors()
            else:
                test_functional_mode()
            print("PASS", names[i])
            passed += 1
        except e:
            print("FAIL", names[i], "-", e)
    print(t"phantom: {passed}/{len(names)} passed")
    if passed != len(names):
        raise Error("phantom tests failed")
