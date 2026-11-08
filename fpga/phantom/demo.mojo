"""Demo: target the phantom NPU from Mojo and run an int8 matmul on it.

Run from fpga/: `pixi run phantom-demo`.
"""

from std.time import perf_counter_ns
from phantom.context import PhantomContext
from phantom.target import Phantom, PhantomTarget, device_kind_string
from systolic.driver import predicted_cycles
from systolic.reference import reference_matmul
from systolic.rng import Rng

comptime CLOCK_HZ = 100.0e6  # same nominal clock as tools/perf.mojo


def main() raises:
    print("target device_kind:", device_kind_string[PhantomTarget]())
    print(
        "  is_accelerator:", Phantom().is_accelerator(),
        " is_gpu:", Phantom().is_gpu(),
    )
    print("host device_kind:  ", device_kind_string())

    var ctx = PhantomContext(8)
    print("device:", ctx.name(), "| api:", ctx.api())

    # A prefill-shaped slice: 64 tokens x 512 hidden -> 64 outputs.
    comptime M = 64
    comptime K = 512
    comptime N = 64
    var rng = Rng(2026)
    var a = List[Int8](capacity=M * K)
    var b = List[Int8](capacity=K * N)
    for _ in range(M * K):
        a.append(rng.scalar[DType.int8]())
    for _ in range(K * N):
        b.append(rng.scalar[DType.int8]())

    var da = ctx.enqueue_create_buffer[DType.int8](M * K)
    var db = ctx.enqueue_create_buffer[DType.int8](K * N)
    var dc = ctx.enqueue_create_buffer[DType.int32](M * N)
    ctx.enqueue_copy(da, a)
    ctx.enqueue_copy(db, b)
    var t0 = perf_counter_ns()
    var cycles = ctx.enqueue_matmul_i8(da, db, dc, M, K, N)
    var wall = Float64(perf_counter_ns() - t0) / 1e9
    var c = List[Int32]()
    ctx.enqueue_copy(c, dc)
    ctx.synchronize()

    var want = reference_matmul[DType.int8, DType.int32](a, b, M, K, N)
    var exact = True
    for i in range(M * N):
        if c[i] != want[i]:
            exact = False
    var macs = M * K * N
    var secs = Float64(cycles) / CLOCK_HZ
    print(t"matmul_i8 M={M} K={K} N={N}: bit_exact={exact}")
    print(
        t"  cycles={cycles} (formula {predicted_cycles(ctx.tile_n(), M, K, N)}),"
        t" PE utilization {Float64(macs) / Float64(cycles * 64) * 100.0}%"
    )
    print(
        t"  simulated @100 MHz: {secs * 1e6} us, {2.0 * Float64(macs) / secs / 1e9} GOP/s"
    )
    print(t"  host wall time of the simulation: {wall} s")
    print(t"  C[0][0..4] = {c[0]} {c[1]} {c[2]} {c[3]}")
    if not exact:
        raise Error("demo matmul mismatch")
