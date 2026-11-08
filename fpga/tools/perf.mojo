"""Cycle numbers for a Llama 3.2 3B prefill-sized matmul (M=512, K=3072, N_out=8192).

Simulating all 16k-65k passes register-by-register is needlessly heavy, so this
simulates the full M=512 strip of output-tile columns (every row tile, one column
tile) cycle-accurately, checks it bit-exact against the reference and against the
formula, then extrapolates by the number of column tiles (passes are identical).
"""

from std.math import ceildiv
from std.time import perf_counter_ns

from systolic.driver import HostConfig, predicted_cycles, run_matmul
from systolic.reference import reference_matmul
from systolic.rng import Rng
from systolic.trace import TraceWriter

comptime M = 512
comptime K = 3072
comptime N_OUT = 8192
comptime CLOCK_HZ = 100.0e6


def strip[N: Int]() raises:
    var rng = Rng(2026)
    var a = List[Int8](capacity=M * K)
    for _ in range(M * K):
        a.append(rng.scalar[DType.int8]())
    var b = List[Int8](capacity=K * N)
    for _ in range(K * N):
        b.append(rng.scalar[DType.int8]())
    var trace = TraceWriter()
    var c = List[Int32]()
    var t0 = perf_counter_ns()
    var s = run_matmul[N](a, b, M, K, N, HostConfig.ideal(), trace, c)
    var sim_s = Float64(perf_counter_ns() - t0) / 1e9
    var want = reference_matmul[DType.int8, DType.int32](a, b, M, K, N)
    var bad = 0
    for i in range(M * N):
        if c[i] != want[i]:
            bad += 1
    var strip_pred = predicted_cycles(N, M, K, N)
    var col_tiles = ceildiv(N_OUT, N)
    var full = s.cycles * col_tiles
    var secs = Float64(full) / CLOCK_HZ
    var macs = Float64(M) * Float64(K) * Float64(N_OUT)
    var ideal = macs / Float64(N * N)
    print(
        t"N={N}: strip M={M} K={K} N_out={N}: {s.cycles} cycles simulated"
        t" (formula {strip_pred}, mismatches {bad}, sim wall {sim_s} s)"
    )
    print(
        t"   full {M}x{K}x{N_OUT}: {s.passes * col_tiles} passes, {full} cycles;"
        t" PE utilization {ideal / Float64(full)}"
    )
    print(
        t"   @100 MHz: {secs * 1e3} ms; {2.0 * macs / secs / 1e9} GOP/s;"
        t" operand stream 2N B/cycle = {Float64(2 * N) * CLOCK_HZ / 1e6} MB/s"
    )


def main() raises:
    strip[8]()
    strip[16]()
