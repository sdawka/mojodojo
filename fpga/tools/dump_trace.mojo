"""Dump per-cycle golden port vectors for one randomized matmul.

pixi run trace -- [--tile 4] [--m 5] [--k 7] [--n 6] [--seed 1]
                  [--format text|jsonl] [--stalls] [--out traces/t.txt]
--tile must be one of 1, 2, 3, 4, 8, 16 (it is a comptime parameter).
"""

from std.sys import argv

from systolic.driver import HostConfig, predicted_cycles, run_matmul
from systolic.reference import reference_matmul
from systolic.rng import Rng
from systolic.trace import TraceWriter, TRACE_TEXT, TRACE_JSONL


def dump[
    N: Int
](
    m: Int, k: Int, n_out: Int, seed: UInt64, mode: Int, stalls: Bool, path: String
) raises:
    var rng = Rng(seed)
    var a = List[Int8](capacity=m * k)
    for _ in range(m * k):
        a.append(rng.scalar[DType.int8]())
    var b = List[Int8](capacity=k * n_out)
    for _ in range(k * n_out):
        b.append(rng.scalar[DType.int8]())
    var cfg = HostConfig.stalling(seed) if stalls else HostConfig.ideal()
    var trace = TraceWriter(mode)
    var c = List[Int32]()
    var s = run_matmul[N](a, b, m, k, n_out, cfg, trace, c)
    var want = reference_matmul[DType.int8, DType.int32](a, b, m, k, n_out)
    for i in range(m * n_out):
        if c[i] != want[i]:
            raise Error(t"simulator mismatch at {i}")
    trace.save(path)
    print(
        t"wrote {trace.lines} cycles to {path}: N={N} M={m} K={k} N_out={n_out}"
        t" seed={seed} cycles={s.cycles} (formula {predicted_cycles(N, m, k, n_out)}"
        t" + in_stalls {s.in_stalls} + out_stalls {s.out_stalls}); result bit-exact"
    )


def main() raises:
    var tile = 4
    var m = 5
    var k = 7
    var n_out = 6
    var seed = 1
    var mode = TRACE_TEXT
    var stalls = False
    var path = String("traces/trace.txt")
    var args = argv()
    var i = 1
    while i < len(args):
        var flag = String(args[i])
        if flag == "--stalls":
            stalls = True
            i += 1
            continue
        if i + 1 >= len(args):
            raise Error(t"missing value for {flag}")
        var val = String(args[i + 1])
        if flag == "--tile":
            tile = Int(val)
        elif flag == "--m":
            m = Int(val)
        elif flag == "--k":
            k = Int(val)
        elif flag == "--n":
            n_out = Int(val)
        elif flag == "--seed":
            seed = Int(val)
        elif flag == "--out":
            path = val
        elif flag == "--format":
            mode = TRACE_JSONL if val == "jsonl" else TRACE_TEXT
        else:
            raise Error(t"unknown flag {flag}")
        i += 2
    var s = UInt64(seed)
    match tile:
        case 1:
            dump[1](m, k, n_out, s, mode, stalls, path)
        case 2:
            dump[2](m, k, n_out, s, mode, stalls, path)
        case 3:
            dump[3](m, k, n_out, s, mode, stalls, path)
        case 4:
            dump[4](m, k, n_out, s, mode, stalls, path)
        case 8:
            dump[8](m, k, n_out, s, mode, stalls, path)
        case 16:
            dump[16](m, k, n_out, s, mode, stalls, path)
        case _:
            raise Error(t"--tile {tile} not instantiated; use 1, 2, 3, 4, 8 or 16")
