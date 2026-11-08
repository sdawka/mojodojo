"""Cycle-accurate spot checks of routed matmul tiles on the register-level tile model.

Usage: mojo run -I .. tests/spot_check.mojo <case_dir>...
Each case dir (written by tests/test_maxop.py) holds
    meta.txt  "N M K N_out stall"   (stall=1: random in_valid/out_ready stalls + garbage)
    a.bin     int8  [M, K]      quantized activations, as fed to the MAX op
    b.bin     int8  [K, N_out]  quantized weights, transposed (B = W^T)
    c.bin     int32 [M, N_out]  what the MAX custom op returned for this window
For each case it steps `systolic.driver.run_matmul[N]` (fpga/systolic, the
cycle-accurate model the RTL matches) and checks
  * the tile's int32 outputs equal the MAX op's, bit for bit, and
  * with an ideal host, the stepped cycle count equals the analytical formula
    ceil(M/N)*ceil(N_out/N)*(K+3N-1) that the adapter reports;
    with stalls, cycles == formula + in_stalls + out_stalls.
Prints one `SPOT` line per case and raises on any mismatch.
"""

from std.sys import argv

from systolic.driver import HostConfig, predicted_cycles, run_matmul
from systolic.trace import TraceWriter


def read_i8(path: String) raises -> List[Int8]:
    var raw: List[UInt8]
    with open(path, "r") as f:
        raw = f.read_bytes()
    var out = List[Int8](capacity=len(raw))
    for i in range(len(raw)):
        out.append(raw[i].cast[DType.int8]())
    return out^


def read_i32(path: String) raises -> List[Int32]:
    var raw: List[UInt8]
    with open(path, "r") as f:
        raw = f.read_bytes()
    var out = List[Int32](capacity=len(raw) // 4)
    for i in range(0, len(raw), 4):
        var u = (
            raw[i].cast[DType.uint32]()
            | (raw[i + 1].cast[DType.uint32]() << 8)
            | (raw[i + 2].cast[DType.uint32]() << 16)
            | (raw[i + 3].cast[DType.uint32]() << 24)
        )
        out.append(u.cast[DType.int32]())
    return out^


def run_case[N: Int](dir: String, m: Int, k: Int, n_out: Int, stall: Bool) raises:
    var a = read_i8(dir + "/a.bin")
    var b = read_i8(dir + "/b.bin")
    var want = read_i32(dir + "/c.bin")
    if len(a) != m * k or len(b) != k * n_out or len(want) != m * n_out:
        raise Error(t"{dir}: file sizes do not match meta")
    var cfg = HostConfig.stalling(UInt64(m * 7 + n_out)) if stall else HostConfig.ideal()
    var trace = TraceWriter()
    var c = List[Int32]()
    var s = run_matmul[N](a, b, m, k, n_out, cfg, trace, c)
    var bad = 0
    for i in range(m * n_out):
        if c[i] != want[i]:
            bad += 1
    var pred = predicted_cycles(N, m, k, n_out)
    var expect = pred + s.in_stalls + s.out_stalls
    print(
        t"SPOT {dir} N={N} M={m} K={k} N_out={n_out} stall={stall}"
        t" cycles={s.cycles} formula={pred} stalls={s.in_stalls + s.out_stalls}"
        t" mismatches={bad}"
    )
    if bad != 0:
        raise Error(t"{dir}: {bad} int32 mismatches vs MAX op")
    if s.cycles != expect:
        raise Error(t"{dir}: cycles {s.cycles} != formula+stalls {expect}")


def main() raises:
    var args = argv()
    for i in range(1, len(args)):
        var dir = String(args[i])
        var meta: String
        with open(dir + "/meta.txt", "r") as f:
            meta = f.read()
        var parts = meta.strip().split(" ")
        var n = Int(parts[0])
        var m = Int(parts[1])
        var k = Int(parts[2])
        var n_out = Int(parts[3])
        var stall = Int(parts[4]) != 0
        if n == 4:
            run_case[4](dir, m, k, n_out, stall)
        elif n == 8:
            run_case[8](dir, m, k, n_out, stall)
        elif n == 16:
            run_case[16](dir, m, k, n_out, stall)
        else:
            raise Error(t"unsupported tile size {n}")
    print("SPOT all", len(args) - 1, "cases passed")
