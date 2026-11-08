"""Randomized register-level programs for tile v2 co-simulation.

    mojo run -I . tile2/prog.mojo -- --tile 8 --seed 1 --jobs 6 --out prog.txt

A program is what firmware would do: load Aᵀ / B (and C for ACCUM) through the
bus, program the CSRs, START, poll STATUS (or idle-wait for irq), read
CYCLES and C back. Illegal accesses while BUSY, rejected STARTs, CSR
read-back/masking, aliasing and resets are mixed in. Every access is driven
through `Tile2Model`, which supplies the expected read data, irq level and
timing; every run's C is also checked against an independent reference
matmul computed from the original matrices (not from the buffers).

Line format (one bus operation per line, all numbers hex except gap/sel/irq):
    W <gap> <addr> <data> <sel> <irq>    write; irq = expected irq in request cycle
    R <gap> <addr> <expect> <irq>        read, compare wb_dat_r in the ACK cycle
    X <gap>                              gap idle cycles, then one rst cycle
    END <cycles> <runs>                  total cycles after the initial reset
`gap` idle cycles (cyc=0) precede each access; an access is 2 cycles.
"""

from std.sys import argv
from std.math import ceildiv

from systolic.rng import Rng
from systolic.tile2 import (
    Tile2Model,
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
    t2_busy_cycles,
)


def hx(v: Int) -> String:
    comptime digits = "0123456789abcdef"
    var s = String()
    var started = False
    for k in range(7, -1, -1):
        var d = (v >> (4 * k)) & 0xF
        if d != 0 or started or k == 0:
            started = True
            s += String(digits[byte=d])
    return s^


struct Prog[N: Int, A_BYTES: Int, B_BYTES: Int, C_BYTES: Int](Movable):
    var model: Tile2Model[Self.N, Self.A_BYTES, Self.B_BYTES, Self.C_BYTES]
    var out: String
    var ops: Int

    def __init__(out self):
        self.model = Tile2Model[Self.N, Self.A_BYTES, Self.B_BYTES, Self.C_BYTES]()
        self.out = String()
        self.ops = 0

    def wr(mut self, gap: Int, addr: Int, data: Int, sel: Int = 15):
        var r = self.model.access(gap, True, addr, data, sel)
        self.out += String(
            "W ", gap, " ", hx(addr), " ", hx(data & 0xFFFFFFFF), " ", sel, " ",
            1 if r.irq else 0, "\n",
        )
        self.ops += 1

    def rd(mut self, gap: Int, addr: Int) -> Int:
        var r = self.model.access(gap, False, addr, 0, 15)
        self.out += String("R ", gap, " ", hx(addr), " ", hx(r.rdata), " ", 1 if r.irq else 0, "\n")
        self.ops += 1
        return r.rdata

    def comment(mut self, s: String):
        self.out += String("# ", s, "\n")


def pick_gap(mut rng: Rng) -> Int:
    var r = rng.range(0, 99)
    if r < 60:
        return 0
    if r < 90:
        return rng.range(1, 3)
    return rng.range(4, 20)


def run_prog[N: Int, A_BYTES: Int, B_BYTES: Int, C_BYTES: Int](
    seed: Int, jobs: Int, big: Bool
) raises -> String:
    var p = Prog[N, A_BYTES, B_BYTES, C_BYTES]()
    var rng = Rng(UInt64(seed))
    p.comment(String("tile2 program N=", N, " A=", A_BYTES, " B=", B_BYTES, " C=", C_BYTES, " seed=", seed))

    # ---- identification, CSR masking, aliasing, read-only behaviour
    if p.rd(0, 0x00) != T2_ID:
        raise Error("model ID")
    _ = p.rd(1, 0x04)
    _ = p.rd(0, 0x3C)
    _ = p.rd(0, T2_CTRL)
    for _ in range(3):
        for off in range(T2_MODE, T2_CYCLES, 4):
            p.wr(pick_gap(rng), off, Int(rng.next() & 0xFFFFFFFF))
        for off in range(0, 0x40, 4):
            _ = p.rd(pick_gap(rng), off + 0x40 * rng.range(0, 511))  # aliased reads
    p.wr(0, T2_ID, 0x12345678)  # RO: ignored
    p.wr(0, T2_CYCLES, 0x12345678)
    _ = p.rd(0, T2_ID)
    _ = p.rd(0, T2_CYCLES)
    var rgap = rng.range(0, 3)
    p.model.reset(rgap)
    p.out += String("X ", rgap, "\n")
    _ = p.rd(0, T2_M)  # 0 after reset

    # ---- rejected START (K = 0)
    p.wr(0, T2_M, 3)
    p.wr(0, T2_K, 0)
    p.wr(0, T2_N, 3)
    p.wr(0, T2_CTRL, 1)
    if p.rd(0, T2_STATUS) != 4:
        raise Error("model ERR_CFG")
    p.wr(0, T2_CTRL, 2)  # CLEAR
    _ = p.rd(0, T2_STATUS)

    var mean_k = 48
    for job in range(jobs):
        # ---- shape and layout
        var m = rng.range(1, 3 * N + 1)
        var kk = rng.range(1, mean_k)
        var nc = rng.range(1, 3 * N + 3)
        if rng.pct(10):
            kk = rng.range(100, 400)
        if big and job == 0:
            m = 2 * N
            nc = 2 * N
            kk = 1000
        var mp = ceildiv(m, N) * N
        var np = ceildiv(nc, N) * N
        var a_ld = mp + N * rng.range(0, 2)
        var b_ld = np + N * rng.range(0, 2)
        var c_ld = 4 * np + 4 * N * rng.range(0, 1)
        kk = min(kk, A_BYTES // a_ld, B_BYTES // b_ld)
        m = min(m, C_BYTES // c_ld)
        mp = ceildiv(m, N) * N
        var a_base = N * rng.range(0, A_BYTES // N - 1)
        var b_base = N * rng.range(0, B_BYTES // N - 1)
        var c_base = 4 * N * rng.range(0, C_BYTES // (4 * N) - 1)
        var accum = rng.pct(30)
        var irq_en = rng.pct(50)
        var extreme = rng.pct(10)
        p.comment(String("job ", job, ": M=", m, " K=", kk, " N=", nc, " accum=", accum))

        # ---- matrices (int8), reference
        var a = List[Int](length=m * kk, fill=0)
        var bm = List[Int](length=kk * nc, fill=0)
        for i in range(m * kk):
            a[i] = -128 if extreme else Int(rng.scalar[DType.int8]())
        for i in range(kk * nc):
            bm[i] = (-128 if rng.pct(50) else 127) if extreme else Int(rng.scalar[DType.int8]())
        var cpre = List[Int](length=m * np, fill=0)

        # ---- load Aᵀ rows [0, mp) (padding bytes are garbage), mixed store widths
        for k in range(kk):
            var w = 0
            while w < mp:
                var word = 0
                for bb in range(4):
                    var mi = w + bb
                    var v = Int(rng.next() & 0xFF)
                    if mi < m:
                        v = a[mi * kk + k] & 0xFF
                    word |= v << (8 * bb)
                var addr = T2_A_WIN + (a_base + k * a_ld + w) % A_BYTES
                if rng.pct(85):
                    p.wr(pick_gap(rng), addr, word)
                else:
                    for bb in range(4):  # byte stores
                        p.wr(0, addr + bb, word, 1 << bb)
                w += 4
        for k in range(kk):
            var w = 0
            while w < np:
                var word = 0
                for bb in range(4):
                    var nj = w + bb
                    var v = Int(rng.next() & 0xFF)
                    if nj < nc:
                        v = bm[k * nc + nj] & 0xFF
                    word |= v << (8 * bb)
                var addr = T2_B_WIN + (b_base + k * b_ld + w) % B_BYTES
                if rng.pct(85):
                    p.wr(pick_gap(rng), addr, word)
                else:
                    p.wr(0, addr, word, 3)  # half-word stores
                    p.wr(0, addr, word, 12)
                w += 4
        # ---- C: preload (accum) or fill with garbage (checks overwrite + untouched padding)
        if accum or rng.pct(40):
            for mi in range(m + (1 if rng.pct(50) else 0)):
                for nj in range(np):
                    var v = Int(rng.next() & 0xFFFFFFFF)
                    if mi < m:
                        cpre[mi * np + nj] = v
                    var addr = T2_C_WIN + (c_base + mi * c_ld + 4 * nj) % C_BYTES
                    if rng.pct(90):
                        p.wr(pick_gap(rng), addr, v)
                    else:
                        p.wr(0, addr, v, 1)
                        p.wr(0, addr, v, 14)
        for mi in range(m):
            for nj in range(np):
                var addr = (c_base + mi * c_ld + 4 * nj) % C_BYTES
                cpre[mi * np + nj] = Int(p.model.c[addr // 4])

        # ---- CSRs (high garbage bits above the buffer size are ignored)
        p.wr(pick_gap(rng), T2_M, m | (rng.range(0, 0xFFFF) << 16))
        p.wr(pick_gap(rng), T2_K, kk)
        p.wr(pick_gap(rng), T2_N, nc)
        p.wr(pick_gap(rng), T2_A_BASE, a_base + A_BYTES * rng.range(0, 3) + rng.range(0, N - 1))
        p.wr(pick_gap(rng), T2_A_LD, a_ld)
        p.wr(pick_gap(rng), T2_B_BASE, b_base + B_BYTES * rng.range(0, 3))
        p.wr(pick_gap(rng), T2_B_LD, b_ld + rng.range(0, N - 1))
        p.wr(pick_gap(rng), T2_C_BASE, c_base + rng.range(0, 4 * N - 1))
        p.wr(pick_gap(rng), T2_C_LD, c_ld)
        p.wr(pick_gap(rng), T2_MODE, (1 if accum else 0) | (2 if irq_en else 0))

        # ---- run
        var expect_cycles = t2_busy_cycles(N, m, kk, nc)
        p.wr(pick_gap(rng), T2_CTRL, 3 if rng.pct(50) else 1)
        var illegal = rng.pct(25)
        if illegal:
            p.wr(rng.range(0, 5), T2_M, 1)  # ignored, ERR_BUSY
            p.wr(0, T2_CTRL, 3)  # ignored (also CLEAR)
            p.wr(0, T2_C_WIN + c_base, 0xDEADBEEF)  # dropped
            _ = p.rd(0, T2_A_WIN + a_base)  # reads 0
            _ = p.rd(0, T2_CYCLES)
        if rng.pct(30):
            # irq-style wait: one access after a long idle gap
            var remaining = p.model.busy_until() - p.model.now
            _ = p.rd(max(0, remaining + rng.range(-2, 2)), T2_STATUS)
        while True:
            var st = p.rd(rng.range(0, 30), T2_STATUS)
            if (st & ST_BUSY) == 0:
                if (st & ST_DONE) == 0:
                    raise Error("model: not done after busy")
                break
        if p.rd(0, T2_CYCLES) != expect_cycles:
            raise Error("model CYCLES")
        _ = p.rd(0, T2_M)

        # ---- read back C (incl. padding columns, one extra row) and check reference
        for mi in range(m + 1):
            for nj in range(np):
                var off = (c_base + mi * c_ld + 4 * nj) % C_BYTES
                var got = p.rd(0 if rng.pct(80) else 1, T2_C_WIN + off)
                if mi < m and nj < nc:
                    var ref_v = cpre[mi * np + nj] if accum else 0
                    for k in range(kk):
                        ref_v += a[mi * kk + k] * bm[k * nc + nj]
                    if got != (ref_v & 0xFFFFFFFF):
                        raise Error(String("model C mismatch vs reference at ", mi, ",", nj))
                elif mi < m:
                    if got != (cpre[mi * np + nj] & 0xFFFFFFFF):
                        raise Error("model wrote C padding")
        if illegal or rng.pct(50):
            p.wr(0, T2_CTRL, 2)  # CLEAR
        if rng.pct(10):
            _ = p.rd(0, T2_A_WIN + (a_base % A_BYTES))  # spot-check operand read-back
            _ = p.rd(0, T2_B_WIN + (b_base % B_BYTES))

    p.out += String("END ", p.model.now, " ", p.model.runs, "\n")
    return p.out.copy()


def main() raises:
    var args = argv()
    var tile = 8
    var seed = 1
    var jobs = 6
    var big = False
    var path = String("prog.txt")
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--tile":
            tile = Int(args[i + 1])
            i += 1
        elif a == "--seed":
            seed = Int(args[i + 1])
            i += 1
        elif a == "--jobs":
            jobs = Int(args[i + 1])
            i += 1
        elif a == "--big":
            big = True
        elif a == "--out":
            path = String(args[i + 1])
            i += 1
        i += 1
    var text: String
    if tile == 8:
        text = run_prog[8, 16384, 32768, 32768](seed, jobs, big)
    elif tile == 4:
        text = run_prog[4, 4096, 4096, 8192](seed, jobs, big)
    else:
        raise Error("--tile must be 8 or 4")
    with open(path, "w") as f:
        f.write(text)
    print(t"wrote {path}")
