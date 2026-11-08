"""Bus-level model of tile v2 (fpga/tile2/INTERFACE.md).

This is the executable spec the generated `tile2_n{N}` RTL is co-simulated
against. It models the Wishbone slave at transaction level with exact cycle
timing: every access is a request cycle plus an ACK cycle (2 cycles), the
caller supplies the idle gap before each access, and the model tracks
absolute time (cycle 0 = first cycle after reset).

A run is computed functionally at the START write, in the hardware's tile and
row order (tm outer, tn inner, drain row r, lane j), with the hardware's
modular address arithmetic. Its timing is the closed form

    BUSY cycles = 2 + ceil(M/N) * ceil(N_cols/N) * (K + 3N)

(1 setup cycle + 1 tail cycle for the pipelined last C write; per pass:
1 start + K feed + (2N-1) flush + N drain). The
co-simulation compares every read (including STATUS polls and CYCLES while
busy) and the total cycle count with the RTL.
"""

from std.math import ceildiv

comptime T2_ID = 0x54494C32
comptime T2_REGION_CSR = 0
comptime T2_REGION_A = 1
comptime T2_REGION_B = 2
comptime T2_REGION_C = 3

# CSR byte offsets.
comptime T2_ID_OFF = 0x00
comptime T2_INFO = 0x04
comptime T2_CTRL = 0x08
comptime T2_STATUS = 0x0C
comptime T2_MODE = 0x10
comptime T2_M = 0x14
comptime T2_K = 0x18
comptime T2_N = 0x1C
comptime T2_A_BASE = 0x20
comptime T2_A_LD = 0x24
comptime T2_B_BASE = 0x28
comptime T2_B_LD = 0x2C
comptime T2_C_BASE = 0x30
comptime T2_C_LD = 0x34
comptime T2_CYCLES = 0x38

# Window byte offsets of the buffers.
comptime T2_A_WIN = 0x08000
comptime T2_B_WIN = 0x10000
comptime T2_C_WIN = 0x18000

comptime ST_BUSY = 1
comptime ST_DONE = 2
comptime ST_ERR_CFG = 4
comptime ST_ERR_BUSY = 8


def ilog2(x: Int) -> Int:
    var n = 0
    while (1 << n) < x:
        n += 1
    return n


def t2_busy_cycles(tile_n: Int, m: Int, k: Int, n: Int) -> Int:
    """Closed-form BUSY length of one run (INTERFACE.md s6)."""
    return 2 + ceildiv(m, tile_n) * ceildiv(n, tile_n) * (k + 3 * tile_n)


@fieldwise_init
struct BusResult(Copyable, Movable):
    var rdata: Int  # 32-bit read data (0 for writes)
    var irq: Bool  # irq output during the request cycle
    var t: Int  # request cycle


struct Tile2Model[N: Int, A_BYTES: Int, B_BYTES: Int, C_BYTES: Int](Movable):
    comptime LN = ilog2(Self.N)
    comptime A_DEPTH = Self.A_BYTES // Self.N
    comptime B_DEPTH = Self.B_BYTES // Self.N
    comptime C_WORDS = Self.C_BYTES // 4
    comptime C_DEPTH = Self.C_WORDS // Self.N
    comptime INFO = (
        Self.N
        | (ilog2(Self.A_BYTES) << 8)
        | (ilog2(Self.B_BYTES) << 16)
        | (ilog2(Self.C_BYTES) << 24)
    )

    var a: List[UInt8]
    var b: List[UInt8]
    var c: List[UInt32]
    var mode: Int
    var m: Int
    var k: Int
    var n: Int
    var a_base_w: Int  # in N-byte rows
    var a_ld_w: Int
    var b_base_w: Int
    var b_ld_w: Int
    var c_base_w: Int  # in N-word rows
    var c_ld_w: Int
    var done: Bool
    var err_cfg: Bool
    var err_busy: Bool
    var running: Bool
    var run_begin: Int  # first BUSY cycle
    var run_end: Int  # first cycle after BUSY
    var cycles_last: Int
    var now: Int  # next free cycle
    var runs: Int

    def __init__(out self):
        comptime assert Self.N >= 4 and (Self.N & (Self.N - 1)) == 0
        self.a = List[UInt8](length=Self.A_BYTES, fill=0)
        self.b = List[UInt8](length=Self.B_BYTES, fill=0)
        self.c = List[UInt32](length=Self.C_WORDS, fill=0)
        self.mode = 0
        self.m = 0
        self.k = 0
        self.n = 0
        self.a_base_w = 0
        self.a_ld_w = 0
        self.b_base_w = 0
        self.b_ld_w = 0
        self.c_base_w = 0
        self.c_ld_w = 0
        self.done = False
        self.err_cfg = False
        self.err_busy = False
        self.running = False
        self.run_begin = 0
        self.run_end = 0
        self.cycles_last = 0
        self.now = 0
        self.runs = 0

    def reset(mut self, gap: Int):
        """`gap` idle cycles, then one cycle with rst=1. Buffers keep contents."""
        self._settle(self.now + gap)
        self.mode = 0
        self.m = 0
        self.k = 0
        self.n = 0
        self.a_base_w = 0
        self.a_ld_w = 0
        self.b_base_w = 0
        self.b_ld_w = 0
        self.c_base_w = 0
        self.c_ld_w = 0
        self.done = False
        self.err_cfg = False
        self.err_busy = False
        self.running = False
        self.cycles_last = 0
        self.now += gap + 1

    def _settle(mut self, t: Int):
        if self.running and t >= self.run_end:
            self.running = False
            self.done = True
            self.cycles_last = self.run_end - self.run_begin

    def busy_until(self) -> Int:
        """First non-BUSY cycle of the current run (or `now` if idle)."""
        return self.run_end if self.running else self.now

    def irq(self) -> Bool:
        return self.done and (self.mode & 2) != 0

    def status(self) -> Int:
        var s = 0
        if self.running:
            s |= ST_BUSY
        if self.done:
            s |= ST_DONE
        if self.err_cfg:
            s |= ST_ERR_CFG
        if self.err_busy:
            s |= ST_ERR_BUSY
        return s

    def _csr_read(self, idx: Int, t: Int) -> Int:
        if idx == 0:
            return T2_ID
        if idx == 1:
            return Self.INFO
        if idx == 3:
            return self.status()
        if idx == 4:
            return self.mode
        if idx == 5:
            return self.m
        if idx == 6:
            return self.k
        if idx == 7:
            return self.n
        if idx == 8:
            return self.a_base_w << Self.LN
        if idx == 9:
            return self.a_ld_w << Self.LN
        if idx == 10:
            return self.b_base_w << Self.LN
        if idx == 11:
            return self.b_ld_w << Self.LN
        if idx == 12:
            return self.c_base_w << (Self.LN + 2)
        if idx == 13:
            return self.c_ld_w << (Self.LN + 2)
        if idx == 14:
            return (t - self.run_begin) if self.running else self.cycles_last
        return 0

    def access(
        mut self, gap: Int, we: Bool, addr: Int, data: Int, sel: Int
    ) -> BusResult:
        """One Wishbone classic access whose request cycle is `now + gap`."""
        var t = self.now + gap
        self._settle(t)
        var irq = self.irq()
        var busy = self.running
        var region = (addr >> 15) & 3
        var off = addr & 0x7FFF
        var rdata = 0
        if busy and (we or region != T2_REGION_CSR):
            self.err_busy = True  # dropped write / zero read
        elif we:
            self._write(region, off, data & 0xFFFFFFFF, sel, t)
        else:
            rdata = self._read(region, off, t)
        self.now = t + 2
        return BusResult(rdata, irq, t)

    def _read(self, region: Int, off: Int, t: Int) -> Int:
        if region == T2_REGION_CSR:
            return self._csr_read((off >> 2) & 15, t)
        if region == T2_REGION_C:
            return Int(self.c[(off >> 2) % Self.C_WORDS])
        var w = 0
        for bb in range(4):
            var byte_i = ((off & ~3) + bb)
            var v: Int
            if region == T2_REGION_A:
                v = Int(self.a[byte_i % Self.A_BYTES])
            else:
                v = Int(self.b[byte_i % Self.B_BYTES])
            w |= v << (8 * bb)
        return w

    def _write(mut self, region: Int, off: Int, data: Int, sel: Int, t: Int):
        if region == T2_REGION_CSR:
            var idx = (off >> 2) & 15
            if idx == 2:
                if data & 2:
                    self.done = False
                    self.err_cfg = False
                    self.err_busy = False
                if data & 1:
                    if self.m != 0 and self.k != 0 and self.n != 0:
                        self._start(t)
                    else:
                        self.err_cfg = True
            elif idx == 4:
                self.mode = data & 3
            elif idx == 5:
                self.m = data & 0xFFFF
            elif idx == 6:
                self.k = data & 0xFFFF
            elif idx == 7:
                self.n = data & 0xFFFF
            elif idx == 8:
                self.a_base_w = (data >> Self.LN) % Self.A_DEPTH
            elif idx == 9:
                self.a_ld_w = (data >> Self.LN) % Self.A_DEPTH
            elif idx == 10:
                self.b_base_w = (data >> Self.LN) % Self.B_DEPTH
            elif idx == 11:
                self.b_ld_w = (data >> Self.LN) % Self.B_DEPTH
            elif idx == 12:
                self.c_base_w = (data >> (Self.LN + 2)) % Self.C_DEPTH
            elif idx == 13:
                self.c_ld_w = (data >> (Self.LN + 2)) % Self.C_DEPTH
            return
        if region == T2_REGION_C:
            var wi = (off >> 2) % Self.C_WORDS
            var old = Int(self.c[wi])
            for bb in range(4):
                if sel & (1 << bb):
                    var mask = 0xFF << (8 * bb)
                    old = (old & ~mask) | (data & mask)
            self.c[wi] = UInt32(old & 0xFFFFFFFF)
            return
        for bb in range(4):
            if sel & (1 << bb):
                var byte_i = (off & ~3) + bb
                var v = UInt8((data >> (8 * bb)) & 0xFF)
                if region == T2_REGION_A:
                    self.a[byte_i % Self.A_BYTES] = v
                else:
                    self.b[byte_i % Self.B_BYTES] = v

    def _start(mut self, t: Int):
        """Accepted START at request cycle t: compute C, schedule BUSY."""
        comptime n_t = Self.N
        self.done = False
        self.running = True
        self.runs += 1
        self.run_begin = t + 1
        self.run_end = self.run_begin + t2_busy_cycles(n_t, self.m, self.k, self.n)
        var accum = (self.mode & 1) != 0
        var mt = ceildiv(self.m, n_t)
        var nt = ceildiv(self.n, n_t)
        for tm in range(mt):
            for tn in range(nt):
                for r in range(n_t):
                    var mi = tm * n_t + r
                    if mi >= self.m:
                        continue
                    var crow = (
                        self.c_base_w + (tm * n_t + r) * self.c_ld_w + tn
                    ) % Self.C_DEPTH
                    for j in range(n_t):
                        var nj = tn * n_t + j
                        if nj >= self.n:
                            continue
                        var acc = 0
                        for kk in range(self.k):
                            var arow = (
                                self.a_base_w + kk * self.a_ld_w + tm
                            ) % Self.A_DEPTH
                            var brow = (
                                self.b_base_w + kk * self.b_ld_w + tn
                            ) % Self.B_DEPTH
                            var av = Int(self.a[arow * n_t + r].cast[DType.int8]())
                            var bv = Int(self.b[brow * n_t + j].cast[DType.int8]())
                            acc += av * bv
                        var wi = crow * n_t + j
                        if accum:
                            acc += Int(self.c[wi])
                        self.c[wi] = UInt32(acc & 0xFFFFFFFF)
