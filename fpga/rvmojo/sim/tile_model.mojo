"""Functional model of the v1 tile Wishbone slave (fpga/soc/rtl/tile_v1_wb.v).

Register map (byte offsets): 0x00 W CTRL start / R STATUS {done,in_ready,idle},
0x04 KLEN, 0x08 A (packed a_col), 0x0C B (packed b_row; pushes a beat),
0x10 CYCLES, 0x14 ID = 0x71110004, 0x40 + 4*(4r+j) C[r][j] (int32).
Bit-exact for results; CYCLES is a nominal K + 2N + 1 (not timing-accurate).
"""

comptime TILE_BASE: UInt32 = 0x82000000
comptime TILE_SPAN: UInt32 = 0x1000
comptime N = 4


def byte_s8(v: UInt32, i: Int) -> Int32:
    return Int32(Int8(UInt8((v >> UInt32(8 * i)) & 0xFF)))


struct TileModel(Movable):
    var ops: Int  # completed passes
    var klen: UInt32
    var a: UInt32
    var running: Bool
    var done: Bool
    var beats: UInt32
    var cycles: UInt32
    var acc: List[Int32]
    var c: List[Int32]
    var fault: Bool  # inject a one-off error (negative test of the self-check)

    def __init__(out self):
        self.fault = False
        self.ops = 0
        self.klen = 0
        self.a = 0
        self.running = False
        self.done = False
        self.beats = 0
        self.cycles = 0
        self.acc = List[Int32](length=N * N, fill=0)
        self.c = List[Int32](length=N * N, fill=0)

    def read(mut self, off: UInt32) -> UInt32:
        var w = Int(off >> 2) & 0x3F
        if w == 0:
            var idle = UInt32(0 if self.running else 1)
            var in_ready = UInt32(1 if self.running else 0)
            var done = UInt32(1 if self.done else 0)
            return idle | (in_ready << 1) | (done << 2)
        if w == 1:
            return self.klen
        if w == 2:
            return self.a
        if w == 4:
            return self.cycles
        if w == 5:
            return 0x71110004
        if w >= 0x10 and w < 0x10 + N * N:
            return UInt32(self.c[w - 0x10])
        return 0

    def write(mut self, off: UInt32, v: UInt32):
        var w = Int(off >> 2) & 0x3F
        if w == 0:
            if not self.running and self.klen != 0:
                self.running = True
                self.done = False
                self.beats = 0
                for i in range(N * N):
                    self.acc[i] = 0
        elif w == 1:
            self.klen = v & 0xFFFF
        elif w == 2:
            self.a = v
        elif w == 3 and self.running:
            for r in range(N):
                var ar = byte_s8(self.a, r)
                for j in range(N):
                    self.acc[r * N + j] += ar * byte_s8(v, j)
            self.beats += 1
            if self.beats == self.klen:
                for i in range(N * N):
                    self.c[i] = self.acc[i]
                if self.fault and self.ops == 3:
                    self.c[5] += 1
                self.running = False
                self.done = True
                self.cycles = self.klen + UInt32(2 * N + 1)
                self.ops += 1
