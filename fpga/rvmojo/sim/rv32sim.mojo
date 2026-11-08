"""Rv32sim: a small RV32IM(+Zicsr subset) instruction-set simulator in Mojo.

Functional (not cycle-accurate) model used to run bare-metal Mojo firmware
built by fpga/rvmojo/build.sh. Loads an ELF32 (PT_LOAD by physical address),
starts at e_entry, and models the memory map of the LiteX sim SoC in fpga/soc
(build/sim/csr.csv) so the same firmware image runs on both:
  rom 0x00000000 (64 KiB)  sram 0x10000000 (64 KiB)  main_ram 0x40000000 (16 MiB)
  uart    0xF0002000  rxtx(+0, write -> host stdout) txfull(+4) = 0,
                      txempty(+0x18) = 1
  simctrl 0xF0001000  finish(+0, write (code<<1)|1 ends the run with `code`),
                      latch(+4), cycles(+8 hi, +12 lo; = retired instructions)
  tile    0x82000000  functional model of the v1 tile wrapper (tile_model.mojo)
  tile2   0x90000000  tile v2 (fpga/tile2/INTERFACE.md) via lane-tile2's
                      bus-level model fpga/systolic/tile2.mojo, clocked at
                      1 cycle per retired instruction (so BUSY is observable)
ecall/ebreak also stop the run (exit status = a0).

usage: rv32sim <fw.elf> [--trace] [--max-insns N] [--tile-fault]
"""

from std.sys import argv, exit
from tile_model import TileModel, TILE_BASE, TILE_SPAN
from systolic.tile2 import Tile2Model  # lane-tile2's executable spec (-I fpga)

comptime TILE2_BASE: UInt32 = 0x90000000  # INTERFACE.md s8 suggestion
comptime TILE2_SPAN: UInt32 = 0x20000
comptime Tile2 = Tile2Model[8, 16384, 32768, 32768]  # default build (INFO 0x0F0F0E08)

comptime ROM_BASE: UInt32 = 0x00000000
comptime ROM_SIZE = 0x10000
comptime SRAM_BASE: UInt32 = 0x10000000
comptime SRAM_SIZE = 0x10000
comptime MAIN_BASE: UInt32 = 0x40000000
comptime MAIN_SIZE = 0x1000000
comptime UART_BASE: UInt32 = 0xF0002000
comptime SIMCTRL_BASE: UInt32 = 0xF0001000


def sext(v: UInt32, bits: Int) -> UInt32:
    var sh = UInt32(32 - bits)
    return UInt32(Int32(v << sh) >> Int32(sh))


struct Machine(Movable):
    var x: List[UInt32]
    var pc: UInt32
    var rom: List[UInt8]
    var sram: List[UInt8]
    var ram: List[UInt8]
    var cycles_latched: UInt64
    var tile: TileModel
    var tile2: Tile2
    var halted: Bool
    var exit_code: Int
    var instret: UInt64
    var uart_line: String
    var trace: Bool

    def __init__(out self):
        self.x = List[UInt32](length=32, fill=0)
        self.pc = 0
        self.rom = List[UInt8](length=ROM_SIZE, fill=0)
        self.sram = List[UInt8](length=SRAM_SIZE, fill=0)
        self.ram = List[UInt8](length=MAIN_SIZE, fill=0)
        self.cycles_latched = 0
        self.tile = TileModel()
        self.tile2 = Tile2()
        self.halted = False
        self.exit_code = 0
        self.instret = 0
        self.uart_line = String()
        self.trace = False

    # ---- memory -------------------------------------------------------------
    def poke8(mut self, addr: UInt32, v: UInt8) raises:
        if addr >= MAIN_BASE and addr < MAIN_BASE + MAIN_SIZE:
            self.ram[Int(addr - MAIN_BASE)] = v
        elif addr >= SRAM_BASE and addr < SRAM_BASE + SRAM_SIZE:
            self.sram[Int(addr - SRAM_BASE)] = v
        elif addr < ROM_BASE + ROM_SIZE:  # loader / firmware writes into ROM
            self.rom[Int(addr - ROM_BASE)] = v
        else:
            raise Error("bus error: byte write to " + hex(Int(addr)) + " pc=" + hex(Int(self.pc)))

    def peek8(self, addr: UInt32) raises -> UInt8:
        if addr >= MAIN_BASE and addr < MAIN_BASE + MAIN_SIZE:
            return self.ram[Int(addr - MAIN_BASE)]
        elif addr >= SRAM_BASE and addr < SRAM_BASE + SRAM_SIZE:
            return self.sram[Int(addr - SRAM_BASE)]
        elif addr < ROM_BASE + ROM_SIZE:
            return self.rom[Int(addr - ROM_BASE)]
        raise Error("bus error: read at " + hex(Int(addr)) + " pc=" + hex(Int(self.pc)))

    def is_mmio(self, addr: UInt32) -> Bool:
        return addr >= 0xF0000000 or (
            addr >= TILE_BASE and addr < TILE_BASE + TILE_SPAN
        ) or (addr >= TILE2_BASE and addr < TILE2_BASE + TILE2_SPAN)

    def tile2_access(mut self, we: Bool, addr: UInt32, size: Int, v: UInt32) -> UInt32:
        """Byte/half/word access to tile2, mapped onto Wishbone sel lanes."""
        var off = Int(addr - TILE2_BASE)
        var lane = off & 3
        var sel = ((1 << size) - 1) << lane
        var gap = max(0, Int(self.instret) - self.tile2.now)
        var r = self.tile2.access(gap, we, off & ~3, Int(v) << (8 * lane), sel)
        return UInt32((r.rdata >> (8 * lane)) & 0xFFFFFFFF)

    def load(mut self, addr: UInt32, size: Int) raises -> UInt32:
        if addr >= TILE2_BASE and addr < TILE2_BASE + TILE2_SPAN:
            var w = self.tile2_access(False, addr, size, 0)
            return w & UInt32((1 << (8 * size)) - 1) if size < 4 else w
        if self.is_mmio(addr):
            if size != 4 or (addr & 3) != 0:
                raise Error("non-word MMIO read at " + hex(Int(addr)))
            if addr >= TILE_BASE and addr < TILE_BASE + TILE_SPAN:
                return self.tile.read(addr - TILE_BASE)
            if addr == UART_BASE + 4:
                return 0  # txfull: never full
            if addr == UART_BASE + 0x18:
                return 1  # txempty
            if addr == SIMCTRL_BASE + 8:
                return UInt32(self.cycles_latched >> 32)
            if addr == SIMCTRL_BASE + 12:
                return UInt32(self.cycles_latched & 0xFFFFFFFF)
            return 0
        var v: UInt32 = 0
        for i in range(size):
            v |= UInt32(self.peek8(addr + UInt32(i))) << UInt32(8 * i)
        return v

    def store(mut self, addr: UInt32, size: Int, v: UInt32) raises:
        if addr >= TILE2_BASE and addr < TILE2_BASE + TILE2_SPAN:
            var m = v & UInt32((1 << (8 * size)) - 1) if size < 4 else v
            _ = self.tile2_access(True, addr, size, m)
            return
        if self.is_mmio(addr):
            if size != 4 or (addr & 3) != 0:
                raise Error("non-word MMIO write at " + hex(Int(addr)))
            if addr >= TILE_BASE and addr < TILE_BASE + TILE_SPAN:
                self.tile.write(addr - TILE_BASE, v)
            elif addr == UART_BASE:
                var c = Int(v & 0xFF)
                if c == 10:
                    print(self.uart_line)
                    self.uart_line = String()
                else:
                    self.uart_line += chr(c)
            elif addr == SIMCTRL_BASE:
                self.halted = True
                self.exit_code = Int(v >> 1)
            elif addr == SIMCTRL_BASE + 4:
                self.cycles_latched = self.instret
            return
        for i in range(size):
            self.poke8(addr + UInt32(i), UInt8((v >> UInt32(8 * i)) & 0xFF))

    # ---- ELF loader ---------------------------------------------------------
    def load_elf(mut self, path: String) raises:
        var data: List[UInt8]
        with open(path, "r") as f:
            data = f.read_bytes()

        def u32(d: List[UInt8], o: Int) -> UInt32:
            return (
                UInt32(d[o]) | (UInt32(d[o + 1]) << 8)
                | (UInt32(d[o + 2]) << 16) | (UInt32(d[o + 3]) << 24)
            )

        def u16(d: List[UInt8], o: Int) -> Int:
            return Int(d[o]) | (Int(d[o + 1]) << 8)

        if len(data) < 52 or data[0] != 0x7F or data[4] != 1 or u16(data, 18) != 243:
            raise Error("not an ELF32 RISC-V file: " + path)
        self.pc = u32(data, 24)
        var phoff = Int(u32(data, 28))
        var phentsize = u16(data, 42)
        var phnum = u16(data, 44)
        for i in range(phnum):
            var ph = phoff + i * phentsize
            if u32(data, ph) != 1:  # PT_LOAD
                continue
            var off = Int(u32(data, ph + 4))
            var paddr = u32(data, ph + 12)
            var filesz = Int(u32(data, ph + 16))
            for j in range(filesz):
                self.poke8(paddr + UInt32(j), data[off + j])

    # ---- CSRs ---------------------------------------------------------------
    def csr_read(self, csr: UInt32) -> UInt32:
        if csr == 0xB00 or csr == 0xC00 or csr == 0xB02 or csr == 0xC02:
            return UInt32(self.instret & 0xFFFFFFFF)
        if csr == 0xB80 or csr == 0xC80 or csr == 0xB82 or csr == 0xC82:
            return UInt32(self.instret >> 32)
        return 0  # mhartid, misa, ... read as zero

    # ---- execute ------------------------------------------------------------
    def step(mut self) raises:
        var inst = self.load(self.pc, 4)
        var op = inst & 0x7F
        var rd = Int((inst >> 7) & 31)
        var f3 = (inst >> 12) & 7
        var rs1 = Int((inst >> 15) & 31)
        var rs2 = Int((inst >> 20) & 31)
        var f7 = inst >> 25
        var a = self.x[rs1]
        var b = self.x[rs2]
        var next_pc = self.pc + 4
        var imm_i = sext(inst >> 20, 12)

        if self.trace:
            print(hex(Int(self.pc)), hex(Int(inst)))

        if op == 0x37:  # LUI
            self.x[rd] = inst & 0xFFFFF000
        elif op == 0x17:  # AUIPC
            self.x[rd] = self.pc + (inst & 0xFFFFF000)
        elif op == 0x6F:  # JAL
            var imm = (
                ((inst >> 31) << 20) | (((inst >> 12) & 0xFF) << 12)
                | (((inst >> 20) & 1) << 11) | (((inst >> 21) & 0x3FF) << 1)
            )
            self.x[rd] = next_pc
            next_pc = self.pc + sext(imm, 21)
        elif op == 0x67:  # JALR
            var t = (a + imm_i) & ~UInt32(1)
            self.x[rd] = next_pc
            next_pc = t
        elif op == 0x63:  # BRANCH
            var imm = (
                ((inst >> 31) << 12) | (((inst >> 7) & 1) << 11)
                | (((inst >> 25) & 0x3F) << 5) | (((inst >> 8) & 0xF) << 1)
            )
            var take: Bool
            if f3 == 0:
                take = a == b
            elif f3 == 1:
                take = a != b
            elif f3 == 4:
                take = Int32(a) < Int32(b)
            elif f3 == 5:
                take = Int32(a) >= Int32(b)
            elif f3 == 6:
                take = a < b
            elif f3 == 7:
                take = a >= b
            else:
                raise Error("bad branch")
            if take:
                next_pc = self.pc + sext(imm, 13)
        elif op == 0x03:  # LOAD
            var addr = a + imm_i
            var v: UInt32
            if f3 == 0:
                v = sext(self.load(addr, 1), 8)
            elif f3 == 1:
                v = sext(self.load(addr, 2), 16)
            elif f3 == 2:
                v = self.load(addr, 4)
            elif f3 == 4:
                v = self.load(addr, 1)
            elif f3 == 5:
                v = self.load(addr, 2)
            else:
                raise Error("bad load")
            self.x[rd] = v
        elif op == 0x23:  # STORE
            var imm = sext(((inst >> 25) << 5) | ((inst >> 7) & 31), 12)
            var size = 1 << Int(f3)
            self.store(a + imm, size, b)
        elif op == 0x13:  # OP-IMM
            var sh = (inst >> 20) & 31
            var r: UInt32
            if f3 == 0:
                r = a + imm_i
            elif f3 == 2:
                r = UInt32(1) if Int32(a) < Int32(imm_i) else 0
            elif f3 == 3:
                r = UInt32(1) if a < imm_i else 0
            elif f3 == 4:
                r = a ^ imm_i
            elif f3 == 6:
                r = a | imm_i
            elif f3 == 7:
                r = a & imm_i
            elif f3 == 1:
                r = a << sh
            elif (f7 & 0x20) != 0:
                r = UInt32(Int32(a) >> Int32(sh))
            else:
                r = a >> sh
            self.x[rd] = r
        elif op == 0x33 and f7 == 1:  # RV32M
            self.x[rd] = mul_div(f3, a, b)
        elif op == 0x33:  # OP
            var sh = b & 31
            var r: UInt32
            if f3 == 0:
                r = a - b if f7 == 0x20 else a + b
            elif f3 == 1:
                r = a << sh
            elif f3 == 2:
                r = UInt32(1) if Int32(a) < Int32(b) else 0
            elif f3 == 3:
                r = UInt32(1) if a < b else 0
            elif f3 == 4:
                r = a ^ b
            elif f3 == 5:
                r = UInt32(Int32(a) >> Int32(sh)) if f7 == 0x20 else a >> sh
            elif f3 == 6:
                r = a | b
            else:
                r = a & b
            self.x[rd] = r
        elif op == 0x0F:  # FENCE
            pass
        elif op == 0x73:  # SYSTEM
            if f3 == 0:  # ecall / ebreak / mret / wfi -> stop
                self.halted = True
                self.exit_code = Int(self.x[10])
            else:
                var csr = inst >> 20
                self.x[rd] = self.csr_read(csr)  # writes ignored
        else:
            raise Error("illegal instruction " + hex(Int(inst)) + " at " + hex(Int(self.pc)))
        self.x[0] = 0
        self.pc = next_pc
        self.instret += 1


def mul_div(f3: UInt32, a: UInt32, b: UInt32) -> UInt32:
    var sa = Int64(Int32(a))
    var sb = Int64(Int32(b))
    if f3 == 0:
        return a * b
    if f3 == 1:
        return UInt32(UInt64(sa * sb) >> 32)
    if f3 == 2:
        return UInt32(UInt64(sa * Int64(b)) >> 32)
    if f3 == 3:
        return UInt32((UInt64(a) * UInt64(b)) >> 32)
    if f3 == 5:
        return 0xFFFFFFFF if b == 0 else a // b
    if f3 == 7:
        return a if b == 0 else a % b
    # signed div/rem with truncation (Mojo `//` on signed ints floors).
    if b == 0:
        return 0xFFFFFFFF if f3 == 4 else a
    if a == 0x80000000 and b == 0xFFFFFFFF:
        return a if f3 == 4 else 0
    var ua = UInt32(0) - a if Int32(a) < 0 else a
    var ub = UInt32(0) - b if Int32(b) < 0 else b
    if f3 == 4:
        var q = ua // ub
        return UInt32(0) - q if (Int32(a) < 0) != (Int32(b) < 0) else q
    var rr = ua % ub
    return UInt32(0) - rr if Int32(a) < 0 else rr


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: rv32sim <fw.elf> [--trace] [--max-insns N]")
        exit(2)
    var m = Machine()
    var max_insns: UInt64 = 200_000_000
    var i = 2
    while i < len(args):
        if args[i] == "--trace":
            m.trace = True
        elif args[i] == "--tile-fault":
            m.tile.fault = True
        elif args[i] == "--max-insns":
            max_insns = UInt64(Int(args[i + 1]))
            i += 1
        i += 1
    m.load_elf(String(args[1]))
    while not m.halted and m.instret < max_insns:
        m.step()
    if m.uart_line.byte_length() > 0:
        print(m.uart_line)
    print("[rv32sim] halted=", m.halted, " exit=", m.exit_code,
          " instret=", m.instret, " tile_ops=", m.tile.ops, " tile2_runs=", m.tile2.runs, sep="")
    exit(m.exit_code if m.halted else 3)
