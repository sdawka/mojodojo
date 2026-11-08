"""Bare-metal RV32 runtime for Mojo firmware: crt0, mem*, MMIO, console.

Everything here is Mojo; there is no C or separate assembler step. Board
addresses come from `-D NAME=<decimal>` defines so the same source targets the
functional simulator (fpga/rvmojo/sim) and the LiteX/VexRiscv SoC (fpga/soc).
"""

from std.sys import get_defined_int, size_of
from std.bit import count_leading_zeros
from std.sys._assembly import inlined_assembly

# --- board configuration (override with -D, decimal values) -----------------
# Defaults are the LiteX sim SoC in fpga/soc (build/sim/csr.csv); rv32sim
# models the same map, so one firmware image runs on both.
comptime CSR_UART = get_defined_int["UART_BASE", 0xF0002000]()
# LiteX UART CSRs (csr_data_width=32): rxtx @+0, txfull @+4.
comptime UART_RXTX = CSR_UART + 0
comptime UART_TXFULL = CSR_UART + 4
comptime UART_TXEMPTY = CSR_UART + 0x18
# LiteX SimCtrl block (fpga/soc/soc.py): finish @+0 (any write ends the
# simulation), latch @+4 (write: snapshot the 64-bit cycle counter),
# cycles @+8 (hi word) / +12 (lo word), csr_ordering_big.
comptime CSR_SIMCTRL = get_defined_int["SIMCTRL_BASE", 0xF0001000]()
comptime SIMCTRL_FINISH = CSR_SIMCTRL + 0
comptime SIMCTRL_LATCH = CSR_SIMCTRL + 4
comptime SIMCTRL_CYCLES_LO = CSR_SIMCTRL + 12


# --- crt0 -------------------------------------------------------------------
# The whole startup lives in a .text.init section emitted from inline asm.
# The linker script places .text.init at the reset vector and provides the
# symbols. `$` is the operand sigil in inline asm, so the global pointer is not
# set up (Mojo code does not need gp; linker relaxation is off).
comptime _CRT0 = """
.pushsection .text.init,"ax",@progbits
.globl _start
.type _start,@function
_start:
    csrci mstatus, 8          # MIE=0: firmware polls, no interrupts
    lla   sp, _stack_top
    lla   t0, _data_lma
    lla   t1, _data_start
    lla   t2, _data_end
1:  bgeu  t1, t2, 2f
    lw    t3, 0(t0)
    sw    t3, 0(t1)
    addi  t0, t0, 4
    addi  t1, t1, 4
    j     1b
2:  lla   t1, _bss_start
    lla   t2, _bss_end
3:  bgeu  t1, t2, 4f
    sw    zero, 0(t1)
    addi  t1, t1, 4
    j     3b
4:  call  fw_main
    call  fw_exit
5:  j     5b
.size _start, .-_start
.popsection
"""


@export
def _crt0_anchor() abi("C"):
    """Never called; exists so the crt0 asm above is emitted into the object."""
    inlined_assembly[_CRT0, NoneType, constraints="", has_side_effect=True]()


# --- compiler support routines ----------------------------------------------
# LLVM emits calls to memset/memcpy/memmove (e.g. Array(fill=0), struct copies).
# Volatile byte accesses stop LLVM from recognising these loops as the very
# idioms they implement (which would recurse).
comptime BytePtr = MutPointer[UInt8, MutUntrackedOrigin]


@export
def memset(dst: BytePtr, c: Int32, n: UInt32) abi("C") -> BytePtr:
    var b = UInt8(c & 0xFF)
    for i in range(Int(n)):
        dst.unsafe_offset(i).unsafe_store[volatile=True](b)
    return dst


@export
def memcpy(dst: BytePtr, src: BytePtr, n: UInt32) abi("C") -> BytePtr:
    for i in range(Int(n)):
        dst.unsafe_offset(i).unsafe_store[volatile=True](
            src.unsafe_offset(i).unsafe_load[volatile=True]()
        )
    return dst


@export
def memmove(dst: BytePtr, src: BytePtr, n: UInt32) abi("C") -> BytePtr:
    if Int(dst) <= Int(src):
        return memcpy(dst, src, n)
    var i = Int(n)
    while i > 0:
        i -= 1
        dst.unsafe_offset(i).unsafe_store[volatile=True](
            src.unsafe_offset(i).unsafe_load[volatile=True]()
        )
    return dst


# --- MMIO -------------------------------------------------------------------
@always_inline
def mmio_write32(addr: Int, value: UInt32):
    MutPointer[UInt32, MutUntrackedOrigin](
        unsafe_from_address=addr
    ).unsafe_store[volatile=True](value)


@always_inline
def mmio_read32(addr: Int) -> UInt32:
    return MutPointer[UInt32, MutUntrackedOrigin](
        unsafe_from_address=addr
    ).unsafe_load[volatile=True]()


@always_inline
def fence():
    inlined_assembly["fence", NoneType, constraints="~{memory}", has_side_effect=True]()


@always_inline
def rdcycle() -> UInt32:
    """The `mcycle` CSR (needs Zicsr + counters in the core)."""
    return inlined_assembly[
        "csrr $0, mcycle", UInt32, constraints="=r", has_side_effect=True
    ]()


def cycles() -> UInt32:
    """Low 32 bits of the SoC cycle counter (SimCtrl; rv32sim: instret)."""
    mmio_write32(SIMCTRL_LATCH, 1)
    return mmio_read32(SIMCTRL_CYCLES_LO)


@always_inline
def heap_base() -> Int:
    """First free RAM byte after .bss (linker symbol `_heap_start`)."""
    return Int(
        inlined_assembly[
            "lla $0, _heap_start", UInt32, constraints="=r", has_side_effect=False
        ]()
    )


struct Arena(Movable):
    """Bump allocator over RAM after .bss; never frees (firmware lifetime)."""

    var top: Int

    def __init__(out self):
        self.top = heap_base()

    def alloc[T: AnyType](mut self, count: Int) -> MutPointer[T, MutUntrackedOrigin]:
        self.top = (self.top + 15) & ~15
        var p = MutPointer[T, MutUntrackedOrigin](unsafe_from_address=self.top)
        self.top += count * size_of[T]()
        return p


# --- console (no String / print: those pull in libc + allocator) -------------
def putc(c: UInt8):
    while mmio_read32(UART_TXFULL) != 0:
        pass
    mmio_write32(UART_RXTX, UInt32(c))


def puts(s: StaticString):
    """Write a string literal (StaticString: no allocation)."""
    var p = s.unsafe_ptr()
    for i in range(s.byte_length()):
        putc(p[unsafe_offset=i])


def put_hex(v: UInt32):
    putc(UInt8(ord("0")))
    putc(UInt8(ord("x")))
    comptime for i in range(8):
        var nib = (v >> UInt32(28 - 4 * i)) & 0xF
        putc(UInt8(nib + 48) if nib < 10 else UInt8(nib + 87))


def put_dec(v: Int32):
    var u: UInt32
    if v < 0:
        putc(UInt8(ord("-")))
        u = UInt32(0) - UInt32(v)
    else:
        u = UInt32(v)
    var buf = SIMD[DType.uint8, 16](0)
    var n = 0
    while True:
        buf[n] = UInt8(u % 10) + 48  # unsigned div: plain `divu`/`remu`
        u = u // 10
        n += 1
        if u == 0:
            break
    while n > 0:
        n -= 1
        putc(buf[n])


@export
def fw_exit(code: Int32) abi("C"):
    """Called by crt0 with fw_main's return value: report it and end the sim.

    Same protocol as fpga/soc/fw/sim.c: print `SIM_FINISH <code>` (run.py
    keys on it), drain the UART, then write SimCtrl.finish. LiteX ignores the
    written value; rv32sim takes the exit status from (code<<1)|1.
    """
    puts("\nSIM_FINISH ")
    put_dec(code)
    putc(10)
    while mmio_read32(UART_TXEMPTY) == 0:
        pass
    for i in range(2000):
        _ = mmio_read32(UART_TXEMPTY)  # volatile: a real delay
    mmio_write32(SIMCTRL_FINISH, (UInt32(code) << 1) | 1)
    while True:
        pass


# --- compiler-rt gap ----------------------------------------------------------
# LiteX's libcompiler_rt.a (linked by build.sh when present) lacks
# __floatundidf, which Mojo's signed floor-division lowering needs. Returns the
# IEEE-754 double bit pattern: under the ilp32 soft-float ABI a double is
# returned in a0/a1 exactly like a UInt64.
@export
def __floatundidf(x: UInt64) abi("C") -> UInt64:
    if x == 0:
        return 0
    var lz = count_leading_zeros(x)
    var e = UInt64(1086) - lz  # 1023 + 63 - lz
    var m = x << lz  # leading one at bit 63
    var mant = (m >> 11) & 0xFFFFFFFFFFFFF
    var rest = m & 0x7FF  # round to nearest, ties to even
    if rest > 0x400 or (rest == 0x400 and (mant & 1) == 1):
        mant += 1
        if mant == 0x10000000000000:
            mant = 0
            e += 1
    return (e << 52) | mant
