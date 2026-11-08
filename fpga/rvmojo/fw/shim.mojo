"""Minimal libc/Mojo-runtime shim so `print`, `String`, `List`, `raise` link.

Single-core, interrupts off, so "atomics" are plain read-modify-write. The heap
is a never-freeing bump allocator after .bss (shared with rt.Arena's start,
so don't mix both in one program). stdout/stderr go to the UART.
Built as its own object (exports are only emitted for the root module).
"""

from rt import heap_base, putc
from std.sys._assembly import inlined_assembly

comptime BytePtr = MutPointer[UInt8, MutUntrackedOrigin]
comptime U32Ptr = MutPointer[UInt32, MutUntrackedOrigin]
comptime U64Ptr = MutPointer[UInt64, MutUntrackedOrigin]


def _heap_top_cell() -> U32Ptr:
    # One word of state at the start of the heap holds the bump pointer.
    return U32Ptr(unsafe_from_address=heap_base())


@export
def KGEN_CompilerRT_AlignedAlloc(align: UInt32, size: UInt32) abi("C") -> BytePtr:
    var cell = _heap_top_cell()
    var top = cell[]
    if top == 0:
        top = UInt32(heap_base() + 16)
    var a = align if align > 4 else 4
    top = (top + a - 1) & ~(a - 1)
    cell[] = top + size
    return BytePtr(unsafe_from_address=Int(top))


@export
def KGEN_CompilerRT_AlignedFree(p: BytePtr) abi("C"):
    pass  # bump allocator: never frees


@export
def KGEN_CompilerRT_GetStackTrace(buf: BytePtr, n: Int32) abi("C") -> Int32:
    return 0


@export
def write(fd: Int32, buf: BytePtr, n: UInt32) abi("C") -> Int32:
    for i in range(Int(n)):
        putc(buf[unsafe_offset=i])
    return Int32(n)


@export
def dup(fd: Int32) abi("C") -> Int32:
    return fd


@export
def fdopen(fd: Int32, mode: BytePtr) abi("C") -> BytePtr:
    return BytePtr(unsafe_from_address=Int(fd) + 0x100)  # non-null token


@export
def fclose(f: BytePtr) abi("C") -> Int32:
    return 0


@export
def fflush(f: BytePtr) abi("C") -> Int32:
    return 0


@export
def KGEN_CompilerRT_fprintf(f: BytePtr, fmt: BytePtr) abi("C") -> Int32:
    """Variadic in C; on ilp32 the fixed args arrive in a0/a1 regardless.
    Prints the format string verbatim (used for abort/bounds messages)."""
    var i = 0
    while fmt[unsafe_offset=i] != 0:
        putc(fmt[unsafe_offset=i])
        i += 1
    return Int32(i)


@export
def __atomic_fetch_add_4(p: U32Ptr, v: UInt32, order: Int32) abi("C") -> UInt32:
    var old = p[]
    p[] = old + v
    return old


@export
def __atomic_fetch_add_8(p: U64Ptr, v: UInt64, order: Int32) abi("C") -> UInt64:
    var old = p[]
    p[] = old + v
    return old


@export
def __atomic_load_4(p: U32Ptr, order: Int32) abi("C") -> UInt32:
    return p.unsafe_load[volatile=True]()


@export
def __atomic_load_8(p: U64Ptr, order: Int32) abi("C") -> UInt64:
    return p.unsafe_load[volatile=True]()


@export
def __atomic_store_4(p: U32Ptr, v: UInt32, order: Int32) abi("C"):
    p.unsafe_store[volatile=True](v)


@export
def __atomic_compare_exchange_4(
    p: U32Ptr, expected: U32Ptr, desired: UInt32, so: Int32, fo: Int32
) abi("C") -> Bool:
    var cur = p[]
    if cur == expected[]:
        p[] = desired
        return True
    expected[] = cur
    return False
