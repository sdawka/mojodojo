"""Self-checking firmware for tile v2: tile result vs a Mojo CPU reference."""

from rt import Arena, puts, put_dec, put_hex, putc, cycles, mmio_read32
from tile_v1 import cpu_matmul_i8
from tile2 import tile2_present, tile2_matmul_i8, T2_ID, T2_INFO, I8Ptr


def fill(p: MutPointer[Int8, MutUntrackedOrigin], n: Int, mut s: UInt32):
    for i in range(n):
        s = s * 1664525 + 1013904223
        p[unsafe_offset=i] = Int8(Int32(s >> 24) - 128)


def run_case(mut arena: Arena, M: Int, K: Int, N: Int, seed: UInt32) -> Int:
    var a = arena.alloc[Int8](M * K)
    var b = arena.alloc[Int8](K * N)
    var c_tile = arena.alloc[Int32](M * N)
    var c_ref = arena.alloc[Int32](M * N)
    var s = seed
    fill(a, M * K, s)
    fill(b, K * N, s)
    for i in range(M * N):
        c_tile[unsafe_offset=i] = 0x5A5A5A5A
    var t0 = cycles()
    var busy = tile2_matmul_i8(I8Ptr(a), I8Ptr(b), c_tile, M, K, N)
    var t1 = cycles()
    cpu_matmul_i8(I8Ptr(a), I8Ptr(b), c_ref, M, K, N)
    var t2 = cycles()
    var bad = 0 if busy >= 0 else 1
    for i in range(M * N):
        if c_tile[unsafe_offset=i] != c_ref[unsafe_offset=i]:
            if bad < 4:
                puts("  mismatch at ")
                put_dec(Int32(i))
                puts(": tile=")
                put_dec(c_tile[unsafe_offset=i])
                puts(" ref=")
                put_dec(c_ref[unsafe_offset=i])
                putc(10)
            bad += 1
    puts("tile2 ")
    put_dec(Int32(M))
    putc(UInt8(ord("x")))
    put_dec(Int32(K))
    putc(UInt8(ord("x")))
    put_dec(Int32(N))
    puts(StaticString(": PASS") if bad == 0 else StaticString(": FAIL"))
    puts("  cycles: driver=")
    put_dec(Int32(t1 - t0))
    puts(" tile-busy=")
    put_dec(Int32(busy))
    puts(" cpu-ref=")
    put_dec(Int32(t2 - t1))
    putc(10)
    return bad


@export
def fw_main() abi("C") -> Int32:
    puts("Mojo rv32 firmware: tile v2 self-check\n")
    puts("ID = ")
    put_hex(mmio_read32(T2_ID))
    puts(" INFO = ")
    put_hex(mmio_read32(T2_INFO))
    putc(10)
    if not tile2_present():
        puts("tile2 not found\n")
        return 2
    var arena = Arena()
    var bad = 0
    bad += run_case(arena, 8, 16, 8, 7)
    bad += run_case(arena, 5, 7, 3, 3)  # ragged everything
    bad += run_case(arena, 16, 64, 24, 11)
    bad += run_case(arena, 8, 1024, 32, 5)  # one run, full A/B capacity
    bad += run_case(arena, 72, 600, 80, 42)  # M, N blocking + K split (ACCUM)
    puts(StaticString("ALL PASS\n") if bad == 0 else StaticString("FAILURES\n"))
    return 0 if bad == 0 else 1
