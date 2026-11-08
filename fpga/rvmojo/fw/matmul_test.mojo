"""Self-checking firmware: drive the tile and compare to a Mojo CPU reference."""

from rt import Arena, puts, put_dec, put_hex, putc, cycles
from tile_v1 import (
    tile_present, tile_matmul_i8, cpu_matmul_i8, I8Ptr, I32Ptr, REG_ID,
)
from rt import mmio_read32


def fill(p: MutPointer[Int8, MutUntrackedOrigin], n: Int, mut s: UInt32):
    for i in range(n):
        s = s * 1664525 + 1013904223
        p[unsafe_offset=i] = Int8(Int32(s >> 24) - 128)  # full int8 range


def run_case(mut arena: Arena, M: Int, K: Int, N: Int, seed: UInt32) -> Int:
    var a = arena.alloc[Int8](M * K)
    var b = arena.alloc[Int8](K * N)
    var c_tile = arena.alloc[Int32](M * N)
    var c_ref = arena.alloc[Int32](M * N)
    var s = seed
    fill(a, M * K, s)
    fill(b, K * N, s)
    for i in range(M * N):
        c_tile[unsafe_offset=i] = 0x5A5A5A5A  # poison
    var t0 = cycles()
    var tile_cyc = tile_matmul_i8(I8Ptr(a), I8Ptr(b), c_tile, M, K, N)
    var t1 = cycles()
    cpu_matmul_i8(I8Ptr(a), I8Ptr(b), c_ref, M, K, N)
    var t2 = cycles()
    var bad = 0
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
    puts("matmul ")
    put_dec(Int32(M))
    putc(UInt8(ord("x")))
    put_dec(Int32(K))
    putc(UInt8(ord("x")))
    put_dec(Int32(N))
    puts(StaticString(": PASS") if bad == 0 else StaticString(": FAIL"))
    puts("  soc cycles: tile-driver=")
    put_dec(Int32(t1 - t0))
    puts(" cpu-ref=")
    put_dec(Int32(t2 - t1))
    puts(" tile-internal=")
    put_dec(Int32(tile_cyc))
    putc(10)
    return bad


@export
def fw_main() abi("C") -> Int32:
    puts("Mojo rv32 firmware: int8 matmul tile self-check\n")
    puts("tile ID = ")
    put_hex(mmio_read32(REG_ID))
    putc(10)
    if not tile_present():
        puts("tile not found\n")
        return 2
    var arena = Arena()
    var bad = 0
    bad += run_case(arena, 4, 1, 4, 7)
    bad += run_case(arena, 4, 16, 4, 11)
    bad += run_case(arena, 6, 9, 10, 3)  # ragged M and N
    bad += run_case(arena, 16, 32, 16, 42)
    puts(StaticString("ALL PASS\n") if bad == 0 else StaticString("FAILURES\n"))
    return 0 if bad == 0 else 1
