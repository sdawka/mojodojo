"""Needs compiler-rt: signed `//` with a runtime divisor (Mojo's floor-div
lowering uses an f64 reciprocal), Float32 on a no-FPU core, UInt64 `//`."""

from rt import puts, put_dec, putc, mmio_read32, CSR_UART


@export
def fw_main() abi("C") -> Int32:
    # Values from a volatile read so nothing constant-folds.
    var z = Int32(mmio_read32(CSR_UART + 4))  # txfull == 0
    var a: Int32 = -17 + z
    var b: Int32 = 5 + z
    var q = a // b  # floor: -4
    var r = a % b  # floor mod: 3
    var f = Float32(a) * 0.5 + Float32(b)  # -3.5
    var big = UInt64(0x1_0000_0000) * UInt64(b + 2) // UInt64(b)  # 6012954214
    puts("-17 // 5 = ")
    put_dec(q)
    puts(", -17 % 5 = ")
    put_dec(r)
    puts(", Int32(f*10) = ")
    put_dec(Int32(f * 10.0))
    puts(", u64 div lo = ")
    put_dec(Int32(big & 0x7FFFFFFF))
    putc(10)
    var ok = q == -4 and r == 3 and Int32(f * 10.0) == -35 and big == 6012954214
    return 0 if ok else 1
