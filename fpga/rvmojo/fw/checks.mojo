"""Integer workloads compiled for both the host and rv32 (differential test).

`checksum()` must give the same value on the host (native Mojo) and on
rv32sim running the riscv32 build; any ISS or codegen bug shows up as a
mismatch. Uses only fixed-width ints (UInt32/Int32/Int64/Int16/Int8), stack
arrays, structs and SIMD -- no allocation or printing.
"""


@fieldwise_init
struct Lcg(TrivialRegisterPassable):
    var s: UInt32

    def next(mut self) -> UInt32:
        self.s = self.s * 1664525 + 1013904223
        return self.s


def mix(h: UInt32, v: UInt32) -> UInt32:
    var x = (h ^ v) * 0x01000193
    return x ^ (x >> 15)


def checksum(seed: UInt32) -> UInt32:
    var rng = Lcg(seed)
    var h: UInt32 = 0x811C9DC5
    # 32x32 -> 64 multiplies (mul/mulh/mulhu/mulhsu), shifts, compares.
    for _ in range(200):
        var a = rng.next()
        var b = rng.next()
        var p = Int64(Int32(a)) * Int64(Int32(b))
        h = mix(h, UInt32(p & 0xFFFFFFFF))
        h = mix(h, UInt32(p >> 32))
        var q = UInt64(a) * UInt64(b)
        h = mix(h, UInt32(q >> 32))
        h = mix(h, a >> (b & 31))
        h = mix(h, UInt32(Int32(a) >> Int32(b & 31)))
        h = mix(h, UInt32(1) if Int32(a) < Int32(b) else 0)
        var d = b | 1
        h = mix(h, a // d)  # unsigned: divu
        h = mix(h, a % d)  # remu
        # signed floor div/mod with a runtime divisor: on rv32 Mojo lowers
        # this via an f64 reciprocal (needs compiler-rt + rt.__floatundidf)
        var sa = Int32(a)
        var sd = Int32(b >> 20) - 2048
        if sd == 0:
            sd = 7
        h = mix(h, UInt32(sa // sd))
        h = mix(h, UInt32(sa % sd))
    # int8 matmul, the tile's job, on stack arrays (exercises lb/sb, memset).
    var A = Array[Int8, 64](fill=0)
    var B = Array[Int8, 64](fill=0)
    for i in range(64):
        A[i] = Int8(Int32(rng.next() >> 24) - 128)
        B[i] = Int8(Int32(rng.next() >> 24) - 128)
    for i in range(8):
        for j in range(8):
            var acc: Int32 = 0
            for k in range(8):
                acc += Int32(A[i * 8 + k]) * Int32(B[k * 8 + j])
            h = mix(h, UInt32(acc))
    # 16-bit and SIMD paths
    var v = SIMD[DType.int16, 8](0)
    for i in range(8):
        v[i] = Int16(Int32(rng.next() >> 16) - 32768)
    var w = (v * v).cast[DType.int32]()
    h = mix(h, UInt32(w.reduce_add()))
    h = mix(h, UInt32(Int32(v.reduce_min())))
    return h
