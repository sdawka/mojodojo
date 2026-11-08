@export
def f32_madd(a: Float32, b: Float32, c: Float32) abi("C") -> Float32:
    return a * b + c

@export
def u64_div(a: UInt64, b: UInt64) abi("C") -> UInt64:
    return a // b

@export
def i8_to_f32(a: Int8) abi("C") -> Float32:
    return Float32(a) * 0.5
