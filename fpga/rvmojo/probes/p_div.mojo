@export
def sdiv_i32(a: Int32, b: Int32) abi("C") -> Int32:
    return a // b

@export
def udiv_u32(a: UInt32, b: UInt32) abi("C") -> UInt32:
    return a // b

@export
def urem_u32(a: UInt32, b: UInt32) abi("C") -> UInt32:
    return a % b

@export
def sdiv_int(a: Int, b: Int) abi("C") -> Int:
    return a // b
