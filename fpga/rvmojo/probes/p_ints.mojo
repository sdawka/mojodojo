@export
def add_i32(a: Int32, b: Int32) abi("C") -> Int32:
    return a + b

@export
def mul_i32(a: Int32, b: Int32) abi("C") -> Int32:
    return a * b

@export
def div_i32(a: Int32, b: Int32) abi("C") -> Int32:
    return a // b

@export
def add_int(a: Int, b: Int) abi("C") -> Int:
    return a + b

@export
def mul_i64(a: Int64, b: Int64) abi("C") -> Int64:
    return a * b

@export
def sum_loop(n: Int32) abi("C") -> Int32:
    var s: Int32 = 0
    for i in range(n):
        s += i * i
    return s
