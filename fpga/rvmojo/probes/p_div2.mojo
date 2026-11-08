@export
def sdiv_const(a: Int32) abi("C") -> Int32:
    return a // 7

@export
def srem_var(a: Int32, b: Int32) abi("C") -> Int32:
    return a % b

@export
def int_floordiv(a: Int, b: Int) abi("C") -> Int:
    return a // b

@export
def int_div_loop(n: Int, b: Int) abi("C") -> Int:
    var s = 0
    for i in range(n):
        s += i // b
    return s
