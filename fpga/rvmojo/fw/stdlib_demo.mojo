"""Uses print / String / List / raise on bare metal, via fw/shim.mojo."""
from std.sys import size_of


def checked_div(a: UInt32, b: UInt32) raises -> UInt32:
    if b == 0:
        raise Error("division by zero")
    return a // b


@export
def fw_main() abi("C") -> Int32:
    print("print() from Mojo on rv32im, Int is", 8 * size_of[Int](), "bits")
    var squares = List[Int32]()
    for i in range(12):
        squares.append(Int32(i * i))
    var total: Int32 = 0
    for v in squares:
        total += v
    var s = String("squares: ")
    s += String(len(squares))
    s += " sum="
    s += String(total)
    print(s)
    try:
        print("10/2 =", checked_div(10, 2))
        print("10/0 =", checked_div(10, 0))
    except e:
        print("caught:", e)
    return 0 if total == 506 else 1
