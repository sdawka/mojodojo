from rt import puts, put_dec, put_hex, cycles


@export
def fw_main() abi("C") -> Int32:
    puts("hello from Mojo on rv32im\n")
    var s: Int32 = 0
    for i in range(10):
        s += Int32(i) * Int32(i)
    puts("sum of squares 0..9 = ")
    put_dec(s)
    puts("\ncycles = ")
    put_hex(cycles())
    puts("\n")
    return 0 if s == 285 else 1
