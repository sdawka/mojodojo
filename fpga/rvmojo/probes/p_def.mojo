from std.sys import get_defined_int

@export
def uart_base() abi("C") -> UInt32:
    return UInt32(get_defined_int["UART_BASE", 0xF0001000]())
