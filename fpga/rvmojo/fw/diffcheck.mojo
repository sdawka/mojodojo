from rt import puts, put_hex
from checks import checksum


@export
def fw_main() abi("C") -> Int32:
    puts("checksum(1)    = ")
    put_hex(checksum(1))
    puts("\nchecksum(1234) = ")
    put_hex(checksum(1234))
    puts("\n")
    return 0
