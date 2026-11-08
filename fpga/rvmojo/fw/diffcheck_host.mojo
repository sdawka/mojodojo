from checks import checksum


def main():
    print("checksum(1)    =", hex(Int(checksum(1))))
    print("checksum(1234) =", hex(Int(checksum(1234))))
