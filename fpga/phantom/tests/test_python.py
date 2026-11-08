"""Python binding test: bit-exact int8 matmul and cycle model via phantom_py.

Run from fpga/: `pixi run phantom-test` (after phantom-build). No numpy.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))
import phantom_py  # noqa: E402


def ref(a, b, m, k, n):
    c = []
    for i in range(m):
        for j in range(n):
            s = sum(a[i * k + kk] * b[kk * n + j] for kk in range(k))
            c.append((s + 2**31) % 2**32 - 2**31)
    return c


def cdiv(x, y):
    return -(-x // y)


def main():
    assert phantom_py.device_kind() == "other(phantom, accelerator=True)"
    rnd = random.Random(1234)
    passed = total = 0
    for tile in (4, 8):
        dev = phantom_py.Device(tile)
        assert dev.tile_n() == tile, dev.tile_n()
        for _ in range(5):
            m, k, n = rnd.randint(1, 20), rnd.randint(1, 40), rnd.randint(1, 20)
            a = [rnd.randint(-128, 127) for _ in range(m * k)]
            b = [rnd.randint(-128, 127) for _ in range(k * n)]
            c, cycles = dev.matmul_i8(a, b, m, k, n)
            want_cycles = cdiv(m, tile) * cdiv(n, tile) * (k + 3 * tile - 1)
            ok = c == ref(a, b, m, k, n) and cycles == want_cycles
            total += 1
            passed += ok
            print("ok  " if ok else "FAIL", f"N={tile} M={m} K={k} N_out={n} cycles={cycles}")
    # bytes are read as two's complement int8
    dev = phantom_py.Device(4)
    c, _ = dev.matmul_i8(bytes([0xFF, 2]), bytes([3, 0x80]), 1, 2, 1)
    total += 1
    passed += c == [-1 * 3 + 2 * -128]
    try:
        dev.matmul_i8([1, 2, 3], [1], 2, 2, 1)
        print("FAIL length check")
    except Exception as e:  # raised Mojo errors surface as Python exceptions
        total += 1
        passed += "expected 4 elements" in str(e)
    print(f"python: {passed}/{total} passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
