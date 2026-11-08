"""Write the generated tile v2 Verilog (pixi run tile2-gen).

Each `comptime V = tile2_verilog[...]()` is rendered by the compiler; the
binary only writes the finished strings. `check_tile2` asserts the structure
at compile time.

Configurations:
  tile2_n8: N=8, A 16 KiB, B 32 KiB, C 32 KiB   (the deliverable, INTERFACE.md)
  tile2_n4: N=4, A 4 KiB,  B 4 KiB,  C 8 KiB    (small: exercises address wrap in co-sim)
"""

from std.sys import argv

from rtlgen.tile2 import check_tile2, tile2_name, tile2_verilog


def emit[N: Int, A_BYTES: Int, B_BYTES: Int, C_BYTES: Int](outdir: String) raises:
    check_tile2[N, A_BYTES, B_BYTES, C_BYTES]()
    comptime V = tile2_verilog[N, A_BYTES, B_BYTES, C_BYTES]()
    comptime NAME = tile2_name[N]()
    var path = String(outdir, "/", NAME, ".v")
    with open(path, "w") as f:
        f.write(V)
    print(t"wrote {path}: {V.byte_length()} bytes (N={N}, A={A_BYTES} B={B_BYTES} C={C_BYTES})")


def main() raises:
    var outdir = String("tile2/rtl")
    var args = argv()
    if len(args) > 1:
        outdir = String(args[1])
    emit[8, 16384, 32768, 32768](outdir)
    emit[4, 4096, 4096, 8192](outdir)
