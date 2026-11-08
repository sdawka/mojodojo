"""Write the generated Verilog for every instantiated tile size.

pixi run rtlgen            # writes rtl/systolic_n{N}_i8_a32.v for N in CONFIGS

All Verilog text is produced at compile time: each `comptime V = ...` below is
evaluated by the compiler, and the executable only contains the finished strings
plus the file writes. The structural `comptime assert`s in `check_structure`
run during compilation too.
"""

from std.sys import argv

from rtlgen.systolic import (
    check_structure,
    datapath_reg_bits,
    module_name,
    systolic_verilog,
)


def emit[N: Int, in_dtype: DType = .int8, acc_dtype: DType = .int32](
    outdir: String
) raises:
    check_structure[N, in_dtype, acc_dtype]()
    comptime V = systolic_verilog[N, in_dtype, acc_dtype]()
    comptime NAME = module_name[N, in_dtype, acc_dtype]()
    comptime DP_BITS = datapath_reg_bits[N, in_dtype, acc_dtype]()
    var path = String(outdir, "/", NAME, ".v")
    with open(path, "w") as f:
        f.write(V)
    print(
        t"wrote {path}: {V.byte_length()} bytes, {N * N} PEs, {DP_BITS} datapath"
        t" register bits + 34 control bits"
    )


def main() raises:
    var outdir = String("rtl")
    var args = argv()
    if len(args) > 1:
        outdir = String(args[1])
    emit[1](outdir)
    emit[2](outdir)
    emit[3](outdir)
    emit[4](outdir)
    emit[8](outdir)
    emit[16](outdir)
