"""Comptime Verilog generator for the systolic tile (lane G1-B).

`rtlgen.systolic.systolic_verilog[N, in_dtype, acc_dtype]()` is meant to be bound
to a `comptime` name: the compiler then builds the netlist and renders the
Verilog text during compilation. See ../README.md, "RTL generation".
"""
