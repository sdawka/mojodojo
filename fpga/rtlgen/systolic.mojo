"""Verilog generator for the systolic tile of ../DESIGN.md.

`systolic_verilog[N, in_dtype, acc_dtype]` is a comptime value: the netlist is
built and rendered by the compiler's comptime interpreter while `rtlgen` is
being compiled. The structural checks in `check_structure` are `comptime
assert`s against the register budget of DESIGN.md §3, so a generator bug that
adds or drops a register fails the build, not the simulation.
"""

from std.sys import bit_width_of

from rtlgen.netlist import DIR_IN, DIR_OUT, Module, Port, Reg, Wire, emit_verilog

comptime K_LEN_BITS = 16
comptime GENERATOR_VERSION = "rtlgen v1"


def dtype_tag(dtype: DType) -> String:
    """`int8` -> `i8`, `int32` -> `a32`-style suffix digits."""
    if dtype == DType.int8:
        return "8"
    if dtype == DType.int16:
        return "16"
    if dtype == DType.int32:
        return "32"
    if dtype == DType.int64:
        return "64"
    return "x"


def module_name[N: Int, in_dtype: DType, acc_dtype: DType]() -> String:
    return String(
        "systolic_n", N, "_i", dtype_tag(in_dtype), "_a", dtype_tag(acc_dtype)
    )


def sv(width: Int, value: Int) -> String:
    """Sized decimal literal."""
    return String(width, "'d", value)


def bits(bus: String, lo: Int, width: Int) -> String:
    return String(bus, "[", lo + width - 1, ":", lo, "]")


def control_block[N: Int]() -> String:
    """FSM, counter and k_len latch (DESIGN.md §4 table), with Moore outputs."""
    comptime K = K_LEN_BITS
    comptime FLUSH_CYCLES = 2 * (N - 1)
    var c = String()
    c += "    // ---- control: fsm / cnt / k_len_q (DESIGN.md section 4) ----\n"
    c += "    reg [1:0] fsm /*verilator public_flat_rd*/;\n"
    c += String("    reg [", K - 1, ":0] cnt /*verilator public_flat_rd*/;\n")
    c += String("    reg [", K - 1, ":0] k_len_q;\n\n")
    c += "    wire st_idle  = (fsm == ST_IDLE);\n"
    c += "    wire st_feed  = (fsm == ST_FEED);\n"
    c += "    wire st_flush = (fsm == ST_FLUSH);\n"
    c += "    wire st_drain = (fsm == ST_DRAIN);\n"
    c += String("    wire start_ok = st_idle && start && (k_len != ", sv(K, 0), ");\n")
    c += "    wire dp_clear = rst || start_ok;\n"
    c += "    wire adv      = (st_feed && in_valid) || st_flush;\n"
    c += "    wire shift    = st_drain && out_ready;\n\n"
    c += "    assign idle      = st_idle;\n"
    c += "    assign in_ready  = st_feed;\n"
    c += "    assign out_valid = st_drain;\n"
    c += String("    assign out_last  = st_drain && (cnt == ", sv(K, N - 1), ");\n\n")
    c += "    always @(posedge clk) begin\n"
    c += "        if (rst) begin\n"
    c += "            fsm <= ST_IDLE;\n"
    c += String("            cnt <= ", sv(K, 0), ";\n")
    c += String("            k_len_q <= ", sv(K, 0), ";\n")
    c += "        end else begin\n"
    c += "            case (fsm)\n"
    c += "                ST_IDLE: if (start_ok) begin\n"
    c += "                    fsm <= ST_FEED;\n"
    c += String("                    cnt <= ", sv(K, 0), ";\n")
    c += "                    k_len_q <= k_len;\n"
    c += "                end\n"
    c += "                ST_FEED: if (in_valid) begin\n"
    c += String(
        "                    if (cnt == k_len_q - ", sv(K, 1), ") begin\n"
    )
    c += String("                        cnt <= ", sv(K, 0), ";\n")
    if N > 1:
        c += "                        fsm <= ST_FLUSH;\n"
    else:
        c += "                        fsm <= ST_DRAIN;  // N == 1: no flush\n"
    c += "                    end else begin\n"
    c += String("                        cnt <= cnt + ", sv(K, 1), ";\n")
    c += "                    end\n"
    c += "                end\n"
    if FLUSH_CYCLES > 0:
        c += "                ST_FLUSH: begin\n"
        c += String(
            "                    if (cnt == ", sv(K, FLUSH_CYCLES - 1), ") begin\n"
        )
        c += String("                        cnt <= ", sv(K, 0), ";\n")
        c += "                        fsm <= ST_DRAIN;\n"
        c += "                    end else begin\n"
        c += String("                        cnt <= cnt + ", sv(K, 1), ";\n")
        c += "                    end\n"
        c += "                end\n"
    c += "                ST_DRAIN: if (out_ready) begin\n"
    c += String("                    if (cnt == ", sv(K, N - 1), ") begin\n")
    c += String("                        cnt <= ", sv(K, 0), ";\n")
    c += "                        fsm <= ST_IDLE;\n"
    c += "                    end else begin\n"
    c += String("                        cnt <= cnt + ", sv(K, 1), ";\n")
    c += "                    end\n"
    c += "                end\n"
    c += "                default: ;\n"
    c += "            endcase\n"
    c += "        end\n"
    c += "    end\n"
    return c^


def build_systolic[N: Int, in_dtype: DType, acc_dtype: DType]() -> Module:
    """Build the netlist of an N x N tile. Indices follow DESIGN.md §3-4."""
    comptime IN = bit_width_of[in_dtype]()
    comptime ACC = bit_width_of[acc_dtype]()
    comptime PROD = 2 * IN  # full signed product width
    var m = Module(module_name[N, in_dtype, acc_dtype]())
    m.header.append(
        String(
            "Generated by fpga/rtlgen (", GENERATOR_VERSION,
            ", Mojo comptime) -- DO NOT EDIT.",
        )
    )
    m.header.append(
        String(
            "Params: N=", N, " IN_BITS=", IN, " ACC_BITS=", ACC,
            " K_LEN_BITS=", K_LEN_BITS,
        )
    )
    m.header.append(
        "Spec: fpga/DESIGN.md (output-stationary int systolic tile v1)."
    )
    m.header.append(
        "Regenerate: cd fpga && pixi run rtlgen. Verify: pixi run cosim."
    )

    m.ports.append(Port("clk", DIR_IN, 1))
    m.ports.append(Port("rst", DIR_IN, 1))
    m.ports.append(Port("start", DIR_IN, 1))
    m.ports.append(Port("k_len", DIR_IN, K_LEN_BITS))
    m.ports.append(Port("in_valid", DIR_IN, 1))
    m.ports.append(Port("a_col", DIR_IN, N * IN))
    m.ports.append(Port("b_row", DIR_IN, N * IN))
    m.ports.append(Port("out_ready", DIR_IN, 1))
    m.ports.append(Port("idle", DIR_OUT, 1))
    m.ports.append(Port("in_ready", DIR_OUT, 1))
    m.ports.append(Port("out_valid", DIR_OUT, 1))
    m.ports.append(Port("out_last", DIR_OUT, 1))
    m.ports.append(Port("out_row", DIR_OUT, N * ACC))

    m.params.append("localparam [1:0] ST_IDLE  = 2'd0;")
    m.params.append("localparam [1:0] ST_FEED  = 2'd1;")
    m.params.append("localparam [1:0] ST_FLUSH = 2'd2;")
    m.params.append("localparam [1:0] ST_DRAIN = 2'd3;")

    m.control = control_block[N]()
    m.clear = "dp_clear"
    m.adv = "adv"
    m.shift = "shift"

    # Edge inputs: the port column/row in FEED, zeros in FLUSH (§4).
    for i in range(N):
        m.wires.append(
            Wire(
                String("ea_", i), IN, True,
                String("st_feed ? ", bits("a_col", i * IN, IN), " : ", sv(IN, 0)),
                "edge",
            )
        )
    for j in range(N):
        m.wires.append(
            Wire(
                String("eb_", j), IN, True,
                String("st_feed ? ", bits("b_row", j * IN, IN), " : ", sv(IN, 0)),
                "edge",
            )
        )

    # Skew delay lines: row i of A (column j of B) has i (j) stages.
    for i in range(1, N):
        for d in range(i):
            var a_src = String("ea_", i) if d == 0 else String(
                "a_skew_", i, "_", d - 1
            )
            var b_src = String("eb_", i) if d == 0 else String(
                "b_skew_", i, "_", d - 1
            )
            m.regs.append(Reg(String("a_skew_", i, "_", d), IN, True, a_src, "", "skew"))
            m.regs.append(Reg(String("b_skew_", i, "_", d), IN, True, b_src, "", "skew"))

    # PE grid.
    for i in range(N):
        for j in range(N):
            var a_in: String
            if j > 0:
                a_in = String("a_reg_", i, "_", j - 1)
            elif i == 0:
                a_in = "ea_0"
            else:
                a_in = String("a_skew_", i, "_", i - 1)
            var b_in: String
            if i > 0:
                b_in = String("b_reg_", i - 1, "_", j)
            elif j == 0:
                b_in = "eb_0"
            else:
                b_in = String("b_skew_", j, "_", j - 1)
            var prod = String("prod_", i, "_", j)
            # Both operands are declared signed, so this is a signed multiply.
            m.wires.append(
                Wire(prod.copy(), PROD, True, String(a_in, " * ", b_in), "mul")
            )
            # sext(prod) to ACC bits (or truncate if ACC < 2*IN): mod 2^ACC.
            var prod_acc: String
            if ACC > PROD:
                prod_acc = String(
                    "{{", ACC - PROD, "{", prod, "[", PROD - 1, "]}}, ", prod, "}"
                )
            elif ACC == PROD:
                prod_acc = prod.copy()
            else:
                prod_acc = bits(prod, 0, ACC)
            var acc = String("acc_", i, "_", j)
            var below = String("acc_", i + 1, "_", j) if i + 1 < N else sv(ACC, 0)
            m.regs.append(Reg(String("a_reg_", i, "_", j), IN, True, a_in, "", "pe_a"))
            m.regs.append(Reg(String("b_reg_", i, "_", j), IN, True, b_in, "", "pe_b"))
            m.regs.append(
                Reg(acc, ACC, True, String(acc, " + ", prod_acc), below, "acc")
            )

    # out_row is wired to accumulator row 0 in every cycle (no mux).
    var out = String("{")
    for j in range(N - 1, -1, -1):
        out += String("acc_0_", j)
        if j > 0:
            out += ", "
    out += "}"
    m.wires.append(Wire("out_row_w", N * ACC, False, out, "out"))
    m.control += "    assign out_row = out_row_w;\n"
    return m^


def check_structure[N: Int, in_dtype: DType, acc_dtype: DType]():
    """Comptime checks of the generated netlist against DESIGN.md §3."""
    comptime IN = bit_width_of[in_dtype]()
    comptime ACC = bit_width_of[acc_dtype]()
    comptime m = build_systolic[N, in_dtype, acc_dtype]()
    comptime assert N >= 1, "tile size must be >= 1"
    comptime assert in_dtype.is_integral() and acc_dtype.is_integral(), (
        "integer datapath only"
    )
    comptime assert m.count_regs("skew") == N * (N - 1), (
        "skew lines must have N(N-1)/2 stages each for A and B"
    )
    comptime assert m.reg_bits("skew") == IN * N * (N - 1)
    comptime assert m.reg_bits("pe_a") == IN * N * N
    comptime assert m.reg_bits("pe_b") == IN * N * N
    comptime assert m.reg_bits("acc") == ACC * N * N
    comptime assert m.count_wires("mul") == N * N, "one multiplier per PE"


def systolic_verilog[N: Int, in_dtype: DType, acc_dtype: DType]() -> String:
    return emit_verilog(build_systolic[N, in_dtype, acc_dtype]())


def datapath_reg_bits[N: Int, in_dtype: DType, acc_dtype: DType]() -> Int:
    var m = build_systolic[N, in_dtype, acc_dtype]()
    return m.reg_bits("skew") + m.reg_bits("pe_a") + m.reg_bits("pe_b") + m.reg_bits(
        "acc"
    )
