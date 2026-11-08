"""Verilog generator for tile v2 (fpga/tile2/INTERFACE.md).

Two modules per configuration, both rendered at compile time:

- `tile2_core_n{N}`: the v1 output-stationary core (`build_systolic`, reused
  unchanged) with one extra pipeline stage: every PE registers its product
  (`prodq_i_j`) and accumulates the registered product one advancing edge
  later. The flush therefore grows from 2(N-1) to 2N-1 cycles. Port contract
  is the v1 one (DESIGN.md section 2).
- `tile2_n{N}`: Wishbone classic slave + CSRs + A/B/C block-RAM buffers +
  sequencer that tiles C (=|+=) A * B over the core.

`tile2_verilog[N, A_BYTES, B_BYTES, C_BYTES]()` returns one file containing
both modules. `check_tile2` holds the comptime structural assertions.
"""

from rtlgen.netlist import Module, Reg, emit_verilog
from rtlgen.systolic import build_systolic, sv

comptime TILE2_GENERATOR = "rtlgen tile2 v2.0"
comptime TILE2_ID = 0x54494C32  # "TIL2"
comptime WB_ADR_BITS = 15  # 128 KiB window, word addressed
comptime SLOT_BYTES = 32768  # region slot (byte address bits [16:15])
comptime K_BITS = 16


def log2i(x: Int) -> Int:
    var n = 0
    while (1 << n) < x:
        n += 1
    return n


def is_pow2(x: Int) -> Bool:
    return x > 0 and (x & (x - 1)) == 0


def core2_name[N: Int]() -> String:
    return String("tile2_core_n", N)


def tile2_name[N: Int]() -> String:
    return String("tile2_n", N)


def core2_flush[N: Int]() -> Int:
    """Flush cycles of the pipelined core: 2(N-1) wavefront + 1 product stage."""
    return 2 * (N - 1) + 1


def tile2_info[N: Int, A_BYTES: Int, B_BYTES: Int, C_BYTES: Int]() -> Int:
    return (
        N
        | (log2i(A_BYTES) << 8)
        | (log2i(B_BYTES) << 16)
        | (log2i(C_BYTES) << 24)
    )


# --------------------------------------------------------------------------
# Core: v1 netlist + registered products
# --------------------------------------------------------------------------


def control_block2[N: Int]() -> String:
    """v1 FSM (DESIGN.md section 4) with FLUSH = 2N-1 cycles."""
    comptime K = K_BITS
    comptime FL = core2_flush[N]()
    var c = String()
    c += "    // ---- control: fsm / cnt / k_len_q (DESIGN.md s4, flush 2N-1) ----\n"
    c += "    reg [1:0] fsm;\n"
    c += String("    reg [", K - 1, ":0] cnt;\n")
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
    c += String("                    if (cnt == k_len_q - ", sv(K, 1), ") begin\n")
    c += String("                        cnt <= ", sv(K, 0), ";\n")
    c += "                        fsm <= ST_FLUSH;\n"
    c += "                    end else begin\n"
    c += String("                        cnt <= cnt + ", sv(K, 1), ";\n")
    c += "                    end\n"
    c += "                end\n"
    c += "                ST_FLUSH: begin\n"
    c += String("                    if (cnt == ", sv(K, FL - 1), ") begin\n")
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
    c += "    assign out_row = out_row_w;\n"
    return c^


def build_core2[N: Int]() -> Module:
    """v1 tile netlist with a registered product stage in every PE."""
    var m = build_systolic[N, DType.int8, DType.int32]()
    m.name = core2_name[N]()
    m.header = List[String]()
    m.header.append(
        String("Generated by fpga/rtlgen (", TILE2_GENERATOR, ", Mojo comptime) -- DO NOT EDIT.")
    )
    m.header.append(
        String("tile v2 core: N=", N, " int8 x int8 -> int32, registered product, flush ", core2_flush[N](), " cycles.")
    )
    m.header.append("Spec: fpga/DESIGN.md (v1 ports) + fpga/tile2/README.md (product stage).")
    var extra = List[Reg]()
    for ref r in m.regs:
        if r.role == "acc":
            # acc_i_j + sext(prod_i_j)  ->  acc_i_j + sext(prodq_i_j)
            r.adv_expr = String(r.adv_expr.replace("prod_", "prodq_"))
    for i in range(N):
        for j in range(N):
            extra.append(
                Reg(
                    String("prodq_", i, "_", j), 16, True,
                    String("prod_", i, "_", j), "", "prodq",
                )
            )
    for r in extra:
        m.regs.append(r.copy())
    m.control = control_block2[N]()
    return m^


# --------------------------------------------------------------------------
# Top: Wishbone slave, CSRs, buffers, sequencer
# --------------------------------------------------------------------------


def zpad(total: Int, used: Int) -> String:
    """`{P{1'b0}}, ` prefix padding `used` bits to `total`, or empty."""
    if total - used <= 0:
        return String()
    return String("{", total - used, "{1'b0}}, ")


def top2[N: Int, A_BYTES: Int, B_BYTES: Int, C_BYTES: Int]() -> String:
    comptime LN = log2i(N)
    comptime GW = LN - 2  # bits selecting a 32-bit group inside an N-byte row
    comptime AW = log2i(A_BYTES // N)  # A rows (N bytes each)
    comptime BW = log2i(B_BYTES // N)
    comptime CW = log2i(C_BYTES // (4 * N))  # C rows (N words each)
    comptime CL = LN + 2  # C alignment bits
    comptime A_LG = log2i(A_BYTES)
    comptime B_LG = log2i(B_BYTES)
    comptime C_LG = log2i(C_BYTES)
    comptime INFO = tile2_info[N, A_BYTES, B_BYTES, C_BYTES]()
    comptime GROUPS = N // 4
    var v = String()
    v += String("module ", tile2_name[N](), " (\n")
    v += "    input  wire        clk,\n"
    v += "    input  wire        rst,\n"
    v += "    input  wire        wb_cyc,\n"
    v += "    input  wire        wb_stb,\n"
    v += "    input  wire        wb_we,\n"
    v += "    /* verilator lint_off UNUSEDSIGNAL */  // high offset bits alias in small buffers\n"
    v += String("    input  wire [", WB_ADR_BITS - 1, ":0] wb_adr,\n")
    v += "    /* verilator lint_on UNUSEDSIGNAL */\n"
    v += "    input  wire [31:0] wb_dat_w,\n"
    v += "    input  wire [3:0]  wb_sel,\n"
    v += "    output wire [31:0] wb_dat_r,\n"
    v += "    output wire        wb_ack,\n"
    v += "    output wire        irq\n"
    v += ");\n\n"
    v += "    localparam [1:0] S_IDLE = 2'd0, S_SETUP = 2'd1, S_RUN = 2'd2, S_TAIL = 2'd3;\n"
    v += "    localparam [1:0] R_CSR = 2'd0, R_A = 2'd1, R_B = 2'd2, R_C = 2'd3;\n\n"

    # ---- CSR state
    v += "    // ---- CSRs and sequencer state (INTERFACE.md s4) ----\n"
    v += "    reg [1:0]  st;\n"
    v += "    reg [1:0]  mode_q;            // [0] ACCUM, [1] IRQ_EN\n"
    v += "    reg [15:0] m_q, k_q, n_q;\n"
    v += String("    reg [", AW - 1, ":0] a_base_w, a_ld_w;   // in N-byte rows\n")
    v += String("    reg [", BW - 1, ":0] b_base_w, b_ld_w;\n")
    v += String("    reg [", CW - 1, ":0] c_base_w, c_ld_w;   // in N-word rows\n")
    v += "    reg        done_q, err_cfg_q, err_busy_q;\n"
    v += "    reg [31:0] cycles_q;\n"
    v += "    wire busy = (st != S_IDLE);\n"
    v += "    assign irq = done_q && mode_q[1];\n\n"

    # ---- bus decode
    v += "    // ---- Wishbone classic decode: registered ACK, 2 cycles per access ----\n"
    v += "    reg        ack_q;\n"
    v += "    wire req = wb_cyc && wb_stb && !ack_q;\n"
    v += "    wire wr  = req && wb_we;\n"
    v += String("    wire [1:0] region = wb_adr[", WB_ADR_BITS - 1, ":", WB_ADR_BITS - 2, "];\n")
    v += "    wire [3:0] csr_idx = wb_adr[3:0];\n"
    v += "    wire dims_ok = (m_q != 16'd0) && (k_q != 16'd0) && (n_q != 16'd0);\n"
    v += "    wire csr_wr = wr && !busy && (region == R_CSR);\n"
    v += "    wire start_acc = csr_wr && (csr_idx == 4'd2) && wb_dat_w[0] && dims_ok;\n"
    v += "    assign wb_ack = ack_q;\n\n"

    v += "    reg [31:0] csr_rv;\n"
    v += "    always @(*) begin\n"
    v += "        case (csr_idx)\n"
    v += String("            4'd0:  csr_rv = 32'h", hex32(TILE2_ID), ";\n")
    v += String("            4'd1:  csr_rv = 32'h", hex32(INFO), ";\n")
    v += "            4'd3:  csr_rv = {28'd0, err_busy_q, err_cfg_q, done_q, busy};\n"
    v += "            4'd4:  csr_rv = {30'd0, mode_q};\n"
    v += "            4'd5:  csr_rv = {16'd0, m_q};\n"
    v += "            4'd6:  csr_rv = {16'd0, k_q};\n"
    v += "            4'd7:  csr_rv = {16'd0, n_q};\n"
    v += String("            4'd8:  csr_rv = {", zpad(32, A_LG), "a_base_w, ", LN, "'d0};\n")
    v += String("            4'd9:  csr_rv = {", zpad(32, A_LG), "a_ld_w, ", LN, "'d0};\n")
    v += String("            4'd10: csr_rv = {", zpad(32, B_LG), "b_base_w, ", LN, "'d0};\n")
    v += String("            4'd11: csr_rv = {", zpad(32, B_LG), "b_ld_w, ", LN, "'d0};\n")
    v += String("            4'd12: csr_rv = {", zpad(32, C_LG), "c_base_w, ", CL, "'d0};\n")
    v += String("            4'd13: csr_rv = {", zpad(32, C_LG), "c_ld_w, ", CL, "'d0};\n")
    v += "            4'd14: csr_rv = cycles_q;\n"
    v += "            default: csr_rv = 32'd0;\n"
    v += "        endcase\n"
    v += "    end\n\n"

    # ---- read-data registers
    v += "    reg [31:0] csr_q;\n"
    v += "    reg [1:0]  rreg_q;\n"
    v += "    reg        rzero_q;\n"
    v += String("    reg [", LN - 1, ":0] lane_q;   // word-in-row select for A/B (low GW bits) and C\n")
    v += "    always @(posedge clk) begin\n"
    v += "        if (rst) begin\n"
    v += "            ack_q <= 1'b0; csr_q <= 32'd0; rreg_q <= 2'd0; rzero_q <= 1'b0;\n"
    v += String("            lane_q <= ", LN, "'d0;\n")
    v += "        end else begin\n"
    v += "            ack_q <= req;\n"
    v += "            if (req) begin\n"
    v += "                csr_q   <= csr_rv;\n"
    v += "                rreg_q  <= region;\n"
    v += "                rzero_q <= busy && (region != R_CSR);\n"
    v += String("                lane_q  <= wb_adr[", LN - 1, ":0];\n")
    v += "            end\n"
    v += "        end\n"
    v += "    end\n\n"

    # ---- core interface wires
    v += "    // ---- core ----\n"
    v += "    wire core_idle, core_ov, core_ol;\n"
    v += "    /* verilator lint_off UNUSEDSIGNAL */\n"
    v += "    wire core_in_ready;\n"
    v += "    /* verilator lint_on UNUSEDSIGNAL */\n"
    v += String("    wire [", 32 * N - 1, ":0] core_row;\n")
    v += String("    reg  [", 8 * N - 1, ":0] a_q, b_q;      // registered operands (padding lanes are not masked: they only reach unwritten outputs)\n")
    v += "    reg  pend;                              // a pass is fetched and waits for start\n"
    v += "    wire core_start = (st == S_RUN) && core_idle && pend;\n"
    v += String("    ", core2_name[N](), " u_core (\n")
    v += "        .clk(clk), .rst(rst), .start(core_start), .k_len(k_q),\n"
    v += "        .in_valid(1'b1), .a_col(a_q), .b_row(b_q), .out_ready(1'b1),\n"
    v += "        .idle(core_idle), .in_ready(core_in_ready), .out_valid(core_ov),\n"
    v += "        .out_last(core_ol), .out_row(core_row)\n"
    v += "    );\n\n"

    # ---- sequencer pointers
    v += "    // ---- sequencer: fetch side (A/B reads, 2 cycles ahead of the core) ----\n"
    v += "    reg        f_busy;\n"
    v += "    reg [15:0] f_k;\n"
    v += "    reg [16:0] f_n0;\n"
    v += String("    reg [", AW - 1, ":0] f_aptr, f_atile;\n")
    v += String("    reg [", BW - 1, ":0] f_bptr, f_btile;\n")
    v += "    // ---- sequencer: drain side (C writes) ----\n"
    v += "    reg [16:0] d_m0, d_n0;\n"
    v += String("    reg [", LN - 1, ":0] d_row;\n")
    v += String("    reg [", CW - 1, ":0] c_ptr, c_tile, c_rowbase;\n")
    v += String("    wire d_last = ((d_m0 + 17'd", N, ") >= {1'b0, m_q}) && ((d_n0 + 17'd", N, ") >= {1'b0, n_q});\n")
    v += "    wire trig = core_ov && core_ol && !d_last;       // last drain row of a non-final pass\n"
    v += "    wire f_issue = (st == S_SETUP) || f_busy || trig;\n"
    v += "    wire f_lastk = (f_k == k_q - 16'd1);\n"
    v += String("    wire f_wrap = (f_n0 + 17'd", N, ") >= {1'b0, n_q};\n")
    v += String("    wire d_wrap = (d_n0 + 17'd", N, ") >= {1'b0, n_q};\n")
    v += String("    wire [", CW - 1, ":0] c_tilestep = c_ld_w << ", LN, ";\n")
    v += String("    wire [", N - 1, ":0] d_lane;\n")
    for i in range(N):
        v += String("    assign d_lane[", i, "]  = (d_n0 + 17'd", i, ") < {1'b0, n_q};\n")
    v += String("    wire d_rowok = (d_m0 + {", 17 - LN, "'d0, d_row}) < {1'b0, m_q};\n")
    v += "\n"

    v += "    always @(posedge clk) begin\n"
    v += "        if (rst || start_acc) begin\n"
    v += "            f_busy <= 1'b0; f_k <= 16'd0; f_n0 <= 17'd0;\n"
    v += "            d_m0 <= 17'd0; d_n0 <= 17'd0; pend <= 1'b0;\n"
    v += String("            d_row <= ", LN, "'d0;\n")
    v += "        end else begin\n"
    v += "            if (st == S_SETUP || trig) pend <= 1'b1;\n"
    v += "            else if (core_start) pend <= 1'b0;\n"
    v += "            if (f_issue) begin\n"
    v += "                if (f_lastk) begin\n"
    v += "                    f_k <= 16'd0; f_busy <= 1'b0;\n"
    v += "                    if (f_wrap) f_n0 <= 17'd0;\n"
    v += String("                    else f_n0 <= f_n0 + 17'd", N, ";\n")
    v += "                end else begin\n"
    v += "                    f_k <= f_k + 16'd1; f_busy <= 1'b1;\n"
    v += "                end\n"
    v += "            end\n"
    v += "            if (core_ov) begin\n"
    v += String("                d_row <= d_row + ", LN, "'d1;\n")
    v += "                if (core_ol) begin\n"
    v += String("                    if (d_wrap) begin d_n0 <= 17'd0; d_m0 <= d_m0 + 17'd", N, "; end\n")
    v += String("                    else d_n0 <= d_n0 + 17'd", N, ";\n")
    v += "                end\n"
    v += "            end\n"
    v += "        end\n"
    v += "    end\n\n"

    # Address pointers: no reset needed beyond start (they are reloaded by START).
    v += "    always @(posedge clk) begin\n"
    v += "        if (start_acc) begin\n"
    v += "            f_aptr <= a_base_w; f_atile <= a_base_w;\n"
    v += "            f_bptr <= b_base_w; f_btile <= b_base_w;\n"
    v += "            c_ptr <= c_base_w; c_tile <= c_base_w; c_rowbase <= c_base_w;\n"
    v += "        end else begin\n"
    v += "            if (f_issue) begin\n"
    v += "                if (f_lastk) begin\n"
    v += "                    if (f_wrap) begin\n"
    v += String("                        f_atile <= f_atile + ", AW, "'d1; f_aptr <= f_atile + ", AW, "'d1;\n")
    v += "                        f_btile <= b_base_w; f_bptr <= b_base_w;\n"
    v += "                    end else begin\n"
    v += "                        f_aptr <= f_atile;\n"
    v += String("                        f_btile <= f_btile + ", BW, "'d1; f_bptr <= f_btile + ", BW, "'d1;\n")
    v += "                    end\n"
    v += "                end else begin\n"
    v += "                    f_aptr <= f_aptr + a_ld_w; f_bptr <= f_bptr + b_ld_w;\n"
    v += "                end\n"
    v += "            end\n"
    v += "            if (core_ov) begin\n"
    v += "                if (core_ol) begin\n"
    v += "                    if (d_wrap) begin\n"
    v += "                        c_rowbase <= c_rowbase + c_tilestep;\n"
    v += "                        c_tile <= c_rowbase + c_tilestep; c_ptr <= c_rowbase + c_tilestep;\n"
    v += "                    end else begin\n"
    v += String("                        c_tile <= c_tile + ", CW, "'d1; c_ptr <= c_tile + ", CW, "'d1;\n")
    v += "                    end\n"
    v += "                end else begin\n"
    v += "                    c_ptr <= c_ptr + c_ld_w;\n"
    v += "                end\n"
    v += "            end\n"
    v += "        end\n"
    v += "    end\n\n"

    # ---- CSR writes + top FSM
    v += "    // ---- CSR writes, status flags, top-level state ----\n"
    v += "    always @(posedge clk) begin\n"
    v += "        if (rst) begin\n"
    v += "            st <= S_IDLE; mode_q <= 2'd0; m_q <= 16'd0; k_q <= 16'd0; n_q <= 16'd0;\n"
    v += String("            a_base_w <= ", AW, "'d0; a_ld_w <= ", AW, "'d0;\n")
    v += String("            b_base_w <= ", BW, "'d0; b_ld_w <= ", BW, "'d0;\n")
    v += String("            c_base_w <= ", CW, "'d0; c_ld_w <= ", CW, "'d0;\n")
    v += "            done_q <= 1'b0; err_cfg_q <= 1'b0; err_busy_q <= 1'b0; cycles_q <= 32'd0;\n"
    v += "        end else begin\n"
    v += "            if (busy) cycles_q <= cycles_q + 32'd1;\n"
    v += "            if (req && busy && (wb_we || region != R_CSR)) err_busy_q <= 1'b1;\n"
    v += "            if (csr_wr) begin\n"
    v += "                case (csr_idx)\n"
    v += "                    4'd2: begin\n"
    v += "                        if (wb_dat_w[1]) begin done_q <= 1'b0; err_cfg_q <= 1'b0; err_busy_q <= 1'b0; end\n"
    v += "                        if (wb_dat_w[0]) begin\n"
    v += "                            if (dims_ok) begin done_q <= 1'b0; cycles_q <= 32'd0; st <= S_SETUP; end\n"
    v += "                            else err_cfg_q <= 1'b1;\n"
    v += "                        end\n"
    v += "                    end\n"
    v += "                    4'd4:  mode_q <= wb_dat_w[1:0];\n"
    v += "                    4'd5:  m_q <= wb_dat_w[15:0];\n"
    v += "                    4'd6:  k_q <= wb_dat_w[15:0];\n"
    v += "                    4'd7:  n_q <= wb_dat_w[15:0];\n"
    v += String("                    4'd8:  a_base_w <= wb_dat_w[", A_LG - 1, ":", LN, "];\n")
    v += String("                    4'd9:  a_ld_w   <= wb_dat_w[", A_LG - 1, ":", LN, "];\n")
    v += String("                    4'd10: b_base_w <= wb_dat_w[", B_LG - 1, ":", LN, "];\n")
    v += String("                    4'd11: b_ld_w   <= wb_dat_w[", B_LG - 1, ":", LN, "];\n")
    v += String("                    4'd12: c_base_w <= wb_dat_w[", C_LG - 1, ":", CL, "];\n")
    v += String("                    4'd13: c_ld_w   <= wb_dat_w[", C_LG - 1, ":", CL, "];\n")
    v += "                    default: ;\n"
    v += "                endcase\n"
    v += "            end\n"
    v += "            case (st)\n"
    v += "                S_SETUP: st <= S_RUN;\n"
    v += "                S_RUN: if (core_ov && core_ol && d_last) st <= S_TAIL;\n"
    v += "                S_TAIL: begin st <= S_IDLE; done_q <= 1'b1; end  // last C write lands\n"
    v += "                default: ;\n"
    v += "            endcase\n"
    v += "        end\n"
    v += "    end\n\n"

    # ---- A / B buffers: N byte-lane memories each, 1 write + 1 sync read port
    for which in range(2):
        var nm = String("a") if which == 0 else String("b")
        var W = AW if which == 0 else BW
        var R = String("R_A") if which == 0 else String("R_B")
        var depth = (A_BYTES if which == 0 else B_BYTES) // N
        v += String("    // ---- ", nm.upper(), " buffer: ", N, " byte lanes x ", depth, " rows ----\n")
        v += String("    wire [", W - 1, ":0] bus_", nm, "row = wb_adr[", GW + W - 1, ":", GW, "];\n")
        v += String("    wire [", W - 1, ":0] ", nm, "_raddr = busy ? f_", nm, "ptr : bus_", nm, "row;\n")
        v += String("    wire ", nm, "_bwr = wr && !busy && (region == ", R, ");\n")
        for i in range(N):
            var g = i // 4
            var b = i % 4
            var grp_ok = String()
            if GW > 0:
                grp_ok = String(" && (wb_adr[", GW - 1, ":0] == ", GW, "'d", g, ")")
            v += String("    reg [7:0] ", nm, "mem_", i, " [0:", depth - 1, "];\n")
            v += String("    reg [7:0] ", nm, "_rd_", i, ";\n")
            v += "    always @(posedge clk) begin\n"
            v += String(
                "        if (", nm, "_bwr", grp_ok, " && wb_sel[", b, "]) ",
                nm, "mem_", i, "[bus_", nm, "row] <= wb_dat_w[", 8 * b + 7, ":", 8 * b, "];\n",
            )
            v += String("        ", nm, "_rd_", i, " <= ", nm, "mem_", i, "[", nm, "_raddr];\n")
            v += "    end\n"
        v += "\n"

    # ---- operand stage: BRAM output -> register
    v += "    // ---- operand stage: issue (cycle t) -> BRAM (t+1) -> register (t+2) ----\n"
    v += "    always @(posedge clk) begin\n"
    var aq = String("{")
    var bq = String("{")
    for i in range(N - 1, -1, -1):
        aq += String("a_rd_", i)
        bq += String("b_rd_", i)
        if i > 0:
            aq += ", "
            bq += ", "
    v += String("        a_q <= ", aq, "};\n")
    v += String("        b_q <= ", bq, "};\n")
    v += "    end\n\n"

    # ---- C buffer: N word lanes, byte enables on the bus side
    comptime CDEPTH = C_BYTES // (4 * N)
    v += String("    // ---- C buffer: ", N, " word lanes x ", CDEPTH, " rows ----\n")
    v += String("    wire [", CW - 1, ":0] bus_crow = wb_adr[", LN + CW - 1, ":", LN, "];\n")
    v += String("    wire [", CW - 1, ":0] c_raddr = busy ? (core_ov ? c_ptr + c_ld_w : c_ptr) : bus_crow;\n")
    v += "    // Drain write is pipelined one cycle: row r (and its old C value for ACCUM)\n"
    v += "    // is registered in drain cycle r and written in the next cycle.\n"
    v += String("    reg  [", CW - 1, ":0] cw_addr_q;\n")
    v += String("    reg  [", N - 1, ":0] cw_we_q;\n")
    v += String("    reg  [", 32 * N - 1, ":0] row_q;\n")
    v += String("    wire [", CW - 1, ":0] c_waddr = busy ? cw_addr_q : bus_crow;\n")
    v += "    wire c_bwr = wr && !busy && (region == R_C);\n"
    v += "    wire [3:0] c_be = busy ? 4'hf : wb_sel;\n"
    v += "    wire d_wr = core_ov && d_rowok;\n"
    v += "    always @(posedge clk) begin\n"
    v += String("        if (rst) cw_we_q <= ", N, "'d0;\n")
    v += String("        else cw_we_q <= d_wr ? d_lane : ", N, "'d0;\n")
    v += "        cw_addr_q <= c_ptr;\n"
    v += "        row_q <= core_row;\n"
    v += "    end\n"
    for j in range(N):
        v += String("    reg [31:0] cmem_", j, " [0:", CDEPTH - 1, "];\n")
        v += String("    reg [31:0] c_rd_", j, ", crd_q_", j, ";\n")
        v += String("    always @(posedge clk) crd_q_", j, " <= mode_q[0] ? c_rd_", j, " : 32'd0;\n")
        v += String(
            "    wire [31:0] c_sum_", j, " = row_q[", 32 * j + 31, ":", 32 * j,
            "] + crd_q_", j, ";\n",
        )
        v += String("    wire [31:0] c_wd_", j, " = busy ? c_sum_", j, " : wb_dat_w;\n")
        v += String(
            "    wire c_we_", j, " = busy ? cw_we_q[", j,
            "] : (c_bwr && (wb_adr[", LN - 1, ":0] == ", LN, "'d", j, "));\n",
        )
        v += "    always @(posedge clk) begin\n"
        v += String("        if (c_we_", j, ") begin\n")
        for b in range(4):
            v += String(
                "            if (c_be[", b, "]) cmem_", j, "[c_waddr][", 8 * b + 7, ":", 8 * b,
                "] <= c_wd_", j, "[", 8 * b + 7, ":", 8 * b, "];\n",
            )
        v += "        end\n"
        v += String("        c_rd_", j, " <= cmem_", j, "[c_raddr];\n")
        v += "    end\n"
    v += "\n"

    # ---- read-data mux (registers only)
    v += "    // ---- read data: from registers (csr_q) and BRAM output registers only ----\n"
    v += "    reg [31:0] a_word, b_word, c_word;\n"
    v += "    always @(*) begin\n"
    if GW > 0:
        v += String("        case (lane_q[", GW - 1, ":0])\n")
        for g in range(GROUPS):
            v += String(
                "            ", GW, "'d", g, ": begin a_word = {a_rd_", 4 * g + 3, ", a_rd_", 4 * g + 2,
                ", a_rd_", 4 * g + 1, ", a_rd_", 4 * g, "}; b_word = {b_rd_", 4 * g + 3,
                ", b_rd_", 4 * g + 2, ", b_rd_", 4 * g + 1, ", b_rd_", 4 * g, "}; end\n",
            )
        v += "            default: begin a_word = 32'd0; b_word = 32'd0; end\n"
        v += "        endcase\n"
    else:
        v += "        a_word = {a_rd_3, a_rd_2, a_rd_1, a_rd_0};\n"
        v += "        b_word = {b_rd_3, b_rd_2, b_rd_1, b_rd_0};\n"
    v += "        case (lane_q)\n"
    for j in range(N):
        v += String("            ", LN, "'d", j, ": c_word = c_rd_", j, ";\n")
    v += "            default: c_word = 32'd0;\n"
    v += "        endcase\n"
    v += "    end\n"
    v += "    assign wb_dat_r = rzero_q ? 32'd0 :\n"
    v += "                      (rreg_q == R_CSR) ? csr_q :\n"
    v += "                      (rreg_q == R_A)   ? a_word :\n"
    v += "                      (rreg_q == R_B)   ? b_word : c_word;\n"
    v += "endmodule\n"
    return v^


def hex32(x: Int) -> String:
    comptime digits = "0123456789ABCDEF"
    var s = String()
    for k in range(7, -1, -1):
        var d = (x >> (4 * k)) & 0xF
        s += String(digits[byte=d])
    return s^


def tile2_verilog[N: Int, A_BYTES: Int, B_BYTES: Int, C_BYTES: Int]() -> String:
    var core = build_core2[N]()
    var v = String("/* verilator lint_off DECLFILENAME */\n")
    v += emit_verilog(core)
    v += "\n"
    v += String("// Generated by fpga/rtlgen (", TILE2_GENERATOR, ", Mojo comptime) -- DO NOT EDIT.\n")
    v += String(
        "// tile v2 top: N=", N, " A=", A_BYTES, " B=", B_BYTES, " C=", C_BYTES,
        " bytes. Contract: fpga/tile2/INTERFACE.md\n",
    )
    v += "// Regenerate: cd fpga && pixi run tile2-gen. Verify: pixi run tile2-cosim.\n"
    v += "`default_nettype none\n\n"
    v += top2[N, A_BYTES, B_BYTES, C_BYTES]()
    v += "\n`default_nettype wire\n"
    return v^


def check_tile2[N: Int, A_BYTES: Int, B_BYTES: Int, C_BYTES: Int]():
    """Comptime structural checks for a tile v2 configuration."""
    comptime assert is_pow2(N) and N >= 4 and N <= 16, "tile2: N must be 4, 8 or 16"
    comptime assert is_pow2(A_BYTES) and is_pow2(B_BYTES) and is_pow2(C_BYTES), (
        "tile2: buffer sizes must be powers of two"
    )
    comptime assert A_BYTES <= SLOT_BYTES and B_BYTES <= SLOT_BYTES and C_BYTES <= SLOT_BYTES, (
        "tile2: each buffer must fit its 32 KiB slot"
    )
    comptime assert A_BYTES >= 4 * N and B_BYTES >= 4 * N and C_BYTES >= 16 * N, (
        "tile2: buffers too small"
    )
    comptime core = build_core2[N]()
    comptime assert core.count_regs("prodq") == N * N, "one product register per PE"
    comptime assert core.reg_bits("prodq") == 16 * N * N
    comptime assert core.count_wires("mul") == N * N, "one multiplier per PE"
    comptime assert core.reg_bits("acc") == 32 * N * N
    comptime assert core.count_regs("skew") == N * (N - 1)
