// Minimal Wishbone (classic, 32-bit, word-addressed) slave around the v1
// systolic tile (fpga/rtl/systolic_n4_i8_a32.v, spec fpga/DESIGN.md).
// Stand-in until tile2 (fpga/tile2/INTERFACE.md) lands. Hand-written.
//
// Register map (byte offsets from the slave base):
//   0x00 W  CTRL    write any value: start a pass with K = KLEN (ignored unless idle)
//   0x00 R  STATUS  bit0 idle, bit1 in_ready (FEED), bit2 done (all N rows captured)
//   0x04 RW KLEN    K for the next pass (1..65535)
//   0x08 RW A       packed a_col: byte i = A[m0+i][k] (int8)
//   0x0C W  B       packed b_row: byte j = B[k][n0+j]; the write pushes one beat
//                   (A,B). The bus ack is held until the tile accepts the beat
//                   (in_ready), so back-to-back writes are flow-controlled.
//                   A push while not in FEED/start-pending is acked and dropped.
//   0x10 R  CYCLES  tile cycles from start to the out_last row (inclusive)
//   0x14 R  ID      0x7111_0004 (tile v1 wrapper, N=4)
//   0x40 + 4*(N*r + j) R  C[r][j] (int32), captured from out_row during DRAIN.
//                   The wrapper drives out_ready=1 (ideal host on the drain side).
`default_nettype none
module tile_v1_wb (
    input  wire        clk,
    input  wire        rst,
    input  wire [29:0] wb_adr,     // word address (within slave window)
    input  wire [31:0] wb_dat_w,
    output reg  [31:0] wb_dat_r,
    input  wire [3:0]  wb_sel,
    input  wire        wb_cyc,
    input  wire        wb_stb,
    output reg         wb_ack,
    input  wire        wb_we
);
    localparam N = 4;
    wire idle, in_ready, out_valid, out_last;
    wire [32*N-1:0] out_row;

    reg  [15:0] klen;
    reg  [31:0] a_q;
    reg         start;
    reg         in_valid;
    reg  [31:0] b_q;
    reg         done;
    reg         running;
    reg  [31:0] cycles;
    reg  [1:0]  row;
    reg  [31:0] c [0:N*N-1];

    systolic_n4_i8_a32 tile (
        .clk(clk), .rst(rst), .start(start), .k_len(klen),
        .in_valid(in_valid), .a_col(a_q), .b_row(b_q), .out_ready(1'b1),
        .idle(idle), .in_ready(in_ready), .out_valid(out_valid),
        .out_last(out_last), .out_row(out_row));

    wire [5:0] widx = wb_adr[5:0];
    wire req = wb_cyc && wb_stb && !wb_ack;
    integer i;

    always @(posedge clk) begin
        start   <= 1'b0;
        wb_ack  <= 1'b0;
        if (rst) begin
            klen <= 16'd0; a_q <= 32'd0; b_q <= 32'd0; in_valid <= 1'b0;
            done <= 1'b0; running <= 1'b0; cycles <= 32'd0; row <= 2'd0;
        end else begin
            // beat handshake: in_valid held until the tile takes it
            if (in_valid && in_ready) begin
                in_valid <= 1'b0;
            end
            // cycle counter / result capture
            if (start) begin
                running <= 1'b1; done <= 1'b0; cycles <= 32'd1; row <= 2'd0;
            end else if (running) begin
                cycles <= cycles + 32'd1;
            end
            if (out_valid) begin
                for (i = 0; i < N; i = i + 1)
                    c[{row, 2'b00} + i] <= out_row[32*i +: 32];
                row <= row + 2'd1;
                if (out_last) begin running <= 1'b0; done <= 1'b1; end
            end
            // bus
            if (req) begin
                if (wb_we) begin
                    case (widx)
                        6'h00: begin
                            if (idle && !running && klen != 16'd0) start <= 1'b1;
                            wb_ack <= 1'b1;
                        end
                        6'h01: begin klen <= wb_dat_w[15:0]; wb_ack <= 1'b1; end
                        6'h02: begin a_q  <= wb_dat_w;       wb_ack <= 1'b1; end
                        6'h03: begin
                            // accept the push once no beat is pending
                            if (!in_valid) begin
                                b_q <= wb_dat_w;
                                in_valid <= running || start; // drop if not in a pass
                                wb_ack <= 1'b1;
                            end
                        end
                        default: wb_ack <= 1'b1;
                    endcase
                end else begin
                    case (widx)
                        6'h00: wb_dat_r <= {29'd0, done, in_ready, idle};
                        6'h01: wb_dat_r <= {16'd0, klen};
                        6'h02: wb_dat_r <= a_q;
                        6'h04: wb_dat_r <= cycles;
                        6'h05: wb_dat_r <= 32'h7111_0004;
                        default: wb_dat_r <= (widx >= 6'h10 && widx < 6'h10 + N*N)
                                             ? c[widx - 6'h10] : 32'd0;
                    endcase
                    wb_ack <= 1'b1;
                end
            end
        end
    end
endmodule
`default_nettype wire
