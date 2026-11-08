// Place-and-route wrapper for timing estimates only (not part of the tile).
//
// The tile has 20 + 16N input pins and 4 + 32N output pins (788 at N=16), more
// than any ECP5 package bonds out. This wrapper feeds every tile input from a
// serial-in shift register and folds every tile output into one registered XOR
// bit, so all tile logic stays live and every tile path is register-to-register.
// It adds about (20 + 16N) + (4 + 32N) FFs and a small XOR tree.
//
// Defines: TILE (tile module name), TILE_N.
`default_nettype none

module synth_wrap (
    input  wire clk,
    input  wire rst_pin,
    input  wire din,
    output reg  dout
);
    localparam N = `TILE_N;
    localparam WI = 19 + 16 * N;  // start, k_len[16], in_valid, out_ready, a_col, b_row
    localparam WO = 4 + 32 * N;
    localparam G = (WO + 15) / 16;

    reg [WI-1:0] sr;
    reg rst_q;
    always @(posedge clk) begin
        sr <= {sr[WI-2:0], din};
        rst_q <= rst_pin;
    end

    wire idle, in_ready, out_valid, out_last;
    wire [32*N-1:0] out_row;
    `TILE u_tile (
        .clk(clk),
        .rst(rst_q),
        .start(sr[0]),
        .k_len(sr[16:1]),
        .in_valid(sr[17]),
        .out_ready(sr[18]),
        .a_col(sr[19 +: 8*N]),
        .b_row(sr[19 + 8*N +: 8*N]),
        .idle(idle),
        .in_ready(in_ready),
        .out_valid(out_valid),
        .out_last(out_last),
        .out_row(out_row)
    );

    reg [16*G-1:0] oq;
    reg [G-1:0] part;
    integer g;
    always @(posedge clk) begin
        oq <= {{(16*G-WO){1'b0}}, out_row, idle, in_ready, out_valid, out_last};
        for (g = 0; g < G; g = g + 1)
            part[g] <= ^oq[16*g +: 16];
        dout <= ^part;
    end
endmodule

`default_nettype wire
