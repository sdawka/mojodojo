"""Per-cycle port trace (golden vectors for the RTL co-simulation).

Row `cyc` holds the inputs driven during cycle `cyc` (sampled at the rising edge
that ends it) and the outputs observed during cycle `cyc` (before that edge),
plus the FSM state/counter as debug columns. See DESIGN.md "Trace format".
"""

from systolic.tile import TileInputs, TileOutputs, K_LEN_BITS

comptime TRACE_OFF = 0
comptime TRACE_TEXT = 1
comptime TRACE_JSONL = 2


struct TraceWriter(Movable):
    var mode: Int
    var buf: String
    var lines: Int

    def __init__(out self, mode: Int = TRACE_OFF):
        self.mode = mode
        self.buf = String()
        self.lines = 0

    def header(mut self, n: Int, in_bits: Int, acc_bits: Int, m: Int, k: Int, n_out: Int):
        if self.mode != TRACE_TEXT:
            return
        self.buf.write(
            "# systolic-tile trace v1 N=", n, " in_bits=", in_bits,
            " acc_bits=", acc_bits, " k_len_bits=", K_LEN_BITS,
            " problem M=", m, " K=", k, " N_out=", n_out, "\n",
        )
        self.buf.write("# cyc rst start k_len in_valid out_ready")
        for i in range(n):
            self.buf.write(" a", i)
        for j in range(n):
            self.buf.write(" b", j)
        self.buf.write(" | idle in_ready out_valid out_last")
        for j in range(n):
            self.buf.write(" r", j)
        self.buf.write(" | st cnt\n")

    def record[
        N: Int, in_dtype: DType, acc_dtype: DType
    ](
        mut self,
        cyc: Int,
        inp: TileInputs[N, in_dtype],
        outp: TileOutputs[N, acc_dtype],
        st: Int,
        cnt: Int,
    ):
        if self.mode == TRACE_OFF:
            return
        self.lines += 1
        if self.mode == TRACE_TEXT:
            self.buf.write(
                cyc, " ", Int(inp.rst), " ", Int(inp.start), " ", inp.k_len,
                " ", Int(inp.in_valid), " ", Int(inp.out_ready),
            )
            for i in range(N):
                self.buf.write(" ", inp.a_col[i])
            for j in range(N):
                self.buf.write(" ", inp.b_row[j])
            self.buf.write(
                " | ", Int(outp.idle), " ", Int(outp.in_ready), " ",
                Int(outp.out_valid), " ", Int(outp.out_last),
            )
            for j in range(N):
                self.buf.write(" ", outp.out_row[j])
            self.buf.write(" | ", st, " ", cnt, "\n")
            return
        # JSONL: one object per cycle.
        self.buf.write(
            '{"cyc":', cyc, ',"in":{"rst":', Int(inp.rst), ',"start":',
            Int(inp.start), ',"k_len":', inp.k_len, ',"in_valid":',
            Int(inp.in_valid), ',"out_ready":', Int(inp.out_ready), ',"a_col":[',
        )
        for i in range(N):
            self.buf.write("," if i else "", inp.a_col[i])
        self.buf.write('],"b_row":[')
        for j in range(N):
            self.buf.write("," if j else "", inp.b_row[j])
        self.buf.write(
            ']},"out":{"idle":', Int(outp.idle), ',"in_ready":', Int(outp.in_ready),
            ',"out_valid":', Int(outp.out_valid), ',"out_last":', Int(outp.out_last),
            ',"out_row":[',
        )
        for j in range(N):
            self.buf.write("," if j else "", outp.out_row[j])
        self.buf.write(']},"dbg":{"st":', st, ',"cnt":', cnt, "}}\n")

    def save(self, path: String) raises:
        with open(path, "w") as f:
            f.write(self.buf)
