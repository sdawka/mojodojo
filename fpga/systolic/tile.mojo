"""Register-level model of one N x N output-stationary systolic tile.

Every architectural register lives in `TileState`. A clock edge is `step()`:
phase 1 computes a complete next state from the current state and the inputs
(reads only `cur`, writes only `nxt`), phase 2 commits it. Outputs are pure
functions of the current registers (Moore), so there is no input->output
combinational path. DESIGN.md is the normative spec; this file implements it.
"""

# FSM encoding (also used in traces as the `st` debug column).
comptime ST_IDLE = 0
comptime ST_FEED = 1
comptime ST_FLUSH = 2
comptime ST_DRAIN = 3

# Width of the `k_len` port; K is latched modulo 2**K_LEN_BITS.
comptime K_LEN_BITS = 16
comptime K_LEN_MAX = (1 << K_LEN_BITS) - 1


struct TileInputs[N: Int, in_dtype: DType](Copyable, Movable):
    """Values on the input ports during one cycle (sampled at the rising edge)."""

    var rst: Bool
    var start: Bool
    var k_len: Int
    var in_valid: Bool
    var a_col: Array[Scalar[Self.in_dtype], Self.N]  # A[m0+i][k], i = 0..N-1
    var b_row: Array[Scalar[Self.in_dtype], Self.N]  # B[k][n0+j], j = 0..N-1
    var out_ready: Bool

    def __init__(out self):
        self.rst = False
        self.start = False
        self.k_len = 0
        self.in_valid = False
        self.a_col = Array[Scalar[Self.in_dtype], Self.N](fill=0)
        self.b_row = Array[Scalar[Self.in_dtype], Self.N](fill=0)
        self.out_ready = False


struct TileOutputs[N: Int, acc_dtype: DType](Copyable, Movable):
    """Values on the output ports during one cycle (functions of registers only)."""

    var idle: Bool
    var in_ready: Bool
    var out_valid: Bool
    var out_last: Bool
    var out_row: Array[Scalar[Self.acc_dtype], Self.N]  # C[m0+r][n0+j]

    def __init__(out self):
        self.idle = False
        self.in_ready = False
        self.out_valid = False
        self.out_last = False
        self.out_row = Array[Scalar[Self.acc_dtype], Self.N](fill=0)


struct TileState[N: Int, in_dtype: DType, acc_dtype: DType](Copyable, Movable):
    """All architectural registers of the tile. Reset value: everything zero."""

    comptime InT = Scalar[Self.in_dtype]
    comptime AccT = Scalar[Self.acc_dtype]
    comptime NN = Self.N * Self.N

    var fsm: Int  # ST_*
    var cnt: Int  # shared beat / flush / drain counter (K_LEN_BITS wide)
    var k_len: Int  # latched at start
    # Skew delay lines. Row i of A has i stages, column j of B has j stages:
    # a_skew[i*N + d] is stage d of row i (d = 0 is nearest the port).
    var a_skew: Array[Self.InT, Self.NN]
    var b_skew: Array[Self.InT, Self.NN]
    # PE registers, row-major [i*N + j].
    var a_reg: Array[Self.InT, Self.NN]  # operand forwarded right
    var b_reg: Array[Self.InT, Self.NN]  # operand forwarded down
    var acc: Array[Self.AccT, Self.NN]  # output-stationary accumulator

    def __init__(out self):
        self.fsm = ST_IDLE
        self.cnt = 0
        self.k_len = 0
        self.a_skew = Array[Self.InT, Self.NN](fill=0)
        self.b_skew = Array[Self.InT, Self.NN](fill=0)
        self.a_reg = Array[Self.InT, Self.NN](fill=0)
        self.b_reg = Array[Self.InT, Self.NN](fill=0)
        self.acc = Array[Self.AccT, Self.NN](fill=0)


struct SystolicTile[
    N: Int, in_dtype: DType = .int8, acc_dtype: DType = .int32
](Movable):
    """N x N output-stationary int8 (by default) systolic tile, int32 accumulate."""

    comptime State = TileState[Self.N, Self.in_dtype, Self.acc_dtype]
    comptime Inputs = TileInputs[Self.N, Self.in_dtype]
    comptime Outputs = TileOutputs[Self.N, Self.acc_dtype]
    comptime FLUSH_CYCLES = 2 * (Self.N - 1)

    var cur: Self.State

    def __init__(out self):
        comptime assert Self.N >= 1, "tile size must be >= 1"
        comptime assert Self.in_dtype.is_integral(), "integer operands only"
        comptime assert Self.acc_dtype.is_integral(), "integer accumulator only"
        self.cur = Self.State()

    @staticmethod
    def pass_cycles(k: Int) -> Int:
        """Cycles of one pass with an ideal host: start + K feed + flush + N drain."""
        return 1 + k + Self.FLUSH_CYCLES + Self.N

    def outputs(self) -> Self.Outputs:
        var o = Self.Outputs()
        o.idle = self.cur.fsm == ST_IDLE
        o.in_ready = self.cur.fsm == ST_FEED
        o.out_valid = self.cur.fsm == ST_DRAIN
        o.out_last = o.out_valid and self.cur.cnt == Self.N - 1
        # Row 0 of the accumulator grid is wired to out_row in every state.
        for j in range(Self.N):
            o.out_row[j] = self.cur.acc[j]
        return o^

    def step(mut self, inp: Self.Inputs):
        """One rising clock edge: compute the full next state, then commit."""
        var nxt = self.cur.copy()
        if inp.rst:
            nxt = Self.State()
        elif self.cur.fsm == ST_IDLE:
            var k = inp.k_len & K_LEN_MAX
            if inp.start and k != 0:
                nxt = Self.State()  # synchronous clear of datapath registers
                nxt.fsm = ST_FEED
                nxt.k_len = k
        elif self.cur.fsm == ST_FEED:
            if inp.in_valid:
                self._advance(nxt, inp.a_col, inp.b_row)
                if self.cur.cnt + 1 == self.cur.k_len:
                    nxt.cnt = 0
                    nxt.fsm = ST_FLUSH if Self.FLUSH_CYCLES > 0 else ST_DRAIN
                else:
                    nxt.cnt = self.cur.cnt + 1
        elif self.cur.fsm == ST_FLUSH:
            var zeros = Array[Scalar[Self.in_dtype], Self.N](fill=0)
            self._advance(nxt, zeros, zeros)
            if self.cur.cnt + 1 == Self.FLUSH_CYCLES:
                nxt.cnt = 0
                nxt.fsm = ST_DRAIN
            else:
                nxt.cnt = self.cur.cnt + 1
        else:  # ST_DRAIN
            if inp.out_ready:
                # Shift accumulator rows up by one; row N-1 fills with zero.
                for i in range(Self.N):
                    for j in range(Self.N):
                        nxt.acc[i * Self.N + j] = (
                            self.cur.acc[(i + 1) * Self.N + j] if i
                            + 1 < Self.N else 0
                        )
                if self.cur.cnt + 1 == Self.N:
                    nxt.cnt = 0
                    nxt.fsm = ST_IDLE
                else:
                    nxt.cnt = self.cur.cnt + 1
        self.cur = nxt^

    def _advance(
        self,
        mut nxt: Self.State,
        a_edge: Array[Scalar[Self.in_dtype], Self.N],
        b_edge: Array[Scalar[Self.in_dtype], Self.N],
    ):
        """Next-state of the skew lines and PE grid for one advancing cycle."""
        comptime n = Self.N
        ref cur = self.cur
        for i in range(n):
            # Skew line shift: stage 0 takes the port, stage d takes stage d-1.
            for d in range(i):
                nxt.a_skew[i * n + d] = a_edge[i] if d == 0 else cur.a_skew[
                    i * n + d - 1
                ]
                nxt.b_skew[i * n + d] = b_edge[i] if d == 0 else cur.b_skew[
                    i * n + d - 1
                ]
        for i in range(n):
            for j in range(n):
                var a_in: Scalar[Self.in_dtype]
                if j > 0:
                    a_in = cur.a_reg[i * n + j - 1]
                elif i == 0:
                    a_in = a_edge[0]
                else:
                    a_in = cur.a_skew[i * n + i - 1]
                var b_in: Scalar[Self.in_dtype]
                if i > 0:
                    b_in = cur.b_reg[(i - 1) * n + j]
                elif j == 0:
                    b_in = b_edge[0]
                else:
                    b_in = cur.b_skew[j * n + j - 1]
                var idx = i * n + j
                nxt.acc[idx] = cur.acc[idx] + a_in.cast[
                    Self.acc_dtype
                ]() * b_in.cast[Self.acc_dtype]()
                nxt.a_reg[idx] = a_in
                nxt.b_reg[idx] = b_in
