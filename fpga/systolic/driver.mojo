"""Host model: tiles an arbitrary M x K x N_out matmul onto the N x N tile.

Output tiles are visited row-major (tm outer, tn inner). Each pass streams the
full K for one (tm, tn) tile; edge tiles are zero-padded on the way in and the
padded outputs are discarded on the way out. The cycle count is measured by
stepping the register model, never computed.
"""

from std.math import ceildiv
from std.sys import bit_width_of
from systolic.rng import Rng
from systolic.tile import SystolicTile, K_LEN_MAX
from systolic.trace import TraceWriter


@fieldwise_init
struct HostConfig(Copyable, Movable):
    """Host behaviour. The defaults are an ideal host (never stalls)."""

    var in_valid_pct: Int  # chance in_valid is high while in_ready
    var out_ready_pct: Int  # chance out_ready is high while out_valid
    var seed: UInt64
    var garbage: Bool  # drive random data / spurious start on don't-care cycles

    @staticmethod
    def ideal() -> Self:
        return Self(100, 100, 1, False)

    @staticmethod
    def stalling(seed: UInt64) -> Self:
        return Self(60, 50, seed, True)


@fieldwise_init
struct RunStats(Copyable, Movable, Writable):
    var cycles: Int  # stepped cycles from the first start to the last out_last, inclusive
    var passes: Int
    var in_stalls: Int  # FEED cycles with in_valid low
    var out_stalls: Int  # DRAIN cycles with out_ready low


def predicted_cycles(tile_n: Int, m: Int, k: Int, n_out: Int) -> Int:
    """Analytical model, ideal host: passes * (K + 3N - 1)."""
    return ceildiv(m, tile_n) * ceildiv(n_out, tile_n) * (k + 3 * tile_n - 1)


def run_matmul[
    N: Int, in_dtype: DType = .int8, acc_dtype: DType = .int32
](
    a: List[Scalar[in_dtype]],
    b: List[Scalar[in_dtype]],
    m: Int,
    k: Int,
    n_out: Int,
    cfg: HostConfig,
    mut trace: TraceWriter,
    mut c: List[Scalar[acc_dtype]],
) raises -> RunStats:
    """Runs C = A @ B on the simulated tile; writes C (m x n_out, row-major)."""
    if k < 1 or k > K_LEN_MAX:
        raise Error(t"K={k} outside [1, {K_LEN_MAX}] supported by k_len port")
    var tile = SystolicTile[N, in_dtype, acc_dtype]()
    var rng = Rng(cfg.seed)
    c = List[Scalar[acc_dtype]](length=m * n_out, fill=0)
    trace.header(N, bit_width_of[in_dtype](), bit_width_of[acc_dtype](), m, k, n_out)

    var tiles_n = ceildiv(n_out, N)
    var total_passes = ceildiv(m, N) * tiles_n

    # One reset cycle; traced as cycle 0 but not counted.
    var rst = tile.Inputs()
    rst.rst = True
    trace.record(0, rst, tile.outputs(), tile.cur.fsm, tile.cur.cnt)
    tile.step(rst)

    var stats = RunStats(0, 0, 0, 0)
    var pass_idx = 0
    var m0 = 0
    var n0 = 0
    var beat = 0
    var row = 0
    while True:
        var o = tile.outputs()
        var inp = tile.Inputs()
        if cfg.garbage:
            # Don't-care inputs: the tile must ignore all of these.
            inp.k_len = rng.range(0, K_LEN_MAX)
            inp.start = not o.idle and rng.pct(30)
            for i in range(N):
                inp.a_col[i] = rng.scalar[in_dtype]()
                inp.b_row[i] = rng.scalar[in_dtype]()
            inp.in_valid = not o.in_ready and rng.pct(50)
            inp.out_ready = not o.out_valid and rng.pct(50)
        if o.idle:
            if pass_idx == total_passes:
                break
            var tm, tn = divmod(pass_idx, tiles_n)
            m0 = tm * N
            n0 = tn * N
            beat = 0
            row = 0
            inp.start = True
            inp.k_len = k
        if o.in_ready:
            inp.in_valid = rng.pct(cfg.in_valid_pct)
            if inp.in_valid:
                for i in range(N):
                    var r = m0 + i
                    inp.a_col[i] = a[r * k + beat] if r < m else 0
                for j in range(N):
                    var col = n0 + j
                    inp.b_row[j] = b[beat * n_out + col] if col < n_out else 0
                beat += 1
            else:
                stats.in_stalls += 1
        if o.out_valid:
            inp.out_ready = rng.pct(cfg.out_ready_pct)
            if inp.out_ready:
                if o.out_last != (row == N - 1):
                    raise Error(t"out_last mismatch at drain row {row}")
                var r = m0 + row
                if r < m:
                    for j in range(N):
                        if n0 + j < n_out:
                            c[r * n_out + n0 + j] = o.out_row[j]
                row += 1
                if o.out_last:
                    pass_idx += 1
            else:
                stats.out_stalls += 1
        stats.cycles += 1
        trace.record(stats.cycles, inp, o, tile.cur.fsm, tile.cur.cnt)
        tile.step(inp)
    stats.passes = total_passes
    return stats^
