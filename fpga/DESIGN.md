# Systolic tile v1: specification

This file specifies an invented int8 matmul tile. `systolic/tile.mojo` is the executable reference model. The RTL (lane G1-B) must match it **cycle-for-cycle at the ports**, using the traces described below.

## 1. Summary

- An N×N grid of processing elements (PEs), **output-stationary**. PE(i,j) owns `C[m0+i][n0+j]`.
- Operands: A is `M×K` and B is `K×N_out`, both signed `IN_BITS` (default int8). The accumulator is signed `ACC_BITS` (default int32) with two's-complement wraparound.
- Each cycle the host presents one **unskewed** column of the A block (`a_col[i] = A[m0+i][k]`) and one row of the B block (`b_row[j] = B[k][n0+j]`). The tile skews them internally with delay lines.
- In each advancing cycle, PE(i,j) does `acc += a_in*b_in`, forwards `a` right and forwards `b` down.
- After K beats and a 2(N−1)-cycle flush, the tile drains results row 0 first, one row per cycle, by shifting the accumulator grid up.
- Comptime parameters: `N`, `in_dtype` (IN_BITS) and `acc_dtype` (ACC_BITS). `K_LEN_BITS = 16` is fixed in v1.

## 2. Ports

All ports are synchronous to `clk` (rising edge). Every output is a function of registers only (Moore), so there is no combinational path from input to output.

| Port | Dir | Width | Meaning |
| --- | --- | --- | --- |
| `clk` | in | 1 | clock |
| `rst` | in | 1 | synchronous, active high. Has priority over everything else. |
| `start` | in | 1 | starts a pass. Accepted only when `idle && k_len != 0`. Ignored otherwise. |
| `k_len` | in | 16 | K for the pass, latched on an accepted start. Valid range is 1..65535. |
| `in_valid` | in | 1 | host offers a beat. A beat transfers on an edge where `in_ready && in_valid`. |
| `a_col[i]` | in | N × IN_BITS | `A[m0+i][k]`, i = 0..N−1. Don't-care unless a beat transfers. |
| `b_row[j]` | in | N × IN_BITS | `B[k][n0+j]`, j = 0..N−1. Don't-care unless a beat transfers. |
| `out_ready` | in | 1 | host accepts a row. A row transfers on an edge where `out_valid && out_ready`. |
| `idle` | out | 1 | `fsm == IDLE` |
| `in_ready` | out | 1 | `fsm == FEED` |
| `out_valid` | out | 1 | `fsm == DRAIN` |
| `out_last` | out | 1 | `fsm == DRAIN && cnt == N−1` |
| `out_row[j]` | out | N × ACC_BITS | **always** `acc[0][j]` (row 0 of the grid, with no mux). During DRAIN cycle r it is `C[m0+r][n0+j]`. |

Suggested Verilog packing: `a_col` is a flat `[N*IN_BITS-1:0]` bus with element 0 in the low bits. `b_row` and `out_row` are packed the same way. Traces list elements in index order 0..N−1.

## 3. Registers and reset

| Register | Width | Count |
| --- | --- | --- |
| `fsm` | 2 | 1. Encoding: IDLE=0, FEED=1, FLUSH=2, DRAIN=3 |
| `cnt` | 16 | 1. A shared counter for beats, flush cycles and drain rows. |
| `k_len_q` | 16 | 1 |
| `a_skew[i][d]` | IN_BITS | Row i has i stages (d = 0..i−1, where stage 0 is nearest the port). Total N(N−1)/2. |
| `b_skew[j][d]` | IN_BITS | Same layout, per column j |
| `a_reg[i][j]`, `b_reg[i][j]` | IN_BITS | N² each |
| `acc[i][j]` | ACC_BITS | N² |

On `rst`, **every** register becomes zero, including `fsm` (IDLE). An accepted `start` does the same to every register except that it sets `fsm=FEED` and `k_len_q=k_len`. In IDLE without an accepted start, all registers hold their values.

## 4. Next-state function (one rising edge)

Define `adv = (fsm==FEED && in_valid) || fsm==FLUSH`. This acts as a global clock enable for the datapath. The edge inputs are `ea = a_col, eb = b_row` in FEED and all zeros in FLUSH.

When `adv` is true, all of the following update simultaneously from current values:
```
a_skew[i][0] <= ea[i];      a_skew[i][d] <= a_skew[i][d-1]      (d >= 1)
b_skew[j][0] <= eb[j];      b_skew[j][d] <= b_skew[j][d-1]
left[i]  = (i == 0) ? ea[0] : a_skew[i][i-1]      // row i delayed i cycles
top[j]   = (j == 0) ? eb[0] : b_skew[j][j-1]      // column j delayed j cycles
a_in(i,j) = (j == 0) ? left[i] : a_reg[i][j-1]
b_in(i,j) = (i == 0) ? top[j]  : b_reg[i-1][j]
acc[i][j]   <= acc[i][j] + sext(a_in) * sext(b_in)   (mod 2^ACC_BITS)
a_reg[i][j] <= a_in;  b_reg[i][j] <= b_in
```
When `adv` is false, the skew lines, `a_reg`, `b_reg` and `acc` hold their values. The exception is DRAIN, described next. A FEED stall therefore freezes the whole wavefront, which keeps stalls trivially correct.

FSM and counter, where `(x)` means "if x":

| State | Condition | Next state |
| --- | --- | --- |
| IDLE | `start && k_len != 0` | clear (see §3), `fsm=FEED`, `k_len_q=k_len` |
| FEED | `in_valid` | `cnt == k_len_q−1` → `cnt=0`, `fsm = (N>1) ? FLUSH : DRAIN`. Otherwise `cnt++`. |
| FLUSH | always | `cnt == 2(N−1)−1` → `cnt=0`, `fsm=DRAIN`. Otherwise `cnt++`. |
| DRAIN | `out_ready` | `acc[i][j] <= acc[i+1][j]`, `acc[N−1][j] <= 0`. Then `cnt == N−1` → `cnt=0`, `fsm=IDLE`. Otherwise `cnt++`. |

Wavefront timing: number the advancing edges of a pass 0, 1, 2, …. Beat k enters at advancing edge k. PE(i,j) accumulates `A[m0+i][k]·B[k][n0+j]` at advancing edge `k+i+j`. The last product reaches PE(N−1,N−1) at edge `(K−1)+2(N−1)`, which is the last FLUSH edge. Before its first product and after its last one, each PE sees only zeros, because the clear on start and the zero flush guarantee it. The accumulator therefore holds exactly the dot product.

## 5. Cycle-count model

The host is ideal when it asserts `start` in the first IDLE cycle, keeps `in_valid=1` in FEED and keeps `out_ready=1` in DRAIN. With an ideal host, one pass takes

```
T_pass(K, N) = 1 (start) + K (feed) + 2(N−1) (flush) + N (drain) = K + 3N − 1
```

These cycles run from the cycle that asserts `start` through the cycle where the `out_last` row is accepted, inclusive. The next `start` can come in the very next cycle.

For an M×K×N_out problem, the host visits output tiles row-major (tm outer, tn inner). It zero-pads edge tiles on input (`a_col[i]=0` for `m0+i ≥ M`, `b_row[j]=0` for `n0+j ≥ N_out`) and discards padded outputs:

```
cycles(M, K, N_out; N) = ceil(M/N) · ceil(N_out/N) · (K + 3N − 1)
                       + (#FEED cycles with in_valid=0) + (#DRAIN cycles with out_ready=0)
```

K is never tiled. One pass covers the whole reduction, so int32 results leave the array exactly once. Overflow is impossible within the port range: `65535 · 128² < 2³¹`, and the test checks this at K = 65535.

The PE utilization of a pass is `K / (K + 3N − 1)`, which is 99.3% at N=8 and 98.5% at N=16 for K = 3072.

## 6. Trace format (golden vectors)

`pixi run trace -- ...` writes one line per cycle. Line `cyc` holds the **inputs driven during cycle `cyc`**, which are sampled at the rising edge that ends it. It also holds the **outputs observed during cycle `cyc`**, before that edge. Cycle 0 is always a reset cycle (`rst=1`).

Text format (`--format text`):
```
# systolic-tile trace v1 N=4 in_bits=8 acc_bits=32 k_len_bits=16 problem M=.. K=.. N_out=..
# cyc rst start k_len in_valid out_ready a0..a{N-1} b0..b{N-1} | idle in_ready out_valid out_last r0..r{N-1} | st cnt
```
All values are signed decimal. `st`/`cnt` are the FSM and counter registers, which are useful for debugging and optional to compare. JSONL (`--format jsonl`) writes the same fields as `{"cyc", "in":{...}, "out":{...}, "dbg":{"st","cnt"}}`.

Harness loop for Verilator: for each line, apply `in`, call `eval()`, compare `out` (all fields, including `out_row` in every cycle), then toggle `clk` 0→1. With `--stalls`, the host randomly drops `in_valid`/`out_ready`. It also drives random data, `k_len`, spurious `start`, and `in_valid`/`out_ready` on cycles where they are don't-cares. That checks that the RTL ignores them.

## 7. Why output-stationary for prefill, and what weight-stationary would change

**Output-stationary (chosen).**
- The int32 partial sums never move, so K can be any length without partial-sum traffic or an external accumulator buffer.
- Each pass yields final results.
- The only wide (32-bit) datapath is the drain shift.
- For prefill (M = 512 rows, K = 3072–8192), the passes are long and the fixed overhead of 3N−1 cycles amortizes to about 1%.
- The host side stays trivial: stream two int8 vectors per cycle and read N int32 rows per pass. That suits a USB-attached FPGA, where read-modify-write of partial sums over the link would be ruinous.
- Cost: both operands stream for every pass. A has `ceil(N_out/N)` reuse misses and B has `ceil(M/N)`, unless the host or on-chip buffers cache an operand block.

**Weight-stationary (not chosen).**
- Each PE would hold one B element, loaded through a shift chain in N cycles per N×N weight block.
- A rows would stream left→right and int32 partial sums would flow top→bottom. The vertical bus becomes ACC_BITS wide instead of IN_BITS.
- Results would emerge skewed at the bottom edge, so a de-skew buffer is needed.
- K must be tiled into K/N slices. Each slice produces partial sums that something outside the array must add, which needs an M×N int32 accumulator SRAM next to the tile.
- WS gives the best weight reuse when M ≫ N: a weight block is loaded once and then M rows stream through.
- Decode (M=1) is poor in both: WS spends N load cycles per row, and OS wastes N−1 of N rows.
- The port contract would change: a `w_load` phase, `psum_in`/`psum_out` buses (or an accumulator SRAM interface), and a skewed output.

**Not in v1, possible in v2:**
- Double-buffered accumulators: drain pass p while pass p+1 computes, giving T ≈ K + 2N − 1.
- Start draining early: row 0 is final at advancing edge K + N − 2.
- On-chip operand buffers so the host does not re-stream A/B for every tile.
