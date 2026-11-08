# fpga/: int8 systolic tile: Mojo simulator (G1-A) and Mojo-generated Verilog (G1-B)

This directory holds a register-level model of an invented N×N output-stationary int8 matmul tile with int32 accumulation. The tile spec and port contract are in [DESIGN.md](DESIGN.md). Verilog for the tile is generated from Mojo at compile time (`rtlgen/`) and checked cycle-for-cycle against the simulator's traces in Verilator (`sim/`).

## Layout
| Path | What |
| --- | --- |
| `systolic/tile.mojo` | `SystolicTile[N, in_dtype=.int8, acc_dtype=.int32]`. All registers are in `TileState`. `step()` computes the next state and then commits it (two-phase). `outputs()` is Moore. |
| `systolic/driver.mojo` | Host model. It tiles M×K×N_out onto the tile with zero padding and counts stepped cycles. `HostConfig.ideal()` / `.stalling(seed)` are the host presets. `predicted_cycles()` is the analytical formula. |
| `systolic/reference.mojo` | Reference integer matmul with wrapping accumulate |
| `systolic/trace.mojo` | Per-cycle port trace writer (text or JSONL) |
| `systolic/rng.mojo` | Deterministic xorshift64* generator, so traces are reproducible |
| `tests/test_systolic.mojo` | Test suite (`pixi run test`) |
| `tools/perf.mojo` | Cycle numbers for the Llama 3.2 3B prefill-sized matmul (`pixi run perf`) |
| `tools/dump_trace.mojo` | Golden-vector dumper (`pixi run trace`) |
| `rtlgen/netlist.mojo` | Small netlist IR (ports, wires, clock-enabled registers) and its Verilog-2005 emitter |
| `rtlgen/systolic.mojo` | `build_systolic[N, in_dtype, acc_dtype]()` builds the tile netlist; `check_structure` holds comptime asserts against DESIGN.md §3 |
| `tools/gen_rtl.mojo` | `pixi run rtlgen`: writes `rtl/systolic_n{N}_i8_a32.v` for N ∈ {1, 2, 3, 4, 8, 16} |
| `rtl/*.v` | Generated Verilog, committed. Do not edit; regenerate. |
| `sim/tb.cpp` | Verilator testbench: replays a JSONL trace and checks every output field (and `st`/`cnt`) every cycle |
| `sim/cosim.sh` | `pixi run cosim`: fresh traces, Verilator builds, full co-sim matrix, negative checks |
| `sim/corrupt_trace.py` | Writes a trace copy with one wrong value (negative check) |
| `sim/synth.sh`, `sim/synth_wrap.v` | `pixi run synth`: yosys `synth_ecp5` + nextpnr-ecp5 for N = 4, 8, 16 |

## Build and test
The pixi project is pinned to the same nightly as `../mojo-max` (mojo 1.3.0.dev2026110805). It is CPU-only.
```bash
cd fpga
pixi install
pixi run test     # 24 tests, about 3 s wall
pixi run perf
```

## Dump traces
```bash
pixi run trace -- --tile 4 --m 5 --k 7 --n 6 --seed 1 --out traces/t4.txt
pixi run trace -- --tile 8 --m 13 --k 20 --n 9 --seed 2 --stalls --format jsonl --out traces/t8.jsonl
```
- `--tile` accepts 1, 2, 3, 4, 8 or 16. N is a comptime parameter, so each value is a separate instantiation, and you add more in `main`.
- `--stalls` turns on random backpressure and drives garbage on don't-care inputs.
- Every dump is checked bit-exact against the reference before it is written.
- `traces/` is gitignored. The format is in DESIGN.md §6.

## RTL generation (Mojo comptime → Verilog)
```bash
pixi run rtlgen   # writes rtl/systolic_n{1,2,3,4,8,16}_i8_a32.v (about 7 s, mostly compile)
```
`build_systolic[N, in_dtype, acc_dtype]()` builds a small netlist (`rtlgen/netlist.mojo`: ports, one-assignment wires, and registers with a common clear / `adv` / `shift` enable structure) for one tile. The FSM is a separately rendered block. `emit_verilog` turns the netlist into Verilog-2005. The register names follow DESIGN.md §3 (`a_skew_i_d`, `a_reg_i_j`, `b_reg_i_j`, `acc_i_j`; `fsm`, `cnt`, `k_len_q`), and each PE's next-state line is the §4 equation.

**What runs at compile time.** In `tools/gen_rtl.mojo`, `comptime V = systolic_verilog[N, .int8, .int32]()` makes the compiler's comptime interpreter do all of the following during `mojo build`:
- build the netlist: `List`s of structs with `String` fields, loops over PEs and skew stages
- run the `comptime assert`s in `check_structure`. These check N(N−1) skew registers, N² multipliers, and the exact §3 bit budget for skew, a_reg, b_reg and acc. A wrong count fails the build, not the simulation. This was checked by asserting a deliberately wrong multiplier count: the build fails with `constraint failed`.
- render the whole Verilog file to a `String`

**What runs at runtime.** The executable contains each module's text as a constant string. It opens `rtl/<name>.v`, writes the string and prints its size. `nm build/gen_rtl` shows no `build_systolic`, `emit_verilog` or `control_block` symbols, only `emit[N=...]` (the file writer) and `main`. `strings` on the binary finds the generated Verilog lines verbatim.

The tile size and widths are compile-time parameters, so every configuration is a separate instantiation, the same way the simulator handles N. The PE loops inside the generator are ordinary `for` loops that the interpreter executes; they do not use `comptime for`, which would unroll the generator's own IR and slow compilation without changing the output. The generated N=16 module is 145 KB, and the whole six-size build takes about 4 s.

Generated RTL properties:
- It is lint-clean under `verilator --lint-only -Wall`.
- Every output is assigned from registers only (Moore): the decode of `fsm`/`cnt`, plus `out_row` wired straight from `acc_0_*`.
- `a_reg` in the last column and `b_reg` in the last row are kept because the spec counts them, but nothing reads them. Yosys prunes them, and a lint pragma marks them.
- `fsm` and `cnt` carry `/*verilator public_flat_rd*/` so that the testbench can compare the trace's `dbg` fields. Yosys ignores the comment.

## Co-simulation (Verilator)
```bash
pixi run cosim          # regenerates rtl/, then builds and runs everything (about 11 s)
QUICK=1 pixi run cosim  # skip the K=65535 trace
```
`sim/cosim.sh` takes these steps:
1. Builds `tools/dump_trace.mojo` once.
2. Verilates each `rtl/*.v` with `sim/tb.cpp`.
3. Writes fresh JSONL traces to `traces/cosim/`. For each N ∈ {1, 2, 3, 4, 8, 16} there are 5 shapes, each with and without `--stalls`: single pass with K=1, 5×7×6, ragged (2N+1)×(2N+3)×(N+3), decode-like 1×33×(2N+1), and 13×20×9. Two more traces are added: N=16 16×300×16 with stalls, and N=4 4×65535×4, the longest legal pass.
4. Replays every trace.

The testbench follows DESIGN.md §6. For each line it drives the inputs with clk low, calls `eval()`, compares `idle`, `in_ready`, `out_valid`, `out_last`, every `out_row[j]`, and the `dbg` `st`/`cnt`, then raises clk. It stops at the first mismatch and prints the field, the RTL and trace values, and the current and previous trace lines.

**Result: 62 traces and 84,823 cycles pass with no mismatch, covering every output field in every cycle for all six tile sizes.**

The script then runs three negative checks, and each must be reported as a mismatch:
- `out_row[2]` + 1 in the first DRAIN cycle of a trace. The testbench reports `MISMATCH ... at cyc 28: out_row[2]: rtl=12965 trace=12966`.
- `out_last` cleared on its first assertion. Reported as `at cyc 31: out_last: rtl=1 trace=0`.
- An RTL mutant made with `sed` in which PE(1,1) zero-extends its product. Reported at cycle 29 on `out_row[1]`.

Verilator's default initialization zeroes all registers. That matches the trace's cycle 0, which shows outputs before the first reset edge with `idle=1`, and the ECP5 power-up state.

## ECP5 synthesis
```bash
pixi run synth          # N = 4, 8, 16; about 19 min, nearly all of it N=16 -nodsp P&R
SIZES="4" pixi run synth
```
Tools: Yosys 0.70+360 (5e740962d) and nextpnr-ecp5 0.11.1-88-g937c770c, both from the YosysHQ oss-cad-suite 2026-11-08 (darwin-arm64) unpacked into `.tools/` (gitignored). Homebrew has no nextpnr-ecp5. Homebrew also provides Verilator 5.054 and Yosys 0.70.

The script runs two synthesis passes per size:
- **Resource counts** come from `synth_ecp5` on the bare tile.
- **fmax** comes from nextpnr-ecp5 (`--package CABGA381 --freq 200 --seed 1`) run on `sim/synth_wrap.v`. The tile has 788 I/O at N=16, so the wrapper feeds the inputs from a serial shift register and XOR-folds the registered outputs. This keeps all logic live, but it adds about 48N+23 FFs, which is why the P&R utilization is higher than the bare counts.

| N | Mapping | LUT4 | CCU2C | FF | MULT18X18D | P&R device | Utilization (wrapped) | fmax |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | DSP | 790 | 272 | 836 | 16 | LFE5U-25F | LUT 5%, DSP 16/28 | 84.2 MHz |
| 8 | DSP | 2,307 | 1,040 | 3,428 | 64 | LFE5U-45F | LUT 10%, DSP 64/72 | 67.5 MHz |
| 16 | DSP | 8,583 | 4,112 | 13,988 | 256 | does not fit (85F has 156) | n/a | n/a |
| 16 | `-nodsp` | 67,964 | 6,160 | 13,988 | 0 | LFE5U-85F | LUT 96% | 60.4 MHz |

- **FF count.** It equals the §3 register budget minus the pruned dead `a_reg`/`b_reg` edge registers. For example, at N=4: 898 − 64 = 834, and 836 are mapped.
- **Critical path.** At N=8 it runs from a register into a combinational MULT18X18D (3.9 ns), then about 4.6 ns of routing from the DSP column to the fabric, then the 32-bit CCU2C accumulate chain.
- **Possible fixes for v2.** A registered product (using the DSP's pipeline registers) or a DSP-side accumulate would remove most of that path. Either would add one cycle of latency, so the flush length in DESIGN.md would change. Packing two int8 multiplies per DSP is another option.

## Test coverage (`pixi run test`: 24/24 pass)
- **Single pass, every tile size N ∈ {1, 2, 3, 4, 8, 16}:** K ∈ {1, 2, 3, N, N+1, 2N+3, 33}. Cycles must equal `K + 3N − 1` exactly.
- **Randomized shapes:** N=2 (60), N=3 (60), N=4 (60), N=8 (40), N=16 (8). M, K and N_out are random, mostly non-multiples of N, with full-range int8 values.
- **Decode-like 1×K×N_out:** N_out ∈ {1, 7, 8, 9, 31}, K ∈ {1, 5, 64, 129}, at N=4 and N=8.
- **K = 1:** a grid of M and N_out values.
- **Extreme values:**
  - all −128 × −128 and 127 × −128, with K up to 1000
  - a mixed {−128, 127, −1, 0, 1, −127} pattern
  - K = 65535 with all −128, which reaches 1,073,725,440 and must not overflow int32
- **Protocol:** 60 randomized runs with backpressure and garbage on don't-care inputs. Cycles must equal the formula plus observed stalls. Also covered: `k_len=0` and a 16-bit-truncated `k_len` are ignored, and a reset mid-pass clears all registers.
- **Width parameters:** int16 operands with an int64 accumulator.
- **Trace:** exactly one line per cycle, in both formats.

Every test checks C bit-exactly against the reference and checks cycles against `predicted_cycles`. A quick mutation check confirmed the suite catches real bugs: shortening the flush by one cycle, or mis-wiring the skew of row 1, makes the single-pass and random tests fail.

## Performance note: Llama 3.2 3B prefill-sized matmul (M=512, K=3072, N_out=8192)
`pixi run perf` simulates one full M=512 strip of output tiles for each tile size. The strip is register by register and bit-exact, and its cycle count equals the formula. The script then multiplies by the number of column tiles; every pass is identical.

| Tile | Passes | Cycles | PE utilization | @100 MHz | Throughput |
| --- | --- | --- | --- | --- | --- |
| N=8 (64 MAC/cycle) | 65,536 | 202,833,920 | 99.3% | 2.03 s | 12.7 GOP/s |
| N=16 (256 MAC/cycle) | 16,384 | 51,101,696 | 98.5% | 0.511 s | 50.4 GOP/s |

Back-of-envelope against the M5 GPU:
- **GPU rate.** Llama 3.2 3B does about 2.82 GMAC (5.64 GFLOP) of linear-layer work per token. That count excludes attention scores and lm_head. Phase-1 MAX bf16 prefill at about 810 tok/s therefore implies about 4.6 TFLOP/s effective on the M5 GPU.
- **This matmul on the GPU.** It is 12.9 GMAC (25.8 GOP), which takes about 5.6 ms on the GPU. The N=16 tile at 100 MHz is about 90× slower; N=8 is about 360× slower.
- **Whole-model prefill rate.** Compute-bound, the tile reaches about 8.9 tok/s at N=16 and about 2.2 tok/s at N=8.
- **The tile is starved by I/O well before it is limited by compute.** With no on-chip operand buffers it must be fed 2N bytes per cycle: 3.2 GB/s at N=16 and 100 MHz. This one matmul streams 16,384 × 3,072 × 32 B ≈ 1.6 GB of operands. Over a realistic USB 2.0 link (tens of MB/s) that is tens of seconds. Even the unique data alone (A 1.5 MB + B 25 MB + C 16.8 MB) is around 1 s.
- **Resources (measured, see "ECP5 synthesis" below).** Yosys maps each 8×8 multiply to a whole MULT18X18D; it does not pack two 9×9 multiplies into one DSP. N=16 needs 256 DSPs, but the LFE5U-85F has 156. In LUT multipliers it uses 96% of an 85F and closes at about 60 MHz. With DSPs, N=8 on a 45F reaches about 68 MHz. 100 MHz is **not** met by the v1 RTL.
- **Conclusion.** This is a verification and learning vehicle, not a speedup. It is interesting for proving the Mojo → RTL → "phantom NPU" path, not for beating the GPU. Getting closer would take on-chip A/B block buffers to cut re-streaming, double-buffered accumulators, and a fast host link.

## Notes and gotchas (Mojo 1.3.0.dev2026110805)
- The fixed-size stack array is `Array[T, n](fill=...)`. `InlineArray` is unknown in this build, which matches the skill.
- `DType` has no `.bit_width()` method. Use `bit_width_of[dtype]()` from `std.sys`.
- `out` cannot be an argument name because it is a soft keyword. The skill documents this.
- `Scalar[uint64].cast[int8]()` truncates, and that truncation is what the RNG relies on. Int32 `+` and `*` wrap.
- The `mojo test` subcommand is gone. Tests run through `mojo run` with `TestSuite.discover_tests`, as the skill says.
- RTL lane: a tool file named `tools/rtlgen.mojo` cannot import the package `rtlgen` ("module 'rtlgen' cannot import itself"), so the entry point is `tools/gen_rtl.mojo`.
- RTL lane: `comptime X = f()` works for a `String` built with loops and `+=`, and for `List`s of structs with `String` fields. The value is materialized as a constant, and `f` is not emitted into the binary.
- RTL lane: github.com release URLs fetched with plain `curl -L` returned an HTML page in this environment; `gh release download` worked.
