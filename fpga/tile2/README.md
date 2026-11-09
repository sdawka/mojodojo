# fpga/tile2/: int8 systolic tile v2 as a Wishbone peripheral

Tile v2 makes the v1 tile (`../DESIGN.md`) usable next to a RISC-V soft core in a LiteX SoC on an ECP5 LFE5U-85F. It adds on-chip operand buffers, a sequencer that tiles a whole M×K·K×N matmul out of those buffers, and a Wishbone classic slave. Everything is generated from Mojo at compile time, the same way as v1. The bus contract is in **[INTERFACE.md](INTERFACE.md)**.

## What changed from v1

| | v1 | v2 |
| --- | --- | --- |
| Host interface | stream N A bytes + N B bytes every cycle on wide ports (788 pins at N=16) | Wishbone classic 32-bit slave + `irq` (90 pins in total) |
| Operands | re-streamed for every output tile | stored once in BRAM: A 16 KiB, B 32 KiB, C 32 KiB (int32) |
| Tiling | done by the host | done by the sequencer (tm outer, tn inner, full K per pass) |
| C = / C += | = only | `MODE.ACCUM` does C += A·B, so firmware can split K across runs |
| PE | combinational mult → add | registered product (`prodq`), so the flush is 2N−1 cycles |
| fmax (N=8, DSP) | 67.5 MHz (45F, bare core) | **96.3–97.4 MHz** (85F, whole block, 3 seeds) |

## Layout

| Path | What |
| --- | --- |
| `INTERFACE.md` | Register and memory map, data layout, timing, programming sequence, LiteX notes |
| `../rtlgen/tile2.mojo` | Generator. `build_core2[N]` reuses v1's `build_systolic` netlist and adds the product registers. `top2[...]` renders the Wishbone/CSR/BRAM/sequencer top. `check_tile2` holds the comptime structural asserts. |
| `gen.mojo` | `pixi run tile2-gen` writes `rtl/tile2_n8.v` (the deliverable) and `rtl/tile2_n4.v` (small buffers for co-sim) |
| `rtl/*.v` | Generated Verilog-2005. Do not edit. Each file holds `tile2_core_n{N}` + `tile2_n{N}` and is lint-clean under `verilator -Wall`. |
| `../systolic/tile2.mojo` | `Tile2Model[N, A_BYTES, B_BYTES, C_BYTES]`, the bus-level model (executable spec) |
| `prog.mojo` | Randomized register-level program generator. It drives the model and records the expected read data, irq and timing. |
| `test_tile2.mojo` | `pixi run tile2-test`: model unit tests (7) |
| `../sim/tile2_tb.cpp` | Verilator Wishbone master that replays a program and checks every access |
| `../sim/tile2_cosim.sh` | `pixi run tile2-cosim`: generate, verilate, run all programs, negative checks |
| `../sim/tile2_corrupt.py` | Writes corrupted program copies for the negative checks |
| `../sim/tile2_synth.sh` | `pixi run tile2-synth`: yosys + nextpnr on the LFE5U-85F |
| `../results/tile2.json` | Machine-readable results |

## Hardware structure (`tile2_n8`)

- **Wishbone slave.** ACK is registered, so every access takes 2 cycles and never stalls. Read data comes only from registers: a CSR read register, or the BRAM output registers plus a registered lane select. Region decode uses byte-address bits [16:15].
- **Buffers.** All are simple dual port (1 write + 1 synchronous read):
  - A and B: N byte-lane memories each (8 × 2048 × 8 and 8 × 4096 × 8).
  - C: N word-lane memories, 8 × 1024 × 32, with bus byte enables.
  - When idle, the bus owns both ports. When busy, the sequencer owns them, so bus buffer accesses are dropped and set `ERR_BUSY`.
  - Yosys maps them to exactly 8 + 16 + 16 = 40 DP16KD.
- **Fetch side.** One Aᵀ row and one B row (N bytes each) are read per cycle. A beat is issued at cycle t, comes out of the BRAM at t+1, and is registered (`a_q`/`b_q`) at t+2, which is the core's FEED cycle for that beat. The first beat of each pass is issued in the SETUP cycle (pass 0) or in the previous pass's last drain cycle. Feeding therefore runs back to back with no bubbles. Pointers are incremental (`+= LD`, `+1` per tile) and wrap modulo the buffer size.
- **Core.** The v1 port contract and FSM, with `in_valid` and `out_ready` tied to 1, so it never stalls.
- **Drain side.** Each drained row is registered together with its old C row (read one cycle earlier for ACCUM), its address and its lane write enables, and is written one cycle later.
  - Rows ≥ M and columns ≥ N are not written. Padding lanes of Aᵀ/B are read but never matter: PE(i,j) only combines lane i of A with lane j of B.
  - The pipelined write costs one tail cycle per run.
- **Timing.**
  - `BUSY = 2 + ceil(M/8)·ceil(N/8)·(K + 24)` cycles.
  - Each pass is 1 start + K feed + 15 flush + 8 drain.
  - At K=1024 the PE utilization is 97.7%.

## Verification

```
pixi run tile2-test     # 7/7 model tests (about 2 s)
pixi run tile2-cosim    # regenerates RTL, 62 programs + 8 negative checks (about 13 s)
```

**The model is pinned two ways.**
- `test_tile2.mojo` drives `Tile2Model` through the bus like firmware and compares C with v1's `reference_matmul`. It covers 25 random shapes with random bases and wrap, ACCUM splitting K (23 + 27), and K=2048 at −128·−128. It also ties the cycle formula to v1's `SystolicTile[8].pass_cycles(K)`, plus 1 per pass and 2 per run.
- `prog.mojo` checks every run's C against an independent triple loop over the original matrices before it writes a program, and stops with an error on any mismatch.

**Co-simulation.** Each program is a firmware-like sequence:
- Reads ID and INFO, then writes random values to every CSR and reads them back through aliased addresses (masking).
- Writes to read-only registers, a reset, and a rejected START (K=0 → `ERR_CFG`).
- 8 matmul jobs, each:
  - Random M, K and N, with stores of mixed width (word, half-word, byte) and random gaps.
  - Random bases anywhere in the buffer, so many layouts wrap around the end.
  - Leading dimensions padded by up to 2 tiles, and garbage in all padding bytes.
  - Garbage above the size and below the alignment of the address registers.
  - ACCUM with a preloaded C (30% of jobs), and IRQ_EN at random.
- During a run: STATUS polls with random gaps (CYCLES is read while busy, so the exact count is checked), illegal writes and reads while busy (25% of jobs), and idle-waits timed to the end of BUSY.
- After each run, CYCLES and the whole C region are read back, including padding columns and one row past M (untouched-ness).

The testbench checks `wb_dat_r` on every read, ack timing on every cycle, the irq level in every request cycle, and the total cycle count at END.

**Result: 62 programs (30 seeds × N=8 and N=4, plus a K=1000 program per config) pass. They cover 486 matmul runs, 1,152,168 cycles, 246,805 writes and 92,631 checked reads.**

**Negative checks.** Each one must be reported as a mismatch, and all 8 are:

| Check | First mismatch reported |
| --- | --- |
| Program: one C value + 1 | `wb_dat_r` |
| Program: END cycles + 1 | total cycles |
| Program: one STATUS poll flipped busy → done | `wb_dat_r` |
| RTL mutant: PE(1,1) zero-extends its product | C value |
| RTL mutant: flush one cycle short | irq rises early |
| RTL mutant: C lane-7 write mask ignores N | padding overwritten |
| RTL mutant: ACCUM ignored in lane 0 | C value |
| RTL mutant: CSR write while busy not blocked | `STATUS.ERR_BUSY` |

**v1 is untouched.** `pixi run test` still passes 24/24 and `pixi run cosim` still passes 62 traces and its negative checks. `rtl/` regenerates byte-identical, because `rtlgen/systolic.mojo` and `rtlgen/netlist.mojo` were not modified and tile2 only calls them.

## Synthesis: LFE5U-85F (`pixi run tile2-synth`, about 1.5 min)

The flow is yosys `synth_ecp5` with DSP mapping, then nextpnr-ecp5 `--85k --package CABGA381 --freq 200`. The top has only 90 I/O, so it is placed bare, without v1's serializing wrapper.

| Resource | Used | Device | % |
| --- | --- | --- | --- |
| TRELLIS_COMB (LUT4 + carry) | 7,216 | 83,640 | 8.6 |
| TRELLIS_FF | 6,137 | 83,640 | 7.3 |
| MULT18X18D | 64 | 156 | 41 |
| DP16KD (18 Kb EBR) | 40 | 208 | 19 |

The Yosys cell counts are LUT4 4,090, CCU2C 1,363, FF 6,137, PFUMX 275 and L6MUX21 49.

**fmax:** 97.38 / 96.55 / 96.34 MHz for seeds 1, 2 and 3.
- The critical path is now inside the core: a register, then the combinational MULT18X18D (3.9 ns), then routing, then the `prodq` register, about 10.3 ns in total. Yosys does not absorb ECP5 DSP pipeline registers, so the multiply stays combinational.
- Before the C write was pipelined, the limit was 79.0 MHz. That path ran from the C EBR's clock-to-out (5.8 ns) through a 32-bit carry chain into the EBR's write data.
- 97 MHz is above typical LiteX ECP5 system clocks (50–75 MHz), so the tile will not limit the SoC clock.

### Room left for the SoC on the 85F

| Block | Logic | FF | DSP | EBR | Source |
| --- | --- | --- | --- | --- | --- |
| tile2_n8 | 7.2k COMB | 6.1k | 64 | 40 | measured (P&R) |
| VexRiscv "standard" (LiteX default: rv32im, I$/D$) | 3.0k LUT4 + 181 CCU2C | 1.5k | 4 | 10 | measured: `VexRiscv.v` from lane-soc's pythondata-cpu-vexriscv `642ecfed`, yosys only |
| VexRiscv "lite" | 2.8k LUT4 + 159 CCU2C | 1.1k | 0 | 4 | measured, same way |
| LiteDRAM SDR controller + PHY | ~1.5–3k | ~1–2k | 0 | ~0–4 | **estimate**, not measured (LiteDRAM not checked out) |
| LiteDRAM ECP5 DDR3 (alternative) | ~3–5k | ~3k | 0 | few | **estimate** |
| LiteX bus, UART, timer, CSR, BIOS ROM, SRAM | ~1–2k | ~1–2k | 0 | ~10–30 | **estimate** |
| **Free after tile2** | **76.4k** | **77.5k** | **92** | **168** | |

A full SoC (tile + CPU + SDRAM + infrastructure) is roughly 15–20k of the 83.6k logic cells, about 70 of the 156 DSPs and about 60–80 of the 208 EBRs. That leaves room for a second `tile2_n8` (128 + 4 DSPs). N=16 still does not fit: it needs 256 DSPs. The estimate rows should be replaced by lane-soc's measured SoC build.

## Performance and limits (honest)

- **Peak compute.** 64 MAC/cycle is 12.4 GOP/s at 97 MHz. A job that fills the buffers (M=8, K=1024, N=32) computes in 4,194 cycles at 62.5 MAC/cycle.
- **The bus is now the bottleneck.** The CPU must store Aᵀ + B (8 KiB + 32 KiB in that example, about 10k word stores at ≥ 2 bus cycles each, more in practice with VexRiscv) and read back C (256 words). Loading takes longer than computing. Two next steps would fix that:
  - A Wishbone master DMA from SDRAM, or LiteX's `WishboneDMAReader`.
  - Double-buffered A/B, so loads overlap the run. This needs bus access during BUSY, which the contract currently forbids on purpose.
- **Measured in the SoC (lane-soc, LiteX VexRiscv in Verilator, `fpga/soc`: `pixi run soc && pixi run tiletest`).** For 8×256×32, loading A/B takes 126,011 cycles, the run 1,181 (CYCLES = 1,122, as the formula predicts; the rest is polling) and reading C 2,996. The CPU-only reference takes 2,090,499 cycles, so the tile is 16× faster end to end, and 97% of that time is spent loading operands over the bus. All 6 test matmuls in the SoC are bit-exact.
- **Capacity.** Default capacity per run is K·roundup(M,8) ≤ 16 KiB, K·roundup(N,8) ≤ 32 KiB and M·roundup(N,8) ≤ 8192 C words. Larger K is split across runs with ACCUM. Larger M and N are split by firmware.
- **Firmware's responsibility.** Overlapping C tiles (`C_LD` too small) are not detected.

## Notes and gotchas

- The generator reuses v1's `build_systolic` and post-processes it. It rewrites each accumulator's `adv_expr` with `String.replace("prod_", "prodq_")` at comptime and appends the `prodq` registers. The v1 file is untouched.
- In `tile2/prog.mojo`, returning a `String` field of a local struct with `^` (`p.out^`) fails with "field destroyed out of the middle of a value". It returns `.copy()` instead.
- `comptime digits = "0123456789ABCDEF"; digits[byte=d]` works inside comptime-evaluated generator code.
- The generated file holds two modules, so it carries `/* verilator lint_off DECLFILENAME */`. `wb_adr` has `UNUSEDSIGNAL` off, because in the small N=4 configuration the high offset bits alias.
