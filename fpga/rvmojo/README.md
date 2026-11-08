# rvmojo: Mojo as bare-metal firmware for an RV32IM soft core

This directory makes Mojo the firmware language for a VexRiscv-class core
(rv32im, no FPU, no OS) that drives our int8 matmul tile over memory-mapped
registers. Mojo now covers both ends of that bus: it generates the tile's
Verilog (`fpga/rtlgen`, `fpga/tile2`) and it is also the code running on the
CPU that drives the tile.

**Result (2026-11-08).** Mojo firmware built here runs on two targets:

1. **The real VexRiscv RTL** inside lane-soc's LiteX Verilator SoC (`fpga/soc`).
   It drives the RTL tile v1 (`fpga/soc/rtl/tile_v1_wb.v` around
   `fpga/rtl/systolic_n4_i8_a32.v`) and gets bit-exact results against a Mojo
   CPU reference.
2. **rv32sim**, a small RV32IM instruction-set simulator written in Mojo
   (`sim/`). It models the same memory map, so one `.bin` runs on both. It
   also hosts lane-tile2's executable spec of tile v2, and the tile v2 driver
   passes against it.

The tile v2 driver also runs on RTL: lane-soc wired lane-tile2's generated
`tile2_n8` into the SoC at 0x90000000, and `tile2_test` passes there, up to
72x600x80, which is 39x faster than the core alone.

Toolchain: Mojo `1.3.0.dev2026110805` from the `mojo-max/` pixi env, plus
`ld.lld` (LLD 24) from that same env. Nothing was downloaded. For
disassembly, `llvm-objdump`/`llvm-objcopy` with RISC-V support come from the
`fpga/soc` pixi env, used read-only. The LiteX build also produces an rv32
`libcompiler_rt.a`, which is linked when present.

## Layout

| Path | What |
| --- | --- |
| `fw/rt.mojo` | Runtime: crt0 written as Mojo inline asm, `memset`/`memcpy`/`memmove`, volatile MMIO helpers, UART console (`puts`/`put_dec`/`put_hex`, no allocation), SimCtrl cycle counter, `Arena` bump allocator, `fw_exit` (`SIM_FINISH` protocol), and `__floatundidf` |
| `fw/shim.mojo` | About 120 lines of libc/Mojo-runtime shim (`write`, `fdopen`, `KGEN_CompilerRT_AlignedAlloc`, `__atomic_*`, ...) so that `print`, `String`, `List` and `raise` work on bare metal. Linked automatically only when the app needs it |
| `fw/tile_v1.mojo` | Driver for tile v1 (4x4, beat-streaming registers): `tile_matmul_i8(a, b, c, M, K, N)`, plus `cpu_matmul_i8` as the reference |
| `fw/tile2.mojo` | Driver for tile v2 (`fpga/tile2/INTERFACE.md` v2.0, buffer-based): `tile2_matmul_i8(a, b, c, M, K, N)`. Reads INFO, blocks over M and N, splits K with MODE.ACCUM, packs operands into 32-bit stores, polls BUSY, checks DONE/ERR |
| `fw/libtile.mojo` | C-ABI exports for linking into C firmware (lane-soc's `fw.mk EXTRA_OBJS=`): `mojo_tile_matmul_i8`, `mojo_tile2_matmul_i8`, `mojo_cpu_matmul_i8` (`const int8_t *a, const int8_t *b, int32_t *c, int M, int K, int N`). Needs only `memset` (picolibc or rt) |
| `fw/matmul_test.mojo`, `fw/tile2_test.mojo` | Self-checking firmware: tile result compared with the CPU reference, prints cycle counts |
| `fw/hello.mojo`, `fw/diffcheck.mojo` (+ `checks.mojo`, `diffcheck_host.mojo`), `fw/stdlib_demo.mojo`, `fw/softfloat_demo.mojo` | Capability demos. `diffcheck` computes the same integer checksum on the host and on rv32 |
| `link.ld.in` | Linker script template. "ROM" holds code, rodata and the .data load image; "RAM" holds .data, .bss, heap and stack |
| `build.sh` | `build.sh fw/<app>.mojo` produces `build/<app>.{elf,bin,lst,map}` |
| `sim/rv32sim.mojo`, `sim/tile_model.mojo` | The Mojo ISS and its tile v1 model |
| `check.sh` | Builds everything, runs it on rv32sim and (if built) the Verilator SoC, and asserts the results |
| `probes/` | Single-feature capability probes. `probes/run_probes.sh` prints the external symbols each probe needs |

## How to run

```sh
fpga/rvmojo/check.sh                    # everything, with assertions (~1 min)

fpga/rvmojo/build.sh fpga/rvmojo/fw/matmul_test.mojo
fpga/rvmojo/build/rv32sim fpga/rvmojo/build/matmul_test.elf [--trace] [--tile-fault]
cd fpga/soc && .venv/bin/python run.py ../rvmojo/build/matmul_test.bin   # Verilator SoC
```

To build the ISS:
`cd mojo-max && pixi run mojo build -I ../fpga/rvmojo/sim -I ../fpga -o ../fpga/rvmojo/build/rv32sim ../fpga/rvmojo/sim/rv32sim.mojo`.

### Build recipe (what `build.sh` does)

```sh
T="--target-triple riscv32-unknown-none-elf --target-cpu generic-rv32 --target-features +m,+zicsr"
mojo build $T --emit object -o rt.o   fw/rt.mojo            # runtime: its own object (see note 1)
mojo build $T -I fw --emit object -o app.o fw/app.mojo     # exports fw_main
[mojo build $T -I fw --emit object -o shim.o fw/shim.mojo] # only if app.o needs it
lld -flavor gnu -m elf32lriscv -nostdlib -static --no-relax -T link.ld \
    -o app.elf rt.o app.o [shim.o] [fpga/soc/build/sim/software/libcompiler_rt/libcompiler_rt.a]
llvm-objcopy -O binary app.elf app.bin                      # main-RAM image for fpga/soc/run.py
```

Board addresses are Mojo `-D` defines, set through environment variables to
`build.sh`: `UART_BASE`, `SIMCTRL_BASE`, `TILE_BASE`, `TILE2_BASE`. The
memory layout uses `ROM_BASE/ROM_SIZE/RAM_BASE/RAM_SIZE`. The defaults match
`fpga/soc/build/sim/csr.csv`:

- main_ram at 0x40000000. The BIOS jumps here. Code goes in the first 1 MiB;
  data, heap and stack go in the next 1 MiB.
- uart at 0xF0002000, simctrl at 0xF0001000.
- tile v1 at 0x82000000, tile2 at 0x90000000.

## Capability matrix (riscv32-unknown-none-elf, rv32im + zicsr, ilp32 soft-float)

Evidence: `probes/run_probes.sh` (undefined symbols per feature), `--emit asm`
on the probes, and the firmware runs below. "Works" means it compiles to
plain RV32IM code with no external symbols.

| Feature | Status | Evidence / notes |
| --- | --- | --- |
| Int32/UInt32/Int8/Int16 arithmetic, loops, compares | **works** | `p_ints`: `add_i32` is `add a0,a1,a0; ret`. `sum_loop` is closed-form `mul`/`mulhu`. diffcheck matches the host |
| `Int` width | **works, 32-bit** | Int is pointer-sized: `size_of[Int]()` lowers to `li a0, 4` (`p_misc int_size`), `add_int` is a single `add`, and stdlib_demo prints "Int is 32 bits". **The feared 64-bit-Int trap does not happen** on riscv32 |
| Int64/UInt64 `*`, shifts | **works** | `mul_i64` is `mul`/`mulhu`. diffcheck uses `mulh`/`mulhu` |
| Unsigned `//` `%` (32-bit) | **works** | plain `divu`/`remu` |
| Signed `//` with a runtime divisor (Int, Int32) | **needs a workaround** | Mojo's floor-division lowering builds an f64 reciprocal, so it needs `__floatundidf __floatunsidf __divdf3 __fixunsdfdi` (`p_div`, `p_div2`). Signed `%` and a constant divisor are fine. Fix used here: link LiteX's `libcompiler_rt.a`, and `rt.mojo` supplies the one missing member, `__floatundidf`, written in Mojo. Verified by `softfloat_demo` (`-17 // 5 = -4`) and the randomized signed div/mod in diffcheck (matches the host). Alternative: use unsigned division |
| UInt64 `//` | **needs a workaround** | needs `__udivdi3` (compiler-rt) |
| Float32/Float64 (no FPU) | **needs a workaround** | soft-float libcalls (`__addsf3 __mulsf3 __floatsisf`), resolved by compiler-rt (`softfloat_demo`). With `+f --target-abi ilp32f`, Mojo emits `fmadd.s`/`fcvt.s.w` for FPU cores |
| Structs (`@fieldwise_init`, methods, TrivialRegisterPassable) | **works** | `struct_dot` is `mul; slli; ret`. The `Lcg` struct runs in diffcheck |
| comptime params, `comptime for` | **works** | `comptime_scale` fully folds. The console's `put_hex` uses `comptime for` |
| SIMD types | **works (scalarized)** | `SIMD[int8,8]` load, cast, `*`, `reduce_add` become `lb`/`mul`/`add` (no V extension). `SIMD[int16,8]` is in diffcheck |
| `Array[T, N]` on the stack, `Span` | **works** | `Array(fill=0)` emits a call to `memset`, which `rt.mojo` provides. `Span(arr)` iteration has no external symbols (`p_span`) |
| Pointer loads/stores to fixed addresses | **works** | `MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=ADDR)` |
| **Volatile MMIO** | **works** | `p.unsafe_store[volatile=True](v)` / `p.unsafe_load[volatile=True]()`. In `p_mmio`, two volatile stores to the same register both survive and the poll loop re-loads. The non-volatile version keeps only the last store, so non-volatile MMIO is elided |
| Inline asm | **works** | `from std.sys._assembly import inlined_assembly`. `csrr $0, mcycle` (`p_misc`), `fence`, and the whole crt0 (`.pushsection .text.init`). Gotchas: `$` is the operand sigil, and `la` assembles to a GOT load because the code is PIC, so crt0 uses `lla` |
| `@export def f(...) abi("C")` | **works** | C ABI entry points (`fw_main`, `memset`, ...) |
| `-D NAME=value` board config | **works (decimal only)** | `get_defined_int["UART_BASE", default]()`. A hex value is rejected ("not an integer"), so `build.sh` converts hex to decimal |
| `StaticString` literals | **works** | land in `.rodata`. `puts(StaticString)` needs no allocation |
| `print`, `String`, `List`, `raise`/`except` | **needs a workaround** | They pull in `KGEN_CompilerRT_AlignedAlloc/AlignedFree/GetStackTrace/fprintf`, `write dup fdopen fclose fflush memcpy`, and `__atomic_fetch_add_4/_8`, `__atomic_load_4` (no A extension). `fw/shim.mojo` implements all of them in Mojo (bump heap, UART stdout, single-core atomics). `stdlib_demo` then prints with `print`, builds a `List`, formats a `String` and catches an `Error` on VexRiscv. Cost: +12 KiB text |
| Startup / no libc | **works** | crt0 is Mojo inline asm (sets sp, clears MIE, copies .data, zeroes .bss, calls `fw_main`, then `fw_exit`). No C file and no separate assembler |
| `-O0` builds | **works** | `matmul_test` at `-O0` needs only `_heap_start` |
| `@export` in an imported module | **gotcha** | Only reachable code from imported modules is emitted. `rt.mojo`'s exports (crt0, `memset`, `fw_exit`) disappear when it is merely imported, so the runtime must be compiled as its own object (note 1) |
| Disassembly with Apple's LLVM tools | n/a | Apple `llvm-objdump` has no RISC-V target, but Apple `llvm-nm` reads the ELF symbols. Use the `fpga/soc` env's LLVM 23 tools |

Note 1: `mojo build --emit object` only emits a module's `@export` functions
when that module is the root of the build. This is why `rt.o`, `shim.o` and
`app.o` are three separate builds.

Not probed: atomics on a core with the A extension, `mojo debug`/DWARF on the
target, interrupts and trap handlers. Firmware polls, and crt0 clears
`mstatus.MIE`. A trap handler would need `mtvec` plus an asm entry, which the
same inline-asm technique should cover.

## Results

`check.sh` passes end to end. Exit codes are asserted, and diffcheck
checksums must equal the host's.

**Differential check** (same Mojo source compiled three ways):

| Target | `checksum(1)` | `checksum(1234)` |
| --- | --- | --- |
| host Mojo (arm64) | 0xe675a896 | 0x3535acdc |
| rv32sim | 0xe675a896 | 0x3535acdc |
| VexRiscv RTL (Verilator) | 0xe675a896 | 0x3535acdc |

These checksums cover 200 rounds of mul/mulh/mulhu/divu/remu, signed floor
div/mod (through compiler-rt), shifts and compares; an 8x8 int8 matmul on
stack arrays; and SIMD int16.

**Tile v1 on the VexRiscv Verilator SoC** (`matmul_test.bin`; cycles from the
SoC's SimCtrl counter at sys_clk):

```
tile ID = 0x71110004
matmul 4x1x4: PASS  soc cycles: tile-driver=876 cpu-ref=603 tile-internal=206
matmul 4x16x4: PASS  soc cycles: tile-driver=2221 cpu-ref=3243 tile-internal=1572
matmul 6x9x10: PASS  soc cycles: tile-driver=5878 cpu-ref=7013 tile-internal=4237
matmul 16x32x16: PASS  soc cycles: tile-driver=50367 cpu-ref=94079 tile-internal=45464
ALL PASS
SIM_FINISH 0
```

The tile is 1.9x faster than the CPU at 16x32x16. The run is feed-bound:
v1 takes one 4+4-byte beat per pair of bus writes, so about 98 CPU cycles go
into each beat (packing plus two uncached stores). "tile-internal" counts
from start to the last output row, so it includes that feed time.

The self-check catches errors: `rv32sim matmul_test.elf --tile-fault` makes
the model corrupt one output word, and the firmware reports `6x9x10: FAIL`,
`FAILURES`, `SIM_FINISH 1`.

**Tile v2 on the VexRiscv Verilator SoC** (`tile2_test.bin`, RTL `tile2_n8`
at 0x90000000; SoC cycles):

```
ID = 0x54494c32 INFO = 0x0f0f0e08
tile2 8x16x8: PASS  cycles: driver=6627 tile-busy=42 cpu-ref=12403
tile2 5x7x3: PASS  cycles: driver=2819 tile-busy=33 cpu-ref=1563
tile2 16x64x24: PASS  cycles: driver=50199 tile-busy=530 cpu-ref=276075
tile2 8x1024x32: PASS  cycles: driver=878478 tile-busy=4194 cpu-ref=8337878
tile2 72x600x80: PASS  cycles: driver=2803882 tile-busy=60492 cpu-ref=109984810
ALL PASS
SIM_FINISH 0
```

End to end, including operand upload, the tile is 39x faster than the CPU at
72x600x80 and 9.5x faster at 8x1024x32. The RTL's busy cycles equal those of
lane-tile2's model, and the contract-rev-1 formula, exactly; for example
60492 at 72x600x80. lane-soc also linked `libtile.o` into C firmware
(`fpga/soc/fw/mojolink`): C calls the Mojo tile v1, tile v2 and CPU functions
on 12x40x10 with 0 mismatches.

**Tile v2 on rv32sim** against lane-tile2's bus model
(`fpga/systolic/tile2.mojo`, N=8, INFO 0x0F0F0E08). Cycles here are retired
instructions because the ISS is functional; the model's BUSY/CYCLES timing is
exact per INTERFACE.md:

```
tile2 8x16x8: PASS  cycles: driver=3601 tile-busy=42 cpu-ref=7898
tile2 5x7x3: PASS  cycles: driver=1278 tile-busy=33 cpu-ref=945
tile2 16x64x24: PASS  cycles: driver=30807 tile-busy=530 cpu-ref=176042
tile2 8x1024x32: PASS  cycles: driver=454250 tile-busy=4194 cpu-ref=1837658
tile2 72x600x80: PASS  cycles: driver=1523737 tile-busy=60492 cpu-ref=24250330   # M/N blocking + 3-way K split (ACCUM)
```

The driver's time is dominated by moving operands over the 32-bit bus, not by
the tile: tile-busy is about 4% of driver time at 72x600x80. Next steps for
speed are DMA, or keeping B resident in the tile and streaming only A.

Other runs (all on both rv32sim and Verilator):

- `hello`: UART output.
- `stdlib_demo`: `print`, `List`, `String`, try/except.
- `softfloat_demo`: signed `//`, Float32, UInt64 `//`.

Sizes (text bytes): hello 2.4 KiB, matmul_test 8.3 KiB, tile2_test 16.5 KiB,
stdlib_demo 14.7 KiB.

## rv32sim (Mojo ISS)

`sim/rv32sim.mojo` is about 380 lines covering RV32IM, plus Zicsr reads of
cycle/instret (other CSRs read 0; writes are ignored), `fence` as a no-op,
and `ecall`/`ebreak`/`mret`/`wfi` as halt. It loads ELF32 PT_LOAD segments by
physical address and starts at `e_entry`. Memory map: LiteX
rom/sram/main_ram, the uart rxtx/txfull/txempty CSRs, and simctrl
finish/latch/cycles. The simctrl cycle counter is the retired-instruction
count.

Devices:

- tile v1 at 0x82000000: `sim/tile_model.mojo`, bit-exact results with a
  nominal CYCLES value.
- tile v2 at 0x90000000: lane-tile2's `Tile2Model`, imported unmodified and
  clocked at one cycle per instruction. Byte and half-word accesses map to
  Wishbone `sel` lanes.

Speed is about 100 MIPS on the M5: 29.4 M instructions take 0.27 s. The ISS
is functional and not cycle-accurate; use the Verilator SoC for timing.

## Status of the hand-off

- **lane-soc**: done. Mojo firmware for tile v1 and tile v2 runs in their
  Vsim through `fpga/soc/run.py`. `check.sh` runs all six firmware images on
  both rv32sim and Verilator. `libtile.o` is linked into their C app
  `mojolink`.
- **lane-tile2**: the register map and the `Tile2Model.access()` API are
  frozen at contract v2.0 rev 1. They will notify this lane before any change.
