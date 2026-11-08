# Phantom NPU (G1-C)

This directory makes the invented int8 systolic tile (`../DESIGN.md`) appear to Mojo as an accelerator. The cycle-accurate simulator in `../systolic` backs it. Every launch steps the register model, and the cycle count each launch returns is measured, not computed. The tests check it against the DESIGN.md formula `ceil(M/N)·ceil(N_out/N)·(K+3N−1)`.

## Result

| Route | Status | What works |
| --- | --- | --- |
| Stdlib target plugin (pure Mojo) | **works** | `phantom.target.Phantom` reports `TargetDeviceKind.other("phantom", True)`, `is_accelerator() == True` and `is_gpu() == False`. |
| HAL driver plugin (C, Modular driver ABI 0.1.0) | **works** | MAX's own NeoDriver client (`max._driver`) loads `libphantom_hal.dylib` through `MODULAR_DRIVER_PLUGINS`. It enumerates one `phantom` accelerator, and allocation, copy, bundle/function load, execute and queue properties run a bit-exact matmul on the simulator. |
| Compiler codegen plugin (C++) | **partially feasible; blocked** | The shipped `mojo` dlopens our plugin and calls `M_KGEN_getTargetTraits`. Codegen for a new triple also needs `TargetLowering`/`TargetBackend`, which only a same-commit source build of the compiler can provide. See "Compiler plugin". |
| FFI fallback (Mojo + C ABI) | **works**, the main route | `libphantom.dylib` (C ABI), the `phantom.context.PhantomContext` DeviceContext-style API and the `phantom_py` Python module. |

`fpga/results/phantom.json` records `route: "ffi-fallback"` because the shipped compiler cannot generate code for a phantom triple. The HAL plugin runs on top of the same C ABI.

## Commands (from `fpga/`)

```
pixi run phantom-build         # libphantom.dylib, phantom_py.so, libphantom_hal.dylib, test_capi
pixi run phantom-test          # C, Mojo, Python and HAL suites
pixi run phantom-demo          # device_kind + a 64x512x64 int8 matmul
pixi run phantom-fetch-src     # sparse toolchain sources into third_party/ (needed by HAL build + spike)
pixi run phantom-plugin-spike  # compiler-plugin feasibility evidence (exits 0 on the expected failure)
```

The HAL suite runs in the `../mojo-max` environment, because it needs the `max` package. `test.sh` skips it when that environment is absent.

## Test results (2026-11-08, mojo 1.3.0.dev2026110805, M-series Mac)

```
== C ABI         capi: 14/14 passed    (12 random shapes at N=4,8 + 2 bounds checks)
== Mojo API      phantom: 6/6 passed   (30 random shapes over N=1,2,4,8,16; edges incl.
                                        K=2000 all -128; two independent devices; errors;
                                        functional mode)
== Python        python: 12/12 passed  (10 random shapes, bytes as int8, length error)
== HAL plugin    hal: 5/5 passed       (through max._driver NeoDriver, mojo-max env)
phantom-test: all suites passed
```

Every matmul is compared element by element with `systolic.reference.reference_matmul` (or the C/Python equivalent). In every case `cycles == predicted_cycles`. The existing `pixi run test` still passes (24/24).

Demo (`pixi run phantom-demo`):

```
target device_kind: other(phantom, accelerator=True)
  is_accelerator: True  is_gpu: False
host device_kind:   cpu
device: phantom-npu (simulated 8x8 int8 systolic tile) | api: phantom
matmul_i8 M=64 K=512 N=64: bit_exact=True
  cycles=34240 (formula 34240), PE utilization 95.70093457943926%
  simulated @100 MHz: 342.4 us, 12.249719626168224 GOP/s
```

The GOP/s figure is simulated tile throughput at the nominal 100 MHz used by `tools/perf.mojo`. It is not host wall time, and no hardware is involved.

## API (for G1-D)

```mojo
from phantom.context import PhantomContext, PhantomBuffer
from phantom.target import Phantom, PhantomTarget, device_kind_string

var ctx = PhantomContext(8)                              # N=8 tile; PHANTOM_LIB picks the .dylib
var a = ctx.enqueue_create_buffer[DType.int8](M * K)     # PhantomBuffer[dtype]{addr, length}
var b = ctx.enqueue_create_buffer[DType.int8](K * N)
var c = ctx.enqueue_create_buffer[DType.int32](M * N)
ctx.enqueue_copy(a, host_a)                              # List or ImmPointer -> device
ctx.enqueue_copy(b, host_b)
var cycles = ctx.enqueue_matmul_i8(a, b, c, M, K, N)     # row-major, int32 wraparound
ctx.enqueue_copy(host_c, c)                              # device -> List (resized) or MutPointer
ctx.synchronize()                                        # no-op: eager, in-order stream
ctx.last_cycles(); ctx.total_cycles(); ctx.launches(); ctx.tile_n()
ctx.set_mode(functional=True)                            # plain matmul + formula cycles (default: cycle-accurate)
ctx.enqueue_free(a)                                      # or let the context close everything
```

- C ABI (`include/phantom.h`): `phantom_open/close/alloc/free/memcpy_h2d/memcpy_d2h/launch_matmul_i8/cycles/total_cycles/launches/tile_n/last_error/set_mode/get_mode`.
  - `phantom_set_mode(h, 1)` switches to functional mode: a plain int32 matmul with formula cycles. Both modes give identical results and cycles. Cycle-accurate is already fast: a 128x3072x256 matmul at N=8 takes 0.14 s, against 0.16 s functional, measured via ctypes.
  - Handles are `int64`. Device addresses are `uint64` of the form `(alloc_id+1)<<32 | offset`, so addresses can be offset like pointers.
  - Negative return values are errors, and `phantom_last_error` returns the message.
  - A real FPGA driver that exports the same symbols can replace the simulator without changing callers.
- Python: `phantom_py.Device(n).matmul_i8(a, b, m, k, n) -> (list[int], cycles)`, plus `phantom_py.device_kind()`. Build with `phantom-build`, then put `phantom/build` on `sys.path`.
- HAL: `MODULAR_DRIVER_PLUGINS=phantom@<abs>/phantom/build/libphantom_hal.dylib`.
  - The device reports `name` "phantom-npu (simulated NxN int8 systolic tile)" and arch `phantom-n<N>`. Set N with `PHANTOM_TILE_N`; the default is 8.
  - `bundle_load` accepts any bundle. The only function is `phantom_matmul_i8`.
  - It takes 6 args of 8 bytes each: device addresses `a, b, c` from `unwrap_memory`, then `m, k, n`.
  - Queue property `phantom_last_cycles` returns the cycles. See `tests/test_hal.mojo`.

Supported tile sizes are N ∈ {1, 2, 4, 8, 16}, which are compile-time instantiations of `run_matmul`. The valid K range is 1..65535, set by the `k_len` port.

## Feasibility evidence

### Toolchain source match

`mojo --version` reports `1.3.0.dev2026110805 (11efb73e)`. That hash is internal and does not exist on GitHub. The public commit for that nightly is the release-pin commit `modular@2841b31631e7d31ae74f0ca20335ed1541cddec1`, "[Release] Pin lockfiles to Mojo 1.3.0.dev2026110805, MAX 26.8.0.dev2026110805", dated 2026-11-08. Its `bazel/public-patches/llvm_source.bzl` pins `llvm-project@fa8c67382e4fa68010fce98bbd267f91b397f80e`, and the binary reports LLVM 24.0.0git. `fetch_toolchain_src.sh` makes sparse, blob-less checkouts of exactly those two commits into `third_party/`. That directory is gitignored and marked `.metadata_never_index`; together the checkouts take about 64 MB.

### Stdlib plugin

`target.mojo` uses Modular's stdlib plugin hooks (`std._plugin`). A `#kgen.target` carries `opaque_plugin = TypeList.of[PhantomPlugin]()`. `PhantomHooks.device_kind = TargetDeviceKind.other("phantom", True)`, and `CompilationTarget.device_kind()`, `is_accelerator()` and `is_gpu()` read that value. This needs no compiler change. The triple stays the host's aarch64 triple, because nothing can generate code for a new triple (next section).

### Compiler plugin (`compiler_plugin/`, `spike/`)

- `max/mojo/max/_plugin/README.md` lists four plugin points: stdlib hooks, MAX hooks, the codegen backend (C++), and the driver (C HAL).
- `Mojo/lib/Target/PluginUtils.cpp` dlopens each library in `MODULAR_COMPILER_PLUGINS` (`;`-separated) or modular.cfg `mojo_plugin_paths`. It then looks up `M_KGEN_getTargetTraits`, `M_KGEN_getTargetLowering` and `M_KGEN_getTargetBackend`.
- `phantom_traits.cpp` implements `TargetTraits`, a C++ vtable interface. It is built against the pinned headers, with generated `llvm-config.h`/`abi-breaking.h` and two small shim headers.
  - The result exports only `_M_KGEN_getTargetTraits` and depends on nothing but libc/libc++.
  - The shipped `mojo` loads it (from `pixi run phantom-plugin-spike`):

```
[phantom-plugin] loaded (dlopen) into pid 20470
[phantom-plugin] M_KGEN_getTargetTraits() called -> 0x1031b8000
[phantom-plugin] matches("arm64-apple-darwin25.5.0") -> false
[phantom-plugin] matches("phantom-unknown-none") -> true
[phantom-plugin] matches("phantom-unknown-none") -> true
custom_triple.mojo:1:1: error: target 'phantom-unknown-none' is not supported by this build
```

The error now comes from the `TargetLowering`/`TargetBackend` registries (`TargetLowering.cpp:60`, `TargetBackend.cpp:137`). Those interfaces take internal KGEN/MLIR types: `CompilationOptions`, `BufferRef`, `ErrorOr`, `KGENDType`, `TargetInfoAttr`, `DIType`, `GeneratorOp`. Their out-of-line base methods and tablegen'd dialect headers belong to the compiler binary.

- `bin/mojo` exports about 18 symbols.
- `libLLVM`/`libMLIR` export upstream APIs only (about 29k and 132k symbols), with no KGEN.

The upstream README says a backend must be built from the same commit and build configuration as the compiler, and that no reference backend is public. **Verdict:** a working codegen plugin needs a full same-commit Bazel build of the compiler. That build was not started (see "Full build request").

### HAL driver plugin (`hal_plugin/phantom_hal.c`)

- Implements all 57 required base symbols in `SDK/include/HAL/M_driver_hal_symbols.inc`, plus the peer-access symbols, for 60 `M_driver_*` exports in total. None are missing.
- The driver, device and memory are trivially synchronous. Events and queues complete immediately.
- `M_driver_destroy` only drops a reference: the simulator is a process-wide singleton, because the NeoDriver client creates and destroys driver handles more than once per process.
- Errors use `M_DRIVER_STATUS_CUSTOM_CODES_BASE` and carry the simulator's message. For example: `bad device address 0x8`, `access [0, 64) outside allocation of 4 bytes`.
- Log with `PHANTOM_HAL_LOG=1`:

```
[phantom-hal] M_driver_create(host interface 0.1.0)
[phantom-hal] opened simulator: 8 x 8 int8 tile
host.driver_count(): 1
driver(0).name(): phantom
host.accelerator_count(): 1
device 0: phantom-npu (simulated 8x8 int8 systolic tile) | arch: phantom-n8
[phantom-hal] bundle_load file_type=phantom, 33 bytes
[phantom-hal] function_load 'phantom_matmul_i8'
[phantom-hal] execute phantom_matmul_i8 M=2 K=32 N=5 (mode GPU) -> 55 cycles
```

Limits: the typed `Device`/`Context`/`Queue` layer of `max._driver` selects HAL hooks from a fixed overlay list (`DefaultHalPlugin`, Metal, CUDA, HIP), and `DeviceContext` compatibility needs a `--config=hal-device-context` source build. The test therefore uses `RawDriver` plus raw symbol calls for device, context and bundle. `std.gpu.host.DeviceContext(api="phantom")` is not reachable without a source build.

### Symbols

```
libphantom.dylib: _phantom_alloc _phantom_close _phantom_cycles _phantom_free _phantom_last_error
                  _phantom_launch_matmul_i8 _phantom_launches _phantom_memcpy_d2h _phantom_memcpy_h2d
                  _phantom_open _phantom_tile_n _phantom_total_cycles
                  deps: @rpath/libKGENCompilerRTShared.dylib (Mojo runtime), libSystem
libphantom_hal.dylib: 60 x _M_driver_* ; deps: @rpath/libphantom.dylib, libSystem
libphantom_compiler_plugin.dylib: _M_KGEN_getTargetTraits ; deps: libc++/libSystem only
```

## Files

| Path | What |
| --- | --- |
| `phantom/target.mojo` | `PhantomTarget`, `Phantom`, stdlib plugin hooks, `device_kind_string` |
| `phantom/device.mojo` | device model: allocations, copies, `launch_matmul_i8` on `systolic.run_matmul` |
| `phantom/context.mojo` | `PhantomContext`/`PhantomBuffer` over the C ABI (`OwnedDLHandle`) |
| `capi.mojo`, `include/phantom.h` | C ABI exported from Mojo (`libphantom.dylib`) |
| `phantom_py.mojo` | Python extension module `phantom_py` |
| `hal_plugin/phantom_hal.c` | Modular HAL driver plugin |
| `compiler_plugin/` | `TargetTraits` plugin, build and spike scripts |
| `spike/` | stdlib-plugin and custom-triple spikes |
| `tests/` | `test_capi.c`, `test_phantom.mojo`, `test_python.py`, `test_hal.mojo` |
| `demo.mojo`, `build.sh`, `test.sh`, `fetch_toolchain_src.sh` | entry points |

## Licenses

- Code in this directory is project code.
- Only the gitignored `third_party/` checkout supplies Modular sources, and nothing is copied out of it. Builds include headers from it:
  - Apache-2.0 WITH LLVM-exception: `Mojo/lib/Target/*`, `PluginUtils.cpp` and the stdlib `std/_plugin` files. These are the headers the compiler-plugin spike compiles against, and the stdlib plugin API that `target.mojo` imports from the installed stdlib. llvm-project headers are also Apache-2.0 WITH LLVM-exception.
  - MAX License (Modular's community license, not OSI): `SDK/include/HAL/*`, `Target/include/Target/DriverPlugin.h`, `MLRT/test/Driver/HAL/*` and `max/_driver` (`max_license` dirs).
- `phantom_hal.c` includes the MAX-licensed HAL headers. `tests/test_hal.mojo` imports `max._driver.max_license.hal` from the installed `max` package.
- Anyone who redistributes `libphantom_hal.dylib` must follow the MAX License terms. The FFI route (`libphantom.dylib`, `phantom.context`, `phantom_py`) uses only the Mojo stdlib, which is Apache-2.0 WITH LLVM-exception.

## Full build request (not started)

A real codegen plugin, meaning Mojo kernels compiled *for* `phantom-unknown-none`, needs a same-commit build of the compiler at `modular@2841b316` with `./bazelw`. This is an estimate, not measured: about 2–4 h wall time on all cores of an M-series Mac (LLVM/MLIR + KGEN), and about 40–80 GB of disk for the Bazel output base and caches. Before committing to the build, a cheap first step is `./bazelw query` on the compiler target, which confirms that every source it needs is in the public tree. The build needs explicit approval.
