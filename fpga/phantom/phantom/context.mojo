"""`DeviceContext`-style host API for the phantom NPU.

The context talks to the device only through the C ABI in
`include/phantom.h`, implemented today by `libphantom.dylib` (the simulator).
A real FPGA driver that exports the same symbols can replace it without
changing callers; set `PHANTOM_LIB` to choose the library.

Stream semantics: one in-order stream per context. Every `enqueue_*` call runs
eagerly and has completed when it returns, so `synchronize()` is a no-op kept
for API parity with `DeviceContext`. Errors raise with the driver's message.
"""

from std.ffi import OwnedDLHandle
from std.os import getenv
from std.sys import size_of


def default_lib_path() -> String:
    """`$PHANTOM_LIB`, else `phantom/build/libphantom.dylib` (run from fpga/)."""
    var p = getenv("PHANTOM_LIB")
    if p.byte_length() > 0:
        return p
    return "phantom/build/libphantom.dylib"


struct PhantomBuffer[dtype: DType](ImplicitlyCopyable, Writable):
    """A device allocation of `length` elements of `dtype`.

    It is a plain handle (like a device pointer): free it with
    `PhantomContext.enqueue_free`, or let the context release everything when
    it is destroyed.
    """

    var addr: UInt64
    var length: Int

    def __init__(out self, addr: UInt64, length: Int):
        self.addr = addr
        self.length = length

    def byte_length(self) -> Int:
        return self.length * size_of[Self.dtype]()

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            t"PhantomBuffer[{Self.dtype}](addr={hex(self.addr)}, len={self.length})"
        )


struct PhantomContext(Movable):
    """Handle to one phantom NPU (an N x N int8 systolic tile)."""

    var _lib: OwnedDLHandle
    var _h: Int64

    def __init__(out self, tile_n: Int = 8, lib_path: String = "") raises:
        """Opens a device. `lib_path` defaults to `default_lib_path()`."""
        self._lib = OwnedDLHandle(
            lib_path if lib_path.byte_length() > 0 else default_lib_path()
        )
        self._h = self._lib.call["phantom_open", Int64](Int64(tile_n))
        if self._h <= 0:
            raise Error(
                t"phantom_open(tile_n={tile_n}) failed: tile_n must be one of 1, 2, 4, 8, 16"
            )

    def __deinit__(deinit self):
        _ = self._lib.call["phantom_close", Int64](self._h)

    # ---- device info -------------------------------------------------------

    def name(self) -> String:
        return String(t"phantom-npu (simulated {self.tile_n()}x{self.tile_n()} int8 systolic tile)")

    def api(self) -> String:
        return "phantom"

    def set_mode(self, functional: Bool) raises:
        """`False` (default): cycle-accurate simulation, cycles measured.
        `True`: functional int32 matmul (same results), cycles from the formula.
        """
        self._check(
            self._lib.call["phantom_set_mode", Int64](self._h, Int64(Int(functional))),
            "set_mode",
        )

    def is_functional(self) -> Bool:
        return self._lib.call["phantom_get_mode", Int64](self._h) == 1

    def tile_n(self) -> Int:
        return Int(self._lib.call["phantom_tile_n", Int64](self._h))

    def last_cycles(self) -> Int:
        """Simulated tile cycles of the most recent launch."""
        return Int(self._lib.call["phantom_cycles", Int64](self._h))

    def total_cycles(self) -> Int:
        return Int(self._lib.call["phantom_total_cycles", Int64](self._h))

    def launches(self) -> Int:
        return Int(self._lib.call["phantom_launches", Int64](self._h))

    def _error(self, what: String) -> Error:
        var buf = List[UInt8](length=512, fill=0)
        var n = self._lib.call["phantom_last_error", Int64](
            self._h, Int(buf.unsafe_ptr()), Int64(len(buf))
        )
        var msg = String(unsafe_from_utf8=Span(buf)[: Int(max(n, 0))])
        return Error(t"{what}: {msg}")

    def _check(self, rc: Int64, what: String) raises:
        if rc < 0:
            raise self._error(what)

    # ---- memory ------------------------------------------------------------

    def enqueue_create_buffer[
        dtype: DType
    ](self, length: Int) raises -> PhantomBuffer[dtype]:
        var addr = self._lib.call["phantom_alloc", UInt64](
            self._h, Int64(length * size_of[dtype]())
        )
        if addr == 0:
            raise self._error(String(t"alloc {length} x {dtype}"))
        return PhantomBuffer[dtype](addr, length)

    def enqueue_free[dtype: DType](self, buf: PhantomBuffer[dtype]) raises:
        self._check(
            self._lib.call["phantom_free", Int64](self._h, buf.addr), "free"
        )

    def enqueue_copy[
        dtype: DType
    ](
        self, dst: PhantomBuffer[dtype], src: ImmPointer[Scalar[dtype], _]
    ) raises:
        """Host -> device, `dst.length` elements."""
        self._check(
            self._lib.call["phantom_memcpy_h2d", Int64](
                self._h, dst.addr, Int(src), Int64(dst.byte_length())
            ),
            "copy host->device",
        )

    def enqueue_copy[
        dtype: DType
    ](
        self, dst: MutPointer[Scalar[dtype], _], src: PhantomBuffer[dtype]
    ) raises:
        """Device -> host, `src.length` elements."""
        self._check(
            self._lib.call["phantom_memcpy_d2h", Int64](
                self._h, Int(dst), src.addr, Int64(src.byte_length())
            ),
            "copy device->host",
        )

    def enqueue_copy[
        dtype: DType
    ](self, dst: PhantomBuffer[dtype], src: List[Scalar[dtype]]) raises:
        """Host list -> device. Lengths must match."""
        if len(src) != dst.length:
            raise Error(t"copy: list has {len(src)} elements, buffer {dst.length}")
        self.enqueue_copy(dst, src.unsafe_ptr())

    def enqueue_copy[
        dtype: DType
    ](self, mut dst: List[Scalar[dtype]], src: PhantomBuffer[dtype]) raises:
        """Device -> host list (resized to the buffer length)."""
        dst.resize(src.length, 0)
        self.enqueue_copy(dst.unsafe_ptr(), src)

    # ---- kernels -----------------------------------------------------------

    def enqueue_matmul_i8(
        self,
        a: PhantomBuffer[DType.int8],
        b: PhantomBuffer[DType.int8],
        c: PhantomBuffer[DType.int32],
        m: Int,
        k: Int,
        n: Int,
    ) raises -> Int:
        """C[m,n] = A[m,k] @ B[k,n] (row-major, int32 wraparound accumulate).

        Returns the simulated tile cycles of this launch.
        """
        var rc = self._lib.call["phantom_launch_matmul_i8", Int64](
            self._h, a.addr, b.addr, c.addr, Int64(m), Int64(k), Int64(n)
        )
        self._check(rc, String(t"matmul_i8 M={m} K={k} N={n}"))
        return Int(rc)

    def synchronize(self):
        """No-op: the stream is eager and in-order."""
        pass
