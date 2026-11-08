"""PhantomDevice: the phantom NPU's device model, backed by the simulator.

The "hardware" is a fixed-function int8 matmul engine: one output-stationary
N x N systolic tile (../../DESIGN.md) behind a simulated device memory. Device
memory lives in this struct, not in host pointers: an address is
`(allocation_index + 1) << 32 | byte_offset`, so host code can only reach it
through the copy functions, as with a real accelerator.

A launch reads A (M x K int8, row-major) and B (K x N_out int8, row-major) from
device memory, streams them through the cycle-accurate tile model
(`systolic.driver.run_matmul`, ideal host), and writes C (M x N_out int32,
row-major, little-endian) back. The cycle count is measured by stepping the
register model, never computed.
"""

from std.memory import unsafe_memcpy
from systolic.driver import HostConfig, run_matmul, predicted_cycles
from systolic.tile import K_LEN_MAX
from systolic.trace import TraceWriter, TRACE_OFF

comptime PHANTOM_OK = 0
comptime PHANTOM_ERR_ARG = -1
comptime PHANTOM_ERR_ADDR = -2
comptime PHANTOM_ERR_OOM = -3
comptime PHANTOM_ERR_SIM = -4

comptime MODE_CYCLE_ACCURATE = 0
"""Step the register model; cycles are measured (default)."""
comptime MODE_FUNCTIONAL = 1
"""Plain int32 matmul; cycles from the DESIGN.md formula (ideal host)."""

comptime SUPPORTED_TILES = [1, 2, 4, 8, 16]
comptime _ADDR_SHIFT = 32
comptime MAX_ALLOC_BYTES = 1 << 31


def supported_tile(n: Int) -> Bool:
    comptime for t in SUPPORTED_TILES:
        if n == t:
            return True
    return False


struct PhantomDevice(Movable):
    var tile_n: Int
    var mem: List[List[UInt8]]
    var live: List[Bool]
    var last_cycles: Int
    var total_cycles: Int
    var launches: Int
    var last_error: String
    var mode: Int

    def __init__(out self, tile_n: Int) raises:
        if not supported_tile(tile_n):
            raise Error(t"unsupported tile size N={tile_n}; built for 1,2,4,8,16")
        self.tile_n = tile_n
        self.mem = []
        self.live = []
        self.last_cycles = 0
        self.total_cycles = 0
        self.launches = 0
        self.last_error = String()
        self.mode = MODE_CYCLE_ACCURATE

    # ---------------------------------------------------------------- memory

    def alloc(mut self, nbytes: Int) raises -> UInt64:
        if nbytes <= 0 or nbytes > MAX_ALLOC_BYTES:
            raise Error(t"alloc of {nbytes} bytes out of range")
        self.mem.append(List[UInt8](length=nbytes, fill=0))
        self.live.append(True)
        return UInt64(len(self.mem)) << _ADDR_SHIFT

    def free(mut self, addr: UInt64) raises:
        var idx = self._index(addr)
        if (addr & 0xFFFFFFFF) != 0:
            raise Error("free of an interior address")
        self.live[idx] = False
        self.mem[idx] = []

    def _index(self, addr: UInt64) raises -> Int:
        var idx = Int(addr >> _ADDR_SHIFT) - 1
        if idx < 0 or idx >= len(self.mem) or not self.live[idx]:
            raise Error(t"bad device address {hex(addr)}")
        return idx

    def _check(self, addr: UInt64, nbytes: Int) raises -> Tuple[Int, Int]:
        var idx = self._index(addr)
        var off = Int(addr & 0xFFFFFFFF)
        if nbytes < 0 or off + nbytes > len(self.mem[idx]):
            raise Error(
                t"access [{off}, {off + nbytes}) outside allocation of"
                t" {len(self.mem[idx])} bytes"
            )
        return (idx, off)

    def copy_h2d(
        mut self, dst: UInt64, src: ImmPointer[UInt8, _], nbytes: Int
    ) raises:
        var idx, off = self._check(dst, nbytes)
        unsafe_memcpy(
            dest=self.mem[idx].unsafe_ptr().unsafe_offset(off),
            src=src,
            count=nbytes,
        )

    def copy_d2h(
        self, dst: MutPointer[UInt8, _], src: UInt64, nbytes: Int
    ) raises:
        var idx, off = self._check(src, nbytes)
        unsafe_memcpy(
            dest=dst,
            src=self.mem[idx].unsafe_ptr().unsafe_offset(off),
            count=nbytes,
        )

    # ---------------------------------------------------------------- launch

    def launch_matmul_i8(
        mut self, a: UInt64, b: UInt64, c: UInt64, m: Int, k: Int, n_out: Int
    ) raises -> Int:
        """C[m, n_out] (int32) = A[m, k] (int8) @ B[k, n_out] (int8).

        Returns the simulated cycle count of this launch.
        """
        if m < 1 or n_out < 1 or k < 1 or k > K_LEN_MAX:
            raise Error(t"bad shape M={m} K={k} N={n_out} (1 <= K <= {K_LEN_MAX})")
        var ia, oa = self._check(a, m * k)
        var ib, ob = self._check(b, k * n_out)
        var ic, oc = self._check(c, m * n_out * 4)
        var av = List[Int8](capacity=m * k)
        var pa = self.mem[ia].unsafe_ptr().unsafe_offset(oa)
        for i in range(m * k):
            av.append(pa[unsafe_offset=i].cast[DType.int8]())
        var bv = List[Int8](capacity=k * n_out)
        var pb = self.mem[ib].unsafe_ptr().unsafe_offset(ob)
        for i in range(k * n_out):
            bv.append(pb[unsafe_offset=i].cast[DType.int8]())

        var cv = List[Int32]()
        var trace = TraceWriter(TRACE_OFF)
        var cycles = 0
        if self.mode == MODE_FUNCTIONAL:
            cv = List[Int32](length=m * n_out, fill=0)
            for i in range(m):
                for kk in range(k):
                    var x = Int32(av[i * k + kk])
                    for j in range(n_out):
                        cv[i * n_out + j] += x * Int32(bv[kk * n_out + j])
            cycles = predicted_cycles(self.tile_n, m, k, n_out)
        comptime for t in SUPPORTED_TILES:
            if self.mode == MODE_CYCLE_ACCURATE and self.tile_n == t:
                var st = run_matmul[t](
                    av, bv, m, k, n_out, HostConfig.ideal(), trace, cv
                )
                cycles = st.cycles
        unsafe_memcpy(
            dest=self.mem[ic].unsafe_ptr().unsafe_offset(oc),
            src=cv.unsafe_ptr().unsafe_bitcast[UInt8](),
            count=m * n_out * 4,
        )
        self.last_cycles = cycles
        self.total_cycles += cycles
        self.launches += 1
        return cycles
