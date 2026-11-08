"""C-ABI driver library for the phantom NPU: `mojo build --emit shared-lib`.

    int64_t phantom_open(int64_t tile_n);            // handle > 0, or < 0 error
    int64_t phantom_close(int64_t h);
    uint64_t phantom_alloc(int64_t h, int64_t bytes);  // device address, 0 = error
    int64_t phantom_free(int64_t h, uint64_t addr);
    int64_t phantom_memcpy_h2d(int64_t h, uint64_t dst, const void *src, int64_t n);
    int64_t phantom_memcpy_d2h(int64_t h, void *dst, uint64_t src, int64_t n);
    int64_t phantom_launch_matmul_i8(int64_t h, uint64_t a, uint64_t b, uint64_t c,
                                     int64_t m, int64_t k, int64_t n);  // cycles or < 0
    int64_t phantom_cycles(int64_t h);         // cycles of the last launch
    int64_t phantom_total_cycles(int64_t h);
    int64_t phantom_launches(int64_t h);
    int64_t phantom_set_mode(int64_t h, int64_t mode);  // 0 cycle-accurate, 1 functional
    int64_t phantom_get_mode(int64_t h);
    int64_t phantom_tile_n(int64_t h);
    int64_t phantom_last_error(int64_t h, char *buf, int64_t buflen);  // bytes written

Host pointers cross the boundary as `intptr_t`-sized integers. Every call is
synchronous: when it returns, the simulated work is complete.
"""

from std.memory import alloc, Layout, OwnedPointer, unsafe_memcpy
from phantom.device import (
    PhantomDevice,
    PHANTOM_OK,
    PHANTOM_ERR_ARG,
    PHANTOM_ERR_ADDR,
    PHANTOM_ERR_SIM,
)


def _dev(h: Int64) -> MutPointer[PhantomDevice, MutUntrackedOrigin]:
    return MutPointer[PhantomDevice, MutUntrackedOrigin](
        unsafe_from_address=Int(h)
    )


@export
def phantom_open(tile_n: Int64) abi("C") -> Int64:
    try:
        var dev = PhantomDevice(Int(tile_n))
        var p = alloc(Layout[PhantomDevice](length=1)).unsafe_leak()
        p.unsafe_write(dev^)
        return Int64(Int(p))
    except:
        return PHANTOM_ERR_ARG


@export
def phantom_close(h: Int64) abi("C") -> Int64:
    if h <= 0:
        return PHANTOM_ERR_ARG
    var owned = OwnedPointer[PhantomDevice](unsafe_from_raw_pointer=_dev(h))
    _ = owned^
    return PHANTOM_OK


@export
def phantom_alloc(h: Int64, nbytes: Int64) abi("C") -> UInt64:
    var d = _dev(h)
    try:
        return d[].alloc(Int(nbytes))
    except e:
        d[].last_error = String(e)
        return 0


@export
def phantom_free(h: Int64, addr: UInt64) abi("C") -> Int64:
    var d = _dev(h)
    try:
        d[].free(addr)
        return PHANTOM_OK
    except e:
        d[].last_error = String(e)
        return PHANTOM_ERR_ADDR


@export
def phantom_memcpy_h2d(
    h: Int64, dst: UInt64, src: Int, nbytes: Int64
) abi("C") -> Int64:
    var d = _dev(h)
    try:
        d[].copy_h2d(
            dst, ImmPointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=src),
            Int(nbytes),
        )
        return PHANTOM_OK
    except e:
        d[].last_error = String(e)
        return PHANTOM_ERR_ADDR


@export
def phantom_memcpy_d2h(
    h: Int64, dst: Int, src: UInt64, nbytes: Int64
) abi("C") -> Int64:
    var d = _dev(h)
    try:
        d[].copy_d2h(
            MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=dst),
            src,
            Int(nbytes),
        )
        return PHANTOM_OK
    except e:
        d[].last_error = String(e)
        return PHANTOM_ERR_ADDR


@export
def phantom_launch_matmul_i8(
    h: Int64, a: UInt64, b: UInt64, c: UInt64, m: Int64, k: Int64, n: Int64
) abi("C") -> Int64:
    var d = _dev(h)
    try:
        return Int64(d[].launch_matmul_i8(a, b, c, Int(m), Int(k), Int(n)))
    except e:
        d[].last_error = String(e)
        return PHANTOM_ERR_SIM


@export
def phantom_cycles(h: Int64) abi("C") -> Int64:
    return Int64(_dev(h)[].last_cycles)


@export
def phantom_total_cycles(h: Int64) abi("C") -> Int64:
    return Int64(_dev(h)[].total_cycles)


@export
def phantom_launches(h: Int64) abi("C") -> Int64:
    return Int64(_dev(h)[].launches)


@export
def phantom_set_mode(h: Int64, mode: Int64) abi("C") -> Int64:
    """0 = cycle-accurate (default), 1 = functional + formula cycles."""
    var d = _dev(h)
    if mode != 0 and mode != 1:
        d[].last_error = String(t"bad mode {mode}; 0 = cycle-accurate, 1 = functional")
        return PHANTOM_ERR_ARG
    d[].mode = Int(mode)
    return PHANTOM_OK


@export
def phantom_get_mode(h: Int64) abi("C") -> Int64:
    return Int64(_dev(h)[].mode)


@export
def phantom_tile_n(h: Int64) abi("C") -> Int64:
    return Int64(_dev(h)[].tile_n)


@export
def phantom_last_error(h: Int64, buf: Int, buflen: Int64) abi("C") -> Int64:
    ref msg = _dev(h)[].last_error
    var n = min(msg.byte_length(), Int(buflen) - 1)
    if n < 0:
        return 0
    var dst = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=buf)
    unsafe_memcpy(dest=dst, src=msg.unsafe_ptr(), count=n)
    dst[unsafe_offset=n] = 0
    return Int64(n)
