"""Drives the phantom NPU through Modular's HAL driver ABI.

Runs in the `../mojo-max` pixi env (it needs the `max` package, whose
NeoDriver client `max._driver` dlopens HAL plugins). Requires
MODULAR_DRIVER_PLUGINS=phantom@<path>/libphantom_hal.dylib.
"""

from std.os import getenv
from std.memory import MaybeUninit
from std.testing import assert_equal, assert_true

from max._driver.max_license.hal.host import Host
from max._driver.max_license.hal.plugin import RawDriver
from max._driver.max_license.hal.plugins import (
    ContextHandle,
    DeviceHandle,
    M_driver_bundle_compilation_options,
    M_driver_memory_view,
    M_driver_slice,
    M_driver_static_bundle,
    OutParam,
    RuntimeBundleHandle,
)


def xorshift(mut s: UInt64) -> UInt64:
    s ^= s >> 12
    s ^= s << 25
    s ^= s >> 27
    return s * 0x2545F4914F6CDD1D


def main() raises:
    var spec = getenv("MODULAR_DRIVER_PLUGINS")
    assert_true(spec.byte_length() > 0, "set MODULAR_DRIVER_PLUGINS")

    # 1. MAX's NeoDriver client enumerates the phantom device.
    var host = Host()
    print("host.driver_count():", host.driver_count())
    print("driver(0).name():", host.driver(0)[].name())
    print("host.accelerator_count():", host.accelerator_count())

    # 2. Raw HAL calls through the same loaded-plugin machinery.
    var raw = RawDriver.load(spec)
    print("api_name():", raw.api_name())
    var dev_u = MaybeUninit[DeviceHandle]()
    _ = raw._raw.base.device_get.get()(
        raw._driver_handle, Int64(0), OutParam[DeviceHandle](to=dev_u)
    )
    var dev = dev_u.unsafe_assume_init()
    var name = List[Int8](length=256, fill=0)
    raw.get_device_property_string["name"](dev, name.unsafe_ptr())
    var arch = List[Int8](length=256, fill=0)
    raw.get_device_property_string["arch"](dev, arch.unsafe_ptr())
    print(
        "device 0:",
        String(unsafe_from_utf8_ptr=name.unsafe_ptr().unsafe_bitcast[UInt8]()),
        "| arch:",
        String(unsafe_from_utf8_ptr=arch.unsafe_ptr().unsafe_bitcast[UInt8]()),
    )
    var ctx_u = MaybeUninit[ContextHandle]()
    _ = raw._raw.base.context_create.get()(
        dev, OutParam[ContextHandle](to=ctx_u)
    )
    var ctx = ctx_u.unsafe_assume_init()
    var queue = raw.create_queue(ctx)

    # Bundle + function: the tile is fixed-function, so the bundle is a
    # manifest and the function is looked up by name.
    var manifest = String("phantom-npu v1: phantom_matmul_i8")
    comptime file_type: StaticString = "phantom"
    var sb = M_driver_static_bundle(
        mapped_data=M_driver_slice(
            data=Pointer(to=manifest.unsafe_ptr()[]),
            size=UInt64(manifest.byte_length()),
        ),
        file_type=Pointer(to=file_type.unsafe_ptr().unsafe_bitcast[Int8]()[]),
        file_type_len=UInt64(file_type.byte_length()),
    )
    var opts = M_driver_bundle_compilation_options(
        debug_level=rebind[ImmPointer[Int8, ImmUntrackedOrigin]]("".ptr()),
        debug_level_len=UInt64(0),
        optimization_level=Int32(-1),
    )
    var rb_u = MaybeUninit[RuntimeBundleHandle]()
    _ = raw._raw.base.bundle_load.get()(
        ctx, Pointer(to=sb), Pointer(to=opts), OutParam[RuntimeBundleHandle](to=rb_u)
    )
    var bundle = rb_u.unsafe_assume_init()
    var fname = String("phantom_matmul_i8")
    var fn_h = raw.load_function(ctx, bundle, fname)

    var tile_n = 8
    var seed: UInt64 = 0xC0FFEE
    var passed = 0
    var total = 0
    for _ in range(5):
        var m = 1 + Int(xorshift(seed) % 20)
        var k = 1 + Int(xorshift(seed) % 48)
        var n = 1 + Int(xorshift(seed) % 20)
        var a = List[Int8](capacity=m * k)
        for _ in range(m * k):
            a.append(Int8(Int(xorshift(seed) % 256) - 128))
        var b = List[Int8](capacity=k * n)
        for _ in range(k * n):
            b.append(Int8(Int(xorshift(seed) % 256) - 128))
        var want = List[Int32](length=m * n, fill=0)
        for i in range(m):
            for j in range(n):
                var s: Int32 = 0
                for kk in range(k):
                    s += Int32(a[i * k + kk]) * Int32(b[kk * n + j])
                want[i * n + j] = s

        var ma = raw.alloc_sync(ctx, UInt64(m * k))
        var mb = raw.alloc_sync(ctx, UInt64(k * n))
        var mc = raw.alloc_sync(ctx, UInt64(m * n * 4))
        raw.copy_to_device_sync(
            ctx, M_driver_memory_view(ma, 0, UInt64(m * k)),
            a.unsafe_ptr().unsafe_bitcast[UInt8](),
        )
        raw.copy_to_device_sync(
            ctx, M_driver_memory_view(mb, 0, UInt64(k * n)),
            b.unsafe_ptr().unsafe_bitcast[UInt8](),
        )
        var argv: List[UInt64] = [
            raw.unwrap_memory(ctx, ma),
            raw.unwrap_memory(ctx, mb),
            raw.unwrap_memory(ctx, mc),
            UInt64(m),
            UInt64(k),
            UInt64(n),
        ]
        var argp = List[OpaquePointer[ImmUntrackedOrigin]]()
        for i in range(6):
            argp.append(
                rebind[OpaquePointer[ImmUntrackedOrigin]](
                    argv.unsafe_ptr().unsafe_offset(i).unsafe_bitcast[NoneType]()
                )
            )
        var sizes: List[UInt64] = [8, 8, 8, 8, 8, 8]
        raw.execute_function(
            queue, fn_h, (UInt32(1), UInt32(1), UInt32(1)),
            (UInt32(1), UInt32(1), UInt32(1)),
            argp.unsafe_ptr(), sizes.unsafe_ptr(), UInt32(6),
        )
        raw.synchronize_queue(queue)
        _ = argv^  # keep the argument storage alive through the launch
        _ = argp^
        _ = sizes^
        var c = List[Int32](length=m * n, fill=0)
        raw.copy_from_device_sync(
            ctx, c.unsafe_ptr().unsafe_bitcast[UInt8](),
            M_driver_memory_view(mc, 0, UInt64(m * n * 4)),
        )
        var cycles = raw.get_queue_property["phantom_last_cycles", Int64](queue)
        var formula = ((m + tile_n - 1) // tile_n) * ((n + tile_n - 1) // tile_n) * (
            k + 3 * tile_n - 1
        )
        var ok = Int(cycles) == formula
        for i in range(m * n):
            if c[i] != want[i]:
                ok = False
        total += 1
        if ok:
            passed += 1
        print(
            "ok  " if ok else "FAIL", t"M={m} K={k} N={n} cycles={cycles} formula={formula}"
        )
        raw.free_sync(ctx, ma)
        raw.free_sync(ctx, mb)
        raw.free_sync(ctx, mc)
    print(t"hal: {passed}/{total} passed")
    assert_equal(passed, total)

