"""Python extension module `phantom_py`: the phantom NPU from Python.

Build (from fpga/): see phantom/build.sh. Then, with phantom/build on
sys.path:

    import phantom_py
    dev = phantom_py.Device(8)                   # 8x8 tile
    c, cycles = dev.matmul_i8(a, b, m, k, n)     # a, b: bytes or int lists
    phantom_py.device_kind()                     # "other(phantom, accelerator=True)"
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from phantom.context import PhantomContext
from phantom.target import PhantomTarget, device_kind_string


struct Device(Defaultable, Movable, Writable):
    var ctx: Optional[PhantomContext]

    def __init__(out self):
        self.ctx = None

    def write_to(self, mut writer: Some[Writer]):
        if self.ctx:
            writer.write(self.ctx.value().name())
        else:
            writer.write("phantom-npu (closed)")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("Device(")
        self.write_to(writer)
        writer.write(")")

    @staticmethod
    def py_init(out self: Device, args: PythonObject, kwargs: PythonObject) raises:
        self = Device()
        var tile_n = Int(py=args[0]) if len(args) > 0 else 8
        self.ctx = PhantomContext(tile_n)

    @staticmethod
    def name(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        return PythonObject(self_ptr[].ctx.value().name())

    @staticmethod
    def tile_n(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        return PythonObject(self_ptr[].ctx.value().tile_n())

    @staticmethod
    def total_cycles(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        return PythonObject(self_ptr[].ctx.value().total_cycles())

    @staticmethod
    def launches(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        return PythonObject(self_ptr[].ctx.value().launches())

    @staticmethod
    def matmul_i8(
        self_ptr: Pointer[Self, MutAnyOrigin],
        a: PythonObject,
        b: PythonObject,
        m: PythonObject,
        k: PythonObject,
        n: PythonObject,
    ) raises -> PythonObject:
        """C = A @ B on the tile. `a`/`b` are row-major int8 sequences (bytes
        are read as two's complement). Returns (list[int] C, cycles)."""
        ref ctx = self_ptr[].ctx.value()
        var M = Int(py=m)
        var K = Int(py=k)
        var N = Int(py=n)
        var ha = _to_i8(a, M * K, "a")
        var hb = _to_i8(b, K * N, "b")
        var da = ctx.enqueue_create_buffer[DType.int8](M * K)
        var db = ctx.enqueue_create_buffer[DType.int8](K * N)
        var dc = ctx.enqueue_create_buffer[DType.int32](M * N)
        var c = List[Int32]()
        var cycles = 0
        try:
            ctx.enqueue_copy(da, ha)
            ctx.enqueue_copy(db, hb)
            cycles = ctx.enqueue_matmul_i8(da, db, dc, M, K, N)
            ctx.enqueue_copy(c, dc)
        finally:
            ctx.enqueue_free(da)
            ctx.enqueue_free(db)
            ctx.enqueue_free(dc)
        var out = Python.list()
        for v in c:
            out.append(Int(v))
        return Python.tuple(out, cycles)


def _to_i8(obj: PythonObject, want: Int, what: StaticString) raises -> List[Int8]:
    var n = Int(py=Python.import_module("builtins").len(obj))
    if n != want:
        raise Error(t"{what}: expected {want} elements, got {n}")
    var out = List[Int8](capacity=n)
    for v in obj:
        var x = Int(py=v)
        if x < -128 or x > 255:
            raise Error(t"{what}: value {x} is not an int8")
        out.append(Int8(x - 256 if x > 127 else x))  # bytes 128..255 -> negative
    return out^


def device_kind() raises -> PythonObject:
    return PythonObject(device_kind_string[PhantomTarget]())


@export
def PyInit_phantom_py() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("phantom_py")
        m.def_function[device_kind]("device_kind")
        _ = (
            m.add_type[Device]("Device")
            .def_py_init[Device.py_init]()
            .def_method[Device.name]("name")
            .def_method[Device.tile_n]("tile_n")
            .def_method[Device.total_cycles]("total_cycles")
            .def_method[Device.launches]("launches")
            .def_method[Device.matmul_i8]("matmul_i8")
        )
        return m.finalize()
    except e:
        abort(String("failed to create module phantom_py: ", e))
