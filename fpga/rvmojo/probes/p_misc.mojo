from std.sys._assembly import inlined_assembly
from std.sys import size_of

@fieldwise_init
struct Vec2(TrivialRegisterPassable):
    var x: Int32
    var y: Int32

    def dot(self, o: Vec2) -> Int32:
        return self.x * o.x + self.y * o.y

@export
def struct_dot(a: Int32, b: Int32) abi("C") -> Int32:
    var v = Vec2(a, b)
    return v.dot(Vec2(b, a))

def scale[N: Int](x: Int32) -> Int32:
    return x * Int32(N)

@export
def comptime_scale(x: Int32) abi("C") -> Int32:
    var acc: Int32 = 0
    comptime for i in range(4):
        acc += scale[i + 1](x)
    return acc

@export
def simd_dot_i8(a: ImmPointer[Int8, ImmUntrackedOrigin], b: ImmPointer[Int8, ImmUntrackedOrigin]) abi("C") -> Int32:
    var va = a.unsafe_load[width=8]().cast[DType.int32]()
    var vb = b.unsafe_load[width=8]().cast[DType.int32]()
    return (va * vb).reduce_add()

@export
def int_size() abi("C") -> Int32:
    return Int32(size_of[Int]())

@export
def read_cycle() abi("C") -> UInt32:
    return inlined_assembly["csrr $0, mcycle", UInt32, constraints="=r", has_side_effect=True]()

@export
def stack_array(n: Int32) abi("C") -> Int32:
    var arr = Array[Int32, 16](fill=0)
    for i in range(16):
        arr[i] = Int32(i) * n
    var s: Int32 = 0
    for i in range(16):
        s += arr[(i * 7) % 16]
    return s
