"""C-ABI exports for linking Mojo tile drivers into C firmware (fpga/soc fw.mk).

int32 C[M][N] = int8 A[M][K] @ int8 B[K][N], row-major.
  uint32_t mojo_tile_matmul_i8 (const int8_t*, const int8_t*, int32_t*, int M, int K, int N); // tile v1; returns tile CYCLES sum
  int32_t  mojo_tile2_matmul_i8(...same...);  // tile v2; busy cycles or -1 on tile error
  void     mojo_cpu_matmul_i8  (...same...);  // reference on the core
Needs no libc symbols (pure MMIO + integer code).
"""

from tile_v1 import tile_matmul_i8, cpu_matmul_i8, I8Ptr, I32Ptr
from tile2 import tile2_matmul_i8


@export
def mojo_tile_matmul_i8(a: I8Ptr, b: I8Ptr, c: I32Ptr, M: Int32, K: Int32, N: Int32) abi("C") -> UInt32:
    return tile_matmul_i8(a, b, c, Int(M), Int(K), Int(N))


@export
def mojo_tile2_matmul_i8(a: I8Ptr, b: I8Ptr, c: I32Ptr, M: Int32, K: Int32, N: Int32) abi("C") -> Int32:
    return Int32(tile2_matmul_i8(a, b, c, Int(M), Int(K), Int(N)))


@export
def mojo_cpu_matmul_i8(a: I8Ptr, b: I8Ptr, c: I32Ptr, M: Int32, K: Int32, N: Int32) abi("C"):
    cpu_matmul_i8(a, b, c, Int(M), Int(K), Int(N))
