"""MAX custom op `phantom_matmul_i8`: routes an int8 W8A8 matmul to the phantom NPU.

Graph contract (CPU device only; the phantom is a host-attached accelerator):
    c[M, N_out] int32 = phantom_matmul_i8(a[M, K] int8, w[N_out, K] int8, site[1] int32)
`site` is a runtime call-site id (layer*4 + projection) written to the launch
log ($PHANTOM_LOG). It is a tensor rather than a kernel parameter so that every
routed layer shares one compiled kernel.
"""

import extensibility
from extensibility import InputTensor, OutputTensor
from max.gpu.host import DeviceContext

from .phantom_adapter import phantom_matmul_i8


@extensibility.register("phantom_matmul_i8")
struct PhantomMatmulI8:
    @staticmethod
    def execute(
        c: OutputTensor[dtype=.int32, rank=2, ...],
        a: InputTensor[dtype=.int8, rank=2, ...],
        w: InputTensor[dtype=.int8, rank=2, ...],
        site: InputTensor[dtype=.int32, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime if ctx.target.is_gpu():
            raise Error("phantom_matmul_i8 runs on the host (CPU) only")
        var m = a.dim_size[0]()
        var k = a.dim_size[1]()
        var n_out = w.dim_size[0]()
        if w.dim_size[1]() != k or c.dim_size[0]() != m or c.dim_size[1]() != n_out:
            raise Error("phantom_matmul_i8: shape mismatch")
        phantom_matmul_i8(
            a.unsafe_ptr(), w.unsafe_ptr(), c.unsafe_ptr(), m, k, n_out,
            Int(site.unsafe_ptr()[unsafe_offset=0]),
        )
