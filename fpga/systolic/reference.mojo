"""Plain reference matmul: C[M,N] = A[M,K] @ B[K,N], integer operands, wrapping accumulate."""


def reference_matmul[
    in_dtype: DType, acc_dtype: DType
](
    a: List[Scalar[in_dtype]], b: List[Scalar[in_dtype]], m: Int, k: Int, n: Int
) -> List[Scalar[acc_dtype]]:
    var c = List[Scalar[acc_dtype]](length=m * n, fill=0)
    for i in range(m):
        for j in range(n):
            var s: Scalar[acc_dtype] = 0
            for kk in range(k):
                s += a[i * k + kk].cast[acc_dtype]() * b[kk * n + j].cast[
                    acc_dtype
                ]()
            c[i * n + j] = s
    return c^
