def total(s: Span[Int32, _]) -> Int32:
    var t: Int32 = 0
    for v in s:
        t += v
    return t

@export
def span_sum(n: Int32) abi("C") -> Int32:
    var arr = Array[Int32, 8](fill=0)
    for i in range(8):
        arr[i] = Int32(i) + n
    return total(Span(arr))
