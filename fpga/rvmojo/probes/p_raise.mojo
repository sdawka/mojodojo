def may_raise(x: Int32) raises -> Int32:
    if x < 0:
        raise Error("neg")
    return x

@export
def use_raise(n: Int32) abi("C") -> Int32:
    try:
        return may_raise(n)
    except:
        return -1
