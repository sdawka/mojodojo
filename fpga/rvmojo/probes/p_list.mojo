@export
def use_list(n: Int32) abi("C") -> Int32:
    var l = List[Int32]()
    for i in range(n):
        l.append(i)
    return l[len(l) - 1]
