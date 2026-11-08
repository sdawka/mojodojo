@export
def use_print(n: Int32) abi("C"):
    print("hello", n)
