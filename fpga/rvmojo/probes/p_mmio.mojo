comptime REG_BASE = 0x80000000

@export
def mmio_poke() abi("C"):
    var p = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=REG_BASE)
    p.unsafe_store[volatile=True](UInt32(1))
    p.unsafe_store[volatile=True](UInt32(2))
    while (p.unsafe_offset(1).unsafe_load[volatile=True]() & 1) == 0:
        pass

@export
def plain_poke() abi("C"):
    var p = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=REG_BASE)
    p.unsafe_store(UInt32(1))
    p.unsafe_store(UInt32(2))
