"""Deterministic xorshift64* PRNG so traces are reproducible across runs and languages."""


struct Rng(Copyable, Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        # Never allow the all-zero state.
        self.s = seed * 0x9E3779B97F4A7C15 + 0x2545F4914F6CDD1D
        if self.s == 0:
            self.s = 1

    def next(mut self) -> UInt64:
        self.s ^= self.s >> 12
        self.s ^= self.s << 25
        self.s ^= self.s >> 27
        return self.s * 0x2545F4914F6CDD1D

    def range(mut self, lo: Int, hi: Int) -> Int:
        """Uniform-ish integer in [lo, hi]."""
        return lo + Int(self.next() % UInt64(hi - lo + 1))

    def pct(mut self, p: Int) -> Bool:
        """True with probability p/100."""
        return self.range(0, 99) < p

    def scalar[dtype: DType](mut self) -> Scalar[dtype]:
        """Uniform over the full range of an integer dtype (truncating cast)."""
        return Scalar[DType.uint64](self.next()).cast[dtype]()
