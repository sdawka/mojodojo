#!/usr/bin/env python3
"""Independent host check of tiletest: regenerate the same xorshift64* int8
matrices in Python, compute C = A@B (int32 wrap), and compare the FNV-1a hash
of C with each TILE_RESULT line in the UART log. Also prints the cycle summary."""
import re, sys

M64 = (1 << 64) - 1


def main(log):
    rng = 0x9E3779B97F4A7C15
    def nxt():
        nonlocal rng
        rng ^= rng >> 12; rng ^= (rng << 25) & M64; rng ^= rng >> 27
        return ((rng * 0x2545F4914F6CDD1D) & M64) >> 32
    s8 = lambda v: ((v & 0xff) ^ 0x80) - 0x80
    lines = [l for l in open(log, errors="replace") if l.startswith("TILE_RESULT")]
    if not lines:
        sys.exit("no TILE_RESULT lines")
    expect = {}
    for (M, K, N) in [(13, 64, 9), (8, 256, 32)]:  # same order as fw/tiletest/main.c
        for trial in range(3):
            A = [[s8(nxt()) for _ in range(K)] for _ in range(M)]
            B = [[s8(nxt()) for _ in range(N)] for _ in range(K)]
            if trial == 2:
                A = [[-128] * K for _ in range(M)]; B = [[-128] * N for _ in range(K)]
            h = 2166136261
            for m in range(M):
                for n in range(N):
                    c = sum(A[m][k] * B[k][n] for k in range(K)) & 0xffffffff
                    h = ((h ^ c) * 16777619) & 0xffffffff
            expect[(M, K, N, trial)] = h
    ok = True
    for l in lines:
        kv = dict(t.split("=") for t in l.split()[1:])
        key = (int(kv["M"]), int(kv["K"]), int(kv["N"]), int(kv["trial"]))
        good = int(kv["fnv"], 16) == expect[key] and kv["mismatches"] == "0"
        ok &= good
        print(f"{kv['tile']} M={key[0]} K={key[1]} N={key[2]} trial {key[3]}: host fnv 0x{expect[key]:08x} "
              f"soc {kv['fnv']} mism={kv['mismatches']} cpu_ref={kv['cpu_ref_cycles']} "
              f"tile_total={kv['tile_total_cycles']} tile_busy={kv['tile_busy_cycles']} "
              f"ideal={kv['ideal_cycles']} -> {'OK' if good else 'BAD'}")
    print("HOSTCHECK", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1])
