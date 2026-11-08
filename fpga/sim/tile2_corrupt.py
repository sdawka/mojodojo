"""Write a copy of a tile2 program with one wrong expectation (negative check).

usage: tile2_corrupt.py <in> <out> cread|end|poll
  cread: a C-window read in the middle of the program expects value+1
  end:   END total cycles + 1
  poll:  the last STATUS read that saw BUSY=1 expects BUSY=0/DONE=1 instead
"""
import sys

src, dst, kind = sys.argv[1:4]
L = open(src).read().splitlines()
if kind == "cread":
    idx = [i for i, l in enumerate(L) if l.startswith("R ") and int(l.split()[2], 16) >= 0x18000]
    i = idx[len(idx) // 2]
    f = L[i].split()
    f[3] = "%x" % ((int(f[3], 16) + 1) & 0xFFFFFFFF)
elif kind == "end":
    i = len(L) - 1
    f = L[i].split()
    f[1] = str(int(f[1]) + 1)
else:
    idx = [i for i, l in enumerate(L) if l.startswith("R ") and l.split()[2] == "c" and int(l.split()[3], 16) & 1]
    i = idx[-1]
    f = L[i].split()
    f[3] = "%x" % ((int(f[3], 16) & ~1) | 2)
L[i] = " ".join(f)
open(dst, "w").write("\n".join(L) + "\n")
