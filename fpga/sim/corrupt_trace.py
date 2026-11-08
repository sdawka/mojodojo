"""Write a copy of a JSONL golden trace with one deliberately wrong output value.

python3 -I sim/corrupt_trace.py <in.jsonl> <out.jsonl> out_row|out_last

out_row:  add 1 to out_row[2] (or the last element if N < 3) on the first DRAIN
          cycle (out_valid == 1).
out_last: clear out_last on the first cycle where it is 1.
The modified cycle is printed so the harness report can be checked against it.
"""

import json
import sys


def main() -> int:
    src, dst, field = sys.argv[1], sys.argv[2], sys.argv[3]
    done = False
    with open(src) as f, open(dst, "w") as g:
        for line in f:
            rec = json.loads(line)
            out = rec["out"]
            hit = False
            if not done and field == "out_row" and out["out_valid"] == 1:
                out["out_row"][min(2, len(out["out_row"]) - 1)] += 1
                hit = True
            elif not done and field == "out_last" and out["out_last"] == 1:
                out["out_last"] = 0
                hit = True
            if hit:
                done = True
                print(f"corrupted {field} at cyc {rec['cyc']}", file=sys.stderr)
            g.write(json.dumps(rec, separators=(",", ":")) + "\n")
    if not done:
        print("nothing to corrupt", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
