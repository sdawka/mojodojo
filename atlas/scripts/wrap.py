"""Run a native benchmark binary, collect its JSON-lines records, write results/<name>.json.

usage: wrap.py NAME [--stdin-shapes] [--meta k=v ...] -- CMD ARGS...
Lines on stdout that parse as JSON objects are records; everything else is echoed to stderr.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import atlas_common as ac  # noqa: E402
import shapes  # noqa: E402


def main():
    argv = sys.argv[1:]
    sep = argv.index("--")
    opts, cmd = argv[:sep], argv[sep + 1 :]
    name = opts[0]
    extra = {"command": cmd}
    stdin_data = None
    for o in opts[1:]:
        if o == "--stdin-shapes":
            lm = os.environ.get("ATLAS_LM_HEAD", "1") == "1"
            stdin_data = "".join(f"{t} {m} {n} {k}\n" for t, m, n, k in shapes.shapes(ac.SMOKE, lm))
        elif o.startswith("--meta"):
            pass
        elif "=" in o:
            k, v = o.split("=", 1)
            extra[k] = v
    p = subprocess.run(cmd, input=stdin_data, capture_output=True, text=True)
    recs = []
    for line in p.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                recs.append(json.loads(line))
                continue
            except json.JSONDecodeError:
                pass
        print(line, file=sys.stderr)
    sys.stderr.write(p.stderr[-4000:])
    # Mojo benches emit raw per-op seconds; convert to throughput stats here.
    for r in recs:
        if "samples_s" in r:
            s = r["samples_s"]
            r.update(ac.stats([r["work_per_op"] / x / 1e9 for x in s]))
            if "bytes_per_call" in r:
                bw = ac.stats([r["bytes_per_call"] / x / 1e9 for x in s])
                r["eff_gbps_median"], r["eff_gbps_max"] = bw["median"], bw["max"]
    extra["returncode"] = p.returncode
    ac.write_results(name, recs, extra)
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
