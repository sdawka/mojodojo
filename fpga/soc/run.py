#!/usr/bin/env python3
"""Run a firmware image on the prebuilt Verilator SoC (soc.py).

Writes the main-RAM $readmemh image into a fresh run directory next to copies
of the ROM/SRAM/CSR init files, starts obj_dir/Vsim there with the UART on
stdout, and stops at the firmware's SIM_FINISH marker (or a timeout).

  python run.py build/sim/fw/hello/hello.bin
  python run.py fw.bin --blob 0x41000000:path/to/data.bin   # extra RAM contents
"""
import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write_init(path, ram_base, ram_size, segments):
    """segments: list of (addr, bytes). Writes a sparse $readmemh file
    (@word-address records) of little-endian 32-bit words; gaps stay zero."""
    total = 0
    with open(path, "w") as f:
        for addr, data in sorted(segments):
            off = addr - ram_base
            if off < 0 or off % 4 or off + len(data) > ram_size:
                sys.exit(f"segment 0x{addr:x}+0x{len(data):x} outside main RAM")
            data = bytes(data) + b"\0" * (-len(data) % 4)
            words = memoryview(data).cast("I")
            f.write(f"@{off // 4:x}\n")
            f.write("\n".join(f"{w:08x}" for w in words))
            f.write("\n")
            total += len(words)
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image")
    p.add_argument("--soc", default=str(HERE / "build" / "sim"))
    p.add_argument("--blob", action="append", default=[], help="ADDR:FILE extra RAM contents")
    p.add_argument("--wad", default=None,
                   help="ADDR:FILE preload a file with a 16-byte header {'SWAD', size} at ADDR")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--timeout", type=float, default=600.0, help="wall seconds")
    p.add_argument("--log", default=None)
    a = p.parse_args()

    soc = Path(a.soc).resolve()
    gw = soc / "gateware"
    regions = (soc / "software/include/generated/regions.ld").read_text()
    m = re.search(r"main_ram : ORIGIN = (0x[0-9a-f]+), LENGTH = (0x[0-9a-f]+)", regions)
    base, size = int(m.group(1), 16), int(m.group(2), 16)

    image = Path(a.image).resolve()
    run = Path(a.run_dir) if a.run_dir else image.parent / "run"
    run.mkdir(parents=True, exist_ok=True)
    for f in gw.glob("*.init"):
        if f.name != "sim_main_ram.init":
            shutil.copy(f, run / f.name)
    shutil.copy(gw / "sim_config.js", run / "sim_config.js")
    mods = run / "modules"
    if mods.is_symlink() or mods.exists():
        mods.unlink()
    mods.symlink_to(gw / "modules")
    segs = [(base, image.read_bytes())]
    for spec in a.blob:
        addr, path = spec.split(":", 1)
        segs.append((int(addr, 0), Path(path).read_bytes()))
    if a.wad:
        addr, path = a.wad.split(":", 1)
        data = Path(path).read_bytes()
        hdr = (0x44415753).to_bytes(4, "little") + len(data).to_bytes(4, "little") + bytes(8)
        segs.append((int(addr, 0), hdr + data))
    t = time.time()
    n = write_init(run / "sim_main_ram.init", base, size, segs)
    print(f"[run] main RAM image: {n} words ({n*4/1048576:.2f} MiB) in {time.time()-t:.1f}s; run dir {run}",
          flush=True)

    log = open(a.log or run / "uart.log", "w")
    t0 = time.time()
    proc = subprocess.Popen([str(gw / "obj_dir" / "Vsim")], cwd=run, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    code, buf = None, b""
    try:
        while True:
            chunk = proc.stdout.read1(65536) if hasattr(proc.stdout, "read1") else proc.stdout.read(1)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk); sys.stdout.flush()
            log.write(chunk.decode("latin-1")); log.flush()
            buf = (buf + chunk)[-4096:]
            mm = re.search(rb"SIM_FINISH (-?\d+)", buf)
            if mm and code is None:
                code = int(mm.group(1))
            if time.time() - t0 > a.timeout:
                print(f"\n[run] timeout after {a.timeout:.0f}s", flush=True)
                proc.send_signal(signal.SIGTERM)
                break
    finally:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    print(f"[run] wall {time.time()-t0:.1f}s, SIM_FINISH={code}", flush=True)
    sys.exit(0 if code == 0 else 1)


if __name__ == "__main__":
    main()
