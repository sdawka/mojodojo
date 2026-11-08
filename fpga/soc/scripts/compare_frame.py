#!/usr/bin/env python3
"""Compare DOOM frames from the SoC UART log against the host build.

  compare_frame.py UART_LOG HOST_HASHES [--host-png host-frame1.png] [--png out.png]

UART_LOG: run.py log with `FRAME n ... sha256=...` lines and an optional
`FB_BEGIN n 320x200 xrgb` ... `FB_END` hex dump (6 hex digits/pixel).
HOST_HASHES: doom/build/doom-c --hash-all output (`n sha256` per line).
Exit 0 iff every SoC frame hash equals the host hash for the same frame
and (if given) the dumped pixels equal the host PNG pixels.
"""
import argparse, re, struct, sys, zlib


def read_png_rgb(path):
    data = open(path, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, w = 8, b"", None
    while pos < len(data):
        n, = struct.unpack(">I", data[pos:pos + 4]); typ = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + n]; pos += 12 + n
        if typ == b"IHDR":
            w, h, bd, ct = struct.unpack(">IIBB", body[:10])
            assert bd == 8 and ct in (2, 6), (bd, ct)
            bpp = 3 if ct == 2 else 4
        elif typ == b"IDAT":
            idat += body
    raw = zlib.decompress(idat); stride = w * bpp
    out, prev, i = [], bytearray(stride), 0
    for _ in range(h):
        f = raw[i]; line = bytearray(raw[i + 1:i + 1 + stride]); i += 1 + stride
        for x in range(stride):
            a = line[x - bpp] if x >= bpp else 0; b = prev[x]; c = prev[x - bpp] if x >= bpp else 0
            if f == 1: line[x] = (line[x] + a) & 255
            elif f == 2: line[x] = (line[x] + b) & 255
            elif f == 3: line[x] = (line[x] + (a + b) // 2) & 255
            elif f == 4:
                p = a + b - c; pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[x] = (line[x] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        out.append([tuple(line[x * bpp:x * bpp + 3]) for x in range(w)]); prev = line
    return out


def write_png(path, rows):
    h, w = len(rows), len(rows[0])
    raw = b"".join(b"\0" + bytes(c for px in r for c in px) for r in rows)
    def chunk(t, b): return struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b))
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("uart_log"); ap.add_argument("host_hashes")
    ap.add_argument("--host-png"); ap.add_argument("--png")
    a = ap.parse_args()
    host = {}
    for line in open(a.host_hashes):
        p = line.split()
        if len(p) == 2: host[int(p[0])] = p[1]
    soc, fb, fbn = {}, [], None
    for line in open(a.uart_log, errors="replace"):
        m = re.match(r"FRAME (\d+) .*sha256=([0-9a-f]{64})", line)
        if m: soc[int(m.group(1))] = m.group(2)
        m = re.match(r"FB_BEGIN (\d+)", line)
        if m: fbn = int(m.group(1))
        m = re.match(r"FB (\d+) ([0-9a-f]+)", line)
        if m:
            hx = m.group(2)
            fb.append([tuple(int(hx[i + k:i + k + 2], 16) for k in (0, 2, 4)) for i in range(0, len(hx), 6)])
    ok = bool(soc)
    for n in sorted(soc):
        same = host.get(n) == soc[n]
        ok &= same
        print(f"frame {n}: soc {soc[n][:16]}.. host {(host.get(n) or 'missing')[:16]}.. {'MATCH' if same else 'DIFFER'}")
    if fb:
        print(f"framebuffer dump: frame {fbn}, {len(fb[0])}x{len(fb)}")
        if a.png: write_png(a.png, fb); print(f"wrote {a.png}")
        if a.host_png:
            hp = read_png_rgb(a.host_png)
            diff = sum(1 for y in range(len(fb)) for x in range(len(fb[0])) if fb[y][x] != hp[y][x])
            print(f"pixel compare vs {a.host_png}: {diff} differing pixels of {len(fb) * len(fb[0])}")
            ok &= diff == 0
    print("COMPARE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
