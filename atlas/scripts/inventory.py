"""Machine + toolchain inventory -> results/inventory.json (no sudo needed)."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import atlas_common as ac  # noqa: E402

ROOT = ac.ATLAS.parent


def sh(cmd: str, cwd: Path | None = None) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120,
                              cwd=cwd).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def sysctl_tree(prefix: str) -> dict:
    out = {}
    for line in sh(f"sysctl {prefix}").splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            v = v.strip()
            out[k] = int(v) if re.fullmatch(r"-?\d+", v) else v
    return out


def main():
    hw = sysctl_tree("hw")
    arm = {k.split(".")[-1]: v for k, v in hw.items() if k.startswith("hw.optional.arm.")}
    gpu = json.loads(sh("system_profiler -json SPDisplaysDataType") or "{}")
    hwp = json.loads(sh("system_profiler -json SPHardwareDataType") or "{}")
    gpu_cores = sh("ioreg -rc AGXAccelerator -d1 | grep -o '\"gpu-core-count\" = [0-9]*'")
    ane = sh("ioreg -rc H11ANEIn -d1 | grep -oE '\"(ANEDevicePropertyKey[A-Za-z]*|DeviceProperties)\"[^\\n]{0,200}' | head -20")
    mlx = sh(f"cd {ac.ATLAS} && uv run --quiet python -c \"import mlx.core as mx, json; print(json.dumps(mx.device_info()))\"")
    inv = {
        "summary": {
            "chip": sh("sysctl -n machdep.cpu.brand_string"),
            "model": hw.get("hw.targettype"),
            "memory_bytes": hw.get("hw.memsize"),
            "p_cores": hw.get("hw.perflevel0.physicalcpu"),
            "e_cores": hw.get("hw.perflevel1.physicalcpu"),
            "p_l1d": hw.get("hw.perflevel0.l1dcachesize"), "p_l2": hw.get("hw.perflevel0.l2cachesize"),
            "e_l1d": hw.get("hw.perflevel1.l1dcachesize"), "e_l2": hw.get("hw.perflevel1.l2cachesize"),
            "cacheline": hw.get("hw.cachelinesize"), "page": hw.get("hw.pagesize"),
            "sme_svl_bytes": arm.get("sme_max_svl_b"),
            "features": {k: arm.get(k) for k in ("FEAT_SME", "FEAT_SME2", "FEAT_SME2p1", "FEAT_SME_F16F16",
                         "FEAT_SME_B16B16", "SME_I8I32", "SME_F16F32", "SME_B16F32", "FEAT_BF16",
                         "FEAT_I8MM", "FEAT_DotProd", "FEAT_FP16", "FEAT_FHM")},
            "gpu_core_count": gpu_cores,
            "gpu_arch_mlx": None,
        },
        "apple_spec": {
            "memory_bandwidth_GBps": 153,
            "cpu": "up to 10-core: 4 performance + 6 efficiency",
            "gpu": "10-core, a Neural Accelerator in each core; 'over 4x peak GPU compute (AI) vs M4'",
            "neural_engine": "16-core (no TOPS figure published for M5)",
            "source": "https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/",
        },
        "sysctl_hw": hw,
        "system_profiler_hardware": hwp,
        "system_profiler_displays": gpu,
        "ane_ioreg": ane,
        "mlx_device_info": mlx,
        "versions": {
            "macos": platform.mac_ver()[0],
            "darwin": platform.release(),
            "clang": sh("clang --version | head -1"),
            "mojo": sh(f"pixi run --manifest-path {ROOT}/mojo-max/pixi.toml mojo --version"),
            "max_pkgs": sh(f"pixi list --manifest-path {ROOT}/mojo-max/pixi.toml 2>/dev/null | grep -E '^(max|mojo)[a-z-]* '"),
            "mlx": sh(f"cd {ac.ATLAS} && uv run --quiet python -c 'import mlx.core as m; print(m.__version__)'"),
            "coremltools": sh(f"cd {ac.ATLAS} && uv run --quiet python -c 'import coremltools as c; print(c.__version__)'"),
            "numpy": sh(f"cd {ac.ATLAS} && uv run --quiet python -c 'import numpy as n; print(n.__version__)'"),
            "llamacpp_commit": sh(f"git -C {ROOT}/llamacpp/src rev-parse --short HEAD"),
            "xcode": sh("xcode-select -p"),
            "metal_compiler": sh("xcrun -f metal 2>&1") or "absent",
        },
        "power": sh("pmset -g batt | head -2"),
        "thermal": sh("pmset -g therm"),
    }
    try:
        inv["summary"]["gpu_arch_mlx"] = json.loads(mlx).get("architecture")
    except Exception:  # noqa: BLE001
        pass
    ac.RESULTS.mkdir(exist_ok=True, parents=True)
    out = ac.RESULTS / "inventory.json"
    out.write_text(json.dumps(inv, indent=1, default=str))
    print(json.dumps(inv["summary"], indent=1))
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
