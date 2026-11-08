"""Part 1: the two toolchains, stage by stage. Content is hand-curated from notes/toolchain-brief.md,
mojo-max/README.md and atlas/README.md; claims marked `inferred` are not stated in those notes."""

from __future__ import annotations

from .sources import ROOT, Reg, esc

LANES = [  # key, title, group, colour class
    ("mojo", "Mojo kernels", "mojo", "t-max"),
    ("maxg", "MAX graph", "mojo", "t-max"),
    ("metal", "Metal (MSL)", "apple", "t-metal"),
    ("mlx", "MLX", "apple", "t-mlx"),
    ("coreml", "Core ML", "apple", "t-coreml"),
    ("accel", "Accelerate / BNNS", "apple", "t-accel"),
]

STAGES = [
    ("src", "Source", "what you write"),
    ("ir", "Front end and IR", "what the compiler sees"),
    ("cg", "Code generation", "how kernels become machine code"),
    ("pkg", "Packaging", "what ships to the device"),
    ("rt", "Runtime and driver", "what runs it"),
    ("hw", "Hardware reached", "on this M5"),
    ("srv", "Serving", "LLM endpoint"),
]

B, M, A = "notes/toolchain-brief.md", "mojo-max/README.md", "atlas/README.md"

# (stage, lane, span, html text, tags, ref). Tags: open, closed, xcode, noxcode, inferred, none.
CELLS = [
    # ---------------------------------------------------------------- Mojo kernels
    ("src", "mojo", 1, "Mojo source with <code>comptime</code> parameters and inline <code>__mlir_op</code>.",
     ["open"], B + " §2"),
    ("ir", "mojo", 1, "<code>MojoParser</code> → KGEN, POP and LIT MLIR dialects. The elaborator's comptime "
     "interpreter runs ordinary Mojo during compilation.", ["open"], B + " §1"),
    ("cg", "mojo", 1, "<code>KGENToLLVM</code> → LLVM, then one backend per target: CPU (AArch64), Metal "
     "(<code>MetalLowering</code> → <code>MetalAIRPass</code> → AIR bitcode), or an out-of-tree plugin "
     "through <code>M_KGEN_getTargetBackend</code>.", ["open"], B + " §1"),
    ("pkg", "mojo", 1, "<code>MetallibWriter</code> packs AIR into a <code>.metallib</code> in-process when "
     "<code>MODULAR_METAL_INTREE_METALLIB=1</code>. The variable is undocumented; it was found in the binary.",
     ["open", "noxcode"], M + " Gotchas"),
    ("rt", "mojo", 1, "<code>DeviceContext</code> over metal-cpp, which sits on Apple's closed Metal.framework. "
     "A vendor-neutral C HAL (<code>M_driver_*.h</code>) is the hook for new devices.", ["open"], B + " §1"),
    ("hw", "mojo", 1, "CPU (NEON; AMX through inline-asm wrappers) and GPU, including the M5 Neural "
     "Accelerators via <code>AppleM5MatMul</code>. No ANE.", ["open"], B + " §3"),
    ("srv", "mojo", 1, "Kernels reach serving through the MAX graph (next lane).", ["none"], B + " §3"),
    # ---------------------------------------------------------------- MAX graph
    ("src", "maxg", 1, "Python <code>max.nn</code> / <code>max.graph</code> (ModuleV3) plus Mojo custom ops.",
     ["open"], B + " §3"),
    ("ir", "maxg", 1, "MO → MOGG / RMO / MOSH graph dialects. Fusion passes such as rms-norm + quant and "
     "layer-norm + rope.", ["open"], B + " §1, §3"),
    ("cg", "maxg", 1, "Each op lowers to a Mojo kernel from <code>max/kernels</code>, compiled with the graph "
     "by the same Mojo compiler.", ["open"], B + " §3"),
    ("pkg", "maxg", 1, "Compiled graph in the MAX cache. On this build each launch recompiles the model "
     "(about 30 s).", ["open", "noxcode"], M + " §2"),
    ("rt", "maxg", 1, "MAX engine (GRT / MEF graph runtime), paged KV cache.", ["open"], B + " §1, §3"),
    ("hw", "maxg", 1, "GPU: Neural Accelerator MMA, simdgroup, int8 / fp4 / fp8 matmul, M5 flash-attention. "
     "CPU: Accelerate-backed matmul.", ["open"], B + " §3"),
    ("srv", "maxg", 1, "<code>max serve</code>: OpenAI-compatible HTTP. <code>max benchmark</code> drives "
     "load against it.", ["open"], M + " §2"),
    # ---------------------------------------------------------------- Metal
    ("src", "metal", 1, "Metal Shading Language (C++-based) kernels; Swift, Objective-C or C++ host code.",
     ["closed"], B + " §1"),
    ("ir", "metal", 1, "The <code>metal</code> front end (clang-based) parses MSL.", ["closed", "xcode"],
     M + " Machine"),
    ("cg", "metal", 1, "Emits AIR, Apple's LLVM-based GPU IR. This is the same IR Mojo emits, which is why "
     "Mojo can skip the Apple front end.", ["closed", "xcode"], B + " §1"),
    ("pkg", "metal", 1, "<code>metallib</code> links AIR into a <code>.metallib</code>. Here "
     "<code>xcrun -f metal</code> fails: Command Line Tools ship no Metal toolchain.", ["closed", "xcode"],
     "atlas/results/inventory.json versions.metal_compiler"),
    ("rt", "metal", 1, "Metal.framework. The driver compiles AIR to the GPU's ISA when a pipeline is "
     "created.", ["closed", "inferred"], "general Metal behaviour; not measured here"),
    ("hw", "metal", 1, "GPU simdgroup matrix ops; Neural Accelerators through Metal 4 tensor APIs.",
     ["closed", "inferred"], A),
    ("srv", "metal", 1, "None. You build your own.", ["none"], ""),
    # ---------------------------------------------------------------- MLX
    ("src", "mlx", 1, "Python, C++ or Swift array code, evaluated lazily.", ["open"], "bench/README.md"),
    ("ir", "mlx", 1, "A lazy compute graph. <code>mx.compile</code> fuses element-wise ops.",
     ["open", "inferred"], ""),
    ("cg", "mlx", 1, "Hand-written Metal kernels. M5 <code>NAX</code> kernels are chosen when the GPU "
     "generation is ≥ 17 and macOS ≥ 26.2.", ["open"], A + " Gotchas"),
    ("pkg", "mlx", 1, "Kernels ship precompiled in the wheel's <code>.metallib</code>, so no Xcode is needed "
     "to run. Fused kernels are JIT-compiled from source by Metal.", ["open", "noxcode", "inferred"], A),
    ("rt", "mlx", 1, "MLX runtime on Metal.framework; unified memory without copies.", ["open"], ""),
    ("hw", "mlx", 1, "GPU and Neural Accelerators (NAX); CPU stream via Accelerate.", ["open", "inferred"], A),
    ("srv", "mlx", 1, "<code>mlx-lm</code>; phase 1 measured it in-process.", ["open"], "bench/README.md"),
    # ---------------------------------------------------------------- Core ML
    ("src", "coreml", 1, "A trained model (PyTorch and others) converted with coremltools, which is open "
     "source.", ["open"], A),
    ("ir", "coreml", 1, "MIL (Model Intermediate Language) program in an <code>.mlpackage</code>.",
     ["open"], A),
    ("cg", "coreml", 1, "Core ML compiles MIL and builds a compute plan that assigns each op to CPU, GPU or "
     "ANE (<code>MLComputePlan</code>).", ["closed"], A),
    ("pkg", "coreml", 1, "Compiled <code>.mlmodelc</code>; ANE programs come from a private compiler.",
     ["closed", "inferred"], ""),
    ("rt", "coreml", 1, "Core ML framework and a private ANE driver. Without Instruments, placement is a "
     "plan, not an observation.", ["closed"], A + " Gotchas"),
    ("hw", "coreml", 1, "CPU, GPU or ANE. The only route to the Neural Engine.", ["closed"], B + " §5"),
    ("srv", "coreml", 1, "Embedded in apps; no server.", ["none"], ""),
    # ---------------------------------------------------------------- Accelerate
    ("src", "accel", 1, "C or Swift calls such as <code>cblas_sgemm</code> and <code>BNNSMatMul</code>.",
     ["closed"], A),
    ("ir", "accel", 1, "No IR: you call precompiled library routines.", ["none"], ""),
    ("cg", "accel", 1, "Hand-tuned kernels inside the library.", ["closed"], ""),
    ("pkg", "accel", 1, "Accelerate.framework, part of macOS.", ["closed", "noxcode"], ""),
    ("rt", "accel", 1, "Routes matmul to the SME unit (the AMX successor).", ["closed"], A),
    ("hw", "accel", 1, "CPU NEON and SME. One SME unit is shared per P-cluster.", ["closed"], "atlas/ATLAS.md §3"),
    ("srv", "accel", 1, "None.", ["none"], ""),
]

# which engines each lane can drive: cpu, gpu, na (GPU Neural Accelerators), ane
REACH = {
    "mojo": {"cpu", "gpu", "na"}, "maxg": {"cpu", "gpu", "na"}, "metal": {"gpu", "na"},
    "mlx": {"cpu", "gpu", "na"}, "coreml": {"cpu", "gpu", "ane"}, "accel": {"cpu"},
}
ENGINE_LABEL = {"cpu": "CPU", "gpu": "GPU", "na": "NA", "ane": "ANE"}

TAG_LABEL = {"open": "open source", "closed": "closed", "xcode": "needs Xcode", "noxcode": "no Xcode",
             "inferred": "inferred"}


def render(reg: Reg) -> str:
    srcs = {p: reg.add(ROOT / p, None, "doc") for p in (B, M, A)}
    lane_idx = {k: i for i, (k, *_r) in enumerate(LANES)}
    st_idx = {k: i for i, (k, *_r) in enumerate(STAGES)}
    h = ['<div class="pipe" role="table" aria-label="Toolchain stages, Mojo/MAX against Apple-native">']
    # group headers
    h.append('<div class="pg pg-mojo" style="grid-column:2 / span 2;grid-row:1" role="columnheader">'
             'Mojo / MAX <span>Modular, open source</span></div>'
             '<div class="pg pg-apple" style="grid-column:5 / span 4;grid-row:1" role="columnheader">'
             'Apple-native <span>mostly closed, ships with macOS / Xcode</span></div>')
    for k, title, grp, cls in LANES:
        c = lane_idx[k] + (2 if grp == "mojo" else 3)
        h.append(f'<div class="lh {cls} g-{grp}" style="grid-column:{c};grid-row:2" role="columnheader">'
                 f'{esc(title)}</div>')
    for sk, title, sub in STAGES:
        r = st_idx[sk] + 3
        h.append(f'<div class="sh" style="grid-row:{r};--o:{r * 10}" role="rowheader">{esc(title)}<small>{esc(sub)}</small></div>'
                 f'<div class="band" style="grid-row:{r}"></div>')
    for sk, lk, span, txt, tags, ref in CELLS:
        grp = LANES[lane_idx[lk]][2]
        r, c = st_idx[sk] + 3, lane_idx[lk] + (2 if grp == "mojo" else 3)
        cls = ["cell", f"g-{grp}"] + [f"tag-{t}" for t in tags]
        chips = "".join(f'<i class="chip c-{t}">{TAG_LABEL[t]}</i>' for t in tags if t in TAG_LABEL)
        reach = ""
        if sk == "hw":
            reach = '<span class="reach">' + "".join(
                f'<b class="{"on" if e in REACH[lk] else "off"}" title="{ENGINE_LABEL[e]}: '
                f'{"reached" if e in REACH[lk] else "not reachable"}">{ENGINE_LABEL[e]}</b>'
                for e in ("cpu", "gpu", "na", "ane")) + "</span>"
        src = next((s for p, s in srcs.items() if ref.startswith(p)), None)
        data = (f' tabindex="0" data-s="{src.id}" data-d="{esc("Basis: " + ref)}"' if src else
                (f' tabindex="0" data-d="{esc("Basis: " + ref)}"' if ref else ""))
        lane_title = LANES[lane_idx[lk]][1]
        h.append(f'<div class="{" ".join(cls)}" style="grid-row:{r};grid-column:{c} / span {span};--o:{r * 10 + c}" role="cell"'
                 f'{data}><span class="m-lane">{esc(lane_title)}</span>{reach}<p>{txt}</p>'
                 f'{f"<span class=chips>{chips}</span>" if chips else ""}</div>')
    h.append("</div>")
    return "".join(h)


def key_html() -> str:
    return ('<ul class="pkey">'
            '<li><span class="k k-open"></span>open source</li>'
            '<li><span class="k k-closed"></span>closed (hatched)</li>'
            '<li><i class="chip c-xcode">needs Xcode</i> missing on this Mac</li>'
            '<li><i class="chip c-noxcode">no Xcode</i> works with Command Line Tools only</li>'
            '<li><i class="chip c-inferred">inferred</i> not stated in the project notes</li>'
            '</ul>')
