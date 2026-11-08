# atlas: M5 processor atlas (G2-M1)

This directory holds reproducible benchmarks of memory bandwidth and matmul throughput on this Apple M5 (32 GB), per compute engine × dtype × shape. They feed the roofline model in G2-M2. The report is [`ATLAS.md`](ATLAS.md), with plots in `plots/` and raw data in `results/*.json`.

## Run it

```bash
cd /Users/sdawka/Code/llming/atlas
uv sync                               # numpy, matplotlib, coremltools 9.0, mlx 0.32.4
ATLAS_SMOKE=1 ./run_all.sh            # ~2-5 min validation run -> results/*_smoke.json (non-authoritative)
./run_all.sh                          # full timed suite, ~45-60 min -> results/*.json + ATLAS.md
```

Run the full suite only on an **idle** machine: no Mojo builds, `max serve`, llama.cpp or MLX jobs, video or browsers, on AC power.
- `run_all.sh` records the load average and only sets `authoritative: true` in the JSON metadata when the 1-min load is ≤ 1.5.
- Sections run strictly one after another, with a 20 s cooldown between them (`ATLAS_COOLDOWN_S`).

Knobs:
- `ATLAS_REPS` (timed reps, ≥5; default 15, smoke 5)
- `ATLAS_WARMUP`
- `ATLAS_COLD_MIB` (minimum weight working set cycled per benchmark so decode shapes run cache-cold; default 256)
- `ATLAS_LM_HEAD=0` (skip the 3072×128256 shapes)
- `ATLAS_SKIP="ane ggml"` (sections: `inventory build cpu_bw cpu_peak accel mlx mlx_nonax mlx_notf32 max_gpu ane ggml report`)
- `ATLAS_GGML_REPS`

To regenerate the report from the existing JSON: `uv run python scripts/report.py`.

## What each piece measures

| section | file | engine / method |
|---|---|---|
| inventory | `scripts/inventory.py` | `sysctl hw.*` (P/E cores, caches, FEAT_SME/SME2/BF16/I8MM…, SVL), GPU cores (ioreg), versions → `results/inventory.json` |
| cpu_bw | `cpu/stream.c` | STREAM-like read/write/copy/triad, 1/2/4/8/10 threads (P-pref QoS) and 6 threads at background QoS (E-cores) |
| cpu_peak | `cpu/peak.c` | register-only peaks: NEON FMLA f32/f16, FMLAL, BFDOT, BFMMLA, SDOT, SMMLA; SME FMOPA f32, f16→f32, bf16→f32, f16→f16, SMOPA i8→i32 |
| accel | `cpu/gemm_accel.c` | Accelerate `cblas_sgemm` (multi- and single-threaded via `BLASSetThreading`) and `BNNSMatMul` f32/f16/bf16 (Accelerate routes these to the SME unit) |
| mlx / mlx_nonax / mlx_notf32 | `gpu/mlx_bench.py` | MLX GPU bandwidth plus matmul f32/f16/bf16 and `quantized_matmul` int8/int4 (g64). Three runs: default (NAX = GPU Neural Accelerators); `MLX_METAL_GPU_ARCH=applegpu_g16g` (forces M4 kernels, no NA); `MLX_ENABLE_TF32=0` |
| max_gpu | `gpu/max_gpu_bench.mojo` | Mojo Metal bandwidth kernels and MAX's Apple kernels called directly from the shipped `linalg` package: `AppleM5MatMul` (Metal 4 MMA on the NA) and `matmul_8x8` (simdgroup, no NA), f16/bf16/f32; `apple_gemv`; int8 W8A8; NVFP4 W4A16. Each result is checked for C == K |
| ane | `ane/coreml_bench.py` | Core ML chains of `linear` layers in fp16, w8, w4 (per-block), lut4 and w8a8, under CPU_ONLY / CPU_AND_GPU / CPU_AND_NE / ALL. Planned device per op comes from `MLComputePlan` |
| ggml | `gpu/ggml_bench.py` | llama.cpp `test-backend-ops perf -o MUL_MAT` on Metal and CPU (f16, bf16, q8_0, q4_0, q4_K) as a cross-check |
| report | `scripts/report.py` | `ATLAS.md` tables, `plots/{bandwidth,square_sweep,decode_gemv,prefill_gemm,roofline}.png` |

Shapes come from `shapes.py`:
- square sizes 256…8192
- Llama 3.2 3B projections at M ∈ {1, 4, 16, 512}, with K×N = 3072×3072, 3072×1024, 3072×8192 and 8192×3072
- lm_head 3072×128256 at M ∈ {1, 512}

Weights are stored [N, K] (nn.Linear). The Mojo file mirrors the shape list, so keep the two in sync.

## Conventions

- Every record has median, min, max, mean, stddev and the rep count.
- GPU timings enqueue `inner` ops back-to-back and synchronize once (pipelined per-op time, as in a model graph), and each rep lasts ~0.1 s.
- `eff_gbps_*` = (weights + activations + output bytes) / time. Use it for M ≤ 16.
- 1 MAC = 2 FLOP. Integer rates are in GOP/s.

## Gotchas found while building this

- Compile `peak.c` with `-mcpu=apple-m4…`, **not** `-march=armv9.x-a+sme2`. armv9 implies SVE, the autovectorizer then emits non-streaming SVE, and Apple cores raise SIGILL on it.
- Mojo GPU kernel args can't be `Int`; use `Int64`. `out` is a reserved argument name. Pointer casts use `unsafe_bitcast`.
- `kbench` is not shipped in the max nightly, but `linalg.mojoc` is. So `max/kernels/benchmarks/gpu/linalg/bench_apple_*.mojo`-style code runs as-is with `mojo build`, with no checkout needed.
- MLX's NAX gate is `arch gen >= 17` and macOS ≥ 26.2. Setting `MLX_METAL_GPU_ARCH=applegpu_g16g` turns it off. MLX fp32 matmul on NAX is TF32 unless `MLX_ENABLE_TF32=0`.
- Core ML per-block int4 weights are *planned on the CPU* even with CPU_AND_NE. LUT-4 (palettized) and int8 weights are planned on the ANE.
- No Xcode means no Instruments or powermetrics, so ANE placement is the plan plus timing, not an observation.
