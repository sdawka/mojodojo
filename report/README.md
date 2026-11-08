# report: Mojo vs Apple Silicon comparison page

`build.py` writes one self-contained HTML page, `~/Desktop/mojo-vs-apple-silicon.html` by default.
All CSS, JS, SVG and data are inline. The page loads no fonts or CDN assets and makes no network
requests. It is regenerated from the result JSON in this repo and reads nothing else, so re-run it
after any lane drops new results.

```sh
cd /Users/sdawka/Code/llming
uv run --project report python report/build.py                 # -> ~/Desktop/mojo-vs-apple-silicon.html
uv run --project report python report/build.py -o /tmp/x.html  # other output path
LLMING_ROOT=/path/to/fixture uv run --project report python report/build.py -o x.html  # alternate data root
```

It uses only the Python standard library (`pyproject.toml` has no dependencies). The build never
runs benchmarks. Missing or malformed inputs give a placeholder (a dash or a "pending" panel), not
an error.

## Page sections

1. Header: the machine (`atlas/results/inventory.json`; serial number and UUID are left out),
   toolchain versions, generation date, and data freshness (sources counted by status).
2. Part 1: the pipelines as aligned lanes, stage by stage: Mojo `mojo build` / MAX graph, and
   Apple's MLX / Core ML / Metal / Accelerate. Each cell is tagged open source / closed /
   needs Xcode / inferred. The cell text is curated in `rlib/pipeline.py` from
   `notes/toolchain-brief.md`, `mojo-max/README.md` and `atlas/README.md`.
3. Part 2: Phase-1 LLM decode tok/s and TTFT by format (MAX, llama.cpp, MLX), memory bandwidth,
   matmul on the same shapes across toolchains, and a roofline drawn as inline SVG in Python.
4. Part 3: Mojo on each engine: CPU, GPU simdgroup vs Neural Accelerators, int8/fp4, the int8 tile
   ladder (G1-A to G1-E), FPGA synthesis, the Phantom NPU simulator, and the DOOM ladder
   (X-DOOM-1 to X-DOOM-4).
5. Caveats, generated from the inputs, and the source list.

## Provenance

Every number goes through `Reg.num()` and carries a source id. Hover, focus or tap a number to see
its file, date and status. Each figure caption lists its sources and status badges.

| status | meaning | set by |
|---|---|---|
| `authoritative` | full run on an idle machine | `authoritative: true` (top level or `meta`) |
| `nonidle` | full run, idle not confirmed | `authoritative: false` |
| `smoke` | short validation run | `smoke: true` (top level or `meta`) |
| `measured` | real measurement with no idle flag | default |
| `derived` | computed by the report | formula sources |
| `doc` | quoted from a README | regex fallbacks |
| `pending` | build output from a lane still running | FPGA `build/synth` fallback |

`smoke` is checked first, then `authoritative`. Dates come from `timestamp`, `meta.timestamp`
or `date`, falling back to the file's mtime.

## Data contract

The paths below are relative to the repo root. Unknown keys are ignored. For a given key, the
newest file wins.

### `bench/results/*.json`: Phase-1 LLM (Part 2)

There are three accepted shapes. The newest file per (runtime, format) is used.

- llama-bench JSON: a list of `{n_prompt, n_gen, avg_ts, stddev_ts, samples_ts, test_time,
  model_size, model_type}`. The runtime is `llama.cpp`; the format comes from the file name.
  Peak RSS is read from a sibling `<name>.time.txt` (`/usr/bin/time -l`), if one exists.
- Generic:
  `{runtime, timestamp, versions:{...}, authoritative?, smoke?, results:[{format, pp_tps_mean,
  pp_tps_std, tg_tps_mean, tg_tps_std, ttft_s_mean, ttft_s_std, peak_mem_gb, reps}]}`.
  `runtime` is one of `mlx`, `max`, `llama-server`, `llama.cpp`.
- `max benchmark` JSON: `{date, mean_ttft_ms, mean_tpot_ms, total_input_tokens, completed,
  num_prompts, metadata:{runtime, format, endpoint}}`. Decode tok/s is computed as `1000 / mean_tpot_ms`.

`format` is a quant label such as `bf16`, `q8_0` or `q4_k_m`. It is grouped into 16-bit, 8-bit
or 4-bit.

### `atlas/results/*.json`: kernels, bandwidth, roofline (Parts 2–3)

```json
{"meta": {"timestamp": "...", "smoke": false, "authoritative": true, "load_avg_at_start": [1.2, 1.1, 1.0]},
 "records": [{"bench": "matmul|bandwidth|...", "engine": "gpu|cpu|ane|cpu_mojo",
              "kernel": "...", "mode": "...", "dtype": "bf16|int8|fp4|...",
              "unit": "GFLOP/s|GB/s|...", "shape": "attn_q_o_M512",
              "M": 512, "N": 3072, "K": 3072, "median": 0.0, "stddev": 0.0, "reps": 0,
              "eff_gbps_median": 0.0, "bytes_per_call": 0, "threads": 0, "qos": "", "op": "", "instr": ""}]}
```

File names follow `atlas/scripts/report.py` `atlas_path()`: `mlx_nax`, `mlx_nonax`, `max_mma`,
`max_8x8`, `max_apple_int8_w8a8`, `max_apple_fp4_w4a16`, `max_apple_gemv`, `sme`, `coreml_*`.
As in that script, a full-run file overrides a `*_smoke.json` file for the same key.
`inventory.json` supplies the header (`summary`, `versions`, `apple_spec`).
The Mojo CPU column in Part 3 shows a C reference value until records with
`engine: "cpu_mojo"` appear.

### `fpga/results/*.json`: tile and synthesis (Part 3)

| file | shape | fallback when absent |
|---|---|---|
| `synth_ecp5.json` | `{meta:{timestamp, tools}, records:[{N, variant, lut4, ff, ccu2c, mult18, fmax_mhz, device, pnr}]}` | `fpga/build/synth/n*.core.stat` (yosys) + `n*.report.json` (nextpnr), status `pending` |
| `perf.json` | `{meta, records:[{N, M, K, N_out, passes, cycles, pe_util, macs}]}` | DESIGN.md §5 formula `ceil(M/N)·ceil(N_out/N)·(K+3N−1)` at M=512, K=3072, N_out=8192, status `derived` |
| `cosim.json` | `{meta, traces, cycles, mismatches}` | regex on `fpga/README.md`, status `doc` |
| `phantom.json` | `{meta, status, summary, gops}` | "pending" panel |

### `doom/results/*.json`: DOOM ladder (Part 3)

One object, or a list of them, per file:

```json
{"tier": "X-DOOM-1", "status": "not-started|in-progress|pass|fail",
 "fps": 35.0, "fps_reference": 35.0, "frame": "x-doom-1.png",
 "date": "2026-11-08T12:00:00Z", "notes": "...", "backend": "Mojo FFI + C renderer",
 "authoritative": true}
```

`frame` is relative to `doom/results/`. It can be png, jpg, webp or gif, and is embedded as a data
URI if it is 400 KB or smaller (otherwise the page notes that it was skipped). The newest file for
each tier wins. Tiers with no file show "not started".

## Layout

- `build.py`: page assembly, figures, caveats
- `rlib/sources.py`: source registry, number formatting, status badges
- `rlib/loaders.py`: one loader per results directory
- `rlib/charts.py`: HTML bar charts and the SVG roofline
- `rlib/pipeline.py`: Part 1 lane and cell content
- `assets/style.css`, `assets/page.js`: inlined into the page (theme tokens, tooltip)
