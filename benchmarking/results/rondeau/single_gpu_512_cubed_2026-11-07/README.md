# Rondeau single-GPU 512³ capacity test — 2026-11-07

**512×512×512 (134,217,728 cells) fits all three dynamics on one Rondeau A100-SXM4-40GB, using Float32.** Anelastic and split-explicit retained finite maximum vertical velocity at Δt = 0.02 s. Explicit produced NaNs at 0.02 s; a separate retry at 0.005 s retained finite velocity.

This cubic grid contains twice the cells of the [preceding 512×512×256 tests](../single_gpu_maximum_resolution_2026-11-07/README.md). The earlier maximum was defined along a fixed 2:2:1 sequence; this follow-up tests the requested cubic shape.

[Attempt records](attempts.csv) · [Study and validation](study.json) · [Memory samples](memory_samples.csv) · [Driver results](measurements.csv) · [Commands](execution/commands.log)

## Results

| Dynamics | Δt (s) | Outcome | Observed peak device memory (GiB) | Final max abs w (m/s) |
|---|---:|---|---:|---:|
| Anelastic | 0.02 | [Fits VRAM; finite w](runs/logs/capacity_anelastic_512x512x512_20261107-200655.out) | 15.71 | 0.00238 |
| Compressible explicit | 0.02 | [Fits VRAM; non-finite w](runs/logs/capacity_compressible_explicit_512x512x512_20261107-200929.out) | 16.41 | non-finite |
| Compressible split-explicit | 0.02 | [Fits VRAM; finite w](runs/logs/capacity_compressible_splitexplicit_512x512x512_20261107-201202.out) | 25.94 | 0.004751 |
| Compressible explicit | 0.005 | [Fits VRAM; finite w](runs/logs/capacity_compressible_explicit_512x512x512_dt0p005_20261107-201617.out) | 16.41 | 0.004821 |

## Conditions and scope

Each case uses one fresh Julia process on physical GPU 1, Float32, WENO5, the existing dry convective-boundary-layer physics, no turbulence closure or microphysics, partition 1×1, one Julia thread, and CUDA memory pooling disabled. Split-explicit dynamics uses 12 acoustic substeps. The GPU had no compute process before each launch.

The initial memory-fit tests run **10 warmup steps plus 10 timed steps at Δt = 0.02 s** (0.4 simulated seconds). The explicit case produced non-finite vertical velocity at this step size. Its separate retry uses the same step counts at **Δt = 0.005 s** (0.1 simulated seconds); both outcomes are retained. Memory fit means construction and all 20 steps complete without a GPU allocation failure. Each run draws fresh unseeded initial perturbations. Maximum vertical velocity is the driver's limited numerical check. Full-field accuracy and long-duration stability are not evaluated; the timing CSV is a diagnostic from one short timing window per case.

Observed peak memory is total device usage sampled at approximately one-second intervals, including CUDA context overhead. Short allocation peaks can be missed.

The existing platform launcher and tracked `benchmarking/distributed/distributed_cbl_benchmark.jl` driver perform every run. Scientific source and package files remained unchanged. Project, Manifest, and MPI preferences match the preceding capacity search byte for byte. Its [verified environment log](../single_gpu_maximum_resolution_2026-11-07/provenance/environment.log) records Julia 1.11.9 and CUDA-aware Open MPI 4.1.7. GPU 1 was released when the tests ended.

## Reproduction and evidence

[execution/commands.log](execution/commands.log) records the environment and every exact command. Use a fresh results directory and CSV when repeating the tests. With that environment sourced, the explicit case is:

```bash
cd "$OCEAN_RESULTS_ROOT/runs"
bash "$OCEAN_HPC_ROOT/tools/run_local.sh" --wait -n 1 --gpu-ids 1 \
    -p "$BREEZE_DISTRIBUTED_DIR" \
    "$BREEZE_DISTRIBUTED_DIR/distributed_cbl_benchmark.jl" \
    --dynamics=compressible_explicit --size=512x512x512 --partition=1x1 \
    --float_type=Float32 --warmup_steps=10 --time_steps=10 \
    --time_step=0.005 --substeps=12 \
    --results_file="$OCEAN_RESULTS_ROOT/measurements_dt0p005.csv"
```

Use `--dynamics=anelastic` or `--dynamics=compressible_splitexplicit` with `--time_step=0.02` for the initial cases. The original explicit `--time_step=0.02` command and its non-finite result remain in the records.

Validation checks model/grid coverage, terminal statuses, commands and GPU selection, CSV/log agreement, memory-sample peaks, unchanged source hashes, and environment identity. [raw_logs.zip](raw_logs.zip) retains the original log bytes; [log_originals.json](log_originals.json) records sizes and SHA-256 hashes. Display copies remove ANSI escape sequences and normalize trailing whitespace. CUDA library-path warnings and PMIx cleanup messages are retained.
