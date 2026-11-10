# Rondeau single-GPU maximum resolution — 2026-11-07

**All three dynamics fit at 512×512×256 and fail with GPU out-of-memory errors at 1024×1024×512.** Thus the maximum in the requested power-of-two sequence is **512×512×256 (67,108,864 cells)** for each dynamics.

The selected grid family keeps Nx = Ny = 2 Nz and doubles all three dimensions at each increase. Each increase has eight times as many cells, so the accepted maximum leaves VRAM headroom.

[Attempt records](attempts.csv) · [Study and validation](study.json) · [Memory samples](memory_samples.csv) · [Driver results](measurements.csv) · [Commands](execution/commands.log)

## Results

| Dynamics | Largest fitting grid | Cells | Observed peak device memory (GiB) | Final max abs w (m/s) | Next grid |
|---|---|---:|---:|---:|---|
| Anelastic | [512×512×256](runs/logs/capacity_anelastic_512x512x256_20261107-193757.out) | 67,108,864 | 8.23 | 0.002775 | [1024×1024×512: GPU OOM](runs/logs/capacity_anelastic_1024x1024x512_20261107-194026.out) |
| Compressible explicit | [512×512×256](runs/logs/capacity_compressible_explicit_512x512x256_20261107-194131.out) | 67,108,864 | 8.61 | 0.005249 | [1024×1024×512: GPU OOM](runs/logs/capacity_compressible_explicit_1024x1024x512_20261107-194359.out) |
| Compressible split-explicit | [512×512×256](runs/logs/capacity_compressible_splitexplicit_512x512x256_20261107-194502.out) | 67,108,864 | 13.48 | 0.005058 | [1024×1024×512: GPU OOM](runs/logs/capacity_compressible_splitexplicit_1024x1024x512_20261107-194806.out) |

## Conditions and scope

One NVIDIA A100-SXM4-40GB on Rondeau, physical GPU 1, was used sequentially for all attempts. That device had no compute process before each launch. GPU 0 was already occupied by another workload. Each attempt starts a fresh Julia process through the existing platform launcher and tracked `benchmarking/distributed/distributed_cbl_benchmark.jl` driver.

All attempts use Float32, WENO5, the existing dry convective-boundary-layer physics, no turbulence closure or microphysics, partition 1×1, one Julia thread, and CUDA memory pooling disabled. Split-explicit dynamics uses 12 acoustic substeps. Julia is 1.11.9; package versions and MPI preferences are retained in `provenance/`.

A fit requires successful model construction, **10 warmup steps plus 10 timed steps at Δt = 0.02 s**, and normal process exit. Each fitting case completed all 20 steps (0.4 simulated seconds) and reported finite maximum vertical velocity. This is a capacity test with a short velocity check; full-field accuracy and long-duration stability were not tested. The per-step timings in the driver CSV are diagnostics from a single ten-step timing window.

The device-memory column is the maximum total GPU 1 usage observed by `nvidia-smi` at approximately one-second intervals, including CUDA context overhead. Short allocation peaks can be missed. The next larger grid was actually attempted for every dynamics and failed with an explicit CUDA out-of-memory error during construction. Missing numerical results for these failed constructions are recorded as empty CSV fields and JSON nulls.

The scientific source, benchmark driver, package environment, and platform launcher remained unchanged. Validation checks include adjacent power-of-two fit/OOM bounds, terminal statuses, commands and GPU selection, CSV/log agreement, recorded memory peaks, and source/environment identity. GPU 1 was released after the search.

## Reproduction

[execution/commands.log](execution/commands.log) records the environment and every exact launch command. Use a fresh results directory and CSV when repeating the search. After sourcing that environment, a fitting explicit case can be run with:

```bash
cd "$OCEAN_RESULTS_ROOT/runs"
bash "$OCEAN_HPC_ROOT/tools/run_local.sh" --wait -n 1 --gpu-ids 1 \
    -p "$BREEZE_DISTRIBUTED_DIR" \
    "$BREEZE_DISTRIBUTED_DIR/distributed_cbl_benchmark.jl" \
    --dynamics=compressible_explicit --size=512x512x256 --partition=1x1 \
    --float_type=Float32 --warmup_steps=10 --time_steps=10 \
    --time_step=0.02 --substeps=12 \
    --results_file="$OCEAN_RESULTS_ROOT/measurements.csv"
```

Repeat with `--size=1024x1024x512` to reproduce the next-size GPU memory failure. Use `--dynamics=anelastic` or `--dynamics=compressible_splitexplicit` for the other dynamics. An expected OOM attempt exits with a nonzero status; the three such failures establish the upper bounds of this completed search.

[raw_logs.zip](raw_logs.zip) preserves original log bytes, with sizes and SHA-256 hashes in [log_originals.json](log_originals.json). Display copies remove ANSI escape sequences and normalize trailing whitespace. CUDA library-path warnings and PMIx cleanup messages remain in the retained logs.
