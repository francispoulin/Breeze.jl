# Distributed (multi-GPU) benchmark

`distributed_cbl_benchmark.jl` times the convective boundary layer (CBL) case of
[`../src/convective_boundary_layer.jl`](../src/convective_boundary_layer.jl) on several GPUs, one GPU per
MPI rank, using Oceananigans' `Distributed` or `NCCLDistributed` architecture. It calls `time_step!` directly, with no
Reactant or `Simulation` overhead, and reports points per second. The single-GPU benchmark in
`../run_benchmarks.jl` does not support `Distributed`, which is why this is a separate driver.

## Status

| Dynamics | Distributed? |
|----------|--------------|
| `compressible_explicit` | Yes. Run on 1 A100 GPU (256×256×128) and on 1 and 2 (128×128×64, partition 2x1); results finite. |
| `compressible_splitexplicit` | Yes. Run on 1 and 2 A100 GPUs (256×256×128, partition 2x1); results finite. Not yet compared field by field with a serial run (the case's initial noise uses `rand()`, which differs between ranks). |
| `anelastic` | **On this branch only**, which merges Breeze PR [#1157](https://github.com/NumericalEarth/Breeze.jl/pull/1157) (not in Breeze `main` yet). On several ranks the pressure solve is Oceananigans' `DistributedFourierTridiagonalPoissonSolver`: a solve takes 2 transposes (MPI `Alltoall`) with `Rx×1`, 4 with `1×Ry` and 8 with `Rx×Ry`. Distributed and 1-GPU runs agree (`max_abs_w` to 7 digits at 1×1, 2×1 and 4×1, Float64, `--random_perturbation=false`, 100 steps, pelee and rondeau, Nov 11, 2026). |

## Requirements

- Julia ≥ 1.11.9.
- MPI.jl configured to match the system launcher. The MPI backend requires a **CUDA-aware** MPI;
  `MPI.has_cuda()` should return `true`.
- The NCCL backend requires NCCL.jl (included in this benchmark project). `--comm=NCCL` loads
  Oceananigans' NCCL extension and verifies the communicator before constructing the model.
- At least as many GPUs as ranks. Each rank must see its own GPU (`CUDA_VISIBLE_DEVICES` per rank, or a
  launcher that assigns devices).

## Usage

From the root of the repository, install once, then run:

```bash
julia --project=benchmarking/distributed -e 'using Pkg; Pkg.instantiate()'

mpiexec -n 4 julia --project=benchmarking/distributed \
    benchmarking/distributed/distributed_cbl_benchmark.jl \
    --dynamics=compressible_splitexplicit --size=512x512x128 --partition=2x2
```

The environment in this folder takes Breeze from the checkout (`[sources]`), so it tests the code you
have checked out.

| Option | Default | Description |
|--------|---------|-------------|
| `--dynamics` | `compressible_splitexplicit` | `anelastic`, `compressible_explicit` or `compressible_splitexplicit` |
| `--size` | `256x256x128` | Global grid, `NxNyxNz` |
| `--partition` | `<ranks>x1` | Ranks in x and y, `RxxRy`; must multiply to the number of ranks |
| `--comm` | `MPI` | `MPI` or `NCCL` for multi-GPU halo exchange; one rank uses the shared serial GPU baseline |
| `--time_steps` | `100` | Timed steps |
| `--warmup_steps` | `10` | Untimed steps; they include compilation |
| `--time_step` | `0.05` | Time step in seconds |
| `--substeps` | `12` | Acoustic substeps (split-explicit only) |
| `--float_type` | `Float32` | `Float32` or `Float64` |
| `--simplified` | `false` | Drop the geostrophic forcing and the field-dependent drag boundary conditions |
| `--random_perturbation` | `true` | `false`: smooth (sine) initial perturbation instead of `rand()`, which differs between ranks. Use it to compare runs on different partitions |
| `--profile` | `false` | Start/stop an external CUDA profiler around the timed steps, after warmup |
| `--nvtx` | `false` | Add NVTX ranges to selected Breeze host methods and halo/solver calls |
| `--random_perturbation` | `true` | Use random initial noise; `false` selects a deterministic perturbation |
| `--results_file` | none | Append one CSV line of results to this file |

For NCCL communication on two GPUs:

```bash
mpiexec -n 2 julia --project=benchmarking/distributed \
    benchmarking/distributed/distributed_cbl_benchmark.jl \
    --comm=NCCL --dynamics=compressible_splitexplicit \
    --size=1024x512x512 --partition=2x1 --time_step=0.02
```

MPI still launches ranks and supplies timing barriers and reductions. NCCL performs GPU halo
communication. The result records the effective backend as `SERIAL`, `MPI`, or `NCCL`; a one-GPU
run has no inter-rank communication and can serve as the reference for both distributed backends.

## Nsight Systems

Pass `--profile=true` and launch each rank under `nsys profile` with
`--capture-range=cudaProfilerApi --capture-range-end=stop`. The driver captures exactly
`--time_steps` steps after `--warmup_steps`, including the final CUDA synchronization.
Model construction, warmup/JIT, and the final field checks are outside the capture.

```bash
CUDA_VISIBLE_DEVICES=1 nsys profile --trace=cuda,nvtx,mpi --mpi-impl=openmpi \
    --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop \
    --output=/local_scratch/ajtse/cbl_explicit_1gpu \
    julia --project=benchmarking/distributed benchmarking/distributed/distributed_cbl_benchmark.jl \
    --dynamics=compressible_explicit --size=512x512x512 --partition=1x1 \
    --time_step=0.005 --warmup_steps=10 --time_steps=10 --profile=true
```

For several ranks, place `nsys profile` inside the MPI launch and give each rank a unique
output prefix. The recorded timing includes profiling overhead and belongs in a separate
results file from unprofiled timings. One GPU remains the shared plain-GPU serial baseline.
For each study, list the exact `.nsys-rep` paths in its `README.md`; do not generate plots
for Nsight runs.

Add `--nvtx=true` to attribute launches to Breeze methods using
[NVTX.jl](https://github.com/JuliaGPU/NVTX.jl). The benchmark re-evaluates an explicit
list of host methods from the current checkout with `NVTX.@range` around their bodies
and halo/solver calls, before model construction and warmup. This replaces methods
only in the profiling process; it changes no source files, kernel bodies, or device
synchronization. It is intended for eager CUDA benchmarks, not Reactant tracing.
The instrumentation fails if a listed method disappears after a source change.

NVTX durations describe CPU submission and blocking. Attribute GPU execution using
CUDA launch correlation IDs rather than treating the host range duration as GPU time.
Nested inclusive ranges must not be added together. Enable `JULIA_NVTX_CALLBACKS=gc`
to record Julia GC events too. `validate_nvtx.jl` checks exact equality of all prognostic
and velocity fields with and without instrumentation for the three serial dynamics:

```bash
julia --project=benchmarking/distributed benchmarking/distributed/validate_nvtx.jl
```

The [one-GPU 512³ NVTX demo](../results/rondeau/nvtx_demo_512_cubed_2026-11-10/README.md)
includes a validated capture and readable summaries with times in milliseconds.

## Reading the output

Rank 0 prints one line:

```
RESULT machine=... dynamics=... float=... size=... partition=... gpus=... steps=... seconds_per_step=... points_per_second=... points_per_second_per_gpu=... max_abs_w=... communication=... fields_finite=...
```

- `points_per_second` uses the **global** number of grid points.
- `max_abs_w` is a blow-up check. If it is not finite, the timing is meaningless. With
  `--random_perturbation=false` it is also a quick consistency check: runs of the same case on different
  partitions should agree to many digits.
- `fields_finite` checks every prognostic and velocity field on every rank after timing. A false
  value excludes the result from performance comparisons, even when the process exits successfully.
- CSV rows include `communication` and `fields_finite`. Use a fresh results CSV when switching
  from an older driver version; incompatible headers are rejected before model construction.
- Small grids are dominated by kernel-launch and halo-exchange latency, so more GPUs can be slower. Judge
  scaling on grids of several million points per GPU, with at least a few hundred timed steps.

`plot_split_explicit_results.py --compare-comm` compares MPI and NCCL weak-scaling times and
efficiencies, using the shared serial reference. `--comm=MPI` or `--comm=NCCL` selects one backend
for the original scaling layout. Plots reject non-finite fields and require an explicit backend
choice when the CSV contains both MPI and NCCL.

## pelee and rondeau (Waterloo fluids group)

`pelee_rondeau/` sets this up on top of the oceananigans-hpc platform, which provides Julia (1.13.2), the
modules, the Julia depot, the system OpenMPI and `tools/run_local.sh` (free-GPU selection, one GPU per rank, logging). It is
specific to those machines and can be ignored elsewhere. `Manifest.toml` is tracked in this folder: it is
the exact package versions the benchmark numbers were measured with (made with the platform's Julia).

```bash
ocean_setup
bash benchmarking/distributed/pelee_rondeau/setup_breeze.sh        # once; packages, system MPI, checks
source benchmarking/distributed/pelee_rondeau/breeze_env.sh        # in every new terminal, after ocean_setup

bash $OCEAN_HPC_ROOT/tools/run_local.sh -n 2 -p $BREEZE_DISTRIBUTED_DIR \
    $BREEZE_DISTRIBUTED_DIR/distributed_cbl_benchmark.jl --dynamics=compressible_explicit --size=256x256x128

bash benchmarking/distributed/pelee_rondeau/run_scaling.sh         # strong and weak scaling sweep
```
