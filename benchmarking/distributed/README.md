# Distributed (multi-GPU) benchmark

`distributed_cbl_benchmark.jl` times the convective boundary layer (CBL) case of
[`../src/convective_boundary_layer.jl`](../src/convective_boundary_layer.jl) on several GPUs, one GPU per
MPI rank, using Oceananigans' `Distributed` architecture. It calls `time_step!` directly, with no
Reactant or `Simulation` overhead, and reports points per second. The single-GPU benchmark in
`../run_benchmarks.jl` does not support `Distributed`, which is why this is a separate driver.

## Status

| Dynamics | Distributed? |
|----------|--------------|
| `compressible_explicit` | Yes. Run on 1 and 2 A100 GPUs (128×128×64, partition 2x1); results finite. |
| `compressible_splitexplicit` | Expected to work (it only needs halo exchange); not yet verified at scale. |
| `anelastic` | **No.** `AnelasticDynamics` builds a serial `FourierTridiagonalPoissonSolver` on each rank's local grid, which does not solve the global pressure problem, so the driver refuses it. Work in progress in Breeze PR [#1157](https://github.com/NumericalEarth/Breeze.jl/pull/1157), which depends on Oceananigans PR [#6385](https://github.com/CliMA/Oceananigans.jl/pull/6385). |

## Requirements

- Julia ≥ 1.11.9.
- MPI.jl configured to use a **CUDA-aware** system MPI. Without it, GPU halo exchange does not work.
  `MPI.has_cuda()` should return `true`.
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
| `--dynamics` | `compressible_splitexplicit` | `compressible_explicit` or `compressible_splitexplicit` |
| `--size` | `256x256x128` | Global grid, `NxNyxNz` |
| `--partition` | `<ranks>x1` | Ranks in x and y, `RxxRy`; must multiply to the number of ranks |
| `--time_steps` | `100` | Timed steps |
| `--warmup_steps` | `10` | Untimed steps; they include compilation |
| `--time_step` | `0.05` | Time step in seconds |
| `--substeps` | `12` | Acoustic substeps (split-explicit only) |
| `--float_type` | `Float32` | `Float32` or `Float64` |
| `--simplified` | `false` | Drop the geostrophic forcing and the field-dependent drag boundary conditions |
| `--results_file` | none | Append one CSV line of results to this file |

## Reading the output

Rank 0 prints one line:

```
RESULT dynamics=... size=... partition=... gpus=... seconds_per_step=... points_per_second=... points_per_second_per_gpu=... max_abs_w=...
```

- `points_per_second` uses the **global** number of grid points.
- `max_abs_w` is a blow-up check. If it is not finite, the timing is meaningless.
- Small grids are dominated by kernel-launch and halo-exchange latency, so more GPUs can be slower. Judge
  scaling on grids of several million points per GPU, with at least a few hundred timed steps.

## pelee and rondeau (Waterloo fluids group)

`pelee_rondeau/` sets this up on top of the oceananigans-hpc platform, which provides the modules, the Julia
depot, the system OpenMPI and `tools/run_local.sh` (free-GPU selection, one GPU per rank, logging). It is
specific to those machines and can be ignored elsewhere.

```bash
ocean_setup
bash benchmarking/distributed/pelee_rondeau/setup_breeze.sh        # once; installs Julia >= 1.11.9, MPI setup
source benchmarking/distributed/pelee_rondeau/breeze_env.sh        # in every new terminal, after ocean_setup

bash $OCEAN_HPC_ROOT/tools/run_local.sh -n 2 -p $BREEZE_DISTRIBUTED_DIR \
    $BREEZE_DISTRIBUTED_DIR/distributed_cbl_benchmark.jl --dynamics=compressible_explicit --size=256x256x128

bash benchmarking/distributed/pelee_rondeau/run_scaling.sh         # strong and weak scaling sweep
```
