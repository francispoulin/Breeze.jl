# Rondeau execution, 2026-11-04

The benchmark source is in this repository on the `distributed-benchmark` branch.
Francis Poulin's seven benchmark commits and their three main-line prerequisites
were copied with `git cherry-pick -x`. The imported file tree matches
`francispoulin/Breeze.jl` at `72c9e851105cf0017a16fb3c122a4170f82c5694`.

This run follows the student guide through section 7. Section 8 experiments are
deferred. The configured sweep contains 17 cases, with 10 warmup steps and 500
timed steps at a fixed time step of 0.05 seconds.

## Locations

| Item | Location |
|------|----------|
| Breeze source | `/u/ajtse/Breeze.jl` |
| Benchmark project | `/u/ajtse/Breeze.jl/benchmarking/distributed` |
| Platform | `/u/ajtse/software/oceananigans-hpc` |
| Julia for Breeze | `/u/ajtse/software/julia-1.11.9` |
| Julia package depot | `/u/ajtse/.julia` |
| Run records | `/u/ajtse/Breeze.jl/benchmarking/results/rondeau/student_guide_2026-11-04` |

## Environment

From a new terminal:

```bash
cd /u/ajtse/Breeze.jl
export OCEAN_SOFTWARE_DIR=/u/ajtse/software
export OCEAN_RESULTS_ROOT="$PWD/benchmarking/results/rondeau/student_guide_2026-11-04"
export OCEAN_DATA_DIR=/u/ajtse/software/ocean_data
export JULIA_DEPOT_PATH=/u/ajtse/.julia
export JULIA_LOAD_PATH=@:@stdlib
export TMPDIR=/u/ajtse/software/tmp/breeze-student-guide-2026-11-04/current
mkdir -p "$TMPDIR"
export TMP="$TMPDIR" TEMP="$TMPDIR" OMPI_MCA_orte_tmpdir_base="$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

source /u/ajtse/software/oceananigans-hpc/env/ocean_setup.sh
source benchmarking/distributed/pelee_rondeau/breeze_env.sh

# Prevent detached local shells from reloading this account's Julia 1.12 settings.
unset BASH_ENV SSH_CLIENT SSH_CONNECTION SSH_TTY SSH2_CLIENT
export JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=2
julia --version
```

The last line must report Julia 1.11.9. MPI and CUDA settings come from the
platform. The benchmark's generated `LocalPreferences.toml` and `Manifest.toml`
are in `benchmarking/distributed`; retained copies record the measured environment.

## Run on four GPUs

The platform selects free GPUs and refuses to launch if fewer than four are free:

```bash
cd "$OCEAN_RESULTS_ROOT/runs"
bash "$OCEAN_HPC_ROOT/tools/run_local.sh" --wait -n 4 \
    -p "$BREEZE_DISTRIBUTED_DIR" \
    "$BREEZE_DISTRIBUTED_DIR/distributed_cbl_benchmark.jl" \
    --dynamics=compressible_splitexplicit --size=256x256x128 \
    --partition=2x2 --time_steps=500
```

For a fresh full sweep, use a new CSV so that completed measurements are not
appended twice:

```bash
bash "$BREEZE_DISTRIBUTED_DIR/pelee_rondeau/run_scaling.sh" \
    --time_steps=500 --max_gpus=4 \
    --results_file="$OCEAN_RESULTS_ROOT/scaling-new.csv"
```

To change the time step for every case, add `--time_step=0.03`. The default
remains 0.05 seconds. With 500 timed steps, these settings cover 15 and 25
simulated seconds respectively; timing per step does not compare equal physical
durations.

## Recorded outcomes

The [run records](../../results/rondeau/student_guide_2026-11-04/) contain the CSV,
logs, source revisions, package manifests, and environment checks. A completed
process can still produce an unstable solution: every result with non-finite
`max_abs_w` is excluded from performance and efficiency comparisons.

The first 13 cases ran before the source was copied into this branch. Their
original logs retain the historical paths. The source files are identical, and
the remaining four cases continue from this repository with the same settings.

The requested [2026-11-05 rerun at Δt = 0.03 s](../../results/rondeau/student_guide_2026-11-05_dt0p03/)
uses this primary repository for all 17 cases. Every case completed with finite
maximum vertical velocity. The report preserves the CSV, comparison, plots,
environment, and command logs separately from the 0.05-second run.

The subsequent [Δt = 0.02 s rerun](../../results/rondeau/student_guide_2026-11-05_dt0p02/)
also completed all 17 cases with finite maximum vertical velocity. It retains the
same grids, partitions, 10 warmup steps, 500 timed steps, and 12 acoustic substeps,
covering 10 simulated seconds per timed window. The results include a comparison
with the 0.03-second run, raw logs, provenance, and updated split-explicit plots.

The [2026-11-07 single-GPU capacity search](../../results/rondeau/single_gpu_maximum_resolution_2026-11-07/)
keeps Float32 precision and the power-of-two grid family Nx = Ny = 2 Nz. All
three dynamics completed 10 warmup and 10 timed steps at 512×512×256. The next
grid, 1024×1024×512, failed with GPU out-of-memory errors for all three. The
report records observed device-memory usage, commands, and both successful and
failed logs. These short runs establish memory-fit bounds at Δt = 0.02 s.

The [512³ follow-up](../../results/rondeau/single_gpu_512_cubed_2026-11-07/)
fits all three dynamics on one GPU with Float32. Anelastic and split-explicit
cases retained finite maximum vertical velocity at Δt = 0.02 s. The explicit
case produced non-finite velocity at that step size; a separate Δt = 0.005 s
retry completed with finite velocity. All attempts used 10 warmup and 10 timed
steps, with both explicit outcomes retained in the report.

The [MPI/NCCL weak-scaling study](../../results/rondeau/weak_scaling_mpi_nccl_2026-11-08_dt0p02/)
completed all ten cases at Δt = 0.02 s, with 512³ cells per GPU on 1, 2, and
4 GPUs, 10 warmup steps, and 500 timed steps. All five split-explicit cases
retained finite prognostic and velocity fields. On four GPUs, MPI measured
1.495404 s/step and NCCL 1.395613 s/step (81.4% and 87.3% weak efficiency).
All five explicit cases produced non-finite fields and are excluded from
performance metrics. The report includes the complete plots, two- and four-GPU
NCCL halo checks, raw measurements, logs, and provenance.
