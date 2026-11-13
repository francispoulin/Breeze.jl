# Fir MPI weak scaling

This series runs anelastic, compressible explicit (CE), and compressible split-explicit
(CSE) with MPI on 1, 2, 4, 8, and 16 H100 GPUs. Every rank owns 512³ cells;
the global grid is (512N)×512×512 with N×1 partition. The physical domain stays
12 km × 12 km × 3 km. All dynamics use Δt = 0.01/N seconds, including anelastic.

The existing distributed CBL driver runs full dry physics, WENO5, Float32,
deterministic smooth perturbations, 10 warmup and 500 timed steps. CSE uses
12 acoustic substeps. Each dynamics has a serial GPU baseline. There are 15
cases and no profiling. Timings represent one window per case.

Source `environment.sh`, then run `julia --startup-file=no benchmarking/distributed/fir/setup.jl`
with its output redirected to `$BREEZE_FIR_ENV/setup.log`. This resolves a separate
Julia 1.12.5 environment under `.runtime`, preserving the tracked Project/Manifest.
Both MPI.jl and OpenMPI_jll use Fir's system CUDA-aware OpenMPI.
Generated environments, depots, caches, and temporary files stay under `.runtime`;
the existing user depot is a fallback for already installed packages.

After checking the source and committing/pushing it on `benchmark-results`:

```bash
source benchmarking/distributed/fir/environment.sh
study="$BREEZE_ROOT/benchmarking/results/fir/weak_scaling_mpi_2026-11-12_nx1"
python3 benchmarking/distributed/fir/study.py prepare "$study"
python3 benchmarking/distributed/fir/study.py submit "$study"
```

Preparation freezes a Git source archive, extracts an isolated source copy, copies
the resolved Julia environment with Breeze pointing at that copy, and records
source/environment hashes. The worker verifies the hashes before and after every
case. Source files in the live checkout can subsequently change without affecting
the frozen jobs. Use a fresh study directory for a new series.

Five independent Slurm jobs allocate exactly 1, 2, 4, 8, and 16 GPUs respectively.
Each runs the three dynamics sequentially. Allocations request eight CPU cores
per rank, 64 GiB of host memory per GPU, and three hours. Every Julia/BLAS/OpenMP
process has one thread. Every `srun` step explicitly requests the allocation's
GPUs with `--gpus-per-node=h100:N` and `--gpu-bind=none`, retaining GPU peer copies.
Oceananigans selects devices by node-local rank. An all-rank preflight rejects
shared/missing GPUs and requires all allocated node GPUs to be visible before
model construction. Each case has a 50-minute
limit. A failed case is recorded and the remaining cases still run.

Validation reuses the existing per-rank validator and additionally checks dynamics,
precision, grid, partition, finite maximum velocity, and H100 hardware evidence.
Each accepted case must contain all ranks' unique placement and finite-field
checks plus one valid timing row. A finite solution is a performance sanity check,
not a field-by-field accuracy comparison.

The CPU finalizer depends on all five allocations with `afterany`. It revalidates
evidence, collects Slurm accounting, marks missing/unfinished cases explicitly,
generates CSV/PNG/SVG comparisons for all three dynamics, and commits/pushes the
study on `benchmark-results`. Its active stdout remains in `.runtime`, so publication
does not stage its own changing log. The finalizer never force-pushes. Publication fails
visibly if the checkout is on a different branch or cannot fast-forward.
The finalizer runs as a Slurm job and survives closing the interactive session.

Regenerate an intermediate report with `python3 benchmarking/distributed/fir/study.py report "$study"`.
After allocations end, finalization is idempotent with
`python3 benchmarking/distributed/fir/study.py finalize "$study" --publish`.
Do not finalize while any allocation is queued or running.

Run the CPU contract checks using the sourced environment, so temporary files
stay under `.runtime`:

```bash
python3 -m unittest discover -s benchmarking/distributed/fir -p 'test_*.py' -v
```
