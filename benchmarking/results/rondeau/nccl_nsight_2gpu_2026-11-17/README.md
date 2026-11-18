# Two-GPU NCCL Nsight: Breeze anelastic and Oceananigans nonhydrostatic

Status: **COMPLETE**.

Global grid 1024×512×512, partition 2×1, 512³ cells per A100, NCCL halo and
transpose communication (MPI only launches ranks and supplies barriers). Each launch
has ten warmup and ten measured steps and installs host NVTX ranges.

| Case | Model | Precision | Δt |
|------|-------|-----------|----|
| `anelastic` | Breeze `AtmosphereModel`, `AnelasticDynamics`, dry CBL, WENO5 | Float32 | 0.005 s |
| `oceananigans_nonhydrostatic` | Oceananigans `NonhydrostaticModel`, suite `rectilinear_buoyancy`, WENO5 | Float64 | 10 s |

Both pressure solves are Oceananigans' distributed FFT solvers, whose transposes
use NCCL here. Δt = 0.005 s follows the AGENTS.md rule for Breeze at this grid;
the Oceananigans case keeps its suite's Float64 and Δt = 10 s (1 km × 1 km × 10 m
cells), so its cost per step matches the Oceananigans weak-scaling series. Both use
Oceananigans 0.117.2 from this benchmark environment, not the Oceananigans checkout.
Compare the two models by where time goes, not by absolute speed.

For each case the order is unprofiled NCCL benchmark, then NCCL Nsight capture.
Nsight records CUDA, NVTX and MPI only during the measured steps
(cudaProfilerApi start/stop).

## Timings and validation

| Launch | State | Unprofiled seconds/step | Maximum absolute w | GPUs |
|--------|-------|-------------------------|--------------------|------|
| benchmark anelastic | COMPLETE | 4.767870e-01 | 3.279216e-05 | 0,1 |
| profiled anelastic | COMPLETE | see trace | 3.279216e-05 | 0,1 |
| benchmark oceananigans_nonhydrostatic | COMPLETE | 6.877646e-01 | 2.523582e-06 | 0,1 |
| profiled oceananigans_nonhydrostatic | COMPLETE | see trace | 2.523582e-06 | 0,1 |

## Trace summary (per rank, from analysis.json)

| Case | Rank | Trace s/step | Kernel launches | GPU active fraction | Halo kernel fraction | Solve kernel fraction |
|------|------|--------------|-----------------|---------------------|----------------------|-----------------------|
| anelastic_NCCL_2g | 0 | 0.4787 | 4730 | 0.982 | 0.052 | 0.436 |
| anelastic_NCCL_2g | 1 | 0.4793 | 4730 | 0.985 | 0.054 | 0.436 |
| oceananigans_nonhydrostatic_NCCL_2g | 0 | 0.6851 | 3270 | 0.999 | 0.086 | 0.542 |
| oceananigans_nonhydrostatic_NCCL_2g | 1 | 0.6885 | 3270 | 0.999 | 0.032 | 0.577 |

Trace seconds per step include profiling overhead; use the unprofiled timings
for performance. Fractions are of summed kernel time, attributed through CUDA
launch correlation IDs to enclosing `call/fill_halo_regions!` and `call/solve!`
ranges. Per-phase and per-kernel totals are in phases.csv and kernels.csv.

Finite-field checks establish a functional run, not field-by-field accuracy.
No plots are produced for Nsight runs.

## Exact Nsight result paths

- `/u/ajtse/Breeze.jl/.runtime/nccl_nsight_2gpu_2026-11-17/raw/anelastic/rank_0/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/nccl_nsight_2gpu_2026-11-17/raw/anelastic/rank_1/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/nccl_nsight_2gpu_2026-11-17/raw/oceananigans_nonhydrostatic/rank_0/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/nccl_nsight_2gpu_2026-11-17/raw/oceananigans_nonhydrostatic/rank_1/profile.nsys-rep`

SQLite exports sit beside each report. captures.json records report sizes,
SHA-256 hashes, kernel and range counts. Per-rank `nsys stats` summaries are in logs/.

## Automatic execution

run_campaign.py runs from the frozen checkout `/u/ajtse/Breeze.jl/.runtime/nccl_nsight_2gpu_2026-11-17/worktree`
(commit `50afa1d432120ef9832d7cc6646996155d94185c`) with the Julia environment
`/u/ajtse/Breeze.jl/.runtime/nccl_nsight_2gpu_2026-11-17/environment`. Every 30 s it looks for two GPUs with no
compute processes and at least 35000 MiB free, and sets
CUDA_VISIBLE_DEVICES to that pair for one launch.
Live state: `/u/ajtse/Breeze.jl/.runtime/nccl_nsight_2gpu_2026-11-17/status.json`. Controller log: `/u/ajtse/Breeze.jl/.runtime/nccl_nsight_2gpu_2026-11-17/logs/controller.log`.
On completion it commits this directory to benchmark-results and pushes.
