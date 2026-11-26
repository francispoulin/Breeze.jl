# One-GPU NCCL-distributed baselines: hydrostatic, nonhydrostatic, anelastic, CE

Status: **COMPLETE**.

One rank on one A100 at 512×512×512 with NCCLDistributed and a 1×1 partition
(--distributed=true) instead of the serial GPU(), so the 1-GPU baselines take the
distributed code path of the 2-GPU runs. A 1×1 partition keeps Periodic local topologies:
Oceananigans adds no halo-communication boundary conditions, buffers or solver transposes
and creates no NCCL communicator, so inter-rank communication kernels stay 2-GPU only.
Arguments match the serial 1-GPU cases: Oceananigans HydrostaticFreeSurfaceModel and
NonhydrostaticModel (rectilinear_buoyancy, Float64, Δt = 10 s) and Breeze anelastic and
compressible explicit (full dry CBL, Float32, Δt = 0.01 s, sine perturbation). NVTX ranges
use the Oceananigans domain for the Oceananigans models and the Breeze domain otherwise.

Each launch takes ten warmup steps, then Nsight records CUDA, NVTX and MPI for
ten steps (cudaProfilerApi start/stop).

**The per-kernel time per step tables are in [kernels_per_step.md](kernels_per_step.md)**
(CSV: kernels_per_step.csv). Each entry sums all launches of a kernel during the
ten captured steps and divides by ten.

**Serial vs NCCL-distributed 1-GPU per-kernel comparison** (difference in ms and %,
efficiency = 100 × T_base/T_compared with the slower rank): [kernel_scaling_serial_vs_distributed_1gpu.md](kernel_scaling_serial_vs_distributed_1gpu.md)
(CSV: kernel_scaling_serial_vs_distributed_1gpu.csv).

## Launches

| Case | State | Maximum absolute w | GPUs | Other users' processes on these GPUs |
|------|-------|--------------------|------|---------------------------------------|
| hydrostatic_1gd | COMPLETE | 2.587143e-06 | 2 | none |
| nonhydrostatic_1gd | COMPLETE | 2.523582e-06 | 2 | none |
| anelastic_1gd | COMPLETE | 6.442626e-05 | 2 | none |
| explicit_1gd | COMPLETE | 2.102493e-03 | 2 | none |

## Trace summary per rank

| Case | Rank | Trace s/step | Kernel ms/step | Launches/step | GPU active fraction | Halo kernel fraction | Solve kernel fraction |
|------|------|--------------|----------------|---------------|---------------------|----------------------|-----------------------|
| hydrostatic_1gd | 0 | 0.2371 | 227.1 | 279 | 0.999 | 0.005 | 0.000 |
| nonhydrostatic_1gd | 0 | 0.5240 | 505.1 | 186 | 1.000 | 0.010 | 0.497 |
| anelastic_1gd | 0 | 0.3874 | 386.4 | 269 | 0.997 | 0.033 | 0.348 |
| explicit_1gd | 0 | 0.2722 | 271.9 | 249 | 0.999 | 0.054 | 0.000 |

Trace seconds per step include profiling overhead. Fractions are of summed
kernel time, attributed through CUDA launch correlation IDs to enclosing
`call/fill_halo_regions!` and `call/solve!` ranges. Per-phase totals are in phases.csv.

Finite-field checks establish a functional run, not field-by-field accuracy.
No plots are produced for Nsight runs.

## Exact Nsight result paths

- `/u/ajtse/Breeze.jl/.runtime/distributed_1gpu_nsight_2026-11-25/raw/hydrostatic_1gd/rank_0/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/distributed_1gpu_nsight_2026-11-25/raw/nonhydrostatic_1gd/rank_0/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/distributed_1gpu_nsight_2026-11-25/raw/anelastic_1gd/rank_0/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/distributed_1gpu_nsight_2026-11-25/raw/explicit_1gd/rank_0/profile.nsys-rep`

SQLite exports sit beside each report. captures.json records report sizes,
SHA-256 hashes, kernel and range counts. Per-rank `nsys stats` summaries are in logs/.

## Automatic execution

run_campaign.py runs from the frozen checkout `/u/ajtse/Breeze.jl/.runtime/distributed_1gpu_nsight_2026-11-25/worktree`
(commit `f069ccff22fa27581dee625965ec1fbcb08750cf`) with the Julia environment `/u/ajtse/Breeze.jl/.runtime/distributed_1gpu_nsight_2026-11-25/environment`.
Every 30 s it looks for GPUs with no compute processes and at least
35000 MiB free, and sets CUDA_VISIBLE_DEVICES for one launch.
Live state: `/u/ajtse/Breeze.jl/.runtime/distributed_1gpu_nsight_2026-11-25/status.json`. Controller log: `/u/ajtse/Breeze.jl/.runtime/distributed_1gpu_nsight_2026-11-25/logs/controller.log`.
On completion it commits this directory to benchmark-results and pushes.
