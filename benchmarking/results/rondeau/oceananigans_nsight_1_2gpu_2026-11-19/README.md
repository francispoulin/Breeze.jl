# Oceananigans hydrostatic and nonhydrostatic: NVTX Nsight on 1 and 2 GPUs

Status: **QUEUED**.

512³ cells per A100: 512×512×512 on one GPU (no inter-rank communication) and
1024×512×512 with a 2×1 partition on two GPUs using NCCL for halos and transposes
(MPI only launches ranks). Oceananigans suite `rectilinear_buoyancy` workload, Float64,
Δt = 10 s, WENO5, one buoyancy tracer. Hydrostatic: `HydrostaticFreeSurfaceModel`,
SplitRungeKutta3, 30-substep `SplitExplicitFreeSurface` with extended halos.
Nonhydrostatic: `NonhydrostaticModel`, RungeKutta3, FFT pressure solver.
Oceananigans 0.117.2 from the Breeze benchmark environment.

Each launch takes ten warmup steps, then Nsight records CUDA, NVTX and MPI for
ten steps (cudaProfilerApi start/stop) with Oceananigans host NVTX ranges.

**The per-kernel time per step tables are in [kernels_per_step.md](kernels_per_step.md)**
(CSV: kernels_per_step.csv). Each entry sums all launches of a kernel during the
ten captured steps and divides by ten.

## Launches

| Case | State | Maximum absolute w | GPUs |
|------|-------|--------------------|------|

Finite-field checks establish a functional run, not field-by-field accuracy.
No plots are produced for Nsight runs.

## Exact Nsight result paths

- `/u/ajtse/Breeze.jl/.runtime/oceananigans_nsight_1_2gpu_2026-11-19/raw/hydrostatic_1g/rank_0/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/oceananigans_nsight_1_2gpu_2026-11-19/raw/hydrostatic_2g/rank_0/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/oceananigans_nsight_1_2gpu_2026-11-19/raw/hydrostatic_2g/rank_1/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/oceananigans_nsight_1_2gpu_2026-11-19/raw/nonhydrostatic_1g/rank_0/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/oceananigans_nsight_1_2gpu_2026-11-19/raw/nonhydrostatic_2g/rank_0/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/oceananigans_nsight_1_2gpu_2026-11-19/raw/nonhydrostatic_2g/rank_1/profile.nsys-rep`

SQLite exports sit beside each report. captures.json records report sizes,
SHA-256 hashes, kernel and range counts. Per-rank `nsys stats` summaries are in logs/.

## Automatic execution

run_campaign.py runs from the frozen checkout `/u/ajtse/Breeze.jl/.runtime/oceananigans_nsight_1_2gpu_2026-11-19/worktree`
(commit `9a9ffe87a72ba71dde075e5c72e679ea33eb5898`) with the Julia environment `/u/ajtse/Breeze.jl/.runtime/oceananigans_nsight_1_2gpu_2026-11-19/environment`.
Every 30 s it looks for GPUs with no compute processes and at least
35000 MiB free, and sets CUDA_VISIBLE_DEVICES for one launch.
Live state: `/u/ajtse/Breeze.jl/.runtime/oceananigans_nsight_1_2gpu_2026-11-19/status.json`. Controller log: `/u/ajtse/Breeze.jl/.runtime/oceananigans_nsight_1_2gpu_2026-11-19/logs/controller.log`.
On completion it commits this directory to benchmark-results and pushes.
