# Oceananigans hydrostatic and nonhydrostatic in Float32: NVTX Nsight on 1 and 2 GPUs

Status: **FAILED**.

Float32 repeat of the Oceananigans captures (2026-11-19 and 2026-11-25 ran Float64), so the
Oceananigans models use the same precision as the Breeze anelastic, CE and CSE captures.
HydrostaticFreeSurfaceModel (SplitRungeKutta3, SplitExplicitFreeSurface, 30 substeps) and
NonhydrostaticModel (RungeKutta3) on the rectilinear_buoyancy workload, WENO5, Δt = 10 s.
Per model: 512×512×512 on one GPU serial (`_1g`), the same on one rank with NCCLDistributed
and a 1×1 partition (`_1gd`, --distributed=true), and 1024×512×512 on two GPUs with NCCL
(`_2g`). Oceananigans host NVTX ranges.

Each launch takes ten warmup steps, then Nsight records CUDA, NVTX and MPI for
ten steps (cudaProfilerApi start/stop).

**The per-kernel time per step tables are in [kernels_per_step.md](kernels_per_step.md)**
(CSV: kernels_per_step.csv). Each entry sums all launches of a kernel during the
ten captured steps and divides by ten.

**1-GPU (NCCL-distributed) vs 2-GPU per-kernel comparison** (difference in ms and %,
efficiency = 100 × T_base/T_compared with the slower rank): [kernel_scaling_1_vs_2gpu.md](kernel_scaling_1_vs_2gpu.md)
(CSV: kernel_scaling_1_vs_2gpu.csv).

## Launches

| Case | State | Maximum absolute w | GPUs | Other users' processes on these GPUs |
|------|-------|--------------------|------|---------------------------------------|
| hydrostatic_1g | COMPLETE | 1.633013e-05 | 0 | none |
| hydrostatic_1gd | COMPLETE | 1.633013e-05 | 0 | none |
| hydrostatic_2g | FAILED | 1.633013e-05 | 0,3 | none |
| nonhydrostatic_1g | COMPLETE | 3.872733e-06 | 0 | none |
| nonhydrostatic_1gd | COMPLETE | 3.873204e-06 | 0 | none |
| nonhydrostatic_2g | COMPLETE | 3.873211e-06 | 0,3 | none |

## Trace summary per rank

| Case | Rank | Trace s/step | Kernel ms/step | Launches/step | GPU active fraction | Halo kernel fraction | Solve kernel fraction |
|------|------|--------------|----------------|---------------|---------------------|----------------------|-----------------------|
| hydrostatic_1g | 0 | 0.1192 | 114.0 | 264 | 0.997 | 0.007 | 0.000 |
| hydrostatic_1gd | 0 | 0.1192 | 114.0 | 264 | 0.997 | 0.007 | 0.000 |
| hydrostatic_2g | 0 | 0.1812 | 179.8 | 405 | 0.996 | 0.316 | 0.000 |
| hydrostatic_2g | 1 | 0.1296 | 126.8 | 405 | 0.995 | 0.017 | 0.000 |
| nonhydrostatic_1g | 0 | 0.2149 | 205.4 | 171 | 0.999 | 0.016 | 0.413 |
| nonhydrostatic_1gd | 0 | 0.2946 | 285.0 | 186 | 0.999 | 0.011 | 0.577 |
| nonhydrostatic_2g | 0 | 0.3873 | 376.3 | 327 | 0.999 | 0.035 | 0.631 |
| nonhydrostatic_2g | 1 | 0.3847 | 375.0 | 327 | 0.998 | 0.029 | 0.635 |

Trace seconds per step include profiling overhead. Fractions are of summed
kernel time, attributed through CUDA launch correlation IDs to enclosing
`call/fill_halo_regions!` and `call/solve!` ranges. Per-phase totals are in phases.csv.

Finite-field checks establish a functional run, not field-by-field accuracy.
No plots are produced for Nsight runs.

## Exact Nsight result paths

- `/u/ajtse/Breeze.jl/.runtime/oceananigans_float32_nsight_2026-11-26/raw/hydrostatic_1g/rank_0/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/oceananigans_float32_nsight_2026-11-26/raw/hydrostatic_1gd/rank_0/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/oceananigans_float32_nsight_2026-11-26/raw/hydrostatic_2g/rank_0/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/oceananigans_float32_nsight_2026-11-26/raw/hydrostatic_2g/rank_1/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/oceananigans_float32_nsight_2026-11-26/raw/nonhydrostatic_1g/rank_0/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/oceananigans_float32_nsight_2026-11-26/raw/nonhydrostatic_1gd/rank_0/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/oceananigans_float32_nsight_2026-11-26/raw/nonhydrostatic_2g/rank_0/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/oceananigans_float32_nsight_2026-11-26/raw/nonhydrostatic_2g/rank_1/profile.nsys-rep`

SQLite exports sit beside each report. captures.json records report sizes,
SHA-256 hashes, kernel and range counts. Per-rank `nsys stats` summaries are in logs/.

## Automatic execution

run_campaign.py runs from the frozen checkout `/u/ajtse/Breeze.jl/.runtime/oceananigans_float32_nsight_2026-11-26/worktree`
(commit `b6b4c1f16018dac7de186300f8a70642ac8ba7ae`) with the Julia environment `/u/ajtse/Breeze.jl/.runtime/oceananigans_float32_nsight_2026-11-26/environment`.
Every 30 s it looks for GPUs with no compute processes and at least
35000 MiB free, and sets CUDA_VISIBLE_DEVICES for one launch.
Live state: `/u/ajtse/Breeze.jl/.runtime/oceananigans_float32_nsight_2026-11-26/status.json`. Controller log: `/u/ajtse/Breeze.jl/.runtime/oceananigans_float32_nsight_2026-11-26/logs/controller.log`.
On completion it commits this directory to benchmark-results and pushes.

## Superseded captures

These launches were captured again; the tables above use only the newer capture.

- `hydrostatic_2g` from 2026-11-26T17:26:23 UTC (superseded 2026-11-26T17:34:53 UTC): reports in `/u/ajtse/Breeze.jl/.runtime/oceananigans_float32_nsight_2026-11-26/raw/superseded/hydrostatic_2g_20261126T173453Z`, logs in `logs/superseded/hydrostatic_2g_20261126T173453Z`. Reason: Rank 0's first NCCL kernel waited 516 ms for rank 1 to start its capture, adding 51.6 ms/step to rank 0 (later NCCL launches: 8.5 ms/step); other 2-GPU captures had 0.3-35 ms first waits.

## Failure

no such table: CUPTI_ACTIVITY_KIND_KERNEL
