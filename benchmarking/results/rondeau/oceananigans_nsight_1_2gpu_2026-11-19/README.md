# Oceananigans hydrostatic and nonhydrostatic: NVTX Nsight on 1 and 2 GPUs

Status: **COMPLETE**.

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

**1-GPU vs 2-GPU per-kernel comparison** (hydrostatic and nonhydrostatic tables, with
difference in ms and %, and weak-scaling efficiency = 100 × T₁/T₂, slowest rank):
[kernel_scaling_1_vs_2gpu.md](kernel_scaling_1_vs_2gpu.md) (CSV: kernel_scaling_1_vs_2gpu.csv).
Regenerate with:

```sh
python3 benchmarking/distributed/kernel_scaling_table.py benchmarking/results/rondeau/oceananigans_nsight_1_2gpu_2026-11-19/analysis.json \
    benchmarking/results/rondeau/oceananigans_nsight_1_2gpu_2026-11-19/kernel_scaling_1_vs_2gpu \
    "Hydrostatic=hydrostatic_1g:hydrostatic_2g" "Nonhydrostatic=nonhydrostatic_1g:nonhydrostatic_2g"
```

## Launches

| Case | State | Maximum absolute w | GPUs |
|------|-------|--------------------|------|
| hydrostatic_1g | COMPLETE | 2.587143e-06 | 1 |
| hydrostatic_2g | COMPLETE | 2.587143e-06 | 1,3 |
| nonhydrostatic_1g | COMPLETE | 2.523582e-06 | 1 |
| nonhydrostatic_2g | COMPLETE | 2.523582e-06 | 1,3 |

## Trace summary per rank

| Case | Rank | Trace s/step | Kernel ms/step | Launches/step | GPU active fraction | Halo kernel fraction | Solve kernel fraction |
|------|------|--------------|----------------|---------------|---------------------|----------------------|-----------------------|
| hydrostatic_1g | 0 | 0.2343 | 224.4 | 279 | 0.999 | 0.005 | 0.000 |
| hydrostatic_2g | 0 | 0.2510 | 247.5 | 420 | 0.997 | 0.035 | 0.000 |
| hydrostatic_2g | 1 | 0.2483 | 244.4 | 420 | 0.998 | 0.021 | 0.000 |
| nonhydrostatic_1g | 0 | 0.4265 | 407.6 | 171 | 0.999 | 0.012 | 0.381 |
| nonhydrostatic_2g | 0 | 0.6885 | 667.6 | 327 | 0.999 | 0.034 | 0.575 |
| nonhydrostatic_2g | 1 | 0.6860 | 715.8 | 327 | 0.999 | 0.093 | 0.538 |

Trace seconds per step include profiling overhead. Fractions are of summed
kernel time, attributed through CUDA launch correlation IDs to enclosing
`call/fill_halo_regions!` and `call/solve!` ranges. Per-phase totals are in phases.csv.

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
