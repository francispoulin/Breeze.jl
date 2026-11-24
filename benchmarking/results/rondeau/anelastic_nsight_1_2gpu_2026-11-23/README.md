# Breeze anelastic: NVTX Nsight on 1 and 2 GPUs

Status: **COMPLETE**.

512³ cells per A100: 512×512×512 on one GPU (no inter-rank communication) and
1024×512×512 with a 2×1 partition on two GPUs using NCCL for halo exchange and the
distributed pressure solve (MPI only launches ranks). Breeze AnelasticDynamics, full dry CBL
on the fixed 12 km × 12 km × 3 km domain, Float32, WENO5, deterministic (sine) perturbation.
Δt = 0.01 s at 512³ and 0.005 s at 1024×512×512, as in the compressible 1-vs-2-GPU campaign
and the 2026-11-17 anelastic NCCL capture. Breeze host NVTX ranges.

Each launch takes ten warmup steps, then Nsight records CUDA, NVTX and MPI for
ten steps (cudaProfilerApi start/stop).

**The per-kernel time per step tables are in [kernels_per_step.md](kernels_per_step.md)**
(CSV: kernels_per_step.csv). Each entry sums all launches of a kernel during the
ten captured steps and divides by ten.

**1-GPU vs 2-GPU per-kernel comparison** (difference in ms and %, weak-scaling
efficiency = 100 × T₁/T₂ with the slower rank): [kernel_scaling_1_vs_2gpu.md](kernel_scaling_1_vs_2gpu.md)
(CSV: kernel_scaling_1_vs_2gpu.csv).

## Launches

| Case | State | Maximum absolute w | GPUs | Other users' processes on these GPUs |
|------|-------|--------------------|------|---------------------------------------|
| anelastic_1g | COMPLETE | 6.429169e-05 | 1 | none |
| anelastic_2g | COMPLETE | 3.279216e-05 | 0,1 | none |

## Trace summary per rank

| Case | Rank | Trace s/step | Kernel ms/step | Launches/step | GPU active fraction | Halo kernel fraction | Solve kernel fraction |
|------|------|--------------|----------------|---------------|---------------------|----------------------|-----------------------|
| anelastic_1g | 0 | 0.2959 | 295.6 | 239 | 0.999 | 0.044 | 0.142 |
| anelastic_2g | 0 | 0.4767 | 471.1 | 473 | 0.994 | 0.058 | 0.433 |
| anelastic_2g | 1 | 0.4744 | 469.0 | 473 | 0.994 | 0.054 | 0.437 |

Trace seconds per step include profiling overhead. Fractions are of summed
kernel time, attributed through CUDA launch correlation IDs to enclosing
`call/fill_halo_regions!` and `call/solve!` ranges. Per-phase totals are in phases.csv.

Finite-field checks establish a functional run, not field-by-field accuracy.
No plots are produced for Nsight runs.

## Exact Nsight result paths

- `/u/ajtse/Breeze.jl/.runtime/anelastic_nsight_1_2gpu_2026-11-23/raw/anelastic_1g/rank_0/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/anelastic_nsight_1_2gpu_2026-11-23/raw/anelastic_2g/rank_0/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/anelastic_nsight_1_2gpu_2026-11-23/raw/anelastic_2g/rank_1/profile.nsys-rep`

SQLite exports sit beside each report. captures.json records report sizes,
SHA-256 hashes, kernel and range counts. Per-rank `nsys stats` summaries are in logs/.

## Automatic execution

run_campaign.py runs from the frozen checkout `/u/ajtse/Breeze.jl/.runtime/anelastic_nsight_1_2gpu_2026-11-23/worktree`
(commit `c25dcb5035e9d8f4940e665f3bf2b9ad714895be`) with the Julia environment `/u/ajtse/Breeze.jl/.runtime/anelastic_nsight_1_2gpu_2026-11-23/environment`.
Every 30 s it looks for GPUs with no compute processes and at least
35000 MiB free, and sets CUDA_VISIBLE_DEVICES for one launch.
Live state: `/u/ajtse/Breeze.jl/.runtime/anelastic_nsight_1_2gpu_2026-11-23/status.json`. Controller log: `/u/ajtse/Breeze.jl/.runtime/anelastic_nsight_1_2gpu_2026-11-23/logs/controller.log`.
On completion it commits this directory to benchmark-results and pushes.
