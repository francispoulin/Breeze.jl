# Breeze compressible explicit and split-explicit: NVTX Nsight on 1 and 2 GPUs

Status: **QUEUED**.

512³ cells per A100: 512×512×512 on one GPU (no inter-rank communication) and
1024×512×512 with a 2×1 partition on two GPUs using NCCL for halo exchange
(MPI only launches ranks). Full dry CBL on the fixed 12 km × 12 km × 3 km domain, Float32,
WENO5, deterministic (sine) perturbation. Δt follows AGENTS.md: 0.01 s at 512³ and
0.005 s at 1024×512×512, the same for explicit (CE) and split-explicit (CSE). CSE uses
12 nominal acoustic substeps. Breeze host NVTX ranges.

Each launch takes ten warmup steps, then Nsight records CUDA, NVTX and MPI for
ten steps (cudaProfilerApi start/stop).

**The per-kernel time per step tables are in [kernels_per_step.md](kernels_per_step.md)**
(CSV: kernels_per_step.csv). Each entry sums all launches of a kernel during the
ten captured steps and divides by ten.

**1-GPU vs 2-GPU per-kernel comparison** (difference in ms and %, weak-scaling
efficiency = 100 × T₁/T₂ with the slower rank): [kernel_scaling_1_vs_2gpu.md](kernel_scaling_1_vs_2gpu.md)
(CSV: kernel_scaling_1_vs_2gpu.csv).

## Launches

| Case | State | Maximum absolute w | GPUs |
|------|-------|--------------------|------|

Finite-field checks establish a functional run, not field-by-field accuracy.
No plots are produced for Nsight runs.

## Exact Nsight result paths

- `/u/ajtse/Breeze.jl/.runtime/compressible_nsight_1_2gpu_2026-11-20/raw/explicit_1g/rank_0/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/compressible_nsight_1_2gpu_2026-11-20/raw/explicit_2g/rank_0/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/compressible_nsight_1_2gpu_2026-11-20/raw/explicit_2g/rank_1/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/compressible_nsight_1_2gpu_2026-11-20/raw/splitexplicit_1g/rank_0/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/compressible_nsight_1_2gpu_2026-11-20/raw/splitexplicit_2g/rank_0/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/compressible_nsight_1_2gpu_2026-11-20/raw/splitexplicit_2g/rank_1/profile.nsys-rep`

SQLite exports sit beside each report. captures.json records report sizes,
SHA-256 hashes, kernel and range counts. Per-rank `nsys stats` summaries are in logs/.

## Automatic execution

run_campaign.py runs from the frozen checkout `/u/ajtse/Breeze.jl/.runtime/compressible_nsight_1_2gpu_2026-11-20/worktree`
(commit `4500870251b2c7f939cfedac68dba469cf6cab3b`) with the Julia environment `/u/ajtse/Breeze.jl/.runtime/compressible_nsight_1_2gpu_2026-11-20/environment`.
Every 30 s it looks for GPUs with no compute processes and at least
35000 MiB free, and sets CUDA_VISIBLE_DEVICES for one launch.
Live state: `/u/ajtse/Breeze.jl/.runtime/compressible_nsight_1_2gpu_2026-11-20/status.json`. Controller log: `/u/ajtse/Breeze.jl/.runtime/compressible_nsight_1_2gpu_2026-11-20/logs/controller.log`.
On completion it commits this directory to benchmark-results and pushes.
