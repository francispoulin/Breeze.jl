# Breeze compressible explicit and split-explicit: NVTX Nsight on 1 and 2 GPUs

Status: **FAILED**.

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

| Case | State | Maximum absolute w | GPUs | Other users' processes on these GPUs |
|------|-------|--------------------|------|---------------------------------------|
| explicit_1g | COMPLETE | 2.102493e-03 | 0 | not monitored |
| explicit_2g | FAILED | 1.610592e-03 | 0,3 | none |
| splitexplicit_1g | COMPLETE | 2.344239e-03 | 0 | not monitored |
| splitexplicit_2g | COMPLETE | 1.746902e-03 | 0,1 | not monitored |

## Trace summary per rank

| Case | Rank | Trace s/step | Kernel ms/step | Launches/step | GPU active fraction | Halo kernel fraction | Solve kernel fraction |
|------|------|--------------|----------------|---------------|---------------------|----------------------|-----------------------|
| explicit_1g | 0 | 0.2723 | 272.0 | 231 | 0.999 | 0.056 | 0.000 |
| splitexplicit_1g | 0 | 1.0552 | 1048.9 | 1030 | 0.999 | 0.060 | 0.143 |
| splitexplicit_2g | 0 | 1.1191 | 1113.3 | 2152 | 0.997 | 0.111 | 0.134 |
| splitexplicit_2g | 1 | 1.1165 | 1110.5 | 2152 | 0.997 | 0.120 | 0.130 |
| explicit_2g | 0 | 0.2847 | 285.6 | 465 | 0.996 | 0.100 | 0.000 |
| explicit_2g | 1 | 0.2837 | 284.6 | 465 | 0.996 | 0.097 | 0.000 |

Trace seconds per step include profiling overhead. Fractions are of summed
kernel time, attributed through CUDA launch correlation IDs to enclosing
`call/fill_halo_regions!` and `call/solve!` ranges. Per-phase totals are in phases.csv.

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

## Superseded captures

These launches were captured again; the tables above use only the newer capture.

- `explicit_2g` from 2026-11-20T21:29:35 UTC (superseded 2026-11-22T05:31:57 UTC): reports in `/u/ajtse/Breeze.jl/.runtime/compressible_nsight_1_2gpu_2026-11-20/raw/superseded/explicit_2g_20261122T053157Z`, logs in `logs/superseded/explicit_2g_20261122T053157Z`. Reason: Rank 1 compute kernels ran about 2.2x slower than rank 0 and the 1-GPU case, consistent with another user's job sharing its GPU during the capture.
- `explicit_2g` from 2026-11-22T07:56:02 UTC (superseded 2026-11-26T18:57:02 UTC): reports in `/u/ajtse/Breeze.jl/.runtime/compressible_nsight_1_2gpu_2026-11-20/raw/superseded/explicit_2g_20261126T185702Z`, logs in `logs/superseded/explicit_2g_20261126T185702Z`. Reason: Launched without environment.sh (default CUDA memory pool, default GC threads, no GC NVTX ranges); recaptured with it.
- `explicit_2g` from 2026-11-26T18:57:02 UTC (superseded 2026-11-26T19:09:37 UTC): reports in `/u/ajtse/Breeze.jl/.runtime/compressible_nsight_1_2gpu_2026-11-20/raw/superseded/explicit_2g_20261126T190937Z`, logs in `logs/superseded/explicit_2g_20261126T190937Z`. Reason: Previous recapture's rank 0 Nsight report held no CUDA kernel activity (intermittent CUPTI failure).

## Failure

no such table: CUPTI_ACTIVITY_KIND_KERNEL
