# Trillium NVTX Nsight: anelastic and CE on 4 and 8 GPUs (NCCL)

Status: **COMPLETE**; 4/4 cases validated.

Breeze AnelasticDynamics and CompressibleDynamics(ExplicitTimeStepping()), full dry CBL on the fixed
12 km × 12 km × 3 km domain, Float32, WENO5, deterministic (sine) perturbation, 512³ cells per H100.
Multi-GPU halo exchange (and the anelastic distributed pressure solve) uses NCCL; MPI only launches
ranks. Eight GPUs span two nodes, so their NCCL traffic crosses the inter-node network.
Each launch takes 10 warmup steps, then Nsight Systems (NVIDIA Nsight Systems version 2024.5.1.113-245134619542v0)
records CUDA, NVTX (Breeze host methods) and MPI for 10 steps
(cudaProfilerApi start/stop). CPU sampling and context switches are off.

| GPUs | Global grid | Partition | Nodes | Δt (s) |
|---|---|---|---|---|
| 4 | 2048×512×512 | 4×1 | 1 | 0.0025 |
| 8 | 4096×512×512 | 8×1 | 2 | 0.00125 |

## Cases

| Case | State | Job | s/step (profiled) | Maximum absolute w | Note |
|---|---|---|---|---|---|
| anelastic_NCCL_4g_4x1 | validated | 1418162 | 7.398668e-01 | 1.884158e-05 |  |
| compressible_explicit_NCCL_4g_4x1 | validated | 1421438 | 7.065025e-01 | 1.327951e-03 |  |
| anelastic_NCCL_8g_8x1 | validated | 1418163 | 9.125264e-01 | 1.525665e-05 |  |
| compressible_explicit_NCCL_8g_8x1 | validated | 1418163 | 6.803111e-01 | 1.065023e-03 |  |

Profiled seconds per step include Nsight overhead; compare them only with other profiled runs.

## Retried captures

A failed attempt is kept in rejected_attempts/ (reports under the raw directory's rejected/) and
excluded from the analysis; the retry runs the frozen `run_case` in a new allocation.

- compressible_explicit_NCCL_4g_4x1: job 1421438; rejected attempt(s) compressible_explicit_NCCL_4g_4x1_job1418162. Reason: nsys exited 13 s after launch with "Failed to find session: profile-<pid>" on ranks 1-3 (rank exit 1) immediately after the previous srun step on the same node; no model code ran

## Trace summary per rank

| Case | Rank | Trace s/step | Kernel ms/step | Launches/step | GPU active fraction | Halo kernel fraction | Solve kernel fraction |
|---|---|---|---|---|---|---|---|
| anelastic_NCCL_4g_4x1 | 0 | 0.2526 | 250.3 | 473 | 0.992 | 0.066 | 0.452 |
| anelastic_NCCL_4g_4x1 | 1 | 0.2483 | 246.0 | 473 | 0.992 | 0.052 | 0.459 |
| anelastic_NCCL_4g_4x1 | 2 | 0.2481 | 245.8 | 473 | 0.991 | 0.049 | 0.460 |
| anelastic_NCCL_4g_4x1 | 3 | 0.2524 | 247.7 | 473 | 0.982 | 0.056 | 0.457 |
| compressible_explicit_NCCL_4g_4x1 | 0 | 0.1437 | 143.6 | 465 | 0.992 | 0.097 | 0.000 |
| compressible_explicit_NCCL_4g_4x1 | 1 | 0.1412 | 141.2 | 465 | 0.993 | 0.081 | 0.000 |
| compressible_explicit_NCCL_4g_4x1 | 2 | 0.1438 | 143.8 | 465 | 0.992 | 0.105 | 0.000 |
| compressible_explicit_NCCL_4g_4x1 | 3 | 0.1407 | 140.7 | 465 | 0.992 | 0.083 | 0.000 |
| anelastic_NCCL_8g_8x1 | 0 | 0.4323 | 430.0 | 473 | 0.994 | 0.153 | 0.567 |
| anelastic_NCCL_8g_8x1 | 1 | 0.4225 | 420.4 | 473 | 0.995 | 0.136 | 0.581 |
| anelastic_NCCL_8g_8x1 | 2 | 0.4227 | 420.4 | 473 | 0.994 | 0.133 | 0.580 |
| anelastic_NCCL_8g_8x1 | 3 | 0.4235 | 421.4 | 473 | 0.994 | 0.109 | 0.604 |
| anelastic_NCCL_8g_8x1 | 4 | 0.4004 | 398.4 | 473 | 0.994 | 0.085 | 0.611 |
| anelastic_NCCL_8g_8x1 | 5 | 0.4037 | 401.6 | 473 | 0.994 | 0.093 | 0.607 |
| anelastic_NCCL_8g_8x1 | 6 | 0.4010 | 399.0 | 473 | 0.994 | 0.085 | 0.612 |
| anelastic_NCCL_8g_8x1 | 7 | 0.4033 | 387.0 | 473 | 0.959 | 0.060 | 0.630 |
| compressible_explicit_NCCL_8g_8x1 | 0 | 0.1569 | 156.8 | 465 | 0.993 | 0.172 | 0.000 |
| compressible_explicit_NCCL_8g_8x1 | 1 | 0.1590 | 158.9 | 465 | 0.993 | 0.192 | 0.000 |
| compressible_explicit_NCCL_8g_8x1 | 2 | 0.1577 | 157.5 | 465 | 0.993 | 0.176 | 0.000 |
| compressible_explicit_NCCL_8g_8x1 | 3 | 0.1575 | 157.4 | 465 | 0.993 | 0.175 | 0.000 |
| compressible_explicit_NCCL_8g_8x1 | 4 | 0.2015 | 201.5 | 465 | 0.995 | 0.355 | 0.000 |
| compressible_explicit_NCCL_8g_8x1 | 5 | 0.2050 | 204.9 | 465 | 0.995 | 0.366 | 0.000 |
| compressible_explicit_NCCL_8g_8x1 | 6 | 0.2018 | 201.7 | 465 | 0.994 | 0.357 | 0.000 |
| compressible_explicit_NCCL_8g_8x1 | 7 | 0.2055 | 205.4 | 465 | 0.995 | 0.368 | 0.000 |

Fractions are of summed kernel time, attributed through CUDA launch correlation IDs to the
enclosing `call/fill_halo_regions!` and `call/solve!` NVTX ranges. Per-phase totals: phases.csv.
Per-kernel time per step: [kernels_per_step.md](kernels_per_step.md). Each larger case against
the 4-GPU case of the same dynamics (slowest rank, weak-scaling efficiency
= 100 × T4/Tₙ; rows sorted by Difference ms, greatest first):
[kernel_scaling.md](kernel_scaling.md).

Finite-field checks on every rank establish a functional run, not field-by-field agreement with a
serial solution. No plots are produced for Nsight runs.

## Exact Nsight result paths

On Trillium; each report has a `profile.sqlite` export beside it. captures.json records sizes,
SHA-256 hashes, kernel and NVTX range counts.

- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_4_8gpu_2026-11-26/raw/anelastic_NCCL_4g_4x1/rank_0/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_4_8gpu_2026-11-26/raw/anelastic_NCCL_4g_4x1/rank_1/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_4_8gpu_2026-11-26/raw/anelastic_NCCL_4g_4x1/rank_2/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_4_8gpu_2026-11-26/raw/anelastic_NCCL_4g_4x1/rank_3/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_4_8gpu_2026-11-26/raw/compressible_explicit_NCCL_4g_4x1/rank_0/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_4_8gpu_2026-11-26/raw/compressible_explicit_NCCL_4g_4x1/rank_1/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_4_8gpu_2026-11-26/raw/compressible_explicit_NCCL_4g_4x1/rank_2/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_4_8gpu_2026-11-26/raw/compressible_explicit_NCCL_4g_4x1/rank_3/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_4_8gpu_2026-11-26/raw/anelastic_NCCL_8g_8x1/rank_0/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_4_8gpu_2026-11-26/raw/anelastic_NCCL_8g_8x1/rank_1/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_4_8gpu_2026-11-26/raw/anelastic_NCCL_8g_8x1/rank_2/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_4_8gpu_2026-11-26/raw/anelastic_NCCL_8g_8x1/rank_3/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_4_8gpu_2026-11-26/raw/anelastic_NCCL_8g_8x1/rank_4/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_4_8gpu_2026-11-26/raw/anelastic_NCCL_8g_8x1/rank_5/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_4_8gpu_2026-11-26/raw/anelastic_NCCL_8g_8x1/rank_6/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_4_8gpu_2026-11-26/raw/anelastic_NCCL_8g_8x1/rank_7/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_4_8gpu_2026-11-26/raw/compressible_explicit_NCCL_8g_8x1/rank_0/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_4_8gpu_2026-11-26/raw/compressible_explicit_NCCL_8g_8x1/rank_1/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_4_8gpu_2026-11-26/raw/compressible_explicit_NCCL_8g_8x1/rank_2/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_4_8gpu_2026-11-26/raw/compressible_explicit_NCCL_8g_8x1/rank_3/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_4_8gpu_2026-11-26/raw/compressible_explicit_NCCL_8g_8x1/rank_4/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_4_8gpu_2026-11-26/raw/compressible_explicit_NCCL_8g_8x1/rank_5/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_4_8gpu_2026-11-26/raw/compressible_explicit_NCCL_8g_8x1/rank_6/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_4_8gpu_2026-11-26/raw/compressible_explicit_NCCL_8g_8x1/rank_7/profile.nsys-rep`

## Reproduction

Frozen source `/scratch/anditse/Breeze.jl/.runtime/studies/nsight_anelastic_ce_4_8gpu_2026-11-26/source` (commit in provenance/source_commit.txt) and Julia environment
`/scratch/anditse/Breeze.jl/.runtime/studies/nsight_anelastic_ce_4_8gpu_2026-11-26/environment`; hashes in provenance/hashes.json are checked before and after every case.
Per-case `command.sh`, rank logs, Nsight export logs and `nsys stats` summaries are in cases/.

```bash
source benchmarking/distributed/trillium/nsight_environment.sh
python3 benchmarking/distributed/rorqual/nsight.py prepare ABSOLUTE_STUDY_DIR --series=trillium_4_8
python3 benchmarking/distributed/rorqual/nsight.py submit ABSOLUTE_STUDY_DIR
python3 benchmarking/distributed/rorqual/nsight.py finalize ABSOLUTE_STUDY_DIR --publish  # after the jobs end
```
