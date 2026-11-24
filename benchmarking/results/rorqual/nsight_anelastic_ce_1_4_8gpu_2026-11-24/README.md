# Rorqual NVTX Nsight: anelastic and CE on 1, 4, and 8 GPUs (NCCL)

Status: **SUBMITTED**; 0/6 cases validated.

Breeze AnelasticDynamics and CompressibleDynamics(ExplicitTimeStepping()), full dry CBL on the fixed
12 km × 12 km × 3 km domain, Float32, WENO5, deterministic (sine) perturbation, 512³ cells per H100.
Multi-GPU halo exchange (and the anelastic distributed pressure solve) uses NCCL; MPI only launches
ranks. Eight GPUs span two nodes, so their NCCL traffic crosses the inter-node network.
Each launch takes 10 warmup steps, then Nsight Systems (NVIDIA Nsight Systems version 2024.5.1.113-245134619542v0)
records CUDA, NVTX (Breeze host methods) and MPI for 10 steps
(cudaProfilerApi start/stop). CPU sampling and context switches are off.

| GPUs | Global grid | Partition | Nodes | Δt (s) |
|---|---|---|---|---|
| 1 | 512×512×512 | 1×1 | 1 | 0.01 |
| 4 | 1024×1024×512 | 4×1 | 1 | 0.005 |
| 8 | 4096×512×512 | 8×1 | 2 | 0.00125 |

## Cases

| Case | State | Job | s/step (profiled) | Maximum absolute w | Note |
|---|---|---|---|---|---|
| anelastic_SERIAL_1g_1x1 | not_run |  |  |  |  |
| compressible_explicit_SERIAL_1g_1x1 | not_run |  |  |  |  |
| anelastic_NCCL_4g_4x1 | not_run |  |  |  |  |
| compressible_explicit_NCCL_4g_4x1 | not_run |  |  |  |  |
| anelastic_NCCL_8g_8x1 | not_run |  |  |  |  |
| compressible_explicit_NCCL_8g_8x1 | not_run |  |  |  |  |

Profiled seconds per step include Nsight overhead; compare them only with other profiled runs.

Finite-field checks on every rank establish a functional run, not field-by-field agreement with a
serial solution. No plots are produced for Nsight runs.

## Exact Nsight result paths

On Rorqual; each report has a `profile.sqlite` export beside it. captures.json records sizes,
SHA-256 hashes, kernel and NVTX range counts.

- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/anelastic_SERIAL_1g_1x1/rank_0/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/compressible_explicit_SERIAL_1g_1x1/rank_0/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/anelastic_NCCL_4g_4x1/rank_0/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/anelastic_NCCL_4g_4x1/rank_1/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/anelastic_NCCL_4g_4x1/rank_2/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/anelastic_NCCL_4g_4x1/rank_3/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/compressible_explicit_NCCL_4g_4x1/rank_0/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/compressible_explicit_NCCL_4g_4x1/rank_1/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/compressible_explicit_NCCL_4g_4x1/rank_2/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/compressible_explicit_NCCL_4g_4x1/rank_3/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/anelastic_NCCL_8g_8x1/rank_0/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/anelastic_NCCL_8g_8x1/rank_1/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/anelastic_NCCL_8g_8x1/rank_2/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/anelastic_NCCL_8g_8x1/rank_3/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/anelastic_NCCL_8g_8x1/rank_4/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/anelastic_NCCL_8g_8x1/rank_5/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/anelastic_NCCL_8g_8x1/rank_6/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/anelastic_NCCL_8g_8x1/rank_7/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/compressible_explicit_NCCL_8g_8x1/rank_0/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/compressible_explicit_NCCL_8g_8x1/rank_1/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/compressible_explicit_NCCL_8g_8x1/rank_2/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/compressible_explicit_NCCL_8g_8x1/rank_3/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/compressible_explicit_NCCL_8g_8x1/rank_4/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/compressible_explicit_NCCL_8g_8x1/rank_5/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/compressible_explicit_NCCL_8g_8x1/rank_6/profile.nsys-rep`
- `/scratch/anditse/breeze_nsight/nsight_anelastic_ce_1_4_8gpu_2026-11-24/raw/compressible_explicit_NCCL_8g_8x1/rank_7/profile.nsys-rep`

## Reproduction

Frozen source `/home/anditse/Breeze.jl/.runtime/studies/nsight_anelastic_ce_1_4_8gpu_2026-11-24/source` (commit in provenance/source_commit.txt) and Julia environment
`/home/anditse/Breeze.jl/.runtime/studies/nsight_anelastic_ce_1_4_8gpu_2026-11-24/environment`; hashes in provenance/hashes.json are checked before and after every case.
Per-case `command.sh`, rank logs, Nsight export logs and `nsys stats` summaries are in cases/.

```bash
source benchmarking/distributed/rorqual/nsight_environment.sh
python3 benchmarking/distributed/rorqual/nsight.py prepare ABSOLUTE_STUDY_DIR
python3 benchmarking/distributed/rorqual/nsight.py submit ABSOLUTE_STUDY_DIR
python3 benchmarking/distributed/rorqual/nsight.py finalize ABSOLUTE_STUDY_DIR --publish  # after the jobs end
```
