# Weak-scaling efficiency at 512³ cells per GPU

Efficiency is **100 × T₁ / Tₙ**, with T in seconds per step and each model normalized
to its own one-GPU time. Local grids stay at 512³ cells; the global grid grows in x.
Ideal weak scaling is 100%. Each point is one measured window; no error bars are implied.
`plot_weak_efficiency.py` regenerates every figure and `weak_efficiency.csv` from the
source measurements below; it skips a figure whose series is incomplete.

Each figure covers one machine. Rondeau uses A100 GPUs and Fir uses H100 GPUs, whose
compute, memory bandwidth and interconnect differ, so neither timings nor efficiencies
are comparable between Rondeau and Fir figures. Compare models only within one figure.

## Breeze: anelastic, compressible explicit, compressible split-explicit

![Breeze weak scaling on Fir](breeze_weak_efficiency_fir.png)

Fir, 1–16 H100s, N×1 partitions, one validated series from
[fir/weak_scaling_mpi_2026-11-12_nx1](../fir/weak_scaling_mpi_2026-11-12_nx1/README.md).
Float32 WENO5 dry CBL, Δt halved per GPU doubling, 500 timed steps, slowest-rank time.

| GPUs | Anelastic | Compressible explicit | Compressible split-explicit |
|---:|---:|---:|---:|
| 2 | 56.9% | 94.9% | 94.2% |
| 4 | 60.3% | 95.5% | 95.5% |
| 8 | 20.5% | 79.0% | 80.0% |
| 16 | 14.8% | 76.9% | 78.5% |

![Breeze weak scaling on Rondeau](breeze_weak_efficiency_rondeau.png)

Rondeau, 1–4 A100s, Δt = 0.005 s at every count. Anelastic is from
[anelastic_weak_scaling_512_2026-11-13](../rondeau/anelastic_weak_scaling_512_2026-11-13/README.md)
(4×1 at four GPUs). Compressible MPI values are the 9–10 November measurements from
[weak_scaling_mpi_nccl_2026-11-09_dt0p005](../rondeau/weak_scaling_mpi_nccl_2026-11-09_dt0p005/README.md),
using 2×2 at four GPUs so each local grid is a 512³ cube. The two studies use different
source revisions, so compare scaling shapes, not absolute costs.

| GPUs | Anelastic | Compressible explicit | Compressible split-explicit |
|---:|---:|---:|---:|
| 2 | 63.6% | 97.8% | 96.0% |
| 4 | 57.8% | 91.5% | 82.4% |

## Oceananigans: hydrostatic and nonhydrostatic

![Oceananigans weak scaling on Fir](oceananigans_weak_efficiency_fir.png)

Fir, 1–16 H100s, from the Oceananigans `gpu_weak_scaling_512_x` baseline series
(`/u/ajtse/Oceananigans.jl/benchmarking/results/<model>/fir/baseline/gpu_weak_scaling_512_x/`).
Float64 RectilinearGrid, periodic x/y, WENO5, one buoyancy tracer, Δt = 10 s, N×1 partitions.
The hydrostatic model uses a split-explicit free surface; the nonhydrostatic model uses the
distributed FFT pressure solver. Times are the median of five ten-step windows (slowest rank).

| GPUs | Hydrostatic | Nonhydrostatic |
|---:|---:|---:|
| 2 | 91.5% | 40.0% |
| 4 | 89.8% | 29.0% |
| 8 | 88.5% | 19.2% |
| 16 | 88.6% | 14.4% |

### Fir nonhydrostatic rerun without profiling (NCCL)

The suite's two-GPU nonhydrostatic point (40%) used MPI. The 1- and 2-GPU cases are rerun as a pure
performance benchmark, with no Nsight capture and no NVTX instrumentation, using NCCL for halos and the
FFT transposes: Breeze's driver (`--dynamics=oceananigans_nonhydrostatic --comm=NCCL --profile=false
--nvtx=false`), the same Float64 workload, 10 warmup and 100 timed steps, three launches per case, median
of the slowest rank. Study: [fir/oceananigans_nonhydrostatic_nccl_2026-11-23](../fir/oceananigans_nonhydrostatic_nccl_2026-11-23/README.md).
Once it validates, the Fir figures take their nonhydrostatic 1- and 2-GPU points from it; the 4–16 GPU
points remain the MPI suite, normalized to the suite's own one-GPU time, and each Fir figure's footer says so.

<!-- fir-nonhydrostatic-nccl -->

Status: **PENDING**; the figures still use the MPI suite at every GPU count.

<!-- /fir-nonhydrostatic-nccl -->

The Breeze and Oceananigans workloads differ (Float32 atmosphere vs Float64 ocean
buoyancy case), so the figures compare each code's scaling, not their absolute speed.

### Rondeau (NCCL)

The Rondeau Oceananigans series was rerun on 18 November with NCCL for multi-GPU halo exchange
and the distributed FFT transposes (`gpu_weak_scaling_512_x_nccl`, Oceananigans commit 627157d42,
config `benchmarking/server/rondeau/weak512_nccl_config.json`). MPI still launches the ranks and
times the windows; one-GPU cases have nothing to exchange and use plain `Distributed`.
The workload is otherwise identical to the Fir series above.

The run uses a separate Julia environment,
`/u/ajtse/Oceananigans.jl/.runtime/envs/benchmarking_nccl` (NCCL.jl 0.1.3, NCCL 2.32.3,
CUDA.jl 6.4.2), because no NCCL.jl release works with the CUDA.jl 6.1 pin in Oceananigans'
`benchmarking/Project.toml`. The NCCL series reruns its own one-GPU baseline in that
environment, so its efficiencies do not depend on the MPI series.

<!-- rondeau-nccl-status -->

Status: **RUNNING**, study state `WAITING_FOR_GPUS`.

| GPUs | Hydrostatic NCCL | Nonhydrostatic NCCL | Hydrostatic MPI | Nonhydrostatic MPI |
|---:|---:|---:|---:|---:|
| 1 | 100.0% (0.277 s) | 100.0% (0.582 s) | 100.0% (0.284 s) | 100.0% (0.587 s) |
| 2 | 94.4% (0.293 s) | 77.1% (0.754 s) | 93.7% (0.302 s) | 78.4% (0.748 s) |
| 4 | — | — | — | — |

Efficiency (median seconds per step). Each column is normalized to its own one-GPU case; the MPI columns are the 15 November series, which never reached four GPUs.

<!-- /rondeau-nccl-status -->

Study: `/u/ajtse/Oceananigans.jl/.runtime/studies/2026-11-18_rondeau_weak512_nccl`.
Watcher log: `/u/ajtse/Breeze.jl/.runtime/weak_scaling_efficiency_2026-11-15/watcher_nccl.log`.
The local backend waits for GPUs with no other processes, so the four-GPU cases start only
when other users' jobs leave all four A100s. When the study exits, the watcher reruns
`plot_weak_efficiency.py`, refreshes the table above, and commits and pushes this directory.

## Cross-code comparisons on Fir

Same Fir data as above (H100, 1–16 GPUs, N×1). In these figures color encodes the code
(blue Breeze, orange Oceananigans) and line style plus marker encode the model, so each
model looks the same in every panel. CE is compressible explicit, CSE compressible split-explicit.
The workloads differ, so compare scaling shapes, not absolute speed.

1. **By code**: Oceananigans (hydrostatic, nonhydrostatic) beside Breeze (anelastic, CE, CSE).

   ![Weak scaling on Fir by code](by_code_weak_efficiency_fir.png)

2. **By pressure treatment**: models with no global pressure solve (hydrostatic, CE, CSE)
   beside models with a global elliptic pressure solve (nonhydrostatic, anelastic).

   ![Weak scaling on Fir by pressure treatment](by_solver_weak_efficiency_fir.png)

3. **All five models** in one panel.

   ![Weak scaling on Fir, all five models](all_models_weak_efficiency_fir.png)

## Cross-code comparisons on Rondeau

The same three sets for Rondeau (A100, 1–4 GPUs). The Oceananigans lines are the NCCL rerun;
the Breeze lines are MPI. Breeze compressible uses 2×2 at four GPUs, and its anelastic and
compressible runs come from different source revisions. Until the Oceananigans four-GPU cases
run, its lines stop at the last measured count, and each figure's footer says so.
Do not compare these figures with the Fir figures.

1. **By code**

   ![Weak scaling on Rondeau by code](by_code_weak_efficiency_rondeau.png)

2. **By pressure treatment**

   ![Weak scaling on Rondeau by pressure treatment](by_solver_weak_efficiency_rondeau.png)

3. **All five models**

   ![Weak scaling on Rondeau, all five models](all_models_weak_efficiency_rondeau.png)
