# Two-GPU NCCL Nsight: Breeze anelastic and Oceananigans nonhydrostatic

Status: **QUEUED**.

Global grid 1024×512×512, partition 2×1, 512³ cells per A100, NCCL halo and
transpose communication (MPI only launches ranks and supplies barriers). Each launch
has ten warmup and ten measured steps and installs host NVTX ranges.

| Case | Model | Precision | Δt |
|------|-------|-----------|----|
| `anelastic` | Breeze `AtmosphereModel`, `AnelasticDynamics`, dry CBL, WENO5 | Float32 | 0.005 s |
| `oceananigans_nonhydrostatic` | Oceananigans `NonhydrostaticModel`, suite `rectilinear_buoyancy`, WENO5 | Float64 | 10 s |

Both pressure solves are Oceananigans' distributed FFT solvers, whose transposes
use NCCL here. Δt = 0.005 s follows the AGENTS.md rule for Breeze at this grid;
the Oceananigans case keeps its suite's Float64 and Δt = 10 s (1 km × 1 km × 10 m
cells), so its cost per step matches the Oceananigans weak-scaling series. Both use
Oceananigans 0.117.2 from this benchmark environment, not the Oceananigans checkout.
Compare the two models by where time goes, not by absolute speed.

For each case the order is unprofiled NCCL benchmark, then NCCL Nsight capture.
Nsight records CUDA, NVTX and MPI only during the measured steps
(cudaProfilerApi start/stop).

## Timings and validation

| Launch | State | Unprofiled seconds/step | Maximum absolute w | GPUs |
|--------|-------|-------------------------|--------------------|------|

Finite-field checks establish a functional run, not field-by-field accuracy.
No plots are produced for Nsight runs.

## Exact Nsight result paths

- `/u/ajtse/Breeze.jl/.runtime/nccl_nsight_2gpu_2026-11-17/raw/anelastic/rank_0/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/nccl_nsight_2gpu_2026-11-17/raw/anelastic/rank_1/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/nccl_nsight_2gpu_2026-11-17/raw/oceananigans_nonhydrostatic/rank_0/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/nccl_nsight_2gpu_2026-11-17/raw/oceananigans_nonhydrostatic/rank_1/profile.nsys-rep`

SQLite exports sit beside each report. captures.json records report sizes,
SHA-256 hashes, kernel and range counts. Per-rank `nsys stats` summaries are in logs/.

## Automatic execution

run_campaign.py runs from the frozen checkout `/u/ajtse/Breeze.jl/.runtime/nccl_nsight_2gpu_2026-11-17/worktree`
(commit `50afa1d432120ef9832d7cc6646996155d94185c`) with the Julia environment
`/u/ajtse/Breeze.jl/.runtime/nccl_nsight_2gpu_2026-11-17/environment`. Every 30 s it looks for two GPUs with no
compute processes and at least 35000 MiB free, and sets
CUDA_VISIBLE_DEVICES to that pair for one launch.
Live state: `/u/ajtse/Breeze.jl/.runtime/nccl_nsight_2gpu_2026-11-17/status.json`. Controller log: `/u/ajtse/Breeze.jl/.runtime/nccl_nsight_2gpu_2026-11-17/logs/controller.log`.
On completion it commits this directory to benchmark-results and pushes.
