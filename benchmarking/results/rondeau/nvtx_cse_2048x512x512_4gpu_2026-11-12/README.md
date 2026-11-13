# Four-GPU split-explicit CBL: NVTX and Nsight

Status: **QUEUED: waiting for preceding checks and free GPUs**.

Global grid 2048×512×512; partition 4×1; local grid 512³ per A100; Float32;
full dry CBL with WENO5 and deterministic perturbations. Δt = 0.0025 s follows
the resolution rule in AGENTS.md. Twelve nominal acoustic substeps give 22
substeps per outer RK step. Each launch has ten warmup and ten measured steps.

The order is NCCL benchmark, MPI benchmark, NCCL capture, MPI capture.
MPI is the requested case without NCCL; NCCL still uses MPI for launch/control.
Both install the existing Breeze host NVTX ranges. Nsight records CUDA, NVTX
and MPI only during measured steps, using cudaProfilerApi start/stop calls.

The detached checkout and Julia environment live under:
`/u/ajtse/Breeze.jl/.runtime/nvtx_cse_2048x512x512_4gpu_2026-11-12`

## Timings and validation

| Launch | State | Unprofiled seconds/step | Maximum absolute w |
|--------|-------|-------------------------|--------------------|

Unprofiled wall timings and captured timings are kept separate. The driver's
profiled timer can include report generation and must not be used as a simulation
performance measurement. Trace windows, GPU activity and nested host ranges are
reported per rank in analysis.json; nested ranges must not be added together.
Finite-field checks establish a functional run, not field-by-field numerical accuracy.
The source-review communication concerns in .agents/nvidia_optimization_advice.md
remain unverified by these dry CBL captures.

## Exact Nsight result paths

These are planned paths until each report is recorded in captures.json.

- `/u/ajtse/Breeze.jl/.runtime/nvtx_cse_2048x512x512_4gpu_2026-11-12/raw/NCCL/rank_0/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/nvtx_cse_2048x512x512_4gpu_2026-11-12/raw/NCCL/rank_1/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/nvtx_cse_2048x512x512_4gpu_2026-11-12/raw/NCCL/rank_2/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/nvtx_cse_2048x512x512_4gpu_2026-11-12/raw/NCCL/rank_3/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/nvtx_cse_2048x512x512_4gpu_2026-11-12/raw/MPI/rank_0/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/nvtx_cse_2048x512x512_4gpu_2026-11-12/raw/MPI/rank_1/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/nvtx_cse_2048x512x512_4gpu_2026-11-12/raw/MPI/rank_2/profile.nsys-rep`
- `/u/ajtse/Breeze.jl/.runtime/nvtx_cse_2048x512x512_4gpu_2026-11-12/raw/MPI/rank_3/profile.nsys-rep`

SQLite exports stay alongside the raw reports. All raw data, temporary files,
MPI sessions, caches and environments stay under /u/ajtse; .runtime is ignored.
The tracked report contains commands, environment/source provenance, per-launch
logs and CSVs, per-rank Nsight summaries and report sizes/SHA-256 hashes.

## Automatic execution

run_campaign.py checks every 30 seconds for four GPUs with no compute processes
and at least 35000 MiB free each. It waits for the previously queued four-GPU
smoke checks to finish successfully, then runs each launch without sharing GPUs.
It verifies the frozen source hashes before launches and after analysis, refuses
to overwrite output, and records failures. A detached watcher survives this chat.
Live state: `/u/ajtse/Breeze.jl/.runtime/nvtx_cse_2048x512x512_4gpu_2026-11-12/status.json`.
Full controller log: `/u/ajtse/Breeze.jl/.runtime/nvtx_cse_2048x512x512_4gpu_2026-11-12/logs/controller.log`.

The live report directory is `/u/ajtse/Breeze.jl/.runtime/nvtx_cse_2048x512x512_4gpu_2026-11-12/worktree/benchmarking/results/rondeau/nvtx_cse_2048x512x512_4gpu_2026-11-12`.
The original checkout's README remains the initial queued snapshot until pulled.

After completion it commits this report directory and immediately attempts a normal
push to origin/NVTX-optimization. Concurrent remote changes cause a push failure
recorded in status.json; it never force-pushes. No plots are produced for this run.
