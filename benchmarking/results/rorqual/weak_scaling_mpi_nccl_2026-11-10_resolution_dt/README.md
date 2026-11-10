# Rorqual compressible MPI/NCCL weak scaling

Submitted as Slurm job **26313924**, one four-node / 16-H100 allocation with a three-hour limit.
See [submission command](submission_command.sh), [submission record](submission.json), and
[live scheduler snapshot](scheduler_status.json). A login-node watcher will retain accounting,
generate the final report/plots, and commit/push results when the allocation ends.

**0/18 cases validated.** See [progress](progress.json) and [case plan](plan.json).

Full dry CBL, Float32, WENO5, fixed 12 km × 12 km × 3 km domain; both dynamics use the same Δt per grid.
10 warmup and 500 timed steps. Split-explicit uses 12 acoustic substeps. No Nsight runs.

| GPUs | Global grid | Partition | Backends | Δt (s) |
|---|---|---|---|---|
| 1 | 512×512×512 | 1×1×1 | Shared SERIAL | 0.01 |
| 2 | 1024×512×512 | 2×1×1 | MPI, NCCL | 0.005 |
| 4 | 1024×1024×512 | 4×1×1 | MPI, NCCL | 0.005 |
| 8 | 4096×512×512 | 8×1×1 | MPI, NCCL | 0.00125 |
| 16 | 8192×512×512 | 16×1×1 | MPI, NCCL | 0.000625 |

One 16-GPU/four-node Slurm allocation runs cases sequentially, one Julia thread per rank,
system CUDA-aware MPI and JULIA_CUDA_MEMORY_POOL=none. Each case has a 30-minute execution cap;
the allocation requests three hours based on previous stepping times plus compilation allowance.

Every accepted case requires finite prognostic and velocity fields on every rank, unique GPU placement,
the requested communicator, and positive finite timing. Timing uses the slowest synchronized rank.
Weak efficiency is 100 × serial seconds/step ÷ distributed seconds/step. Finiteness does not
establish agreement with a serial solution. Each grid advances 500 steps, so simulated durations vary with Δt.

[Source revision](provenance/source_commit.txt), [working-tree patch](provenance/source_changes.patch),
[frozen hashes](provenance/hashes.json), [environment setup](provenance/setup.log),
[manifest](provenance/Manifest.toml), and [source archive](provenance/source.tar.gz).
Per-case directories contain exact commands, environment records, per-rank logs, final device-memory
snapshots and sampled per-process GPU memory (MiB, every five seconds; sampled maxima can miss peaks).

| Case | State | s/step | Points/s | Weak efficiency |
|---|---|---|---|---|
| compressible_explicit_SERIAL_1g_1x1 | pending |  |  |  |
| compressible_explicit_MPI_2g_2x1 | pending |  |  |  |
| compressible_explicit_NCCL_2g_2x1 | pending |  |  |  |
| compressible_explicit_MPI_4g_4x1 | pending |  |  |  |
| compressible_explicit_NCCL_4g_4x1 | pending |  |  |  |
| compressible_explicit_MPI_8g_8x1 | pending |  |  |  |
| compressible_explicit_NCCL_8g_8x1 | pending |  |  |  |
| compressible_explicit_MPI_16g_16x1 | pending |  |  |  |
| compressible_explicit_NCCL_16g_16x1 | pending |  |  |  |
| compressible_splitexplicit_SERIAL_1g_1x1 | pending |  |  |  |
| compressible_splitexplicit_MPI_2g_2x1 | pending |  |  |  |
| compressible_splitexplicit_NCCL_2g_2x1 | pending |  |  |  |
| compressible_splitexplicit_MPI_4g_4x1 | pending |  |  |  |
| compressible_splitexplicit_NCCL_4g_4x1 | pending |  |  |  |
| compressible_splitexplicit_MPI_8g_8x1 | pending |  |  |  |
| compressible_splitexplicit_NCCL_8g_8x1 | pending |  |  |  |
| compressible_splitexplicit_MPI_16g_16x1 | pending |  |  |  |
| compressible_splitexplicit_NCCL_16g_16x1 | pending |  |  |  |
