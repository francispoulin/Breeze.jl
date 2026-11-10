# Rorqual unprofiled weak scaling

Run both compressible dynamics sequentially in one four-node / 16-H100 Slurm allocation.
The agreed matrix has 18 cases: one serial baseline per dynamics and MPI/NCCL at 2, 4, 8,
and 16 GPUs. Four GPUs use global 1024×1024×512 and partition 4×1; eight use 4096×512×512
and 8×1; sixteen use 8192×512×512 and 16×1. Every rank has 512³ cells, with one Julia thread.
There are no Nsight cases or alternative four-GPU partitions.

The tracked driver fixes the physical domain and defaults to WENO5. The runner explicitly sets
Float32, full CBL, 10 warmup steps, 500 timed steps, and 12 acoustic substeps. Δt is 0.01 s
at 512³ and halves whenever spatial resolution doubles: 0.005 s on the 2-/4-GPU grids,
0.00125 s on the 8-GPU grid, and 0.000625 s (1/1600 s) on the 16-GPU grid. Both dynamics
use the same Δt on the same grid, as required by AGENTS.md.
The allocation requests three hours, rather than 18 × 30 minutes, based on the previous
measurements (roughly two hours of stepping). Each individual step has a 30-minute limit.

From the repository root, source `environment.sh`, then run the tracked
`../pelee_rondeau/setup_breeze.sh`. Its final CUDA check must run on a GPU node; login nodes
have no GPUs. Verify the MPI preferences and complete package installation before preparing
provenance. The series repeats the CUDA-aware MPI/GPU preflight inside the allocation.

Prepare a fresh study with `python3 benchmarking/distributed/rorqual/prepare.py ABSOLUTE_STUDY_DIR`.
Commit the source first, so the recorded revision identifies the driver and controller. The
preparation records source hashes, an archive, Julia Project/Manifest/LocalPreferences, and any
working-tree patch. It refuses to overwrite a prepared study. The source/environment hashes
must remain unchanged throughout the series. Do not edit the driver, environment, or runner
while the job is queued or running.

Submit from the repository root with
`sbatch --parsable --output=ABSOLUTE_STUDY_DIR/slurm-%j.out benchmarking/distributed/rorqual/series.sh ABSOLUTE_STUDY_DIR`.
Record that exact command and returned job ID in the study. The job runs every case through
`srun --exact`, using only the required ranks and nodes while retaining the full allocation.
All allocated GPUs remain visible; Oceananigans selects the GPU by node-local MPI rank.

Each case records its exact command, environment, per-rank UUID/communicator/thread placement,
per-rank field validation, final device memory, five-second process-memory samples, CSV timing,
and machine-readable status. Validation checks every rank, unique placement, communicator,
finite fields, 500 steps, and positive finite timing. A failed case is retained and the next case
runs; failures are excluded from derived metrics. The report uses shared serial baselines for
weak efficiency and produces comparison CSV/plots. It explicitly states that finite values do
not demonstrate numerical agreement with a serial solution.

Run `python3 -m unittest discover -s benchmarking/distributed/rorqual -p 'test_*.py'` for validator tests.
After execution, rerun `report.py STUDY_DIR` using the plotting virtual environment, inspect all
case status/log files and Slurm accounting, and commit/push the completed results on
`benchmark-results`. Do not label queued, running, or failed cases as completed measurements.

To retain results after the interactive session ends, start `watch.py STUDY_DIR JOB_ID` with
`nohup` on the login node. It records scheduler/accounting state, regenerates reports, commits
only the study directory on `benchmark-results`, and pushes to `origin` after the allocation ends.
It records incomplete studies as incomplete; its commit message states the validated case count.
GitHub authentication must be available on the login node for the final push.
