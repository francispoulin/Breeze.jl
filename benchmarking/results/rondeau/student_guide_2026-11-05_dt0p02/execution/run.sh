#!/bin/bash
# Reproduce the 17-case student-guide sweep with only the time step changed.
set -eo pipefail
cd /u/ajtse/Breeze.jl
export OCEAN_SOFTWARE_DIR=/u/ajtse/software
export OCEAN_RESULTS_ROOT="$PWD/benchmarking/results/rondeau/student_guide_2026-11-05_dt0p02"
export OCEAN_DATA_DIR=/u/ajtse/software/ocean_data
export JULIA_DEPOT_PATH=/u/ajtse/.julia JULIA_LOAD_PATH=@:@stdlib
export TMPDIR=/u/ajtse/software/tmp/breeze-student-guide-2026-11-05-dt0p02
mkdir -p "$TMPDIR" "$OCEAN_RESULTS_ROOT/runs" "$OCEAN_RESULTS_ROOT/provenance"
export TMP="$TMPDIR" TEMP="$TMPDIR" OMPI_MCA_orte_tmpdir_base="$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
unset BASH_ENV SSH_CLIENT SSH_CONNECTION SSH_TTY SSH2_CLIENT
source /u/ajtse/software/oceananigans-hpc/env/ocean_setup.sh
source benchmarking/distributed/pelee_rondeau/breeze_env.sh
unset BASH_ENV SSH_CLIENT SSH_CONNECTION SSH_TTY SSH2_CLIENT
export JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=2
test ! -e "$OCEAN_RESULTS_ROOT/scaling.csv"

trap 'status=$?; printf "%s\n" "$status" > "$OCEAN_RESULTS_ROOT/execution/exit_code.txt"; date -u "+%FT%TZ" > "$OCEAN_RESULTS_ROOT/execution/finished_utc.txt"' EXIT
date -u '+%FT%TZ' > "$OCEAN_RESULTS_ROOT/execution/started_utc.txt"
git status --short > "$OCEAN_RESULTS_ROOT/provenance/working_tree_before.txt"
git ls-tree -r HEAD src > "$OCEAN_RESULTS_ROOT/provenance/source_tree.txt"
cp "$OCEAN_HPC_ROOT/tools/run_local.sh" "$OCEAN_RESULTS_ROOT/provenance/run_local.sh"
sha256sum "$OCEAN_HPC_ROOT/tools/run_local.sh" "$OCEAN_HPC_ROOT/env/ocean_setup.sh" "$OCEAN_HPC_ROOT/env/fluidsgpu.sh" > "$OCEAN_RESULTS_ROOT/provenance/platform_sha256.txt"
git rev-parse HEAD > "$OCEAN_RESULTS_ROOT/provenance/source_commit.txt"
git diff --unified=0 -- benchmarking/distributed/pelee_rondeau/run_scaling.sh > "$OCEAN_RESULTS_ROOT/provenance/sweep.patch"
git diff --exit-code HEAD -- src Project.toml benchmarking/src/convective_boundary_layer.jl benchmarking/distributed/distributed_cbl_benchmark.jl
git -C "$OCEAN_HPC_ROOT" rev-parse HEAD > "$OCEAN_RESULTS_ROOT/provenance/platform_commit.txt"
cp "$BREEZE_DISTRIBUTED_DIR/Project.toml" "$BREEZE_DISTRIBUTED_DIR/Manifest.toml" "$BREEZE_DISTRIBUTED_DIR/LocalPreferences.toml" "$OCEAN_RESULTS_ROOT/provenance/"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv > "$OCEAN_RESULTS_ROOT/provenance/gpus.csv"
sha256sum src/Breeze.jl benchmarking/src/convective_boundary_layer.jl benchmarking/distributed/distributed_cbl_benchmark.jl benchmarking/distributed/pelee_rondeau/run_scaling.sh > "$OCEAN_RESULTS_ROOT/provenance/source_sha256.txt"
julia --project="$BREEZE_DISTRIBUTED_DIR" -e 'using MPI, CUDA, Breeze; MPI.Init(); @assert VERSION == v"1.11.9"; @assert MPI.has_cuda(); @assert CUDA.functional(); println("Julia: ", VERSION); println("Breeze source: ", pathof(Breeze)); println("Project: ", Base.active_project()); println("MPI CUDA-aware: ", MPI.has_cuda()); println(MPI.Get_library_version()); println("CUDA devices: ", collect(CUDA.devices())); println("Threads: ", Threads.nthreads()); println("Memory pool in workers: run_local.sh sets none"); MPI.Finalize()' > "$OCEAN_RESULTS_ROOT/provenance/environment.log" 2>&1
cat "$OCEAN_RESULTS_ROOT/provenance/environment.log"
cd "$OCEAN_RESULTS_ROOT/runs"
bash "$BREEZE_DISTRIBUTED_DIR/pelee_rondeau/run_scaling.sh" \
    --time_steps=500 --time_step=0.02 --max_gpus=4 \
    --results_file="$OCEAN_RESULTS_ROOT/scaling.csv"
date -u '+%FT%TZ' > "$OCEAN_RESULTS_ROOT/execution/finished_utc.txt"
