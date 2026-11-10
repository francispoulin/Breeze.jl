#!/bin/bash
set -eo pipefail
cd /u/ajtse/Breeze.jl
export OCEAN_SOFTWARE_DIR=/u/ajtse/software
export OCEAN_RESULTS_ROOT=/u/ajtse/Breeze.jl/benchmarking/results/rondeau/weak_scaling_mpi_nccl_2026-11-09_dt0p005
export OCEAN_DATA_DIR=/u/ajtse/software/ocean_data
export JULIA_DEPOT_PATH=/u/ajtse/.julia JULIA_LOAD_PATH=@:@stdlib
export TMPDIR=/u/ajtse/software/tmp/breeze-weak-scaling-mpi-nccl-2026-11-09-dt0p005
mkdir -p "$TMPDIR"
export TMP="$TMPDIR" TEMP="$TMPDIR" OMPI_MCA_orte_tmpdir_base="$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
unset BASH_ENV SSH_CLIENT SSH_CONNECTION SSH_TTY SSH2_CLIENT
source /u/ajtse/software/oceananigans-hpc/env/ocean_setup.sh
source benchmarking/distributed/pelee_rondeau/breeze_env.sh
unset BASH_ENV SSH_CLIENT SSH_CONNECTION SSH_TTY SSH2_CLIENT
export JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=2
unset CUDA_VISIBLE_DEVICES
export JULIA_CUDA_MEMORY_POOL=none NCCL_DEBUG=WARN
