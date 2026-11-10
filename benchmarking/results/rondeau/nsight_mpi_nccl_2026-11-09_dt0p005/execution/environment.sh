#!/bin/bash
set -eo pipefail
source /u/ajtse/Breeze.jl/benchmarking/results/rondeau/weak_scaling_mpi_nccl_2026-11-09_dt0p005/execution/environment.sh
export PATH=/opt/uw/nsight/2024.5.1/bin:$PATH
export OCEAN_RESULTS_ROOT=/u/ajtse/Breeze.jl/benchmarking/results/rondeau/nsight_mpi_nccl_2026-11-09_dt0p005
export BREEZE_DISTRIBUTED_DIR=/local_scratch/ajtse/breeze_nsight_mpi_nccl_2026-11-09/source/benchmarking/distributed
export TMPDIR=/local_scratch/ajtse/breeze_nsight_mpi_nccl_2026-11-09/tmp
mkdir -p "$TMPDIR"
export TMP="$TMPDIR" TEMP="$TMPDIR" OMPI_MCA_orte_tmpdir_base="$TMPDIR"
export JULIA_NUM_GC_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
export NSYS_MPI_STORE_TEAMS_PER_RANK=1 NCCL_DEBUG=WARN
