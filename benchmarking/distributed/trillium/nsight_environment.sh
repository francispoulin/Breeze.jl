#!/usr/bin/env bash
# Source for the Trillium NVTX/Nsight series (rorqual/nsight.py --series=trillium_4_8) on trig-login01
# and inside its GPU jobs. It reuses the Trillium module stack and Julia environment (environment.sh;
# Nsight Systems comes with cuda/12.6) with its own temporary directory. Reports go to $SCRATCH.
source "$(dirname "${BASH_SOURCE[0]}")/environment.sh" || return 1
export TMPDIR="$BREEZE_RUNTIME_ROOT/tmp/trillium_nsight"
export TMP="$TMPDIR" TEMP="$TMPDIR" OMPI_MCA_orte_tmpdir_base="$TMPDIR/mpi" CUDA_CACHE_PATH="$TMPDIR/cuda_cache"
export BREEZE_NSIGHT_SCRATCH="${BREEZE_NSIGHT_SCRATCH:-$SCRATCH/breeze_nsight}"
# GC ranges on the NVTX timeline; per-rank MPI communicator tables in the reports.
export JULIA_NVTX_CALLBACKS=gc NSYS_MPI_STORE_TEAMS_PER_RANK=1
mkdir -p "$TMPDIR/mpi" "$CUDA_CACHE_PATH" "$BREEZE_NSIGHT_SCRATCH"
