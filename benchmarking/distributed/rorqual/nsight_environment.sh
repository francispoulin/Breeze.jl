#!/usr/bin/env bash
# Source for the Rorqual NVTX/Nsight study (nsight.py): setup, submission, execution and finalization.
# Rorqual has the same Alliance module stack as Fir (Julia 1.12.5, CUDA-aware OpenMPI 4.1.5, Nsight
# Systems from cuda/12.6), so this reuses fir/environment.sh with its own Julia environment and
# temporary directory. Nsight reports are large, so they go to /scratch rather than /home.
breeze_nsight_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export BREEZE_FIR_ENV="${BREEZE_FIR_ENV:-${BREEZE_RUNTIME_ROOT:-$breeze_nsight_root/.runtime}/environments/rorqual_nsight}"
unset breeze_nsight_root
source "$(dirname "${BASH_SOURCE[0]}")/../fir/environment.sh" || return 1
export TMPDIR="$BREEZE_RUNTIME_ROOT/tmp/rorqual_nsight"
export TMP="$TMPDIR" TEMP="$TMPDIR" OMPI_MCA_orte_tmpdir_base="$TMPDIR/mpi" CUDA_CACHE_PATH="$TMPDIR/cuda_cache"
export BREEZE_NSIGHT_SCRATCH="${BREEZE_NSIGHT_SCRATCH:-/scratch/$USER/breeze_nsight}"
# GC ranges on the NVTX timeline; per-rank MPI communicator tables in the reports.
export JULIA_NVTX_CALLBACKS=gc NSYS_MPI_STORE_TEAMS_PER_RANK=1
mkdir -p "$TMPDIR/mpi" "$CUDA_CACHE_PATH" "$BREEZE_NSIGHT_SCRATCH"
