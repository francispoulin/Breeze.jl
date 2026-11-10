#!/bin/bash
# Rondeau environment used by this study. The resolved project is retained in provenance/.
set -eo pipefail
source /u/ajtse/.config/breeze/rondeau-mpi/activate.sh
export JULIA_DEPOT_PATH=/u/ajtse/.julia
export JULIA_LOAD_PATH=@:@stdlib
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_NUM_THREADS=1 JULIA_NUM_GC_THREADS=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export JULIA_CUDA_MEMORY_POOL=none JULIA_NVTX_CALLBACKS=gc
export TMPDIR=/local_scratch/ajtse/breeze_nvtx_2026-11-10/tmp
mkdir -p "$TMPDIR"
