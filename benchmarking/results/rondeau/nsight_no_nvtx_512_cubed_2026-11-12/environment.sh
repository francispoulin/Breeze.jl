#!/bin/bash
# Same package, MPI, threading and memory-pool configuration as the NVTX demo.
# All new temporary, cache, environment and trace files stay under /u/ajtse.
export PATH="/opt/uw/openmpi/4.1.7-with-cuda/bin:$PATH"
export LD_LIBRARY_PATH="/opt/uw/openmpi/4.1.7-with-cuda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export JULIA_DEPOT_PATH=/u/ajtse/.julia
export JULIA_LOAD_PATH=@:@stdlib
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_NUM_THREADS=1 JULIA_NUM_GC_THREADS=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export JULIA_CUDA_MEMORY_POOL=none
unset JULIA_NVTX_CALLBACKS
profile_runtime=/u/ajtse/Breeze.jl/.runtime/nsight_no_nvtx_512_cubed_2026-11-12
export TMPDIR="$profile_runtime/tmp" TMP="$profile_runtime/tmp" TEMP="$profile_runtime/tmp"
export OMPI_MCA_orte_tmpdir_base="$profile_runtime/mpi"
export XDG_CACHE_HOME="$profile_runtime/cache"
export CUDA_CACHE_PATH="$profile_runtime/cuda_cache"
mkdir -p "$TMPDIR" "$OMPI_MCA_orte_tmpdir_base" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"
