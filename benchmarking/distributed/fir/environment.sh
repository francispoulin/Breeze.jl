#!/usr/bin/env bash
# Source for Fir setup, submissions, and execution; keep all generated files in .runtime.
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 julia/1.12.5 cuda/12.6 python/3.11 scipy-stack/2025a || return 1
export BREEZE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export BREEZE_RUNTIME_ROOT="${BREEZE_RUNTIME_ROOT:-$BREEZE_ROOT/.runtime}"
export BREEZE_FIR_ENV="${BREEZE_FIR_ENV:-$BREEZE_RUNTIME_ROOT/environments/fir_weak512}"
export TMPDIR="$BREEZE_RUNTIME_ROOT/tmp/fir_weak512"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export OMPI_MCA_orte_tmpdir_base="$TMPDIR/mpi"
export XDG_CACHE_HOME="$BREEZE_RUNTIME_ROOT/cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/python"
export JULIA_DEPOT_PATH="$BREEZE_RUNTIME_ROOT/depot:$HOME/.julia:"
export JULIA_NUM_THREADS=1 JULIA_NUM_GC_THREADS=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export JULIA_NUM_PRECOMPILE_TASKS=2 JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_CUDA_MEMORY_POOL=none
export OPAL_PREFIX="$EBROOTOPENMPI"
mkdir -p "$BREEZE_FIR_ENV" "$TMPDIR/mpi" "$MPLCONFIGDIR"
