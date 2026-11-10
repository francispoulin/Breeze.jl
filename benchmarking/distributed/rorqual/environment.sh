#!/bin/bash
# Source before setup, submission, and every case; keep this unchanged during a series.
export JULIA_DEPOT_PATH=/scratch/anditse/julia_depot:
export OCEAN_SOFTWARE_DIR=/scratch/anditse/software
source /home/anditse/Oceananigans-DRAC/env/ocean_setup.sh || return 1
source "$(dirname "${BASH_SOURCE[0]}")/../pelee_rondeau/breeze_env.sh" || return 1
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export JULIA_CUDA_MEMORY_POOL=none JULIA_NUM_PRECOMPILE_TASKS=2
