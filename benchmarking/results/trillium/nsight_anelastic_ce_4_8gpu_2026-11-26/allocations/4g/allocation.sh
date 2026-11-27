#!/usr/bin/env bash
set -eo pipefail
export BREEZE_RUNTIME_ROOT=/scratch/anditse/Breeze.jl/.runtime
export BREEZE_FIR_ENV=/scratch/anditse/Breeze.jl/.runtime/studies/nsight_anelastic_ce_4_8gpu_2026-11-26/environment
source /scratch/anditse/Breeze.jl/.runtime/studies/nsight_anelastic_ce_4_8gpu_2026-11-26/source/benchmarking/distributed/trillium/nsight_environment.sh
python3 /scratch/anditse/Breeze.jl/.runtime/studies/nsight_anelastic_ce_4_8gpu_2026-11-26/source/benchmarking/distributed/rorqual/nsight.py run-size /scratch/anditse/Breeze.jl/studies/nsight_anelastic_ce_4_8gpu_2026-11-26 4
