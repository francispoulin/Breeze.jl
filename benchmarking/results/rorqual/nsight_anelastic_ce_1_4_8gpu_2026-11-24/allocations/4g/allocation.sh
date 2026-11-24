#!/usr/bin/env bash
set -eo pipefail
export BREEZE_RUNTIME_ROOT=/home/anditse/Breeze.jl/.runtime
export BREEZE_FIR_ENV=/home/anditse/Breeze.jl/.runtime/studies/nsight_anelastic_ce_1_4_8gpu_2026-11-24/environment
source /home/anditse/Breeze.jl/.runtime/studies/nsight_anelastic_ce_1_4_8gpu_2026-11-24/source/benchmarking/distributed/rorqual/nsight_environment.sh
python3 /home/anditse/Breeze.jl/.runtime/studies/nsight_anelastic_ce_1_4_8gpu_2026-11-24/source/benchmarking/distributed/rorqual/nsight.py run-size /home/anditse/Breeze.jl/benchmarking/results/rorqual/nsight_anelastic_ce_1_4_8gpu_2026-11-24 4
