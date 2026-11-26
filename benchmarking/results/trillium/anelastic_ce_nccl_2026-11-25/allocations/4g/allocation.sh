#!/usr/bin/env bash
set -eo pipefail
export BREEZE_RUNTIME_ROOT=/scratch/anditse/Breeze.jl/.runtime
export BREEZE_FIR_ENV=/scratch/anditse/Breeze.jl/.runtime/studies/anelastic_ce_nccl_2026-11-25/environment
source /scratch/anditse/Breeze.jl/.runtime/studies/anelastic_ce_nccl_2026-11-25/source/benchmarking/distributed/trillium/environment.sh
python3 /scratch/anditse/Breeze.jl/.runtime/studies/anelastic_ce_nccl_2026-11-25/source/benchmarking/distributed/fir/study.py run-size /scratch/anditse/Breeze.jl/studies/anelastic_ce_nccl_2026-11-25 4
