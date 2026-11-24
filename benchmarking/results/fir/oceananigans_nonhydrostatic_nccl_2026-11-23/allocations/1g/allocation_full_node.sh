#!/usr/bin/env bash
set -eo pipefail
export BREEZE_RUNTIME_ROOT=/home/anditse/Breeze.jl/.runtime
export BREEZE_FIR_ENV=/home/anditse/Breeze.jl/.runtime/studies/oceananigans_nonhydrostatic_nccl_2026-11-23/environment
source /home/anditse/Breeze.jl/.runtime/studies/oceananigans_nonhydrostatic_nccl_2026-11-23/source/benchmarking/distributed/fir/environment.sh
python3 /home/anditse/Breeze.jl/.runtime/studies/oceananigans_nonhydrostatic_nccl_2026-11-23/source/benchmarking/distributed/fir/study.py run-size /home/anditse/Breeze.jl/benchmarking/results/fir/oceananigans_nonhydrostatic_nccl_2026-11-23 1 --node-gpus=4
