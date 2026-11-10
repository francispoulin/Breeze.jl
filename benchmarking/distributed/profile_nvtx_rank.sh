#!/bin/bash
set -euo pipefail
project=$1
julia=$2
driver=$3
trace_directory=$4
shift 4
rank=${OMPI_COMM_WORLD_RANK:-0}
# Oceananigans assigns node-local ranks to distinct visible GPUs. Keep peers visible
# so GPU-aware MPI and NCCL can use direct GPU communication.
export CUDA_VISIBLE_DEVICES=0,1,2,3
if [[ ${OMPI_COMM_WORLD_SIZE:-1} == 1 ]]; then
    export CUDA_VISIBLE_DEVICES=0
fi
mkdir -p "$trace_directory/rank_$rank"
exec nsys profile --trace=cuda,nvtx \
    --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
    --capture-range-end=stop --force-overwrite=false \
    --output="$trace_directory/rank_$rank/profile" \
    "$julia" --threads=1 --project="$project" "$driver" "$@"
