#!/bin/bash
set -euo pipefail
project=$1
julia=$2
driver=$3
directory=$4
shift 4
rank=${OMPI_COMM_WORLD_RANK:?MPI rank is required}
# Keep all peers visible; Oceananigans chooses one device per node-local rank.
export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir -p "$directory/rank_$rank"
exec nsys profile --trace=cuda,nvtx,mpi --mpi-impl=openmpi \
    --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
    --capture-range-end=stop --force-overwrite=false --wait=primary \
    --output="$directory/rank_$rank/profile" \
    "$julia" --startup-file=no --threads=1 --project="$project" "$driver" "$@"
