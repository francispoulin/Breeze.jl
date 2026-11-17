#!/bin/bash
# Run one MPI rank under Nsight Systems; the capture covers only the driver's
# cudaProfilerApi window (the measured steps after warmup).
set -euo pipefail
project=$1
julia=$2
driver=$3
directory=$4
shift 4
rank=${OMPI_COMM_WORLD_RANK:?MPI rank is required}
# CUDA_VISIBLE_DEVICES lists the two free GPUs chosen by the controller; Oceananigans
# picks one device per node-local rank.
: "${CUDA_VISIBLE_DEVICES:?}"
mkdir -p "$directory/rank_$rank"
exec nsys profile --trace=cuda,nvtx,mpi --mpi-impl=openmpi \
    --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
    --capture-range-end=stop --force-overwrite=false --wait=primary \
    --output="$directory/rank_$rank/profile" \
    "$julia" --startup-file=no --threads=1 --project="$project" "$driver" "$@"
