#!/bin/bash
set -euo pipefail
export PATH=/opt/uw/nsight/2024.5.1/bin:$PATH
case_directory=$1
raw_directory=$2
project=$3
driver=$4
shift 4
rank=${OMPI_COMM_WORLD_RANK:-0}
local_rank=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
IFS=, read -r -a selected_gpus <<< "$NSIGHT_GPU_IDS"
export CUDA_VISIBLE_DEVICES=${selected_gpus[$local_rank]}
rank_directory="$case_directory/rank_$rank"
trace_directory="$raw_directory/rank_$rank"
mkdir -p "$rank_directory" "$trace_directory"
nsys profile --trace=cuda,nvtx,mpi --mpi-impl=openmpi --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=false \
    --output="$trace_directory/profile" \
    julia --threads=1 --project="$project" "$driver" "$@" > "$rank_directory/profile.out" 2>&1
test -s "$trace_directory/profile.nsys-rep"
nsys stats --report=cuda_gpu_kern_sum --format=csv --force-export=true --force-overwrite=false \
    --output="$rank_directory/summary" "$trace_directory/profile.nsys-rep" > "$rank_directory/stats.out" 2>&1
for report in cuda_api_sum mpi_event_sum nvtx_sum; do
    status=0
    nsys stats --report="$report" --format=csv --force-overwrite=false \
        --output="$rank_directory/summary" "$trace_directory/profile.nsys-rep" \
        > "$rank_directory/$report.out" 2>&1 || status=$?
    printf '%s\n' "$status" > "$rank_directory/${report}_exit_code.txt"
done
printf '0\n' > "$rank_directory/exit_code.txt"
