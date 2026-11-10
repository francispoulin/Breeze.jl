#!/bin/bash
set -euo pipefail
case_directory=$1
shift
rank=${SLURM_PROCID:?not launched by srun}
if [[ ${SLURM_LOCALID:-0} == 0 ]]; then
    nvidia-smi -q > "$case_directory/hardware_rank_$rank.txt"
fi
# Oceananigans selects the node-local GPU using MPI's shared-memory rank.
# Keep every allocated GPU visible; do not mask to one GPU per task.
(
    while true; do
        date -u +%FT%TZ
        nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits || true
        sleep 5
    done
) > "$case_directory/memory_rank_$rank.csv" 2>&1 &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true' EXIT
"$@" > "$case_directory/rank_$rank.log" 2>&1
