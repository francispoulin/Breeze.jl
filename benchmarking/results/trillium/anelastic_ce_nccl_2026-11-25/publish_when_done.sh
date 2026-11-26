#!/usr/bin/env bash
set -eo pipefail
export BREEZE_RUNTIME_ROOT=/scratch/anditse/Breeze.jl/.runtime
export BREEZE_FIR_ENV=/scratch/anditse/Breeze.jl/.runtime/studies/anelastic_ce_nccl_2026-11-25/environment
source /scratch/anditse/Breeze.jl/.runtime/studies/anelastic_ce_nccl_2026-11-25/source/benchmarking/distributed/trillium/environment.sh
jobs=$(python3 -c 'import json, sys; print(",".join(job["job_id"] for job in json.load(open(sys.argv[1]))["jobs"]))' /scratch/anditse/Breeze.jl/studies/anelastic_ce_nccl_2026-11-25/submission.json)
while [[ -n "$(squeue --noheader --jobs="$jobs" 2>/dev/null)" ]]; do sleep 600; done
python3 /scratch/anditse/Breeze.jl/.runtime/studies/anelastic_ce_nccl_2026-11-25/source/benchmarking/distributed/fir/study.py finalize /scratch/anditse/Breeze.jl/studies/anelastic_ce_nccl_2026-11-25 --publish
