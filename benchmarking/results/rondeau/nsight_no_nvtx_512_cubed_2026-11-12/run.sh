#!/bin/bash
set -euo pipefail
cd /u/ajtse/Breeze.jl
source benchmarking/results/rondeau/nsight_no_nvtx_512_cubed_2026-11-12/environment.sh
run_results=/u/ajtse/Breeze.jl/benchmarking/results/rondeau/nsight_no_nvtx_512_cubed_2026-11-12
report_prefix="$profile_runtime/raw/profile"
test ! -e "$report_prefix.nsys-rep"
test ! -e "$run_results/profiled_measurements.csv"
test -z "$(nvidia-smi --id=2 --query-compute-apps=pid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES=2

{
    /u/ajtse/.local/opt/julia-1.12.7/bin/julia --version
    nsys --version
    mpiexec --version
    nvidia-smi
    env | sort | rg '^(CUDA_|JULIA_|OMPI_|OMP_|OPENBLAS_|TMPDIR=|TMP=|TEMP=|XDG_CACHE_HOME=)'
} > "$run_results/provenance/environment.log" 2>&1
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv \
    > "$run_results/provenance/gpu_processes_before.csv"
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv \
    > "$run_results/provenance/gpus_before.csv"

command=(
    mpiexec --bind-to none -n 1
    nsys profile --trace=cuda --sample=none --cpuctxsw=none
    --capture-range=cudaProfilerApi --capture-range-end=stop
    --force-overwrite=false --wait=primary --output="$report_prefix"
    /u/ajtse/.local/opt/julia-1.12.7/bin/julia --threads=1
    --project="$profile_runtime/environment"
    /u/ajtse/Breeze.jl/benchmarking/distributed/distributed_cbl_benchmark.jl
    --dynamics=compressible_splitexplicit --comm=MPI
    --size=512x512x512 --partition=1x1 --float_type=Float32
    --time_step=0.02 --substeps=12 --simplified=false
    --warmup_steps=10 --time_steps=10 --profile=true --nvtx=false
    --random_perturbation=false --results_file="$run_results/profiled_measurements.csv"
)
printf '%q' "${command[0]}" > "$run_results/command.sh"
printf ' %q' "${command[@]:1}" >> "$run_results/command.sh"
printf '\n' >> "$run_results/command.sh"
"${command[@]}" > "$run_results/run.log" 2>&1

nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv \
    > "$run_results/provenance/gpu_processes_after.csv"
nsys export --type=sqlite --force-overwrite=false --output="$report_prefix.sqlite" \
    "$report_prefix.nsys-rep" > "$run_results/export.log" 2>&1
nsys stats --report=cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
    --format=csv --output="$run_results/summary" "$report_prefix.nsys-rep" \
    > "$run_results/stats.log" 2>&1
