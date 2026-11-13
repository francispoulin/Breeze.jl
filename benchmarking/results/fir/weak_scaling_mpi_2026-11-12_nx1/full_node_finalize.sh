#!/usr/bin/env bash
set -eo pipefail
export BREEZE_RUNTIME_ROOT=/home/anditse/Breeze.jl/.runtime
export BREEZE_FIR_ENV=/home/anditse/Breeze.jl/.runtime/studies/weak_scaling_mpi_2026-11-12_nx1/environment
source /home/anditse/Breeze.jl/.runtime/studies/weak_scaling_mpi_2026-11-12_nx1/full_node_source/benchmarking/distributed/fir/environment.sh
python3 -c 'import json,hashlib,pathlib; h=json.load(open('"'"'/home/anditse/Breeze.jl/benchmarking/results/fir/weak_scaling_mpi_2026-11-12_nx1/provenance/full_node_hashes.json'"'"')); assert all(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==d for p,d in h.items()), '"'"'Full-node controller changed'"'"''
python3 /home/anditse/Breeze.jl/.runtime/studies/weak_scaling_mpi_2026-11-12_nx1/full_node_source/benchmarking/distributed/fir/study.py finalize /home/anditse/Breeze.jl/benchmarking/results/fir/weak_scaling_mpi_2026-11-12_nx1 --publish
