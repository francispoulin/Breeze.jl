#!/bin/bash
#SBATCH --account=def-fpoulin_gpu
#SBATCH --job-name=breeze_weak18
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=03:00:00
cd "${SLURM_SUBMIT_DIR:?}"
source benchmarking/distributed/rorqual/environment.sh || exit 1
set -eo pipefail
study=$1
python3 "$BREEZE_DISTRIBUTED_DIR/rorqual/run_series.py" "$study"
