#!/usr/bin/env bash
set -eo pipefail
export BREEZE_RUNTIME_ROOT=/home/anditse/Breeze.jl/.runtime
export BREEZE_FIR_ENV=/home/anditse/Breeze.jl/.runtime/studies/nsight_anelastic_ce_1_4_8gpu_2026-11-24/environment
source /home/anditse/Breeze.jl/.runtime/studies/nsight_anelastic_ce_1_4_8gpu_2026-11-24/source/benchmarking/distributed/rorqual/nsight_environment.sh
module -t list > /home/anditse/Breeze.jl/benchmarking/results/rorqual/nsight_anelastic_ce_1_4_8gpu_2026-11-24/allocations/4g_retry1/environment.log 2>&1
python3 -c 'import json, sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); from rorqual import nsight; study = Path(sys.argv[2]); plan = {c['"'"'id'"'"']: c for c in json.loads((study / '"'"'plan.json'"'"').read_text())}; sys.exit(0 if all([nsight.run_case(study, plan[name]) for name in sys.argv[3:]]) else 1)' /home/anditse/Breeze.jl/.runtime/studies/nsight_anelastic_ce_1_4_8gpu_2026-11-24/source/benchmarking/distributed /home/anditse/Breeze.jl/benchmarking/results/rorqual/nsight_anelastic_ce_1_4_8gpu_2026-11-24 compressible_explicit_NCCL_4g_4x1
