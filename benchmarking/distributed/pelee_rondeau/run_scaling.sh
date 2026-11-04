#!/bin/bash
# Strong and weak scaling sweep of the distributed CBL benchmark on pelee / rondeau, for the two
# compressible dynamics, plus the 1-GPU baseline of the anelastic model (which is not distributed
# yet). After `ocean_setup` and `source breeze_env.sh`:
#
#     bash benchmarking/distributed/pelee_rondeau/run_scaling.sh
#     bash benchmarking/distributed/pelee_rondeau/run_scaling.sh --time_steps=1000 --max_gpus=2
#
# Options: --time_steps (default 500), --max_gpus (default 4), --results_file (CSV, appended to;
# default $OCEAN_RESULTS_ROOT/Breeze/scaling.csv). Each run is attached (run_local.sh --wait), one
# after the other; its log is in logs/ of the folder you start from.
#
# Strong scaling: 256x256x128 on 1, 2 and 4 GPUs. Weak scaling: 256x256x128 per 1 GPU, then the
# grid doubles in x for 2 GPUs and in x and y for 4 GPUs (about 8M points per GPU throughout).

: "${OCEAN_HPC_ROOT:?run ocean_setup first}"
: "${BREEZE_DISTRIBUTED_DIR:?source breeze_env.sh first}"

busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 > 500 {print "GPU " $1 ": " $2 " MiB"}')
[[ -n $busy ]] && { echo "GPUs already in use, aborting:"; echo "$busy"; exit 1; }

time_steps=500
max_gpus=4
results_file="${OCEAN_RESULTS_ROOT:-$PWD}/Breeze/scaling.csv"
for argument in "$@"; do
    case "$argument" in
        --time_steps=*)   time_steps="${argument#--time_steps=}" ;;
        --max_gpus=*)     max_gpus="${argument#--max_gpus=}" ;;
        --results_file=*) results_file="${argument#--results_file=}" ;;
        *) echo "unknown option: $argument"; exit 1 ;;
    esac
done
mkdir -p "$(dirname "$results_file")"

# gpus, global size, partition
cases=("1 256x256x128 1x1" "2 256x256x128 2x1" "2 256x256x128 1x2" "4 256x256x128 2x2" "4 256x256x128 4x1" "4 256x256x128 1x4" "2 512x256x128 2x1" "4 512x512x128 2x2")

for dynamics in anelastic compressible_explicit compressible_splitexplicit; do
    for case in "${cases[@]}"; do
        read -r gpus size partition <<<"$case"
        (( gpus > max_gpus )) && continue
        [[ $dynamics == anelastic && $gpus -gt 1 ]] && continue     # not distributed yet
        echo "== $dynamics, $gpus GPU(s), $size, partition $partition"
        bash "$OCEAN_HPC_ROOT/tools/run_local.sh" -n "$gpus" -p "$BREEZE_DISTRIBUTED_DIR" --wait \
            "$BREEZE_DISTRIBUTED_DIR/distributed_cbl_benchmark.jl" \
            --dynamics="$dynamics" --size="$size" --partition="$partition" \
            --time_steps="$time_steps" --results_file="$results_file" \
            || echo "   FAILED: $dynamics, $gpus GPU(s), $size"
    done
done

echo "Results: $results_file"
