#!/bin/bash
# Source this AFTER `ocean_setup`, in every new terminal, before running the distributed benchmark:
#
#     ocean_setup
#     source benchmarking/distributed/pelee_rondeau/breeze_env.sh
#
# Julia, modules, depot, system MPI and run_local.sh all come from ocean_setup (the oceananigans-hpc
# platform). This checks that its Julia is new enough for Breeze and sets $BREEZE_DISTRIBUTED_DIR.

: "${OCEAN_HPC_ROOT:?run ocean_setup first}"

export BREEZE_DISTRIBUTED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

breeze_julia="$(julia --version 2>/dev/null | awk '{print $3}')"
if [[ "$(printf '%s\n' 1.11.9 "$breeze_julia" | sort -V | head -1)" != 1.11.9 ]]; then
    echo "Breeze needs Julia >= 1.11.9; ocean_setup gives '${breeze_julia:-none}'. Update the platform:"
    echo "    bash \$OCEAN_HPC_ROOT/setup/update.sh"
    unset breeze_julia
    return 1 2>/dev/null || exit 1
fi
echo "Breeze environment: julia $breeze_julia, project=$BREEZE_DISTRIBUTED_DIR"
unset breeze_julia
