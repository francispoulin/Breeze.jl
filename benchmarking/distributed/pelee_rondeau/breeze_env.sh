#!/bin/bash
# Source this AFTER `ocean_setup`, in every new terminal, before running the distributed benchmark:
#
#     ocean_setup
#     source benchmarking/distributed/pelee_rondeau/breeze_env.sh
#
# ocean_setup gives the platform's Julia (1.10.10); Breeze needs >= 1.11.9, so put that Julia first
# on PATH. Everything else (modules, depot, system MPI, run_local.sh) comes from ocean_setup.

: "${OCEAN_HPC_ROOT:?run ocean_setup first}"

# Julia installs live next to the oceananigans-hpc clone, as the platform's own does.
export BREEZE_SOFTWARE_DIR="${BREEZE_SOFTWARE_DIR:-$(dirname "$OCEAN_HPC_ROOT")}"
export BREEZE_JULIA_VERSION="${BREEZE_JULIA_VERSION:-1.11.9}"
export BREEZE_DISTRIBUTED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PATH="$BREEZE_SOFTWARE_DIR/julia-$BREEZE_JULIA_VERSION/bin:$PATH"

echo "Breeze environment: $(julia --version 2>&1), project=$BREEZE_DISTRIBUTED_DIR"
