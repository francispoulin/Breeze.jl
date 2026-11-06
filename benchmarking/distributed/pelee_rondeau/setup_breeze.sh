#!/bin/bash
# Setup of the distributed benchmark on the Waterloo fluids group's servers (pelee, rondeau), on top
# of the oceananigans-hpc platform. After `ocean_setup`, from anywhere; again after the platform
# moves to a new Julia, or after a pull that changes Manifest.toml:
#
#     bash benchmarking/distributed/pelee_rondeau/setup_breeze.sh
#
# It (1) brings Manifest.toml up to date: made afresh if another Julia made it, else resolved, which
# keeps every version that Project.toml (and Breeze's own compat) still allows; (2) installs the packages and points MPI at the system CUDA-aware OpenMPI with the
# platform's configure_mpi.sh, and (3) checks that CUDA and CUDA-aware MPI work.

: "${OCEAN_HPC_ROOT:?run ocean_setup first}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/breeze_env.sh" >/dev/null || exit 1
project="$BREEZE_DISTRIBUTED_DIR"

set -eo pipefail          # after the environment is loaded, as module commands can return harmless errors

echo "== 1. Manifest"
julia_version="$(julia --startup-file=no -e 'print(VERSION)')"
manifest="$project/Manifest.toml"
if [[ -f "$manifest" ]] && ! grep -q "^julia_version = \"$julia_version\"" "$manifest"; then
    echo "   made with another Julia ($(grep -m1 julia_version "$manifest")): making it afresh"
    rm "$manifest"
fi
JULIA_PKG_PRECOMPILE_AUTO=0 julia --project="$project" -e 'using Pkg; Pkg.resolve()'
grep -m1 julia_version "$manifest"

echo "== 2. Packages and system MPI (oceananigans-hpc's configure_mpi.sh)"
bash "$OCEAN_HPC_ROOT/setup/configure_mpi.sh" "$project"

echo "== 3. CUDA and CUDA-aware MPI"
julia --project="$project" -e '
    using MPI, CUDA, Breeze
    println("   CUDA functional: ", CUDA.functional(), "; MPI.has_cuda() (CUDA-aware MPI): ", MPI.has_cuda())
    CUDA.functional() && MPI.has_cuda() || exit(1)'

echo
[[ -z "$(git -C "$project" status --porcelain -- Manifest.toml)" ]] ||
    echo "Manifest.toml is new or changed: commit it, so the versions behind your benchmark numbers are recorded."
echo "Setup done. In a new terminal:"
echo "    ocean_setup && source $here/breeze_env.sh"
echo "    bash \$OCEAN_HPC_ROOT/tools/run_local.sh -n 2 -p \$BREEZE_DISTRIBUTED_DIR \$BREEZE_DISTRIBUTED_DIR/distributed_cbl_benchmark.jl --dynamics=compressible_explicit --size=256x256x128"
