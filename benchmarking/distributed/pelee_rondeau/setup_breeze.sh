#!/bin/bash
# One-time setup of the distributed benchmark on the Waterloo fluids group's servers (pelee,
# rondeau), on top of the oceananigans-hpc platform. After `ocean_setup`, from anywhere:
#
#     bash benchmarking/distributed/pelee_rondeau/setup_breeze.sh
#
# It (1) installs Julia $BREEZE_JULIA_VERSION next to the platform's Julia (Breeze needs >= 1.11.9),
# (2) installs the packages of benchmarking/distributed (Breeze comes from this checkout),
# (3) points MPI at the system CUDA-aware OpenMPI with the platform's configure_mpi.sh, and
# (4) precompiles and checks that CUDA and CUDA-aware MPI work.

: "${OCEAN_HPC_ROOT:?run ocean_setup first}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/breeze_env.sh" >/dev/null
project="$BREEZE_DISTRIBUTED_DIR"

set -eo pipefail          # after the environment is loaded, as module commands can return harmless errors

echo "== 1. Julia $BREEZE_JULIA_VERSION"
julia_dir="$BREEZE_SOFTWARE_DIR/julia-$BREEZE_JULIA_VERSION"
if [[ ! -x "$julia_dir/bin/julia" ]]; then
    tarball="julia-$BREEZE_JULIA_VERSION-linux-x86_64.tar.gz"
    series="${BREEZE_JULIA_VERSION%.*}"
    work="$(mktemp -d)"
    curl -fsSL -o "$work/$tarball" "https://julialang-s3.julialang.org/bin/linux/x64/$series/$tarball"
    curl -fsSL -o "$work/checksums" "https://julialang-s3.julialang.org/bin/checksums/julia-$BREEZE_JULIA_VERSION.sha256"
    (cd "$work" && grep " $tarball\$" checksums | sha256sum --check -)     # stops here if missing or wrong
    mkdir -p "$julia_dir"
    tar -xzf "$work/$tarball" -C "$julia_dir" --strip-components=1
    rm -rf "$work"
fi
julia --version

echo "== 2. Packages"
JULIA_PKG_PRECOMPILE_AUTO=0 julia --project="$project" -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'

echo "== 3. System MPI"
bash "$OCEAN_HPC_ROOT/setup/configure_mpi.sh" "$project"

echo "== 4. Precompile and check"
julia --project="$project" -e '
    using Pkg; Pkg.precompile()
    using MPI, CUDA, Breeze
    println("   Breeze, MPI and CUDA load; CUDA functional: ", CUDA.functional())
    println("   MPI.has_cuda() (CUDA-aware MPI): ", MPI.has_cuda())
'
grep -m1 julia_version "$project/Manifest.toml"

echo
echo "Setup done. In a new terminal:"
echo "    ocean_setup && source $here/breeze_env.sh"
echo "    bash \$OCEAN_HPC_ROOT/tools/run_local.sh -n 2 -p \$BREEZE_DISTRIBUTED_DIR \$BREEZE_DISTRIBUTED_DIR/distributed_cbl_benchmark.jl --dynamics=compressible_explicit --size=256x256x128"
