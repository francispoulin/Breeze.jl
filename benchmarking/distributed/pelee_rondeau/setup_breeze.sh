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
# A Manifest made by another Julia series (e.g. 1.10) cannot be resolved by 1.11: start afresh.
manifest="$project/Manifest.toml"
if [[ -f "$manifest" ]] && ! grep -q "^julia_version = \"${BREEZE_JULIA_VERSION%.*}\." "$manifest"; then
    echo "   removing a Manifest made by another Julia series"
    rm "$manifest"
fi
JULIA_PKG_PRECOMPILE_AUTO=0 julia --project="$project" -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'

echo "== 3. System MPI"
# configure_mpi.sh sources ocean_setup.sh, which puts the platform's Julia 1.10.10 first on PATH,
# and its package step then fails on a Manifest made by Julia 1.11 (StyledStrings is a stdlib only
# in 1.11). The environment is already loaded here, with the right Julia first, so run the script
# with that one line removed. Stop if the platform's script no longer has exactly one such line.
configure_mpi="$OCEAN_HPC_ROOT/setup/configure_mpi.sh"
if [[ "$(grep -c 'ocean_setup.sh"$' "$configure_mpi")" -ne 1 ]]; then
    echo "ERROR: expected exactly one ocean_setup.sh line in $configure_mpi; has the platform changed?" >&2
    exit 1
fi
bash <(sed '/ocean_setup.sh"$/d' "$configure_mpi") "$project"

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
