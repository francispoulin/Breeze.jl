export PATH="/opt/uw/openmpi/4.1.7-with-cuda/bin:$PATH"
export LD_LIBRARY_PATH="/opt/uw/openmpi/4.1.7-with-cuda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export JULIA_DEPOT_PATH=/u/ajtse/.julia JULIA_LOAD_PATH=@:@stdlib
export JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_NUM_THREADS=1 JULIA_NUM_GC_THREADS=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_CUDA_MEMORY_POOL=none
export TMPDIR=/u/ajtse/Breeze.jl/.runtime/nvtx_cse_2048x512x512_4gpu_2026-11-12/tmp
export TMP="$TMPDIR" TEMP="$TMPDIR" OMPI_MCA_orte_tmpdir_base="$TMPDIR/mpi"
export CUDA_CACHE_PATH="$TMPDIR/cuda_cache" XDG_CACHE_HOME="$TMPDIR/cache"
mkdir -p "$OMPI_MCA_orte_tmpdir_base" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"
export JULIA_NVTX_CALLBACKS=gc NSYS_MPI_STORE_TEAMS_PER_RANK=1 CUDA_VISIBLE_DEVICES=0,1,2,3
