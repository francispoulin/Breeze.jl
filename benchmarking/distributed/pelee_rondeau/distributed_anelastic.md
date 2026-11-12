# Anelastic on several GPUs (pelee, rondeau)

The branch `claude/distributed-anelastic` of `francispoulin/Breeze.jl` runs Breeze's anelastic model on
several GPUs. It is the benchmark branch plus Breeze PR
[#1157](https://github.com/NumericalEarth/Breeze.jl/pull/1157) (distributed anelastic pressure solver),
which is not in Breeze `main` yet. When #1157 is merged, this moves back to `fjp/distributed-benchmark`.

Checked on Nov 11, 2026: 1, 2 and 4 GPUs give the same `max |w|` to 7 digits after 1000 steps.

## Switch your clone to the branch

Use the clone from `student_guide.md` (`.../scratch/$USER/software/Breeze.jl`; `origin` is
`francispoulin/Breeze.jl`):

```bash
ocean_setup
cd /mnt/autofs/bow.math/scratch/$USER/software/Breeze.jl
git status -s                                       # should print nothing; commit or stash your changes first
git fetch origin
git switch claude/distributed-anelastic             # creates the local branch, tracking origin
bash benchmarking/distributed/pelee_rondeau/setup_breeze.sh
```

Later updates: `git pull`, then `setup_breeze.sh` again if `Manifest.toml` changed. To go back:
`git switch fjp/distributed-benchmark`.

## Run

As in `student_guide.md`, from your runs folder:

```bash
ocean_setup
source /mnt/autofs/bow.math/scratch/$USER/software/Breeze.jl/benchmarking/distributed/pelee_rondeau/breeze_env.sh
cd /mnt/autofs/bow.math/scratch/$USER/software/breeze_runs

# Check: same max_abs_w on 1 and 2 GPUs (the smooth initial perturbation is the same on every rank)
for p in "1 1x1" "2 2x1"; do read -r n part <<<"$p"
  bash $OCEAN_HPC_ROOT/tools/run_local.sh --gpus $n --project $BREEZE_DISTRIBUTED_DIR --wait -j check_$part \
    $BREEZE_DISTRIBUTED_DIR/distributed_cbl_benchmark.jl --dynamics=anelastic --partition=$part \
    --float_type=Float64 --random_perturbation=false --time_steps=100
done
grep -h RESULT $(ls -t logs/check_*.out | head -2)
```

Expected: `max_abs_w=1.646577e-03` on both lines. Then use any of the driver's options (`README.md` in
`benchmarking/distributed/`), for example `--gpus 4 ... --partition=4x1 --size=1024x256x128`.

## Rules for the partition

- `--partition=RxxRy` with `Rx × Ry` = number of GPUs. No split in z.
- `Nx` and `Ny` must both be divisible by the number of GPUs (the solver transposes the data).
- `Rx×1` uses the fewest transposes; `2x2` works but is the slowest.
- `--random_perturbation=false` whenever you compare runs on different partitions: the default `rand()`
  noise differs from rank to rank.

## In your own script

Nothing anelastic-specific: build the model on a distributed architecture, and the pressure solver
becomes the distributed one.

```julia
using MPI; MPI.Init()
using Oceananigans, Breeze
using Oceananigans.DistributedComputations: Distributed, Partition

arch = Distributed(GPU(); partition = Partition(4, 1))
grid = RectilinearGrid(arch; size = (1024, 256, 128), ...)    # global size
model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state), ...)
```

Start it with `run_local.sh --gpus 4 --project $BREEZE_DISTRIBUTED_DIR your_script.jl`.

If something fails, send Francis the end of the log and `git log --oneline -1`.
