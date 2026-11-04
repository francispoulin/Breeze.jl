# Student guide: the Breeze multi-GPU benchmark on pelee and rondeau

This guide takes you from nothing to a scaling table for three atmospheric models (Breeze.jl), using
the group's GPU servers and the oceananigans-hpc platform. Read it once through, then work down it.

## 1. The big picture

Two pieces, each doing one job:

| Piece | Job | Where |
|-------|-----|-------|
| **oceananigans-hpc** (the platform) | Everything about the machine: modules, CUDA-aware MPI, the Julia package depot, `ocean_setup`, and `run_local.sh`, which starts runs on free GPUs | `.../software/oceananigans-hpc` |
| **This benchmark** (a branch of Breeze.jl) | The physics case and the timing: `distributed_cbl_benchmark.jl` builds the convective boundary layer (CBL) on `n` GPUs and reports how fast it steps | `.../software/Breeze.jl/benchmarking/distributed` |

You never configure MPI or CUDA by hand. The platform does that, and the benchmark uses it. The platform's
own guide to the servers is
[`docs/fluidsgpu.md`](https://github.com/francispoulin/oceananigans-hpc/blob/main/docs/fluidsgpu.md);
read it first if you have not used `run_local.sh` before.

**The three models** (the `dynamics` of Breeze's `AtmosphereModel`):

| `--dynamics` | What it is | Distributed over several GPUs? |
|--------------|-----------|--------------------------------|
| `anelastic` | Filters sound waves; needs a global pressure (Poisson) solve every step | **No, 1 GPU only** (work in progress in Breeze PR #1157) |
| `compressible_explicit` | Fully compressible; sound waves resolved with the main time step, so a small step | Yes |
| `compressible_splitexplicit` | Fully compressible; sound waves handled by short substeps inside each (larger) step | Yes |

The compressible models need only neighbour (halo) exchange between GPUs. The anelastic model needs a
global solve, which is the hard part to distribute. That difference is what this benchmark is about.

## 2. Before you start

- You need an account on pelee and rondeau. They share a drive, so you can work from either.
- Your own folder on the shared drive: `/mnt/autofs/bow.math/scratch/$USER`. The platform keeps its large Julia depot there.
- The platform cloned and set up once, in a folder named `software` inside your scratch folder
  (steps in `docs/fluidsgpu.md`; at the end `bash tools/check_env.sh` prints "All checks OK."):

  ```bash
  mkdir -p /mnt/autofs/bow.math/scratch/$USER/software && cd /mnt/autofs/bow.math/scratch/$USER/software
  git clone git@github.com:francispoulin/oceananigans-hpc.git
  ```

  and the alias in `~/.bashrc`:

  ```bash
  alias ocean_setup='source /mnt/autofs/bow.math/scratch/'$USER'/software/oceananigans-hpc/env/ocean_setup.sh'
  ```

- The Breeze branch, **next to** `oceananigans-hpc` (the scripts find the platform from there). Francis
  will tell you the exact remote and branch name:

  ```bash
  cd /mnt/autofs/bow.math/scratch/$USER/software
  git clone git@github.com:<owner>/Breeze.jl.git && cd Breeze.jl
  git switch fjp/distributed-benchmark
  ```

## 3. One-time setup of the benchmark

```bash
ocean_setup
bash benchmarking/distributed/pelee_rondeau/setup_breeze.sh
```

This takes a while (it compiles Breeze and its dependencies). What it does, so you know what to look for:

1. Installs **Julia 1.11.9** next to the platform's Julia 1.10.10. Breeze needs 1.11.9 or newer; the platform's Julia is older.
2. Installs the packages listed in `benchmarking/distributed/Project.toml`. Breeze comes from your checkout, so you test the code you have.
3. Points MPI at the system CUDA-aware OpenMPI (the platform's `configure_mpi.sh`).
4. Compiles and prints two lines you should check: `CUDA functional: true` and `MPI.has_cuda() ... true`.

Last, it prints the Manifest's Julia version, which should be 1.11.9.

## 4. Every new terminal

```bash
ocean_setup
source benchmarking/distributed/pelee_rondeau/breeze_env.sh      # puts Julia 1.11.9 first
```

The second line matters: without it you are on the platform's Julia 1.10.10, which cannot load this
environment. It also sets `$BREEZE_DISTRIBUTED_DIR` (this benchmark's folder), used below.

## 5. A first run: 1 GPU

```bash
bash $OCEAN_HPC_ROOT/tools/run_local.sh -n 1 -p $BREEZE_DISTRIBUTED_DIR \
    $BREEZE_DISTRIBUTED_DIR/distributed_cbl_benchmark.jl \
    --dynamics=compressible_explicit --size=256x256x128 --time_steps=300
```

Read the command in three parts: `run_local.sh -n 1 -p <project>` is the platform (1 GPU, which Julia
project); then the script; then the script's own options. `run_local.sh`:

- picks a GPU with no one else on it, and **refuses to start** if not enough are free (it tells you who is using them);
- gives each MPI rank only its own GPU, and runs in the background (it survives logging out);
- prints a **log file** and a **process ID**:

  ```bash
  tail -f <the log path it printed>      # follow the run (Ctrl-C only stops tail)
  kill <pid>                              # stop the run, all ranks
  ```

The first run spends a few minutes compiling GPU kernels before anything is timed. When it finishes, the
log ends with one line like:

```
RESULT dynamics=compressible_explicit float=Float32 size=256x256x128 partition=1x1 gpus=1 steps=300 seconds_per_step=... points_per_second=... points_per_second_per_gpu=... max_abs_w=...
```

- `points_per_second` uses the **global** number of grid points; higher is faster.
- `max_abs_w` is a blow-up check. If it is `NaN` or huge, the run went unstable and the timing means nothing.

## 6. Several GPUs

```bash
bash $OCEAN_HPC_ROOT/tools/run_local.sh -n 2 -p $BREEZE_DISTRIBUTED_DIR \
    $BREEZE_DISTRIBUTED_DIR/distributed_cbl_benchmark.jl \
    --dynamics=compressible_explicit --size=256x256x128 --partition=2x1
```

`--partition=RxxRy` says how many pieces to cut the domain into in x and y; `Rx * Ry` must equal `-n`. A
4-GPU run can be `--partition=2x2`, `4x1` or `1x4`. The size must divide evenly.

**Be considerate: pelee and rondeau are shared.** A 4-GPU run takes the whole machine. Check `nvidia-smi`
first, tell the group before long 4-GPU sweeps, and do not change `JULIA_NUM_THREADS` (the platform sets it
to 1 on purpose, because the machines have been short of CPUs).

## 7. The scaling sweep

```bash
bash benchmarking/distributed/pelee_rondeau/run_scaling.sh
```

It runs, one after the other, for each model: **strong scaling** (256×256×128 on 1, 2 and 4 GPUs) and
**weak scaling** (about 8M points per GPU: 256×256×128, 512×256×128, 512×512×128 on 1, 2, 4 GPUs). Anelastic
is run on 1 GPU only. Results are appended to a CSV, `$OCEAN_RESULTS_ROOT/Breeze/scaling.csv` by default.
Options: `--time_steps=1000`, `--max_gpus=2`, `--results_file=<file>`.

How to read the table (with `t(N)` the `seconds_per_step` on `N` GPUs):

- strong-scaling efficiency = `t(1) / (N * t(N))`: 1 is perfect, since the same problem should be N times faster;
- weak-scaling efficiency = `t(1) / t(N)`: 1 is perfect, since the time per step should not change.

## 8. Things to think about (and try)

1. **Time per step is not the whole story.** The benchmark's default step is `--time_step=0.05` s, small
   enough that all three models are stable. But the compressible-explicit model is limited by the sound
   speed, while the anelastic and split-explicit models can usually take much larger steps. What
   matters for a real simulation is the cost per **simulated** second. For each model, find the largest
   stable `--time_step` (watch `max_abs_w`), then compare the cost per simulated second.
2. **Partitions.** For 4 GPUs compare `2x2`, `4x1` and `1x4`. Which communicates the least? Does it matter on this machine?
3. **Size.** At about 1M points per GPU the GPUs are idle most of the time, and 2 GPUs can be slower than 1.
   At what size per GPU does scaling become good? Plot efficiency against points per GPU.
4. **Precision.** `--float_type=Float64` against `Float32`: how much slower, and is the answer visibly different?
5. **Physics options.** `--simplified=true` drops the geostrophic forcing and the surface drag. What does it cost?
6. **Substeps.** For `compressible_splitexplicit`, vary `--substeps` (default 12). Where does it stop being stable?

## 9. When something goes wrong

| Symptom | Likely cause and what to do |
|---------|-----------------------------|
| `not a git repository` | You are in the wrong folder. `cd` into the `Breeze.jl` folder for git commands. |
| `run_local.sh` refuses to start | Not enough free GPUs. `nvidia-smi` shows who is using them; use fewer or wait. |
| Nothing prints for several minutes | First-run compilation. Check `nvidia-smi` shows the GPU busy. After about 15 minutes, look at the log. |
| A multi-GPU run seems to hang after the timed steps | Rank 0 may have crashed while the others wait. Read the log from the top for the first `ERROR`, then `kill <pid>`. |
| `max_abs_w=NaN` | The run blew up. Reduce `--time_step`. |
| MPI errors at the start, or `MPI.has_cuda()` is `false` | MPI is not set up for this folder. Run `bash $OCEAN_HPC_ROOT/tools/check_env.sh $BREEZE_DISTRIBUTED_DIR`, then redo section 3. |
| `check_env.sh` warns that the Julia version differs | Expected here: this benchmark uses Julia 1.11.9, the platform's tests use 1.10.10. |
| Out of GPU memory | Smaller `--size`, or more GPUs. |
| `Resource temporarily unavailable` while compiling | Too many compile tasks. The platform sets 2; do not raise `JULIA_NUM_PRECOMPILE_TASKS`. |

When you ask Francis for help, send: the **command** you ran, the **log file** (or its last 40 lines), and the
output of `git log --oneline -1` in `Breeze.jl`.

## 10. Where things live

| What | Where |
|------|-------|
| This benchmark | `.../software/Breeze.jl/benchmarking/distributed/` (`README.md` is the generic version of this guide) |
| The platform | `.../software/oceananigans-hpc` (`$OCEAN_HPC_ROOT`) |
| Julia 1.10.10 and 1.11.9 | `.../software/julia-1.10.10`, `.../software/julia-1.11.9` |
| Julia packages (the depot) | `/mnt/autofs/bow.math/scratch/$USER/julia_depot` |
| Logs | `logs/` in the folder you ran `run_local.sh` from |
| Scaling results | `$OCEAN_RESULTS_ROOT/Breeze/scaling.csv` |
