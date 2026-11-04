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
  git clone git@github.com:francispoulin/Breeze.jl.git && cd Breeze.jl
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

Source `breeze_env.sh` **last**: it overrides other setups such as `ocean_setup_13`. Do not run `ocean_setup`
again after it in the same terminal, or you are back on Julia 1.10.10.

### Julia: which version, and how it gets installed

- **Which.** This benchmark uses **Julia 1.11.9**. Breeze's `Project.toml` requires 1.11.9 or newer,
  and the platform's tests and its own Julia stay on 1.10.10. The two live side by side, and
  `breeze_env.sh` chooses 1.11.9 for you. Do not switch Julia versions on your own: the shared depot keeps
  compiled files per Julia version, so a switch means a long recompile, and the platform moves to a newer
  Julia only together with a planned Oceananigans upgrade.
- **How it is installed.** You do nothing: `setup_breeze.sh` (section 3) downloads the official Linux
  tarball from julialang.org, checks its SHA-256 against the checksum julialang.org publishes, and
  unpacks it into `.../software/julia-1.11.9`. All projects share that one install, so it happens once.
- **Check it.** After `source breeze_env.sh`, `julia --version` should print 1.11.9 and `which julia`
  should point into `.../software/julia-1.11.9/bin`.
- **If the download fails** (for example no internet access), do it by hand. On a machine that can
  reach julialang.org, download `julia-1.11.9-linux-x86_64.tar.gz` from
  `https://julialang-s3.julialang.org/bin/linux/x64/1.11/` and the checksum file
  `https://julialang-s3.julialang.org/bin/checksums/julia-1.11.9.sha256`, copy both into the
  `software` folder, and then:

  ```bash
  cd /mnt/autofs/bow.math/scratch/$USER/software
  grep " julia-1.11.9-linux-x86_64.tar.gz$" julia-1.11.9.sha256 | sha256sum --check -    # must say OK
  mkdir -p julia-1.11.9 && tar -xzf julia-1.11.9-linux-x86_64.tar.gz -C julia-1.11.9 --strip-components=1
  ```

  Then run `setup_breeze.sh` again; it sees the install and skips the download.
- **Not the module.** The `julia/1.10.8` module on the servers is too old, and `ocean_setup` already
  gives you the platform's 1.10.10. Use only the 1.11.9 from `breeze_env.sh` for this benchmark.

## 5. A first run: 1 GPU

`run_local.sh` writes its logs to `logs/` in the folder you start it from. Start from a scratch folder
**outside** the repository, so logs do not clutter `git status`:

```bash
mkdir -p ../breeze_runs && cd ../breeze_runs      # from inside Breeze.jl; once
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
RESULT machine=pelee dynamics=compressible_explicit float=Float32 size=256x256x128 partition=1x1 gpus=1 steps=300 seconds_per_step=... points_per_second=... points_per_second_per_gpu=... max_abs_w=...
```

- `points_per_second` uses the **global** number of grid points; higher is faster.
- `max_abs_w` is a blow-up check. If it is `NaN` or huge, the run went unstable and the timing means nothing.

### What you should roughly see

Measured on pelee (A100-40GB), Float32, 256x256x128, `--time_step=0.05`, 100 timed steps:

| Dynamics | GPUs | ms per step | points per second |
|----------|------|-------------|-------------------|
| `anelastic` | 1 | 22.8 | 3.67e8 |
| `compressible_explicit` | 1 | 22.8 | 3.67e8 |
| `compressible_splitexplicit` | 1 | 82.4 | 1.02e8 |
| `compressible_splitexplicit` | 2 (2x1) | 69.3 | 1.21e8 (6.05e7 per GPU) |

`max_abs_w` is about 3e-2 in all of them. Things to keep in mind:

- Cost **per step** is not cost per **simulated second**. Split-explicit does extra substeps each step but can
  take a larger `--time_step`, so compare the models at the largest stable step of each.
- Two GPUs gave a 1.19x speedup (59% efficiency) on this grid: halo exchange costs a lot on a grid this small.
- Compare numbers from the same machine only, and repeat runs: a single 100-step run is noisy.

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
| `check_env.sh` warns that the Julia version differs | Expected here: this benchmark uses Julia 1.11.9, the platform's tests use 1.10.10. Ignore its advice to run `install_julia.sh`. |
| CUDA warnings that a library is "loaded from a system path" | Harmless so far; the runs are correct. |
| `Unsatisfiable requirements ... StyledStrings` | A Manifest made by another Julia is in the folder. `setup_breeze.sh` now removes it; rerun section 3. |
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
