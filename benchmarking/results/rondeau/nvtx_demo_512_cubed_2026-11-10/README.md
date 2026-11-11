# One-GPU NVTX demo at 512³, 2026-11-10

Status: **COMPLETE and validated**. All fields are finite; the capture contains
10 outer steps, 220 acoustic substeps and 10300 GPU kernels. All kernels are
attributed to Breeze NVTX ranges.

The existing tracked CBL benchmark runs split-explicit compressible dynamics on
one A100 with a 512×512×512 grid (134217728 cells), Float32, WENO5, full dry CBL
forcing and boundary conditions, and deterministic initial perturbations.
The simulation time step is 0.02 s. Ten warmup steps precede ten captured steps;
the requested 12 acoustic substeps are distributed as 4 + 6 + 12 per RK step,
giving 22 acoustic substeps per outer step.

This changes the grid from the earlier 128×128×64 demo while retaining its
dynamics, time step, warmup/capture counts, precision and initial-condition choice.
The earlier capture at `/local_scratch/ajtse/breeze_nvtx_demo.nsys-rep` is preserved.
This is the benchmark's serial `GPU()` branch; halo fills operate locally.

The Nsight result file is:

`/local_scratch/ajtse/breeze_nvtx_demo_512_cubed_2026-11-10/profiles/compressible_splitexplicit_SERIAL_1g/rank_0/profile.nsys-rep`

Report size: 1070118 bytes (1.02 MiB). Its SQLite export
is `/local_scratch/ajtse/breeze_nvtx_demo_512_cubed_2026-11-10/profiles/compressible_splitexplicit_SERIAL_1g/rank_0/profile.sqlite`.

[Exact launch](commands.log), [capture state](captures.json),
[resolved environment and source hashes](provenance/).

CUDA and NVTX traces cover only the steps after warmup and final GPU
synchronization. Model setup, JIT/warmup and field checks are outside the capture.
The driver's elapsed time includes Nsight report generation; execution timing
must be read from the capture. NVTX host ranges measure CPU submission/waiting;
GPU projections include the interval between their first and last associated GPU
operation. Parent and child ranges overlap and must not be added together.

The timeline below was added on 2026-11-11 at the user's request, using this
existing capture. The simulation and profiler were not rerun.

## Timeline like the NVTX.jl demo

The image in the [NVTX.jl README](https://github.com/JuliaGPU/NVTX.jl) shows the
NVIDIA Nsight Systems GUI. Our original `.nsys-rep` already contains the NVTX
ranges and CUDA events needed for that view.

- [Interactive timeline, standalone HTML](timeline.html): open in a browser;
  scroll to zoom, drag to pan, hover for names and durations.
- [PNG overview and first-step detail](timeline.png), [vector SVG](timeline.svg).
- [Projection counts, timing totals and input hash](timeline.json).

These exports distinguish host NVTX ranges, GPU-projected NVTX ranges and actual
CUDA kernel/copy execution. A level is the range's nesting depth on its host
thread. The projection spans its first to last associated GPU operation,
including gaps; correlation follows the CUDA launch API. Outer-step numbers
identify the same invocation on the host and GPU projection rows. The CPU may
submit step 2 while the GPU still runs step 1. Parent and child bars overlap.
The one-GPU capture has no inter-rank communication or MPI API trace rows.

Every Breeze range's projected total duration, instance count and associated GPU
operation count was checked against the saved official
`summary_nvtx_gpu_proj_sum.csv`; all match exactly. The export includes all
10300 kernels, 60 copies and 3461 completed host ranges. Regression tests cover
asynchronous launches, nesting, copies, thread isolation and omitted empty ranges.

To use the original Nsight GUI from a desktop session on Rondeau:

```bash
nsys-ui /local_scratch/ajtse/breeze_nvtx_demo_512_cubed_2026-11-10/profiles/compressible_splitexplicit_SERIAL_1g/rank_0/profile.nsys-rep
```

For a laptop or workstation, run this **on that machine**, then open the downloaded
report in Nsight Systems using **File > Open**:

```bash
scp ajtse@rondeau:/local_scratch/ajtse/breeze_nvtx_demo_512_cubed_2026-11-10/profiles/compressible_splitexplicit_SERIAL_1g/rank_0/profile.nsys-rep ~/Downloads/breeze_512_cubed.nsys-rep
```

Use Nsight Systems **2024.5.1 or newer** to match the capture's CLI version.
Expand the Julia process/thread's **NVTX** rows, the **Breeze** domain and the
GPU's **CUDA** stream rows. Zoom into one `time_step!`, then into its acoustic
substeps or `solve!` calls. See the
[NVIDIA GUI instructions](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#reading-your-report-in-gui).

Regenerate the offline exports from the repository root with Python 3 and
Matplotlib installed:

```bash
python3 -B benchmarking/distributed/render_nvtx_timeline.py \
  /local_scratch/ajtse/breeze_nvtx_demo_512_cubed_2026-11-10/profiles/compressible_splitexplicit_SERIAL_1g/rank_0/profile.sqlite \
  --output-prefix benchmarking/results/rondeau/nvtx_demo_512_cubed_2026-11-10/timeline \
  --title 'Breeze 512³ · split-explicit CBL · one A100'

python3 -B benchmarking/distributed/test_nvtx_timeline.py
```

## Read the results

The captured execution window is **10.5174 s total**, or
**1.0517 s/outer step**. This is a profiled measurement.

- [Readable text summary, with times in milliseconds](summary.txt)
- [NVTX ranges in milliseconds, CSV](nvtx_ranges_ms.csv)
- [Original Nsight projection summary, nanoseconds](summary_nvtx_gpu_proj_sum.csv)
- [Per-kernel execution totals](kernels.csv) and [exclusive phase totals](phases.csv)
- [Trace-window metrics](rank_summary.csv), [full analysis](analysis.json), [verification](verification.json)

NVTX projection totals include child ranges and intervals between GPU operations.
Kernel execution totals sum each kernel instance separately. Neither reports SM
utilization. The one-GPU halo ranges describe local periodic/boundary work.

Open the readable summary on Rondeau:

```bash
less -S /u/ajtse/Breeze.jl/benchmarking/results/rondeau/nvtx_demo_512_cubed_2026-11-10/summary.txt
```

The [launch log](commands.log) records the exact command. It uses the retained
Julia 1.12.7 profiling environment from the preceding NVTX study.
