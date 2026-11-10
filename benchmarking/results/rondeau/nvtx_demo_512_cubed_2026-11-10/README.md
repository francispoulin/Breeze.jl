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

No plots are generated for this Nsight run.

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
