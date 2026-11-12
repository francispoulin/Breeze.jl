# One-GPU 512³ Nsight demo without NVTX, 2026-11-12

Status: **COMPLETE; capture validation PASS**. This reruns the
[512³ NVTX demo](../nvtx_demo_512_cubed_2026-11-10/README.md) with Breeze
instrumentation disabled (`--nvtx=false`), Nsight CUDA tracing only
(`--trace=cuda`), and `JULIA_NVTX_CALLBACKS` unset.

The scientific configuration is unchanged: split-explicit compressible dry CBL,
Float32, WENO5, a 512×512×512 grid, one MPI rank using the serial `GPU()` branch,
time step 0.02 s, nominal 12 acoustic substeps, deterministic initial perturbations,
10 warmup steps and 10 captured steps. The physical domain is 12×12×3 km.
Surface heating, geostrophic forcing and drag boundary conditions are retained.
The CUDA memory pool remains disabled and Julia uses one thread.

The exact Project, Manifest and MPI preferences were copied from the NVTX demo.
All 140 recorded source hashes match that demo. Julia is 1.12.7, Oceananigans
0.117.2, CUDA 6.4.2 and Nsight Systems 2024.5.1. NVTX remains a package dependency
of the copied environment; application ranges and NVTX collection are disabled.

Physical **GPU 2**, A100-SXM4-40GB, UUID
`0d23b51f-58fb-31df-e839-8dd85117df68`, was free at launch. The original demo used
GPU 0, which is occupied for this run. Other GPUs have active jobs, as recorded
in [GPU processes before launch](provenance/gpu_processes_before.csv).

The Nsight result path is:

`/u/ajtse/Breeze.jl/.runtime/nsight_no_nvtx_512_cubed_2026-11-12/raw/profile.nsys-rep`

The SQLite export path is:

`/u/ajtse/Breeze.jl/.runtime/nsight_no_nvtx_512_cubed_2026-11-12/raw/profile.sqlite`

## Results and validation

- **Zero NVTX events**; the recorded capture type is CUDA only.
- **10300 GPU kernels and 60 memory copies**. Every kernel's launch count
  matches the NVTX demo, including 220 tridiagonal-solver kernels.
- All prognostic and velocity fields are finite. The reported maximum absolute
  vertical velocity is `3.864660e-03`, matching the earlier demo at CSV precision.
  This is not a field-by-field comparison with that run.
- The Nsight capture duration is **10.572117 s**. The first-to-last GPU operation
  span is **10.531505 s**, averaging **1.053150 s/step** over the ten captured
  steps. This span includes GPU gaps. Summed kernel duration is **10.469129 s**.
- Report size is **1104468 bytes**. SHA-256, environment/source checks, event
  counts and timing definitions are retained in [verification.json](verification.json).
- GPU 2 has no remaining benchmark process after the run.

[Readable summary](summary.txt), [per-kernel totals](kernels.csv),
[original Nsight kernel summary](summary_cuda_gpu_kern_sum.csv),
[CUDA API summary](summary_cuda_api_sum.csv),
[memory-copy summary](summary_cuda_gpu_mem_time_sum.csv),
[exact invocation](command.sh), [application/profiler log](run.log),
[export log](export.log), [stats log](stats.log), and [provenance](provenance/).

After the finite-field `RESULT` line, the MPI launcher printed
`PMIX ERROR: NO-PERMISSIONS in file dstore_base.c at line 237`.
The launcher and complete run script exited with status 0; the report and SQLite
export completed, and all checks above passed. The message is retained in the log
and verification record; its cause was not diagnosed in this capture task.

All new environment, temporary, MPI, cache and raw report files are inside
`/u/ajtse/Breeze.jl/.runtime/nsight_no_nvtx_512_cubed_2026-11-12/`, which is already
ignored by Git. No new files are written to `/tmp` or `/local_scratch` by the
configured temporary/output paths. No plots are generated for this Nsight run.

## Reproduction and inspection

[run.sh](run.sh) invokes the existing tracked
`benchmarking/distributed/distributed_cbl_benchmark.jl` directly through Nsight
and the single-rank MPI launcher. It refuses to overwrite the report or result
CSV, and checks that GPU 2 has no compute processes before starting.
[environment.sh](environment.sh) records all environment changes without
sourcing the older activation script that creates a directory under `/tmp`.

The local environment contains copies of [Project.toml](provenance/Project.toml),
[Manifest.toml](provenance/Manifest.toml) and
[LocalPreferences.toml](provenance/LocalPreferences.toml). To reproduce, create
the environment directory under the runtime path above and copy those three
files there, then run `bash run.sh` from this directory. Use fresh raw/result
paths for another capture; the saved launch refuses overwrites.

Download the report from your laptop or workstation, then open it in Nsight
Systems 2024.5.1 or newer:

```bash
scp ajtse@rondeau.math.private.uwaterloo.ca:/u/ajtse/Breeze.jl/.runtime/nsight_no_nvtx_512_cubed_2026-11-12/raw/profile.nsys-rep ~/Downloads/breeze_512_cubed_no_nvtx.nsys-rep
```

The CUDA API and GPU stream rows contain the captured work. There are no Breeze
NVTX range labels. Profiling begins after warmup using the application's CUDA
profiler start/stop calls; the final GPU synchronization is included.

The driver's elapsed time includes Nsight report generation. Use capture/GPU
event timestamps to discuss execution timing. This is one profiled window on a
different physical GPU and a shared host; it does not establish NVTX overhead
or a performance improvement over the earlier demo.
