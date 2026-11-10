# Breeze NVTX optimization study, 2026-11-10

Complete: **7/7 cases, 19 rank traces, 533000 captured kernels**. All kernels are attributed
to Breeze ranges. All cases have finite fields; the annotation checks passed 81 assertions.

## Optimization targets from serial 512³ captures

1. **Reduce work in the acoustic pipeline.** Split-explicit spends 65.8% of summed
   kernel duration inside `acoustic_substep!`. Post-solve recovery is 14.9%, the
   batched tridiagonal solve 14.3%, predictors 10.0%, the horizontal step 8.1%,
   divergence damping 8.0%, and vertical RHS construction 7.1%. Each of those six
   kernels runs 220 times in ten outer steps. Focus on repeated array traffic and
   stage-invariant solver work.

   In [`_post_solve_recovery!`](../../../../src/CompressibleEquations/acoustic_substepping.jl),
   three `avg` arrays are read and written every substep, even for this dry case.
   That is approximately 70.9 GB of nominal Float32 array traffic per outer step
   at 512³ × 22 substeps, before caching. An explicit dry/no-tracer specialization
   could omit accumulation, finalization and related halo fills when their consumers
   are absent. The whole 14.9% kernel still performs essential density/θ recovery,
   so this is an optimization budget, not a predicted 14.9% speedup.

   **TODO for Oceananigans:** support factoring the acoustic tridiagonal matrix once per RK
   stage, then reusing it for successive RHS solves. Breeze freezes θᴸ and Cᴸ per
   stage and uses a constant Δτ within that stage, while the current coefficient
   tags evaluate entries in each solve. A factorization/repeated-RHS API would
   belong in Oceananigans; Breeze would own invalidation on changes to stage state,
   Δτ, damping, grid and sponge. Extra factor storage and numerical behavior must
   be measured before treating this as an improvement.

2. **Specialize known dry configurations.** The moisture WENO tendency consumes
   12.2% of explicit-compressible kernel duration and 11.0% of anelastic kernel
   duration although this case supplies no water or water sources. It is the only
   `compute_scalar_tendency!` field in this case. An explicit dry invariant could
   eliminate that tendency and its RK update/halo work, and could permit aliasing
   dry and total density. `microphysics=nothing` alone does not imply zero moisture:
   vapor and tracers can still be transported without a microphysics scheme.
   See [`compute_tendencies!`](../../../../src/AtmosphereModels/update_atmosphere_model_state.jl)
   and [`compute_total_density!`](../../../../src/CompressibleEquations/compressible_time_stepping.jl).

3. **Avoid duplicate thermodynamic inversion.** The explicit-compressible auxiliary
   thermodynamics kernel takes 4.8% and the following coupled temperature/pressure
   kernel another 4.2%. The first diagnoses T; the second computes T again before
   pressure. Reusing the diagnosed temperature, or a carefully fused dry path,
   could reduce inversion and memory work. Moist saturation adjustment and energy
   conservation require separate validation before applying the same change to moist
   runs. Their combined 9.0% kernel share is the affected budget, not an expected gain.
   See [`compute_auxiliary_variables!`](../../../../src/AtmosphereModels/update_atmosphere_model_state.jl)
   and [`compute_auxiliary_dynamics_variables!`](../../../../src/CompressibleEquations/compressible_time_stepping.jl).

4. **Profile WENO kernels in more detail.** Momentum, potential-temperature and
   moisture tendencies together take 63.5% of explicit-compressible and 57.5% of
   anelastic kernel time. These are the first targets for Nsight Compute measurements
   of memory throughput, instruction cost and register pressure. NVTX/Nsight Systems
   identifies their time budget but does not establish a bandwidth or occupancy limit.
   The anelastic pressure solve, its source construction and momentum correction
   together account for 20.7% of kernel duration in this setup.

The serial GPU timeline is 99.9% active for all three dynamics. This is consistent
with prioritizing GPU work at this resolution. Host halo ranges include waiting
for preceding kernels; their durations are not isolated network costs. Halo-launched
GPU kernels take 5.6% of explicit and 6.0% of split-explicit kernel duration.

## Distributed communication

Split-explicit MPI shows **25.2–26.1% gaps** in the recorded GPU activity timeline.
Its NCCL counterpart is 98.9–99.3% active, with 12.2–12.8% of summed kernel duration
in NCCL send/receive kernels. Reducing or overlapping repeated halo work is a
substantial candidate for the distributed split-explicit path under this workload.

The four-GPU explicit MPI capture has 86.4–86.7% timeline activity across ranks.
NCCL has 99.4%, with send/receive kernels accounting for 10.4–14.0% of summed
kernel duration. NCCL kernels include waiting for peers and data, so timeline
activity alone does not measure arithmetic utilization. These observations support
examining the halo schedule and opportunities to overlap communication with useful
work. They do not isolate network bandwidth or establish an unprofiled speedup.

For explicit dynamics, annotated halo-fill calls occur 39 times per outer step;
split-explicit issues 225 per step in the serial capture, including the repeated
acoustic work. Inspect redundant fills and opportunities to batch independent
fields while preserving stencil and boundary-condition dependencies. The shared
communication implementation belongs in Oceananigans; Breeze controls when it is
called. GPU halo work and host halo durations describe different costs.

The study uses one Julia thread per rank. In this resolved Oceananigans 0.117.2,
the MPI progress worker requires a thread pool with more than one thread and
`MPI.THREAD_MULTIPLE`. Comparing two threads per rank with the same unprofiled
workload is a useful configuration experiment before adding new communication
code. A gain from that change has not been measured here.

MPI traces also contain millions of `cuEventQuery` calls. Their host durations
overlap device execution and must not be interpreted as additive wall-time costs.
The analyzer aggregates these calls in SQLite and only loads kernel launch events
for NVTX attribution, keeping analysis memory bounded by the useful event set.

Each result is one short capture without repeated trials. CUDA/NVTX tracing changes
execution cost, especially in API-heavy communication paths. Unprofiled runs with
repeats are required to evaluate gains from any proposed change.

## Measurement protocol

This study uses [NVTX.jl](https://github.com/JuliaGPU/NVTX.jl) to connect CUDA launches
to Breeze host methods, including nested halo-fill and solver calls. Instrumentation
is implemented in `benchmarking/distributed/nvtx_instrumentation.jl` and enabled by
the tracked CBL benchmark's `--nvtx=true` option. It re-evaluates selected host
methods from the checkout without modifying their scientific expressions or adding
GPU synchronizations. Kernels remain unchanged.

The dry CBL uses Float32, WENO5, surface heating, geostrophic forcing and surface drag,
without turbulence closure, microphysics or radiation. Serial anelastic, explicit
compressible and split-explicit compressible runs use 512³ cells. Both compressible
methods also run on four A100s with MPI and NCCL, a 2×2 partition and global
1024×1024×512 cells (512³ per GPU). Initial perturbations are deterministic.
Explicit uses Δt=0.005 s; anelastic and split-explicit use Δt=0.02 s.
Split-explicit requests 12 substeps with the default proportional distribution:
4, 6 and 12 acoustic substeps across the three RK stages, 22 per outer step.

Each capture excludes model setup and 10 warmup steps, then records 10 steps.
Only CUDA and NVTX are traced; MPI API tracing, CPU sampling and context-switch
tracing are disabled. MPI/NCCL halo costs remain visible through annotated host
calls and the associated CUDA activity, with less tracer work in the communication path.
These are profiling measurements, not controlled unprofiled benchmark comparisons.
The driver's `profiled_measurements.csv` includes time spent stopping the profiler
and generating reports. For example, explicit serial records 0.736 s/step in the
driver while its captured execution window is about 0.272 s/step. Use
`rank_summary.csv` for trace-window durations; retain the driver CSV as raw logging.
Kernel fractions refer to summed GPU kernel durations, not wall time or utilization.
Kernel activity on the timeline is measured as the union of kernel and memcpy
intervals; it is not hardware occupancy or an SM utilization counter.

`analyze_nvtx.py` follows CUDA correlation IDs from launch API calls inside a
host range to the resulting GPU kernels, even when the GPU executes after the
host range ends. Each kernel belongs exclusively to the innermost annotated
Breeze function; inclusive acoustic-substep totals and host range times are
reported separately. Inclusive ranges overlap and must not be added together.

The Julia environment is isolated at
`/local_scratch/ajtse/breeze_nvtx_2026-11-10/environment` because the checked-in
benchmark manifest was generated with a different Julia version. The root package
dependencies and checked-in manifest are unchanged. The resolved environment,
source hashes, exact commands and capture checks are retained beside this file.

No plots are generated for this Nsight study.

## Trace-window measurements

Ranges below span ranks, not repeated trials. The time-step sizes differ between
explicit and split-explicit; these per-step times do not compare simulated-time efficiency.

| Dynamics | Backend | GPUs | Δt (s) | Trace s/step | GPU timeline active | Halo share of kernel time |
|---|---|---:|---:|---:|---:|---:|
| anelastic | SERIAL | 1 | 0.02 | 0.2969 | 99.9% | 4.4% |
| compressible_explicit | SERIAL | 1 | 0.005 | 0.2723 | 99.9% | 5.6% |
| compressible_splitexplicit | SERIAL | 1 | 0.02 | 1.0523 | 99.9% | 6.0% |
| compressible_explicit | MPI | 4 | 0.005 | 0.3393–0.3397 | 86.4–86.7% | 6.1% |
| compressible_explicit | NCCL | 4 | 0.005 | 0.3075–0.3197 | 99.4% | 16.2–19.6% |
| compressible_splitexplicit | MPI | 4 | 0.02 | 1.5412–1.5433 | 73.9–74.8% | 7.6–7.7% |
| compressible_splitexplicit | NCCL | 4 | 0.02 | 1.2289–1.2305 | 98.9–99.3% | 18.2–18.9% |

Machine-readable results: [rank summary](rank_summary.csv), [kernel totals](kernels.csv),
[exclusive phase totals](phases.csv), [full analysis](analysis.json),
[capture manifest with SHA-256 hashes](captures.json), [verification](verification.json).
The `calls` field inside phase/kernel JSON metrics counts kernel launches; `host_ranges.calls`
counts NVTX range invocations.

## Exact capture paths on Rondeau

Every primary capture is validated. Each SQLite export is beside its report as `profile.sqlite`.

| Case | Rank | Nsight result file | MiB |
|---|---:|---|---:|
| anelastic_SERIAL_1g | 0 | `/local_scratch/ajtse/breeze_nvtx_2026-11-10/profiles/anelastic_SERIAL_1g/rank_0/profile.nsys-rep` | 0.53 |
| compressible_explicit_SERIAL_1g | 0 | `/local_scratch/ajtse/breeze_nvtx_2026-11-10/profiles/compressible_explicit_SERIAL_1g/rank_0/profile.nsys-rep` | 0.51 |
| compressible_splitexplicit_SERIAL_1g | 0 | `/local_scratch/ajtse/breeze_nvtx_2026-11-10/profiles/compressible_splitexplicit_SERIAL_1g/rank_0/profile.nsys-rep` | 1.02 |
| compressible_explicit_MPI_4g | 0 | `/local_scratch/ajtse/breeze_nvtx_2026-11-10/profiles_distributed/compressible_explicit_MPI_4g/rank_0/profile.nsys-rep` | 83.38 |
| compressible_explicit_MPI_4g | 1 | `/local_scratch/ajtse/breeze_nvtx_2026-11-10/profiles_distributed/compressible_explicit_MPI_4g/rank_1/profile.nsys-rep` | 80.78 |
| compressible_explicit_MPI_4g | 2 | `/local_scratch/ajtse/breeze_nvtx_2026-11-10/profiles_distributed/compressible_explicit_MPI_4g/rank_2/profile.nsys-rep` | 84.28 |
| compressible_explicit_MPI_4g | 3 | `/local_scratch/ajtse/breeze_nvtx_2026-11-10/profiles_distributed/compressible_explicit_MPI_4g/rank_3/profile.nsys-rep` | 81.30 |
| compressible_explicit_NCCL_4g | 0 | `/local_scratch/ajtse/breeze_nvtx_2026-11-10/profiles_distributed/compressible_explicit_NCCL_4g/rank_0/profile.nsys-rep` | 2.36 |
| compressible_explicit_NCCL_4g | 1 | `/local_scratch/ajtse/breeze_nvtx_2026-11-10/profiles_distributed/compressible_explicit_NCCL_4g/rank_1/profile.nsys-rep` | 2.35 |
| compressible_explicit_NCCL_4g | 2 | `/local_scratch/ajtse/breeze_nvtx_2026-11-10/profiles_distributed/compressible_explicit_NCCL_4g/rank_2/profile.nsys-rep` | 2.36 |
| compressible_explicit_NCCL_4g | 3 | `/local_scratch/ajtse/breeze_nvtx_2026-11-10/profiles_distributed/compressible_explicit_NCCL_4g/rank_3/profile.nsys-rep` | 2.35 |
| compressible_splitexplicit_MPI_4g | 0 | `/local_scratch/ajtse/breeze_nvtx_2026-11-10/profiles_distributed/compressible_splitexplicit_MPI_4g/rank_0/profile.nsys-rep` | 335.36 |
| compressible_splitexplicit_MPI_4g | 1 | `/local_scratch/ajtse/breeze_nvtx_2026-11-10/profiles_distributed/compressible_splitexplicit_MPI_4g/rank_1/profile.nsys-rep` | 335.25 |
| compressible_splitexplicit_MPI_4g | 2 | `/local_scratch/ajtse/breeze_nvtx_2026-11-10/profiles_distributed/compressible_splitexplicit_MPI_4g/rank_2/profile.nsys-rep` | 343.80 |
| compressible_splitexplicit_MPI_4g | 3 | `/local_scratch/ajtse/breeze_nvtx_2026-11-10/profiles_distributed/compressible_splitexplicit_MPI_4g/rank_3/profile.nsys-rep` | 332.46 |
| compressible_splitexplicit_NCCL_4g | 0 | `/local_scratch/ajtse/breeze_nvtx_2026-11-10/profiles_distributed/compressible_splitexplicit_NCCL_4g/rank_0/profile.nsys-rep` | 9.84 |
| compressible_splitexplicit_NCCL_4g | 1 | `/local_scratch/ajtse/breeze_nvtx_2026-11-10/profiles_distributed/compressible_splitexplicit_NCCL_4g/rank_1/profile.nsys-rep` | 9.81 |
| compressible_splitexplicit_NCCL_4g | 2 | `/local_scratch/ajtse/breeze_nvtx_2026-11-10/profiles_distributed/compressible_splitexplicit_NCCL_4g/rank_2/profile.nsys-rep` | 9.82 |
| compressible_splitexplicit_NCCL_4g | 3 | `/local_scratch/ajtse/breeze_nvtx_2026-11-10/profiles_distributed/compressible_splitexplicit_NCCL_4g/rank_3/profile.nsys-rep` | 9.81 |

The separate 64³/two-step pilot is `/local_scratch/ajtse/breeze_nvtx_2026-11-10/pilot/profile.nsys-rep` (0.28 MiB).
Its [analysis](pilot_analysis.json) and the official `nvtx_gpu_proj_sum` CSV are retained
(the official CSV is `/local_scratch/ajtse/breeze_nvtx_2026-11-10/pilot/summary_nvtx_gpu_proj_sum.csv`).

The initial MPI attempt below is excluded: it finished with finite fields but its controller
used an incorrect GPU-ordinal mapping. [Validation notes](validation_notes.md) explain
the corrected direct-UUID check and recollection. The initial controller state and raw
timing rows are preserved in [initial_collection.json](initial_collection.json) and
[initial_profiled_measurements.csv](initial_profiled_measurements.csv).

| Excluded attempt | Rank | Nsight result file | MiB |
|---|---:|---|---:|
| initial masked MPI explicit | 0 | `/local_scratch/ajtse/breeze_nvtx_2026-11-10/profiles/compressible_explicit_MPI_4g/rank_0/profile.nsys-rep` | 86.41 |
| initial masked MPI explicit | 1 | `/local_scratch/ajtse/breeze_nvtx_2026-11-10/profiles/compressible_explicit_MPI_4g/rank_1/profile.nsys-rep` | 86.67 |
| initial masked MPI explicit | 2 | `/local_scratch/ajtse/breeze_nvtx_2026-11-10/profiles/compressible_explicit_MPI_4g/rank_2/profile.nsys-rep` | 82.90 |
| initial masked MPI explicit | 3 | `/local_scratch/ajtse/breeze_nvtx_2026-11-10/profiles/compressible_explicit_MPI_4g/rank_3/profile.nsys-rep` | 82.84 |

## Reproduction and validation

The scientific source hashes for all 136 library/case files match between collections
and the checkout. NVTX instrumentation is identical. The benchmark driver gained runtime
GPU UUID logging between serial and distributed collection; this logging occurs before
construction/warmup. Distributed ranks see all four devices and Oceananigans assigns
one to each rank.

Resolved environments and source hashes: [serial provenance](provenance/),
[distributed provenance](distributed/provenance/). Exact launches: [initial commands](commands.log)
(the last MPI attempt is excluded), [validated distributed commands](distributed/commands.log).
[Annotation validation log](validation.log), [validation notes](validation_notes.md).

Checks establish complete captures, correct rank/device assignment, finite fields,
and unchanged serial GPU results under annotations. Distributed fields were not compared
with a serial solution. Microphysics, radiation and turbulence-closure costs are outside
this early-time dry CBL study.

To collect a fresh suite from the repository root using the retained environment:

```bash
source benchmarking/results/rondeau/nvtx_optimization_2026-11-10/environment.sh
python3 -u benchmarking/distributed/run_nvtx_profiles.py \
    --project=/local_scratch/ajtse/breeze_nvtx_2026-11-10/environment \
    --julia=/u/ajtse/.local/opt/julia-1.12.7/bin/julia \
    --scratch=/local_scratch/ajtse/breeze_nvtx_repeat/profiles \
    --results=benchmarking/results/rondeau/nvtx_optimization_repeat
python3 -u -B benchmarking/distributed/analyze_nvtx.py \
    benchmarking/results/rondeau/nvtx_optimization_repeat/captures.json
```

Choose fresh output paths; the controller refuses an existing capture manifest.
