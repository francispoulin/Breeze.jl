# NVIDIA engineer's advice on distributed GPU optimization

Source: Jonathan's message supplied by the user; summary recorded on 2026-11-12.
The review covered Breeze branch `fjp/distributed-benchmark` at commit `4aaa00e`
with Oceananigans 0.117.2. It was a **source review without GPU testing or
benchmarking**. Performance explanations and correctness concerns below are
hypotheses to verify. Check the current checkout before treating an item as open
or applying a proposed change.

## First priority: asynchronous communication correctness

Use synchronous prognostic-field halo fills as an initial correctness baseline.
Investigate these concerns before relying on asynchronous benchmark timings:

| Concern from the review | What to check |
|-------------------------|---------------|
| Missing completion and growing queues | Breeze requests asynchronous prognostic fills without an apparent `synchronize_communication!` call. NCCL queues deferred unpacks, and later synchronous fills apparently do not drain that queue. Check completion paths and queue growth over many steps. |
| Send-buffer reuse | Check whether packing can overwrite a buffer before an earlier asynchronous NCCL send finishes reading it; verify stream dependencies and buffer lifetime. |
| Stale moisture halos | The review found no later synchronous refill or completion/unpack for asynchronously filled moisture. Check both MPI and NCCL with nonzero moisture and tracers; zero moisture in the dry benchmark can mask the issue. Another completion path may have been missed. |

## Likely communication costs

- Distributed halos exchange one field at a time; a 2×2 partition also sends
  separate corner messages.
- MPI repeatedly waits for GPU packing, posts messages, waits for completion,
  and launches unpacking. These CPU/GPU handoffs may add substantial overhead.
- NCCL uses a separate communication stream and stream dependencies, avoiding
  those explicit CPU waits. This may explain part of its improvement; the review
  did not measure the contribution.
- Reuse Oceananigans' local-only halo fills and communication/computation overlap
  patterns where appropriate. Its 2-D split-explicit free-surface solver uses
  wide halos to reduce exchange frequency; adapting that approach to Breeze's
  3-D acoustic solver would require more work.

## Profiling and comparison plan

After checking correctness, compare one GPU and four-GPU partitions **4×1,
1×4, and 2×2**, using both MPI and NCCL for distributed runs. Measure wall time,
GPU gaps, exchange counts, and CPU polling/wait time.

Suggested NVTX ranges:

- Whole time step in the benchmark driver.
- RK stages and `update_state!` in `acoustic_runge_kutta_3.jl`.
- Acoustic loop, individual substeps, compute stages, and intervening halo fills
  in `acoustic_substepping.jl`.
- Packing, completion/posting waits, unpacking, and corner exchanges inside
  Oceananigans when finer attribution is needed.

The reviewed code had NVTX transitively through CUDA but no application ranges
identified by Jonathan. Check existing instrumentation before adding ranges.
Declare NVTX directly in the environment that imports it. Bracket a few steps
after warmup with profiler start/stop calls to match the `nsys` capture flags.

## Small changes to test individually

1. Remove immediate duplicate density/momentum halo fills at stage end.
2. For the periodic CBL, test removing the two post-damping momentum fills that
   are followed by another fill before the next horizontal-stencil use.
3. Retain only local boundary fills for the density perturbation, if sufficient.
4. Copy already-valid halos into the saved thermodynamic perturbation instead
   of exchanging that saved field again.

Each candidate needs before/after numerical comparisons with **identical initial
conditions**. Check terrain and open-boundary cases separately; conclusions from
the periodic dry CBL do not establish correctness there.

## Pressure solver and anelastic baseline

- Investigate the large one-to-two-GPU pressure-solver slowdown, even within a
  node. Separate local FFT/permutation work from communication before choosing
  an alternative solver.
- Obtain an anelastic baseline early because it is the intended application.
  Start with N×1: in the reviewed implementation, that path uses two transposes
  per solve versus eight on 2×2. The 1×N layout does not use that optimized path.

## Additional experiments and larger projects

- Test the CUDA memory pool with NCCL; the reviewed launcher defaults it off.
- Consider CUDA graphs later to reduce kernel-launch latency.
- Use the correctness checks and measurements to decide whether to pursue
  batched multi-field distributed exchange or a more involved wide-halo scheme.
- If exploring cuFFTMp integration, Jonathan offered to take questions to
  NVIDIA's math-libraries team.

Order of work: establish communication correctness, profile and compare layouts,
test the small changes separately, then choose a larger optimization project.
