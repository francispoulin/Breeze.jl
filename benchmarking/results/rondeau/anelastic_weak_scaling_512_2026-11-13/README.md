# Anelastic weak scaling: 512³ cells per A100

Status: **QUEUED: waiting for preceding NVTX/Nsight campaign**.

Three unprofiled dry-CBL cases use the tracked distributed benchmark driver on
benchmark-results: Float32, WENO5, deterministic initial perturbations, Δt=0.005 s,
ten warmup steps and 500 timed steps. MPI handles distributed communication; the
one-GPU case uses the serial GPU architecture. Each case has one thread per rank
and the CUDA memory pool disabled. No NVTX instrumentation or Nsight capture is enabled.

| GPUs | Global grid | Partition | Seconds/step | Weak efficiency | State |
|------|-------------|-----------|--------------|-----------------|-------|
| 1 | 512×512×512 | 1×1 | pending | pending | PENDING |
| 2 | 1024×512×512 | 2×1 | pending | pending | PENDING |
| 4 | 2048×512×512 | 4×1 | pending | pending | PENDING |

Efficiency is **100 × T₁ / Tₙ**, with fixed 512³ cells per GPU. It is not divided
by GPU count. Each local grid remains cubic; the global grid grows in x. N×1 is
the initial anelastic layout suggested in .agents/nvidia_optimization_advice.md.
Seconds per step use the driver's slowest-rank wall time divided by 500. Each case
has one measured window; no uncertainty estimates are implied by these curves.

## Queue and provenance

This watcher waits for the already queued split-explicit NVTX/Nsight campaign to
reach a terminal state, then runs the three anelastic cases sequentially. It checks
every 30 seconds for enough GPUs with no compute processes and at least 30000 MiB
free each. Idle compute contexts also block selection; no existing jobs are stopped.
A failure in the preceding profiler campaign does not cancel this independent study.

Live state: `/u/ajtse/Breeze.jl/.runtime/anelastic_weak_scaling_512_2026-11-13/status.json`.
Controller log: `/u/ajtse/Breeze.jl/.runtime/anelastic_weak_scaling_512_2026-11-13/logs/controller.log`.
Frozen checkout and environment: `/u/ajtse/Breeze.jl/.runtime/anelastic_weak_scaling_512_2026-11-13`.

Source and environment hashes are checked before execution and after the last case.
Per-case logs, commands and GPU inventories are retained. Placement, communicator,
CSV geometry and finite fields on every rank are required. These are finite-field
checks; they do not establish serial-solution agreement or long-term stability.

## Plots

When all three cases succeed, plot_efficiency.py creates anelastic_weak_efficiency
and breeze_weak_efficiency in PNG and SVG. The Breeze figure has exactly three
model curves: anelastic, compressible explicit and compressible split-explicit.
Its MPI compressible data are archived 9–10 November measurements copied into
provenance/compressible_measurements.csv; the four-GPU compressible layout is 2×2
on 1024×1024×512, keeping the same 512³ local grid. Anelastic uses 4×1 on
2048×512×512. All models use Δt=0.005 s. Source revisions differ between archived and new
studies; this is a comparison of each model's measured scaling, not a same-revision
comparison of absolute model costs. All points use unprofiled seconds per step.
Matching 512³ Oceananigans measurements are still needed for the separate Oceananigans figure.

After completion the watcher fast-forwards from origin/benchmark-results, commits
only this report and its figures, then immediately attempts a normal push. It never
force-pushes. Incoming conflicting edits or a rejected push are recorded in status.json.
All task files, logs, temporary files, caches and environments remain inside /u/ajtse.
