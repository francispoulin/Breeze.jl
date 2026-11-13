# Five-model weak-scaling efficiency figures

Status: **QUEUED: strict successful-anelastic dependency**.

Both figure jobs depend on a successful anelastic weak-scaling study. The gate
requires completed 1-, 2- and 4-GPU cases, exit code zero, expected 512³ cells/GPU,
positive finite timings, finite fields on every rank, correct communicator and
distinct GPU placements. A single case finishing or a successful process exit alone
does not unlock plotting. A failed, cancelled or invalid anelastic study skips both jobs.

| Figure | Model curves | Job state |
|--------|--------------|-----------|
| Breeze | Anelastic, compressible explicit, compressible split-explicit | WAITING_FOR_ANELASTIC |
| Oceananigans | Hydrostatic, nonhydrostatic | WAITING_FOR_ANELASTIC |

Each figure uses 1, 2 and 4 A100 GPUs, 512³ cells per GPU and efficiency
100 × T₁/Tₙ. PNG and SVG outputs are produced under this report directory.

The Breeze figure uses the new anelastic study and archived unprofiled MPI
compressible measurements from 9–10 November. At four GPUs, anelastic uses 4×1
and the compressible models use 2×2; all local grids remain 512³. The studies use
different source revisions. Input paths, hashes and normalized timings are retained.

The Oceananigans job additionally waits for complete matching hydrostatic and
nonhydrostatic series in the Rondeau canonical result directories. It requires
COMPLETED suite cases, finite rank results, A100 metadata, unprofiled GPU execution
and exactly 512³ cells per GPU. Each model's points must belong to one consistent
series/configuration/version. Dagger and archived directories are excluded.
Oceananigans times use the median of the slowest-rank synchronized windows per step.
Missing data leave the job waiting; no benchmark is launched and no points are invented.

## Queue

Live job states: `/u/ajtse/Breeze.jl/.runtime/efficiency_plots_after_anelastic_2026-11-13/status.json`.
Watcher log: `/u/ajtse/Breeze.jl/.runtime/efficiency_plots_after_anelastic_2026-11-13/logs/watcher.log`.
The watcher polls every 30 seconds and survives this chat. Rendering starts only
after the dependency gate passes. The Breeze job can finish while Oceananigans
continues waiting for its data. If the dependency becomes invalid, rendering stops.

Final figures: breeze_weak_efficiency.png/.svg and oceananigans_weak_efficiency.png/.svg.
The live figures are in the frozen worktree under the runtime path above; the
original checkout retains its queued README until pulled after publication.
When both jobs finish, the watcher fast-forwards from origin/benchmark-results,
commits only this report and its figures, then immediately attempts a normal push.
No merge commit or force-push is used. All temporary files and caches stay in /u/ajtse.
