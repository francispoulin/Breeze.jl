# Queued move to the Francis Poulin fork

Requested on November 9, 2026. Current commits are pushed to
`https://github.com/AndiJingzhiTse/Breeze.jl` as soon as GitHub authentication is available.
The source branch is `distributed-benchmark`; the isolated profiling-support branch is also preserved.
The committed `todo.txt` and the explicitly labeled in-progress result snapshot are included.

The checkout migration waits for all 14 timestep/partition cases, all six requested Nsight cases,
their final validation and commits, and termination of their controllers. It then:

1. Incorporates the isolated profiling-support commit and pushes the final source/results backup.
2. Saves a full local Git bundle, then clones `francispoulin/Breeze.jl` into a staging directory.
3. Creates `benchmark-results` from `origin/fjp/distributed-benchmark` and applies our benchmark,
   result, instruction and workflow-note changes relative to the previously imported fork tree.
4. Preserves the fork's scientific source, root dependencies, newer benchmark environment and setup.
   The reviewed driver integration supports both `--comm` and the fork's `--communicator` spelling.
5. Commits and pushes `benchmark-results` to the personal fork. The Francis Poulin base branch is not modified.
6. Archives the old checkout at `/u/ajtse/Breeze.jl.before-fjp-2026-11-09`, repairs the link to the
   profiling worktree, and installs the new clone at `/u/ajtse/Breeze.jl` on `benchmark-results`.

Raw Nsight captures remain in `/local_scratch/ajtse/breeze_nsight_mpi_nccl_2026-11-09/`.
Historical results retain their measured source revisions and environments; they do not describe
the newer fork's scientific code. The archived checkout and source snapshots preserve reproduction evidence.

No clone, branch transfer, or checkout replacement occurs before the running workflows finish.
Uncommitted work, an unexpected driver update, merge conflicts, a push failure, or an existing backup/staging
directory stops the migration for review before checkout replacement. Pushes never force-update branches.

The queue's live state and log are outside the checkout so they survive its replacement:

- `/u/ajtse/.codex/queues/breeze-fjp-migration-2026-11-09/status.json`
- `/u/ajtse/.codex/queues/breeze-fjp-migration-2026-11-09/worker.log`

Configuration: [queue.json](queue.json). Worker: [run_migration.py](run_migration.py).

Start only one worker, from the repository root:

```bash
python3 -u benchmarking/queued_fork_migration/run_migration.py
```
