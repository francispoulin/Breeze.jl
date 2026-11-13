# Four-GPU benchmark smoke checks, Rondeau, 2026-11-12

Status: **AUTOMATIC LAUNCH CONFIGURED**. The watcher checks every 30 seconds and
starts the prepared checks once all four GPUs have no compute processes and at
least 8 GiB free each. Existing idle processes also block launch. It does not
stop other users' jobs or enable shared-GPU testing.

The prepared matrix runs the existing tracked
`benchmarking/distributed/distributed_cbl_benchmark.jl` from `benchmark-results`
with anelastic, compressible explicit and compressible split-explicit dynamics,
each with MPI and NCCL: six cases total.

Each case uses four ranks on four A100s, global 32×32×32, partition 4×1,
deterministic initial conditions, time step 0.005 s, one warmup step and three
timed steps. Anelastic uses Float64; the compressible cases use Float32.
This extends the preceding successful one/two-GPU smoke check using the same
package environment and matching scientific source hashes. It is not a scaling
or performance measurement. No Nsight profiling is requested.

The [controller](run_smoke.py) requires four free GPUs by default. Shared use
requires `--allow-shared-gpus`; all GPUs must still have at least 8 GiB available.
It records placement, per-rank finite-field checks and exact commands for every
case. The [plan](plan.json), [GPU inventory](logs/gpu_inventory.csv),
[existing processes](logs/existing_gpu_processes.csv), and [provenance](provenance/)
are retained. All new files are under `/u/ajtse`.

## Automatic launch

[wait_for_gpus.py](wait_for_gpus.py) runs as a detached background process, so it
continues waiting after the chat turn ends. It verifies the recorded source
hashes before launching and keeps the test checkout separate from the main
working directory. The controller checks availability again before every case;
if another job appears between cases, it stops rather than sharing GPUs.

Live state and PID are retained at:

```text
/u/ajtse/Breeze.jl/.runtime/benchmark_results_four_gpu_smoke_2026-11-12/watcher_status.json
/u/ajtse/Breeze.jl/.runtime/benchmark_results_four_gpu_smoke_2026-11-12/watcher.pid
```

Watcher and controller output go to `logs/watcher.log` and `logs/controller.log`
under that runtime directory. After running, it writes `outcome.md`, full case
logs and `results.json`. With `--publish-results`, it commits the results from
the isolated checkout and attempts a normal fast-forward push to
`origin/benchmark-results`; it never force-pushes. It fast-forwards from that branch
after measurements finish and before committing. If the remote branch has
advanced incompatibly, the live state records a publication failure for
manual review. The local `benchmark-results` branch can then be updated by pulling.

The availability and source-change gate has six CPU regression checks:

```bash
python3 -B test_gpu_gate.py
```

## Branch consolidation

The waiting job was moved to the consolidated `benchmark-results` source before any
cases started. The numerical source matches d1c2558. The driver now also
supports optional NVTX; these smoke checks leave it disabled. Source hashes and the
frozen environment were refreshed before restarting the watcher.
