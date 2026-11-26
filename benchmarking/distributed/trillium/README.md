# Trillium weak scaling

The [Fir runner](../fir/README.md) (`fir/study.py`) also drives Trillium's H100 nodes with
`prepare --machine=trillium`; `--series` picks the workload from `SERIES` as on Fir.
`breeze_anelastic_ce_nccl` is a pure performance benchmark of anelastic and compressible
explicit (CE) on 1, 2, 4, 8, and 16 GPUs: 512³ cells per GPU, N×1 partitions, Δt = 0.01/N s,
Float32, WENO5, 10 warmup and 500 timed steps, NCCL halo exchange, no Nsight and no NVTX.

Trillium differs from Fir in ways the runner handles:

- `/home` is read-only on compute nodes. `environment.sh` keeps the runtime, Julia environment,
  depot, and temporary files under `$SCRATCH/Breeze.jl/.runtime`, and the live study goes on
  `$SCRATCH`. Publication copies it into the checkout (`--publish-directory`).
- Jobs get 1 GPU or whole 4-GPU nodes, and `--mem` is rejected. The 2-GPU case reserves a
  full node and runs 2 ranks; the unused GPUs do not contribute to the measurement.
- The account is `def-fpoulin`; each rank gets its 24-core share of the node.
- There is no CPU partition for a Slurm finalizer. `publish_when_done.sh` runs on
  `trig-login01`, waits until the study's jobs leave the queue, then finalizes and publishes.
- The scratch depot starts empty; `setup.jl` adds the General registry when none is reachable.

All commands run on `trig-login01` from the checkout on `benchmark-results`:

```bash
source benchmarking/distributed/trillium/environment.sh
julia --startup-file=no benchmarking/distributed/fir/setup.jl > "$BREEZE_FIR_ENV/setup.log" 2>&1

name=anelastic_ce_nccl_$(date +%F)
study="$SCRATCH/Breeze.jl/studies/$name"
python3 benchmarking/distributed/fir/study.py prepare "$study" --machine=trillium \
    --series=breeze_anelastic_ce_nccl \
    --publish-directory="$BREEZE_ROOT/benchmarking/results/trillium/$name"
python3 benchmarking/distributed/fir/study.py submit "$study"
nohup bash "$study/publish_when_done.sh" > "$BREEZE_RUNTIME_ROOT/studies/$name/publish.log" 2>&1 &
```

If the watcher dies (for example, a login-node reboot), finalize by hand after all jobs end:
`python3 benchmarking/distributed/fir/study.py finalize "$study" --publish`.
