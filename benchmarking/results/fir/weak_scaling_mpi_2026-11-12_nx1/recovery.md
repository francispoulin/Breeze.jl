# Two-GPU recovery

The initial two-GPU allocation (3715782) completed stepping but both MPI ranks reported the same GPU UUID and `CUDA_VISIBLE_DEVICES=1`. All three measurements were rejected before inclusion in metrics. Full original commands, logs, and measurements remain under [rejected_attempts/3715782/](rejected_attempts/3715782/).

The production recovery explicitly requests all allocated node GPUs on each step. Its architecture/GPU preflight requires distinct selected host/UUID pairs and full visibility of the allocated GPUs before model construction. Only the three two-GPU cases are rerun. The original model source, driver, package environment, initial conditions, grid, precision, time steps, warmup, and timed-step count are unchanged. The 12 accepted measurements at 1/4/8/16 GPUs remain intact.

The initial finalizer generated valid 12-case CSV/plots but could not commit them because CSV carriage returns triggered Git whitespace checks. The updated reporter emits LF line endings and publication normalizes bytes, including CRLF-only differences. The original failed finalization log and accounting are retained under [rejected_attempts/finalizer_3715786/](rejected_attempts/finalizer_3715786/).

Recovery uses a separate frozen controller archive and hashes under provenance/. Its Julia environment still points to the original frozen scientific source. The recovery allocation requests two GPUs for one hour; its after-any finalizer revalidates all 15 cases, regenerates reports, commits results, and pushes them. Finalizer stdout stays under the runtime directory so it does not stage a changing copy of itself.

## GPU visibility validation

The first recovery allocation (3819903) established distinct GPU UUIDs but Oceananigans warned that one-device-per-rank masking can prevent direct GPU peer copies. It and its dependent finalizer (3819904) were cancelled before accepting any performance measurements. Their full evidence is retained under [rejected_attempts/3819903/](rejected_attempts/3819903/).

The production recovery explicitly requests both allocated GPUs on every `srun` step with `--gpus-per-node=h100:2 --gpu-bind=none`. Its preflight requires two visible devices per rank and distinct selected GPU UUIDs. This retains the peer-copy communication configuration of the validated larger runs. No scientific source or environment changes.

## Full-node reservation for two active GPUs

The partial-node recovery 3823308 still exposed one device to both ranks despite an explicit two-GPU step request. Its placement gate rejected the allocation before timing; its complete evidence is retained under [rejected_attempts/3823308/](rejected_attempts/3823308/). Finalizer 3823309 correctly published the 12 valid cases and returned incomplete status.

The next recovery reserves all four H100s and four task slots on one node, then launches exactly two MPI ranks and two model subdomains with all four allocated devices visible. Only two GPUs perform the benchmark; two GPUs remain unused. The global grid is still 1024×512×512 and the partition remains 2×1. This matches the full-node GPU visibility of the validated larger cases. The preflight requires four visible GPUs on both ranks and two distinct selected UUIDs. Extra reserved GPUs are recorded in submission/accounting and excluded from throughput and efficiency denominators. Scientific source, environment, precision, and stepping parameters remain unchanged.
