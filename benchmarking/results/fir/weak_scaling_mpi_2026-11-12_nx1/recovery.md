# Two-GPU recovery

The initial two-GPU allocation (3715782) completed stepping but both MPI ranks reported the same GPU UUID and `CUDA_VISIBLE_DEVICES=1`. All three measurements were rejected before inclusion in metrics. Full original commands, logs, and measurements remain under [rejected_attempts/3715782/](rejected_attempts/3715782/).

The recovery launcher requests one H100 per task and binds one task per GPU. An all-rank architecture/GPU preflight requires distinct host/UUID pairs before model construction. Only the three two-GPU cases are rerun. The original model source, driver, package environment, initial conditions, grid, precision, time steps, warmup, and timed-step count are unchanged. The 12 accepted measurements at 1/4/8/16 GPUs remain intact.

The initial finalizer generated valid 12-case CSV/plots but could not commit them because CSV carriage returns triggered Git whitespace checks. The updated reporter emits LF line endings and publication normalizes bytes, including CRLF-only differences. The original failed finalization log and accounting are retained under [rejected_attempts/finalizer_3715786/](rejected_attempts/finalizer_3715786/).

Recovery uses a separate frozen controller archive and hashes under provenance/. Its Julia environment still points to the original frozen scientific source. The recovery allocation requests two GPUs for one hour; its after-any finalizer revalidates all 15 cases, regenerates reports, commits results, and pushes them. Finalizer stdout stays under the runtime directory so it does not stage a changing copy of itself.
