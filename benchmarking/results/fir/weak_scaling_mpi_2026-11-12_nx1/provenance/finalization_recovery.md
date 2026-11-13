# Finalization recovery

GPU allocation 3868383 completed successfully with all three two-GPU cases validated. Automated CPU finalizer 3868384 failed during its initial controller checksum read with `OSError: [Errno 5] Input/output error`, before finalization started. The failed stdout and Slurm accounting are retained beside this note.

The login node verified all 186 original source/environment hashes and all 180 frozen full-node controller hashes, and confirmed 15 validated case records. See final_hash_verification.log. Finalization is rerun from the login node using the same frozen controller:

```bash
python3 /home/anditse/Breeze.jl/.runtime/studies/weak_scaling_mpi_2026-11-12_nx1/full_node_source/benchmarking/distributed/fir/study.py finalize /home/anditse/Breeze.jl/benchmarking/results/fir/weak_scaling_mpi_2026-11-12_nx1 --publish
```

Its active stdout is retained at `/home/anditse/Breeze.jl/.runtime/studies/weak_scaling_mpi_2026-11-12_nx1/manual_finalization.log`. The final status, comparison CSV, report, and Git history record the outcome; the failed CPU job is not relabeled as successful. No GPU simulations are rerun for this filesystem error.
