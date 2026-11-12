# Post-rebase benchmark smoke test, Rondeau, 2026-11-12

Status: **PASS, with MPI environment warnings recorded below**. All five GPU
cases completed and reported finite fields. Source: `benchmark-results` at
`74b3599`, rebased onto
`origin/claude/distributed-anelastic`. The original `NVTX-optimization` checkout
and its uncommitted `todo.txt` changes are preserved separately.

These are small functional checks of the existing
`benchmarking/distributed/distributed_cbl_benchmark.jl` entry point and supporting
scripts. They are not performance measurements or a full numerical validation.
No scientific source or existing benchmark script is modified for the tests.

## Lightweight checks

- Rorqual result-validation tests: `Ran 7 tests ... OK`.
- Queued-migration gate tests: `Ran 5 tests ... OK`.
- Six distributed-benchmark shell scripts pass `bash -n`.
- Nine distributed/migration Python files pass syntax parsing.
- `plot_split_explicit_results.py --help` exits successfully.
- The plotting CLI renders valid PNG and SVG files from an archived benchmark
  CSV. This checks rendering; it does not create new performance measurements.
- `breeze_env.sh` sources successfully with the installed Julia.
- The scaling launcher dry run issues 24 mocked commands, including eight
  anelastic cases. It forwards the requested time step.
- CUDA and CUDA-aware MPI preflight succeeds, and `pathof(Breeze)` points into
  the `benchmark-results` worktree.

See [basic checks](basic_checks.json), [validation test output](logs/validation_tests.log),
[queue test output](logs/queue_tests.log), [plot CLI help](logs/plot_help.log), and
[environment preflight](logs/preflight.log).

The unit-test summary lines are:

```text
Ran 7 tests in 0.025s
OK

Ran 5 tests in 0.068s
OK
```

## Runtime checks

All cases use the full dry CBL, WENO5, global grid 32×32×32, deterministic smooth
perturbations, time step 0.005 s, one warmup step, three timed steps and no
profiling. The launcher runs the tracked driver directly through `mpiexec`.
Single-GPU cases use physical GPU 1; two-GPU cases use GPUs 1 and 2, both A100s.
Other GPUs have unrelated jobs, so the full production scaling sweep is not run.

| Dynamics | Backend | GPUs / partition | Precision | Result |
|----------|---------|------------------|-----------|--------|
| Anelastic | Serial | 1 / 1×1 | Float64 | PASS; finite fields |
| Compressible explicit | Serial | 1 / 1×1 | Float32 | PASS; finite fields |
| Compressible split-explicit | Serial | 1 / 1×1 | Float32 | PASS; finite fields |
| Anelastic | MPI | 2 / 2×1 | Float64 | PASS; finite fields |
| Anelastic | NCCL | 2 / 2×1 | Float64 | PASS; finite fields |

The three anelastic measurements agree at CSV precision:
`max_abs_w=2.957905e-06`. The explicit and split-explicit serial measurements
report `1.485664e-04` and `1.485390e-04`, respectively. All ranks report
`VALIDATION ... fields_finite=true`; each command exits 0 and writes one result
row. Compilation/startup dominate elapsed process time; the timed loops finish
in under half a second per case.

A runtime case passes only when the process exits successfully, every rank
reports finite fields, placement shows distinct GPUs and the expected
communicator, and the measurement CSV has the expected backend and step count.
The test does not compare complete fields between serial and distributed runs.

[run_smoke.py](run_smoke.py) records each exact invocation, full output and CSV
under [logs/](logs/). [results.json](results.json) records the case outcomes and
placement. It includes launcher messages rather than silently discarding them.

## Environment warnings

Each GPU run prints a post-result
`PMIX ERROR: NO-PERMISSIONS in file dstore_base.c at line 237` despite exiting 0.
The preflight also prints PMIx messages. The two-rank launches warn that the
CUDA shared-memory backing file is on a network filesystem. Temporary and MPI
session files were deliberately placed under `/u/ajtse`, on NFS, to follow the
storage rule. These messages are retained in the logs; their cause was not
diagnosed or fixed in this smoke-test task. The finite-field, placement,
communicator and output checks above still passed.

## Environment and scope

Julia 1.12.7, Breeze 0.15.0, Oceananigans 0.117.2, CUDA 6.4.2, MPI.jl 0.20.29,
NCCL.jl 0.1.3 and system CUDA-aware Open MPI 4.1.7. Julia and BLAS use one thread;
the CUDA memory pool is disabled. The isolated project was resolved and
instantiated with this checkout as its Breeze path dependency.

The tracked distributed-benchmark Manifest was generated with Julia 1.13.2.
It is left unchanged; these runtime checks use an isolated Julia 1.12.7
environment. `setup_breeze.sh` is syntax-checked rather than executed against
the tracked project. No Slurm job, full scaling sweep or GPU profiling capture
is launched.

[Environment exports](environment.sh), [resolved environment](provenance/), and
[source hashes](provenance/source_sha256.json) are retained. All task environments,
temporary files, caches and logs are under
`/u/ajtse/Breeze.jl/.runtime/benchmark_results_smoke_2026-11-12/` while running.
