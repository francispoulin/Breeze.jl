# Four-GPU smoke checks: PASS

Completed cases: 6/6; controller exit: 0.

| Case | Status | Maximum absolute vertical velocity |
|------|--------|------------------------------------|
| anelastic_MPI_4g | PASS | 2.957905e-06 |
| anelastic_NCCL_4g | PASS | 2.957905e-06 |
| compressible_explicit_MPI_4g | PASS | 1.485664e-04 |
| compressible_explicit_NCCL_4g | PASS | 1.485664e-04 |
| compressible_splitexplicit_MPI_4g | PASS | 1.485390e-04 |
| compressible_splitexplicit_NCCL_4g | PASS | 1.485390e-04 |

See results.json and per-case logs for rank placement, finite-field checks and launcher messages.
These are functional checks, not performance or full numerical-validation results.
