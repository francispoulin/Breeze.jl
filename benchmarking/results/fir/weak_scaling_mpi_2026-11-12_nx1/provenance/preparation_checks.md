# Preparation checks

- Controller contract checks: `Ran 9 tests in 1.234s`, `OK`; see [contract_tests.log](contract_tests.log).
- All six generated allocation/finalizer scripts pass `bash -n`.
- Both the 16-GPU request and CPU finalizer request pass Slurm `--test-only`; see [GPU](slurm_gpu_dry_run.log) and [CPU](slurm_cpu_dry_run.log) output. Test-only job numbers are scheduler estimates, not submitted allocations.
- Package loading succeeds with `MPI_cuda=true` and system OpenMPI; see [package_preflight.log](package_preflight.log).
- Setup initially logged a linker failure while caching OceananigansCUDAExt. Loading the package stack rebuilt that extension successfully. The setup log retains the diagnostic while omitting the failed process environment dump.
- Frozen-source package loading and its Breeze path are recorded in [frozen_package_preflight.log](frozen_package_preflight.log).
- These are preparation checks, not GPU performance measurements. GPU preflight and every rank's field/placement evidence are recorded by the allocations.

The source revision is committed locally. Its initial `git push` failed because Fir's Git configuration refers to a missing `credential-manager`, with no GitHub credentials available. Measurements and local retention can continue; publication requires working Git authentication. No credentials are stored with the study.
