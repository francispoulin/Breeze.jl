# Kernel time per time step: scaling comparison

GPU execution time per time step, summed over all launches of each kernel and averaged
over the captured steps. Multi-rank cases use each kernel's slowest rank.
Difference = scaled − base. Efficiency = 100 × base / scaled (weak scaling: 100% means
the kernel takes as long with the same work per GPU). A dash marks a kernel that only
one case launches. Rows are sorted by the scaled case's time.

## Compressible explicit (CE)

Base `explicit_1g` (1 GPU), scaled `explicit_2g` (2 GPUs).

| Kernel | Launches/step 1→2 GPU | 1 GPU ms/step | 2 GPU ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `gpu_compute_potential_temperature_tendency_` | 3 → 3 | 39.153 | 39.078 | -0.074 | -0.2% | 100.2 |
| `gpu_compute_z_momentum_tendency_` | 3 → 3 | 34.088 | 34.131 | +0.043 | +0.1% | 99.9 |
| `gpu_compute_y_momentum_tendency_` | 3 → 3 | 33.462 | 33.487 | +0.025 | +0.1% | 99.9 |
| `gpu_compute_x_momentum_tendency_` | 3 → 3 | 33.077 | 33.086 | +0.009 | +0.0% | 100.0 |
| `gpu_compute_scalar_tendency_` | 3 → 3 | 33.056 | 33.073 | +0.017 | +0.1% | 99.9 |
| `gpu__ssp_rk3_substep_` | 18 → 18 | 30.248 | 30.238 | -0.010 | -0.0% | 100.0 |
| `ncclDevKernel_SendRecv` | 0 → 63 | — | 16.287 | — | — | — |
| `gpu_broadcast_kernel_cartesian` | 6 → 132 | 11.367 | 15.683 | +4.315 | +38.0% | 72.5 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 → 3 | 12.960 | 12.958 | -0.002 | -0.0% | 100.0 |
| `gpu__compute_temperature_and_pressure_` | 3 → 3 | 11.464 | 11.466 | +0.002 | +0.0% | 100.0 |
| `gpu__compute_velocities_` | 3 → 3 | 9.162 | 9.109 | -0.053 | -0.6% | 100.6 |
| `gpu__copy_from_buffer_` | 0 → 90 | — | 5.098 | — | — | — |
| `gpu__compute_density_tendency_` | 3 → 3 | 5.029 | 5.029 | +0.000 | +0.0% | 100.0 |
| `gpu__compute_total_density_` | 3 → 3 | 3.812 | 3.832 | +0.020 | +0.5% | 99.5 |
| `gpu__fill_periodic_south_and_north_halo_` | 57 → 63 | 1.741 | 1.624 | -0.117 | -6.7% | 107.2 |
| `gpu__fill_bottom_and_top_halo_` | 54 → 60 | 0.459 | 0.493 | +0.034 | +7.5% | 93.0 |
| `gpu__compute_z_bcs_` | 9 → 9 | 0.056 | 0.056 | -0.000 | -0.2% | 100.2 |
| `gpu__fill_periodic_west_and_east_halo_` | 57 → 0 | 12.902 | — | — | — | — |
| **All kernels** | 231 → 465 | 272.035 | 284.688 | +12.652 | +4.7% | 95.6 |

## Compressible split-explicit (CSE)

Base `splitexplicit_1g` (1 GPU), scaled `splitexplicit_2g` (2 GPUs).

| Kernel | Launches/step 1→2 GPU | 1 GPU ms/step | 2 GPU ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `gpu__post_solve_recovery_` | 22 → 22 | 155.788 | 156.354 | +0.566 | +0.4% | 99.6 |
| `gpu_solve_batched_tridiagonal_system_kernel_` | 22 → 22 | 149.607 | 149.548 | -0.059 | -0.0% | 100.0 |
| `gpu__build_predictors_` | 22 → 22 | 104.174 | 104.114 | -0.060 | -0.1% | 100.1 |
| `gpu__explicit_horizontal_step_` | 22 → 22 | 84.530 | 84.315 | -0.215 | -0.3% | 100.3 |
| `ncclDevKernel_SendRecv` | 0 → 285 | — | 83.755 | — | — | — |
| `gpu__thermal_divergence_damping_` | 22 → 22 | 83.474 | 82.958 | -0.516 | -0.6% | 100.6 |
| `gpu__build_vertical_rhs_` | 22 → 22 | 74.780 | 74.528 | -0.252 | -0.3% | 100.3 |
| `gpu_compute_potential_temperature_tendency_` | 3 → 3 | 39.156 | 39.110 | -0.046 | -0.1% | 100.1 |
| `gpu_compute_y_momentum_tendency_` | 3 → 3 | 33.210 | 33.208 | -0.002 | -0.0% | 100.0 |
| `gpu_compute_scalar_tendency_` | 3 → 3 | 33.072 | 33.061 | -0.011 | -0.0% | 100.0 |
| `gpu_compute_z_momentum_tendency_` | 3 → 3 | 32.863 | 32.864 | +0.001 | +0.0% | 100.0 |
| `gpu_compute_x_momentum_tendency_` | 3 → 3 | 32.845 | 32.850 | +0.006 | +0.0% | 100.0 |
| `gpu_broadcast_kernel_cartesian` | 6 → 576 | 11.369 | 30.233 | +18.864 | +165.9% | 37.6 |
| `gpu__copy_from_buffer_` | 0 → 534 | — | 30.154 | — | — | — |
| `gpu__recover_full_state_` | 3 → 3 | 21.437 | 21.370 | -0.067 | -0.3% | 100.3 |
| `gpu__compute_velocities_` | 6 → 6 | 18.318 | 18.221 | -0.097 | -0.5% | 100.5 |
| `gpu__compute_linearization_basic_state_` | 4 → 4 | 17.512 | 17.498 | -0.013 | -0.1% | 100.1 |
| `gpu__finalize_time_averaged_velocity_` | 3 → 3 | 15.459 | 15.411 | -0.048 | -0.3% | 100.3 |
| `gpu__initialize_stage_perturbations_` | 3 → 3 | 15.263 | 15.309 | +0.046 | +0.3% | 99.7 |
| `gpu__zero_stage_workspaces_` | 3 → 3 | 13.438 | 13.529 | +0.091 | +0.7% | 99.3 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 → 3 | 12.970 | 12.946 | -0.024 | -0.2% | 100.2 |
| `gpu__compute_temperature_and_pressure_` | 3 → 3 | 11.469 | 11.463 | -0.006 | -0.1% | 100.1 |
| `gpu__fill_periodic_south_and_north_halo_` | 279 → 285 | 7.713 | 7.010 | -0.704 | -9.1% | 110.0 |
| `gpu__assemble_slow_vertical_momentum_tendency_` | 3 → 3 | 5.052 | 5.051 | -0.002 | -0.0% | 100.0 |
| `gpu__compute_density_tendency_` | 3 → 3 | 5.031 | 5.030 | -0.001 | -0.0% | 100.0 |
| `gpu__compute_total_density_` | 3 → 3 | 3.811 | 3.828 | +0.017 | +0.4% | 99.6 |
| `gpu__initialize_perturbation_with_rewind_` | 3 → 3 | 3.794 | 3.788 | -0.006 | -0.2% | 100.2 |
| `gpu__rk3_substep_` | 3 → 3 | 3.797 | 3.778 | -0.019 | -0.5% | 100.5 |
| `gpu__fill_bottom_and_top_halo_` | 267 → 273 | 2.050 | 2.014 | -0.036 | -1.8% | 101.8 |
| `gpu__compute_z_bcs_` | 9 → 9 | 0.059 | 0.057 | -0.002 | -3.5% | 103.6 |
| `gpu__fill_periodic_west_and_east_halo_` | 279 → 0 | 56.864 | — | — | — | — |
| **All kernels** | 1030 → 2152 | 1048.905 | 1113.338 | +64.433 | +6.1% | 94.2 |
