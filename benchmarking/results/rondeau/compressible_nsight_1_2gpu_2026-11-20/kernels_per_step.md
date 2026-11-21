# GPU kernel time per time step

Mean over the captured steps of the summed duration of every launch of each kernel;
launches with the same short name are one row. Times are GPU execution in
milliseconds; kernels on one GPU can overlap, so the column sum can exceed the
step's wall time. NCCL kernels include time spent waiting for the peer rank.

## explicit_1g

1 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step |
|---|---|---|
| `gpu_compute_potential_temperature_tendency_` | 3 | 39.153 |
| `gpu_compute_z_momentum_tendency_` | 3 | 34.088 |
| `gpu_compute_y_momentum_tendency_` | 3 | 33.462 |
| `gpu_compute_x_momentum_tendency_` | 3 | 33.077 |
| `gpu_compute_scalar_tendency_` | 3 | 33.056 |
| `gpu__ssp_rk3_substep_` | 18 | 30.248 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 | 12.960 |
| `gpu__fill_periodic_west_and_east_halo_` | 57 | 12.902 |
| `gpu__compute_temperature_and_pressure_` | 3 | 11.464 |
| `gpu_broadcast_kernel_cartesian` | 6 | 11.367 |
| `gpu__compute_velocities_` | 3 | 9.162 |
| `gpu__compute_density_tendency_` | 3 | 5.029 |
| `gpu__compute_total_density_` | 3 | 3.812 |
| `gpu__fill_periodic_south_and_north_halo_` | 57 | 1.741 |
| `gpu__fill_bottom_and_top_halo_` | 54 | 0.459 |
| `gpu__compute_z_bcs_` | 9 | 0.056 |
| **All kernels** | 231 | **272.035** |

## explicit_2g

2 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step | Rank 1 ms/step |
|---|---|---|---|
| `ncclDevKernel_SendRecv` | 63 | 371.534 | 36.745 |
| `gpu_compute_potential_temperature_tendency_` | 3 | 39.072 | 88.470 |
| `gpu_compute_z_momentum_tendency_` | 3 | 34.128 | 77.334 |
| `gpu_compute_y_momentum_tendency_` | 3 | 33.484 | 76.369 |
| `gpu_compute_scalar_tendency_` | 3 | 33.064 | 74.080 |
| `gpu_compute_x_momentum_tendency_` | 3 | 33.088 | 73.908 |
| `gpu__ssp_rk3_substep_` | 18 | 30.253 | 71.097 |
| `gpu_broadcast_kernel_cartesian` | 132 | 15.629 | 36.033 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 | 12.955 | 30.366 |
| `gpu__compute_temperature_and_pressure_` | 3 | 11.466 | 26.929 |
| `gpu__compute_velocities_` | 3 | 9.112 | 21.939 |
| `gpu__copy_from_buffer_` | 90 | 5.095 | 10.851 |
| `gpu__compute_density_tendency_` | 3 | 5.038 | 10.615 |
| `gpu__compute_total_density_` | 3 | 3.833 | 8.872 |
| `gpu__fill_periodic_south_and_north_halo_` | 63 | 1.592 | 3.173 |
| `gpu__fill_bottom_and_top_halo_` | 60 | 0.476 | 0.988 |
| `gpu__compute_z_bcs_` | 9 | 0.055 | 0.056 |
| **All kernels** | 465 | **639.876** | **647.826** |

## splitexplicit_1g

1 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step |
|---|---|---|
| `gpu__post_solve_recovery_` | 22 | 155.788 |
| `gpu_solve_batched_tridiagonal_system_kernel_` | 22 | 149.607 |
| `gpu__build_predictors_` | 22 | 104.174 |
| `gpu__explicit_horizontal_step_` | 22 | 84.530 |
| `gpu__thermal_divergence_damping_` | 22 | 83.474 |
| `gpu__build_vertical_rhs_` | 22 | 74.780 |
| `gpu__fill_periodic_west_and_east_halo_` | 279 | 56.864 |
| `gpu_compute_potential_temperature_tendency_` | 3 | 39.156 |
| `gpu_compute_y_momentum_tendency_` | 3 | 33.210 |
| `gpu_compute_scalar_tendency_` | 3 | 33.072 |
| `gpu_compute_z_momentum_tendency_` | 3 | 32.863 |
| `gpu_compute_x_momentum_tendency_` | 3 | 32.845 |
| `gpu__recover_full_state_` | 3 | 21.437 |
| `gpu__compute_velocities_` | 6 | 18.318 |
| `gpu__compute_linearization_basic_state_` | 4 | 17.512 |
| `gpu__finalize_time_averaged_velocity_` | 3 | 15.459 |
| `gpu__initialize_stage_perturbations_` | 3 | 15.263 |
| `gpu__zero_stage_workspaces_` | 3 | 13.438 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 | 12.970 |
| `gpu__compute_temperature_and_pressure_` | 3 | 11.469 |
| `gpu_broadcast_kernel_cartesian` | 6 | 11.369 |
| `gpu__fill_periodic_south_and_north_halo_` | 279 | 7.713 |
| `gpu__assemble_slow_vertical_momentum_tendency_` | 3 | 5.052 |
| `gpu__compute_density_tendency_` | 3 | 5.031 |
| `gpu__compute_total_density_` | 3 | 3.811 |
| `gpu__rk3_substep_` | 3 | 3.797 |
| `gpu__initialize_perturbation_with_rewind_` | 3 | 3.794 |
| `gpu__fill_bottom_and_top_halo_` | 267 | 2.050 |
| `gpu__compute_z_bcs_` | 9 | 0.059 |
| **All kernels** | 1030 | **1048.905** |

## splitexplicit_2g

2 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step | Rank 1 ms/step |
|---|---|---|---|
| `gpu__post_solve_recovery_` | 22 | 156.354 | 150.023 |
| `gpu_solve_batched_tridiagonal_system_kernel_` | 22 | 149.548 | 144.313 |
| `gpu__build_predictors_` | 22 | 104.114 | 104.113 |
| `gpu__explicit_horizontal_step_` | 22 | 84.315 | 83.773 |
| `ncclDevKernel_SendRecv` | 285 | 73.955 | 83.755 |
| `gpu__thermal_divergence_damping_` | 22 | 82.837 | 82.958 |
| `gpu__build_vertical_rhs_` | 22 | 74.528 | 74.510 |
| `gpu_compute_potential_temperature_tendency_` | 3 | 39.110 | 39.082 |
| `gpu_compute_y_momentum_tendency_` | 3 | 33.199 | 33.208 |
| `gpu_compute_scalar_tendency_` | 3 | 33.061 | 33.051 |
| `gpu_compute_z_momentum_tendency_` | 3 | 32.857 | 32.864 |
| `gpu_compute_x_momentum_tendency_` | 3 | 32.843 | 32.850 |
| `gpu_broadcast_kernel_cartesian` | 576 | 30.176 | 30.233 |
| `gpu__copy_from_buffer_` | 534 | 30.154 | 30.112 |
| `gpu__recover_full_state_` | 3 | 21.370 | 21.281 |
| `gpu__compute_velocities_` | 6 | 18.212 | 18.221 |
| `gpu__compute_linearization_basic_state_` | 4 | 17.496 | 17.498 |
| `gpu__finalize_time_averaged_velocity_` | 3 | 15.411 | 15.020 |
| `gpu__initialize_stage_perturbations_` | 3 | 15.305 | 15.309 |
| `gpu__zero_stage_workspaces_` | 3 | 13.529 | 13.499 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 | 12.946 | 12.930 |
| `gpu__compute_temperature_and_pressure_` | 3 | 11.463 | 11.461 |
| `gpu__fill_periodic_south_and_north_halo_` | 285 | 7.010 | 6.955 |
| `gpu__assemble_slow_vertical_momentum_tendency_` | 3 | 5.051 | 5.049 |
| `gpu__compute_density_tendency_` | 3 | 5.030 | 5.028 |
| `gpu__compute_total_density_` | 3 | 3.828 | 3.828 |
| `gpu__initialize_perturbation_with_rewind_` | 3 | 3.788 | 3.788 |
| `gpu__rk3_substep_` | 3 | 3.778 | 3.777 |
| `gpu__fill_bottom_and_top_halo_` | 273 | 2.014 | 1.998 |
| `gpu__compute_z_bcs_` | 9 | 0.057 | 0.056 |
| **All kernels** | 2152 | **1113.338** | **1110.542** |
