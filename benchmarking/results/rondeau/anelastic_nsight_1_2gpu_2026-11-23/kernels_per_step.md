# GPU kernel time per time step

Mean over the captured steps of the summed duration of every launch of each kernel;
launches with the same short name are one row. Times are GPU execution in
milliseconds; kernels on one GPU can overlap, so the column sum can exceed the
step's wall time. NCCL kernels include time spent waiting for the peer rank.

## anelastic_1g

1 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step |
|---|---|---|
| `gpu_compute_potential_temperature_tendency_` | 3 | 36.925 |
| `gpu_compute_z_momentum_tendency_` | 3 | 35.390 |
| `gpu_compute_y_momentum_tendency_` | 3 | 32.835 |
| `gpu_compute_scalar_tendency_` | 3 | 32.689 |
| `gpu_compute_x_momentum_tendency_` | 3 | 32.573 |
| `gpu__ssp_rk3_substep_` | 15 | 25.178 |
| `gpu_solve_batched_tridiagonal_system_kernel_` | 3 | 14.467 |
| `gpu__fill_periodic_west_and_east_halo_` | 51 | 11.008 |
| `gpu__pressure_correct_momentum_` | 3 | 10.704 |
| `vector_fft` | 6 | 9.493 |
| `gpu_broadcast_kernel_cartesian` | 5 | 9.478 |
| `regular_fft` | 6 | 9.449 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 | 8.682 |
| `gpu__compute_velocities_` | 3 | 8.093 |
| `gpu__compute_anelastic_source_term_` | 3 | 6.145 |
| `scal_kernel_ref` | 3 | 4.667 |
| `gpu_copy_real_component_` | 3 | 3.779 |
| `gpu_fill_kernel_` | 3 | 2.073 |
| `gpu__fill_periodic_south_and_north_halo_` | 51 | 1.484 |
| `gpu__fill_bottom_and_top_halo_` | 51 | 0.388 |
| `gpu__compute_z_bcs_` | 9 | 0.054 |
| `partial_mapreduce_grid` | 3 | 0.015 |
| `gpu_broadcast_kernel_linear` | 3 | 0.015 |
| **All kernels** | 239 | **295.583** |

## anelastic_2g

2 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step | Rank 1 ms/step |
|---|---|---|---|
| `gpu_permutedims_kernel_` | 12 | 73.383 | 73.365 |
| `ncclDevKernel_SendRecv` | 60 | 59.062 | 58.280 |
| `gpu_compute_potential_temperature_tendency_` | 3 | 37.019 | 37.086 |
| `gpu_compute_z_momentum_tendency_` | 3 | 35.472 | 35.503 |
| `gpu_compute_y_momentum_tendency_` | 3 | 32.853 | 32.869 |
| `gpu_compute_scalar_tendency_` | 3 | 32.701 | 32.718 |
| `gpu_compute_x_momentum_tendency_` | 3 | 32.609 | 32.611 |
| `gpu__ssp_rk3_substep_` | 15 | 25.189 | 25.207 |
| `vector_fft` | 12 | 18.935 | 18.939 |
| `gpu_broadcast_kernel_cartesian` | 119 | 18.780 | 18.617 |
| `gpu_solve_batched_tridiagonal_system_kernel_` | 3 | 14.404 | 14.392 |
| `gpu__pack_buffer_x_to_y_` | 3 | 13.401 | 13.410 |
| `gpu__pressure_correct_momentum_` | 3 | 11.385 | 11.365 |
| `scal_kernel_ref` | 6 | 9.334 | 9.334 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 | 8.687 | 8.702 |
| `gpu__compute_velocities_` | 3 | 8.097 | 8.095 |
| `gpu__compute_anelastic_source_term_` | 3 | 6.101 | 6.099 |
| `gpu__unpack_buffer_y_from_x_` | 3 | 5.020 | 5.024 |
| `gpu__unpack_buffer_x_from_y_` | 3 | 4.962 | 4.959 |
| `gpu__pack_buffer_y_to_x_` | 3 | 4.827 | 4.825 |
| `gpu__copy_from_buffer_` | 78 | 4.418 | 4.404 |
| `partial_mapreduce_grid` | 6 | 4.233 | 4.235 |
| `gpu__copy_real_component_` | 3 | 3.730 | 3.729 |
| `gpu__broadcast_kernel_` | 3 | 2.684 | 2.685 |
| `gpu__fill_periodic_south_and_north_halo_` | 54 | 1.494 | 1.406 |
| `gpu__fill_bottom_and_top_halo_` | 54 | 0.505 | 0.436 |
| `gpu__compute_z_bcs_` | 9 | 0.055 | 0.055 |
| **All kernels** | 473 | **469.342** | **468.351** |
