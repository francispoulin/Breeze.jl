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
| `gpu_permutedims_kernel_` | 12 | 73.909 | 74.157 |
| `ncclDevKernel_SendRecv` | 60 | 60.809 | 58.870 |
| `gpu_compute_potential_temperature_tendency_` | 3 | 36.992 | 37.035 |
| `gpu_compute_z_momentum_tendency_` | 3 | 35.389 | 35.428 |
| `gpu_compute_y_momentum_tendency_` | 3 | 32.843 | 32.849 |
| `gpu_compute_scalar_tendency_` | 3 | 32.703 | 32.724 |
| `gpu_compute_x_momentum_tendency_` | 3 | 32.605 | 32.608 |
| `gpu__ssp_rk3_substep_` | 15 | 25.176 | 25.160 |
| `vector_fft` | 12 | 18.937 | 18.916 |
| `gpu_broadcast_kernel_cartesian` | 119 | 18.651 | 18.791 |
| `gpu_solve_batched_tridiagonal_system_kernel_` | 3 | 14.458 | 14.428 |
| `gpu__pack_buffer_x_to_y_` | 3 | 13.014 | 13.020 |
| `gpu__pressure_correct_momentum_` | 3 | 11.407 | 10.712 |
| `scal_kernel_ref` | 6 | 9.338 | 9.338 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 | 8.826 | 8.833 |
| `gpu__compute_velocities_` | 3 | 8.007 | 8.014 |
| `gpu__compute_anelastic_source_term_` | 3 | 6.108 | 6.107 |
| `gpu__unpack_buffer_y_from_x_` | 3 | 5.055 | 5.042 |
| `gpu__unpack_buffer_x_from_y_` | 3 | 5.021 | 5.042 |
| `gpu__pack_buffer_y_to_x_` | 3 | 4.843 | 4.845 |
| `gpu__copy_from_buffer_` | 78 | 4.410 | 4.422 |
| `partial_mapreduce_grid` | 6 | 4.231 | 4.231 |
| `gpu__copy_real_component_` | 3 | 3.744 | 3.740 |
| `gpu__broadcast_kernel_` | 3 | 2.677 | 2.677 |
| `gpu__fill_periodic_south_and_north_halo_` | 54 | 1.466 | 1.458 |
| `gpu__fill_bottom_and_top_halo_` | 54 | 0.440 | 0.471 |
| `gpu__compute_z_bcs_` | 9 | 0.055 | 0.055 |
| **All kernels** | 473 | **471.115** | **468.974** |
