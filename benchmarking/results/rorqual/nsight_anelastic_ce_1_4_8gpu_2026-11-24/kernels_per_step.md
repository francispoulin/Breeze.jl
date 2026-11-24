# GPU kernel time per time step

Mean over the captured steps of the summed duration of every launch of each kernel;
launches with the same short name are one row. Times are GPU execution in
milliseconds; kernels on one GPU can overlap, so the column sum can exceed the
step's wall time. NCCL kernels include time spent waiting for the peer rank.

## anelastic_SERIAL_1g_1x1

1 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step |
|---|---|---|
| `gpu_compute_potential_temperature_tendency_` | 3 | 18.101 |
| `gpu_compute_z_momentum_tendency_` | 3 | 16.333 |
| `gpu__ssp_rk3_substep_` | 15 | 15.825 |
| `gpu_compute_y_momentum_tendency_` | 3 | 14.486 |
| `gpu_compute_x_momentum_tendency_` | 3 | 14.451 |
| `gpu_compute_scalar_tendency_` | 3 | 14.049 |
| `gpu__pressure_correct_momentum_` | 3 | 7.729 |
| `gpu_solve_batched_tridiagonal_system_kernel_` | 3 | 7.351 |
| `gpu_broadcast_kernel_cartesian` | 5 | 5.868 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 | 5.734 |
| `gpu__compute_velocities_` | 3 | 5.041 |
| `regular_fft` | 6 | 4.775 |
| `vector_fft` | 6 | 4.272 |
| `gpu__fill_periodic_west_and_east_halo_` | 51 | 4.203 |
| `gpu__compute_anelastic_source_term_` | 3 | 2.961 |
| `scal_kernel_ref` | 3 | 2.250 |
| `gpu_copy_real_component_` | 3 | 1.923 |
| `gpu_fill_kernel_` | 3 | 0.978 |
| `gpu__fill_periodic_south_and_north_halo_` | 51 | 0.930 |
| `gpu__fill_bottom_and_top_halo_` | 51 | 0.207 |
| `gpu__compute_z_bcs_` | 9 | 0.030 |
| `partial_mapreduce_grid` | 3 | 0.010 |
| `gpu_broadcast_kernel_linear` | 3 | 0.008 |
| **All kernels** | 239 | **147.514** |

## compressible_explicit_SERIAL_1g_1x1

1 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step |
|---|---|---|
| `gpu__ssp_rk3_substep_` | 18 | 18.995 |
| `gpu_compute_potential_temperature_tendency_` | 3 | 18.759 |
| `gpu_compute_z_momentum_tendency_` | 3 | 15.442 |
| `gpu_compute_y_momentum_tendency_` | 3 | 15.151 |
| `gpu_compute_x_momentum_tendency_` | 3 | 15.102 |
| `gpu_compute_scalar_tendency_` | 3 | 14.455 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 | 7.802 |
| `gpu_broadcast_kernel_cartesian` | 6 | 7.029 |
| `gpu__compute_temperature_and_pressure_` | 3 | 6.781 |
| `gpu__compute_velocities_` | 3 | 5.417 |
| `gpu__fill_periodic_west_and_east_halo_` | 57 | 4.978 |
| `gpu__compute_density_tendency_` | 3 | 2.481 |
| `gpu__compute_total_density_` | 3 | 2.024 |
| `gpu__fill_periodic_south_and_north_halo_` | 57 | 1.075 |
| `gpu__fill_bottom_and_top_halo_` | 54 | 0.248 |
| `gpu__compute_z_bcs_` | 9 | 0.031 |
| **All kernels** | 231 | **135.769** |

## anelastic_NCCL_4g_4x1

4 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step | Rank 1 ms/step | Rank 2 ms/step | Rank 3 ms/step |
|---|---|---|---|---|---|
| `gpu_permutedims_kernel_` | 12 | 49.911 | 49.924 | 49.969 | 49.954 |
| `ncclDevKernel_SendRecv` | 60 | 24.071 | 23.955 | 24.554 | 23.886 |
| `gpu_compute_potential_temperature_tendency_` | 3 | 18.044 | 18.076 | 18.095 | 18.055 |
| `gpu_compute_z_momentum_tendency_` | 3 | 16.267 | 16.307 | 16.305 | 16.281 |
| `gpu__ssp_rk3_substep_` | 15 | 16.143 | 16.112 | 16.140 | 16.130 |
| `gpu_compute_y_momentum_tendency_` | 3 | 14.454 | 14.464 | 14.462 | 14.457 |
| `gpu_compute_x_momentum_tendency_` | 3 | 14.412 | 14.440 | 14.436 | 14.422 |
| `gpu_compute_scalar_tendency_` | 3 | 14.015 | 14.029 | 14.039 | 14.019 |
| `gpu_broadcast_kernel_cartesian` | 119 | 13.135 | 13.140 | 13.138 | 13.127 |
| `vector_fft` | 12 | 8.581 | 8.582 | 8.583 | 8.580 |
| `gpu__pressure_correct_momentum_` | 3 | 7.870 | 7.870 | 7.872 | 7.869 |
| `gpu_solve_batched_tridiagonal_system_kernel_` | 3 | 7.413 | 7.419 | 7.413 | 7.417 |
| `gpu__pack_buffer_x_to_y_` | 3 | 6.205 | 6.208 | 6.205 | 6.207 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 | 6.008 | 6.001 | 6.037 | 6.118 |
| `gpu__compute_velocities_` | 3 | 5.574 | 5.561 | 5.571 | 5.563 |
| `scal_kernel_ref` | 6 | 4.520 | 4.519 | 4.520 | 4.520 |
| `gpu__copy_from_buffer_` | 78 | 3.989 | 3.988 | 3.974 | 3.992 |
| `gpu__unpack_buffer_x_from_y_` | 3 | 2.998 | 2.998 | 2.998 | 2.998 |
| `gpu__compute_anelastic_source_term_` | 3 | 2.983 | 2.983 | 2.983 | 2.982 |
| `gpu__unpack_buffer_y_from_x_` | 3 | 2.865 | 2.865 | 2.865 | 2.865 |
| `gpu__pack_buffer_y_to_x_` | 3 | 2.471 | 2.472 | 2.472 | 2.471 |
| `partial_mapreduce_grid` | 6 | 2.418 | 2.418 | 2.416 | 2.418 |
| `gpu__copy_real_component_` | 3 | 2.022 | 2.022 | 2.022 | 2.022 |
| `gpu__broadcast_kernel_` | 3 | 1.870 | 1.874 | 1.869 | 1.870 |
| `gpu__fill_periodic_south_and_north_halo_` | 54 | 0.502 | 0.503 | 0.501 | 0.506 |
| `gpu__fill_bottom_and_top_halo_` | 54 | 0.235 | 0.234 | 0.233 | 0.237 |
| `gpu__compute_z_bcs_` | 9 | 0.030 | 0.030 | 0.030 | 0.030 |
| **All kernels** | 473 | **249.008** | **248.996** | **249.704** | **248.997** |

## compressible_explicit_NCCL_4g_4x1

4 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step | Rank 1 ms/step | Rank 2 ms/step | Rank 3 ms/step |
|---|---|---|---|---|---|
| `gpu__ssp_rk3_substep_` | 18 | 19.360 | 19.273 | 19.202 | 19.208 |
| `gpu_compute_potential_temperature_tendency_` | 3 | 18.775 | 18.702 | 18.705 | 18.771 |
| `gpu_compute_z_momentum_tendency_` | 3 | 15.440 | 15.352 | 15.361 | 15.431 |
| `gpu_compute_y_momentum_tendency_` | 3 | 15.155 | 15.065 | 15.077 | 15.147 |
| `gpu_compute_x_momentum_tendency_` | 3 | 15.118 | 15.036 | 15.045 | 15.111 |
| `gpu_compute_scalar_tendency_` | 3 | 14.471 | 14.377 | 14.388 | 14.472 |
| `gpu_broadcast_kernel_cartesian` | 132 | 11.722 | 11.644 | 11.619 | 11.704 |
| `ncclDevKernel_SendRecv` | 63 | 7.815 | 8.955 | 8.792 | 9.402 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 | 8.006 | 7.983 | 7.967 | 7.994 |
| `gpu__compute_temperature_and_pressure_` | 3 | 6.841 | 6.810 | 6.802 | 6.828 |
| `gpu__compute_velocities_` | 3 | 6.019 | 6.026 | 5.983 | 5.979 |
| `gpu__copy_from_buffer_` | 90 | 4.573 | 4.598 | 4.586 | 4.586 |
| `gpu__compute_density_tendency_` | 3 | 2.639 | 2.642 | 2.599 | 2.600 |
| `gpu__compute_total_density_` | 3 | 2.140 | 2.130 | 2.109 | 2.115 |
| `gpu__fill_periodic_south_and_north_halo_` | 63 | 0.578 | 0.577 | 0.569 | 0.577 |
| `gpu__fill_bottom_and_top_halo_` | 60 | 0.292 | 0.288 | 0.282 | 0.291 |
| `gpu__compute_z_bcs_` | 9 | 0.032 | 0.032 | 0.032 | 0.032 |
| **All kernels** | 465 | **148.978** | **149.490** | **149.117** | **150.250** |

## anelastic_NCCL_8g_8x1

8 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step | Rank 1 ms/step | Rank 2 ms/step | Rank 3 ms/step | Rank 4 ms/step | Rank 5 ms/step | Rank 6 ms/step | Rank 7 ms/step |
|---|---|---|---|---|---|---|---|---|---|
| `ncclDevKernel_SendRecv` | 60 | 555.537 | 554.592 | 555.988 | 555.415 | 594.352 | 593.913 | 593.958 | 590.383 |
| `gpu_permutedims_kernel_` | 12 | 52.336 | 52.311 | 52.299 | 52.307 | 52.317 | 52.321 | 52.296 | 52.299 |
| `gpu_compute_potential_temperature_tendency_` | 3 | 18.136 | 18.170 | 18.084 | 18.161 | 18.105 | 18.162 | 18.157 | 18.146 |
| `gpu_compute_z_momentum_tendency_` | 3 | 16.384 | 16.400 | 16.324 | 16.392 | 16.361 | 16.394 | 16.383 | 16.378 |
| `gpu__ssp_rk3_substep_` | 15 | 15.917 | 15.886 | 15.820 | 15.822 | 15.927 | 15.929 | 15.941 | 15.931 |
| `gpu_compute_y_momentum_tendency_` | 3 | 14.502 | 14.506 | 14.429 | 14.503 | 14.468 | 14.505 | 14.497 | 14.499 |
| `gpu_compute_x_momentum_tendency_` | 3 | 14.482 | 14.488 | 14.418 | 14.486 | 14.452 | 14.487 | 14.477 | 14.476 |
| `gpu_compute_scalar_tendency_` | 3 | 14.078 | 14.080 | 14.007 | 14.078 | 14.045 | 14.078 | 14.076 | 14.072 |
| `gpu_broadcast_kernel_cartesian` | 119 | 11.174 | 11.177 | 11.120 | 11.163 | 11.145 | 11.180 | 11.168 | 11.167 |
| `vector_fft` | 12 | 8.606 | 8.604 | 8.527 | 8.527 | 8.607 | 8.608 | 8.607 | 8.608 |
| `gpu__pressure_correct_momentum_` | 3 | 7.777 | 7.769 | 7.751 | 7.752 | 7.779 | 7.776 | 7.773 | 7.781 |
| `gpu_solve_batched_tridiagonal_system_kernel_` | 3 | 7.408 | 7.404 | 7.290 | 7.299 | 7.406 | 7.402 | 7.405 | 7.406 |
| `gpu__pack_buffer_x_to_y_` | 3 | 6.285 | 6.357 | 6.339 | 6.277 | 6.287 | 6.355 | 6.357 | 6.282 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 | 5.726 | 5.738 | 5.760 | 5.722 | 5.742 | 5.751 | 5.779 | 5.799 |
| `gpu__compute_velocities_` | 3 | 5.037 | 5.064 | 5.018 | 4.994 | 5.040 | 5.054 | 5.027 | 5.034 |
| `scal_kernel_ref` | 6 | 4.519 | 4.519 | 4.500 | 4.499 | 4.518 | 4.520 | 4.520 | 4.519 |
| `gpu__unpack_buffer_x_from_y_` | 3 | 3.009 | 3.014 | 3.001 | 3.000 | 3.007 | 3.014 | 3.010 | 3.009 |
| `gpu__compute_anelastic_source_term_` | 3 | 2.985 | 2.991 | 2.956 | 2.957 | 2.984 | 2.990 | 2.985 | 2.985 |
| `gpu__unpack_buffer_y_from_x_` | 3 | 2.899 | 2.912 | 2.892 | 2.885 | 2.896 | 2.911 | 2.908 | 2.898 |
| `gpu__pack_buffer_y_to_x_` | 3 | 2.472 | 2.470 | 2.455 | 2.458 | 2.474 | 2.469 | 2.469 | 2.474 |
| `partial_mapreduce_grid` | 6 | 2.426 | 2.427 | 2.413 | 2.425 | 2.421 | 2.425 | 2.427 | 2.426 |
| `gpu__copy_from_buffer_` | 78 | 2.031 | 2.107 | 2.030 | 2.025 | 2.032 | 2.108 | 2.033 | 2.031 |
| `gpu__copy_real_component_` | 3 | 1.932 | 1.933 | 1.915 | 1.915 | 1.931 | 1.933 | 1.933 | 1.932 |
| `gpu__broadcast_kernel_` | 3 | 1.863 | 1.858 | 1.856 | 1.860 | 1.860 | 1.860 | 1.861 | 1.861 |
| `gpu__fill_periodic_south_and_north_halo_` | 54 | 0.877 | 0.878 | 0.875 | 0.875 | 0.875 | 0.877 | 0.874 | 0.875 |
| `gpu__fill_bottom_and_top_halo_` | 54 | 0.214 | 0.215 | 0.214 | 0.212 | 0.214 | 0.213 | 0.214 | 0.212 |
| `gpu__compute_z_bcs_` | 9 | 0.032 | 0.032 | 0.032 | 0.032 | 0.032 | 0.032 | 0.032 | 0.031 |
| **All kernels** | 473 | **778.646** | **777.903** | **778.312** | **778.042** | **817.276** | **817.266** | **817.167** | **813.512** |

## compressible_explicit_NCCL_8g_8x1

8 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step | Rank 1 ms/step | Rank 2 ms/step | Rank 3 ms/step | Rank 4 ms/step | Rank 5 ms/step | Rank 6 ms/step | Rank 7 ms/step |
|---|---|---|---|---|---|---|---|---|---|
| `ncclDevKernel_SendRecv` | 63 | 29.545 | 30.273 | 29.038 | 29.230 | 36.125 | 35.672 | 35.085 | 36.123 |
| `gpu__ssp_rk3_substep_` | 18 | 19.149 | 19.086 | 19.024 | 19.013 | 19.137 | 19.139 | 19.148 | 19.144 |
| `gpu_compute_potential_temperature_tendency_` | 3 | 18.844 | 18.763 | 18.758 | 18.831 | 18.853 | 18.860 | 18.845 | 18.856 |
| `gpu_compute_z_momentum_tendency_` | 3 | 15.489 | 15.407 | 15.417 | 15.480 | 15.483 | 15.488 | 15.491 | 15.495 |
| `gpu_compute_y_momentum_tendency_` | 3 | 15.208 | 15.123 | 15.132 | 15.199 | 15.203 | 15.207 | 15.212 | 15.214 |
| `gpu_compute_x_momentum_tendency_` | 3 | 15.158 | 15.075 | 15.085 | 15.149 | 15.154 | 15.161 | 15.166 | 15.166 |
| `gpu_compute_scalar_tendency_` | 3 | 14.515 | 14.432 | 14.434 | 14.505 | 14.509 | 14.512 | 14.515 | 14.520 |
| `gpu_broadcast_kernel_cartesian` | 132 | 9.415 | 9.379 | 9.370 | 9.409 | 9.398 | 9.436 | 9.413 | 9.418 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 | 7.840 | 7.790 | 7.804 | 7.828 | 7.856 | 7.881 | 7.852 | 7.834 |
| `gpu__compute_temperature_and_pressure_` | 3 | 6.815 | 6.791 | 6.775 | 6.795 | 6.812 | 6.820 | 6.816 | 6.810 |
| `gpu__compute_velocities_` | 3 | 5.484 | 5.497 | 5.438 | 5.442 | 5.477 | 5.481 | 5.480 | 5.477 |
| `gpu__compute_density_tendency_` | 3 | 2.510 | 2.511 | 2.479 | 2.479 | 2.508 | 2.508 | 2.509 | 2.509 |
| `gpu__copy_from_buffer_` | 90 | 2.339 | 2.428 | 2.338 | 2.343 | 2.345 | 2.418 | 2.340 | 2.339 |
| `gpu__compute_total_density_` | 3 | 2.050 | 2.053 | 2.033 | 2.034 | 2.049 | 2.047 | 2.049 | 2.048 |
| `gpu__fill_periodic_south_and_north_halo_` | 63 | 1.009 | 1.016 | 1.007 | 1.007 | 1.006 | 1.018 | 1.011 | 1.015 |
| `gpu__fill_bottom_and_top_halo_` | 60 | 0.263 | 0.266 | 0.260 | 0.263 | 0.260 | 0.265 | 0.261 | 0.263 |
| `gpu__compute_z_bcs_` | 9 | 0.032 | 0.031 | 0.032 | 0.031 | 0.032 | 0.032 | 0.032 | 0.032 |
| **All kernels** | 465 | **165.666** | **165.923** | **164.422** | **165.038** | **172.207** | **171.946** | **171.223** | **172.262** |
