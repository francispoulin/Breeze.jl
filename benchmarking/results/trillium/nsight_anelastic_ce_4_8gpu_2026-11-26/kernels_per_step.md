# GPU kernel time per time step

Mean over the captured steps of the summed duration of every launch of each kernel;
launches with the same short name are one row. Times are GPU execution in
milliseconds; kernels on one GPU can overlap, so the column sum can exceed the
step's wall time. NCCL kernels include time spent waiting for the peer rank.

## anelastic_NCCL_4g_4x1

4 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step | Rank 1 ms/step | Rank 2 ms/step | Rank 3 ms/step |
|---|---|---|---|---|---|
| `gpu_permutedims_kernel_` | 12 | 52.355 | 52.232 | 52.207 | 52.357 |
| `ncclDevKernel_SendRecv` | 60 | 27.706 | 24.139 | 23.387 | 25.201 |
| `gpu_compute_potential_temperature_tendency_` | 3 | 18.138 | 18.052 | 18.139 | 18.126 |
| `gpu_compute_z_momentum_tendency_` | 3 | 16.373 | 16.301 | 16.380 | 16.361 |
| `gpu__ssp_rk3_substep_` | 15 | 15.837 | 15.811 | 15.772 | 15.815 |
| `gpu_compute_y_momentum_tendency_` | 3 | 14.504 | 14.410 | 14.500 | 14.489 |
| `gpu_compute_x_momentum_tendency_` | 3 | 14.474 | 14.387 | 14.478 | 14.456 |
| `gpu_compute_scalar_tendency_` | 3 | 14.069 | 13.983 | 14.065 | 14.060 |
| `gpu_broadcast_kernel_cartesian` | 119 | 11.252 | 11.192 | 11.246 | 11.241 |
| `vector_fft` | 12 | 8.526 | 8.527 | 8.590 | 8.527 |
| `gpu__pressure_correct_momentum_` | 3 | 7.757 | 7.767 | 7.744 | 7.753 |
| `gpu_solve_batched_tridiagonal_system_kernel_` | 3 | 7.298 | 7.298 | 7.298 | 7.300 |
| `gpu__pack_buffer_x_to_y_` | 3 | 5.967 | 5.902 | 5.896 | 5.962 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 | 5.756 | 5.712 | 5.752 | 5.725 |
| `gpu__compute_velocities_` | 3 | 5.016 | 5.092 | 5.053 | 5.034 |
| `scal_kernel_ref` | 6 | 4.500 | 4.498 | 4.499 | 4.499 |
| `gpu__unpack_buffer_x_from_y_` | 3 | 3.001 | 2.994 | 3.003 | 2.999 |
| `gpu__compute_anelastic_source_term_` | 3 | 2.953 | 2.951 | 2.953 | 2.952 |
| `gpu__unpack_buffer_y_from_x_` | 3 | 2.910 | 2.904 | 2.911 | 2.907 |
| `gpu__pack_buffer_y_to_x_` | 3 | 2.454 | 2.454 | 2.454 | 2.454 |
| `partial_mapreduce_grid` | 6 | 2.421 | 2.407 | 2.421 | 2.419 |
| `gpu__copy_from_buffer_` | 78 | 2.087 | 2.094 | 2.085 | 2.095 |
| `gpu__copy_real_component_` | 3 | 1.913 | 1.912 | 1.912 | 1.912 |
| `gpu__broadcast_kernel_` | 3 | 1.854 | 1.854 | 1.855 | 1.852 |
| `gpu__fill_periodic_south_and_north_halo_` | 54 | 0.898 | 0.895 | 0.897 | 0.895 |
| `gpu__fill_bottom_and_top_halo_` | 54 | 0.251 | 0.245 | 0.245 | 0.245 |
| `gpu__compute_z_bcs_` | 9 | 0.031 | 0.030 | 0.031 | 0.031 |
| **All kernels** | 473 | **250.299** | **246.043** | **245.773** | **247.667** |

## compressible_explicit_NCCL_4g_4x1

4 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step | Rank 1 ms/step | Rank 2 ms/step | Rank 3 ms/step |
|---|---|---|---|---|---|
| `gpu__ssp_rk3_substep_` | 18 | 18.935 | 18.934 | 17.758 | 18.389 |
| `gpu_compute_potential_temperature_tendency_` | 3 | 18.802 | 18.808 | 18.828 | 18.787 |
| `gpu_compute_z_momentum_tendency_` | 3 | 15.480 | 15.489 | 15.500 | 15.466 |
| `gpu_compute_y_momentum_tendency_` | 3 | 15.195 | 15.199 | 15.215 | 15.178 |
| `gpu_compute_x_momentum_tendency_` | 3 | 15.148 | 15.149 | 15.167 | 15.130 |
| `gpu_compute_scalar_tendency_` | 3 | 14.501 | 14.506 | 14.511 | 14.482 |
| `gpu_broadcast_kernel_cartesian` | 132 | 9.516 | 9.514 | 9.517 | 9.496 |
| `ncclDevKernel_SendRecv` | 63 | 7.708 | 5.259 | 8.916 | 5.432 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 | 7.875 | 7.864 | 7.839 | 7.843 |
| `gpu__compute_temperature_and_pressure_` | 3 | 6.799 | 6.800 | 6.808 | 6.791 |
| `gpu__compute_velocities_` | 3 | 5.387 | 5.394 | 5.396 | 5.391 |
| `gpu__compute_density_tendency_` | 3 | 2.477 | 2.478 | 2.481 | 2.481 |
| `gpu__copy_from_buffer_` | 90 | 2.414 | 2.422 | 2.429 | 2.423 |
| `gpu__compute_total_density_` | 3 | 2.026 | 2.027 | 2.026 | 2.028 |
| `gpu__fill_periodic_south_and_north_halo_` | 63 | 1.038 | 1.036 | 1.042 | 1.040 |
| `gpu__fill_bottom_and_top_halo_` | 60 | 0.292 | 0.286 | 0.289 | 0.289 |
| `gpu__compute_z_bcs_` | 9 | 0.031 | 0.031 | 0.032 | 0.031 |
| **All kernels** | 465 | **143.626** | **141.196** | **143.755** | **140.678** |

## anelastic_NCCL_8g_8x1

8 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step | Rank 1 ms/step | Rank 2 ms/step | Rank 3 ms/step | Rank 4 ms/step | Rank 5 ms/step | Rank 6 ms/step | Rank 7 ms/step |
|---|---|---|---|---|---|---|---|---|---|
| `ncclDevKernel_SendRecv` | 60 | 207.516 | 198.820 | 197.857 | 198.617 | 175.452 | 179.247 | 176.156 | 164.747 |
| `gpu_permutedims_kernel_` | 12 | 52.206 | 52.770 | 52.193 | 52.219 | 52.345 | 52.373 | 52.354 | 52.321 |
| `gpu_compute_potential_temperature_tendency_` | 3 | 18.136 | 18.157 | 18.157 | 18.162 | 18.164 | 18.074 | 18.152 | 18.051 |
| `gpu_compute_z_momentum_tendency_` | 3 | 16.364 | 16.382 | 16.389 | 16.392 | 16.395 | 16.314 | 16.380 | 16.290 |
| `gpu__ssp_rk3_substep_` | 15 | 15.737 | 14.603 | 15.762 | 15.816 | 15.838 | 15.796 | 15.825 | 15.831 |
| `gpu_compute_y_momentum_tendency_` | 3 | 14.495 | 14.497 | 14.508 | 14.511 | 14.504 | 14.421 | 14.495 | 14.408 |
| `gpu_compute_x_momentum_tendency_` | 3 | 14.475 | 14.481 | 14.501 | 14.496 | 14.488 | 14.408 | 14.478 | 14.392 |
| `gpu_compute_scalar_tendency_` | 3 | 14.067 | 14.074 | 14.077 | 14.079 | 14.078 | 14.000 | 14.072 | 13.988 |
| `gpu_broadcast_kernel_cartesian` | 119 | 11.169 | 11.155 | 11.168 | 11.181 | 11.181 | 11.121 | 11.159 | 11.123 |
| `vector_fft` | 12 | 8.588 | 8.515 | 8.590 | 8.526 | 8.525 | 8.526 | 8.527 | 8.526 |
| `gpu__pressure_correct_momentum_` | 3 | 7.737 | 7.347 | 7.740 | 7.758 | 7.754 | 7.757 | 7.762 | 7.760 |
| `gpu_solve_batched_tridiagonal_system_kernel_` | 3 | 7.285 | 7.291 | 7.290 | 7.297 | 7.294 | 7.293 | 7.296 | 7.291 |
| `gpu__pack_buffer_x_to_y_` | 3 | 6.282 | 6.441 | 6.275 | 6.334 | 6.348 | 6.341 | 6.345 | 6.344 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 | 5.722 | 5.734 | 5.731 | 5.765 | 5.737 | 5.730 | 5.774 | 5.736 |
| `gpu__compute_velocities_` | 3 | 5.008 | 5.006 | 5.026 | 5.055 | 5.032 | 5.040 | 5.003 | 5.011 |
| `scal_kernel_ref` | 6 | 4.499 | 4.500 | 4.500 | 4.501 | 4.500 | 4.499 | 4.501 | 4.499 |
| `gpu__unpack_buffer_x_from_y_` | 3 | 3.008 | 3.009 | 3.010 | 3.010 | 3.009 | 3.003 | 3.010 | 3.003 |
| `gpu__compute_anelastic_source_term_` | 3 | 2.957 | 2.957 | 2.956 | 2.957 | 2.958 | 2.957 | 2.958 | 2.956 |
| `gpu__unpack_buffer_y_from_x_` | 3 | 2.897 | 2.897 | 2.899 | 2.899 | 2.899 | 2.895 | 2.898 | 2.894 |
| `gpu__pack_buffer_y_to_x_` | 3 | 2.454 | 2.457 | 2.454 | 2.457 | 2.458 | 2.456 | 2.456 | 2.455 |
| `partial_mapreduce_grid` | 6 | 2.421 | 2.423 | 2.424 | 2.426 | 2.426 | 2.414 | 2.426 | 2.411 |
| `gpu__copy_from_buffer_` | 78 | 2.070 | 2.034 | 2.032 | 2.074 | 2.072 | 2.032 | 2.033 | 2.074 |
| `gpu__copy_real_component_` | 3 | 1.913 | 1.915 | 1.913 | 1.915 | 1.916 | 1.915 | 1.916 | 1.914 |
| `gpu__broadcast_kernel_` | 3 | 1.851 | 1.853 | 1.850 | 1.856 | 1.858 | 1.858 | 1.860 | 1.855 |
| `gpu__fill_periodic_south_and_north_halo_` | 54 | 0.875 | 0.871 | 0.872 | 0.875 | 0.877 | 0.873 | 0.877 | 0.876 |
| `gpu__fill_bottom_and_top_halo_` | 54 | 0.214 | 0.212 | 0.213 | 0.213 | 0.214 | 0.212 | 0.213 | 0.213 |
| `gpu__compute_z_bcs_` | 9 | 0.031 | 0.032 | 0.031 | 0.031 | 0.032 | 0.032 | 0.032 | 0.032 |
| **All kernels** | 473 | **429.979** | **420.429** | **420.418** | **421.422** | **398.351** | **401.588** | **398.959** | **386.999** |

## compressible_explicit_NCCL_8g_8x1

8 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step | Rank 1 ms/step | Rank 2 ms/step | Rank 3 ms/step | Rank 4 ms/step | Rank 5 ms/step | Rank 6 ms/step | Rank 7 ms/step |
|---|---|---|---|---|---|---|---|---|---|
| `ncclDevKernel_SendRecv` | 63 | 20.982 | 24.510 | 21.803 | 21.438 | 65.503 | 69.061 | 66.067 | 69.605 |
| `gpu__ssp_rk3_substep_` | 18 | 18.926 | 17.554 | 18.925 | 18.985 | 19.007 | 18.975 | 18.998 | 19.016 |
| `gpu_compute_potential_temperature_tendency_` | 3 | 18.817 | 18.843 | 18.832 | 18.840 | 18.838 | 18.844 | 18.796 | 18.816 |
| `gpu_compute_z_momentum_tendency_` | 3 | 15.470 | 15.474 | 15.474 | 15.479 | 15.488 | 15.483 | 15.446 | 15.464 |
| `gpu_compute_y_momentum_tendency_` | 3 | 15.194 | 15.195 | 15.199 | 15.200 | 15.205 | 15.201 | 15.163 | 15.182 |
| `gpu_compute_x_momentum_tendency_` | 3 | 15.151 | 15.147 | 15.159 | 15.151 | 15.157 | 15.152 | 15.123 | 15.134 |
| `gpu_compute_scalar_tendency_` | 3 | 14.508 | 14.510 | 14.506 | 14.518 | 14.512 | 14.506 | 14.463 | 14.485 |
| `gpu_broadcast_kernel_cartesian` | 132 | 9.428 | 9.403 | 9.409 | 9.433 | 9.438 | 9.406 | 9.389 | 9.420 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 | 7.859 | 7.835 | 7.851 | 7.838 | 7.839 | 7.826 | 7.831 | 7.819 |
| `gpu__compute_temperature_and_pressure_` | 3 | 6.794 | 6.789 | 6.795 | 6.797 | 6.799 | 6.791 | 6.784 | 6.785 |
| `gpu__compute_velocities_` | 3 | 5.442 | 5.435 | 5.434 | 5.444 | 5.441 | 5.453 | 5.440 | 5.449 |
| `gpu__compute_density_tendency_` | 3 | 2.477 | 2.483 | 2.477 | 2.479 | 2.480 | 2.484 | 2.479 | 2.479 |
| `gpu__copy_from_buffer_` | 90 | 2.397 | 2.345 | 2.350 | 2.404 | 2.397 | 2.353 | 2.347 | 2.402 |
| `gpu__compute_total_density_` | 3 | 2.033 | 2.034 | 2.033 | 2.040 | 2.041 | 2.037 | 2.035 | 2.036 |
| `gpu__fill_periodic_south_and_north_halo_` | 63 | 1.011 | 1.007 | 1.007 | 1.016 | 1.022 | 1.010 | 1.007 | 1.021 |
| `gpu__fill_bottom_and_top_halo_` | 60 | 0.263 | 0.263 | 0.262 | 0.264 | 0.266 | 0.263 | 0.262 | 0.264 |
| `gpu__compute_z_bcs_` | 9 | 0.031 | 0.032 | 0.031 | 0.031 | 0.032 | 0.032 | 0.032 | 0.032 |
| **All kernels** | 465 | **156.784** | **158.859** | **157.549** | **157.357** | **201.465** | **204.877** | **201.661** | **205.408** |
