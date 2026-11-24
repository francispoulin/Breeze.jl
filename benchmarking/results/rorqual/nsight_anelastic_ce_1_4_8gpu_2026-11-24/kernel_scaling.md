# Kernel time per time step: scaling comparison

GPU execution time per time step, summed over all launches of each kernel and averaged
over the captured steps. Multi-rank cases use each kernel's slowest rank.
Difference = scaled − base. Efficiency = 100 × base / scaled (weak scaling: 100% means
the kernel takes as long with the same work per GPU). A dash marks a kernel that only
one case launches. Rows are sorted by the scaled case's time, then the base case's.

## Anelastic 1 vs 4 GPUs

Base `anelastic_SERIAL_1g_1x1` (1 GPU), scaled `anelastic_NCCL_4g_4x1` (4 GPUs).

| Kernel | Launches/step 1→4 GPU | 1 GPU ms/step | 4 GPU ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `gpu_permutedims_kernel_` | 0 → 12 | — | 49.969 | — | — | — |
| `ncclDevKernel_SendRecv` | 0 → 60 | — | 24.554 | — | — | — |
| `gpu_compute_potential_temperature_tendency_` | 3 → 3 | 18.101 | 18.095 | -0.005 | -0.0% | 100.0 |
| `gpu_compute_z_momentum_tendency_` | 3 → 3 | 16.333 | 16.307 | -0.025 | -0.2% | 100.2 |
| `gpu__ssp_rk3_substep_` | 15 → 15 | 15.825 | 16.143 | +0.318 | +2.0% | 98.0 |
| `gpu_compute_y_momentum_tendency_` | 3 → 3 | 14.486 | 14.464 | -0.022 | -0.2% | 100.2 |
| `gpu_compute_x_momentum_tendency_` | 3 → 3 | 14.451 | 14.440 | -0.010 | -0.1% | 100.1 |
| `gpu_compute_scalar_tendency_` | 3 → 3 | 14.049 | 14.039 | -0.009 | -0.1% | 100.1 |
| `gpu_broadcast_kernel_cartesian` | 5 → 119 | 5.868 | 13.140 | +7.272 | +123.9% | 44.7 |
| `vector_fft` | 6 → 12 | 4.272 | 8.583 | +4.311 | +100.9% | 49.8 |
| `gpu__pressure_correct_momentum_` | 3 → 3 | 7.729 | 7.872 | +0.143 | +1.9% | 98.2 |
| `gpu_solve_batched_tridiagonal_system_kernel_` | 3 → 3 | 7.351 | 7.419 | +0.069 | +0.9% | 99.1 |
| `gpu__pack_buffer_x_to_y_` | 0 → 3 | — | 6.208 | — | — | — |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 → 3 | 5.734 | 6.118 | +0.384 | +6.7% | 93.7 |
| `gpu__compute_velocities_` | 3 → 3 | 5.041 | 5.574 | +0.533 | +10.6% | 90.4 |
| `scal_kernel_ref` | 3 → 6 | 2.250 | 4.520 | +2.270 | +100.9% | 49.8 |
| `gpu__copy_from_buffer_` | 0 → 78 | — | 3.992 | — | — | — |
| `gpu__unpack_buffer_x_from_y_` | 0 → 3 | — | 2.998 | — | — | — |
| `gpu__compute_anelastic_source_term_` | 3 → 3 | 2.961 | 2.983 | +0.022 | +0.7% | 99.3 |
| `gpu__unpack_buffer_y_from_x_` | 0 → 3 | — | 2.865 | — | — | — |
| `gpu__pack_buffer_y_to_x_` | 0 → 3 | — | 2.472 | — | — | — |
| `partial_mapreduce_grid` | 3 → 6 | 0.010 | 2.418 | +2.408 | +24371.5% | 0.4 |
| `gpu__copy_real_component_` | 0 → 3 | — | 2.022 | — | — | — |
| `gpu__broadcast_kernel_` | 0 → 3 | — | 1.874 | — | — | — |
| `gpu__fill_periodic_south_and_north_halo_` | 51 → 54 | 0.930 | 0.506 | -0.424 | -45.6% | 183.8 |
| `gpu__fill_bottom_and_top_halo_` | 51 → 54 | 0.207 | 0.237 | +0.030 | +14.6% | 87.3 |
| `gpu__compute_z_bcs_` | 9 → 9 | 0.030 | 0.030 | -0.000 | -0.9% | 100.9 |
| `regular_fft` | 6 → 0 | 4.775 | — | — | — | — |
| `gpu__fill_periodic_west_and_east_halo_` | 51 → 0 | 4.203 | — | — | — | — |
| `gpu_copy_real_component_` | 3 → 0 | 1.923 | — | — | — | — |
| `gpu_fill_kernel_` | 3 → 0 | 0.978 | — | — | — | — |
| `gpu_broadcast_kernel_linear` | 3 → 0 | 0.008 | — | — | — | — |
| **All kernels** | 239 → 473 | 147.514 | 249.704 | +102.190 | +69.3% | 59.1 |

## Anelastic 1 vs 8 GPUs

Base `anelastic_SERIAL_1g_1x1` (1 GPU), scaled `anelastic_NCCL_8g_8x1` (8 GPUs).

| Kernel | Launches/step 1→8 GPU | 1 GPU ms/step | 8 GPU ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `ncclDevKernel_SendRecv` | 0 → 60 | — | 594.352 | — | — | — |
| `gpu_permutedims_kernel_` | 0 → 12 | — | 52.336 | — | — | — |
| `gpu_compute_potential_temperature_tendency_` | 3 → 3 | 18.101 | 18.170 | +0.069 | +0.4% | 99.6 |
| `gpu_compute_z_momentum_tendency_` | 3 → 3 | 16.333 | 16.400 | +0.068 | +0.4% | 99.6 |
| `gpu__ssp_rk3_substep_` | 15 → 15 | 15.825 | 15.941 | +0.116 | +0.7% | 99.3 |
| `gpu_compute_y_momentum_tendency_` | 3 → 3 | 14.486 | 14.506 | +0.019 | +0.1% | 99.9 |
| `gpu_compute_x_momentum_tendency_` | 3 → 3 | 14.451 | 14.488 | +0.038 | +0.3% | 99.7 |
| `gpu_compute_scalar_tendency_` | 3 → 3 | 14.049 | 14.080 | +0.032 | +0.2% | 99.8 |
| `gpu_broadcast_kernel_cartesian` | 5 → 119 | 5.868 | 11.180 | +5.312 | +90.5% | 52.5 |
| `vector_fft` | 6 → 12 | 4.272 | 8.608 | +4.336 | +101.5% | 49.6 |
| `gpu__pressure_correct_momentum_` | 3 → 3 | 7.729 | 7.781 | +0.051 | +0.7% | 99.3 |
| `gpu_solve_batched_tridiagonal_system_kernel_` | 3 → 3 | 7.351 | 7.408 | +0.058 | +0.8% | 99.2 |
| `gpu__pack_buffer_x_to_y_` | 0 → 3 | — | 6.357 | — | — | — |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 → 3 | 5.734 | 5.799 | +0.065 | +1.1% | 98.9 |
| `gpu__compute_velocities_` | 3 → 3 | 5.041 | 5.064 | +0.023 | +0.5% | 99.5 |
| `scal_kernel_ref` | 3 → 6 | 2.250 | 4.520 | +2.270 | +100.9% | 49.8 |
| `gpu__unpack_buffer_x_from_y_` | 0 → 3 | — | 3.014 | — | — | — |
| `gpu__compute_anelastic_source_term_` | 3 → 3 | 2.961 | 2.991 | +0.030 | +1.0% | 99.0 |
| `gpu__unpack_buffer_y_from_x_` | 0 → 3 | — | 2.912 | — | — | — |
| `gpu__pack_buffer_y_to_x_` | 0 → 3 | — | 2.474 | — | — | — |
| `partial_mapreduce_grid` | 3 → 6 | 0.010 | 2.427 | +2.417 | +24464.7% | 0.4 |
| `gpu__copy_from_buffer_` | 0 → 78 | — | 2.108 | — | — | — |
| `gpu__copy_real_component_` | 0 → 3 | — | 1.933 | — | — | — |
| `gpu__broadcast_kernel_` | 0 → 3 | — | 1.863 | — | — | — |
| `gpu__fill_periodic_south_and_north_halo_` | 51 → 54 | 0.930 | 0.878 | -0.051 | -5.5% | 105.9 |
| `gpu__fill_bottom_and_top_halo_` | 51 → 54 | 0.207 | 0.215 | +0.008 | +4.0% | 96.2 |
| `gpu__compute_z_bcs_` | 9 → 9 | 0.030 | 0.032 | +0.002 | +5.3% | 94.9 |
| `regular_fft` | 6 → 0 | 4.775 | — | — | — | — |
| `gpu__fill_periodic_west_and_east_halo_` | 51 → 0 | 4.203 | — | — | — | — |
| `gpu_copy_real_component_` | 3 → 0 | 1.923 | — | — | — | — |
| `gpu_fill_kernel_` | 3 → 0 | 0.978 | — | — | — | — |
| `gpu_broadcast_kernel_linear` | 3 → 0 | 0.008 | — | — | — | — |
| **All kernels** | 239 → 473 | 147.514 | 817.276 | +669.762 | +454.0% | 18.0 |

## CE 1 vs 4 GPUs

Base `compressible_explicit_SERIAL_1g_1x1` (1 GPU), scaled `compressible_explicit_NCCL_4g_4x1` (4 GPUs).

| Kernel | Launches/step 1→4 GPU | 1 GPU ms/step | 4 GPU ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `gpu__ssp_rk3_substep_` | 18 → 18 | 18.995 | 19.360 | +0.365 | +1.9% | 98.1 |
| `gpu_compute_potential_temperature_tendency_` | 3 → 3 | 18.759 | 18.775 | +0.016 | +0.1% | 99.9 |
| `gpu_compute_z_momentum_tendency_` | 3 → 3 | 15.442 | 15.440 | -0.002 | -0.0% | 100.0 |
| `gpu_compute_y_momentum_tendency_` | 3 → 3 | 15.151 | 15.155 | +0.004 | +0.0% | 100.0 |
| `gpu_compute_x_momentum_tendency_` | 3 → 3 | 15.102 | 15.118 | +0.016 | +0.1% | 99.9 |
| `gpu_compute_scalar_tendency_` | 3 → 3 | 14.455 | 14.472 | +0.017 | +0.1% | 99.9 |
| `gpu_broadcast_kernel_cartesian` | 6 → 132 | 7.029 | 11.722 | +4.693 | +66.8% | 60.0 |
| `ncclDevKernel_SendRecv` | 0 → 63 | — | 9.402 | — | — | — |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 → 3 | 7.802 | 8.006 | +0.205 | +2.6% | 97.4 |
| `gpu__compute_temperature_and_pressure_` | 3 → 3 | 6.781 | 6.841 | +0.061 | +0.9% | 99.1 |
| `gpu__compute_velocities_` | 3 → 3 | 5.417 | 6.026 | +0.609 | +11.2% | 89.9 |
| `gpu__copy_from_buffer_` | 0 → 90 | — | 4.598 | — | — | — |
| `gpu__compute_density_tendency_` | 3 → 3 | 2.481 | 2.642 | +0.161 | +6.5% | 93.9 |
| `gpu__compute_total_density_` | 3 → 3 | 2.024 | 2.140 | +0.116 | +5.8% | 94.6 |
| `gpu__fill_periodic_south_and_north_halo_` | 57 → 63 | 1.075 | 0.578 | -0.497 | -46.2% | 185.9 |
| `gpu__fill_bottom_and_top_halo_` | 54 → 60 | 0.248 | 0.292 | +0.044 | +17.8% | 84.9 |
| `gpu__compute_z_bcs_` | 9 → 9 | 0.031 | 0.032 | +0.001 | +4.4% | 95.8 |
| `gpu__fill_periodic_west_and_east_halo_` | 57 → 0 | 4.978 | — | — | — | — |
| **All kernels** | 231 → 465 | 135.769 | 150.250 | +14.481 | +10.7% | 90.4 |

## CE 1 vs 8 GPUs

Base `compressible_explicit_SERIAL_1g_1x1` (1 GPU), scaled `compressible_explicit_NCCL_8g_8x1` (8 GPUs).

| Kernel | Launches/step 1→8 GPU | 1 GPU ms/step | 8 GPU ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `ncclDevKernel_SendRecv` | 0 → 63 | — | 36.125 | — | — | — |
| `gpu__ssp_rk3_substep_` | 18 → 18 | 18.995 | 19.149 | +0.154 | +0.8% | 99.2 |
| `gpu_compute_potential_temperature_tendency_` | 3 → 3 | 18.759 | 18.860 | +0.101 | +0.5% | 99.5 |
| `gpu_compute_z_momentum_tendency_` | 3 → 3 | 15.442 | 15.495 | +0.053 | +0.3% | 99.7 |
| `gpu_compute_y_momentum_tendency_` | 3 → 3 | 15.151 | 15.214 | +0.063 | +0.4% | 99.6 |
| `gpu_compute_x_momentum_tendency_` | 3 → 3 | 15.102 | 15.166 | +0.064 | +0.4% | 99.6 |
| `gpu_compute_scalar_tendency_` | 3 → 3 | 14.455 | 14.520 | +0.065 | +0.5% | 99.6 |
| `gpu_broadcast_kernel_cartesian` | 6 → 132 | 7.029 | 9.436 | +2.407 | +34.2% | 74.5 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 → 3 | 7.802 | 7.881 | +0.079 | +1.0% | 99.0 |
| `gpu__compute_temperature_and_pressure_` | 3 → 3 | 6.781 | 6.820 | +0.039 | +0.6% | 99.4 |
| `gpu__compute_velocities_` | 3 → 3 | 5.417 | 5.497 | +0.080 | +1.5% | 98.6 |
| `gpu__compute_density_tendency_` | 3 → 3 | 2.481 | 2.511 | +0.030 | +1.2% | 98.8 |
| `gpu__copy_from_buffer_` | 0 → 90 | — | 2.428 | — | — | — |
| `gpu__compute_total_density_` | 3 → 3 | 2.024 | 2.053 | +0.029 | +1.4% | 98.6 |
| `gpu__fill_periodic_south_and_north_halo_` | 57 → 63 | 1.075 | 1.018 | -0.057 | -5.3% | 105.6 |
| `gpu__fill_bottom_and_top_halo_` | 54 → 60 | 0.248 | 0.266 | +0.017 | +7.0% | 93.4 |
| `gpu__compute_z_bcs_` | 9 → 9 | 0.031 | 0.032 | +0.001 | +3.8% | 96.4 |
| `gpu__fill_periodic_west_and_east_halo_` | 57 → 0 | 4.978 | — | — | — | — |
| **All kernels** | 231 → 465 | 135.769 | 172.262 | +36.493 | +26.9% | 78.8 |
