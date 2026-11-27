# Kernel time per time step: scaling comparison

GPU execution time per time step, summed over all launches of each kernel and averaged
over the captured steps. Multi-rank cases use each kernel's slowest rank.
Difference = scaled − base. Efficiency = 100 × base / scaled (weak scaling: 100% means
the kernel takes as long with the same work per GPU). A dash marks a kernel that only
one case launches. Rows are sorted by Difference ms, greatest first (a missing kernel counts as 0 ms).

## Anelastic 4 vs 8 GPUs

Base `anelastic_NCCL_4g_4x1` (4 GPUs), compared `anelastic_NCCL_8g_8x1` (8 GPUs).

| Kernel | Launches/step 4→8 GPU | 4 GPU ms/step | 8 GPU ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `ncclDevKernel_SendRecv` | 60 → 60 | 27.706 | 207.516 | +179.811 | +649.0% | 13.4 |
| `gpu__pack_buffer_x_to_y_` | 3 → 3 | 5.967 | 6.441 | +0.474 | +8.0% | 92.6 |
| `gpu_permutedims_kernel_` | 12 → 12 | 52.357 | 52.770 | +0.413 | +0.8% | 99.2 |
| `gpu_compute_potential_temperature_tendency_` | 3 → 3 | 18.139 | 18.164 | +0.025 | +0.1% | 99.9 |
| `gpu_compute_x_momentum_tendency_` | 3 → 3 | 14.478 | 14.501 | +0.023 | +0.2% | 99.8 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 → 3 | 5.756 | 5.774 | +0.018 | +0.3% | 99.7 |
| `gpu_compute_z_momentum_tendency_` | 3 → 3 | 16.380 | 16.395 | +0.015 | +0.1% | 99.9 |
| `gpu_compute_scalar_tendency_` | 3 → 3 | 14.069 | 14.079 | +0.010 | +0.1% | 99.9 |
| `gpu__unpack_buffer_x_from_y_` | 3 → 3 | 3.003 | 3.010 | +0.007 | +0.2% | 99.8 |
| `gpu_compute_y_momentum_tendency_` | 3 → 3 | 14.504 | 14.511 | +0.007 | +0.0% | 100.0 |
| `gpu__broadcast_kernel_` | 3 → 3 | 1.855 | 1.860 | +0.005 | +0.3% | 99.8 |
| `gpu__compute_anelastic_source_term_` | 3 → 3 | 2.953 | 2.958 | +0.005 | +0.2% | 99.8 |
| `partial_mapreduce_grid` | 6 → 6 | 2.421 | 2.426 | +0.004 | +0.2% | 99.8 |
| `gpu__copy_real_component_` | 3 → 3 | 1.913 | 1.916 | +0.004 | +0.2% | 99.8 |
| `gpu__pack_buffer_y_to_x_` | 3 → 3 | 2.454 | 2.458 | +0.003 | +0.1% | 99.9 |
| `gpu__ssp_rk3_substep_` | 15 → 15 | 15.837 | 15.838 | +0.002 | +0.0% | 100.0 |
| `scal_kernel_ref` | 6 → 6 | 4.500 | 4.501 | +0.001 | +0.0% | 100.0 |
| `gpu__compute_z_bcs_` | 9 → 9 | 0.031 | 0.032 | +0.001 | +3.0% | 97.1 |
| `vector_fft` | 12 → 12 | 8.590 | 8.590 | +0.000 | +0.0% | 100.0 |
| `gpu_solve_batched_tridiagonal_system_kernel_` | 3 → 3 | 7.300 | 7.297 | -0.002 | -0.0% | 100.0 |
| `gpu__pressure_correct_momentum_` | 3 → 3 | 7.767 | 7.762 | -0.005 | -0.1% | 100.1 |
| `gpu__unpack_buffer_y_from_x_` | 3 → 3 | 2.911 | 2.899 | -0.012 | -0.4% | 100.4 |
| `gpu__fill_periodic_south_and_north_halo_` | 54 → 54 | 0.898 | 0.877 | -0.021 | -2.3% | 102.4 |
| `gpu__copy_from_buffer_` | 78 → 78 | 2.095 | 2.074 | -0.021 | -1.0% | 101.0 |
| `gpu__compute_velocities_` | 3 → 3 | 5.092 | 5.055 | -0.037 | -0.7% | 100.7 |
| `gpu__fill_bottom_and_top_halo_` | 54 → 54 | 0.251 | 0.214 | -0.037 | -14.7% | 117.2 |
| `gpu_broadcast_kernel_cartesian` | 119 → 119 | 11.252 | 11.181 | -0.071 | -0.6% | 100.6 |
| **All kernels** | 473 → 473 | 250.299 | 429.979 | +179.679 | +71.8% | 58.2 |

## CE 4 vs 8 GPUs

Base `compressible_explicit_NCCL_4g_4x1` (4 GPUs), compared `compressible_explicit_NCCL_8g_8x1` (8 GPUs).

| Kernel | Launches/step 4→8 GPU | 4 GPU ms/step | 8 GPU ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `ncclDevKernel_SendRecv` | 63 → 63 | 8.916 | 69.605 | +60.689 | +680.7% | 12.8 |
| `gpu__ssp_rk3_substep_` | 18 → 18 | 18.935 | 19.016 | +0.081 | +0.4% | 99.6 |
| `gpu__compute_velocities_` | 3 → 3 | 5.396 | 5.453 | +0.057 | +1.1% | 99.0 |
| `gpu_compute_potential_temperature_tendency_` | 3 → 3 | 18.828 | 18.844 | +0.016 | +0.1% | 99.9 |
| `gpu__compute_total_density_` | 3 → 3 | 2.028 | 2.041 | +0.014 | +0.7% | 99.3 |
| `gpu_compute_scalar_tendency_` | 3 → 3 | 14.511 | 14.518 | +0.007 | +0.0% | 100.0 |
| `gpu__compute_density_tendency_` | 3 → 3 | 2.481 | 2.484 | +0.002 | +0.1% | 99.9 |
| `gpu__compute_z_bcs_` | 9 → 9 | 0.032 | 0.032 | +0.000 | +0.4% | 99.6 |
| `gpu_compute_x_momentum_tendency_` | 3 → 3 | 15.167 | 15.159 | -0.008 | -0.1% | 100.1 |
| `gpu_compute_y_momentum_tendency_` | 3 → 3 | 15.215 | 15.205 | -0.009 | -0.1% | 100.1 |
| `gpu__compute_temperature_and_pressure_` | 3 → 3 | 6.808 | 6.799 | -0.010 | -0.1% | 100.1 |
| `gpu_compute_z_momentum_tendency_` | 3 → 3 | 15.500 | 15.488 | -0.013 | -0.1% | 100.1 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 → 3 | 7.875 | 7.859 | -0.017 | -0.2% | 100.2 |
| `gpu__fill_periodic_south_and_north_halo_` | 63 → 63 | 1.042 | 1.022 | -0.020 | -1.9% | 101.9 |
| `gpu__copy_from_buffer_` | 90 → 90 | 2.429 | 2.404 | -0.026 | -1.1% | 101.1 |
| `gpu__fill_bottom_and_top_halo_` | 60 → 60 | 0.292 | 0.266 | -0.027 | -9.2% | 110.1 |
| `gpu_broadcast_kernel_cartesian` | 132 → 132 | 9.517 | 9.438 | -0.080 | -0.8% | 100.8 |
| **All kernels** | 465 → 465 | 143.755 | 205.408 | +61.653 | +42.9% | 70.0 |
