# Kernel time per time step: scaling comparison

GPU execution time per time step, summed over all launches of each kernel and averaged
over the captured steps. Multi-rank cases use each kernel's slowest rank.
Difference = scaled − base. Efficiency = 100 × base / scaled (weak scaling: 100% means
the kernel takes as long with the same work per GPU). A dash marks a kernel that only
one case launches. Rows are sorted by the scaled case's time, then the base case's.

## Anelastic

Base `anelastic_1g` (1 GPU), scaled `anelastic_2g` (2 GPUs).

| Kernel | Launches/step 1→2 GPU | 1 GPU ms/step | 2 GPU ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `gpu_permutedims_kernel_` | 0 → 12 | — | 74.157 | — | — | — |
| `ncclDevKernel_SendRecv` | 0 → 60 | — | 60.809 | — | — | — |
| `gpu_compute_potential_temperature_tendency_` | 3 → 3 | 36.925 | 37.035 | +0.111 | +0.3% | 99.7 |
| `gpu_compute_z_momentum_tendency_` | 3 → 3 | 35.390 | 35.428 | +0.038 | +0.1% | 99.9 |
| `gpu_compute_y_momentum_tendency_` | 3 → 3 | 32.835 | 32.849 | +0.014 | +0.0% | 100.0 |
| `gpu_compute_scalar_tendency_` | 3 → 3 | 32.689 | 32.724 | +0.035 | +0.1% | 99.9 |
| `gpu_compute_x_momentum_tendency_` | 3 → 3 | 32.573 | 32.608 | +0.035 | +0.1% | 99.9 |
| `gpu__ssp_rk3_substep_` | 15 → 15 | 25.178 | 25.176 | -0.003 | -0.0% | 100.0 |
| `vector_fft` | 6 → 12 | 9.493 | 18.937 | +9.444 | +99.5% | 50.1 |
| `gpu_broadcast_kernel_cartesian` | 5 → 119 | 9.478 | 18.791 | +9.313 | +98.3% | 50.4 |
| `gpu_solve_batched_tridiagonal_system_kernel_` | 3 → 3 | 14.467 | 14.458 | -0.009 | -0.1% | 100.1 |
| `gpu__pack_buffer_x_to_y_` | 0 → 3 | — | 13.020 | — | — | — |
| `gpu__pressure_correct_momentum_` | 3 → 3 | 10.704 | 11.407 | +0.703 | +6.6% | 93.8 |
| `scal_kernel_ref` | 3 → 6 | 4.667 | 9.338 | +4.671 | +100.1% | 50.0 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 → 3 | 8.682 | 8.833 | +0.151 | +1.7% | 98.3 |
| `gpu__compute_velocities_` | 3 → 3 | 8.093 | 8.014 | -0.079 | -1.0% | 101.0 |
| `gpu__compute_anelastic_source_term_` | 3 → 3 | 6.145 | 6.108 | -0.037 | -0.6% | 100.6 |
| `gpu__unpack_buffer_y_from_x_` | 0 → 3 | — | 5.055 | — | — | — |
| `gpu__unpack_buffer_x_from_y_` | 0 → 3 | — | 5.042 | — | — | — |
| `gpu__pack_buffer_y_to_x_` | 0 → 3 | — | 4.845 | — | — | — |
| `gpu__copy_from_buffer_` | 0 → 78 | — | 4.422 | — | — | — |
| `partial_mapreduce_grid` | 3 → 6 | 0.015 | 4.231 | +4.216 | +27855.7% | 0.4 |
| `gpu__copy_real_component_` | 0 → 3 | — | 3.744 | — | — | — |
| `gpu__broadcast_kernel_` | 0 → 3 | — | 2.677 | — | — | — |
| `gpu__fill_periodic_south_and_north_halo_` | 51 → 54 | 1.484 | 1.466 | -0.018 | -1.2% | 101.2 |
| `gpu__fill_bottom_and_top_halo_` | 51 → 54 | 0.388 | 0.471 | +0.082 | +21.2% | 82.5 |
| `gpu__compute_z_bcs_` | 9 → 9 | 0.054 | 0.055 | +0.001 | +1.6% | 98.4 |
| `gpu__fill_periodic_west_and_east_halo_` | 51 → 0 | 11.008 | — | — | — | — |
| `regular_fft` | 6 → 0 | 9.449 | — | — | — | — |
| `gpu_copy_real_component_` | 3 → 0 | 3.779 | — | — | — | — |
| `gpu_fill_kernel_` | 3 → 0 | 2.073 | — | — | — | — |
| `gpu_broadcast_kernel_linear` | 3 → 0 | 0.015 | — | — | — | — |
| **All kernels** | 239 → 473 | 295.583 | 471.115 | +175.531 | +59.4% | 62.7 |

## Compressible explicit (CE)

Base `explicit_1g` (1 GPU), scaled `explicit_2g` (2 GPUs).

| Kernel | Launches/step 1→2 GPU | 1 GPU ms/step | 2 GPU ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `gpu_compute_potential_temperature_tendency_` | 3 → 3 | 39.153 | 39.126 | -0.027 | -0.1% | 100.1 |
| `gpu_compute_z_momentum_tendency_` | 3 → 3 | 34.088 | 34.138 | +0.051 | +0.1% | 99.9 |
| `gpu_compute_y_momentum_tendency_` | 3 → 3 | 33.462 | 33.496 | +0.034 | +0.1% | 99.9 |
| `gpu_compute_x_momentum_tendency_` | 3 → 3 | 33.077 | 33.096 | +0.019 | +0.1% | 99.9 |
| `gpu_compute_scalar_tendency_` | 3 → 3 | 33.056 | 33.084 | +0.029 | +0.1% | 99.9 |
| `gpu__ssp_rk3_substep_` | 18 → 18 | 30.248 | 30.270 | +0.022 | +0.1% | 99.9 |
| `ncclDevKernel_SendRecv` | 0 → 63 | — | 17.082 | — | — | — |
| `gpu_broadcast_kernel_cartesian` | 6 → 132 | 11.367 | 15.714 | +4.346 | +38.2% | 72.3 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 → 3 | 12.960 | 13.001 | +0.041 | +0.3% | 99.7 |
| `gpu__compute_temperature_and_pressure_` | 3 → 3 | 11.464 | 11.466 | +0.001 | +0.0% | 100.0 |
| `gpu__compute_velocities_` | 3 → 3 | 9.162 | 9.103 | -0.059 | -0.6% | 100.7 |
| `gpu__copy_from_buffer_` | 0 → 90 | — | 5.092 | — | — | — |
| `gpu__compute_density_tendency_` | 3 → 3 | 5.029 | 5.035 | +0.006 | +0.1% | 99.9 |
| `gpu__compute_total_density_` | 3 → 3 | 3.812 | 3.825 | +0.012 | +0.3% | 99.7 |
| `gpu__fill_periodic_south_and_north_halo_` | 57 → 63 | 1.741 | 1.609 | -0.132 | -7.6% | 108.2 |
| `gpu__fill_bottom_and_top_halo_` | 54 → 60 | 0.459 | 0.502 | +0.043 | +9.3% | 91.5 |
| `gpu__compute_z_bcs_` | 9 → 9 | 0.056 | 0.055 | -0.001 | -1.2% | 101.3 |
| `gpu__fill_periodic_west_and_east_halo_` | 57 → 0 | 12.902 | — | — | — | — |
| **All kernels** | 231 → 465 | 272.035 | 285.617 | +13.582 | +5.0% | 95.2 |
