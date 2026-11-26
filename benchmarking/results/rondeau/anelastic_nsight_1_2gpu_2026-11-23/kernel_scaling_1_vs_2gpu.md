# Kernel time per time step: scaling comparison

GPU execution time per time step, summed over all launches of each kernel and averaged
over the captured steps. Multi-rank cases use each kernel's slowest rank.
Difference = scaled − base. Efficiency = 100 × base / scaled (weak scaling: 100% means
the kernel takes as long with the same work per GPU). A dash marks a kernel that only
one case launches. Rows are sorted by the scaled case's time.

## Anelastic

Base `anelastic_1g` (1 GPU), scaled `anelastic_2g` (2 GPUs).

| Kernel | Launches/step 1→2 GPU | 1 GPU ms/step | 2 GPU ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `gpu_permutedims_kernel_` | 0 → 12 | — | 73.383 | — | — | — |
| `ncclDevKernel_SendRecv` | 0 → 60 | — | 59.062 | — | — | — |
| `gpu_compute_potential_temperature_tendency_` | 3 → 3 | 36.925 | 37.086 | +0.162 | +0.4% | 99.6 |
| `gpu_compute_z_momentum_tendency_` | 3 → 3 | 35.390 | 35.503 | +0.113 | +0.3% | 99.7 |
| `gpu_compute_y_momentum_tendency_` | 3 → 3 | 32.835 | 32.869 | +0.034 | +0.1% | 99.9 |
| `gpu_compute_scalar_tendency_` | 3 → 3 | 32.689 | 32.718 | +0.029 | +0.1% | 99.9 |
| `gpu_compute_x_momentum_tendency_` | 3 → 3 | 32.573 | 32.611 | +0.038 | +0.1% | 99.9 |
| `gpu__ssp_rk3_substep_` | 15 → 15 | 25.178 | 25.207 | +0.028 | +0.1% | 99.9 |
| `vector_fft` | 6 → 12 | 9.493 | 18.939 | +9.446 | +99.5% | 50.1 |
| `gpu_broadcast_kernel_cartesian` | 5 → 119 | 9.478 | 18.780 | +9.303 | +98.2% | 50.5 |
| `gpu_solve_batched_tridiagonal_system_kernel_` | 3 → 3 | 14.467 | 14.404 | -0.062 | -0.4% | 100.4 |
| `gpu__pack_buffer_x_to_y_` | 0 → 3 | — | 13.410 | — | — | — |
| `gpu__pressure_correct_momentum_` | 3 → 3 | 10.704 | 11.385 | +0.681 | +6.4% | 94.0 |
| `scal_kernel_ref` | 3 → 6 | 4.667 | 9.334 | +4.667 | +100.0% | 50.0 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 → 3 | 8.682 | 8.702 | +0.020 | +0.2% | 99.8 |
| `gpu__compute_velocities_` | 3 → 3 | 8.093 | 8.097 | +0.004 | +0.1% | 99.9 |
| `gpu__compute_anelastic_source_term_` | 3 → 3 | 6.145 | 6.101 | -0.044 | -0.7% | 100.7 |
| `gpu__unpack_buffer_y_from_x_` | 0 → 3 | — | 5.024 | — | — | — |
| `gpu__unpack_buffer_x_from_y_` | 0 → 3 | — | 4.962 | — | — | — |
| `gpu__pack_buffer_y_to_x_` | 0 → 3 | — | 4.827 | — | — | — |
| `gpu__copy_from_buffer_` | 0 → 78 | — | 4.418 | — | — | — |
| `partial_mapreduce_grid` | 3 → 6 | 0.015 | 4.235 | +4.220 | +27881.6% | 0.4 |
| `gpu__copy_real_component_` | 0 → 3 | — | 3.730 | — | — | — |
| `gpu__broadcast_kernel_` | 0 → 3 | — | 2.685 | — | — | — |
| `gpu__fill_periodic_south_and_north_halo_` | 51 → 54 | 1.484 | 1.494 | +0.010 | +0.7% | 99.3 |
| `gpu__fill_bottom_and_top_halo_` | 51 → 54 | 0.388 | 0.505 | +0.116 | +30.0% | 76.9 |
| `gpu__compute_z_bcs_` | 9 → 9 | 0.054 | 0.055 | +0.001 | +1.2% | 98.9 |
| `gpu_broadcast_kernel_linear` | 3 → 0 | 0.015 | — | — | — | — |
| `regular_fft` | 6 → 0 | 9.449 | — | — | — | — |
| `gpu_fill_kernel_` | 3 → 0 | 2.073 | — | — | — | — |
| `gpu__fill_periodic_west_and_east_halo_` | 51 → 0 | 11.008 | — | — | — | — |
| `gpu_copy_real_component_` | 3 → 0 | 3.779 | — | — | — | — |
| **All kernels** | 239 → 473 | 295.583 | 469.342 | +173.758 | +58.8% | 63.0 |
