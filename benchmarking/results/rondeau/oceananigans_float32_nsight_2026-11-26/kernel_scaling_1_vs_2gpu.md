# Kernel time per time step: scaling comparison

GPU execution time per time step, summed over all launches of each kernel and averaged
over the captured steps. Multi-rank cases use each kernel's slowest rank.
Difference = scaled − base. Efficiency = 100 × base / scaled (weak scaling: 100% means
the kernel takes as long with the same work per GPU). A dash marks a kernel that only
one case launches. Rows are sorted by the scaled case's time, then the base case's.

## Hydrostatic

Base `hydrostatic_1gd` (1 GPU), compared `hydrostatic_2g` (2 GPUs).

| Kernel | Launches/step 1→2 GPU | 1 GPU ms/step | 2 GPU ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `gpu_compute_hydrostatic_free_surface_Gv_` | 3 → 9 | 15.954 | 16.881 | +0.927 | +5.8% | 94.5 |
| `gpu_compute_hydrostatic_free_surface_Gu_` | 3 → 9 | 15.905 | 16.853 | +0.947 | +6.0% | 94.4 |
| `gpu_compute_hydrostatic_free_surface_Gc_` | 3 → 9 | 14.937 | 16.031 | +1.094 | +7.3% | 93.2 |
| `gpu__compute_w_from_continuity_` | 6 → 6 | 10.312 | 10.414 | +0.101 | +1.0% | 99.0 |
| `gpu__compute_barotropic_mode_` | 6 → 6 | 10.129 | 10.034 | -0.096 | -0.9% | 101.0 |
| `gpu__barotropic_split_explicit_corrector_` | 3 → 3 | 9.008 | 8.997 | -0.011 | -0.1% | 100.1 |
| `gpu__compute_split_explicit_transport_velocities_` | 3 → 3 | 8.558 | 8.612 | +0.054 | +0.6% | 99.4 |
| `gpu__rk_substep_field_` | 6 → 6 | 7.567 | 7.585 | +0.018 | +0.2% | 99.8 |
| `gpu__update_hydrostatic_pressure_` | 3 → 9 | 4.505 | 6.337 | +1.832 | +40.7% | 71.1 |
| `ncclDevKernel_SendRecv` | 0 → 30 | — | 6.208 | — | — | — |
| `gpu__compute_realized_barotropic_tendency_` | 3 → 3 | 5.958 | 5.924 | -0.033 | -0.6% | 100.6 |
| `gpu_broadcast_kernel_cartesian` | 14 → 74 | 3.804 | 4.571 | +0.767 | +20.2% | 83.2 |
| `gpu__rk_substep_tracer_field_` | 3 → 3 | 3.767 | 3.766 | -0.002 | -0.0% | 100.0 |
| `gpu__copy_from_buffer_` | 0 → 60 | — | 1.036 | — | — | — |
| `gpu__cache_tracer_fields_` | 1 → 1 | 0.887 | 0.887 | -0.000 | -0.0% | 100.0 |
| `gpu__split_explicit_free_surface_` | 63 → 63 | 0.551 | 0.617 | +0.066 | +12.0% | 89.3 |
| `gpu__split_explicit_barotropic_velocity_` | 63 → 63 | 0.542 | 0.612 | +0.070 | +12.9% | 88.5 |
| `gpu__fill_periodic_south_and_north_halo_` | 33 → 33 | 0.272 | 0.281 | +0.009 | +3.4% | 96.8 |
| `gpu__fill_bottom_and_top_halo_` | 12 → 12 | 0.083 | 0.082 | -0.001 | -1.5% | 101.5 |
| `gpu__update_split_explicit_state_` | 3 → 3 | 0.019 | 0.019 | +0.001 | +3.2% | 96.9 |
| `gpu__fill_periodic_west_and_east_halo_` | 33 → 0 | 1.228 | — | — | — | — |
| **All kernels** | 264 → 405 | 113.985 | 125.669 | +11.684 | +10.3% | 90.7 |

## Nonhydrostatic

Base `nonhydrostatic_1gd` (1 GPU), compared `nonhydrostatic_2g` (2 GPUs).

| Kernel | Launches/step 1→2 GPU | 1 GPU ms/step | 2 GPU ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `gpu_permutedims_kernel_` | 12 → 12 | 74.067 | 73.383 | -0.685 | -0.9% | 100.9 |
| `ncclDevKernel_SendRecv` | 0 → 30 | — | 62.907 | — | — | — |
| `gpu_broadcast_kernel_cartesian` | 9 → 57 | 28.063 | 29.447 | +1.384 | +4.9% | 95.3 |
| `vector_fft` | 12 → 12 | 18.928 | 18.956 | +0.029 | +0.2% | 99.8 |
| `gpu__rk3_substep_field_` | 12 → 12 | 18.363 | 18.351 | -0.012 | -0.1% | 100.1 |
| `gpu_compute_Gu_` | 3 → 9 | 16.458 | 17.646 | +1.188 | +7.2% | 93.3 |
| `gpu_compute_Gw_` | 3 → 9 | 16.121 | 17.118 | +0.998 | +6.2% | 94.2 |
| `gpu_compute_Gv_` | 3 → 9 | 16.185 | 17.069 | +0.883 | +5.5% | 94.8 |
| `gpu_compute_Gc_` | 3 → 9 | 15.270 | 16.231 | +0.962 | +6.3% | 94.1 |
| `scal_kernel_ref` | 9 → 9 | 13.988 | 13.996 | +0.008 | +0.1% | 99.9 |
| `gpu__pack_buffer_x_to_y_` | 0 → 3 | — | 13.430 | — | — | — |
| `gpu__make_pressure_correction_` | 3 → 3 | 11.329 | 11.312 | -0.016 | -0.1% | 100.1 |
| `gpu__cache_field_tendencies_` | 12 → 12 | 10.699 | 10.646 | -0.053 | -0.5% | 100.5 |
| `regular_fft` | 6 → 6 | 10.529 | 10.456 | -0.074 | -0.7% | 100.7 |
| `gpu__update_hydrostatic_pressure_` | 3 → 9 | 4.569 | 7.058 | +2.489 | +54.5% | 64.7 |
| `gpu__compute_source_term_` | 3 → 3 | 6.078 | 6.080 | +0.002 | +0.0% | 100.0 |
| `gpu__spectral_poisson_solve_` | 3 → 3 | 5.109 | 5.074 | -0.034 | -0.7% | 100.7 |
| `gpu__unpack_buffer_y_from_x_` | 0 → 3 | — | 5.025 | — | — | — |
| `gpu__unpack_buffer_x_from_y_` | 0 → 3 | — | 4.968 | — | — | — |
| `gpu__pack_buffer_y_to_x_` | 0 → 3 | — | 4.847 | — | — | — |
| `gpu_unpermute_z_indices_` | 3 → 3 | 4.805 | 4.811 | +0.006 | +0.1% | 99.9 |
| `gpu_permute_z_indices_` | 3 → 3 | 4.782 | 4.785 | +0.003 | +0.1% | 99.9 |
| `gpu__copy_real_component_` | 3 → 3 | 3.724 | 3.727 | +0.002 | +0.1% | 99.9 |
| `gpu__divide_by_` | 3 → 3 | 2.702 | 2.687 | -0.015 | -0.6% | 100.6 |
| `gpu__copy_from_buffer_` | 0 → 48 | — | 2.378 | — | — | — |
| `gpu__fill_periodic_south_and_north_halo_` | 27 → 27 | 0.478 | 0.490 | +0.012 | +2.5% | 97.6 |
| `gpu__fill_bottom_and_top_halo_` | 24 → 24 | 0.171 | 0.198 | +0.027 | +15.6% | 86.5 |
| `gpu__fill_periodic_west_and_east_halo_` | 27 → 0 | 2.577 | — | — | — | — |
| **All kernels** | 186 → 327 | 284.994 | 383.030 | +98.036 | +34.4% | 74.4 |
