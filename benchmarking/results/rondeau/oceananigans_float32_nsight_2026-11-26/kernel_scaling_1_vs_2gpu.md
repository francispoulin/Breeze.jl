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
| `ncclDevKernel_SendRecv` | 0 → 30 | — | 60.168 | — | — | — |
| `gpu_compute_hydrostatic_free_surface_Gu_` | 3 → 9 | 15.905 | 17.514 | +1.610 | +10.1% | 90.8 |
| `gpu_compute_hydrostatic_free_surface_Gv_` | 3 → 9 | 15.956 | 17.479 | +1.523 | +9.5% | 91.3 |
| `gpu_compute_hydrostatic_free_surface_Gc_` | 3 → 9 | 14.932 | 16.611 | +1.679 | +11.2% | 89.9 |
| `gpu__compute_barotropic_mode_` | 6 → 6 | 10.075 | 10.550 | +0.475 | +4.7% | 95.5 |
| `gpu__compute_w_from_continuity_` | 6 → 6 | 10.288 | 10.514 | +0.226 | +2.2% | 97.9 |
| `gpu__barotropic_split_explicit_corrector_` | 3 → 3 | 9.024 | 9.339 | +0.316 | +3.5% | 96.6 |
| `gpu__compute_split_explicit_transport_velocities_` | 3 → 3 | 8.623 | 8.865 | +0.242 | +2.8% | 97.3 |
| `gpu__rk_substep_field_` | 6 → 6 | 7.577 | 7.597 | +0.019 | +0.3% | 99.7 |
| `gpu__update_hydrostatic_pressure_` | 3 → 9 | 4.478 | 6.403 | +1.924 | +43.0% | 69.9 |
| `gpu__compute_realized_barotropic_tendency_` | 3 → 3 | 5.960 | 5.909 | -0.051 | -0.8% | 100.9 |
| `gpu_broadcast_kernel_cartesian` | 14 → 74 | 3.805 | 4.680 | +0.875 | +23.0% | 81.3 |
| `gpu__rk_substep_tracer_field_` | 3 → 3 | 3.766 | 3.778 | +0.012 | +0.3% | 99.7 |
| `gpu__copy_from_buffer_` | 0 → 60 | — | 1.050 | — | — | — |
| `gpu__cache_tracer_fields_` | 1 → 1 | 0.885 | 0.902 | +0.016 | +1.8% | 98.2 |
| `gpu__split_explicit_barotropic_velocity_` | 63 → 63 | 0.554 | 0.636 | +0.082 | +14.8% | 87.1 |
| `gpu__split_explicit_free_surface_` | 63 → 63 | 0.561 | 0.630 | +0.069 | +12.3% | 89.1 |
| `gpu__fill_periodic_south_and_north_halo_` | 33 → 33 | 0.269 | 0.299 | +0.030 | +11.2% | 89.9 |
| `gpu__fill_bottom_and_top_halo_` | 12 → 12 | 0.081 | 0.083 | +0.002 | +2.1% | 97.9 |
| `gpu__update_split_explicit_state_` | 3 → 3 | 0.019 | 0.020 | +0.002 | +8.0% | 92.6 |
| `gpu__fill_periodic_west_and_east_halo_` | 33 → 0 | 1.226 | — | — | — | — |
| **All kernels** | 264 → 405 | 113.986 | 179.831 | +65.845 | +57.8% | 63.4 |

## Nonhydrostatic

Base `nonhydrostatic_1gd` (1 GPU), compared `nonhydrostatic_2g` (2 GPUs).

| Kernel | Launches/step 1→2 GPU | 1 GPU ms/step | 2 GPU ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `gpu_permutedims_kernel_` | 12 → 12 | 74.770 | 75.266 | +0.496 | +0.7% | 99.3 |
| `ncclDevKernel_SendRecv` | 0 → 30 | — | 53.081 | — | — | — |
| `gpu_broadcast_kernel_cartesian` | 9 → 57 | 28.023 | 29.939 | +1.915 | +6.8% | 93.6 |
| `vector_fft` | 12 → 12 | 18.933 | 18.963 | +0.030 | +0.2% | 99.8 |
| `gpu__rk3_substep_field_` | 12 → 12 | 18.366 | 18.405 | +0.039 | +0.2% | 99.8 |
| `gpu_compute_Gu_` | 3 → 9 | 16.308 | 17.456 | +1.148 | +7.0% | 93.4 |
| `gpu_compute_Gw_` | 3 → 9 | 15.984 | 17.270 | +1.287 | +8.1% | 92.5 |
| `gpu_compute_Gv_` | 3 → 9 | 16.032 | 17.213 | +1.181 | +7.4% | 93.1 |
| `gpu_compute_Gc_` | 3 → 9 | 15.189 | 16.384 | +1.195 | +7.9% | 92.7 |
| `scal_kernel_ref` | 9 → 9 | 13.988 | 14.005 | +0.017 | +0.1% | 99.9 |
| `gpu__pack_buffer_x_to_y_` | 0 → 3 | — | 13.335 | — | — | — |
| `gpu__make_pressure_correction_` | 3 → 3 | 11.388 | 11.576 | +0.188 | +1.7% | 98.4 |
| `gpu__cache_field_tendencies_` | 12 → 12 | 10.627 | 10.731 | +0.104 | +1.0% | 99.0 |
| `regular_fft` | 6 → 6 | 10.446 | 10.643 | +0.197 | +1.9% | 98.2 |
| `gpu__update_hydrostatic_pressure_` | 3 → 9 | 4.537 | 6.975 | +2.437 | +53.7% | 65.1 |
| `gpu__compute_source_term_` | 3 → 3 | 6.079 | 6.077 | -0.002 | -0.0% | 100.0 |
| `gpu__spectral_poisson_solve_` | 3 → 3 | 5.088 | 5.160 | +0.073 | +1.4% | 98.6 |
| `gpu__unpack_buffer_y_from_x_` | 0 → 3 | — | 5.036 | — | — | — |
| `gpu__unpack_buffer_x_from_y_` | 0 → 3 | — | 5.002 | — | — | — |
| `gpu__pack_buffer_y_to_x_` | 0 → 3 | — | 4.857 | — | — | — |
| `gpu_unpermute_z_indices_` | 3 → 3 | 4.818 | 4.822 | +0.004 | +0.1% | 99.9 |
| `gpu_permute_z_indices_` | 3 → 3 | 4.788 | 4.793 | +0.005 | +0.1% | 99.9 |
| `gpu__copy_real_component_` | 3 → 3 | 3.730 | 3.730 | +0.001 | +0.0% | 100.0 |
| `gpu__divide_by_` | 3 → 3 | 2.683 | 2.707 | +0.024 | +0.9% | 99.1 |
| `gpu__copy_from_buffer_` | 0 → 48 | — | 2.381 | — | — | — |
| `gpu__fill_periodic_south_and_north_halo_` | 27 → 27 | 0.480 | 0.488 | +0.008 | +1.6% | 98.4 |
| `gpu__fill_bottom_and_top_halo_` | 24 → 24 | 0.169 | 0.200 | +0.031 | +18.1% | 84.6 |
| `gpu__fill_periodic_west_and_east_halo_` | 27 → 0 | 2.588 | — | — | — | — |
| **All kernels** | 186 → 327 | 285.014 | 376.302 | +91.288 | +32.0% | 75.7 |
