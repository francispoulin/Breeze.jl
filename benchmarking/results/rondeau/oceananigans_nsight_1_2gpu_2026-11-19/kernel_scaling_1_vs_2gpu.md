# Kernel time per time step: scaling comparison

GPU execution time per time step, summed over all launches of each kernel and averaged
over the captured steps. Multi-rank cases use each kernel's slowest rank.
Difference = scaled − base. Efficiency = 100 × base / scaled (weak scaling: 100% means
the kernel takes as long with the same work per GPU). A dash marks a kernel that only
one case launches. Rows are sorted by the scaled case's time.

## Hydrostatic

Base `hydrostatic_1g` (1 GPU), scaled `hydrostatic_2g` (2 GPUs).

| Kernel | Launches/step 1→2 GPU | 1 GPU ms/step | 2 GPU ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `gpu_compute_hydrostatic_free_surface_Gv_` | 3 → 9 | 36.167 | 38.368 | +2.201 | +6.1% | 94.3 |
| `gpu_compute_hydrostatic_free_surface_Gu_` | 3 → 9 | 35.636 | 38.292 | +2.656 | +7.5% | 93.1 |
| `gpu_compute_hydrostatic_free_surface_Gc_` | 3 → 9 | 34.921 | 37.181 | +2.260 | +6.5% | 93.9 |
| `gpu__compute_w_from_continuity_` | 6 → 6 | 24.711 | 25.494 | +0.783 | +3.2% | 96.9 |
| `gpu__compute_barotropic_mode_` | 6 → 6 | 14.537 | 15.661 | +1.125 | +7.7% | 92.8 |
| `gpu__barotropic_split_explicit_corrector_` | 3 → 3 | 14.000 | 14.394 | +0.394 | +2.8% | 97.3 |
| `gpu__rk_substep_field_` | 6 → 6 | 14.332 | 14.351 | +0.019 | +0.1% | 99.9 |
| `gpu__compute_split_explicit_transport_velocities_` | 3 → 3 | 12.664 | 13.250 | +0.586 | +4.6% | 95.6 |
| `ncclDevKernel_SendRecv` | 0 → 30 | — | 12.756 | — | — | — |
| `gpu__compute_realized_barotropic_tendency_` | 3 → 3 | 11.267 | 11.546 | +0.280 | +2.5% | 97.6 |
| `gpu__update_hydrostatic_pressure_` | 3 → 9 | 9.452 | 11.238 | +1.786 | +18.9% | 84.1 |
| `gpu__rk_substep_tracer_field_` | 3 → 3 | 7.167 | 7.177 | +0.010 | +0.1% | 99.9 |
| `gpu_broadcast_kernel_cartesian` | 14 → 74 | 3.975 | 4.856 | +0.882 | +22.2% | 81.8 |
| `gpu__cache_tracer_fields_` | 1 → 1 | 1.644 | 1.645 | +0.001 | +0.0% | 100.0 |
| `gpu__copy_from_buffer_` | 0 → 60 | — | 1.173 | — | — | — |
| `gpu__split_explicit_barotropic_velocity_` | 63 → 63 | 0.724 | 0.831 | +0.107 | +14.8% | 87.1 |
| `gpu__split_explicit_free_surface_` | 63 → 63 | 0.734 | 0.804 | +0.069 | +9.5% | 91.4 |
| `gpu__fill_periodic_south_and_north_halo_` | 33 → 33 | 0.390 | 0.420 | +0.030 | +7.7% | 92.9 |
| `gpu__fill_bottom_and_top_halo_` | 12 → 12 | 0.120 | 0.110 | -0.010 | -8.4% | 109.1 |
| `gpu_fill_kernel_` | 15 → 15 | 0.040 | 0.043 | +0.003 | +7.2% | 93.3 |
| `gpu__update_split_explicit_state_` | 3 → 3 | 0.023 | 0.025 | +0.001 | +5.6% | 94.7 |
| `gpu__fill_periodic_west_and_east_halo_` | 33 → 0 | 1.863 | — | — | — | — |
| **All kernels** | 279 → 420 | 224.367 | 247.527 | +23.160 | +10.3% | 90.6 |

## Nonhydrostatic

Base `nonhydrostatic_1g` (1 GPU), scaled `nonhydrostatic_2g` (2 GPUs).

| Kernel | Launches/step 1→2 GPU | 1 GPU ms/step | 2 GPU ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `ncclDevKernel_SendRecv` | 0 → 30 | — | 150.650 | — | — | — |
| `gpu_permutedims_kernel_` | 0 → 12 | — | 87.590 | — | — | — |
| `gpu_broadcast_kernel_cartesian` | 9 → 57 | 43.057 | 44.859 | +1.802 | +4.2% | 96.0 |
| `gpu_compute_Gu_` | 3 → 9 | 36.489 | 40.160 | +3.671 | +10.1% | 90.9 |
| `gpu_compute_Gv_` | 3 → 9 | 36.953 | 40.152 | +3.199 | +8.7% | 92.0 |
| `gpu_compute_Gw_` | 3 → 9 | 37.093 | 39.469 | +2.376 | +6.4% | 94.0 |
| `vector_fft` | 6 → 12 | 18.689 | 37.445 | +18.756 | +100.4% | 49.9 |
| `gpu_compute_Gc_` | 3 → 9 | 35.239 | 37.387 | +2.148 | +6.1% | 94.3 |
| `gpu__rk3_substep_field_` | 12 → 12 | 36.195 | 36.620 | +0.425 | +1.2% | 98.8 |
| `scal_kernel_ref` | 6 → 9 | 18.678 | 28.060 | +9.382 | +50.2% | 66.6 |
| `regular_fft` | 12 → 6 | 39.500 | 20.535 | -18.964 | -48.0% | 192.3 |
| `gpu__cache_field_tendencies_` | 12 → 12 | 19.770 | 19.795 | +0.024 | +0.1% | 99.9 |
| `gpu__make_pressure_correction_` | 3 → 3 | 19.077 | 19.487 | +0.409 | +2.1% | 97.9 |
| `gpu__pack_buffer_x_to_y_` | 0 → 3 | — | 16.293 | — | — | — |
| `gpu__update_hydrostatic_pressure_` | 3 → 9 | 9.259 | 12.215 | +2.957 | +31.9% | 75.8 |
| `gpu__compute_source_term_` | 3 → 3 | 11.709 | 11.904 | +0.195 | +1.7% | 98.4 |
| `gpu__unpack_buffer_y_from_x_` | 0 → 3 | — | 9.743 | — | — | — |
| `gpu__spectral_poisson_solve_` | 3 → 3 | 9.468 | 9.460 | -0.008 | -0.1% | 100.1 |
| `gpu__pack_buffer_y_to_x_` | 0 → 3 | — | 9.409 | — | — | — |
| `gpu__unpack_buffer_x_from_y_` | 0 → 3 | — | 9.358 | — | — | — |
| `gpu_unpermute_z_indices_` | 3 → 3 | 9.304 | 9.335 | +0.031 | +0.3% | 99.7 |
| `gpu_permute_z_indices_` | 3 → 3 | 9.283 | 9.319 | +0.037 | +0.4% | 99.6 |
| `gpu__copy_real_component_` | 0 → 3 | — | 7.217 | — | — | — |
| `gpu__divide_by_` | 3 → 3 | 5.658 | 5.861 | +0.203 | +3.6% | 96.5 |
| `gpu__copy_from_buffer_` | 0 → 48 | — | 2.673 | — | — | — |
| `gpu__fill_periodic_south_and_north_halo_` | 27 → 27 | 0.743 | 0.721 | -0.022 | -3.0% | 103.1 |
| `gpu__fill_bottom_and_top_halo_` | 24 → 24 | 0.255 | 0.233 | -0.022 | -8.6% | 109.4 |
| `gpu_copy_real_component_` | 3 → 0 | 7.170 | — | — | — | — |
| `gpu__fill_periodic_west_and_east_halo_` | 27 → 0 | 4.000 | — | — | — | — |
| **All kernels** | 171 → 327 | 407.588 | 715.756 | +308.168 | +75.6% | 56.9 |
