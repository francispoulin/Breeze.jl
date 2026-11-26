# GPU kernel time per time step

Mean over the captured steps of the summed duration of every launch of each kernel;
launches with the same short name are one row. Times are GPU execution in
milliseconds; kernels on one GPU can overlap, so the column sum can exceed the
step's wall time. NCCL kernels include time spent waiting for the peer rank.

## hydrostatic_1g

1 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step |
|---|---|---|
| `gpu_compute_hydrostatic_free_surface_Gv_` | 3 | 15.950 |
| `gpu_compute_hydrostatic_free_surface_Gu_` | 3 | 15.905 |
| `gpu_compute_hydrostatic_free_surface_Gc_` | 3 | 14.930 |
| `gpu__compute_w_from_continuity_` | 6 | 10.298 |
| `gpu__compute_barotropic_mode_` | 6 | 10.116 |
| `gpu__barotropic_split_explicit_corrector_` | 3 | 9.012 |
| `gpu__compute_split_explicit_transport_velocities_` | 3 | 8.551 |
| `gpu__rk_substep_field_` | 6 | 7.566 |
| `gpu__compute_realized_barotropic_tendency_` | 3 | 5.943 |
| `gpu__update_hydrostatic_pressure_` | 3 | 4.496 |
| `gpu_broadcast_kernel_cartesian` | 14 | 3.803 |
| `gpu__rk_substep_tracer_field_` | 3 | 3.767 |
| `gpu__fill_periodic_west_and_east_halo_` | 33 | 1.242 |
| `gpu__cache_tracer_fields_` | 1 | 0.887 |
| `gpu__split_explicit_free_surface_` | 63 | 0.574 |
| `gpu__split_explicit_barotropic_velocity_` | 63 | 0.562 |
| `gpu__fill_periodic_south_and_north_halo_` | 33 | 0.271 |
| `gpu__fill_bottom_and_top_halo_` | 12 | 0.083 |
| `gpu__update_split_explicit_state_` | 3 | 0.019 |
| **All kernels** | 264 | **113.975** |

## hydrostatic_1gd

1 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step |
|---|---|---|
| `gpu_compute_hydrostatic_free_surface_Gv_` | 3 | 15.954 |
| `gpu_compute_hydrostatic_free_surface_Gu_` | 3 | 15.905 |
| `gpu_compute_hydrostatic_free_surface_Gc_` | 3 | 14.937 |
| `gpu__compute_w_from_continuity_` | 6 | 10.312 |
| `gpu__compute_barotropic_mode_` | 6 | 10.129 |
| `gpu__barotropic_split_explicit_corrector_` | 3 | 9.008 |
| `gpu__compute_split_explicit_transport_velocities_` | 3 | 8.558 |
| `gpu__rk_substep_field_` | 6 | 7.567 |
| `gpu__compute_realized_barotropic_tendency_` | 3 | 5.958 |
| `gpu__update_hydrostatic_pressure_` | 3 | 4.505 |
| `gpu_broadcast_kernel_cartesian` | 14 | 3.804 |
| `gpu__rk_substep_tracer_field_` | 3 | 3.767 |
| `gpu__fill_periodic_west_and_east_halo_` | 33 | 1.228 |
| `gpu__cache_tracer_fields_` | 1 | 0.887 |
| `gpu__split_explicit_free_surface_` | 63 | 0.551 |
| `gpu__split_explicit_barotropic_velocity_` | 63 | 0.542 |
| `gpu__fill_periodic_south_and_north_halo_` | 33 | 0.272 |
| `gpu__fill_bottom_and_top_halo_` | 12 | 0.083 |
| `gpu__update_split_explicit_state_` | 3 | 0.019 |
| **All kernels** | 264 | **113.985** |

## hydrostatic_2g

2 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step | Rank 1 ms/step |
|---|---|---|---|
| `gpu_compute_hydrostatic_free_surface_Gv_` | 9 | 16.868 | 16.881 |
| `gpu_compute_hydrostatic_free_surface_Gu_` | 9 | 16.841 | 16.853 |
| `gpu_compute_hydrostatic_free_surface_Gc_` | 9 | 16.024 | 16.031 |
| `gpu__compute_w_from_continuity_` | 6 | 10.380 | 10.414 |
| `gpu__compute_barotropic_mode_` | 6 | 10.034 | 9.987 |
| `gpu__barotropic_split_explicit_corrector_` | 3 | 8.990 | 8.997 |
| `gpu__compute_split_explicit_transport_velocities_` | 3 | 8.612 | 8.610 |
| `gpu__rk_substep_field_` | 6 | 7.585 | 7.585 |
| `gpu__update_hydrostatic_pressure_` | 9 | 6.337 | 6.318 |
| `ncclDevKernel_SendRecv` | 30 | 4.535 | 6.208 |
| `gpu__compute_realized_barotropic_tendency_` | 3 | 5.906 | 5.924 |
| `gpu_broadcast_kernel_cartesian` | 74 | 4.566 | 4.571 |
| `gpu__rk_substep_tracer_field_` | 3 | 3.766 | 3.765 |
| `gpu__copy_from_buffer_` | 60 | 1.036 | 1.032 |
| `gpu__cache_tracer_fields_` | 1 | 0.887 | 0.886 |
| `gpu__split_explicit_free_surface_` | 63 | 0.612 | 0.617 |
| `gpu__split_explicit_barotropic_velocity_` | 63 | 0.607 | 0.612 |
| `gpu__fill_periodic_south_and_north_halo_` | 33 | 0.281 | 0.281 |
| `gpu__fill_bottom_and_top_halo_` | 12 | 0.082 | 0.079 |
| `gpu__update_split_explicit_state_` | 3 | 0.019 | 0.019 |
| **All kernels** | 405 | **123.967** | **125.669** |

## nonhydrostatic_1gd

1 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step |
|---|---|---|
| `gpu_permutedims_kernel_` | 12 | 74.067 |
| `gpu_broadcast_kernel_cartesian` | 9 | 28.063 |
| `vector_fft` | 12 | 18.928 |
| `gpu__rk3_substep_field_` | 12 | 18.363 |
| `gpu_compute_Gu_` | 3 | 16.458 |
| `gpu_compute_Gv_` | 3 | 16.185 |
| `gpu_compute_Gw_` | 3 | 16.121 |
| `gpu_compute_Gc_` | 3 | 15.270 |
| `scal_kernel_ref` | 9 | 13.988 |
| `gpu__make_pressure_correction_` | 3 | 11.329 |
| `gpu__cache_field_tendencies_` | 12 | 10.699 |
| `regular_fft` | 6 | 10.529 |
| `gpu__compute_source_term_` | 3 | 6.078 |
| `gpu__spectral_poisson_solve_` | 3 | 5.109 |
| `gpu_unpermute_z_indices_` | 3 | 4.805 |
| `gpu_permute_z_indices_` | 3 | 4.782 |
| `gpu__update_hydrostatic_pressure_` | 3 | 4.569 |
| `gpu__copy_real_component_` | 3 | 3.724 |
| `gpu__divide_by_` | 3 | 2.702 |
| `gpu__fill_periodic_west_and_east_halo_` | 27 | 2.577 |
| `gpu__fill_periodic_south_and_north_halo_` | 27 | 0.478 |
| `gpu__fill_bottom_and_top_halo_` | 24 | 0.171 |
| **All kernels** | 186 | **284.994** |

## nonhydrostatic_2g

2 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step | Rank 1 ms/step |
|---|---|---|---|
| `gpu_permutedims_kernel_` | 12 | 73.383 | 73.342 |
| `ncclDevKernel_SendRecv` | 30 | 62.907 | 50.751 |
| `gpu_broadcast_kernel_cartesian` | 57 | 29.447 | 29.439 |
| `vector_fft` | 12 | 18.956 | 18.956 |
| `gpu__rk3_substep_field_` | 12 | 18.348 | 18.351 |
| `gpu_compute_Gu_` | 9 | 17.646 | 17.254 |
| `gpu_compute_Gw_` | 9 | 17.117 | 17.118 |
| `gpu_compute_Gv_` | 9 | 17.050 | 17.069 |
| `gpu_compute_Gc_` | 9 | 16.230 | 16.231 |
| `scal_kernel_ref` | 9 | 13.996 | 13.984 |
| `gpu__pack_buffer_x_to_y_` | 3 | 13.430 | 13.429 |
| `gpu__make_pressure_correction_` | 3 | 11.312 | 11.294 |
| `gpu__cache_field_tendencies_` | 12 | 10.646 | 10.623 |
| `regular_fft` | 6 | 10.455 | 10.456 |
| `gpu__update_hydrostatic_pressure_` | 9 | 7.058 | 6.991 |
| `gpu__compute_source_term_` | 3 | 6.080 | 6.076 |
| `gpu__spectral_poisson_solve_` | 3 | 5.074 | 5.070 |
| `gpu__unpack_buffer_y_from_x_` | 3 | 5.011 | 5.025 |
| `gpu__unpack_buffer_x_from_y_` | 3 | 4.968 | 4.940 |
| `gpu__pack_buffer_y_to_x_` | 3 | 4.847 | 4.842 |
| `gpu_unpermute_z_indices_` | 3 | 4.811 | 4.806 |
| `gpu_permute_z_indices_` | 3 | 4.785 | 4.780 |
| `gpu__copy_real_component_` | 3 | 3.727 | 3.723 |
| `gpu__divide_by_` | 3 | 2.685 | 2.687 |
| `gpu__copy_from_buffer_` | 48 | 2.378 | 2.378 |
| `gpu__fill_periodic_south_and_north_halo_` | 27 | 0.490 | 0.469 |
| `gpu__fill_bottom_and_top_halo_` | 24 | 0.191 | 0.198 |
| **All kernels** | 327 | **383.030** | **370.283** |
