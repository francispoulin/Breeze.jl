# GPU kernel time per time step

Mean over the captured steps of the summed duration of every launch of each kernel;
launches with the same short name are one row. Times are GPU execution in
milliseconds; kernels on one GPU can overlap, so the column sum can exceed the
step's wall time. NCCL kernels include time spent waiting for the peer rank.

## hydrostatic_1g

1 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step |
|---|---|---|
| `gpu_compute_hydrostatic_free_surface_Gv_` | 3 | 36.167 |
| `gpu_compute_hydrostatic_free_surface_Gu_` | 3 | 35.636 |
| `gpu_compute_hydrostatic_free_surface_Gc_` | 3 | 34.921 |
| `gpu__compute_w_from_continuity_` | 6 | 24.711 |
| `gpu__compute_barotropic_mode_` | 6 | 14.537 |
| `gpu__rk_substep_field_` | 6 | 14.332 |
| `gpu__barotropic_split_explicit_corrector_` | 3 | 14.000 |
| `gpu__compute_split_explicit_transport_velocities_` | 3 | 12.664 |
| `gpu__compute_realized_barotropic_tendency_` | 3 | 11.267 |
| `gpu__update_hydrostatic_pressure_` | 3 | 9.452 |
| `gpu__rk_substep_tracer_field_` | 3 | 7.167 |
| `gpu_broadcast_kernel_cartesian` | 14 | 3.975 |
| `gpu__fill_periodic_west_and_east_halo_` | 33 | 1.863 |
| `gpu__cache_tracer_fields_` | 1 | 1.644 |
| `gpu__split_explicit_free_surface_` | 63 | 0.734 |
| `gpu__split_explicit_barotropic_velocity_` | 63 | 0.724 |
| `gpu__fill_periodic_south_and_north_halo_` | 33 | 0.390 |
| `gpu__fill_bottom_and_top_halo_` | 12 | 0.120 |
| `gpu_fill_kernel_` | 15 | 0.040 |
| `gpu__update_split_explicit_state_` | 3 | 0.023 |
| **All kernels** | 279 | **224.367** |

## hydrostatic_2g

2 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step | Rank 1 ms/step |
|---|---|---|---|
| `gpu_compute_hydrostatic_free_surface_Gv_` | 9 | 38.363 | 38.368 |
| `gpu_compute_hydrostatic_free_surface_Gu_` | 9 | 38.292 | 38.292 |
| `gpu_compute_hydrostatic_free_surface_Gc_` | 9 | 37.181 | 37.136 |
| `gpu__compute_w_from_continuity_` | 6 | 24.614 | 25.494 |
| `gpu__compute_barotropic_mode_` | 6 | 15.412 | 15.661 |
| `gpu__barotropic_split_explicit_corrector_` | 3 | 14.089 | 14.394 |
| `gpu__rk_substep_field_` | 6 | 14.351 | 14.330 |
| `gpu__compute_split_explicit_transport_velocities_` | 3 | 12.805 | 13.250 |
| `ncclDevKernel_SendRecv` | 30 | 12.756 | 7.710 |
| `gpu__compute_realized_barotropic_tendency_` | 3 | 11.415 | 11.546 |
| `gpu__update_hydrostatic_pressure_` | 9 | 11.238 | 11.202 |
| `gpu__rk_substep_tracer_field_` | 3 | 7.177 | 7.168 |
| `gpu_broadcast_kernel_cartesian` | 74 | 4.856 | 4.836 |
| `gpu__cache_tracer_fields_` | 1 | 1.645 | 1.643 |
| `gpu__copy_from_buffer_` | 60 | 1.152 | 1.173 |
| `gpu__split_explicit_barotropic_velocity_` | 63 | 0.814 | 0.831 |
| `gpu__split_explicit_free_surface_` | 63 | 0.792 | 0.804 |
| `gpu__fill_periodic_south_and_north_halo_` | 33 | 0.401 | 0.420 |
| `gpu__fill_bottom_and_top_halo_` | 12 | 0.107 | 0.110 |
| `gpu_fill_kernel_` | 15 | 0.043 | 0.042 |
| `gpu__update_split_explicit_state_` | 3 | 0.024 | 0.025 |
| **All kernels** | 420 | **247.527** | **244.436** |

## nonhydrostatic_1g

1 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step |
|---|---|---|
| `gpu_broadcast_kernel_cartesian` | 9 | 43.057 |
| `regular_fft` | 12 | 39.500 |
| `gpu_compute_Gw_` | 3 | 37.093 |
| `gpu_compute_Gv_` | 3 | 36.953 |
| `gpu_compute_Gu_` | 3 | 36.489 |
| `gpu__rk3_substep_field_` | 12 | 36.195 |
| `gpu_compute_Gc_` | 3 | 35.239 |
| `gpu__cache_field_tendencies_` | 12 | 19.770 |
| `gpu__make_pressure_correction_` | 3 | 19.077 |
| `vector_fft` | 6 | 18.689 |
| `scal_kernel_ref` | 6 | 18.678 |
| `gpu__compute_source_term_` | 3 | 11.709 |
| `gpu__spectral_poisson_solve_` | 3 | 9.468 |
| `gpu_unpermute_z_indices_` | 3 | 9.304 |
| `gpu_permute_z_indices_` | 3 | 9.283 |
| `gpu__update_hydrostatic_pressure_` | 3 | 9.259 |
| `gpu_copy_real_component_` | 3 | 7.170 |
| `gpu__divide_by_` | 3 | 5.658 |
| `gpu__fill_periodic_west_and_east_halo_` | 27 | 4.000 |
| `gpu__fill_periodic_south_and_north_halo_` | 27 | 0.743 |
| `gpu__fill_bottom_and_top_halo_` | 24 | 0.255 |
| **All kernels** | 171 | **407.588** |

## nonhydrostatic_2g

2 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step | Rank 1 ms/step |
|---|---|---|---|
| `ncclDevKernel_SendRecv` | 30 | 105.969 | 150.650 |
| `gpu_permutedims_kernel_` | 12 | 87.590 | 87.569 |
| `gpu_broadcast_kernel_cartesian` | 57 | 44.804 | 44.859 |
| `gpu_compute_Gu_` | 9 | 39.137 | 40.160 |
| `gpu_compute_Gv_` | 9 | 39.298 | 40.152 |
| `gpu_compute_Gw_` | 9 | 39.469 | 39.412 |
| `vector_fft` | 12 | 37.361 | 37.445 |
| `gpu_compute_Gc_` | 9 | 37.387 | 37.387 |
| `gpu__rk3_substep_field_` | 12 | 36.202 | 36.620 |
| `scal_kernel_ref` | 9 | 28.050 | 28.060 |
| `regular_fft` | 6 | 20.535 | 20.467 |
| `gpu__cache_field_tendencies_` | 12 | 19.795 | 19.790 |
| `gpu__make_pressure_correction_` | 3 | 19.062 | 19.487 |
| `gpu__pack_buffer_x_to_y_` | 3 | 16.280 | 16.293 |
| `gpu__update_hydrostatic_pressure_` | 9 | 11.807 | 12.215 |
| `gpu__compute_source_term_` | 3 | 11.831 | 11.904 |
| `gpu__unpack_buffer_y_from_x_` | 3 | 9.743 | 9.719 |
| `gpu__spectral_poisson_solve_` | 3 | 9.460 | 9.460 |
| `gpu__pack_buffer_y_to_x_` | 3 | 9.409 | 9.408 |
| `gpu__unpack_buffer_x_from_y_` | 3 | 9.355 | 9.358 |
| `gpu_unpermute_z_indices_` | 3 | 9.335 | 9.331 |
| `gpu_permute_z_indices_` | 3 | 9.319 | 9.310 |
| `gpu__copy_real_component_` | 3 | 7.187 | 7.217 |
| `gpu__divide_by_` | 3 | 5.665 | 5.861 |
| `gpu__copy_from_buffer_` | 48 | 2.673 | 2.670 |
| `gpu__fill_periodic_south_and_north_halo_` | 27 | 0.673 | 0.721 |
| `gpu__fill_bottom_and_top_halo_` | 24 | 0.228 | 0.233 |
| **All kernels** | 327 | **667.627** | **715.756** |
