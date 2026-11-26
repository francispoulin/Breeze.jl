# GPU kernel time per time step

Mean over the captured steps of the summed duration of every launch of each kernel;
launches with the same short name are one row. Times are GPU execution in
milliseconds; kernels on one GPU can overlap, so the column sum can exceed the
step's wall time. NCCL kernels include time spent waiting for the peer rank.

## hydrostatic_1g

1 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step |
|---|---|---|
| `gpu_compute_hydrostatic_free_surface_Gv_` | 3 | 15.957 |
| `gpu_compute_hydrostatic_free_surface_Gu_` | 3 | 15.906 |
| `gpu_compute_hydrostatic_free_surface_Gc_` | 3 | 14.938 |
| `gpu__compute_w_from_continuity_` | 6 | 10.283 |
| `gpu__compute_barotropic_mode_` | 6 | 10.075 |
| `gpu__barotropic_split_explicit_corrector_` | 3 | 9.032 |
| `gpu__compute_split_explicit_transport_velocities_` | 3 | 8.623 |
| `gpu__rk_substep_field_` | 6 | 7.577 |
| `gpu__compute_realized_barotropic_tendency_` | 3 | 5.925 |
| `gpu__update_hydrostatic_pressure_` | 3 | 4.486 |
| `gpu_broadcast_kernel_cartesian` | 14 | 3.805 |
| `gpu__rk_substep_tracer_field_` | 3 | 3.768 |
| `gpu__fill_periodic_west_and_east_halo_` | 33 | 1.245 |
| `gpu__cache_tracer_fields_` | 1 | 0.885 |
| `gpu__split_explicit_free_surface_` | 63 | 0.580 |
| `gpu__split_explicit_barotropic_velocity_` | 63 | 0.567 |
| `gpu__fill_periodic_south_and_north_halo_` | 33 | 0.271 |
| `gpu__fill_bottom_and_top_halo_` | 12 | 0.082 |
| `gpu__update_split_explicit_state_` | 3 | 0.019 |
| **All kernels** | 264 | **114.025** |

## hydrostatic_1gd

1 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step |
|---|---|---|
| `gpu_compute_hydrostatic_free_surface_Gv_` | 3 | 15.956 |
| `gpu_compute_hydrostatic_free_surface_Gu_` | 3 | 15.905 |
| `gpu_compute_hydrostatic_free_surface_Gc_` | 3 | 14.932 |
| `gpu__compute_w_from_continuity_` | 6 | 10.288 |
| `gpu__compute_barotropic_mode_` | 6 | 10.075 |
| `gpu__barotropic_split_explicit_corrector_` | 3 | 9.024 |
| `gpu__compute_split_explicit_transport_velocities_` | 3 | 8.623 |
| `gpu__rk_substep_field_` | 6 | 7.577 |
| `gpu__compute_realized_barotropic_tendency_` | 3 | 5.960 |
| `gpu__update_hydrostatic_pressure_` | 3 | 4.478 |
| `gpu_broadcast_kernel_cartesian` | 14 | 3.805 |
| `gpu__rk_substep_tracer_field_` | 3 | 3.766 |
| `gpu__fill_periodic_west_and_east_halo_` | 33 | 1.226 |
| `gpu__cache_tracer_fields_` | 1 | 0.885 |
| `gpu__split_explicit_free_surface_` | 63 | 0.561 |
| `gpu__split_explicit_barotropic_velocity_` | 63 | 0.554 |
| `gpu__fill_periodic_south_and_north_halo_` | 33 | 0.269 |
| `gpu__fill_bottom_and_top_halo_` | 12 | 0.081 |
| `gpu__update_split_explicit_state_` | 3 | 0.019 |
| **All kernels** | 264 | **113.986** |

## hydrostatic_2g

2 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step | Rank 1 ms/step |
|---|---|---|---|
| `ncclDevKernel_SendRecv` | 30 | 60.168 | 3.983 |
| `gpu_compute_hydrostatic_free_surface_Gu_` | 9 | 16.841 | 17.514 |
| `gpu_compute_hydrostatic_free_surface_Gv_` | 9 | 16.864 | 17.479 |
| `gpu_compute_hydrostatic_free_surface_Gc_` | 9 | 16.039 | 16.611 |
| `gpu__compute_barotropic_mode_` | 6 | 10.071 | 10.550 |
| `gpu__compute_w_from_continuity_` | 6 | 10.362 | 10.514 |
| `gpu__barotropic_split_explicit_corrector_` | 3 | 9.127 | 9.339 |
| `gpu__compute_split_explicit_transport_velocities_` | 3 | 8.651 | 8.865 |
| `gpu__rk_substep_field_` | 6 | 7.584 | 7.597 |
| `gpu__update_hydrostatic_pressure_` | 9 | 6.338 | 6.403 |
| `gpu__compute_realized_barotropic_tendency_` | 3 | 5.909 | 5.894 |
| `gpu_broadcast_kernel_cartesian` | 74 | 4.566 | 4.680 |
| `gpu__rk_substep_tracer_field_` | 3 | 3.771 | 3.778 |
| `gpu__copy_from_buffer_` | 60 | 1.038 | 1.050 |
| `gpu__cache_tracer_fields_` | 1 | 0.885 | 0.902 |
| `gpu__split_explicit_barotropic_velocity_` | 63 | 0.613 | 0.636 |
| `gpu__split_explicit_free_surface_` | 63 | 0.617 | 0.630 |
| `gpu__fill_periodic_south_and_north_halo_` | 33 | 0.286 | 0.299 |
| `gpu__fill_bottom_and_top_halo_` | 12 | 0.081 | 0.083 |
| `gpu__update_split_explicit_state_` | 3 | 0.020 | 0.020 |
| **All kernels** | 405 | **179.831** | **126.828** |

## nonhydrostatic_1g

1 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step |
|---|---|---|
| `gpu_broadcast_kernel_cartesian` | 9 | 27.711 |
| `regular_fft` | 12 | 19.945 |
| `gpu__rk3_substep_field_` | 12 | 18.375 |
| `gpu_compute_Gu_` | 3 | 16.309 |
| `gpu_compute_Gv_` | 3 | 16.030 |
| `gpu_compute_Gw_` | 3 | 15.973 |
| `gpu_compute_Gc_` | 3 | 15.192 |
| `gpu__make_pressure_correction_` | 3 | 11.399 |
| `gpu__cache_field_tendencies_` | 12 | 10.616 |
| `vector_fft` | 6 | 9.484 |
| `scal_kernel_ref` | 6 | 9.320 |
| `gpu__compute_source_term_` | 3 | 6.087 |
| `gpu__spectral_poisson_solve_` | 3 | 5.085 |
| `gpu_unpermute_z_indices_` | 3 | 4.824 |
| `gpu_permute_z_indices_` | 3 | 4.792 |
| `gpu__update_hydrostatic_pressure_` | 3 | 4.531 |
| `gpu_copy_real_component_` | 3 | 3.735 |
| `gpu__divide_by_` | 3 | 2.684 |
| `gpu__fill_periodic_west_and_east_halo_` | 27 | 2.615 |
| `gpu__fill_periodic_south_and_north_halo_` | 27 | 0.480 |
| `gpu__fill_bottom_and_top_halo_` | 24 | 0.170 |
| **All kernels** | 171 | **205.356** |

## nonhydrostatic_1gd

1 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step |
|---|---|---|
| `gpu_permutedims_kernel_` | 12 | 74.770 |
| `gpu_broadcast_kernel_cartesian` | 9 | 28.023 |
| `vector_fft` | 12 | 18.933 |
| `gpu__rk3_substep_field_` | 12 | 18.366 |
| `gpu_compute_Gu_` | 3 | 16.308 |
| `gpu_compute_Gv_` | 3 | 16.032 |
| `gpu_compute_Gw_` | 3 | 15.984 |
| `gpu_compute_Gc_` | 3 | 15.189 |
| `scal_kernel_ref` | 9 | 13.988 |
| `gpu__make_pressure_correction_` | 3 | 11.388 |
| `gpu__cache_field_tendencies_` | 12 | 10.627 |
| `regular_fft` | 6 | 10.446 |
| `gpu__compute_source_term_` | 3 | 6.079 |
| `gpu__spectral_poisson_solve_` | 3 | 5.088 |
| `gpu_unpermute_z_indices_` | 3 | 4.818 |
| `gpu_permute_z_indices_` | 3 | 4.788 |
| `gpu__update_hydrostatic_pressure_` | 3 | 4.537 |
| `gpu__copy_real_component_` | 3 | 3.730 |
| `gpu__divide_by_` | 3 | 2.683 |
| `gpu__fill_periodic_west_and_east_halo_` | 27 | 2.588 |
| `gpu__fill_periodic_south_and_north_halo_` | 27 | 0.480 |
| `gpu__fill_bottom_and_top_halo_` | 24 | 0.169 |
| **All kernels** | 186 | **285.014** |

## nonhydrostatic_2g

2 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step | Rank 1 ms/step |
|---|---|---|---|
| `gpu_permutedims_kernel_` | 12 | 75.266 | 75.260 |
| `ncclDevKernel_SendRecv` | 30 | 53.081 | 51.785 |
| `gpu_broadcast_kernel_cartesian` | 57 | 29.939 | 29.904 |
| `vector_fft` | 12 | 18.919 | 18.963 |
| `gpu__rk3_substep_field_` | 12 | 18.366 | 18.405 |
| `gpu_compute_Gu_` | 9 | 17.423 | 17.456 |
| `gpu_compute_Gw_` | 9 | 17.265 | 17.270 |
| `gpu_compute_Gv_` | 9 | 17.209 | 17.213 |
| `gpu_compute_Gc_` | 9 | 16.377 | 16.384 |
| `scal_kernel_ref` | 9 | 14.005 | 14.000 |
| `gpu__pack_buffer_x_to_y_` | 3 | 13.335 | 13.282 |
| `gpu__make_pressure_correction_` | 3 | 11.576 | 11.555 |
| `gpu__cache_field_tendencies_` | 12 | 10.731 | 10.724 |
| `regular_fft` | 6 | 10.643 | 10.611 |
| `gpu__update_hydrostatic_pressure_` | 9 | 6.967 | 6.975 |
| `gpu__compute_source_term_` | 3 | 6.074 | 6.077 |
| `gpu__spectral_poisson_solve_` | 3 | 5.160 | 5.136 |
| `gpu__unpack_buffer_y_from_x_` | 3 | 5.036 | 5.021 |
| `gpu__unpack_buffer_x_from_y_` | 3 | 4.987 | 5.002 |
| `gpu__pack_buffer_y_to_x_` | 3 | 4.853 | 4.857 |
| `gpu_unpermute_z_indices_` | 3 | 4.816 | 4.822 |
| `gpu_permute_z_indices_` | 3 | 4.789 | 4.793 |
| `gpu__copy_real_component_` | 3 | 3.729 | 3.730 |
| `gpu__divide_by_` | 3 | 2.703 | 2.707 |
| `gpu__copy_from_buffer_` | 48 | 2.381 | 2.379 |
| `gpu__fill_periodic_south_and_north_halo_` | 27 | 0.473 | 0.488 |
| `gpu__fill_bottom_and_top_halo_` | 24 | 0.196 | 0.200 |
| **All kernels** | 327 | **376.302** | **374.998** |
