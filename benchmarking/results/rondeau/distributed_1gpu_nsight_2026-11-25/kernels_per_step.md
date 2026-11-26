# GPU kernel time per time step

Mean over the captured steps of the summed duration of every launch of each kernel;
launches with the same short name are one row. Times are GPU execution in
milliseconds; kernels on one GPU can overlap, so the column sum can exceed the
step's wall time. NCCL kernels include time spent waiting for the peer rank.

## hydrostatic_1gd

1 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step |
|---|---|---|
| `gpu_compute_hydrostatic_free_surface_Gv_` | 3 | 36.147 |
| `gpu_compute_hydrostatic_free_surface_Gu_` | 3 | 35.618 |
| `gpu_compute_hydrostatic_free_surface_Gc_` | 3 | 34.946 |
| `gpu__compute_w_from_continuity_` | 6 | 25.425 |
| `gpu__compute_barotropic_mode_` | 6 | 15.191 |
| `gpu__barotropic_split_explicit_corrector_` | 3 | 14.378 |
| `gpu__rk_substep_field_` | 6 | 14.328 |
| `gpu__compute_split_explicit_transport_velocities_` | 3 | 13.203 |
| `gpu__compute_realized_barotropic_tendency_` | 3 | 11.472 |
| `gpu__update_hydrostatic_pressure_` | 3 | 9.757 |
| `gpu__rk_substep_tracer_field_` | 3 | 7.167 |
| `gpu_broadcast_kernel_cartesian` | 14 | 3.975 |
| `gpu__fill_periodic_west_and_east_halo_` | 33 | 1.875 |
| `gpu__cache_tracer_fields_` | 1 | 1.644 |
| `gpu__split_explicit_free_surface_` | 63 | 0.721 |
| `gpu__split_explicit_barotropic_velocity_` | 63 | 0.705 |
| `gpu__fill_periodic_south_and_north_halo_` | 33 | 0.393 |
| `gpu__fill_bottom_and_top_halo_` | 12 | 0.119 |
| `gpu_fill_kernel_` | 15 | 0.041 |
| `gpu__update_split_explicit_state_` | 3 | 0.023 |
| **All kernels** | 279 | **227.131** |

## nonhydrostatic_1gd

1 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step |
|---|---|---|
| `gpu_permutedims_kernel_` | 12 | 87.140 |
| `gpu_broadcast_kernel_cartesian` | 9 | 43.075 |
| `vector_fft` | 12 | 37.379 |
| `gpu_compute_Gw_` | 3 | 37.093 |
| `gpu_compute_Gv_` | 3 | 36.943 |
| `gpu__rk3_substep_field_` | 12 | 36.585 |
| `gpu_compute_Gu_` | 3 | 36.481 |
| `gpu_compute_Gc_` | 3 | 35.264 |
| `scal_kernel_ref` | 9 | 28.033 |
| `regular_fft` | 6 | 20.276 |
| `gpu__cache_field_tendencies_` | 12 | 19.774 |
| `gpu__make_pressure_correction_` | 3 | 19.589 |
| `gpu__compute_source_term_` | 3 | 11.717 |
| `gpu__update_hydrostatic_pressure_` | 3 | 9.652 |
| `gpu__spectral_poisson_solve_` | 3 | 9.483 |
| `gpu_unpermute_z_indices_` | 3 | 9.318 |
| `gpu_permute_z_indices_` | 3 | 9.288 |
| `gpu__copy_real_component_` | 3 | 7.169 |
| `gpu__divide_by_` | 3 | 5.858 |
| `gpu__fill_periodic_west_and_east_halo_` | 27 | 3.981 |
| `gpu__fill_periodic_south_and_north_halo_` | 27 | 0.756 |
| `gpu__fill_bottom_and_top_halo_` | 24 | 0.254 |
| **All kernels** | 186 | **505.109** |

## anelastic_1gd

1 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step |
|---|---|---|
| `gpu_permutedims_kernel_` | 12 | 73.385 |
| `gpu_compute_potential_temperature_tendency_` | 3 | 36.979 |
| `gpu_compute_z_momentum_tendency_` | 3 | 35.392 |
| `gpu_compute_y_momentum_tendency_` | 3 | 32.842 |
| `gpu_compute_scalar_tendency_` | 3 | 32.691 |
| `gpu_compute_x_momentum_tendency_` | 3 | 32.576 |
| `gpu__ssp_rk3_substep_` | 15 | 25.191 |
| `vector_fft` | 12 | 18.941 |
| `gpu_broadcast_kernel_cartesian` | 11 | 15.130 |
| `gpu_solve_batched_tridiagonal_system_kernel_` | 3 | 14.388 |
| `gpu__pressure_correct_momentum_` | 3 | 11.385 |
| `gpu__fill_periodic_west_and_east_halo_` | 54 | 10.648 |
| `scal_kernel_ref` | 6 | 9.335 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 | 8.803 |
| `gpu__compute_velocities_` | 3 | 8.067 |
| `gpu__compute_anelastic_source_term_` | 3 | 6.121 |
| `partial_mapreduce_grid` | 6 | 4.234 |
| `gpu__copy_real_component_` | 3 | 3.729 |
| `gpu__broadcast_kernel_` | 3 | 2.680 |
| `gpu__fill_periodic_south_and_north_halo_` | 54 | 1.538 |
| `gpu__fill_bottom_and_top_halo_` | 54 | 0.398 |
| `gpu__compute_z_bcs_` | 9 | 0.055 |
| **All kernels** | 269 | **384.506** |

## explicit_1gd

1 rank(s), 10 captured steps.

| Kernel | Launches/step | Rank 0 ms/step |
|---|---|---|
| `gpu_compute_potential_temperature_tendency_` | 3 | 39.130 |
| `gpu_compute_z_momentum_tendency_` | 3 | 34.091 |
| `gpu_compute_y_momentum_tendency_` | 3 | 33.459 |
| `gpu_compute_x_momentum_tendency_` | 3 | 33.070 |
| `gpu_compute_scalar_tendency_` | 3 | 33.056 |
| `gpu__ssp_rk3_substep_` | 18 | 30.235 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 | 12.963 |
| `gpu__fill_periodic_west_and_east_halo_` | 63 | 12.443 |
| `gpu__compute_temperature_and_pressure_` | 3 | 11.467 |
| `gpu_broadcast_kernel_cartesian` | 6 | 11.371 |
| `gpu__compute_velocities_` | 3 | 9.159 |
| `gpu__compute_density_tendency_` | 3 | 5.029 |
| `gpu__compute_total_density_` | 3 | 3.811 |
| `gpu__fill_periodic_south_and_north_halo_` | 63 | 1.789 |
| `gpu__fill_bottom_and_top_halo_` | 60 | 0.474 |
| `gpu__compute_z_bcs_` | 9 | 0.055 |
| **All kernels** | 249 | **271.600** |
