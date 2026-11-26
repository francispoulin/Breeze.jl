# Kernel time per time step: scaling comparison

GPU execution time per time step, summed over all launches of each kernel and averaged
over the captured steps. Multi-rank cases use each kernel's slowest rank.
Difference = scaled − base. Efficiency = 100 × base / scaled (weak scaling: 100% means
the kernel takes as long with the same work per GPU). A dash marks a kernel that only
one case launches. Rows are sorted by the scaled case's time, then the base case's.

## Hydrostatic

Base `hydrostatic_1g` (1 GPU), compared `hydrostatic_1gd` (1 GPU).

| Kernel | Launches/step base→compared | `hydrostatic_1g` ms/step | `hydrostatic_1gd` ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `gpu_compute_hydrostatic_free_surface_Gv_` | 3 → 3 | 36.167 | 36.147 | -0.019 | -0.1% | 100.1 |
| `gpu_compute_hydrostatic_free_surface_Gu_` | 3 → 3 | 35.636 | 35.618 | -0.018 | -0.1% | 100.1 |
| `gpu_compute_hydrostatic_free_surface_Gc_` | 3 → 3 | 34.921 | 34.946 | +0.025 | +0.1% | 99.9 |
| `gpu__compute_w_from_continuity_` | 6 → 6 | 24.711 | 25.425 | +0.713 | +2.9% | 97.2 |
| `gpu__compute_barotropic_mode_` | 6 → 6 | 14.537 | 15.191 | +0.655 | +4.5% | 95.7 |
| `gpu__barotropic_split_explicit_corrector_` | 3 → 3 | 14.000 | 14.378 | +0.378 | +2.7% | 97.4 |
| `gpu__rk_substep_field_` | 6 → 6 | 14.332 | 14.328 | -0.004 | -0.0% | 100.0 |
| `gpu__compute_split_explicit_transport_velocities_` | 3 → 3 | 12.664 | 13.203 | +0.539 | +4.3% | 95.9 |
| `gpu__compute_realized_barotropic_tendency_` | 3 → 3 | 11.267 | 11.472 | +0.205 | +1.8% | 98.2 |
| `gpu__update_hydrostatic_pressure_` | 3 → 3 | 9.452 | 9.757 | +0.305 | +3.2% | 96.9 |
| `gpu__rk_substep_tracer_field_` | 3 → 3 | 7.167 | 7.167 | +0.000 | +0.0% | 100.0 |
| `gpu_broadcast_kernel_cartesian` | 14 → 14 | 3.975 | 3.975 | +0.000 | +0.0% | 100.0 |
| `gpu__fill_periodic_west_and_east_halo_` | 33 → 33 | 1.863 | 1.875 | +0.012 | +0.6% | 99.4 |
| `gpu__cache_tracer_fields_` | 1 → 1 | 1.644 | 1.644 | +0.001 | +0.0% | 100.0 |
| `gpu__split_explicit_free_surface_` | 63 → 63 | 0.734 | 0.721 | -0.013 | -1.7% | 101.8 |
| `gpu__split_explicit_barotropic_velocity_` | 63 → 63 | 0.724 | 0.705 | -0.019 | -2.6% | 102.7 |
| `gpu__fill_periodic_south_and_north_halo_` | 33 → 33 | 0.390 | 0.393 | +0.003 | +0.9% | 99.1 |
| `gpu__fill_bottom_and_top_halo_` | 12 → 12 | 0.120 | 0.119 | -0.000 | -0.4% | 100.4 |
| `gpu_fill_kernel_` | 15 → 15 | 0.040 | 0.041 | +0.001 | +2.4% | 97.7 |
| `gpu__update_split_explicit_state_` | 3 → 3 | 0.023 | 0.023 | +0.000 | +0.7% | 99.3 |
| **All kernels** | 279 → 279 | 224.367 | 227.131 | +2.764 | +1.2% | 98.8 |

## Nonhydrostatic

Base `nonhydrostatic_1g` (1 GPU), compared `nonhydrostatic_1gd` (1 GPU).

| Kernel | Launches/step base→compared | `nonhydrostatic_1g` ms/step | `nonhydrostatic_1gd` ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `gpu_permutedims_kernel_` | 0 → 12 | — | 87.140 | — | — | — |
| `gpu_broadcast_kernel_cartesian` | 9 → 9 | 43.057 | 43.075 | +0.018 | +0.0% | 100.0 |
| `vector_fft` | 6 → 12 | 18.689 | 37.379 | +18.690 | +100.0% | 50.0 |
| `gpu_compute_Gw_` | 3 → 3 | 37.093 | 37.093 | +0.001 | +0.0% | 100.0 |
| `gpu_compute_Gv_` | 3 → 3 | 36.953 | 36.943 | -0.010 | -0.0% | 100.0 |
| `gpu__rk3_substep_field_` | 12 → 12 | 36.195 | 36.585 | +0.390 | +1.1% | 98.9 |
| `gpu_compute_Gu_` | 3 → 3 | 36.489 | 36.481 | -0.007 | -0.0% | 100.0 |
| `gpu_compute_Gc_` | 3 → 3 | 35.239 | 35.264 | +0.025 | +0.1% | 99.9 |
| `scal_kernel_ref` | 6 → 9 | 18.678 | 28.033 | +9.355 | +50.1% | 66.6 |
| `regular_fft` | 12 → 6 | 39.500 | 20.276 | -19.223 | -48.7% | 194.8 |
| `gpu__cache_field_tendencies_` | 12 → 12 | 19.770 | 19.774 | +0.004 | +0.0% | 100.0 |
| `gpu__make_pressure_correction_` | 3 → 3 | 19.077 | 19.589 | +0.512 | +2.7% | 97.4 |
| `gpu__compute_source_term_` | 3 → 3 | 11.709 | 11.717 | +0.007 | +0.1% | 99.9 |
| `gpu__update_hydrostatic_pressure_` | 3 → 3 | 9.259 | 9.652 | +0.393 | +4.2% | 95.9 |
| `gpu__spectral_poisson_solve_` | 3 → 3 | 9.468 | 9.483 | +0.016 | +0.2% | 99.8 |
| `gpu_unpermute_z_indices_` | 3 → 3 | 9.304 | 9.318 | +0.015 | +0.2% | 99.8 |
| `gpu_permute_z_indices_` | 3 → 3 | 9.283 | 9.288 | +0.005 | +0.1% | 99.9 |
| `gpu__copy_real_component_` | 0 → 3 | — | 7.169 | — | — | — |
| `gpu__divide_by_` | 3 → 3 | 5.658 | 5.858 | +0.200 | +3.5% | 96.6 |
| `gpu__fill_periodic_west_and_east_halo_` | 27 → 27 | 4.000 | 3.981 | -0.020 | -0.5% | 100.5 |
| `gpu__fill_periodic_south_and_north_halo_` | 27 → 27 | 0.743 | 0.756 | +0.013 | +1.7% | 98.3 |
| `gpu__fill_bottom_and_top_halo_` | 24 → 24 | 0.255 | 0.254 | -0.002 | -0.6% | 100.6 |
| `gpu_copy_real_component_` | 3 → 0 | 7.170 | — | — | — | — |
| **All kernels** | 171 → 186 | 407.588 | 505.109 | +97.520 | +23.9% | 80.7 |

## Anelastic

Base `anelastic_1g` (1 GPU), compared `anelastic_1gd` (1 GPU).

| Kernel | Launches/step base→compared | `anelastic_1g` ms/step | `anelastic_1gd` ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `gpu_permutedims_kernel_` | 0 → 12 | — | 75.335 | — | — | — |
| `gpu_compute_potential_temperature_tendency_` | 3 → 3 | 36.925 | 36.953 | +0.028 | +0.1% | 99.9 |
| `gpu_compute_z_momentum_tendency_` | 3 → 3 | 35.390 | 35.409 | +0.019 | +0.1% | 99.9 |
| `gpu_compute_y_momentum_tendency_` | 3 → 3 | 32.835 | 32.828 | -0.007 | -0.0% | 100.0 |
| `gpu_compute_scalar_tendency_` | 3 → 3 | 32.689 | 32.678 | -0.011 | -0.0% | 100.0 |
| `gpu_compute_x_momentum_tendency_` | 3 → 3 | 32.573 | 32.573 | +0.000 | +0.0% | 100.0 |
| `gpu__ssp_rk3_substep_` | 15 → 15 | 25.178 | 25.194 | +0.015 | +0.1% | 99.9 |
| `vector_fft` | 6 → 12 | 9.493 | 18.937 | +9.443 | +99.5% | 50.1 |
| `gpu_broadcast_kernel_cartesian` | 5 → 11 | 9.478 | 15.142 | +5.664 | +59.8% | 62.6 |
| `gpu_solve_batched_tridiagonal_system_kernel_` | 3 → 3 | 14.467 | 14.451 | -0.015 | -0.1% | 100.1 |
| `gpu__pressure_correct_momentum_` | 3 → 3 | 10.704 | 11.388 | +0.684 | +6.4% | 94.0 |
| `gpu__fill_periodic_west_and_east_halo_` | 51 → 54 | 11.008 | 10.650 | -0.357 | -3.2% | 103.4 |
| `scal_kernel_ref` | 3 → 6 | 4.667 | 9.338 | +4.671 | +100.1% | 50.0 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 → 3 | 8.682 | 8.682 | +0.000 | +0.0% | 100.0 |
| `gpu__compute_velocities_` | 3 → 3 | 8.093 | 8.091 | -0.002 | -0.0% | 100.0 |
| `gpu__compute_anelastic_source_term_` | 3 → 3 | 6.145 | 6.128 | -0.016 | -0.3% | 100.3 |
| `partial_mapreduce_grid` | 3 → 6 | 0.015 | 4.235 | +4.219 | +27877.1% | 0.4 |
| `gpu__copy_real_component_` | 0 → 3 | — | 3.739 | — | — | — |
| `gpu__broadcast_kernel_` | 0 → 3 | — | 2.678 | — | — | — |
| `gpu__fill_periodic_south_and_north_halo_` | 51 → 54 | 1.484 | 1.535 | +0.051 | +3.5% | 96.7 |
| `gpu__fill_bottom_and_top_halo_` | 51 → 54 | 0.388 | 0.394 | +0.006 | +1.5% | 98.5 |
| `gpu__compute_z_bcs_` | 9 → 9 | 0.054 | 0.054 | -0.000 | -0.2% | 100.2 |
| `regular_fft` | 6 → 0 | 9.449 | — | — | — | — |
| `gpu_copy_real_component_` | 3 → 0 | 3.779 | — | — | — | — |
| `gpu_fill_kernel_` | 3 → 0 | 2.073 | — | — | — | — |
| `gpu_broadcast_kernel_linear` | 3 → 0 | 0.015 | — | — | — | — |
| **All kernels** | 239 → 269 | 295.583 | 386.413 | +90.830 | +30.7% | 76.5 |

## Compressible explicit (CE)

Base `explicit_1g` (1 GPU), compared `explicit_1gd` (1 GPU).

| Kernel | Launches/step base→compared | `explicit_1g` ms/step | `explicit_1gd` ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `gpu_compute_potential_temperature_tendency_` | 3 → 3 | 39.153 | 39.147 | -0.006 | -0.0% | 100.0 |
| `gpu_compute_z_momentum_tendency_` | 3 → 3 | 34.088 | 34.150 | +0.063 | +0.2% | 99.8 |
| `gpu_compute_y_momentum_tendency_` | 3 → 3 | 33.462 | 33.479 | +0.018 | +0.1% | 99.9 |
| `gpu_compute_x_momentum_tendency_` | 3 → 3 | 33.077 | 33.094 | +0.017 | +0.1% | 99.9 |
| `gpu_compute_scalar_tendency_` | 3 → 3 | 33.056 | 33.060 | +0.004 | +0.0% | 100.0 |
| `gpu__ssp_rk3_substep_` | 18 → 18 | 30.248 | 30.273 | +0.025 | +0.1% | 99.9 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 → 3 | 12.960 | 13.015 | +0.055 | +0.4% | 99.6 |
| `gpu__fill_periodic_west_and_east_halo_` | 57 → 63 | 12.902 | 12.481 | -0.421 | -3.3% | 103.4 |
| `gpu__compute_temperature_and_pressure_` | 3 → 3 | 11.464 | 11.490 | +0.026 | +0.2% | 99.8 |
| `gpu_broadcast_kernel_cartesian` | 6 → 6 | 11.367 | 11.370 | +0.002 | +0.0% | 100.0 |
| `gpu__compute_velocities_` | 3 → 3 | 9.162 | 9.152 | -0.010 | -0.1% | 100.1 |
| `gpu__compute_density_tendency_` | 3 → 3 | 5.029 | 5.028 | -0.000 | -0.0% | 100.0 |
| `gpu__compute_total_density_` | 3 → 3 | 3.812 | 3.824 | +0.011 | +0.3% | 99.7 |
| `gpu__fill_periodic_south_and_north_halo_` | 57 → 63 | 1.741 | 1.772 | +0.031 | +1.8% | 98.2 |
| `gpu__fill_bottom_and_top_halo_` | 54 → 60 | 0.459 | 0.479 | +0.020 | +4.3% | 95.9 |
| `gpu__compute_z_bcs_` | 9 → 9 | 0.056 | 0.055 | -0.000 | -0.9% | 100.9 |
| **All kernels** | 231 → 249 | 272.035 | 271.870 | -0.165 | -0.1% | 100.1 |
