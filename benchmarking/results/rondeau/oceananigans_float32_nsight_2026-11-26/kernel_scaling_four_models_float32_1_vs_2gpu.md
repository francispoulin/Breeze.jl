# Kernel time per time step: scaling comparison

GPU execution time per time step, summed over all launches of each kernel and averaged
over the captured steps. Multi-rank cases use each kernel's slowest rank.
Difference = scaled − base. Efficiency = 100 × base / scaled (weak scaling: 100% means
the kernel takes as long with the same work per GPU). A dash marks a kernel that only
one case launches. Rows are sorted by the scaled case's time, then the base case's.

## Hydrostatic (Oceananigans)

Base `hydrostatic_1gd` (1 GPU), compared `hydrostatic_2g` (2 GPUs).

Float32 for all four models. Base is the one-GPU NCCL-distributed capture (--distributed=true, partition 1×1), so both cases take the distributed code path; a 1×1 partition has no halo buffers or NCCL, so communication kernels remain 2-GPU only. Every capture ran with environment.sh (JULIA_CUDA_MEMORY_POOL=none).

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

## Nonhydrostatic (Oceananigans)

Base `nonhydrostatic_1gd` (1 GPU), compared `nonhydrostatic_2g` (2 GPUs).

Float32 for all four models. Base is the one-GPU NCCL-distributed capture (--distributed=true, partition 1×1), so both cases take the distributed code path; a 1×1 partition has no halo buffers or NCCL, so communication kernels remain 2-GPU only. Every capture ran with environment.sh (JULIA_CUDA_MEMORY_POOL=none).

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

## Anelastic (Breeze)

Base `anelastic_1gd` (1 GPU), compared `anelastic_2g` (2 GPUs).

Float32 for all four models. Base is the one-GPU NCCL-distributed capture (--distributed=true, partition 1×1), so both cases take the distributed code path; a 1×1 partition has no halo buffers or NCCL, so communication kernels remain 2-GPU only. Every capture ran with environment.sh (JULIA_CUDA_MEMORY_POOL=none).

| Kernel | Launches/step 1→2 GPU | 1 GPU ms/step | 2 GPU ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `gpu_permutedims_kernel_` | 12 → 12 | 73.385 | 73.383 | -0.002 | -0.0% | 100.0 |
| `ncclDevKernel_SendRecv` | 0 → 60 | — | 59.062 | — | — | — |
| `gpu_compute_potential_temperature_tendency_` | 3 → 3 | 36.979 | 37.086 | +0.107 | +0.3% | 99.7 |
| `gpu_compute_z_momentum_tendency_` | 3 → 3 | 35.392 | 35.503 | +0.111 | +0.3% | 99.7 |
| `gpu_compute_y_momentum_tendency_` | 3 → 3 | 32.842 | 32.869 | +0.027 | +0.1% | 99.9 |
| `gpu_compute_scalar_tendency_` | 3 → 3 | 32.691 | 32.718 | +0.027 | +0.1% | 99.9 |
| `gpu_compute_x_momentum_tendency_` | 3 → 3 | 32.576 | 32.611 | +0.035 | +0.1% | 99.9 |
| `gpu__ssp_rk3_substep_` | 15 → 15 | 25.191 | 25.207 | +0.016 | +0.1% | 99.9 |
| `vector_fft` | 12 → 12 | 18.941 | 18.939 | -0.002 | -0.0% | 100.0 |
| `gpu_broadcast_kernel_cartesian` | 11 → 119 | 15.130 | 18.780 | +3.650 | +24.1% | 80.6 |
| `gpu_solve_batched_tridiagonal_system_kernel_` | 3 → 3 | 14.388 | 14.404 | +0.017 | +0.1% | 99.9 |
| `gpu__pack_buffer_x_to_y_` | 0 → 3 | — | 13.410 | — | — | — |
| `gpu__pressure_correct_momentum_` | 3 → 3 | 11.385 | 11.385 | -0.000 | -0.0% | 100.0 |
| `scal_kernel_ref` | 6 → 6 | 9.335 | 9.334 | -0.000 | -0.0% | 100.0 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 → 3 | 8.803 | 8.702 | -0.101 | -1.1% | 101.2 |
| `gpu__compute_velocities_` | 3 → 3 | 8.067 | 8.097 | +0.031 | +0.4% | 99.6 |
| `gpu__compute_anelastic_source_term_` | 3 → 3 | 6.121 | 6.101 | -0.020 | -0.3% | 100.3 |
| `gpu__unpack_buffer_y_from_x_` | 0 → 3 | — | 5.024 | — | — | — |
| `gpu__unpack_buffer_x_from_y_` | 0 → 3 | — | 4.962 | — | — | — |
| `gpu__pack_buffer_y_to_x_` | 0 → 3 | — | 4.827 | — | — | — |
| `gpu__copy_from_buffer_` | 0 → 78 | — | 4.418 | — | — | — |
| `partial_mapreduce_grid` | 6 → 6 | 4.234 | 4.235 | +0.001 | +0.0% | 100.0 |
| `gpu__copy_real_component_` | 3 → 3 | 3.729 | 3.730 | +0.001 | +0.0% | 100.0 |
| `gpu__broadcast_kernel_` | 3 → 3 | 2.680 | 2.685 | +0.006 | +0.2% | 99.8 |
| `gpu__fill_periodic_south_and_north_halo_` | 54 → 54 | 1.538 | 1.494 | -0.044 | -2.8% | 102.9 |
| `gpu__fill_bottom_and_top_halo_` | 54 → 54 | 0.398 | 0.505 | +0.107 | +27.0% | 78.8 |
| `gpu__compute_z_bcs_` | 9 → 9 | 0.055 | 0.055 | +0.000 | +0.5% | 99.5 |
| `gpu__fill_periodic_west_and_east_halo_` | 54 → 0 | 10.648 | — | — | — | — |
| **All kernels** | 269 → 473 | 384.506 | 469.342 | +84.835 | +22.1% | 81.9 |

## Compressible explicit (CE, Breeze)

Base `explicit_1gd` (1 GPU), compared `explicit_2g` (2 GPUs).

Float32 for all four models. Base is the one-GPU NCCL-distributed capture (--distributed=true, partition 1×1), so both cases take the distributed code path; a 1×1 partition has no halo buffers or NCCL, so communication kernels remain 2-GPU only. Every capture ran with environment.sh (JULIA_CUDA_MEMORY_POOL=none).

| Kernel | Launches/step 1→2 GPU | 1 GPU ms/step | 2 GPU ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `gpu_compute_potential_temperature_tendency_` | 3 → 3 | 39.130 | 39.078 | -0.052 | -0.1% | 100.1 |
| `gpu_compute_z_momentum_tendency_` | 3 → 3 | 34.091 | 34.131 | +0.040 | +0.1% | 99.9 |
| `gpu_compute_y_momentum_tendency_` | 3 → 3 | 33.459 | 33.487 | +0.029 | +0.1% | 99.9 |
| `gpu_compute_x_momentum_tendency_` | 3 → 3 | 33.070 | 33.086 | +0.016 | +0.0% | 100.0 |
| `gpu_compute_scalar_tendency_` | 3 → 3 | 33.056 | 33.073 | +0.017 | +0.1% | 99.9 |
| `gpu__ssp_rk3_substep_` | 18 → 18 | 30.235 | 30.238 | +0.003 | +0.0% | 100.0 |
| `ncclDevKernel_SendRecv` | 0 → 63 | — | 16.287 | — | — | — |
| `gpu_broadcast_kernel_cartesian` | 6 → 132 | 11.371 | 15.683 | +4.312 | +37.9% | 72.5 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 → 3 | 12.963 | 12.958 | -0.005 | -0.0% | 100.0 |
| `gpu__compute_temperature_and_pressure_` | 3 → 3 | 11.467 | 11.466 | -0.001 | -0.0% | 100.0 |
| `gpu__compute_velocities_` | 3 → 3 | 9.159 | 9.109 | -0.050 | -0.5% | 100.6 |
| `gpu__copy_from_buffer_` | 0 → 90 | — | 5.098 | — | — | — |
| `gpu__compute_density_tendency_` | 3 → 3 | 5.029 | 5.029 | +0.000 | +0.0% | 100.0 |
| `gpu__compute_total_density_` | 3 → 3 | 3.811 | 3.832 | +0.021 | +0.6% | 99.4 |
| `gpu__fill_periodic_south_and_north_halo_` | 63 → 63 | 1.789 | 1.624 | -0.164 | -9.2% | 110.1 |
| `gpu__fill_bottom_and_top_halo_` | 60 → 60 | 0.474 | 0.493 | +0.019 | +4.0% | 96.1 |
| `gpu__compute_z_bcs_` | 9 → 9 | 0.055 | 0.056 | +0.001 | +1.3% | 98.7 |
| `gpu__fill_periodic_west_and_east_halo_` | 63 → 0 | 12.443 | — | — | — | — |
| **All kernels** | 249 → 465 | 271.600 | 284.688 | +13.087 | +4.8% | 95.4 |
