# Kernel time per time step: scaling comparison

GPU execution time per time step, summed over all launches of each kernel and averaged
over the captured steps. Multi-rank cases use each kernel's slowest rank.
Difference = scaled − base. Efficiency = 100 × base / scaled (weak scaling: 100% means
the kernel takes as long with the same work per GPU). A dash marks a kernel that only
one case launches. Rows are sorted by the scaled case's time, then the base case's.

## Hydrostatic

Base `hydrostatic_1gd` (1 GPU), compared `hydrostatic_2g` (2 GPUs).

Base is the one-GPU NCCL-distributed capture (--distributed=true, partition 1×1; see ../distributed_1gpu_nsight_2026-11-25), so both cases take the distributed code path. A 1×1 partition has no halo-communication buffers or NCCL, so communication kernels remain 2-GPU only; the serial-vs-distributed baseline comparison is in ../distributed_1gpu_nsight_2026-11-25/kernel_scaling_serial_vs_distributed_1gpu.md.

| Kernel | Launches/step 1→2 GPU | 1 GPU ms/step | 2 GPU ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `gpu_compute_hydrostatic_free_surface_Gv_` | 3 → 9 | 36.147 | 38.368 | +2.221 | +6.1% | 94.2 |
| `gpu_compute_hydrostatic_free_surface_Gu_` | 3 → 9 | 35.618 | 38.292 | +2.674 | +7.5% | 93.0 |
| `gpu_compute_hydrostatic_free_surface_Gc_` | 3 → 9 | 34.946 | 37.181 | +2.235 | +6.4% | 94.0 |
| `gpu__compute_w_from_continuity_` | 6 → 6 | 25.425 | 25.494 | +0.070 | +0.3% | 99.7 |
| `gpu__compute_barotropic_mode_` | 6 → 6 | 15.191 | 15.661 | +0.470 | +3.1% | 97.0 |
| `gpu__barotropic_split_explicit_corrector_` | 3 → 3 | 14.378 | 14.394 | +0.016 | +0.1% | 99.9 |
| `gpu__rk_substep_field_` | 6 → 6 | 14.328 | 14.351 | +0.023 | +0.2% | 99.8 |
| `gpu__compute_split_explicit_transport_velocities_` | 3 → 3 | 13.203 | 13.250 | +0.047 | +0.4% | 99.6 |
| `ncclDevKernel_SendRecv` | 0 → 30 | — | 12.756 | — | — | — |
| `gpu__compute_realized_barotropic_tendency_` | 3 → 3 | 11.472 | 11.546 | +0.074 | +0.6% | 99.4 |
| `gpu__update_hydrostatic_pressure_` | 3 → 9 | 9.757 | 11.238 | +1.481 | +15.2% | 86.8 |
| `gpu__rk_substep_tracer_field_` | 3 → 3 | 7.167 | 7.177 | +0.009 | +0.1% | 99.9 |
| `gpu_broadcast_kernel_cartesian` | 14 → 74 | 3.975 | 4.856 | +0.881 | +22.2% | 81.9 |
| `gpu__cache_tracer_fields_` | 1 → 1 | 1.644 | 1.645 | +0.000 | +0.0% | 100.0 |
| `gpu__copy_from_buffer_` | 0 → 60 | — | 1.173 | — | — | — |
| `gpu__split_explicit_barotropic_velocity_` | 63 → 63 | 0.705 | 0.831 | +0.126 | +17.9% | 84.8 |
| `gpu__split_explicit_free_surface_` | 63 → 63 | 0.721 | 0.804 | +0.082 | +11.4% | 89.8 |
| `gpu__fill_periodic_south_and_north_halo_` | 33 → 33 | 0.393 | 0.420 | +0.026 | +6.7% | 93.7 |
| `gpu__fill_bottom_and_top_halo_` | 12 → 12 | 0.119 | 0.110 | -0.010 | -8.0% | 108.8 |
| `gpu_fill_kernel_` | 15 → 15 | 0.041 | 0.043 | +0.002 | +4.7% | 95.5 |
| `gpu__update_split_explicit_state_` | 3 → 3 | 0.023 | 0.025 | +0.001 | +4.8% | 95.4 |
| `gpu__fill_periodic_west_and_east_halo_` | 33 → 0 | 1.875 | — | — | — | — |
| **All kernels** | 279 → 420 | 227.131 | 247.527 | +20.396 | +9.0% | 91.8 |

## Nonhydrostatic

Base `nonhydrostatic_1gd` (1 GPU), compared `nonhydrostatic_2g` (2 GPUs).

Base is the one-GPU NCCL-distributed capture (--distributed=true, partition 1×1; see ../distributed_1gpu_nsight_2026-11-25), so both cases take the distributed code path. A 1×1 partition has no halo-communication buffers or NCCL, so communication kernels remain 2-GPU only; the serial-vs-distributed baseline comparison is in ../distributed_1gpu_nsight_2026-11-25/kernel_scaling_serial_vs_distributed_1gpu.md.

| Kernel | Launches/step 1→2 GPU | 1 GPU ms/step | 2 GPU ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `ncclDevKernel_SendRecv` | 0 → 30 | — | 150.650 | — | — | — |
| `gpu_permutedims_kernel_` | 12 → 12 | 87.140 | 87.590 | +0.450 | +0.5% | 99.5 |
| `gpu_broadcast_kernel_cartesian` | 9 → 57 | 43.075 | 44.859 | +1.784 | +4.1% | 96.0 |
| `gpu_compute_Gu_` | 3 → 9 | 36.481 | 40.160 | +3.679 | +10.1% | 90.8 |
| `gpu_compute_Gv_` | 3 → 9 | 36.943 | 40.152 | +3.209 | +8.7% | 92.0 |
| `gpu_compute_Gw_` | 3 → 9 | 37.093 | 39.469 | +2.376 | +6.4% | 94.0 |
| `vector_fft` | 12 → 12 | 37.379 | 37.445 | +0.066 | +0.2% | 99.8 |
| `gpu_compute_Gc_` | 3 → 9 | 35.264 | 37.387 | +2.123 | +6.0% | 94.3 |
| `gpu__rk3_substep_field_` | 12 → 12 | 36.585 | 36.620 | +0.035 | +0.1% | 99.9 |
| `scal_kernel_ref` | 9 → 9 | 28.033 | 28.060 | +0.027 | +0.1% | 99.9 |
| `regular_fft` | 6 → 6 | 20.276 | 20.535 | +0.259 | +1.3% | 98.7 |
| `gpu__cache_field_tendencies_` | 12 → 12 | 19.774 | 19.795 | +0.020 | +0.1% | 99.9 |
| `gpu__make_pressure_correction_` | 3 → 3 | 19.589 | 19.487 | -0.103 | -0.5% | 100.5 |
| `gpu__pack_buffer_x_to_y_` | 0 → 3 | — | 16.293 | — | — | — |
| `gpu__update_hydrostatic_pressure_` | 3 → 9 | 9.652 | 12.215 | +2.563 | +26.6% | 79.0 |
| `gpu__compute_source_term_` | 3 → 3 | 11.717 | 11.904 | +0.187 | +1.6% | 98.4 |
| `gpu__unpack_buffer_y_from_x_` | 0 → 3 | — | 9.743 | — | — | — |
| `gpu__spectral_poisson_solve_` | 3 → 3 | 9.483 | 9.460 | -0.023 | -0.2% | 100.2 |
| `gpu__pack_buffer_y_to_x_` | 0 → 3 | — | 9.409 | — | — | — |
| `gpu__unpack_buffer_x_from_y_` | 0 → 3 | — | 9.358 | — | — | — |
| `gpu_unpermute_z_indices_` | 3 → 3 | 9.318 | 9.335 | +0.017 | +0.2% | 99.8 |
| `gpu_permute_z_indices_` | 3 → 3 | 9.288 | 9.319 | +0.032 | +0.3% | 99.7 |
| `gpu__copy_real_component_` | 3 → 3 | 7.169 | 7.217 | +0.047 | +0.7% | 99.3 |
| `gpu__divide_by_` | 3 → 3 | 5.858 | 5.861 | +0.003 | +0.1% | 99.9 |
| `gpu__copy_from_buffer_` | 0 → 48 | — | 2.673 | — | — | — |
| `gpu__fill_periodic_south_and_north_halo_` | 27 → 27 | 0.756 | 0.721 | -0.035 | -4.7% | 104.9 |
| `gpu__fill_bottom_and_top_halo_` | 24 → 24 | 0.254 | 0.233 | -0.020 | -8.0% | 108.7 |
| `gpu__fill_periodic_west_and_east_halo_` | 27 → 0 | 3.981 | — | — | — | — |
| **All kernels** | 186 → 327 | 505.109 | 715.756 | +210.647 | +41.7% | 70.6 |
