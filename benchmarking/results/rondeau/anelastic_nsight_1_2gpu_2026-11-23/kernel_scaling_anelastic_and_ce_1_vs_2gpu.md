# Kernel time per time step: scaling comparison

GPU execution time per time step, summed over all launches of each kernel and averaged
over the captured steps. Multi-rank cases use each kernel's slowest rank.
Difference = scaled − base. Efficiency = 100 × base / scaled (weak scaling: 100% means
the kernel takes as long with the same work per GPU). A dash marks a kernel that only
one case launches. Rows are sorted by the scaled case's time, then the base case's.

## Anelastic

Base `anelastic_1gd` (1 GPU), compared `anelastic_2g` (2 GPUs).

Base is the one-GPU NCCL-distributed capture (--distributed=true, partition 1×1; see ../distributed_1gpu_nsight_2026-11-25), so both cases take the distributed code path. A 1×1 partition has no halo-communication buffers or NCCL, so communication kernels remain 2-GPU only; the serial-vs-distributed baseline comparison is in ../distributed_1gpu_nsight_2026-11-25/kernel_scaling_serial_vs_distributed_1gpu.md.

| Kernel | Launches/step 1→2 GPU | 1 GPU ms/step | 2 GPU ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `gpu_permutedims_kernel_` | 12 → 12 | 75.335 | 74.157 | -1.178 | -1.6% | 101.6 |
| `ncclDevKernel_SendRecv` | 0 → 60 | — | 60.809 | — | — | — |
| `gpu_compute_potential_temperature_tendency_` | 3 → 3 | 36.953 | 37.035 | +0.083 | +0.2% | 99.8 |
| `gpu_compute_z_momentum_tendency_` | 3 → 3 | 35.409 | 35.428 | +0.019 | +0.1% | 99.9 |
| `gpu_compute_y_momentum_tendency_` | 3 → 3 | 32.828 | 32.849 | +0.021 | +0.1% | 99.9 |
| `gpu_compute_scalar_tendency_` | 3 → 3 | 32.678 | 32.724 | +0.046 | +0.1% | 99.9 |
| `gpu_compute_x_momentum_tendency_` | 3 → 3 | 32.573 | 32.608 | +0.035 | +0.1% | 99.9 |
| `gpu__ssp_rk3_substep_` | 15 → 15 | 25.194 | 25.176 | -0.018 | -0.1% | 100.1 |
| `vector_fft` | 12 → 12 | 18.937 | 18.937 | +0.001 | +0.0% | 100.0 |
| `gpu_broadcast_kernel_cartesian` | 11 → 119 | 15.142 | 18.791 | +3.649 | +24.1% | 80.6 |
| `gpu_solve_batched_tridiagonal_system_kernel_` | 3 → 3 | 14.451 | 14.458 | +0.006 | +0.0% | 100.0 |
| `gpu__pack_buffer_x_to_y_` | 0 → 3 | — | 13.020 | — | — | — |
| `gpu__pressure_correct_momentum_` | 3 → 3 | 11.388 | 11.407 | +0.020 | +0.2% | 99.8 |
| `scal_kernel_ref` | 6 → 6 | 9.338 | 9.338 | -0.000 | -0.0% | 100.0 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 → 3 | 8.682 | 8.833 | +0.151 | +1.7% | 98.3 |
| `gpu__compute_velocities_` | 3 → 3 | 8.091 | 8.014 | -0.077 | -1.0% | 101.0 |
| `gpu__compute_anelastic_source_term_` | 3 → 3 | 6.128 | 6.108 | -0.021 | -0.3% | 100.3 |
| `gpu__unpack_buffer_y_from_x_` | 0 → 3 | — | 5.055 | — | — | — |
| `gpu__unpack_buffer_x_from_y_` | 0 → 3 | — | 5.042 | — | — | — |
| `gpu__pack_buffer_y_to_x_` | 0 → 3 | — | 4.845 | — | — | — |
| `gpu__copy_from_buffer_` | 0 → 78 | — | 4.422 | — | — | — |
| `partial_mapreduce_grid` | 6 → 6 | 4.235 | 4.231 | -0.003 | -0.1% | 100.1 |
| `gpu__copy_real_component_` | 3 → 3 | 3.739 | 3.744 | +0.005 | +0.1% | 99.9 |
| `gpu__broadcast_kernel_` | 3 → 3 | 2.678 | 2.677 | -0.001 | -0.0% | 100.0 |
| `gpu__fill_periodic_south_and_north_halo_` | 54 → 54 | 1.535 | 1.466 | -0.069 | -4.5% | 104.7 |
| `gpu__fill_bottom_and_top_halo_` | 54 → 54 | 0.394 | 0.471 | +0.076 | +19.4% | 83.8 |
| `gpu__compute_z_bcs_` | 9 → 9 | 0.054 | 0.055 | +0.001 | +1.8% | 98.3 |
| `gpu__fill_periodic_west_and_east_halo_` | 54 → 0 | 10.650 | — | — | — | — |
| **All kernels** | 269 → 473 | 386.413 | 471.115 | +84.702 | +21.9% | 82.0 |

## Compressible explicit (CE)

Base `explicit_1gd` (1 GPU), compared `explicit_2g` (2 GPUs).

Base is the one-GPU NCCL-distributed capture (--distributed=true, partition 1×1; see ../distributed_1gpu_nsight_2026-11-25), so both cases take the distributed code path. A 1×1 partition has no halo-communication buffers or NCCL, so communication kernels remain 2-GPU only; the serial-vs-distributed baseline comparison is in ../distributed_1gpu_nsight_2026-11-25/kernel_scaling_serial_vs_distributed_1gpu.md.

| Kernel | Launches/step 1→2 GPU | 1 GPU ms/step | 2 GPU ms/step | Difference ms | Difference % | Efficiency % |
|---|---|---|---|---|---|---|
| `gpu_compute_potential_temperature_tendency_` | 3 → 3 | 39.147 | 39.126 | -0.021 | -0.1% | 100.1 |
| `gpu_compute_z_momentum_tendency_` | 3 → 3 | 34.150 | 34.138 | -0.012 | -0.0% | 100.0 |
| `gpu_compute_y_momentum_tendency_` | 3 → 3 | 33.479 | 33.496 | +0.016 | +0.0% | 100.0 |
| `gpu_compute_x_momentum_tendency_` | 3 → 3 | 33.094 | 33.096 | +0.002 | +0.0% | 100.0 |
| `gpu_compute_scalar_tendency_` | 3 → 3 | 33.060 | 33.084 | +0.025 | +0.1% | 99.9 |
| `gpu__ssp_rk3_substep_` | 18 → 18 | 30.273 | 30.270 | -0.003 | -0.0% | 100.0 |
| `ncclDevKernel_SendRecv` | 0 → 63 | — | 17.082 | — | — | — |
| `gpu_broadcast_kernel_cartesian` | 6 → 132 | 11.370 | 15.714 | +4.344 | +38.2% | 72.4 |
| `gpu__compute_auxiliary_thermodynamic_variables_` | 3 → 3 | 13.015 | 13.001 | -0.014 | -0.1% | 100.1 |
| `gpu__compute_temperature_and_pressure_` | 3 → 3 | 11.490 | 11.466 | -0.025 | -0.2% | 100.2 |
| `gpu__compute_velocities_` | 3 → 3 | 9.152 | 9.103 | -0.049 | -0.5% | 100.5 |
| `gpu__copy_from_buffer_` | 0 → 90 | — | 5.092 | — | — | — |
| `gpu__compute_density_tendency_` | 3 → 3 | 5.028 | 5.035 | +0.007 | +0.1% | 99.9 |
| `gpu__compute_total_density_` | 3 → 3 | 3.824 | 3.825 | +0.001 | +0.0% | 100.0 |
| `gpu__fill_periodic_south_and_north_halo_` | 63 → 63 | 1.772 | 1.609 | -0.164 | -9.2% | 110.2 |
| `gpu__fill_bottom_and_top_halo_` | 60 → 60 | 0.479 | 0.502 | +0.023 | +4.8% | 95.4 |
| `gpu__compute_z_bcs_` | 9 → 9 | 0.055 | 0.055 | -0.000 | -0.4% | 100.4 |
| `gpu__fill_periodic_west_and_east_halo_` | 63 → 0 | 12.481 | — | — | — | — |
| **All kernels** | 249 → 465 | 271.870 | 285.617 | +13.747 | +5.1% | 95.2 |
