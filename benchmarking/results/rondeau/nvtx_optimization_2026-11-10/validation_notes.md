# Validation notes

`validate_nvtx.jl` passed 81 assertions: 26 uninstrumented repeat-control field
comparisons and 55 checks covering exact field equality, finite fields and matching
field keys after NVTX instrumentation. It covers serial anelastic, explicit and
split-explicit GPU dynamics on deterministic 16³ CBL models, three steps at Δt=0.005 s.
The second pass activates the NVTX API, so it also exercises the active range path.

An initial check rebuilt separate models and failed four exact anelastic field
comparisons while passing 51 checks. The final check resets and reuses each model
and solver plan to isolate instrumentation from model construction. Both its
uninstrumented repeat controls and annotated comparisons are bitwise identical.
This does not establish bitwise reproducibility across separately constructed
anelastic models. The initial diagnostic is retained in
`/local_scratch/ajtse/breeze_nvtx_2026-11-10/separate_model_validation.log`.

A separate 64³ split-explicit pilot under Nsight captured two steps after two
warmups. It has 2060 GPU kernels, all attributed to Breeze NVTX ranges, 44 acoustic
substep calls and finite fields. The official Nsight `nvtx_gpu_proj_sum` report
independently reports two outer steps and 44 acoustic substeps. Its 2072 GPU
operations include 12 memcpy operations in addition to the 2060 kernels.

An asynchronous synthetic trace checks that the Python analyzer follows CUDA
correlation IDs when kernels execute after their host ranges end, attributes
nested phases exclusively, and merges overlapping GPU intervals for activity.

The production capture controller checks finite prognostic and velocity fields on
every rank, nonempty traces and CUDA/NVTX tables, and distinct physical GPU UUIDs
across ranks. These are profiling checks; they do not compare the distributed
solution against a serial solution.

The first four-rank MPI capture completed with finite fields, but the controller
rejected its GPU-identity metadata. Nsight 2024.5 reports local device ordinal 0
for each process when `CUDA_VISIBLE_DEVICES` exposes only one GPU; joining that
ordinal to Nsight's physical GPU table falsely assigned GPU 0 to every rank.
The fixed controller reads the active device UUID directly from Julia and checks
it against the trace's hardware inventory. Distributed captures are recollected
with all four devices visible, letting Oceananigans select one per local rank
and retaining peer visibility. The initial MPI attempt is excluded from analysis.
