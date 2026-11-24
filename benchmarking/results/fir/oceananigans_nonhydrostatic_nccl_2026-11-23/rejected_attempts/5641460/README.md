# Rejected 1-GPU allocation 5641460

The retry on fc10512 (fc10612 excluded) also failed in the GPU preflight before any case started,
this time with `Out of GPU memory` from `cuDevicePrimaryCtxRetain`, that is, while creating the CUDA
context and before any model allocation. Together with 5468404 (no visible GPU) this points to the
one-GPU partial-node step, not a single node. The two-GPU cases, which reserve a full node, ran.

The next retry runs `run-size 1 --node-gpus=4`: one rank and one 512³ subdomain on one GPU, with all
four H100s of the node reserved and visible, exactly as the two-GPU cases. The unused GPUs do not enter
the measured grid, throughput, or efficiency. Source, environment, and plan are unchanged.
Finalizer 5641461 republished 3/6 cases; its log is here.

Full-node retry 5648839 (finalizer 5648840) was cancelled while pending, before it started, because its
batch script lacked the `run-size` line; it was resubmitted unchanged otherwise as 5648886 (finalizer 5648887).
