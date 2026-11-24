# Rejected 1-GPU allocation 5468404

Slurm allocated one H100 on fc10612 (`AllocTRES ... gres/gpu:h100=1`), but the GPU preflight
step could not see it: `ArgumentError: ... a CUDA GPU was not found!` No case started, so no
measurement was produced or rejected. The identical 1-GPU submission ran correctly on fc10516 on
12 November, so the node is excluded from the retry. The 2-GPU allocation 5468405 was unaffected.
Finalizer 5468406 published the three validated 2-GPU cases and reported 3/6; its log is here.
The empty `not_run` case records it wrote for the 1-GPU cases were removed so the retry can create them.
