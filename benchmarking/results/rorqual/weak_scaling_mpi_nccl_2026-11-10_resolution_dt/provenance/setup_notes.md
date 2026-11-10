# Setup notes

The tracked setup script completed dependency installation, system MPI configuration, and precompilation.
Its final CUDA-functional check returned false on the login node, which has no GPU; CUDA-aware MPI
returned true. The series repeats the full GPU/MPI preflight inside the allocation before timing.
The first attempt on scratch storage was stopped due to severe I/O stalls. The retained environment
uses Julia 1.13.2 and an exclusive depot under /home/anditse; Project/Manifest package versions were preserved.
The Rorqual runner and AGENTS.md were committed before source hashes and the archive were captured.

Validation before submission: seven Python validator/matrix tests passed; the report smoke test
excluded failed cases, checked shared-baseline efficiency, and produced PNG/SVG artifacts in a temporary directory.
The Julia driver parsed successfully and CUDA exposes both free_memory and total_memory.
