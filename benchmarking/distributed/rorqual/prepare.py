"""Freeze provenance and prepare (but do not submit) the agreed 18-case study."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from run_case import REPO, PROJECT, write_json

study = Path(sys.argv[1]).resolve()
if (study / "plan.json").exists():
    raise RuntimeError("Study already prepared; use a fresh directory")
(study / "provenance").mkdir(parents=True, exist_ok=True)
(study / "cases").mkdir(exist_ok=True)
plan = []
for dynamics in ("compressible_explicit", "compressible_splitexplicit"):
    for gpus, size, partition in ((1, "512x512x512", "1x1"), (2, "1024x512x512", "2x1"),
                                   (4, "1024x1024x512", "4x1"), (8, "4096x512x512", "8x1"),
                                   (16, "8192x512x512", "16x1")):
        for backend in (("MPI",) if gpus == 1 else ("MPI", "NCCL")):
            plan.append(dict(id=f"{dynamics}_{'SERIAL' if gpus == 1 else backend}_{gpus}g_{partition}",
                             dynamics=dynamics, gpus=gpus, size=size, partition=partition, backend=backend))
write_json(study / "plan.json", plan)
files = subprocess.check_output(["git", "ls-files", "src", "ext", "benchmarking/src"], cwd=REPO, text=True).splitlines()
paths = [REPO / name for name in files] + [REPO / "Project.toml", PROJECT / "distributed_cbl_benchmark.jl"]
paths += list((PROJECT / "rorqual").glob("*.py")) + list((PROJECT / "rorqual").glob("*.sh"))
paths += [PROJECT / name for name in ("Project.toml", "Manifest.toml", "LocalPreferences.toml")]
paths += [Path("/home/anditse/Oceananigans-DRAC/env") / name for name in ("ocean_setup.sh", "drac.sh", "julia.sh")]
write_json(study / "provenance/hashes.json", {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
for name in ("Project.toml", "Manifest.toml", "LocalPreferences.toml"):
    shutil.copy2(PROJECT / name, study / "provenance" / name)
for arguments, name in ((["git", "rev-parse", "HEAD"], "source_commit.txt"), (["git", "status", "--short"], "source_status.txt"),
                        (["git", "diff", "--binary"], "source_changes.patch")):
    (study / "provenance" / name).write_text(subprocess.check_output(arguments, cwd=REPO, text=True))
subprocess.run(["tar", "-czf", str(study / "provenance/source.tar.gz"), "-C", str(REPO),
                "src", "ext", "Project.toml", "benchmarking/src", "benchmarking/distributed"], check=True)
print(f"Prepared {len(plan)} cases in {study}")
