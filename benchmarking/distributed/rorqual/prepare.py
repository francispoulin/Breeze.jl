"""Freeze provenance and prepare (but do not submit) the agreed 18-case study."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from run_case import REPO, PROJECT, write_json, build_plan

study = Path(sys.argv[1]).resolve()
if (study / "plan.json").exists():
    raise RuntimeError("Study already prepared; use a fresh directory")
(study / "provenance").mkdir(parents=True, exist_ok=True)
(study / "cases").mkdir(exist_ok=True)
plan = build_plan()
write_json(study / "plan.json", plan)
files = subprocess.check_output(["git", "ls-files", "src", "ext", "benchmarking/src"], cwd=REPO, text=True).splitlines()
paths = [REPO / name for name in files] + [REPO / "AGENTS.md", REPO / "Project.toml", PROJECT / "distributed_cbl_benchmark.jl"]
paths += list((PROJECT / "rorqual").glob("*.py")) + list((PROJECT / "rorqual").glob("*.sh"))
paths += [PROJECT / name for name in ("Project.toml", "Manifest.toml", "LocalPreferences.toml")]
paths += [Path("/home/anditse/Oceananigans-DRAC/env") / name for name in ("ocean_setup.sh", "drac.sh", "julia.sh")]
write_json(study / "provenance/hashes.json", {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
for name in ("Project.toml", "Manifest.toml", "LocalPreferences.toml"):
    shutil.copy2(PROJECT / name, study / "provenance" / name)
for arguments, name in ((["git", "rev-parse", "HEAD"], "source_commit.txt"), (["git", "status", "--short"], "source_status.txt"),
                        (["git", "diff", "--binary"], "source_changes.patch")):
    (study / "provenance" / name).write_text(subprocess.check_output(arguments, cwd=REPO, text=True))
subprocess.run(["tar", "--exclude=__pycache__", "-czf", str(study / "provenance/source.tar.gz"), "-C", str(REPO),
                "src", "ext", "AGENTS.md", "Project.toml", "benchmarking/src", "benchmarking/distributed"], check=True)
print(f"Prepared {len(plan)} cases in {study}")
