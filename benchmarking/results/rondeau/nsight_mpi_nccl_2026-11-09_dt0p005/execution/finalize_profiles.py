"""Verify Nsight artifacts, preserve original logs, and commit this completed study."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess


STUDY = Path(__file__).resolve().parent.parent
REPO = STUDY.parents[3]
SCRATCH = Path("/local_scratch/ajtse/breeze_nsight_mpi_nccl_2026-11-09")


def main():
    state = json.loads((STUDY / "study.json").read_text())
    assert state["status"] == "COMPLETE"
    assert len(state["cases"]) == 6 and all(case["state"] == "COMPLETE" and case["fields_finite"] for case in state["cases"])
    profiles = json.loads((STUDY / "profile_manifest.json").read_text())
    assert len(profiles) == 18
    for profile in profiles:
        path = Path(profile["path"])
        assert path.stat().st_size == profile["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == profile["sha256"]
        with sqlite3.connect(Path(profile["sqlite"]).as_uri() + "?mode=ro", uri=True) as connection:
            assert connection.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0] == profile["captured_kernel_events"]
        assert profile["path"] in (STUDY / "README.md").read_text()
    controller_log = Path(os.environ.get("NSIGHT_CONTROLLER_LOG", str(STUDY / "execution/controller.log")))
    if controller_log.resolve() != (STUDY / "execution/controller.log").resolve():
        shutil.copy2(controller_log, STUDY / "execution/controller.log")
    originals = {}
    ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    for path in sorted(STUDY.rglob("*")):
        if not path.is_file() or path.suffix not in (".out", ".log"):
            continue
        relative = path.relative_to(STUDY)
        destination = SCRATCH / "original_logs" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        originals[str(relative)] = {"original_path": str(destination),
                                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
        text = ansi.sub("", path.read_text())
        path.write_text("\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n")
    (STUDY / "provenance/original_logs.json").write_text(json.dumps(originals, indent=2) + "\n")
    for path in STUDY.rglob("*"):
        if path.is_file() and path.suffix == ".csv":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()).rstrip() + "\n")
    validation_path = STUDY / "validation/results.json"
    validation = json.loads(validation_path.read_text())
    validation.update(all_capture_sizes_hashes_kernel_counts_and_readme_paths_verified=True,
                      original_logs=str(SCRATCH / "original_logs"),
                      finalized_utc=datetime.now(timezone.utc).isoformat())
    validation_path.write_text(json.dumps(validation, indent=2) + "\n")
    assert not list(STUDY.rglob("*.nsys-rep")) and not list(STUDY.rglob("*.sqlite"))
    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=REPO, text=True)
    if staged.strip():
        raise RuntimeError("Other staged work exists; automatic Nsight result commit withheld")
    subprocess.run(["git", "add", "--", str(STUDY)], cwd=REPO, check=True)
    subprocess.run(["git", "diff", "--cached", "--check", "--", str(STUDY)], cwd=REPO, check=True)
    subprocess.run(["git", "commit", "-m", "Record six Nsight MPI/NCCL captures using measured faster four-GPU partitions"],
                   cwd=REPO, check=True)
    print("Nsight study verified and committed; raw reports remain in local scratch.", flush=True)


if __name__ == "__main__":
    main()
