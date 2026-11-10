"""Publish verified progress, then archive logs and commit the completed study."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile


STUDY = Path(__file__).resolve().parent.parent
REPO = STUDY.parents[3]


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    previous = None
    while True:
        state = json.loads((STUDY / "study.json").read_text())
        signature = (state["status"], tuple(case["state"] for case in state["cases"]))
        if signature != previous:
            result = subprocess.run([sys.executable, str(STUDY / "execution/analyze_results.py")])
            if result.returncode == 0:
                previous = signature
                print(timestamp(), "Report updated", signature, flush=True)
        if state["status"] == "COMPLETE":
            if result.returncode != 0:
                raise RuntimeError("Final validation failed; results were not committed")
            break
        if state["status"] == "FAILED":
            raise RuntimeError("A benchmark case failed; inspect study.json and logs")
        record = json.loads((STUDY / "execution/controller_process.json").read_text())
        process = Path("/proc") / str(record["pid"])
        try:
            command = process.joinpath("cmdline").read_bytes().decode()
        except FileNotFoundError:
            command = ""
        if "run_series.py" not in command:
            raise RuntimeError("Controller exited before study completion; results remain partial")
        time.sleep(20)
    hashes = {}
    logs = sorted(path for path in STUDY.rglob("*") if path.suffix in (".log", ".out"))
    with zipfile.ZipFile(STUDY / "original_logs.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in logs:
            name = str(path.relative_to(STUDY))
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.write(path, name)
    (STUDY / "original_logs_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    with zipfile.ZipFile(STUDY / "original_logs.zip") as archive:
        assert archive.testzip() is None
        assert all(hashlib.sha256(archive.read(name)).hexdigest() == expected for name, expected in hashes.items())
    ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    for path in logs:
        text = ansi.sub("", path.read_text())
        path.write_text("\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n")
    for path in STUDY.rglob("*"):
        if not path.is_file() or path.suffix in (".png", ".gz", ".zip", ".pyc"):
            continue
        text = path.read_text()
        path.write_text("\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n")
    for directory in STUDY.rglob("__pycache__"):
        shutil.rmtree(directory)
    subprocess.run([sys.executable, str(STUDY / "execution/analyze_results.py")], check=True)
    validation = json.loads((STUDY / "validation/results.json").read_text())
    assert validation["complete_case_coverage"]
    assert validation["valid_performance_cases"] == 14, "Non-finite cases remain; automatic success commit withheld"
    from PIL import Image
    import xml.etree.ElementTree as ET
    for path in STUDY.glob("*.png"):
        with Image.open(path) as picture:
            picture.verify()
    for path in STUDY.glob("*.svg"):
        ET.parse(path)
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()).rstrip() + "\n")
    report = STUDY / "README.md"
    report.write_text(report.read_text().rstrip() + "\n\nOriginal logs are preserved byte-for-byte in [original_logs.zip](original_logs.zip) with\n"
                      "[SHA-256 hashes](original_logs_sha256.json); display logs have ANSI escapes and trailing whitespace removed.\n")
    validation.update(original_log_archive_hashes_verified=True, png_svg_integrity_verified=True,
                      finalized_utc=timestamp())
    (STUDY / "validation/results.json").write_text(json.dumps(validation, indent=2) + "\n")
    subprocess.run(["git", "diff", "--check", "--", str(STUDY)], cwd=REPO, check=True)
    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=REPO, text=True)
    if staged.strip():
        raise RuntimeError("Other staged work exists; automatic study commit withheld")
    subprocess.run(["git", "add", "--", str(STUDY)], cwd=REPO, check=True)
    retained = [str(path) for path in STUDY.iterdir() if path.suffix in (".png", ".svg", ".zip")]
    if retained:
        subprocess.run(["git", "add", "-f", "--", *retained], cwd=REPO, check=True)
    subprocess.run(["git", "diff", "--cached", "--check", "--", str(STUDY)], cwd=REPO, check=True)
    subprocess.run(["git", "commit", "-m", "Record MPI/NCCL scaling at timestep 0.005 with both four-GPU partitions"],
                   cwd=REPO, check=True)
    print(timestamp(), "Study complete, validated, and committed", flush=True)


if __name__ == "__main__":
    main()
