"""Run and validate one Slurm case using the frozen study environment."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[3]
PROJECT = REPO / "benchmarking/distributed"


def build_plan():
    plan = []
    for dynamics in ("compressible_explicit", "compressible_splitexplicit"):
        for gpus, size, partition in ((1, "512x512x512", "1x1"), (2, "1024x512x512", "2x1"),
                                       (4, "1024x1024x512", "4x1"), (8, "4096x512x512", "8x1"),
                                       (16, "8192x512x512", "16x1")):
            for backend in (("MPI",) if gpus == 1 else ("MPI", "NCCL")):
                plan.append(dict(id=f"{dynamics}_{'SERIAL' if gpus == 1 else backend}_{gpus}g_{partition}",
                                 dynamics=dynamics, gpus=gpus, size=size, partition=partition, backend=backend,
                                 time_step=0.01 * 512 / max(map(int, size.split("x")))))
    return plan


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def check_sources(study):
    hashes = json.loads((study / "provenance/hashes.json").read_text())
    for name, digest in hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen source/environment changed: {name}")


def validate(case, directory, returncode):
    if returncode:
        raise RuntimeError(f"srun failed with exit code {returncode}")
    placements = set()
    for rank in range(case["gpus"]):
        log = (directory / f"rank_{rank}.log").read_text()
        placement = re.findall(r"PLACEMENT rank=(\d+) host=(\S+) gpu_uuid=(\S+) communicator=(\S+) threads=(\d+) pool=(\S+)", log)
        if len(placement) != 1:
            raise RuntimeError(f"Rank {rank}: missing/duplicate placement")
        found_rank, host, uuid, communicator, threads, pool = placement[0]
        expected = "none" if case["gpus"] == 1 else ("NCCLCommunicator" if case["backend"] == "NCCL" else "Comm")
        if (int(found_rank), communicator, threads, pool) != (rank, expected, "1", "none"):
            raise RuntimeError(f"Rank {rank}: unexpected placement/environment {placement}")
        placements.add((host, uuid))
        if not re.search(fr"VALIDATION rank={rank} fields_finite=true gpu_used_bytes=\d+", log):
            raise RuntimeError(f"Rank {rank}: finite-field validation missing")
    if len(placements) != case["gpus"]:
        raise RuntimeError("Ranks share a GPU")
    with (directory / "measurement.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError("Expected one measurement")
    row = rows[0]
    expected_backend = "SERIAL" if case["gpus"] == 1 else case["backend"]
    if row["fields_finite"] != "true" or row["communication"] != expected_backend:
        raise RuntimeError("Invalid finite-field/backend measurement")
    if int(row["steps"]) != 500 or int(row["gpus"]) != case["gpus"]:
        raise RuntimeError("Wrong number of steps or GPUs")
    if not all(math.isfinite(float(row[key])) and float(row[key]) > 0 for key in ("seconds_per_step", "points_per_second")):
        raise RuntimeError("Invalid timing")
    samples = []
    for path in directory.glob("memory_rank_*.csv"):
        for line in path.read_text().splitlines():
            columns = line.split(",")
            if len(columns) == 3:
                try:
                    samples.append(float(columns[2].strip()))
                except ValueError:
                    pass
    row["max_sampled_process_memory_mib"] = max(samples) if samples else ""
    return row


def main():
    study = Path(sys.argv[1]).resolve()
    case = next(item for item in json.loads((study / "plan.json").read_text()) if item["id"] == sys.argv[2])
    directory = study / "cases" / case["id"]
    directory.mkdir(parents=True, exist_ok=False)
    status = {"case": case, "started": timestamp(), "job_id": os.environ["SLURM_JOB_ID"], "state": "running"}
    write_json(directory / "status.json", status)
    try:
        check_sources(study)
        environment_command = 'module -t list 2>&1; julia --version; mpiexec --version; nvidia-smi; env | sort | rg "^(JULIA|OPENBLAS|OMP_|CUDA|UCX|OMPI|OCEAN|OPAL|SLURM|EBROOT|LOADEDMODULES)"'
        with (directory / "environment.log").open("w") as handle:
            subprocess.run(["bash", "-c", environment_command], stdout=handle, stderr=subprocess.STDOUT, check=True)
        command = ["srun", "--mpi=pmix", "--exact", "--time=00:30:00", "--kill-on-bad-exit=1", "--cpu-bind=none",
                   "--nodes=" + str(max(1, case["gpus"] // 4)), "--ntasks=" + str(case["gpus"]),
                   "--ntasks-per-node=" + str(min(4, case["gpus"])),
                   "bash", str(PROJECT / "rorqual/rank.sh"), str(directory),
                   "julia", "--startup-file=no", "--threads=1", "--project=" + str(PROJECT),
                   str(PROJECT / "distributed_cbl_benchmark.jl"),
                   "--dynamics=" + case["dynamics"], "--comm=" + case["backend"],
                   "--size=" + case["size"], "--partition=" + case["partition"],
                   "--float_type=Float32", "--simplified=false", "--time_step=" + str(case["time_step"]),
                   "--warmup_steps=10", "--time_steps=500", "--substeps=12", "--profile=false",
                   "--results_file=" + str(directory / "measurement.csv")]
        (directory / "command.sh").write_text(shlex.join(command) + "\n")
        with (directory / "launcher.log").open("w") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, timeout=30 * 60)
        status["measurement"] = validate(case, directory, result.returncode)
        check_sources(study)
        status["state"] = "validated"
    except Exception as error:
        status["state"] = "failed"
        status["error"] = str(error)
        raise
    finally:
        status["finished"] = timestamp()
        write_json(directory / "status.json", status)


if __name__ == "__main__":
    main()
