"""Rerun both dynamics and compare two four-GPU partitions with the tracked driver."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone


STUDY = Path(__file__).resolve().parent.parent
REPO = STUDY.parents[3]
PREVIOUS = STUDY.parent / "weak_scaling_mpi_nccl_2026-11-08_dt0p02"
LAUNCHER = Path("/u/ajtse/software/oceananigans-hpc/tools/run_local.sh")
DRIVER = REPO / "benchmarking/distributed/distributed_cbl_benchmark.jl"
PROJECT = REPO / "benchmarking/distributed"


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def output(arguments):
    return subprocess.check_output(arguments, text=True).strip()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_snapshot():
    gpu_rows = list(csv.reader(output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]).splitlines(), skipinitialspace=True))
    processes = list(csv.reader(output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]).splitlines(), skipinitialspace=True))
    return gpu_rows, processes


def worker_record(pid):
    process = Path("/proc") / str(pid)
    try:
        environment = dict(item.split("=", 1) for item in
                           process.joinpath("environ").read_bytes().decode().split("\0")
                           if "=" in item)
        libraries = sorted({line.split()[-1] for line in process.joinpath("maps").read_text().splitlines()
                            if any(name in line for name in ("libmpi.so", "libnccl.so"))})
        return {"pid": int(pid), "executable": str(process.joinpath("exe").resolve()),
                "environment": {key: environment.get(key) for key in
                                ("CUDA_VISIBLE_DEVICES", "JULIA_NUM_THREADS", "JULIA_CUDA_MEMORY_POOL",
                                 "OMPI_COMM_WORLD_RANK")}, "library_mappings": libraries}
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return {"pid": int(pid), "unavailable": True}


def check_sources(hashes):
    changed = [name for name, digest in hashes.items() if sha256(Path(name)) != digest]
    if changed:
        raise RuntimeError(f"Source/environment changed during study: {changed}")


def main():
    if (STUDY / "study.json").exists() or (STUDY / "measurements.csv").exists():
        raise RuntimeError("Study already exists; refusing to duplicate or mix results")
    for folder in ("provenance", "runs", "validation"):
        (STUDY / folder).mkdir(exist_ok=True)
    source_names = output(["git", "-C", str(REPO), "ls-files", "src", "ext", "benchmarking/src"]).splitlines()
    paths = [REPO / name for name in source_names] + [
        DRIVER, LAUNCHER, REPO / "Project.toml", PROJECT / "Project.toml",
        PROJECT / "Manifest.toml", PROJECT / "LocalPreferences.toml",
        REPO / "benchmarking/distributed/pelee_rondeau/breeze_env.sh",
        REPO / "benchmarking/distributed/plot_split_explicit_results.py",
    ]
    hashes = {str(path): sha256(path) for path in paths}
    write_json(STUDY / "provenance/source_sha256.json", hashes)
    for name in ("Project.toml", "Manifest.toml", "LocalPreferences.toml"):
        shutil.copy2(PROJECT / name, STUDY / "provenance" / name)
        if sha256(PROJECT / name) != sha256(PREVIOUS / "provenance" / name):
            raise RuntimeError(f"Environment differs from previous study: {name}")
    (STUDY / "provenance/source_commit.txt").write_text(output(["git", "-C", str(REPO), "rev-parse", "HEAD"]) + "\n")
    (STUDY / "provenance/source_tree_before.txt").write_text(output(["git", "-C", str(REPO), "status", "--short"]) + "\n")
    archive_paths = source_names + ["Project.toml", "benchmarking/distributed/distributed_cbl_benchmark.jl"]
    subprocess.run(["tar", "-czf", str(STUDY / "provenance/source.tar.gz"), "-C", str(REPO), *archive_paths], check=True)
    prior = json.loads((PREVIOUS / "study.json").read_text())
    cases = [{key: case[key] for key in ("id", "dynamics", "communication", "gpus", "size", "partition")}
             for case in prior["cases"]]
    extra_cases = []
    for case in cases:
        if case["gpus"] == 4:
            case["id"] += "_2x2"
            extra = dict(case)
            extra["id"] = extra["id"].replace("_2x2", "_4x1")
            extra["partition"] = [4, 1]
            extra_cases.append(extra)
    cases.extend(extra_cases)
    for case in cases:
        case["state"] = "PENDING"
    state = {key: prior[key] for key in ("dynamics", "gpus", "cells_per_gpu", "grid_sizes", "partitions",
                                      "communications", "configured_cases", "float_type", "warmup_steps",
                                      "timed_steps", "acoustic_substeps", "julia_threads_per_rank", "cuda_memory_pool")}
    state.update(status="RUNNING", created_utc=timestamp(), time_step_seconds=0.005,
                 previous_study=PREVIOUS.name, cases=cases,
                 configured_cases=len(cases), four_gpu_partitions=[[2, 2], [4, 1]],
                 numerical_scope="Final prognostic and velocity field finiteness on every rank; no serial model agreement test",
                 gpu_poll_seconds=10)
    write_json(STUDY / "study.json", state)
    write_json(STUDY / "execution/controller_process.json",
               {"pid": os.getpid(), "started_utc": timestamp(), "command": ["python3", str(Path(__file__).resolve())]})
    sample_path = STUDY / "memory_samples.csv"
    with sample_path.open("w", newline="") as stream:
        csv.writer(stream).writerow(("utc", "case", "gpu", "used_mib", "utilization_percent", "compute_pids"))
    commands = STUDY / "execution/commands.log"
    commands.write_text("# Source execution/environment.sh, then run from each case's runs directory.\n")
    for case in cases:
        check_sources(hashes)
        while True:
            devices, processes = gpu_snapshot()
            busy = {row[0] for row in processes}
            free = [int(row[0]) for row in devices if row[1] not in busy]
            if len(free) >= case["gpus"]:
                gpu_ids = free[:case["gpus"]]
                break
            if case["state"] != "WAITING_FOR_GPUS":
                case.update(state="WAITING_FOR_GPUS", waiting_since_utc=timestamp())
                write_json(STUDY / "study.json", state)
                print(f'{timestamp()} WAITING {case["id"]}; free GPUs: {free}', flush=True)
            time.sleep(10)
        arguments = ["bash", str(LAUNCHER), "--wait", "-n", str(case["gpus"]),
                     "--gpu-ids", ",".join(map(str, gpu_ids)), "-p", str(PROJECT),
                     "--job-name", "dt0p005_" + case["id"], str(DRIVER),
                     "--comm=" + ("MPI" if case["communication"] == "SERIAL" else case["communication"]),
                     "--dynamics=" + case["dynamics"], "--size=" + "x".join(map(str, case["size"])),
                     "--partition=" + "x".join(map(str, case["partition"])), "--float_type=Float32",
                     "--warmup_steps=10", "--time_steps=500", "--time_step=0.005", "--substeps=12",
                     "--results_file=" + str(STUDY / "measurements.csv")]
        with commands.open("a") as stream:
            stream.write("cd " + shlex.quote(str(STUDY / "runs")) + "\n" + shlex.join(arguments) + "\n\n")
        case.update(state="RUNNING", started_utc=timestamp(), gpu_ids=gpu_ids, peak_mib={}, workers=[])
        case["prelaunch_gpu_processes"] = processes
        write_json(STUDY / "study.json", state)
        print(f'{timestamp()} START {case["id"]} GPUs={gpu_ids}', flush=True)
        launched = time.monotonic()
        launch_log = STUDY / "execution" / (case["id"] + ".log")
        with launch_log.open("w") as stream:
            process = subprocess.Popen(arguments, cwd=STUDY / "runs", stdout=stream, stderr=subprocess.STDOUT)
            while process.poll() is None:
                devices, processes = gpu_snapshot()
                seen = {worker["pid"]: worker for worker in case["workers"]}
                with sample_path.open("a", newline="") as sample_stream:
                    writer = csv.writer(sample_stream)
                    for gpu, uuid, used, utilization in devices:
                        if int(gpu) not in gpu_ids:
                            continue
                        pids = [int(row[1]) for row in processes if row[0] == uuid]
                        writer.writerow((timestamp(), case["id"], gpu, used, utilization, ";".join(map(str, pids))))
                        case["peak_mib"][gpu] = max(case["peak_mib"].get(gpu, 0), int(used))
                        for pid in pids:
                            record = worker_record(pid)
                            if "unavailable" not in record:
                                seen[pid] = record
                case["workers"] = list(seen.values())
                write_json(STUDY / "study.json", state)
                time.sleep(10)
        text = launch_log.read_text()
        log_match = re.search(r"^log:\s+(.*)$", text, re.MULTILINE)
        case.update(exit_code=process.returncode, finished_utc=timestamp(),
                    elapsed_wall_seconds=round(time.monotonic() - launched, 3))
        if log_match:
            log = Path(log_match[1].strip())
            case["log"] = str(log.relative_to(STUDY))
            benchmark_text = log.read_text()
            case["driver_exit_zero"] = bool(re.search(r"finished, exit status 0$", benchmark_text, re.MULTILINE))
        with (STUDY / "measurements.csv").open(newline="") as stream:
            rows = [row for row in csv.DictReader(stream)
                    if row["dynamics"] == case["dynamics"] and row["communication"] == case["communication"]
                    and int(row["gpus"]) == case["gpus"]
                    and [int(row["Rx"]), int(row["Ry"])] == case["partition"]]
        if process.returncode != 0 or not case.get("driver_exit_zero") or len(rows) != 1:
            case["state"] = "FAILED"
            state["status"] = "FAILED"
            write_json(STUDY / "study.json", state)
            raise RuntimeError(f'Case did not complete successfully: {case["id"]}')
        row = rows[0]
        finite = row["fields_finite"] == "true" and math.isfinite(float(row["max_abs_w"]))
        case.update(state="COMPLETE", seconds_per_step=float(row["seconds_per_step"]),
                    fields_finite=finite, max_abs_w=float(row["max_abs_w"]) if finite else row["max_abs_w"],
                    performance_valid=finite, numerical_status="FINITE_FIELDS" if finite else "NONFINITE_FIELDS")
        check_sources(hashes)
        write_json(STUDY / "study.json", state)
        print(f'{timestamp()} COMPLETE {case["id"]} seconds_per_step={case["seconds_per_step"]} finite={finite}', flush=True)
    state.update(status="COMPLETE", finished_utc=timestamp(), sources_unchanged=True)
    write_json(STUDY / "study.json", state)
    print(f'{timestamp()} COMPLETE all {len(cases)} cases', flush=True)


if __name__ == "__main__":
    main()
