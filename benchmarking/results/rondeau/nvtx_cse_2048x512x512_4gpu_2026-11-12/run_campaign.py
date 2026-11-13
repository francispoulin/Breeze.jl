"""Run the requested NCCL/MPI CBL benchmarks and NVTX/Nsight captures once GPUs are free."""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import sqlite3
import subprocess
import sys
import time


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verify_sources(root, hashes):
    for name, digest in hashes.items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen source changed: {path}")


def gpus_available(processes, inventory):
    try:
        rows = list(csv.reader(inventory.splitlines()))
        return (not processes.strip() and len(rows) == 4
                and {int(row[0]) for row in rows} == {0, 1, 2, 3}
                and all(int(row[1]) >= 35000 for row in rows))
    except (ValueError, IndexError):
        return False


def predecessor_ready(predecessor):
    if predecessor["status"] == "COMPLETE" or (
            predecessor.get("tests_status") == "PASS" and predecessor.get("completed_cases") == 6
            and predecessor.get("controller_exit_code") == 0):
        return True
    if predecessor["status"] in ("FAILED", "ERROR", "CANCELLED", "RESULTS_READY_PUSH_FAILED"):
        raise RuntimeError(f"Preceding smoke checks require review: {predecessor}")
    return False


def command_for(root, results, runtime, plan, backend, profiling):
    driver = root / "benchmarking/distributed/distributed_cbl_benchmark.jl"
    command = ["timeout", "--foreground", "--kill-after=30s", "1200s",
               "mpiexec", "--bind-to", "none", "-n", "4"]
    if profiling:
        command += ["bash", str(results / "profile_rank.sh"), str(runtime / "environment"),
                    plan["julia"], str(driver), str(runtime / "raw" / backend)]
    else:
        command += [plan["julia"], "--startup-file=no", "--threads=1",
                    "--project=" + str(runtime / "environment"), str(driver)]
    kind = "profiled" if profiling else "benchmark"
    command += ["--dynamics=compressible_splitexplicit", f"--comm={backend}",
                "--size=2048x512x512", "--partition=4x1", "--float_type=Float32",
                "--time_step=0.0025", "--substeps=12", "--warmup_steps=10", "--time_steps=10",
                "--simplified=false", "--random_perturbation=false", "--nvtx=true",
                "--profile=" + str(profiling).lower(),
                "--results_file=" + str(results / "logs" / f"{kind}_{backend}.csv")]
    return command


def validate_run(log, csv_path, backend):
    placement = re.findall(r"PLACEMENT rank=(\d+) host=(\S+) gpu_uuid=(\S+) communicator=(\S+)", log)
    validation = re.findall(r"VALIDATION rank=(\d+) fields_finite=(\S+)", log)
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    communicator = "NCCLCommunicator" if backend == "NCCL" else "Comm"
    if (len(placement) != 4 or {int(row[0]) for row in placement} != set(range(4))
            or len({row[2] for row in placement}) != 4
            or any(row[3] != communicator for row in placement)
            or len(validation) != 4 or {int(row[0]) for row in validation} != set(range(4))
            or any(row[1] != "true" for row in validation) or len(rows) != 1):
        raise RuntimeError("Missing or invalid four-rank placement, communicator, or finite-field checks")
    expected = dict(dynamics="compressible_splitexplicit", float="Float32", Nx="2048", Ny="512",
                    Nz="512", Rx="4", Ry="1", gpus="4", steps="10", communication=backend,
                    fields_finite="true")
    if any(rows[0].get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Unexpected measurement: {rows[0]}")
    if (not math.isfinite(float(rows[0]["max_abs_w"]))
            or not math.isfinite(float(rows[0]["seconds_per_step"]))
            or float(rows[0]["seconds_per_step"]) <= 0):
        raise RuntimeError(f"Invalid numeric measurement: {rows[0]}")
    return dict(placements=placement, rank_validation=validation, measurement=rows[0],
                launcher_messages=[line for line in log.splitlines() if "PMIX ERROR:" in line])


def describe_outcome(results, runtime, cases, state, error=None):
    lines = ["# Four-GPU split-explicit CBL: NVTX and Nsight", "", f"Status: **{state}**.", "",
             "Global grid 2048×512×512; partition 4×1; local grid 512³ per A100; Float32;",
             "full dry CBL with WENO5 and deterministic perturbations. Δt = 0.0025 s follows",
             "the resolution rule in AGENTS.md. Twelve nominal acoustic substeps give 22",
             "substeps per outer RK step. Each launch has ten warmup and ten measured steps.", "",
             "The order is NCCL benchmark, MPI benchmark, NCCL capture, MPI capture.",
             "MPI is the requested case without NCCL; NCCL still uses MPI for launch/control.",
             "Both install the existing Breeze host NVTX ranges. Nsight records CUDA, NVTX",
             "and MPI only during measured steps, using cudaProfilerApi start/stop calls.", "",
             "The detached checkout and Julia environment live under:", f"`{runtime}`", "",
             "## Timings and validation", "",
             "| Launch | State | Unprofiled seconds/step | Maximum absolute w |",
             "|--------|-------|-------------------------|--------------------|"]
    for case in cases:
        measurement = case.get("measurement", {})
        timing = measurement.get("seconds_per_step", "pending") if case["kind"] == "benchmark" else "see trace"
        lines.append(f"| {case['kind']} {case['backend']} | {case['state']} | {timing} | {measurement.get('max_abs_w', 'pending')} |")
    lines += ["", "Unprofiled wall timings and captured timings are kept separate. The driver's",
              "profiled timer can include report generation and must not be used as a simulation",
              "performance measurement. Trace windows, GPU activity and nested host ranges are",
              "reported per rank in analysis.json; nested ranges must not be added together.",
              "Finite-field checks establish a functional run, not field-by-field numerical accuracy.",
              "The source-review communication concerns in .agents/nvidia_optimization_advice.md",
              "remain unverified by these dry CBL captures.", "", "## Exact Nsight result paths", "",
              "These are planned paths until each report is recorded in captures.json.", ""]
    for backend in ("NCCL", "MPI"):
        for rank in range(4):
            lines.append(f"- `{runtime}/raw/{backend}/rank_{rank}/profile.nsys-rep`")
    lines += ["", "SQLite exports stay alongside the raw reports. All raw data, temporary files,",
              "MPI sessions, caches and environments stay under /u/ajtse; .runtime is ignored.",
              "The tracked report contains commands, environment/source provenance, per-launch",
              "logs and CSVs, per-rank Nsight summaries and report sizes/SHA-256 hashes.", "",
              "## Automatic execution", "",
              "run_campaign.py checks every 30 seconds for four GPUs with no compute processes",
              "and at least 35000 MiB free each. It waits for the previously queued four-GPU",
              "smoke checks to finish successfully, then runs each launch without sharing GPUs.",
              "It verifies the frozen source hashes before launches and after analysis, refuses",
              "to overwrite output, and records failures. A detached watcher survives this chat.",
              f"Live state: `{runtime}/status.json`.",
              f"Full controller log: `{runtime}/logs/controller.log`.", "",
              f"The live report directory is `{runtime}/worktree/{results.relative_to(results.parents[3])}`.",
              "The original checkout's README remains the initial queued snapshot until pulled.", "",
              "After completion it commits this report directory and immediately attempts a normal",
              "push to origin/NVTX-optimization. Concurrent remote changes cause a push failure",
              "recorded in status.json; it never force-pushes. No plots are produced for this run."]
    if error:
        lines += ["", "## Failure", "", error]
    (results / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--publish-results", action="store_true")
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    results = Path(__file__).resolve().parent
    root = results.parents[3]
    plan = json.loads((results / "plan.json").read_text())
    hashes = json.loads((results / "provenance/source_sha256.json").read_text())
    cases = []
    captures = []
    child = None

    def status(state, **details):
        write_json(runtime / "status.json", dict(status=state, pid=os.getpid(),
                   updated_utc=datetime.now(timezone.utc).isoformat(), results=str(results),
                   completed_launches=sum(case["state"] == "COMPLETE" for case in cases), **details))

    def stop(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        status("CANCELLED")
        raise SystemExit(128 + signum)

    def wait_for_gpus():
        while True:
            verify_sources(root, hashes)
            try:
                processes = subprocess.check_output(["nvidia-smi", "--id=0,1,2,3",
                    "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
                inventory = subprocess.check_output(["nvidia-smi",
                    "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
                if gpus_available(processes, inventory):
                    return
                status("WAITING_FOR_FOUR_FREE_GPUS", occupied_pids=processes.splitlines(), inventory=inventory.splitlines())
            except subprocess.CalledProcessError as error:
                status("WAITING_FOR_VALID_GPU_STATUS", error=str(error))
            time.sleep(30)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (runtime / "watcher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (runtime / "watcher.pid").write_text(str(os.getpid()) + "\n")
        failure = None
        try:
            if (results / "runs.json").exists():
                raise RuntimeError("This is a one-shot controller; choose fresh paths for another run")
            predecessor_path = Path(plan["predecessor_status"])
            while True:
                verify_sources(root, hashes)
                predecessor = json.loads(predecessor_path.read_text())
                if predecessor_ready(predecessor):
                    break
                status("WAITING_FOR_PRECEDING_SMOKE_CHECKS", predecessor_status=predecessor["status"])
                time.sleep(30)
            for profiling in (False, True):
                for backend in ("NCCL", "MPI"):
                    wait_for_gpus()
                    kind = "profiled" if profiling else "benchmark"
                    case = dict(kind=kind, backend=backend, state="RUNNING")
                    cases.append(case)
                    write_json(results / "runs.json", cases)
                    command = command_for(root, results, runtime, plan, backend, profiling)
                    case["command"] = command
                    prefix = results / "logs" / f"{kind}_{backend}"
                    if prefix.with_suffix(".csv").exists():
                        raise RuntimeError(f"Output already exists: {prefix}")
                    prefix.with_suffix(".command.sh").write_text(shlex.join(command) + "\n")
                    with prefix.with_suffix(".log").open("w") as output:
                        child = subprocess.Popen(command, cwd=root, stdout=output,
                                                 stderr=subprocess.STDOUT, start_new_session=True)
                        status("RUNNING", launch=f"{kind}_{backend}", controller_child_pid=child.pid)
                        case["exit_code"] = child.wait()
                    if case["exit_code"] != 0:
                        raise RuntimeError(f"{kind} {backend} failed with exit {case['exit_code']}")
                    log = prefix.with_suffix(".log").read_text()
                    case.update(validate_run(log, prefix.with_suffix(".csv"), backend))
                    if profiling:
                        record = dict(case=f"compressible_splitexplicit_{backend}_4g", backend=backend,
                                      state="RUNNING", steps=10, warmup=10, size=[2048, 512, 512],
                                      partition=[4, 1], time_step=0.0025, profiles=[])
                        captures.append(record)
                        devices = {int(row[0]): row[2].lower().removeprefix("gpu-") for row in case["placements"]}
                        for rank in range(4):
                            report = runtime / "raw" / backend / f"rank_{rank}" / "profile.nsys-rep"
                            database = report.with_suffix(".sqlite")
                            if not report.is_file() or report.stat().st_size == 0:
                                raise RuntimeError(f"Missing report: {report}")
                            with (results / "logs" / f"export_{backend}_{rank}.log").open("w") as output:
                                subprocess.run(["nsys", "export", "--type=sqlite", f"--output={database}", str(report)],
                                               check=True, stdout=output, stderr=subprocess.STDOUT, timeout=300)
                            with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
                                kernels = connection.execute("SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
                                ranges = connection.execute("SELECT count(*) FROM NVTX_EVENTS WHERE end IS NOT NULL").fetchone()[0]
                                known = {row[0].lower().removeprefix("gpu-") for row in connection.execute("SELECT uuid FROM TARGET_INFO_GPU")}
                            if not kernels or not ranges or devices[rank] not in known:
                                raise RuntimeError(f"Invalid CUDA/NVTX/device capture: {report}")
                            record["profiles"].append(dict(rank=rank, report=str(report), sqlite=str(database),
                                bytes=report.stat().st_size, sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
                                kernels=kernels, ranges=ranges, gpu_uuid=devices[rank]))
                            with (results / "logs" / f"summary_{backend}_{rank}.txt").open("w") as output:
                                subprocess.run(["nsys", "stats", "--report=nvtx_sum,nvtx_gpu_proj_sum,cuda_gpu_kern_sum,cuda_api_sum",
                                                str(database)], check=True, stdout=output, stderr=subprocess.STDOUT, timeout=300)
                        record["state"] = "COMPLETE"
                        write_json(results / "captures.json", captures)
                    case["state"] = "COMPLETE"
                    write_json(results / "runs.json", cases)
                    describe_outcome(results, runtime, cases, "RUNNING")
            status("ANALYZING")
            sys.path.insert(0, str(root / "benchmarking/distributed"))
            from analyze_nvtx import analyze
            analysis = []
            for capture in captures:
                for profile in capture["profiles"]:
                    metrics = analyze(profile["sqlite"], 10)
                    analysis.append(dict(case=capture["case"], rank=profile["rank"], **metrics))
            write_json(results / "analysis.json", analysis)
            verify_sources(root, hashes)
        except Exception as error:
            failure = str(error)
            if cases and cases[-1]["state"] != "COMPLETE":
                cases[-1].update(state="FAILED", error=failure)
            print(f"FAILED: {error}", flush=True)
        write_json(results / "runs.json", cases)
        write_json(results / "captures.json", captures)
        state = "FAILED" if failure else "COMPLETE"
        describe_outcome(results, runtime, cases, state, failure)
        publication = {}
        if args.publish_results:
            try:
                remote = subprocess.check_output(["git", "remote", "get-url", "--push", "origin"], cwd=root, text=True).strip()
                if remote != plan["origin_push_url"]:
                    raise RuntimeError("Push destination changed")
                if subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=root, text=True).strip():
                    raise RuntimeError("Unrelated staged files prevent automatic result publication")
                for path in results.rglob("*"):
                    if path.is_file():
                        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()).rstrip() + "\n")
                subprocess.run(["git", "add", "--", str(results.relative_to(root))], cwd=root, check=True)
                subprocess.run(["git", "diff", "--cached", "--check"], cwd=root, check=True)
                subprocess.run(["git", "commit", "-m", f"Record four-GPU NVTX campaign: {state}"], cwd=root, check=True)
                publication["results_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
                push = subprocess.run(["git", "push", "origin", "HEAD:refs/heads/NVTX-optimization"], cwd=root)
                publication["push_exit_code"] = push.returncode
                if push.returncode:
                    state = "RESULTS_READY_PUSH_FAILED"
            except Exception as error:
                publication["publication_error"] = str(error)
                state = "RESULTS_READY_PUBLICATION_FAILED"
        status(state, error=failure, **publication)
        return int(bool(failure or publication.get("push_exit_code") or publication.get("publication_error")))


if __name__ == "__main__":
    sys.exit(main())
