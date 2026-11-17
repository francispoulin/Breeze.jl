"""Two-GPU NCCL timings and NVTX/Nsight captures of Breeze anelastic and Oceananigans
nonhydrostatic, launched from a frozen checkout once two GPUs are free."""

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

CASES = ("anelastic", "oceananigans_nonhydrostatic")
RANKS = 2


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verify_frozen(checkout, commit):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=checkout, text=True).strip()
    if head != commit or dirty:
        raise RuntimeError(f"Frozen checkout changed: HEAD={head} dirty={bool(dirty)}")


def free_gpu_pair(minimum_free_mib):
    """Return the two lowest-index GPUs with no compute processes and enough free memory."""
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free",
                                         "--format=csv,noheader,nounits"], text=True)
    busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid",
                                    "--format=csv,noheader"], text=True)
    busy = {line.strip() for line in busy.splitlines() if line.strip()}
    free = []
    for row in csv.reader(inventory.splitlines()):
        index, uuid, memory = (field.strip() for field in row)
        if uuid not in busy and int(memory) >= minimum_free_mib:
            free.append(index)
    return (free[:RANKS] if len(free) >= RANKS else None), inventory.splitlines(), sorted(busy)


def command_for(checkout, results, runtime, plan, case, profiling):
    driver = checkout / "benchmarking/distributed/distributed_cbl_benchmark.jl"
    command = ["timeout", "--foreground", "--kill-after=30s", "2400s",
               "mpiexec", "--bind-to", "none", "-n", str(RANKS)]
    if profiling:
        command += ["bash", str(results / "profile_rank.sh"), str(runtime / "environment"),
                    plan["julia"], str(driver), str(runtime / "raw" / case)]
    else:
        command += [plan["julia"], "--startup-file=no", "--threads=1",
                    "--project=" + str(runtime / "environment"), str(driver)]
    kind = "profiled" if profiling else "benchmark"
    command += plan["cases"][case]["arguments"]
    command += ["--comm=NCCL", "--size=1024x512x512", "--partition=2x1",
                f"--warmup_steps={plan['warmup_steps']}", f"--time_steps={plan['measured_steps']}",
                "--nvtx=true", "--profile=" + str(profiling).lower(),
                "--results_file=" + str(results / "logs" / f"{kind}_{case}.csv")]
    return command


def validate_run(log, csv_path, case, plan):
    placement = re.findall(r"PLACEMENT rank=(\d+) host=(\S+) gpu_uuid=(\S+) communicator=(\S+)", log)
    validation = re.findall(r"VALIDATION rank=(\d+) fields_finite=(\S+) gpu_used_bytes=(\d+)", log)
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    if (len(placement) != RANKS or {int(row[0]) for row in placement} != set(range(RANKS))
            or len({row[2] for row in placement}) != RANKS
            or any(row[3] != "NCCLCommunicator" for row in placement)
            or len(validation) != RANKS or any(row[1] != "true" for row in validation)
            or len(rows) != 1):
        raise RuntimeError("Missing or invalid two-rank placement, NCCL communicator, or finite-field checks")
    float_type = next(a for a in plan["cases"][case]["arguments"] if a.startswith("--float_type="))
    expected = dict(dynamics=case, float=float_type.split("=")[1], Nx="1024", Ny="512", Nz="512",
                    Rx="2", Ry="1", gpus="2", steps=str(plan["measured_steps"]),
                    communication="NCCL", fields_finite="true")
    if any(rows[0].get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Unexpected measurement: {rows[0]}")
    if (not math.isfinite(float(rows[0]["max_abs_w"]))
            or not math.isfinite(float(rows[0]["seconds_per_step"]))
            or float(rows[0]["seconds_per_step"]) <= 0):
        raise RuntimeError(f"Invalid numeric measurement: {rows[0]}")
    return dict(placements=placement, rank_validation=validation, measurement=rows[0])


def record_capture(results, runtime, case, run, plan):
    record = dict(case=f"{case}_NCCL_2g", state="RUNNING", steps=plan["measured_steps"],
                  source_commit=run["source_commit"], profiles=[])
    devices = {int(row[0]): row[2].lower().removeprefix("gpu-") for row in run["placements"]}
    for rank in range(RANKS):
        report = runtime / "raw" / case / f"rank_{rank}" / "profile.nsys-rep"
        database = report.with_suffix(".sqlite")
        if not report.is_file() or report.stat().st_size == 0:
            raise RuntimeError(f"Missing report: {report}")
        with (results / "logs" / f"export_{case}_{rank}.log").open("w") as output:
            subprocess.run(["nsys", "export", "--type=sqlite", f"--output={database}", str(report)],
                           check=True, stdout=output, stderr=subprocess.STDOUT, timeout=900)
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            kernels = connection.execute("SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
            ranges = connection.execute("SELECT count(*) FROM NVTX_EVENTS WHERE end IS NOT NULL").fetchone()[0]
            known = {row[0].lower().removeprefix("gpu-")
                     for row in connection.execute("SELECT uuid FROM TARGET_INFO_GPU")}
        if not kernels or not ranges or devices[rank] not in known:
            raise RuntimeError(f"Invalid CUDA/NVTX/device capture: {report}")
        record["profiles"].append(dict(rank=rank, report=str(report), sqlite=str(database),
                                       bytes=report.stat().st_size,
                                       sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
                                       kernels=kernels, ranges=ranges, gpu_uuid=devices[rank]))
        with (results / "logs" / f"summary_{case}_{rank}.txt").open("w") as output:
            subprocess.run(["nsys", "stats", "--report=nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum",
                            str(database)], check=True, stdout=output, stderr=subprocess.STDOUT, timeout=900)
    record["state"] = "COMPLETE"
    return record


def analysis_rows(results):
    path = results / "rank_summary.csv"
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def describe_outcome(results, runtime, plan, runs, state, error=None):
    lines = ["# Two-GPU NCCL Nsight: Breeze anelastic and Oceananigans nonhydrostatic", "",
             f"Status: **{state}**.", "",
             "Global grid 1024×512×512, partition 2×1, 512³ cells per A100, NCCL halo and",
             "transpose communication (MPI only launches ranks and supplies barriers). Each launch",
             "has ten warmup and ten measured steps and installs host NVTX ranges.", "",
             "| Case | Model | Precision | Δt |", "|------|-------|-----------|----|",
             "| `anelastic` | Breeze `AtmosphereModel`, `AnelasticDynamics`, dry CBL, WENO5 | Float32 | 0.005 s |",
             "| `oceananigans_nonhydrostatic` | Oceananigans `NonhydrostaticModel`, suite `rectilinear_buoyancy`, WENO5 | Float64 | 10 s |",
             "", "Both pressure solves are Oceananigans' distributed FFT solvers, whose transposes",
             "use NCCL here. Δt = 0.005 s follows the AGENTS.md rule for Breeze at this grid;",
             "the Oceananigans case keeps its suite's Float64 and Δt = 10 s (1 km × 1 km × 10 m",
             "cells), so its cost per step matches the Oceananigans weak-scaling series. Both use",
             "Oceananigans 0.117.2 from this benchmark environment, not the Oceananigans checkout.",
             "Compare the two models by where time goes, not by absolute speed.", "",
             "For each case the order is unprofiled NCCL benchmark, then NCCL Nsight capture.",
             "Nsight records CUDA, NVTX and MPI only during the measured steps",
             "(cudaProfilerApi start/stop).", "",
             "## Timings and validation", "",
             "| Launch | State | Unprofiled seconds/step | Maximum absolute w | GPUs |",
             "|--------|-------|-------------------------|--------------------|------|"]
    for run in runs:
        measurement = run.get("measurement", {})
        timing = measurement.get("seconds_per_step", "pending") if run["kind"] == "benchmark" else "see trace"
        lines.append(f"| {run['kind']} {run['case']} | {run['state']} | {timing} | "
                     f"{measurement.get('max_abs_w', 'pending')} | {run.get('cuda_visible_devices', '')} |")
    rows = analysis_rows(results)
    if rows:
        lines += ["", "## Trace summary (per rank, from analysis.json)", "",
                  "| Case | Rank | Trace s/step | Kernel launches | GPU active fraction | Halo kernel fraction | Solve kernel fraction |",
                  "|------|------|--------------|-----------------|---------------------|----------------------|-----------------------|"]
        for row in rows:
            lines.append(f"| {row['case']} | {row['rank']} | {float(row['trace_seconds_per_step']):.4f} | "
                         f"{row['kernel_launches']} | {float(row['timeline_active_fraction']):.3f} | "
                         f"{float(row['halo_kernel_fraction']):.3f} | {float(row['solve_kernel_fraction']):.3f} |")
        lines += ["", "Trace seconds per step include profiling overhead; use the unprofiled timings",
                  "for performance. Fractions are of summed kernel time, attributed through CUDA",
                  "launch correlation IDs to enclosing `call/fill_halo_regions!` and `call/solve!`",
                  "ranges. Per-phase and per-kernel totals are in phases.csv and kernels.csv."]
    lines += ["", "Finite-field checks establish a functional run, not field-by-field accuracy.",
              "No plots are produced for Nsight runs.", "", "## Exact Nsight result paths", ""]
    for case in CASES:
        for rank in range(RANKS):
            lines.append(f"- `{runtime}/raw/{case}/rank_{rank}/profile.nsys-rep`")
    lines += ["", "SQLite exports sit beside each report. captures.json records report sizes,",
              "SHA-256 hashes, kernel and range counts. Per-rank `nsys stats` summaries are in logs/.",
              "", "## Automatic execution", "",
              f"run_campaign.py runs from the frozen checkout `{plan['frozen_checkout']}`",
              f"(commit `{plan.get('frozen_commit', 'pending')}`) with the Julia environment",
              f"`{runtime}/environment`. Every {plan['poll_seconds']} s it looks for two GPUs with no",
              f"compute processes and at least {plan['minimum_free_mib']} MiB free, and sets",
              "CUDA_VISIBLE_DEVICES to that pair for one launch.",
              f"Live state: `{runtime}/status.json`. Controller log: `{runtime}/logs/controller.log`.",
              "On completion it commits this directory to benchmark-results and pushes."]
    if error:
        lines += ["", "## Failure", "", error]
    (results / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish-results", action="store_true")
    args = parser.parse_args()
    results = Path(__file__).resolve().parent
    root = results.parents[3]
    plan = json.loads((results / "plan.json").read_text())
    runtime = Path(plan["runtime"])
    checkout = Path(plan["frozen_checkout"])
    runs, captures, child = [], [], None

    def status(state, **details):
        write_json(runtime / "status.json", dict(status=state, pid=os.getpid(),
                   updated_utc=datetime.now(timezone.utc).isoformat(), results=str(results),
                   completed_launches=sum(run["state"] == "COMPLETE" for run in runs), **details))

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
            verify_frozen(checkout, plan["frozen_commit"])
            try:
                pair, inventory, busy = free_gpu_pair(plan["minimum_free_mib"])
                if pair:
                    return pair
                status("WAITING_FOR_TWO_FREE_GPUS", inventory=inventory, busy_gpu_uuids=busy)
            except (subprocess.CalledProcessError, ValueError) as error:
                status("WAITING_FOR_VALID_GPU_STATUS", error=str(error))
            time.sleep(plan["poll_seconds"])

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (runtime / "watcher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (runtime / "watcher.pid").write_text(str(os.getpid()) + "\n")
        failure = None
        state = "COMPLETE"
        try:
            if (results / "runs.json").exists():
                raise RuntimeError("This is a one-shot controller; choose fresh paths for another run")
            for case in CASES:
                for profiling in (False, True):
                    pair = wait_for_gpus()
                    kind = "profiled" if profiling else "benchmark"
                    run = dict(kind=kind, case=case, state="RUNNING", source_commit=plan["frozen_commit"],
                               cuda_visible_devices=",".join(pair),
                               started_utc=datetime.now(timezone.utc).isoformat())
                    runs.append(run)
                    write_json(results / "runs.json", runs)
                    command = command_for(checkout, results, runtime, plan, case, profiling)
                    run["command"] = command
                    prefix = results / "logs" / f"{kind}_{case}"
                    if prefix.with_suffix(".csv").exists():
                        raise RuntimeError(f"Output already exists: {prefix}")
                    prefix.with_suffix(".command.sh").write_text(
                        f"CUDA_VISIBLE_DEVICES={run['cuda_visible_devices']} " + shlex.join(command) + "\n")
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=run["cuda_visible_devices"])
                    with prefix.with_suffix(".log").open("w") as output:
                        child = subprocess.Popen(command, cwd=checkout, stdout=output, env=environment,
                                                 stderr=subprocess.STDOUT, start_new_session=True)
                        status("RUNNING", launch=f"{kind}_{case}", controller_child_pid=child.pid,
                               cuda_visible_devices=run["cuda_visible_devices"])
                        run["exit_code"] = child.wait()
                    run["finished_utc"] = datetime.now(timezone.utc).isoformat()
                    if run["exit_code"] != 0:
                        raise RuntimeError(f"{kind} {case} failed with exit {run['exit_code']}")
                    run.update(validate_run(prefix.with_suffix(".log").read_text(),
                                            prefix.with_suffix(".csv"), case, plan))
                    if profiling:
                        status("EXPORTING", launch=f"{kind}_{case}")
                        captures.append(record_capture(results, runtime, case, run, plan))
                        write_json(results / "captures.json", captures)
                    run["state"] = "COMPLETE"
                    write_json(results / "runs.json", runs)
                    describe_outcome(results, runtime, plan, runs, "RUNNING")
            status("ANALYZING")
            analyzer = checkout / "benchmarking/distributed/analyze_nvtx.py"
            with (runtime / "logs" / "analysis.log").open("w") as output:
                analysis = subprocess.run([sys.executable, str(analyzer), str(results / "captures.json")],
                                          stdout=output, stderr=subprocess.STDOUT, timeout=7200)
            if analysis.returncode:
                state = "COMPLETE_ANALYSIS_FAILED"
                failure = f"Runs and captures completed; analyze_nvtx.py failed (see {runtime}/logs/analysis.log)"
            verify_frozen(checkout, plan["frozen_commit"])
        except Exception as error:
            failure = str(error)
            state = "FAILED"
            if runs and runs[-1]["state"] != "COMPLETE":
                runs[-1].update(state="FAILED", error=failure)
            print(f"FAILED: {error}", flush=True)
        write_json(results / "runs.json", runs)
        write_json(results / "captures.json", captures)
        describe_outcome(results, runtime, plan, runs, state, failure)
        publication = {}
        if args.publish_results:
            try:
                remote = subprocess.check_output(["git", "remote", "get-url", "--push", "origin"],
                                                 cwd=root, text=True).strip()
                if remote != plan["origin_push_url"]:
                    raise RuntimeError("Push destination changed")
                branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
                if branch != plan["publish_branch"]:
                    raise RuntimeError(f"Checkout is on {branch}, not {plan['publish_branch']}")
                if subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=root, text=True).strip():
                    raise RuntimeError("Unrelated staged files prevent automatic result publication")
                subprocess.run(["git", "pull", "--ff-only"], cwd=root, check=True)
                for path in results.rglob("*"):
                    if path.is_file() and path.suffix != ".pending":
                        text = path.read_text()
                        path.write_text("\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n")
                relative = str(results.relative_to(root))
                subprocess.run(["git", "add", "--", relative], cwd=root, check=True)
                subprocess.run(["git", "diff", "--cached", "--check"], cwd=root, check=True)
                subprocess.run(["git", "commit", "-m", f"Record two-GPU NCCL Nsight campaign: {state}\n\n"
                                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"], cwd=root, check=True)
                publication["results_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                                        cwd=root, text=True).strip()
                push = subprocess.run(["git", "push"], cwd=root)
                publication["push_exit_code"] = push.returncode
                if push.returncode:
                    state = "RESULTS_READY_PUSH_FAILED"
            except Exception as error:
                publication["publication_error"] = str(error)
                state = "RESULTS_READY_PUBLICATION_FAILED"
        status(state, error=failure, **publication)
        return int(state != "COMPLETE")


if __name__ == "__main__":
    sys.exit(main())
