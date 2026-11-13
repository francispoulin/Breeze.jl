"""Wait for the existing campaign, run anelastic weak scaling, plot and publish its results."""

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


def available_gpus(inventory, processes):
    try:
        inventory_rows = list(csv.reader(inventory.splitlines()))
        process_rows = list(csv.reader(processes.splitlines()))
        if any(len(row) != 2 or not row[1].strip().isdigit() for row in process_rows if row):
            return []
        occupied = {row[0].strip().lower() for row in process_rows if row}
        devices = [(int(row[0]), row[1].strip().lower(), int(row[2])) for row in inventory_rows]
        if ({index for index, _, _ in devices} != {0, 1, 2, 3} or len(devices) != 4
                or len({uuid for _, uuid, _ in devices}) != 4):
            return []
        if not occupied.issubset({uuid for _, uuid, _ in devices}):
            return []
        return sorted(index for index, uuid, memory in devices if uuid not in occupied and memory >= 30000)
    except (ValueError, IndexError):
        return []


def predecessor_finished(record):
    return record["status"] in {"COMPLETE", "FAILED", "ERROR", "CANCELLED",
                                "RESULTS_READY_PUSH_FAILED", "RESULTS_READY_PUBLICATION_FAILED"}


def command_for(root, results, runtime, plan, ranks):
    return ["timeout", "--foreground", "--kill-after=30s", "7200s", "mpiexec", "--bind-to", "none",
            "-n", str(ranks), plan["julia"], "--startup-file=no", "--threads=1",
            "--project=" + str(runtime / "environment"),
            str(root / "benchmarking/distributed/distributed_cbl_benchmark.jl"),
            "--dynamics=anelastic", "--comm=MPI", f"--size={512*ranks}x512x512",
            f"--partition={ranks}x1", "--float_type=Float32", "--time_step=0.005",
            "--warmup_steps=10", "--time_steps=500", "--simplified=false",
            "--random_perturbation=false", "--profile=false", "--nvtx=false",
            "--results_file=" + str(results / "logs" / f"anelastic_{ranks}g.csv")]


def validate_run(text, path, ranks):
    placement = re.findall(r"PLACEMENT rank=(\d+) host=(\S+) gpu_uuid=(\S+) communicator=(\S+)", text)
    validation = re.findall(r"VALIDATION rank=(\d+) fields_finite=(\S+)", text)
    communicator = "none" if ranks == 1 else "Comm"
    if (len(placement) != ranks or {int(p[0]) for p in placement} != set(range(ranks))
            or len({p[2] for p in placement}) != ranks or any(p[3] != communicator for p in placement)
            or len(validation) != ranks or {int(v[0]) for v in validation} != set(range(ranks))
            or any(v[1] != "true" for v in validation)):
        raise RuntimeError("Incomplete or invalid rank/device/communicator/finite-field validation")
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    expected = dict(dynamics="anelastic", float="Float32", Nx=str(512*ranks), Ny="512", Nz="512",
                    Rx=str(ranks), Ry="1", gpus=str(ranks), steps="500", fields_finite="true",
                    communication="SERIAL" if ranks == 1 else "MPI")
    if len(rows) != 1 or any(rows[0].get(key) != value for key, value in expected.items()):
        raise RuntimeError("The measurement does not match the requested anelastic weak-scaling case")
    if any(not math.isfinite(float(rows[0][key])) for key in ("seconds_per_step", "max_abs_w")):
        raise RuntimeError("Non-finite timing or velocity")
    if float(rows[0]["seconds_per_step"]) <= 0:
        raise RuntimeError("Nonpositive timing")
    return dict(measurement=rows[0], placements=placement, rank_validation=validation,
                launcher_messages=[line for line in text.splitlines() if "PMIX ERROR:" in line])


def write_report(results, runtime, cases, state, error=None):
    lines = ["# Anelastic weak scaling: 512³ cells per A100", "", f"Status: **{state}**.", "",
             "Three unprofiled dry-CBL cases use the tracked distributed benchmark driver on",
             "benchmark-results: Float32, WENO5, deterministic initial perturbations, Δt=0.005 s,",
             "ten warmup steps and 500 timed steps. MPI handles distributed communication; the",
             "one-GPU case uses the serial GPU architecture. Each case has one thread per rank",
             "and the CUDA memory pool disabled. No NVTX instrumentation or Nsight capture is enabled.", "",
             "| GPUs | Global grid | Partition | Seconds/step | Weak efficiency | State |",
             "|------|-------------|-----------|--------------|-----------------|-------|"]
    baseline = next((float(c["measurement"]["seconds_per_step"]) for c in cases
                     if c["ranks"] == 1 and c["state"] == "COMPLETE"), None)
    for ranks in (1, 2, 4):
        case = next((c for c in cases if c["ranks"] == ranks), {})
        timing = case.get("measurement", {}).get("seconds_per_step", "pending")
        efficiency = f"{100*baseline/float(timing):.2f}%" if baseline and timing != "pending" else "pending"
        lines.append(f"| {ranks} | {512*ranks}×512×512 | {ranks}×1 | {timing} | {efficiency} | {case.get('state', 'PENDING')} |")
    lines += ["", "Efficiency is **100 × T₁ / Tₙ**, with fixed 512³ cells per GPU. It is not divided",
              "by GPU count. Each local grid remains cubic; the global grid grows in x. N×1 is",
              "the initial anelastic layout suggested in .agents/nvidia_optimization_advice.md.",
              "Seconds per step use the driver's slowest-rank wall time divided by 500. Each case",
              "has one measured window; no uncertainty estimates are implied by these curves.", "",
              "## Queue and provenance", "",
              "This watcher waits for the already queued split-explicit NVTX/Nsight campaign to",
              "reach a terminal state, then runs the three anelastic cases sequentially. It checks",
              "every 30 seconds for enough GPUs with no compute processes and at least 30000 MiB",
              "free each. Idle compute contexts also block selection; no existing jobs are stopped.",
              "A failure in the preceding profiler campaign does not cancel this independent study.", "",
              f"Live state: `{runtime}/status.json`.", f"Controller log: `{runtime}/logs/controller.log`.",
              f"Frozen checkout and environment: `{runtime}`.", "",
              "Source and environment hashes are checked before execution and after the last case.",
              "Per-case logs, commands and GPU inventories are retained. Placement, communicator,",
              "CSV geometry and finite fields on every rank are required. These are finite-field",
              "checks; they do not establish serial-solution agreement or long-term stability.", "",
              "## Plots", "",
              "When all three cases succeed, plot_efficiency.py creates anelastic_weak_efficiency",
              "and breeze_weak_efficiency in PNG and SVG. The Breeze figure has exactly three",
              "model curves: anelastic, compressible explicit and compressible split-explicit.",
              "Its MPI compressible data are archived 9–10 November measurements copied into",
              "provenance/compressible_measurements.csv; the four-GPU compressible layout is 2×2",
              "on 1024×1024×512, keeping the same 512³ local grid. Anelastic uses 4×1 on",
              "2048×512×512. All models use Δt=0.005 s. Source revisions differ between archived and new",
              "studies; this is a comparison of each model's measured scaling, not a same-revision",
              "comparison of absolute model costs. All points use unprofiled seconds per step.",
              "Matching 512³ Oceananigans measurements are still needed for the separate Oceananigans figure.", "",
              "After completion the watcher fast-forwards from origin/benchmark-results, commits",
              "only this report and its figures, then immediately attempts a normal push. It never",
              "force-pushes. Incoming conflicting edits or a rejected push are recorded in status.json.",
              "All task files, logs, temporary files, caches and environments remain inside /u/ajtse."]
    if state == "COMPLETE":
        lines += ["", "![Anelastic weak-scaling efficiency](anelastic_weak_efficiency.png)", "",
                  "![Breeze weak-scaling efficiency](breeze_weak_efficiency.png)"]
    if error:
        lines += ["", "## Failure", "", error]
    (results / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--publish-results", action="store_true")
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    results = Path(__file__).resolve().parent
    root = results.parents[3]
    plan = json.loads((results / "plan.json").read_text())
    hashes = json.loads((results / "provenance/source_sha256.json").read_text())
    cases = []
    child = None

    def status(state, **details):
        write_json(runtime / "status.json", dict(status=state, pid=os.getpid(),
                   updated_utc=datetime.now(timezone.utc).isoformat(), results=str(results),
                   completed_cases=sum(c["state"] == "COMPLETE" for c in cases), **details))

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

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (runtime / "watcher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (runtime / "watcher.pid").write_text(str(os.getpid()) + "\n")
        failure = None
        try:
            if (results / "results.json").exists():
                raise RuntimeError("Choose a fresh study path; results.json already exists")
            while True:
                verify_sources(root, hashes)
                predecessor = json.loads(Path(plan["predecessor_status"]).read_text())
                if predecessor_finished(predecessor):
                    break
                status("WAITING_FOR_PRECEDING_NVTX_CAMPAIGN", predecessor_status=predecessor["status"])
                time.sleep(30)
            for ranks in (1, 2, 4):
                while True:
                    verify_sources(root, hashes)
                    try:
                        inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free",
                                                             "--format=csv,noheader,nounits"], text=True)
                        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                                             "--format=csv,noheader,nounits"], text=True)
                        available = available_gpus(inventory, processes)
                        if len(available) >= ranks:
                            devices = available[:ranks]
                            break
                        status("WAITING_FOR_FREE_GPUS", requested_gpus=ranks, free_gpu_indices=available,
                               inventory=inventory.splitlines(), occupied_processes=processes.splitlines())
                    except subprocess.CalledProcessError as error:
                        status("WAITING_FOR_VALID_GPU_STATUS", error=str(error))
                    time.sleep(30)
                case = dict(ranks=ranks, state="RUNNING", gpu_indices=devices,
                            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
                cases.append(case)
                write_json(results / "results.json", dict(cases=cases))
                command = command_for(root, results, runtime, plan, ranks)
                case["command"] = command
                prefix = results / "logs" / f"anelastic_{ranks}g"
                if prefix.with_suffix(".csv").exists():
                    raise RuntimeError(f"Output already exists: {prefix}")
                prefix.with_suffix(".gpu_before.csv").write_text(inventory)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
                prefix.with_suffix(".command.sh").write_text("CUDA_VISIBLE_DEVICES=" + env["CUDA_VISIBLE_DEVICES"]
                                                            + " " + shlex.join(command) + "\n")
                with prefix.with_suffix(".log").open("w") as output:
                    child = subprocess.Popen(command, cwd=root, env=env, stdout=output,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    status("RUNNING", current_case=f"anelastic_{ranks}g", child_pid=child.pid, gpu_indices=devices)
                    case["exit_code"] = child.wait()
                if case["exit_code"]:
                    raise RuntimeError(f"{ranks}-GPU anelastic case exited with {case['exit_code']}")
                case.update(validate_run(prefix.with_suffix(".log").read_text(), prefix.with_suffix(".csv"), ranks))
                selected_uuids = {row[1].strip().lower().removeprefix("gpu-")
                                  for row in csv.reader(inventory.splitlines()) if int(row[0]) in devices}
                actual_uuids = {row[2].lower().removeprefix("gpu-") for row in case["placements"]}
                if actual_uuids != selected_uuids:
                    raise RuntimeError("Rank GPU UUIDs do not match the selected free devices")
                case["state"] = "COMPLETE"
                write_json(results / "results.json", dict(cases=cases))
                write_report(results, runtime, cases, "RUNNING")
            verify_sources(root, hashes)
            status("RENDERING_PLOTS")
            from plot_efficiency import render
            render(results, [c["measurement"] for c in cases])
        except Exception as error:
            failure = str(error)
            if cases and cases[-1]["state"] != "COMPLETE":
                cases[-1].update(state="FAILED", error=failure)
            print(f"FAILED: {error}", flush=True)
        outcome = "FAILED" if failure else "COMPLETE"
        write_json(results / "results.json", dict(status=outcome, error=failure, cases=cases))
        write_report(results, runtime, cases, outcome, failure)
        publication = {}
        if args.publish_results:
            try:
                remote = subprocess.check_output(["git", "remote", "get-url", "--push", "origin"], cwd=root, text=True).strip()
                if remote != plan["origin_push_url"]:
                    raise RuntimeError("Push destination changed")
                if subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=root, text=True).strip():
                    raise RuntimeError("Unrelated staged files prevent automatic publication")
                subprocess.run(["git", "pull", "--ff-only", "origin", "benchmark-results"], cwd=root, check=True)
                for path in results.rglob("*"):
                    if path.is_file() and path.suffix != ".png":
                        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()).rstrip() + "\n")
                subprocess.run(["git", "add", "--", str(results.relative_to(root))], cwd=root, check=True)
                figures = [str(p.relative_to(root)) for pattern in ("*.png", "*.svg") for p in results.glob(pattern)]
                if figures:
                    subprocess.run(["git", "add", "-f", "--", *figures], cwd=root, check=True)
                subprocess.run(["git", "diff", "--cached", "--check"], cwd=root, check=True)
                subprocess.run(["git", "commit", "-m", f"Record anelastic 512-cubed weak scaling: {outcome}"], cwd=root, check=True)
                publication["results_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
                push = subprocess.run(["git", "push", "origin", "HEAD:refs/heads/benchmark-results"], cwd=root)
                publication["push_exit_code"] = push.returncode
                if push.returncode:
                    outcome = "RESULTS_READY_PUSH_FAILED"
            except Exception as error:
                outcome = "RESULTS_READY_PUBLICATION_FAILED"
                publication["publication_error"] = str(error)
        status(outcome, error=failure, **publication)
        return int(bool(failure or publication.get("push_exit_code") or publication.get("publication_error")))


if __name__ == "__main__":
    sys.exit(main())
