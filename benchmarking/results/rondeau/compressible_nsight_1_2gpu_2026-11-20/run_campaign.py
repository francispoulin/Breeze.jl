"""NVTX/Nsight captures of the cases in plan.json on one or two GPUs (NCCL), launched from a
frozen checkout once enough GPUs are free, followed by per-time-step kernel tables and
1-vs-2-GPU kernel comparisons."""

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
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_environment(results):
    """Apply environment.sh to this controller, so launches, exports and analyses do not depend on
    the shell that started it. CUDA_VISIBLE_DEVICES is set per launch."""
    output = subprocess.check_output(["bash", "-c", 'CUDA_VISIBLE_DEVICES=none; source "$1"; env -0',
                                      "bash", str(results / "environment.sh")])
    loaded = dict(entry.split("=", 1) for entry in output.decode().split("\0") if "=" in entry)
    loaded.pop("CUDA_VISIBLE_DEVICES")
    os.environ.clear()
    os.environ.update(loaded)
    if os.environ.get("JULIA_CUDA_MEMORY_POOL") != "none":
        raise RuntimeError("environment.sh did not set JULIA_CUDA_MEMORY_POOL=none")


def case_name(case):
    return case["name"]


def verify_frozen(checkout, commit):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=checkout, text=True).strip()
    if head != commit or dirty:
        raise RuntimeError(f"Frozen checkout changed: HEAD={head} dirty={bool(dirty)}")


def free_gpus(count, minimum_free_mib):
    """Return the `count` lowest-index GPUs with no compute processes and enough free memory."""
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
    return (free[:count] if len(free) >= count else None), inventory.splitlines(), sorted(busy)


def gpu_uuids(indices):
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid",
                                         "--format=csv,noheader"], text=True)
    uuids = dict((field.strip() for field in row) for row in csv.reader(inventory.splitlines()))
    return {uuids[index] for index in indices}


def watch_foreign_processes(uuids, done, foreign, interval=5):
    """Record compute processes of other users on the launch's GPUs until `done` is set."""
    while not done.is_set():
        try:
            listing = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                               "--format=csv,noheader,nounits"], text=True)
            for row in csv.reader(listing.splitlines()):
                uuid, pid, memory = (field.strip() for field in row)
                if uuid in uuids:
                    try:
                        owner = os.stat(f"/proc/{pid}").st_uid
                    except OSError:
                        owner = None
                    if owner != os.getuid():
                        entry = foreign.setdefault(f"{uuid}:{pid}", dict(gpu_uuid=uuid, pid=int(pid), uid=owner,
                                                                         max_used_mib=0, samples=0))
                        entry["max_used_mib"] = max(entry["max_used_mib"], int(memory or 0))
                        entry["samples"] += 1
        except (subprocess.CalledProcessError, ValueError):
            pass
        done.wait(interval)


def supersede(results, runtime, name, runs, captures):
    """Move an earlier launch's reports, logs and records aside before it is captured again."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = f"{name}_{stamp}"
    old_run = next(run for run in runs if run["case"] == name)
    raw = runtime / "raw" / name
    if raw.exists():
        (runtime / "raw" / "superseded").mkdir(exist_ok=True)
        raw.rename(runtime / "raw" / "superseded" / archive)
    logs = results / "logs" / "superseded" / archive
    logs.mkdir(parents=True)
    for path in (results / "logs").glob("*"):
        if path.is_file() and re.fullmatch(rf"(profiled|export|summary)_{re.escape(name)}(_\d+)?\.\w+(\.\w+)?", path.name):
            shutil.move(str(path), logs / path.name)
    superseded_path = results / "superseded_runs.json"
    superseded = json.loads(superseded_path.read_text()) if superseded_path.exists() else []
    superseded.append(dict(run=old_run, capture=next((c for c in captures if c["case"] == name), None),
                           superseded_utc=datetime.now(timezone.utc).isoformat(),
                           reports=str(runtime / "raw" / "superseded" / archive),
                           logs=str(logs.relative_to(results))))
    write_json(superseded_path, superseded)
    runs[:] = [run for run in runs if run["case"] != name]
    captures[:] = [capture for capture in captures if capture["case"] != name]


def command_for(checkout, results, runtime, plan, case):
    name = case_name(case)
    driver = checkout / "benchmarking/distributed/distributed_cbl_benchmark.jl"
    command = ["timeout", "--foreground", "--kill-after=30s", "2400s",
               "mpiexec", "--bind-to", "none", "-n", str(case["ranks"]),
               "bash", str(results / "profile_rank.sh"), str(runtime / "environment"),
               plan["julia"], str(driver), str(runtime / "raw" / name)]
    command += [f"--dynamics={case['dynamics']}", f"--size={case['size']}",
                f"--partition={case['partition']}", *plan["common_arguments"], *case["arguments"],
                f"--warmup_steps={plan['warmup_steps']}", f"--time_steps={plan['measured_steps']}",
                "--nvtx=true", "--profile=true",
                "--results_file=" + str(results / "logs" / f"profiled_{name}.csv")]
    return command


def validate_run(log, csv_path, case, plan):
    ranks = case["ranks"]
    placement = re.findall(r"PLACEMENT rank=(\d+) host=(\S+) gpu_uuid=(\S+) communicator=(\S+)", log)
    pools = re.findall(r"PLACEMENT rank=\d+ .* pool=(\S+)", log)
    if len(pools) != case["ranks"] or any(pool != "none" for pool in pools):
        raise RuntimeError(f"Launch did not run with JULIA_CUDA_MEMORY_POOL=none: {pools}")
    validation = re.findall(r"VALIDATION rank=(\d+) fields_finite=(\S+) gpu_used_bytes=(\d+)", log)
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    communicator = "NCCLCommunicator" if ranks > 1 else "none"
    if (len(placement) != ranks or {int(row[0]) for row in placement} != set(range(ranks))
            or len({row[2] for row in placement}) != ranks
            or any(row[3] != communicator for row in placement)
            or len(validation) != ranks or any(row[1] != "true" for row in validation)
            or len(rows) != 1):
        raise RuntimeError("Missing or invalid placement, communicator, or finite-field checks")
    Nx, Ny, Nz = case["size"].split("x")
    Rx, Ry = case["partition"].split("x")
    float_type = next(a for a in plan["common_arguments"] + case["arguments"] if a.startswith("--float_type="))
    expected = dict(dynamics=case["dynamics"], float=float_type.split("=")[1], Nx=Nx, Ny=Ny, Nz=Nz, Rx=Rx, Ry=Ry,
                    gpus=str(ranks), steps=str(plan["measured_steps"]),
                    communication="NCCL" if ranks > 1 else "SERIAL", fields_finite="true")
    if any(rows[0].get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Unexpected measurement: {rows[0]}")
    if not math.isfinite(float(rows[0]["max_abs_w"])):
        raise RuntimeError(f"Invalid numeric measurement: {rows[0]}")
    return dict(placements=placement, rank_validation=validation, measurement=rows[0])


def record_capture(results, runtime, case, run, plan):
    name = case_name(case)
    record = dict(case=name, state="RUNNING", steps=plan["measured_steps"],
                  source_commit=run["source_commit"], profiles=[])
    devices = {int(row[0]): row[2].lower().removeprefix("gpu-") for row in run["placements"]}
    for rank in range(case["ranks"]):
        report = runtime / "raw" / name / f"rank_{rank}" / "profile.nsys-rep"
        database = report.with_suffix(".sqlite")
        if not report.is_file() or report.stat().st_size == 0:
            raise RuntimeError(f"Missing report: {report}")
        with (results / "logs" / f"export_{name}_{rank}.log").open("w") as output:
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
        with (results / "logs" / f"summary_{name}_{rank}.txt").open("w") as output:
            subprocess.run(["nsys", "stats", "--report=nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum",
                            str(database)], check=True, stdout=output, stderr=subprocess.STDOUT, timeout=900)
    record["state"] = "COMPLETE"
    return record


def describe_outcome(results, runtime, plan, runs, state, error=None):
    lines = [f"# {plan['title']}", "", f"Status: **{state}**.", "", *plan["description"], "",
             "Each launch takes ten warmup steps, then Nsight records CUDA, NVTX and MPI for",
             "ten steps (cudaProfilerApi start/stop).", "",
             "**The per-kernel time per step tables are in [kernels_per_step.md](kernels_per_step.md)**",
             "(CSV: kernels_per_step.csv). Each entry sums all launches of a kernel during the",
             "ten captured steps and divides by ten.", "",
             "**1-GPU vs 2-GPU per-kernel comparison** (difference in ms and %, weak-scaling",
             "efficiency = 100 × T₁/T₂ with the slower rank): [kernel_scaling_1_vs_2gpu.md](kernel_scaling_1_vs_2gpu.md)",
             "(CSV: kernel_scaling_1_vs_2gpu.csv).", "",
             "## Launches", "",
             "| Case | State | Maximum absolute w | GPUs | Other users' processes on these GPUs |",
             "|------|-------|--------------------|------|---------------------------------------|"]
    order = [case_name(case) for case in plan["cases"]]
    for run in sorted(runs, key=lambda run: order.index(run["case"])):
        foreign = run.get("foreign_processes")
        foreign = "not monitored" if foreign is None else (", ".join(
            f"pid {p['pid']} ({p['max_used_mib']} MiB)" for p in foreign) or "none")
        lines.append(f"| {run['case']} | {run['state']} | "
                     f"{run.get('measurement', {}).get('max_abs_w', 'pending')} | {run.get('cuda_visible_devices', '')} | {foreign} |")
    summary = results / "rank_summary.csv"
    if summary.exists():
        with summary.open() as stream:
            rows = list(csv.DictReader(stream))
        lines += ["", "## Trace summary per rank", "",
                  "| Case | Rank | Trace s/step | Kernel ms/step | Launches/step | GPU active fraction | Halo kernel fraction | Solve kernel fraction |",
                  "|------|------|--------------|----------------|---------------|---------------------|----------------------|-----------------------|"]
        for row in rows:
            steps = int(row["steps"])
            lines.append(f"| {row['case']} | {row['rank']} | {float(row['trace_seconds_per_step']):.4f} | "
                         f"{1e3 * float(row['summed_kernel_seconds']) / steps:.1f} | "
                         f"{int(row['kernel_launches']) / steps:g} | {float(row['timeline_active_fraction']):.3f} | "
                         f"{float(row['halo_kernel_fraction']):.3f} | {float(row['solve_kernel_fraction']):.3f} |")
        lines += ["", "Trace seconds per step include profiling overhead. Fractions are of summed",
                  "kernel time, attributed through CUDA launch correlation IDs to enclosing",
                  "`call/fill_halo_regions!` and `call/solve!` ranges. Per-phase totals are in phases.csv."]
    lines += ["", "Finite-field checks establish a functional run, not field-by-field accuracy.",
              "No plots are produced for Nsight runs.", "", "## Exact Nsight result paths", ""]
    for case in plan["cases"]:
        for rank in range(case["ranks"]):
            lines.append(f"- `{runtime}/raw/{case_name(case)}/rank_{rank}/profile.nsys-rep`")
    lines += ["", "SQLite exports sit beside each report. captures.json records report sizes,",
              "SHA-256 hashes, kernel and range counts. Per-rank `nsys stats` summaries are in logs/.",
              "", "## Automatic execution", "",
              f"run_campaign.py runs from the frozen checkout `{plan['frozen_checkout']}`",
              f"(commit `{plan['frozen_commit']}`) with the Julia environment `{runtime}/environment`.",
              f"Every {plan['poll_seconds']} s it looks for GPUs with no compute processes and at least",
              f"{plan['minimum_free_mib']} MiB free, and sets CUDA_VISIBLE_DEVICES for one launch.",
              f"Live state: `{runtime}/status.json`. Controller log: `{runtime}/logs/controller.log`.",
              "On completion it commits this directory to benchmark-results and pushes."]
    superseded_path = results / "superseded_runs.json"
    if superseded_path.exists():
        lines += ["", "## Superseded captures", "",
                  "These launches were captured again; the tables above use only the newer capture.", ""]
        for entry in json.loads(superseded_path.read_text()):
            lines.append(f"- `{entry['run']['case']}` from {entry['run']['started_utc'][:19]} UTC "
                         f"(superseded {entry['superseded_utc'][:19]} UTC): reports in `{entry['reports']}`, "
                         f"logs in `{entry['logs']}`. Reason: {entry.get('reason', 'not recorded')}")
    if error:
        lines += ["", "## Failure", "", error]
    (results / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish-results", action="store_true")
    parser.add_argument("--rerun", action="append", default=[], metavar="CASE",
                        help="capture CASE again, superseding its earlier capture, then redo the analysis")
    parser.add_argument("--reason", default="", help="why the --rerun cases are captured again")
    args = parser.parse_args()
    results = Path(__file__).resolve().parent
    root = results.parents[3]
    plan = json.loads((results / "plan.json").read_text())
    runtime = Path(plan["runtime"])
    load_environment(results)
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

    def wait_for_gpus(count):
        while True:
            verify_frozen(checkout, plan["frozen_commit"])
            try:
                gpus, inventory, busy = free_gpus(count, plan["minimum_free_mib"])
                if gpus:
                    return gpus
                status(f"WAITING_FOR_{count}_FREE_GPUS", inventory=inventory, busy_gpu_uuids=busy)
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
            selected = plan["cases"]
            if args.rerun:
                names = {case_name(case) for case in plan["cases"]}
                if not set(args.rerun) <= names:
                    raise RuntimeError(f"Unknown cases: {sorted(set(args.rerun) - names)}")
                runs[:] = json.loads((results / "runs.json").read_text())
                captures[:] = json.loads((results / "captures.json").read_text())
                for name in args.rerun:
                    supersede(results, runtime, name, runs, captures)
                    superseded_path = results / "superseded_runs.json"
                    superseded = json.loads(superseded_path.read_text())
                    superseded[-1]["reason"] = args.reason or "not recorded"
                    write_json(superseded_path, superseded)
                selected = [case for case in plan["cases"] if case_name(case) in args.rerun]
            elif (results / "runs.json").exists():
                raise RuntimeError("Captures exist; pass --rerun CASE to capture a case again")
            for case in selected:
                name = case_name(case)
                gpus = wait_for_gpus(case["ranks"])
                run = dict(case=name, state="RUNNING", source_commit=plan["frozen_commit"],
                           cuda_visible_devices=",".join(gpus),
                           started_utc=datetime.now(timezone.utc).isoformat())
                runs.append(run)
                write_json(results / "runs.json", runs)
                command = command_for(checkout, results, runtime, plan, case)
                run["command"] = command
                prefix = results / "logs" / f"profiled_{name}"
                if prefix.with_suffix(".csv").exists():
                    raise RuntimeError(f"Output already exists: {prefix}")
                prefix.with_suffix(".command.sh").write_text(
                    f"CUDA_VISIBLE_DEVICES={run['cuda_visible_devices']} " + shlex.join(command) + "\n")
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=run["cuda_visible_devices"])
                with prefix.with_suffix(".log").open("w") as output:
                    child = subprocess.Popen(command, cwd=checkout, stdout=output, env=environment,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    status("RUNNING", launch=name, controller_child_pid=child.pid,
                           cuda_visible_devices=run["cuda_visible_devices"])
                    done, foreign = threading.Event(), {}
                    watcher = threading.Thread(target=watch_foreign_processes, daemon=True,
                                               args=(gpu_uuids(gpus), done, foreign))
                    watcher.start()
                    run["exit_code"] = child.wait()
                    done.set()
                    watcher.join()
                    run["foreign_processes"] = list(foreign.values())
                run["finished_utc"] = datetime.now(timezone.utc).isoformat()
                if run["exit_code"] != 0:
                    raise RuntimeError(f"{name} failed with exit {run['exit_code']}")
                run.update(validate_run(prefix.with_suffix(".log").read_text(),
                                        prefix.with_suffix(".csv"), case, plan))
                status("EXPORTING", launch=name)
                captures.append(record_capture(results, runtime, case, run, plan))
                write_json(results / "captures.json", captures)
                run["state"] = "COMPLETE"
                write_json(results / "runs.json", runs)
                describe_outcome(results, runtime, plan, runs, "RUNNING")
            status("ANALYZING")
            tools = checkout / "benchmarking/distributed"
            with (runtime / "logs" / "analysis.log").open("w") as output:
                steps = (("analyze_nvtx.py", [str(results / "captures.json")]),
                         ("kernel_step_table.py", [str(results / "analysis.json")]),
                         ("kernel_scaling_table.py", [str(results / "analysis.json"),
                                                      str(results / "kernel_scaling_1_vs_2gpu"), *plan["comparisons"]]))
                for script, arguments in steps:
                    analysis = subprocess.run([sys.executable, str(tools / script), *arguments],
                                              stdout=output, stderr=subprocess.STDOUT, timeout=7200)
                    if analysis.returncode:
                        state = "COMPLETE_ANALYSIS_FAILED"
                        failure = f"Captures completed; {script} failed (see {runtime}/logs/analysis.log)"
                        break
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
                    if path.is_file() and path.suffix != ".pending" and "__pycache__" not in path.parts:
                        text = path.read_text()
                        path.write_text("\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n")
                relative = str(results.relative_to(root))
                subprocess.run(["git", "add", "--", relative], cwd=root, check=True)
                subprocess.run(["git", "diff", "--cached", "--check"], cwd=root, check=True)
                subprocess.run(["git", "commit", "-m", f"{plan['commit_subject']}: {state}\n\n"
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
