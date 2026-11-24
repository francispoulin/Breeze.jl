"""Prepare, submit, run, and finalize the Rorqual NVTX/Nsight study: anelastic and compressible
explicit (CE) dynamics on 1, 4, and 8 H100 GPUs with NCCL halo exchange.

    prepare STUDY     freeze the source and Julia environment, write the plan and job scripts
    submit STUDY      submit one Slurm allocation per GPU count
    run-size STUDY N  (inside an allocation) capture every case on N GPUs, export and check reports
    finalize STUDY    (login node, after the allocations end) analyze, write README, commit, push
    retry STUDY CASE  move a failed case aside and capture it again in a new allocation
    report STUDY      rewrite README.md from the current state
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sqlite3
import subprocess
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fir.study import check_sources, read_json, run_text, validate_measurement
from rorqual.run_case import timestamp, write_json

REPO = Path(__file__).resolve().parents[3]
DYNAMICS = ("anelastic", "compressible_explicit")
# 512³ cells per GPU on the fixed 12 km × 12 km × 3 km domain; Δt halves for every doubling of
# spatial resolution and is the same for both dynamics on the same grid (AGENTS.md).
GRIDS = ((1, "512x512x512", "1x1", 0.01),
         (4, "1024x1024x512", "4x1", 0.005),
         (8, "4096x512x512", "8x1", 0.00125))
GPU_COUNTS = tuple(grid[0] for grid in GRIDS)
WARMUP_STEPS = 10
CAPTURED_STEPS = 10
CASE_SECONDS = 2400
TITLES = dict(anelastic="Anelastic", compressible_explicit="CE")


def build_plan():
    plan = []
    for gpus, size, partition, time_step in GRIDS:
        for dynamics in DYNAMICS:
            backend = "SERIAL" if gpus == 1 else "NCCL"
            plan.append(dict(id=f"{dynamics}_{backend}_{gpus}g_{partition}", dynamics=dynamics, gpus=gpus,
                             size=size, partition=partition, backend="NCCL", time_step=time_step,
                             float_type="Float32", timed_steps=CAPTURED_STEPS))
    return plan


def comparisons(plan):
    """kernel_scaling_table.py arguments: each multi-GPU case against its dynamics' one-GPU case."""
    pairs = []
    for dynamics in DYNAMICS:
        cases = [case for case in plan if case["dynamics"] == dynamics]
        base = next(case for case in cases if case["gpus"] == 1)
        for case in cases:
            if case["gpus"] > 1:
                pairs.append(f"{TITLES[dynamics]} 1 vs {case['gpus']} GPUs={base['id']}:{case['id']}")
    return pairs


def nodes(count):
    return max(1, count // 4)


def gpus_per_node(count):
    return min(4, count)


def prepare(study):
    if study.exists():
        raise RuntimeError("Use a fresh study directory")
    if run_text(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO).strip():
        raise RuntimeError("Commit the source before freezing the study")
    runtime = REPO / ".runtime/studies" / study.name
    scratch = Path(os.environ["BREEZE_NSIGHT_SCRATCH"]) / study.name
    if runtime.exists() or scratch.exists():
        raise RuntimeError("Runtime or scratch directory for this study already exists")
    source = runtime / "source"
    environment = runtime / "environment"
    source.mkdir(parents=True)
    environment.mkdir()
    (scratch / "raw").mkdir(parents=True)
    for name in ("provenance", "cases", "allocations"):
        (study / name).mkdir(parents=True)
    (study / "provenance/source_commit.txt").write_text(run_text(["git", "rev-parse", "HEAD"], cwd=REPO))
    paths = ["src", "ext", "Project.toml", "AGENTS.md", "benchmarking/src", "benchmarking/distributed/fir",
             "benchmarking/distributed/rorqual", "benchmarking/distributed/distributed_cbl_benchmark.jl",
             "benchmarking/distributed/oceananigans_rectilinear_buoyancy.jl",
             "benchmarking/distributed/nvtx_instrumentation.jl", "benchmarking/distributed/analyze_nvtx.py",
             "benchmarking/distributed/kernel_step_table.py", "benchmarking/distributed/kernel_scaling_table.py"]
    if (REPO / "Artifacts.toml").exists():
        paths.append("Artifacts.toml")
    archive = study / "provenance/source.tar.gz"
    subprocess.run(["git", "archive", "--format=tar.gz", "--output=" + str(archive), "HEAD", "--", *paths],
                   cwd=REPO, check=True)
    with tarfile.open(archive) as handle:
        handle.extractall(source, filter="data")
    installed = Path(os.environ["BREEZE_FIR_ENV"])
    for name in ("Project.toml", "Manifest.toml", "LocalPreferences.toml"):
        contents = (installed / name).read_text().replace(f'path = "{REPO}"', f'path = "{source}"')
        (environment / name).write_text(contents)
        (study / "provenance" / name).write_text(contents)
    (study / "provenance/setup.log").write_text((installed / "setup.log").read_text())
    config = dict(repository=str(REPO), source=str(source), environment=str(environment), runtime=str(runtime),
                  runtime_root=str(REPO / ".runtime"), raw=str(scratch / "raw"), machine="rorqual",
                  prepared=timestamp(), warmup_steps=WARMUP_STEPS, captured_steps=CAPTURED_STEPS,
                  float_type="Float32", backend="NCCL", random_perturbation=False, simplified=False,
                  nvtx=True, profile=True, nsight_trace="cuda,nvtx,mpi", case_limit_seconds=CASE_SECONDS,
                  allocation_limit="01:30:00", nsys=run_text(["nsys", "--version"]).strip())
    plan = build_plan()
    write_json(study / "config.json", config)
    write_json(study / "plan.json", plan)
    files = [path for path in source.rglob("*") if path.is_file()]
    files += list(environment.iterdir()) + [study / "plan.json", study / "config.json"]
    write_json(study / "provenance/hashes.json",
               {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    worker = source / "benchmarking/distributed/rorqual/nsight.py"
    prefix = ("#!/usr/bin/env bash\nset -eo pipefail\n"
              + "export BREEZE_RUNTIME_ROOT=" + shlex.quote(config["runtime_root"]) + "\n"
              + "export BREEZE_FIR_ENV=" + shlex.quote(str(environment)) + "\n"
              + "source " + shlex.quote(str(source / "benchmarking/distributed/rorqual/nsight_environment.sh")) + "\n")
    for count in GPU_COUNTS:
        directory = study / "allocations" / f"{count}g"
        directory.mkdir()
        (directory / "allocation.sh").write_text(prefix + shlex.join(["python3", str(worker), "run-size", str(study), str(count)]) + "\n")
    render_report(study)
    print(f"Prepared {len(plan)} cases: {study}")


def submit(study):
    check_sources(study)
    if (study / "submission.json").exists():
        raise RuntimeError("Study already submitted; refusing duplicate jobs")
    config = read_json(study / "config.json")
    submission = dict(submitted=timestamp(), jobs=[], state="submitting")
    write_json(study / "submission.json", submission)
    for count in GPU_COUNTS:
        directory = study / "allocations" / f"{count}g"
        per_node = gpus_per_node(count)
        command = ["sbatch", "--parsable", "--account=def-fpoulin_gpu", f"--job-name=breeze_nsight_{count}g",
                   f"--nodes={nodes(count)}", f"--ntasks={count}", f"--ntasks-per-node={per_node}",
                   f"--gpus-per-node=h100:{per_node}", "--cpus-per-task=8", f"--mem={64 * per_node}G",
                   "--time=" + config["allocation_limit"], "--distribution=block:block",
                   "--chdir=" + config["repository"], "--output=" + str(directory / "slurm-%j.out"),
                   str(directory / "allocation.sh")]
        (directory / "submission_command.sh").write_text(shlex.join(command) + "\n")
        job = run_text(command).strip().split(";")[0]
        submission["jobs"].append(dict(gpus=count, job_id=job, command=command))
        write_json(study / "submission.json", submission)
        print(f"Submitted {count} GPUs: {job}", flush=True)
    submission["state"] = "submitted"
    write_json(study / "submission.json", submission)
    render_report(study)


def case_command(study, case):
    config = read_json(study / "config.json")
    source = Path(config["source"])
    count = case["gpus"]
    output = Path(config["raw"]) / case["id"] / "rank_%q{SLURM_PROCID}" / "profile"
    return ["srun", "--mpi=pmix", "--exact", "--kill-on-bad-exit=1", f"--time={CASE_SECONDS // 60}",
            "--cpu-bind=none", f"--gpus-per-node=h100:{gpus_per_node(count)}", "--gpu-bind=none",
            "--distribution=block:block", f"--nodes={nodes(count)}", f"--ntasks={count}",
            f"--ntasks-per-node={gpus_per_node(count)}",
            "bash", str(source / "benchmarking/distributed/rorqual/rank.sh"), str(study / "cases" / case["id"]),
            "nsys", "profile", "--trace=" + config["nsight_trace"], "--mpi-impl=openmpi", "--sample=none",
            "--cpuctxsw=none", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
            "--force-overwrite=false", "--wait=primary", "--output=" + str(output),
            "julia", "--startup-file=no", "--threads=1", "--project=" + config["environment"],
            str(source / "benchmarking/distributed/distributed_cbl_benchmark.jl"),
            "--dynamics=" + case["dynamics"], "--comm=" + case["backend"], "--size=" + case["size"],
            "--partition=" + case["partition"], "--float_type=" + case["float_type"], "--simplified=false",
            "--random_perturbation=false", "--time_step=" + str(case["time_step"]),
            f"--warmup_steps={WARMUP_STEPS}", f"--time_steps={case['timed_steps']}",
            "--nvtx=true", "--profile=true", "--results_file=" + str(study / "cases" / case["id"] / "measurement.csv")]


def check_capture(study, case, measurement_directory):
    """Export every rank's report to SQLite and require kernels, NVTX ranges and the rank's own GPU."""
    raw = Path(read_json(study / "config.json")["raw"]) / case["id"]
    devices = {}
    for rank in range(case["gpus"]):
        log = (measurement_directory / f"rank_{rank}.log").read_text()
        devices[rank] = re.search(r"PLACEMENT rank=\d+ host=\S+ gpu_uuid=(\S+)", log)[1].lower().removeprefix("gpu-")
    profiles = []
    for rank in range(case["gpus"]):
        report = raw / f"rank_{rank}" / "profile.nsys-rep"
        database = report.with_suffix(".sqlite")
        if not report.is_file() or report.stat().st_size == 0:
            raise RuntimeError(f"Missing report: {report}")
        with (measurement_directory / f"export_rank_{rank}.log").open("w") as output:
            subprocess.run(["nsys", "export", "--type=sqlite", f"--output={database}", str(report)],
                           check=True, stdout=output, stderr=subprocess.STDOUT, timeout=1800)
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            kernels = connection.execute("SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
            ranges = connection.execute("SELECT count(*) FROM NVTX_EVENTS WHERE end IS NOT NULL").fetchone()[0]
            known = {row[0].lower().removeprefix("gpu-") for row in connection.execute("SELECT uuid FROM TARGET_INFO_GPU")}
        if not kernels or not ranges or devices[rank] not in known:
            raise RuntimeError(f"Invalid CUDA/NVTX/device capture: {report}")
        with (measurement_directory / f"summary_rank_{rank}.txt").open("w") as output:
            subprocess.run(["nsys", "stats", "--report=nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum", str(database)],
                           check=True, stdout=output, stderr=subprocess.STDOUT, timeout=1800)
        profiles.append(dict(rank=rank, report=str(report), sqlite=str(database), bytes=report.stat().st_size,
                             sha256=hashlib.sha256(report.read_bytes()).hexdigest(), kernels=kernels,
                             ranges=ranges, gpu_uuid=devices[rank]))
    return dict(case=case["id"], steps=case["timed_steps"], profiles=profiles)


def run_case(study, case):
    directory = study / "cases" / case["id"]
    directory.mkdir(exist_ok=False)
    raw = Path(read_json(study / "config.json")["raw"]) / case["id"]
    for rank in range(case["gpus"]):
        (raw / f"rank_{rank}").mkdir(parents=True, exist_ok=False)
    status = dict(case=case, state="running", started=timestamp(), job_id=os.environ["SLURM_JOB_ID"])
    write_json(directory / "status.json", status)
    try:
        check_sources(study)
        command = case_command(study, case)
        (directory / "command.sh").write_text(shlex.join(command) + "\n")
        with (directory / "launcher.log").open("w") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, timeout=CASE_SECONDS + 120)
        status["measurement"] = validate_measurement(case, directory, result.returncode)
        status["capture"] = check_capture(study, case, directory)
        check_sources(study)
        status["state"] = "validated"
    except Exception as error:
        status.update(state="failed", error=str(error))
        print(f"FAILED {case['id']}: {error}", flush=True)
    finally:
        status["finished"] = timestamp()
        write_json(directory / "status.json", status)
    return status["state"] == "validated"


def run_size(study, count):
    check_sources(study)
    directory = study / "allocations" / f"{count}g"
    allocation = dict(job_id=os.environ["SLURM_JOB_ID"], gpus=count, started=timestamp(), state="running", cases=[])
    write_json(directory / "status.json", allocation)
    try:
        capture = 'module -t list 2>&1; julia --version; mpiexec --version; nsys --version; scontrol show job "$SLURM_JOB_ID"'
        with (directory / "environment.log").open("w") as handle:
            subprocess.run(["bash", "-c", capture], stdout=handle, stderr=subprocess.STDOUT, check=True)
        for case in read_json(study / "plan.json"):
            if case["gpus"] == count:
                print(f"STARTING {case['id']} {timestamp()}", flush=True)
                allocation["cases"].append(dict(id=case["id"], validated=run_case(study, case)))
                write_json(directory / "status.json", allocation)
        allocation["state"] = "validated" if all(item["validated"] for item in allocation["cases"]) else "failed"
    except Exception as error:
        allocation.update(state="failed", error=str(error))
        print(f"Allocation failed: {error}", flush=True)
    finally:
        allocation["finished"] = timestamp()
        write_json(directory / "status.json", allocation)
    return 0 if allocation["state"] == "validated" else 1


def retry(study, case_ids):
    """Move failed cases (all on one GPU count) aside and submit one allocation that captures them again.

    The allocation calls `run_case` from the frozen source, so a retried capture uses the same code and
    environment as the original cases. Rejected attempts stay in rejected_attempts/ and are not analyzed."""
    check_sources(study)
    config = read_json(study / "config.json")
    plan = {case["id"]: case for case in read_json(study / "plan.json")}
    states = case_states(study)
    if not case_ids or any(name not in plan or states[name]["state"] != "failed" for name in case_ids):
        raise RuntimeError("Retry only failed cases of this study")
    counts = {plan[name]["gpus"] for name in case_ids}
    if len(counts) != 1:
        raise RuntimeError("Retry cases of one GPU count at a time")
    count = counts.pop()
    submission = read_json(study / "submission.json")
    attempt = len(submission.setdefault("retry_jobs", [])) + 1
    for name in case_ids:
        label = f"{name}_job{states[name]['job_id']}"
        (study / "rejected_attempts").mkdir(exist_ok=True)
        (study / "cases" / name).rename(study / "rejected_attempts" / label)
        raw = Path(config["raw"])
        (raw / "rejected").mkdir(exist_ok=True)
        (raw / name).rename(raw / "rejected" / label)
    directory = study / "allocations" / f"{count}g_retry{attempt}"
    directory.mkdir()
    frozen = Path(config["source"]) / "benchmarking/distributed"
    program = ("import json, sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); from rorqual import nsight; "
               "study = Path(sys.argv[2]); plan = {c['id']: c for c in json.loads((study / 'plan.json').read_text())}; "
               "sys.exit(0 if all([nsight.run_case(study, plan[name]) for name in sys.argv[3:]]) else 1)")
    script = ("#!/usr/bin/env bash\nset -eo pipefail\n"
              + "export BREEZE_RUNTIME_ROOT=" + shlex.quote(config["runtime_root"]) + "\n"
              + "export BREEZE_FIR_ENV=" + shlex.quote(config["environment"]) + "\n"
              + "source " + shlex.quote(str(frozen / "rorqual/nsight_environment.sh")) + "\n"
              + "module -t list > " + shlex.quote(str(directory / "environment.log")) + " 2>&1\n"
              + shlex.join(["python3", "-c", program, str(frozen), str(study), *case_ids]) + "\n")
    (directory / "allocation.sh").write_text(script)
    per_node = gpus_per_node(count)
    command = ["sbatch", "--parsable", "--account=def-fpoulin_gpu", f"--job-name=breeze_nsight_{count}g_retry",
               f"--nodes={nodes(count)}", f"--ntasks={count}", f"--ntasks-per-node={per_node}",
               f"--gpus-per-node=h100:{per_node}", "--cpus-per-task=8", f"--mem={64 * per_node}G",
               "--time=" + config["allocation_limit"], "--distribution=block:block",
               "--chdir=" + config["repository"], "--output=" + str(directory / "slurm-%j.out"),
               str(directory / "allocation.sh")]
    (directory / "submission_command.sh").write_text(shlex.join(command) + "\n")
    job = run_text(command).strip().split(";")[0]
    submission["retry_jobs"].append(dict(gpus=count, job_id=job, cases=case_ids, command=command,
                                         rejected=[f"{name}_job{states[name]['job_id']}" for name in case_ids],
                                         submitted=timestamp()))
    write_json(study / "submission.json", submission)
    render_report(study)
    print(f"Retry of {', '.join(case_ids)} on {count} GPUs: {job}")


def case_states(study):
    states = {}
    for case in read_json(study / "plan.json"):
        path = study / "cases" / case["id"] / "status.json"
        states[case["id"]] = read_json(path) if path.exists() else dict(case=case, state="not_run")
    return states


def render_report(study):
    import csv
    config = read_json(study / "config.json")
    plan = read_json(study / "plan.json")
    states = case_states(study)
    validated = sum(state["state"] == "validated" for state in states.values())
    final = read_json(study / "final_status.json") if (study / "final_status.json").exists() else None
    status = final["state"].upper() if final else ("SUBMITTED" if (study / "submission.json").exists() else "PREPARED")
    lines = ["# Rorqual NVTX Nsight: anelastic and CE on 1, 4, and 8 GPUs (NCCL)", "",
             f"Status: **{status}**; {validated}/{len(plan)} cases validated.", "",
             "Breeze AnelasticDynamics and CompressibleDynamics(ExplicitTimeStepping()), full dry CBL on the fixed",
             "12 km × 12 km × 3 km domain, Float32, WENO5, deterministic (sine) perturbation, 512³ cells per H100.",
             "Multi-GPU halo exchange (and the anelastic distributed pressure solve) uses NCCL; MPI only launches",
             "ranks. Eight GPUs span two nodes, so their NCCL traffic crosses the inter-node network.",
             f"Each launch takes {config['warmup_steps']} warmup steps, then Nsight Systems ({config['nsys']})",
             f"records CUDA, NVTX (Breeze host methods) and MPI for {config['captured_steps']} steps",
             "(cudaProfilerApi start/stop). CPU sampling and context switches are off.", "",
             "| GPUs | Global grid | Partition | Nodes | Δt (s) |", "|---|---|---|---|---|"]
    for gpus, size, partition, time_step in GRIDS:
        lines.append(f"| {gpus} | {size.replace('x', '×')} | {partition.replace('x', '×')} | {nodes(gpus)} | {time_step} |")
    lines += ["", "## Cases", "", "| Case | State | Job | s/step (profiled) | Maximum absolute w | Note |",
              "|---|---|---|---|---|---|"]
    for case in plan:
        state = states[case["id"]]
        measurement = state.get("measurement", {})
        lines.append(f"| {case['id']} | {state['state']} | {state.get('job_id', '')} | "
                     f"{measurement.get('seconds_per_step', '')} | {measurement.get('max_abs_w', '')} | {state.get('error', '')} |")
    lines += ["", "Profiled seconds per step include Nsight overhead; compare them only with other profiled runs."]
    submission = read_json(study / "submission.json") if (study / "submission.json").exists() else {}
    if submission.get("retry_jobs"):
        lines += ["", "## Retried captures", "",
                  "A failed attempt is kept in rejected_attempts/ (reports under the raw directory's rejected/) and",
                  "excluded from the analysis; the retry runs the frozen `run_case` in a new allocation.", ""]
        for item in submission["retry_jobs"]:
            lines.append(f"- {', '.join(item['cases'])}: job {item['job_id']}; rejected attempt(s) "
                         f"{', '.join(item['rejected'])}. Reason: {item.get('reason', 'see the rejected attempt')}")
    summary = study / "rank_summary.csv"
    if summary.exists():
        with summary.open() as stream:
            rows = list(csv.DictReader(stream))
        lines += ["", "## Trace summary per rank", "",
                  "| Case | Rank | Trace s/step | Kernel ms/step | Launches/step | GPU active fraction | Halo kernel fraction | Solve kernel fraction |",
                  "|---|---|---|---|---|---|---|---|"]
        for row in rows:
            steps = int(row["steps"])
            lines.append(f"| {row['case']} | {row['rank']} | {float(row['trace_seconds_per_step']):.4f} | "
                         f"{1e3 * float(row['summed_kernel_seconds']) / steps:.1f} | {int(row['kernel_launches']) / steps:g} | "
                         f"{float(row['timeline_active_fraction']):.3f} | {float(row['halo_kernel_fraction']):.3f} | "
                         f"{float(row['solve_kernel_fraction']):.3f} |")
        lines += ["", "Fractions are of summed kernel time, attributed through CUDA launch correlation IDs to the",
                  "enclosing `call/fill_halo_regions!` and `call/solve!` NVTX ranges. Per-phase totals: phases.csv.",
                  "Per-kernel time per step: [kernels_per_step.md](kernels_per_step.md). Each multi-GPU case against",
                  "the one-GPU case of the same dynamics (slowest rank, weak-scaling efficiency = 100 × T₁/Tₙ):",
                  "[kernel_scaling.md](kernel_scaling.md)."]
    lines += ["", "Finite-field checks on every rank establish a functional run, not field-by-field agreement with a",
              "serial solution. No plots are produced for Nsight runs.", "", "## Exact Nsight result paths", "",
              "On Rorqual; each report has a `profile.sqlite` export beside it. captures.json records sizes,",
              "SHA-256 hashes, kernel and NVTX range counts.", ""]
    for case in plan:
        for rank in range(case["gpus"]):
            lines.append(f"- `{config['raw']}/{case['id']}/rank_{rank}/profile.nsys-rep`")
    lines += ["", "## Reproduction", "",
              f"Frozen source `{config['source']}` (commit in provenance/source_commit.txt) and Julia environment",
              f"`{config['environment']}`; hashes in provenance/hashes.json are checked before and after every case.",
              "Per-case `command.sh`, rank logs, Nsight export logs and `nsys stats` summaries are in cases/.", "",
              "```bash", "source benchmarking/distributed/rorqual/nsight_environment.sh",
              "python3 benchmarking/distributed/rorqual/nsight.py prepare ABSOLUTE_STUDY_DIR",
              "python3 benchmarking/distributed/rorqual/nsight.py submit ABSOLUTE_STUDY_DIR",
              "python3 benchmarking/distributed/rorqual/nsight.py finalize ABSOLUTE_STUDY_DIR --publish  # after the jobs end",
              "```"]
    (study / "README.md").write_text("\n".join(lines) + "\n")


def finalize(study, publish):
    submission = read_json(study / "submission.json")
    jobs = [item["job_id"] for item in submission["jobs"] + submission.get("retry_jobs", [])]
    queued = run_text(["squeue", "--me", "--noheader", "--format=%i|%T"])
    active = [line.strip() for line in queued.splitlines() if line.split("|", 1)[0].strip() in jobs]
    if active:
        raise RuntimeError("Refusing to finalize queued/running allocations: " + ", ".join(active))
    (study / "slurm_accounting.txt").write_text(run_text(
        ["sacct", "--jobs=" + ",".join(jobs), "--parsable2",
         "--format=JobIDRaw,JobName,State,ExitCode,Elapsed,AllocTRES,NodeList"]))
    plan = read_json(study / "plan.json")
    for case_id, state in case_states(study).items():
        if state["state"] == "running":
            state.update(state="failed", error="Allocation ended before the case completed")
        elif state["state"] == "not_run":
            state["error"] = "Allocation ended before this case produced evidence"
        (study / "cases" / case_id).mkdir(exist_ok=True)
        write_json(study / "cases" / case_id / "status.json", state)
    states = case_states(study)
    # analyze_nvtx.py reads only captures marked COMPLETE; only validated cases are listed.
    captures = [dict(state["capture"], state="COMPLETE") for state in states.values() if state["state"] == "validated"]
    write_json(study / "captures.json", captures)
    validated = len(captures)
    analysis_error = None
    if captures:
        tools = Path(read_json(study / "config.json")["source"]) / "benchmarking/distributed"
        pairs = [pair for pair in comparisons(plan)
                 if all(states[name]["state"] == "validated" for name in pair.split("=", 1)[1].split(":"))]
        steps = [("analyze_nvtx.py", [str(study / "captures.json")]),
                 ("kernel_step_table.py", [str(study / "analysis.json")])]
        if pairs:
            steps.append(("kernel_scaling_table.py", [str(study / "analysis.json"), str(study / "kernel_scaling"), *pairs]))
        with (study / "analysis.log").open("w") as output:
            for script, arguments in steps:
                if subprocess.run([sys.executable, str(tools / script), *arguments], stdout=output,
                                  stderr=subprocess.STDOUT, timeout=7200).returncode:
                    analysis_error = f"{script} failed; see analysis.log"
                    break
    write_json(study / "final_status.json", dict(finished=timestamp(), validated=validated, expected=len(plan),
               state="complete" if validated == len(plan) and not analysis_error else "incomplete",
               analysis_error=analysis_error))
    render_report(study)
    if publish:
        publish_study(study, f"Record Rorqual anelastic and CE NVTX Nsight on 1, 4, 8 GPUs: {validated}/{len(plan)} cases validated")
    print(f"Final validation: {validated}/{len(plan)} cases; analysis error: {analysis_error}")
    return 0 if validated == len(plan) and not analysis_error else 1


def publish_study(study, message):
    repository = REPO
    if run_text(["git", "branch", "--show-current"], cwd=repository).strip() != "benchmark-results":
        raise RuntimeError("Refusing publication from a different branch")
    for path in study.rglob("*"):
        if path.is_file() and path.suffix not in (".gz",) and "__pycache__" not in path.parts:
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                continue
            cleaned = "\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n" if text else ""
            if cleaned.encode() != path.read_bytes():
                path.write_text(cleaned)
    relative = str(study.relative_to(repository))
    subprocess.run(["git", "add", "--force", "--", relative], cwd=repository, check=True)
    subprocess.run(["git", "diff", "--cached", "--check", "--", relative], cwd=repository, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--", relative], cwd=repository).returncode:
        subprocess.run(["git", "commit", "-m", message + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
                        "--", relative], cwd=repository, check=True)
    subprocess.run(["git", "pull", "--rebase"], cwd=repository, check=True)
    subprocess.run(["git", "push"], cwd=repository, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("prepare", "submit", "report"):
        commands.add_parser(name).add_argument("study", type=Path)
    again = commands.add_parser("retry")
    again.add_argument("study", type=Path)
    again.add_argument("cases", nargs="+")
    again.add_argument("--reason", required=True)
    worker = commands.add_parser("run-size")
    worker.add_argument("study", type=Path)
    worker.add_argument("count", type=int, choices=GPU_COUNTS)
    finish = commands.add_parser("finalize")
    finish.add_argument("study", type=Path)
    finish.add_argument("--publish", action="store_true")
    arguments = parser.parse_args()
    study = arguments.study.resolve()
    if arguments.action == "prepare":
        prepare(study)
    elif arguments.action == "submit":
        submit(study)
    elif arguments.action == "retry":
        retry(study, arguments.cases)
        submission = read_json(study / "submission.json")
        submission["retry_jobs"][-1]["reason"] = arguments.reason
        write_json(study / "submission.json", submission)
        render_report(study)
    elif arguments.action == "run-size":
        return run_size(study, arguments.count)
    elif arguments.action == "finalize":
        return finalize(study, arguments.publish)
    else:
        render_report(study)
    return 0


if __name__ == "__main__":
    sys.exit(main())
