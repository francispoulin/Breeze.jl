"""Prepare, submit, run, and retain the agreed Fir MPI weak-scaling series."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rorqual.run_case import timestamp, validate, write_json

REPO = Path(__file__).resolve().parents[3]
DYNAMICS = ("anelastic", "compressible_explicit", "compressible_splitexplicit")
GPU_COUNTS = (1, 2, 4, 8, 16)
CASE_SECONDS = 3000


def build_plan():
    return [dict(id=f"{dynamics}_{'SERIAL' if count == 1 else 'MPI'}_{count}g_{count}x1",
                 dynamics=dynamics, gpus=count, size=f"{512 * count}x512x512",
                 partition=f"{count}x1", backend="MPI", time_step=0.01 / count)
            for count in GPU_COUNTS for dynamics in DYNAMICS]


def read_json(path):
    return json.loads(path.read_text())


def run_text(command, **kwargs):
    return subprocess.check_output(command, text=True, **kwargs)


def check_sources(study):
    for name, expected in read_json(study / "provenance/hashes.json").items():
        path = Path(name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen source/environment changed: {path}")


def validate_preflight_placement(output, count):
    placements = re.findall(r"PREFLIGHT rank=(\d+) host=(\S+) gpu_uuid=(\S+)", output)
    if len(placements) != count or {int(item[0]) for item in placements} != set(range(count)):
        raise RuntimeError("Missing or duplicate rank in GPU placement preflight")
    if len({(host, uuid) for _, host, uuid in placements}) != count:
        raise RuntimeError("Ranks share a GPU in placement preflight")
    return placements


def prepare(study):
    if study.exists():
        raise RuntimeError("Use a fresh study directory")
    if run_text(["git", "status", "--porcelain"], cwd=REPO).strip():
        raise RuntimeError("Commit the source before freezing the study")
    runtime = REPO / ".runtime/studies" / study.name
    if runtime.exists():
        raise RuntimeError("Runtime study already exists")
    source = runtime / "source"
    environment = runtime / "environment"
    source.mkdir(parents=True)
    environment.mkdir()
    (study / "provenance").mkdir(parents=True)
    (study / "cases").mkdir()
    (study / "allocations").mkdir()
    (study / "provenance/source_commit.txt").write_text(run_text(["git", "rev-parse", "HEAD"], cwd=REPO))
    paths = ["src", "ext", "Project.toml", "AGENTS.md", "benchmarking/src",
             "benchmarking/distributed/fir", "benchmarking/distributed/rorqual/run_case.py",
             "benchmarking/distributed/rorqual/rank.sh", "benchmarking/distributed/distributed_cbl_benchmark.jl"]
    if (REPO / "Artifacts.toml").exists():
        paths.append("Artifacts.toml")
    archive = study / "provenance/source.tar.gz"
    subprocess.run(["git", "archive", "--format=tar.gz", "--output=" + str(archive), "HEAD", "--", *paths], cwd=REPO, check=True)
    with tarfile.open(archive) as handle:
        handle.extractall(source, filter="data")
    installed = Path(os.environ["BREEZE_FIR_ENV"])
    for name in ("Project.toml", "Manifest.toml", "LocalPreferences.toml"):
        contents = (installed / name).read_text().replace(f'path = "{REPO}"', f'path = "{source}"')
        (environment / name).write_text(contents)
        (study / "provenance" / name).write_text(contents)
    (study / "provenance/setup.log").write_text((installed / "setup.log").read_text())
    config = dict(repository=str(REPO), source=str(source), environment=str(environment),
                  runtime=str(runtime), runtime_root=str(REPO / ".runtime"), machine="fir",
                  prepared=timestamp(), warmup_steps=10, timed_steps=500, substeps=12,
                  float_type="Float32", random_perturbation=False, simplified=False,
                  case_limit_seconds=CASE_SECONDS, allocation_limit="03:00:00")
    write_json(study / "config.json", config)
    write_json(study / "plan.json", build_plan())
    files = [path for path in source.rglob("*") if path.is_file()]
    files += list(environment.iterdir()) + [study / "plan.json", study / "config.json"]
    write_json(study / "provenance/hashes.json", {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    write_json(study / "progress.json", dict(state="prepared", expected=len(build_plan()), validated=0, checked=timestamp()))
    generate_scripts(study)
    render_report(study)
    print(f"Prepared {len(build_plan())} cases: {study}")


def script_prefix(config):
    return ("#!/usr/bin/env bash\nset -eo pipefail\n"
            + "export BREEZE_RUNTIME_ROOT=" + shlex.quote(config["runtime_root"]) + "\n"
            + "export BREEZE_FIR_ENV=" + shlex.quote(config["environment"]) + "\n"
            + "source " + shlex.quote(str(Path(config["source"]) / "benchmarking/distributed/fir/environment.sh")) + "\n")


def generate_scripts(study):
    config = read_json(study / "config.json")
    worker = Path(config["source"]) / "benchmarking/distributed/fir/study.py"
    for count in GPU_COUNTS:
        directory = study / "allocations" / f"{count}g"
        directory.mkdir()
        command = ["python3", str(worker), "run-size", str(study), str(count)]
        (directory / "allocation.sh").write_text(script_prefix(config) + shlex.join(command) + "\n")
    command = ["python3", str(worker), "finalize", str(study), "--publish"]
    (study / "finalize.sh").write_text(script_prefix(config) + shlex.join(command) + "\n")


def submit(study):
    check_sources(study)
    if (study / "submission.json").exists():
        raise RuntimeError("Study already submitted; refusing duplicate jobs")
    submission = dict(submitted=timestamp(), jobs=[], state="submitting")
    write_json(study / "submission.json", submission)
    for count in GPU_COUNTS:
        directory = study / "allocations" / f"{count}g"
        nodes, per_node = max(1, count // 4), min(4, count)
        command = ["sbatch", "--parsable", "--account=def-fpoulin_gpu", "--job-name=breeze_mpi_" + str(count) + "g",
                   "--nodes=" + str(nodes), "--ntasks=" + str(count), "--ntasks-per-node=" + str(per_node),
                   "--gpus-per-node=h100:" + str(per_node), "--cpus-per-task=8", "--mem=" + str(64 * per_node) + "G",
                   "--time=03:00:00", "--distribution=block:block", "--chdir=" + config_path(study, "repository"),
                   "--output=" + str(directory / "slurm-%j.out"), str(directory / "allocation.sh")]
        (directory / "submission_command.sh").write_text(shlex.join(command) + "\n")
        job = run_text(command).strip().split(";")[0]
        submission["jobs"].append(dict(gpus=count, job_id=job, command=command))
        write_json(study / "submission.json", submission)
        print(f"Submitted {count} GPUs: {job}", flush=True)
    dependency = "afterany:" + ":".join(item["job_id"] for item in submission["jobs"])
    command = ["sbatch", "--parsable", "--account=def-fpoulin", "--job-name=breeze_finalize", "--nodes=1",
               "--ntasks=1", "--cpus-per-task=1", "--mem=4G", "--time=00:20:00", "--dependency=" + dependency,
               "--chdir=" + config_path(study, "repository"), "--output=" + str(Path(config_path(study, "runtime")) / "finalize-%j.out"),
               str(study / "finalize.sh")]
    (study / "finalize_submission_command.sh").write_text(shlex.join(command) + "\n")
    submission.update(finalizer_job_id=run_text(command).strip().split(";")[0], finalizer_command=command, state="submitted")
    write_json(study / "submission.json", submission)
    render_report(study)
    print(f"Finalizer: {submission['finalizer_job_id']}")


def config_path(study, name):
    return read_json(study / "config.json")[name]


def case_command(case, study):
    config = read_json(study / "config.json")
    source = Path(config["source"])
    directory = study / "cases" / case["id"]
    count = case["gpus"]
    return ["srun", "--mpi=pmix", "--exact", "--kill-on-bad-exit=1", "--time=00:50:00", "--cpu-bind=none",
            "--gpus-per-task=h100:1", "--gpu-bind=single:1", "--distribution=block:block", "--nodes=" + str(max(1, count // 4)),
            "--ntasks=" + str(count), "--ntasks-per-node=" + str(min(4, count)),
            "bash", str(source / "benchmarking/distributed/rorqual/rank.sh"), str(directory),
            "julia", "--startup-file=no", "--threads=1", "--project=" + config["environment"],
            str(source / "benchmarking/distributed/distributed_cbl_benchmark.jl"),
            "--dynamics=" + case["dynamics"], "--comm=MPI", "--size=" + case["size"],
            "--partition=" + case["partition"], "--float_type=Float32", "--simplified=false",
            "--random_perturbation=false", "--time_step=" + str(case["time_step"]),
            "--warmup_steps=10", "--time_steps=500", "--substeps=12", "--profile=false", "--nvtx=false",
            "--results_file=" + str(directory / "measurement.csv")]


def validate_measurement(case, directory, returncode):
    row = validate(case, directory, returncode)
    dimensions = tuple(map(int, case["size"].split("x")))
    if (row["dynamics"], row["float"], *(int(row[key]) for key in ("Nx", "Ny", "Nz", "Rx", "Ry"))) != (
            case["dynamics"], "Float32", *dimensions, case["gpus"], 1):
        raise RuntimeError("Measurement dynamics, precision, grid, or partition differs from plan")
    maximum_velocity = float(row["max_abs_w"])
    if not math.isfinite(maximum_velocity) or maximum_velocity < 0:
        raise RuntimeError("Invalid maximum vertical velocity")
    hardware = list(directory.glob("hardware_rank_*.txt"))
    if len(hardware) != max(1, case["gpus"] // 4) or any("H100" not in path.read_text() for path in hardware):
        raise RuntimeError("Missing H100 hardware evidence on an allocated node")
    return row


def run_case(study, case):
    directory = study / "cases" / case["id"]
    directory.mkdir(exist_ok=False)
    status = dict(case=case, state="running", started=timestamp(), job_id=os.environ["SLURM_JOB_ID"])
    write_json(directory / "status.json", status)
    try:
        check_sources(study)
        command = case_command(case, study)
        (directory / "command.sh").write_text(shlex.join(command) + "\n")
        with (directory / "launcher.log").open("w") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, timeout=CASE_SECONDS + 60)
        status["measurement"] = validate_measurement(case, directory, result.returncode)
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
        capture = 'module -t list 2>&1; julia --version; mpiexec --version; scontrol show job "$SLURM_JOB_ID"'
        with (directory / "environment.log").open("w") as handle:
            subprocess.run(["bash", "-c", capture], stdout=handle, stderr=subprocess.STDOUT, check=True)
        preflight = ('using MPI; MPI.Init(); using CUDA, Oceananigans, Breeze, Printf; '
                     'using Oceananigans.DistributedComputations: Distributed, Partition; '
                     'count = MPI.Comm_size(MPI.COMM_WORLD); rank = MPI.Comm_rank(MPI.COMM_WORLD); '
                     'arch = count == 1 ? GPU() : Distributed(GPU(); partition=Partition(count, 1)); '
                     '@printf("PREFLIGHT rank=%d host=%s gpu_uuid=%s\\n", rank, gethostname(), string(CUDA.uuid(CUDA.device()))); '
                     'CUDA.functional() && MPI.has_cuda() || error("CUDA-aware MPI and a functional GPU are required"); '
                     'if rank == 0; CUDA.versioninfo(); MPI.versioninfo(); println("Breeze path: ", pathof(Breeze)); end; '
                     'MPI.Barrier(MPI.COMM_WORLD)')
        command = ["srun", "--mpi=pmix", "--exact", "--kill-on-bad-exit=1", "--nodes=" + str(max(1, count // 4)),
                   "--ntasks=" + str(count), "--ntasks-per-node=" + str(min(4, count)), "--cpu-bind=none",
                   "--gpus-per-task=h100:1", "--gpu-bind=single:1", "--distribution=block:block",
                   "julia", "--startup-file=no", "--threads=1", "--project=" + config_path(study, "environment"), "-e", preflight]
        (directory / "preflight_command.sh").write_text(shlex.join(command) + "\n")
        with (directory / "gpu_preflight.log").open("w") as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=900)
        placements = validate_preflight_placement((directory / "gpu_preflight.log").read_text(), count)
        write_json(directory / "gpu_placement.json", dict(placements=placements, verified=timestamp()))
        for case in read_json(study / "plan.json"):
            if case["gpus"] == count:
                print(f"STARTING {case['id']}", flush=True)
                passed = run_case(study, case)
                allocation["cases"].append(dict(id=case["id"], validated=passed))
                write_json(directory / "status.json", allocation)
        allocation["state"] = "validated" if all(item["validated"] for item in allocation["cases"]) else "failed"
    except Exception as error:
        allocation.update(state="failed", error=str(error))
        print(f"Allocation failed: {error}", flush=True)
    finally:
        allocation["finished"] = timestamp()
        write_json(directory / "status.json", allocation)
    return 0 if allocation["state"] == "validated" else 1


def render_report(study):
    from report import report
    report(study)


def finalize(study, publish):
    submission = read_json(study / "submission.json")
    allocations = submission["jobs"] + submission.get("retry_jobs", [])
    expected_jobs = {item["job_id"] for item in allocations}
    queued = run_text(["squeue", "--me", "--noheader", "--format=%i|%T"])
    active = [line.strip() for line in queued.splitlines() if line.split("|", 1)[0].strip() in expected_jobs]
    if active:
        raise RuntimeError("Refusing to finalize queued/running allocations: " + ", ".join(active))
    jobs = ",".join(item["job_id"] for item in allocations)
    accounting = run_text(["sacct", "--jobs=" + jobs, "--parsable2",
                           "--format=JobIDRaw,JobName,State,ExitCode,Elapsed,AllocTRES,NodeList"])
    (study / "slurm_accounting.txt").write_text(accounting)
    plan = read_json(study / "plan.json")
    for case in plan:
        directory = study / "cases" / case["id"]
        path = directory / "status.json"
        status = read_json(path) if path.exists() else dict(case=case, state="not_run", error="Allocation ended before this case produced evidence")
        if status["state"] == "running":
            status.update(state="failed", error="Allocation ended before the case completed")
        if status["state"] == "validated":
            try:
                status["measurement"] = validate_measurement(case, directory, 0)
            except Exception as error:
                status.update(state="failed", error="Final validation: " + str(error))
        directory.mkdir(exist_ok=True)
        write_json(path, status)
    try:
        check_sources(study)
        source_error = None
    except RuntimeError as error:
        source_error = str(error)
        for case in plan:
            path = study / "cases" / case["id"] / "status.json"
            status = read_json(path)
            if status["state"] == "validated":
                status.update(state="failed", error=source_error)
                write_json(path, status)
    validated = sum(read_json(study / "cases" / case["id"] / "status.json")["state"] == "validated" for case in plan)
    write_json(study / "final_status.json", dict(finished=timestamp(), validated=validated, expected=len(plan),
               state="validated" if validated == len(plan) else "incomplete", source_error=source_error))
    render_report(study)
    if publish:
        repository = Path(config_path(study, "repository"))
        if run_text(["git", "branch", "--show-current"], cwd=repository).strip() != "benchmark-results":
            raise RuntimeError("Refusing publication from a different branch")
        # Pull before edits, as required by AGENTS.md. Generated results are scoped to this study.
        subprocess.run(["git", "pull", "--ff-only"], cwd=repository, check=True)
        for path in study.rglob("*"):
            if path.is_file() and path.suffix not in (".gz", ".png"):
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
            subprocess.run(["git", "commit", "-m", f"Record Fir MPI weak scaling: {validated}/{len(plan)} cases validated", "--", relative], cwd=repository, check=True)
            subprocess.run(["git", "push"], cwd=repository, check=True)
    print(f"Final validation: {validated}/{len(plan)} cases")
    return 0 if validated == len(plan) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("prepare", "submit", "report"):
        commands.add_parser(name).add_argument("study", type=Path)
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
    elif arguments.action == "run-size":
        return run_size(study, arguments.count)
    elif arguments.action == "finalize":
        return finalize(study, arguments.publish)
    else:
        render_report(study)
    return 0


if __name__ == "__main__":
    sys.exit(main())
