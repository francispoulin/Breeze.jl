"""Wait for the measured partition comparison, then capture six warmed Nsight cases."""

import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


STUDY = Path(__file__).resolve().parent.parent
REPO = STUDY.parents[3]
BENCHMARK = STUDY.parent / "weak_scaling_mpi_nccl_2026-11-09_dt0p005"
SCRATCH = Path("/local_scratch/ajtse/breeze_nsight_mpi_nccl_2026-11-09")
SOURCE = SCRATCH / "source"
PROJECT = SOURCE / "benchmarking/distributed"
DRIVER = PROJECT / "distributed_cbl_benchmark.jl"
WRAPPER = STUDY / "execution/profile_rank.sh"
specification = importlib.util.spec_from_file_location("benchmark_helpers", BENCHMARK / "execution/run_series.py")
helpers = importlib.util.module_from_spec(specification)
specification.loader.exec_module(helpers)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(state):
    helpers.write_json(STUDY / "study.json", state)
    profiles = [profile for case in state["cases"] for profile in case.get("profiles", [])]
    helpers.write_json(STUDY / "profile_manifest.json", profiles)
    now = datetime.now(ZoneInfo("America/Toronto")).strftime("%Y-%m-%d %H:%M %Z")
    lines = ["# Rondeau Nsight Systems: MPI and NCCL at Δt = 0.005 s", "",
             f'Updated {now}. **{state["status"]}**; '
             f'{sum(case["state"] == "COMPLETE" for case in state["cases"])}/6 requested cases complete.', "",
             "Both compressible explicit and split-explicit dynamics use Float32, WENO5, the full dry CBL setup,",
             "10 warmup steps, then exactly 10 captured main steps at Δt = 0.005 s. Split-explicit uses 12 acoustic substeps.",
             "The one-GPU grid is 512×512×512; the four-GPU grid is 1024×1024×512 (512³ cells/GPU).",
             "Each dynamics has one shared plain-GPU serial baseline. Four-GPU MPI/NCCL use the faster observed",
             "partition separately for each dynamics/backend, selected from finite 4×1 and 2×2 rerun cases.", "",
             "The controller waits for the entire [unprofiled rerun](../" + BENCHMARK.name + "/README.md) and its final validation",
             "before starting, then waits for the selected devices to have no compute processes. Profiled timings include",
             "profiling overhead and remain separate from unprofiled benchmark timings.", "",
             "## Capture locations", "",
             "Paths below are exact absolute paths on Rondeau. Pending paths are planned locations, not existing results.", "",
             "| Dynamics | Backend | GPUs | Partition | State | Rank | Nsight result file | Size (MiB) |",
             "|---|---|---|---|---|---|---|---|"]
    for case in state["cases"]:
        by_rank = {profile["rank"]: profile for profile in case.get("profiles", [])}
        for rank in range(case["gpus"]):
            profile = by_rank.get(rank)
            path = SCRATCH / "profiles" / case["id"] / f"rank_{rank}/profile.nsys-rep"
            partition = "×".join(map(str, case["partition"])) if case.get("partition") else "awaiting rerun"
            size = f'{profile["bytes"] / 1024**2:.2f}' if profile else "pending"
            lines.append(f'| {case["dynamics"]} | {case["communication"]} | {case["gpus"]} | {partition} | '
                         f'{case["state"]} | {rank} | `{path}` | {size} |')
    lines.extend(["", "Each completed rank also has `profile.sqlite` beside its `.nsys-rep` for trace validation.",
                  "The [profile manifest](profile_manifest.json) records sizes, SHA-256 hashes, SQLite paths and captured kernel counts.", ""])
    if state.get("partition_selection"):
        lines.extend(["## Partition selection from unprofiled measurements", "",
                      "| Dynamics | Backend | 2×2 s/step | 4×1 s/step | Selected |", "|---|---|---|---|---|"])
        for selection in state["partition_selection"]:
            lines.append(f'| {selection["dynamics"]} | {selection["communication"]} | '
                         f'{selection["seconds_per_step_2x2"]:.6f} | {selection["seconds_per_step_4x1"]:.6f} | '
                         f'{selection["partition"][0]}×{selection["partition"][1]} |')
        lines.extend(["", "Selections use the lower observed mean from one timing window per layout, without repeats or error bars.",
                      "Selected layouts may differ across backends; the profiles therefore describe each measured faster configuration.", ""])
    lines.extend(["## Reproduction and validation", "",
                  "The scientific source and resolved environment match the unprofiled rerun. Profiling uses an isolated",
                  f'checkout at `{SOURCE}`; the tracked driver adds an optional `--profile=true` capture range.',
                  "The main checkout and active benchmark source are left unchanged. [Driver patch](provenance/profile_driver.patch),",
                  "[source revision](provenance/source_commit.txt), [environment](provenance/environment.log),",
                  "[study state](study.json), [exact commands](execution/commands.log), [controller log](execution/controller.log).", "",
                  "Nsight traces CUDA, MPI (Open MPI), and NVTX. CPU sampling and context-switch tracing are disabled.",
                  "CUDA profiler start/stop brackets only the requested time steps and their final device synchronization;",
                  "construction, warmup/JIT, and post-run field checks are outside the capture. This Nsight version has no",
                  "dedicated NCCL API trace option; NCCL activity appears through CUDA kernels and any library NVTX ranges.", "",
                  "Before the six requested captures, separate small-grid serial-explicit and two-rank NCCL split-explicit",
                  "pilots validate capture/export and finite fields. They are recorded under `pilots/` and excluded from the six-case count.",
                  "Every requested case requires finite prognostic and velocity fields on every rank, a successful profiler exit,",
                  "a nonempty `.nsys-rep`, and a nonempty CUDA kernel table in each exported SQLite database.",
                  "GPU/process samples verify device assignment and loaded MPI/NCCL libraries. These checks establish",
                  "capture validity and field finiteness; they do not compare distributed model solutions with a serial solution.", "",
                  "No plots are generated for Nsight runs. Raw captures stay in local scratch; this README lists their locations.", "",
                  "From the repository root, launch a fresh study after choosing a new study/scratch directory:", "", "```bash",
                  f'source {STUDY}/execution/environment.sh',
                  f'python3 -u {STUDY}/execution/run_profiles.py > {STUDY}/execution/controller.log 2>&1', "```", "",
                  "The controller refuses an existing study to avoid duplicate captures.", ""])
    (STUDY / "README.md").write_text("\n".join(lines).rstrip() + "\n")


def wait_for_benchmark(state):
    while True:
        benchmark = json.loads((BENCHMARK / "study.json").read_text())
        if benchmark["status"] in ("FAILED", "INTERRUPTED"):
            raise RuntimeError("Unprofiled rerun failed; partition selection cannot proceed")
        validation_file = BENCHMARK / "validation/results.json"
        validation = json.loads(validation_file.read_text()) if validation_file.exists() else {}
        if benchmark["status"] == "COMPLETE" and validation.get("finalized_utc"):
            assert validation["complete_case_coverage"] and validation["valid_performance_cases"] == 14
            return benchmark
        controller = json.loads((BENCHMARK / "execution/controller_process.json").read_text())
        if benchmark["status"] != "COMPLETE" and not (Path("/proc") / str(controller["pid"])).exists():
            raise RuntimeError("Unprofiled rerun controller exited before completion")
        time.sleep(20)


def choose_partitions(state, benchmark):
    selections = []
    for case in state["cases"]:
        if case["gpus"] == 1:
            continue
        pair = [candidate for candidate in benchmark["cases"] if candidate["gpus"] == 4
                and candidate["dynamics"] == case["dynamics"] and candidate["communication"] == case["communication"]]
        assert len(pair) == 2 and {tuple(item["partition"]) for item in pair} == {(2, 2), (4, 1)}
        assert all(item["state"] == "COMPLETE" and item["fields_finite"] and item["performance_valid"] for item in pair)
        assert all(math.isfinite(item["seconds_per_step"]) and item["seconds_per_step"] > 0 for item in pair)
        chosen = min(pair, key=lambda item: (item["seconds_per_step"], item["partition"]))
        case["partition"] = chosen["partition"]
        selections.append({"dynamics": case["dynamics"], "communication": case["communication"],
                           "partition": chosen["partition"], "benchmark_case": chosen["id"],
                           **{f'seconds_per_step_{item["partition"][0]}x{item["partition"][1]}': item["seconds_per_step"]
                              for item in pair}})
    state["partition_selection"] = selections
    helpers.write_json(STUDY / "partition_selection.json", selections)


def verify_sources():
    hashes = json.loads((BENCHMARK / "provenance/source_sha256.json").read_text())
    for name, expected in hashes.items():
        path = Path(name)
        if path.is_relative_to(REPO):
            relative = path.relative_to(REPO)
            if relative.parts[0] in ("src", "ext") or str(relative).startswith("benchmarking/src/"):
                assert digest(SOURCE / relative) == expected, f"Scientific source drift: {relative}"
    for name in ("Project.toml", "Manifest.toml", "LocalPreferences.toml"):
        original = REPO / "benchmarking/distributed" / name
        assert digest(PROJECT / name) == hashes[str(original)]
    pinned = (STUDY / "provenance/source_commit.txt").read_text().strip()
    assert helpers.output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"]) == pinned
    assert helpers.output(["git", "-C", str(SOURCE), "diff", "--name-only"]) == ""
    hashes = json.loads((STUDY / "provenance/profiling_source_sha256.json").read_text())
    assert all(digest(Path(path)) == expected for path, expected in hashes.items()), "Profiling source changed"


def run_case(state, case, pilot=False):
    verify_sources()
    case_root = STUDY / ("pilots" if pilot else "cases") / case["id"]
    raw_root = SCRATCH / ("pilots" if pilot else "profiles") / case["id"]
    case_root.mkdir(parents=True, exist_ok=True)
    while True:
        devices, processes = helpers.gpu_snapshot()
        busy = {row[0] for row in processes}
        available = [int(row[0]) for row in devices if row[1] not in busy]
        if len(available) >= case["gpus"]:
            gpu_ids = available[:case["gpus"]]
            break
        if case["state"] != "WAITING_FOR_GPUS":
            case.update(state="WAITING_FOR_GPUS", waiting_since_utc=timestamp())
            save(state)
            print(timestamp(), "WAITING_FOR_GPUS", case["id"], flush=True)
        time.sleep(10)
    steps = 2 if pilot else 10
    results = STUDY / ("validation/pilot_measurements.csv" if pilot else "profiled_measurements.csv")
    arguments = ["bash", str(WRAPPER), str(case_root), str(raw_root), str(PROJECT), str(DRIVER),
                 "--dynamics=" + case["dynamics"], "--comm=" + ("MPI" if case["communication"] == "SERIAL" else case["communication"]),
                 "--size=" + "x".join(map(str, case["size"])), "--partition=" + "x".join(map(str, case["partition"])),
                 "--time_step=0.005", "--float_type=Float32", f"--warmup_steps={steps}", f"--time_steps={steps}",
                 "--substeps=12", "--profile=true", "--results_file=" + str(results)]
    if case["gpus"] > 1:
        arguments = ["mpiexec", "-n", str(case["gpus"]), "--bind-to", "none", *arguments]
    environment = dict(os.environ, NSIGHT_GPU_IDS=",".join(map(str, gpu_ids)))
    with (STUDY / "execution/commands.log").open("a") as stream:
        stream.write("NSIGHT_GPU_IDS=" + environment["NSIGHT_GPU_IDS"] + " " + shlex.join(arguments) + "\n\n")
    case.update(state="RUNNING", gpu_ids=gpu_ids, started_utc=timestamp(), workers=[], peak_mib={})
    save(state)
    print(timestamp(), "START", case["id"], "GPUs", gpu_ids, flush=True)
    with (case_root / "launch.log").open("w") as log:
        process = subprocess.Popen(arguments, cwd=SOURCE, env=environment, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            devices, processes = helpers.gpu_snapshot()
            workers = {worker["pid"]: worker for worker in case["workers"]}
            with (STUDY / "memory_samples.csv").open("a", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                for gpu, uuid, used, utilization in devices:
                    if int(gpu) not in gpu_ids:
                        continue
                    pids = [int(row[1]) for row in processes if row[0] == uuid]
                    writer.writerow((timestamp(), case["id"], gpu, used, utilization, ";".join(map(str, pids))))
                    case["peak_mib"][gpu] = max(case["peak_mib"].get(gpu, 0), int(used))
                    for pid in pids:
                        worker = helpers.worker_record(pid)
                        if "unavailable" not in worker:
                            workers[pid] = worker
            case["workers"] = list(workers.values())
            helpers.write_json(STUDY / "study.json", state)
            time.sleep(10)
    case.update(exit_code=process.returncode, finished_utc=timestamp())
    assert process.returncode == 0, f'Profiler/worker failed: {case["id"]}'
    with results.open(newline="") as stream:
        matches = [row for row in csv.DictReader(stream) if row["dynamics"] == case["dynamics"]
                   and row["communication"] == case["communication"] and int(row["gpus"]) == case["gpus"]]
    assert len(matches) == 1 and matches[0]["fields_finite"] == "true"
    assert [int(matches[0][name]) for name in ("Nx", "Ny", "Nz")] == case["size"]
    assert [int(matches[0][name]) for name in ("Rx", "Ry")] == case["partition"]
    assert int(matches[0]["steps"]) == steps and matches[0]["float"] == "Float32"
    assert math.isfinite(float(matches[0]["max_abs_w"]))
    case.update(fields_finite=True, seconds_per_step=float(matches[0]["seconds_per_step"]),
                max_abs_w=float(matches[0]["max_abs_w"]))
    assert len(case["workers"]) == case["gpus"], "Unexpected compute processes on selected GPUs"
    assert {int(worker["environment"]["CUDA_VISIBLE_DEVICES"]) for worker in case["workers"]} == set(gpu_ids)
    assert all(worker["environment"]["JULIA_NUM_THREADS"] == "1" for worker in case["workers"])
    assert all(worker["environment"]["JULIA_CUDA_MEMORY_POOL"] == "none" for worker in case["workers"])
    assert all(any("/opt/uw/openmpi/4.1.7-with-cuda/" in name for name in worker["library_mappings"])
               for worker in case["workers"])
    if case["communication"] == "NCCL":
        assert all(any("libnccl.so" in name for name in worker["library_mappings"]) for worker in case["workers"])
    case["profiles"] = []
    for rank in range(case["gpus"]):
        rank_directory = case_root / f"rank_{rank}"
        text = (rank_directory / "profile.out").read_text()
        assert f"PROFILE_START rank={rank} steps={steps}" in text and f"PROFILE_END rank={rank} steps={steps}" in text
        if rank == 0:
            assert "Δt = 0.005" in text and "profile = true" in text
            if case["communication"] == "NCCL":
                assert "NCCLCommunicator" in text
        assert (rank_directory / "exit_code.txt").read_text().strip() == "0"
        path = raw_root / f"rank_{rank}/profile.nsys-rep"
        database = path.with_suffix(".sqlite")
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            kernels = connection.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
        assert kernels > 0 and path.stat().st_size > 0
        case["profiles"].append({"rank": rank, "path": str(path), "bytes": path.stat().st_size,
                                 "sha256": digest(path), "sqlite": str(database), "captured_kernel_events": kernels})
    verify_sources()
    case["state"] = "COMPLETE"
    save(state)
    print(timestamp(), "COMPLETE", case["id"], "finite fields; all rank captures verified", flush=True)


def main():
    if (STUDY / "study.json").exists():
        raise RuntimeError("Study already exists; refusing duplicate profiles")
    for directory in ("cases", "pilots", "provenance", "validation"):
        (STUDY / directory).mkdir(exist_ok=True)
    cases = [{"id": dynamics + "_" + backend + f"_{gpus}g", "dynamics": dynamics, "communication": backend,
              "gpus": gpus, "size": [512, 512, 512] if gpus == 1 else [1024, 1024, 512],
              "partition": [1, 1] if gpus == 1 else None, "state": "PENDING"}
             for dynamics in ("compressible_explicit", "compressible_splitexplicit")
             for backend, gpus in (("SERIAL", 1), ("MPI", 4), ("NCCL", 4))]
    pilots = [{"id": "serial_explicit", "dynamics": "compressible_explicit", "communication": "SERIAL",
               "gpus": 1, "size": [32, 32, 16], "partition": [1, 1], "state": "PENDING"},
              {"id": "nccl_splitexplicit", "dynamics": "compressible_splitexplicit", "communication": "NCCL",
               "gpus": 2, "size": [64, 32, 16], "partition": [2, 1], "state": "PENDING"}]
    state = {"status": "WAITING_FOR_BENCHMARK", "created_utc": timestamp(), "time_step_seconds": 0.005,
             "warmup_steps": 10, "captured_steps": 10, "configured_cases": 6, "cases": cases, "pilots": pilots,
             "source": str(SOURCE), "raw_profiles_root": str(SCRATCH / "profiles"), "partition_reference": str(BENCHMARK)}
    helpers.write_json(STUDY / "execution/controller_process.json",
                       {"pid": os.getpid(), "started_utc": timestamp(),
                        "log": os.environ.get("NSIGHT_CONTROLLER_LOG", str(STUDY / "execution/controller.log"))})
    (STUDY / "execution/commands.log").write_text("# Source execution/environment.sh before these commands.\n")
    with (STUDY / "memory_samples.csv").open("w", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerow(("utc", "case", "gpu", "used_mib", "utilization_percent", "compute_pids"))
    save(state)
    print(timestamp(), "WAITING_FOR_BENCHMARK", BENCHMARK, flush=True)
    try:
        benchmark = wait_for_benchmark(state)
        choose_partitions(state, benchmark)
        state["status"] = "VALIDATING_PROFILE_PILOTS"
        save(state)
        for pilot in pilots:
            run_case(state, pilot, pilot=True)
        state["status"] = "RUNNING"
        save(state)
        for case in cases:
            run_case(state, case)
        state.update(status="COMPLETE", finished_utc=timestamp())
        save(state)
        helpers.write_json(STUDY / "validation/results.json",
                           {"configured_cases": 6, "completed_cases": 6, "verified_rank_captures": 18,
                            "all_fields_finite": True, "all_rank_cuda_kernel_tables_nonempty": True,
                            "isolated_scientific_source_and_environment_match_benchmark": True,
                            "selected_partitions_from_complete_finite_unprofiled_comparison": True,
                            "profile_window": "10 main steps after 10 warmup steps at dt=0.005 s",
                            "finished_utc": timestamp()})
        print(timestamp(), "COMPLETE six cases / eighteen requested rank reports", flush=True)
        subprocess.run([sys.executable, str(STUDY / "execution/finalize_profiles.py")], check=True)
    except Exception as error:
        state.update(status="FAILED", error=str(error), failed_utc=timestamp())
        save(state)
        raise


if __name__ == "__main__":
    main()
