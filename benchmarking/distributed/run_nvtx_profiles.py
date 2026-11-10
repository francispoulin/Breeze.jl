"""Capture the three serial dynamics and both compressible MPI/NCCL paths.

Requires a configured CUDA-aware Open MPI environment and an instantiated Julia
project containing the distributed benchmark dependencies. Raw traces stay in
--scratch; logs, commands, provenance, and the capture manifest go in --results.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sqlite3
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--size", default="512x512x512")
    parser.add_argument("--steps", default=10, type=int)
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.results.mkdir(parents=True, exist_ok=True)
    manifest_path = args.results / "captures.json"
    if manifest_path.exists():
        raise RuntimeError("Choose fresh result paths; captures.json already exists")
    provenance = args.results / "provenance"
    provenance.mkdir()
    for name in ("Project.toml", "Manifest.toml", "LocalPreferences.toml"):
        if (args.project / name).exists():
            shutil.copy2(args.project / name, provenance / name)
    sources = [*root.glob("src/**/*.jl"), root / "benchmarking/src/convective_boundary_layer.jl",
               *Path(__file__).parent.glob("*.jl"), Path(__file__)]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (provenance / "source_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    (provenance / "source_commit.txt").write_text(subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True))
    with (provenance / "environment.log").open("w") as output:
        for command in ([args.julia, "--version"], ["nsys", "--version"],
                        ["mpiexec", "--version"], ["nvidia-smi"]):
            subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
    cases = [(name, "SERIAL", 1) for name in
             ("anelastic", "compressible_explicit", "compressible_splitexplicit")]
    cases += [(name, backend, 4) for name in
              ("compressible_explicit", "compressible_splitexplicit") for backend in ("MPI", "NCCL")]
    nx, ny, nz = map(int, args.size.split("x"))
    records = []
    environment = os.environ.copy()
    environment.update(JULIA_NUM_THREADS="1", JULIA_NUM_GC_THREADS="1",
                       JULIA_PKG_PRECOMPILE_AUTO="0", JULIA_NVTX_CALLBACKS="gc",
                       JULIA_CUDA_MEMORY_POOL="none", NSYS_MPI_STORE_TEAMS_PER_RANK="1")
    for dynamics, backend, ranks in cases:
        case = f"{dynamics}_{backend}_{ranks}g"
        if args.only and case not in args.only:
            continue
        processes = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if processes.strip():
            raise RuntimeError(f"GPUs occupied before {case}: {processes}")
        directory = args.scratch / case
        record = dict(case=case, dynamics=dynamics, backend=backend, ranks=ranks,
                      state="RUNNING", steps=args.steps, warmup=args.warmup,
                      size=[nx if ranks == 1 else 2 * nx, ny if ranks == 1 else 2 * ny, nz],
                      partition=[1, 1] if ranks == 1 else [2, 2],
                      time_step=0.005 if dynamics == "compressible_explicit" else 0.02)
        records.append(record)
        manifest_path.write_text(json.dumps(records, indent=2) + "\n")
        command = ["mpiexec", "--bind-to", "none", "-n", str(ranks), "bash",
                   str(root / "benchmarking/distributed/profile_nvtx_rank.sh"),
                   str(args.project), args.julia,
                   str(root / "benchmarking/distributed/distributed_cbl_benchmark.jl"), str(directory),
                   f"--dynamics={dynamics}", f"--comm={'MPI' if ranks == 1 else backend}",
                   "--size=" + "x".join(map(str, record["size"])),
                   "--partition=" + "x".join(map(str, record["partition"])),
                   f"--warmup_steps={args.warmup}", f"--time_steps={args.steps}",
                   f"--time_step={record['time_step']}", "--substeps=12", "--profile=true",
                   "--nvtx=true", "--random_perturbation=false",
                   f"--results_file={args.results / 'profiled_measurements.csv'}"]
        with (args.results / "commands.log").open("a") as output:
            output.write(shlex.join(command) + "\n")
        print(f"START {case}", flush=True)
        start = time.time()
        try:
            log = args.results / f"{case}.log"
            with log.open("w") as output:
                subprocess.run(command, env=environment, stdout=output, stderr=subprocess.STDOUT,
                               check=True, cwd=root)
            if "fields_finite=true" not in log.read_text():
                raise RuntimeError(f"Field finiteness check missing: {log}")
            devices_by_rank = {int(rank): uuid.lower().removeprefix("gpu-") for rank, uuid in
                               re.findall(r"DEVICE rank=(\d+) uuid=([\w-]+)", log.read_text())}
            assert len(devices_by_rank) == ranks, "Missing runtime GPU UUIDs"
            record["profiles"] = []
            for rank in range(ranks):
                report = directory / f"rank_{rank}/profile.nsys-rep"
                database = report.with_suffix(".sqlite")
                subprocess.run(["nsys", "export", "--type=sqlite", f"--output={database}", str(report)],
                               check=True, stdout=subprocess.DEVNULL)
                with sqlite3.connect(database) as connection:
                    kernels = connection.execute("SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
                    ranges = connection.execute("SELECT count(*) FROM NVTX_EVENTS WHERE end IS NOT NULL").fetchone()[0]
                    known_devices = {row[0].lower() for row in connection.execute("SELECT uuid FROM TARGET_INFO_GPU")}
                assert kernels > 0 and ranges > 0
                assert devices_by_rank[rank] in known_devices
                record["profiles"].append(dict(rank=rank, report=str(report), sqlite=str(database),
                    bytes=report.stat().st_size, sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
                    kernels=kernels, ranges=ranges, gpu_uuid=devices_by_rank[rank]))
            assert len({profile["gpu_uuid"] for profile in record["profiles"]}) == ranks
            record.update(state="COMPLETE", elapsed_seconds=time.time() - start)
            print(f"COMPLETE {case}", flush=True)
        except BaseException as error:
            record.update(state="FAILED", error=str(error))
            raise
        finally:
            manifest_path.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
