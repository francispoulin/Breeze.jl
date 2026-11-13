"""Small functional checks of the existing distributed CBL command-line driver."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--allow-shared-gpus", action="store_true")
    args = parser.parse_args()
    results = Path(__file__).resolve().parent
    root = results.parents[3]
    driver = root / "benchmarking/distributed/distributed_cbl_benchmark.jl"
    julia = "/u/ajtse/.local/opt/julia-1.12.7/bin/julia"
    cases = [(dynamics, backend, "Float64" if dynamics == "anelastic" else "Float32")
             for dynamics in ("anelastic", "compressible_explicit", "compressible_splitexplicit")
             for backend in ("MPI", "NCCL")]
    (results / "logs").mkdir(exist_ok=True)
    records = []
    for dynamics, backend, precision in cases:
        ranks = 4
        case = f"{dynamics}_{backend}_{ranks}g"
        gpus = "0,1,2,3"
        processes = subprocess.check_output(
            ["nvidia-smi", "--id=" + gpus, "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if processes.strip() and not args.allow_shared_gpus:
            raise RuntimeError(f"Selected GPUs occupied before {case}: {processes}")
        inventory = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        if any(int(line.split(",")[1]) < 8192 for line in inventory.splitlines()):
            raise RuntimeError("A selected GPU has less than 8 GiB available")
        (results / "logs" / f"{case}.gpu_before.csv").write_text(inventory)
        csv_path = results / "logs" / f"{case}.csv"
        assert not csv_path.exists(), "Use fresh output paths for a new smoke test"
        command = ["timeout", "--foreground", "--kill-after=10s", "300s", "mpiexec", "--bind-to", "none", "-n", str(ranks),
                   julia, "--threads=1", "--project=" + str(args.runtime / "environment"), str(driver),
                   "--dynamics=" + dynamics, "--comm=" + ("MPI" if backend == "SERIAL" else backend),
                   "--size=32x32x32", "--partition=4x1",
                   "--float_type=" + precision, "--time_step=0.005", "--warmup_steps=1", "--time_steps=3",
                   "--random_perturbation=false", "--profile=false", "--results_file=" + str(csv_path)]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpus
        (results / "logs" / f"{case}.command.sh").write_text(
            f"CUDA_VISIBLE_DEVICES={gpus} " + shlex.join(command) + "\n")
        print("RUN", case, flush=True)
        started = time.monotonic()
        log_path = results / "logs" / f"{case}.log"
        with log_path.open("w") as output:
            process = subprocess.run(command, cwd=root, env=environment, stdout=output, stderr=subprocess.STDOUT)
        log = log_path.read_text()
        placement = re.findall(r"PLACEMENT rank=(\d+) host=(\S+) gpu_uuid=(\S+) communicator=(\S+)", log)
        validation = re.findall(r"VALIDATION rank=(\d+) fields_finite=(\S+)", log)
        measurements = list(csv.DictReader(csv_path.open())) if csv_path.exists() else []
        expected_communicator = "none" if ranks == 1 else ("NCCLCommunicator" if backend == "NCCL" else "Comm")
        passed = (process.returncode == 0 and len(measurements) == 1 and len(placement) == ranks
                  and {int(p[0]) for p in placement} == set(range(ranks))
                  and len({p[2] for p in placement}) == ranks
                  and all(p[3] == expected_communicator for p in placement)
                  and len(validation) == ranks and all(v[1] == "true" for v in validation)
                  and measurements[0]["fields_finite"] == "true"
                  and measurements[0]["communication"] == backend and measurements[0]["steps"] == "3")
        record = dict(case=case, status="PASS" if passed else "FAIL", exit_code=process.returncode,
                      elapsed_including_startup_seconds=time.monotonic() - started,
                      precision=precision, ranks=ranks, gpu_indices=gpus, placements=placement,
                      shared_gpus_allowed=args.allow_shared_gpus, existing_gpu_pids=processes.splitlines(),
                      rank_validation=validation, measurement=measurements[0] if measurements else None,
                      launcher_messages=[line for line in log.splitlines() if "PMIX ERROR:" in line])
        records.append(record)
        (results / "results.json").write_text(json.dumps(dict(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(), size=[32, 32, 32],
            warmup_steps=1, timed_steps=3, time_step=0.005, cases=records), indent=2) + "\n")
        print(record["status"], case, "exit=" + str(process.returncode), flush=True)
    print("SUMMARY", sum(r["status"] == "PASS" for r in records), "/", len(records), "cases passed", flush=True)


if __name__ == "__main__":
    main()
