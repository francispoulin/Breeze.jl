"""Run the whole unprofiled series sequentially inside one 16-GPU allocation."""
import json
import os
from pathlib import Path
import subprocess
import sys
from run_case import check_sources, timestamp, write_json

study = Path(sys.argv[1]).resolve()
check_sources(study)
state = {"job_id": os.environ["SLURM_JOB_ID"], "started": timestamp(), "state": "running", "cases": []}
write_json(study / "progress.json", state)
# Fail before the series if the installed packages or GPU-aware MPI cannot load.
with (study / "provenance/gpu_preflight.log").open("w") as handle:
    subprocess.run(["srun", "--exact", "--nodes=1", "--ntasks=1", "--ntasks-per-node=1", "--cpu-bind=none",
                    "julia", "--startup-file=no", "--threads=1",
                    "--project=" + str(Path(__file__).resolve().parents[1]), "-e",
                    'using MPI, CUDA, Oceananigans, Breeze, NCCL; MPI.Init(); CUDA.versioninfo(); MPI.versioninfo(); println("NCCL loaded"); CUDA.functional() && MPI.has_cuda() || error("CUDA-aware MPI required")'],
                   stdout=handle, stderr=subprocess.STDOUT, check=True)
failed = False
for case in json.loads((study / "plan.json").read_text()):
    print(f"Starting {case['id']}", flush=True)
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("run_case.py")), str(study), case["id"]])
    failed |= result.returncode != 0
    state["cases"].append({"id": case["id"], "exit_code": result.returncode})
    write_json(study / "progress.json", state)
state.update(finished=timestamp(), state="failed" if failed else "validated")
write_json(study / "progress.json", state)
subprocess.run(["/scratch/anditse/breeze_plotting/bin/python", str(Path(__file__).with_name("report.py")), str(study)], check=True)
sys.exit(int(failed))
