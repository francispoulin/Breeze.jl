"""Watch Slurm from the login node, then report, commit, and push the study."""
import json
import getpass
from pathlib import Path
import subprocess
import sys
import time
from run_case import REPO, timestamp, write_json

study = Path(sys.argv[1]).resolve()
job = sys.argv[2]
while True:
    result = subprocess.run(["squeue", "--noheader", "--user=" + getpass.getuser(), "--format=%i|%T|%R|%M"], text=True, capture_output=True)
    if result.returncode:
        print(result.stderr, flush=True)
        time.sleep(30)
        continue
    state = next((line.strip() for line in result.stdout.splitlines() if line.split("|", 1)[0].strip() == job), "")
    write_json(study / "scheduler_status.json", {"job_id": job, "checked": timestamp(), "squeue": state})
    if not state:
        break
    time.sleep(30)
accounting = subprocess.check_output(["sacct", "--jobs=" + job, "--parsable2", "--format=JobIDRaw,JobName,State,ExitCode,Elapsed,AllocTRES,NodeList"], text=True)
(study / "slurm_accounting.txt").write_text(accounting)
subprocess.run(["/scratch/anditse/breeze_plotting/bin/python", str(Path(__file__).with_name("report.py")), str(study)], check=True)
validated = 0
for case in json.loads((study / "plan.json").read_text()):
    path = study / "cases" / case["id"] / "status.json"
    validated += path.exists() and json.loads(path.read_text())["state"] == "validated"
write_json(study / "final_status.json", {"job_id": job, "finished": timestamp(), "validated": validated,
                                       "expected": 18, "state": "validated" if validated == 18 else "incomplete"})
branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
if branch != "benchmark-results":
    raise RuntimeError("Refusing to commit results on a different branch")
relative = str(study.relative_to(REPO))
subprocess.run(["git", "add", "--force", "--", relative], cwd=REPO, check=True)
subprocess.run(["git", "diff", "--cached", "--check", "--", relative], cwd=REPO, check=True)
subprocess.run(["git", "commit", "-m", f"Record Rorqual weak scaling: {validated}/18 cases validated", "--", relative], cwd=REPO, check=True)
subprocess.run(["git", "push", "origin", "benchmark-results"], cwd=REPO, check=True)
