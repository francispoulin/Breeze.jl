"""Wait for four free GPUs, run the prepared checks once, and retain their results."""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def can_launch(processes, inventory):
    rows = list(csv.reader(inventory.splitlines()))
    try:
        return (not processes.strip() and len(rows) == 4 and
                {int(row[0]) for row in rows} == {0, 1, 2, 3} and
                all(int(row[1]) >= 8192 for row in rows))
    except (ValueError, IndexError):
        return False


def verify_sources(root, hashes):
    for name, digest in hashes.items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Queued source changed or disappeared: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--publish-results", action="store_true")
    args = parser.parse_args()
    results = Path(__file__).resolve().parent
    root = results.parents[3]
    status_path = args.runtime / "watcher_status.json"
    hashes = json.loads((results / "provenance/source_sha256.json").read_text())
    child = None

    def status(state, **details):
        record = dict(status=state, pid=os.getpid(),
                      updated_utc=datetime.now(timezone.utc).isoformat(),
                      results=str(results), poll_seconds=args.poll_seconds, **details)
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(status_path)

    def stop(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
        status("CANCELLED")
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (args.runtime / "watcher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (args.runtime / "watcher.pid").write_text(str(os.getpid()) + "\n")
        try:
            while True:
                verify_sources(root, hashes)
                try:
                    processes = subprocess.check_output(
                        ["nvidia-smi", "--id=0,1,2,3", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
                    inventory = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
                except subprocess.CalledProcessError as error:
                    status("WAITING_FOR_VALID_GPU_STATUS", error=str(error))
                    time.sleep(args.poll_seconds)
                    continue
                if can_launch(processes, inventory):
                    break
                status("WAITING_FOR_FOUR_FREE_GPUS", occupied_pids=processes.splitlines(), inventory=inventory.splitlines())
                time.sleep(args.poll_seconds)
            verify_sources(root, hashes)
            status("RUNNING")
            with (args.runtime / "logs/controller.log").open("w") as output:
                child = subprocess.Popen(
                    [sys.executable, "-B", str(results / "run_smoke.py"), "--runtime", str(args.runtime)],
                    cwd=root, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                status("RUNNING", controller_pid=child.pid)
                exit_code = child.wait()
            verify_sources(root, hashes)
            path = results / "results.json"
            cases = json.loads(path.read_text())["cases"] if path.exists() else []
            passed = exit_code == 0 and len(cases) == 6 and all(c["status"] == "PASS" for c in cases)
            outcome = "PASS" if passed else "FAIL"
            summary = [f"# Four-GPU smoke checks: {outcome}", "",
                       f"Completed cases: {len(cases)}/6; controller exit: {exit_code}.", "",
                       "| Case | Status | Maximum absolute vertical velocity |", "|------|--------|------------------------------------|"]
            for case in cases:
                value = (case["measurement"] or {}).get("max_abs_w", "unavailable")
                summary.append(f"| {case['case']} | {case['status']} | {value} |")
            summary += ["", "See results.json and per-case logs for rank placement, finite-field checks and launcher messages.",
                        "These are functional checks, not performance or full numerical-validation results."]
            (results / "outcome.md").write_text("\n".join(summary) + "\n")
            (results / "logs/controller.log").write_bytes((args.runtime / "logs/controller.log").read_bytes())
            publication = {}
            if args.publish_results:
                queued_state = json.loads((args.runtime / "state.json").read_text())
                remote = subprocess.check_output(["git", "remote", "get-url", "--push", "origin"], cwd=root, text=True).strip()
                if remote != queued_state["origin_push_url"]:
                    raise RuntimeError("Push destination changed since queueing; results need manual review")
                if subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=root, text=True).strip():
                    raise RuntimeError("Results ready, but unrelated staged changes prevent automatic publication")
                # Do this only after the checks finish; measurements use the frozen checkout.
                subprocess.run(["git", "fetch", "origin", "benchmark-results"], cwd=root, check=True)
                subprocess.run(["git", "merge", "--ff-only", "origin/benchmark-results"], cwd=root, check=True)
                for path in results.rglob("*"):
                    if path.is_file():
                        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()).rstrip() + "\n")
                relative = str(results.relative_to(root))
                subprocess.run(["git", "add", "--", relative], cwd=root, check=True)
                subprocess.run(["git", "diff", "--cached", "--check"], cwd=root, check=True)
                subprocess.run(["git", "commit", "-m", f"Record four-GPU smoke check results: {outcome}"], cwd=root, check=True)
                commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
                push = subprocess.run(["git", "push", "origin", "HEAD:refs/heads/benchmark-results"], cwd=root)
                publication = dict(results_commit=commit, push_exit_code=push.returncode)
            state = "COMPLETE" if passed else "FAILED"
            if publication.get("push_exit_code", 0):
                state = "RESULTS_READY_PUSH_FAILED"
            status(state, tests_status=outcome,
                   completed_cases=len(cases), controller_exit_code=exit_code, **publication)
        except Exception as error:
            status("ERROR", error=str(error))
            raise


if __name__ == "__main__":
    main()
