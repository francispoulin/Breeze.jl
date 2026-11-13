"""Queue two efficiency figures, with a strict successful-anelastic dependency."""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from ocean_inputs import discover_complete_series


TERMINAL = {"COMPLETE", "FAILED", "ERROR", "CANCELLED", "RESULTS_READY_PUSH_FAILED",
            "RESULTS_READY_PUBLICATION_FAILED"}
SUCCESSFUL = {"COMPLETE", "RESULTS_READY_PUSH_FAILED", "RESULTS_READY_PUBLICATION_FAILED"}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def anelastic_gate(status, data):
    if status.get("status") not in TERMINAL:
        return "WAIT"
    if status.get("status") not in SUCCESSFUL or not isinstance(data, dict) or data.get("status") != "COMPLETE":
        return "SKIP"
    cases = data.get("cases", [])
    if len(cases) != 3 or {case.get("ranks") for case in cases} != {1, 2, 4}:
        return "SKIP"
    try:
        for case in cases:
            n = case["ranks"]
            row = case["measurement"]
            expected = dict(dynamics="anelastic", gpus=str(n), Nx=str(512*n), Ny="512", Nz="512",
                            Rx=str(n), Ry="1", steps="500", float="Float32", fields_finite="true",
                            communication="SERIAL" if n == 1 else "MPI")
            if (case["state"] != "COMPLETE" or case["exit_code"] != 0
                    or any(row.get(key) != value for key, value in expected.items())
                    or not math.isfinite(float(row["seconds_per_step"])) or float(row["seconds_per_step"]) <= 0
                    or not math.isfinite(float(row["max_abs_w"]))):
                return "SKIP"
            placements = case["placements"]
            validation = case["rank_validation"]
            if (len(placements) != n or {int(p[0]) for p in placements} != set(range(n))
                    or len({p[2] for p in placements}) != n
                    or any(p[3] != ("none" if n == 1 else "Comm") for p in placements)
                    or len(validation) != n or {int(v[0]) for v in validation} != set(range(n))
                    or any(v[1] != "true" for v in validation)):
                return "SKIP"
        if len({case["source_commit"] for case in cases}) != 1:
            return "SKIP"
    except (KeyError, TypeError, ValueError, IndexError):
        return "SKIP"
    return "READY"


def verify_sources(root, hashes):
    for name, digest in hashes.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen plotting source changed: {name}")


def write_report(results, runtime, tasks, state):
    lines = ["# Five-model weak-scaling efficiency figures", "", f"Status: **{state}**.", "",
             "Both figure jobs depend on a successful anelastic weak-scaling study. The gate",
             "requires completed 1-, 2- and 4-GPU cases, exit code zero, expected 512³ cells/GPU,",
             "positive finite timings, finite fields on every rank, correct communicator and",
             "distinct GPU placements. A single case finishing or a successful process exit alone",
             "does not unlock plotting. A failed, cancelled or invalid anelastic study skips both jobs.", "",
             "| Figure | Model curves | Job state |", "|--------|--------------|-----------|"]
    lines += [f"| Breeze | Anelastic, compressible explicit, compressible split-explicit | {tasks['breeze']} |",
              f"| Oceananigans | Hydrostatic, nonhydrostatic | {tasks['oceananigans']} |", "",
              "Each figure uses 1, 2 and 4 A100 GPUs, 512³ cells per GPU and efficiency",
              "100 × T₁/Tₙ. PNG and SVG outputs are produced under this report directory.", "",
              "The Breeze figure uses the new anelastic study and archived unprofiled MPI",
              "compressible measurements from 9–10 November. At four GPUs, anelastic uses 4×1",
              "and the compressible models use 2×2; all local grids remain 512³. The studies use",
              "different source revisions. Input paths, hashes and normalized timings are retained.", "",
              "The Oceananigans job additionally waits for complete matching hydrostatic and",
              "nonhydrostatic series in the Rondeau canonical result directories. It requires",
              "COMPLETED suite cases, finite rank results, A100 metadata, unprofiled GPU execution",
              "and exactly 512³ cells per GPU. Each model's points must belong to one consistent",
              "series/configuration/version. Dagger and archived directories are excluded.",
              "Oceananigans times use the median of the slowest-rank synchronized windows per step.",
              "Missing data leave the job waiting; no benchmark is launched and no points are invented.", "",
              "## Queue", "",
              f"Live job states: `{runtime}/status.json`.", f"Watcher log: `{runtime}/logs/watcher.log`.",
              "The watcher polls every 30 seconds and survives this chat. Rendering starts only",
              "after the dependency gate passes. The Breeze job can finish while Oceananigans",
              "continues waiting for its data. If the dependency becomes invalid, rendering stops.", "",
              "Final figures: breeze_weak_efficiency.png/.svg and oceananigans_weak_efficiency.png/.svg.",
              "The live figures are in the frozen worktree under the runtime path above; the",
              "original checkout retains its queued README until pulled after publication.",
              "When both jobs finish, the watcher fast-forwards from origin/benchmark-results,",
              "commits only this report and its figures, then immediately attempts a normal push.",
              "No merge commit or force-push is used. All temporary files and caches stay in /u/ajtse."]
    for name in ("breeze", "oceananigans"):
        if tasks[name] == "COMPLETE":
            lines += ["", f"![{name} weak-scaling efficiency]({name}_weak_efficiency.png)"]
    (results / "README.md").write_text("\n".join(lines) + "\n")


def render_breeze(root, results, data):
    helper = root / "benchmarking/results/rondeau/anelastic_weak_scaling_512_2026-11-13"
    sys.path.insert(0, str(helper))
    from plot_efficiency import draw_curves, weak_efficiencies
    curves = [("Anelastic", weak_efficiencies([c["measurement"] for c in data["cases"]]))]
    with (helper / "provenance/compressible_measurements.csv").open() as stream:
        archived = list(csv.DictReader(stream))
    for model, label in [("compressible_explicit", "Compressible explicit"),
                         ("compressible_splitexplicit", "Compressible split-explicit")]:
        rows = [r for r in archived if r["dynamics"] == model and r["communication"] in ("SERIAL", "MPI")
                and (int(r["gpus"]) < 4 or (r["Rx"], r["Ry"]) == ("2", "2"))]
        curves.append((label, weak_efficiencies(rows)))
    write_json(results / "breeze_inputs.json", dict(anelastic_results=data, curves=curves))
    draw_curves(curves, "Breeze weak-scaling efficiency",
                "4 GPUs: anelastic 4×1; compressible models 2×2 · Unprofiled timings\n"
                "Compressible data: 9–10 Nov; anelastic: new study. Source revisions differ.",
                results / "breeze_weak_efficiency")


def render_ocean(root, results, series):
    helper = root / "benchmarking/results/rondeau/anelastic_weak_scaling_512_2026-11-13"
    sys.path.insert(0, str(helper))
    from plot_efficiency import draw_curves, weak_efficiencies
    curves = [(label, weak_efficiencies(series[name])) for name, label in
              [("hydrostatic", "Hydrostatic"), ("nonhydrostatic", "Nonhydrostatic")]]
    write_json(results / "oceananigans_inputs.json", series)
    draw_curves(curves, "Oceananigans weak-scaling efficiency",
                "Unprofiled timings · Median of slowest-rank windows per step · Efficiency = 100 × T₁/Tₙ",
                results / "oceananigans_weak_efficiency")


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
    tasks = dict(breeze="WAITING_FOR_ANELASTIC", oceananigans="WAITING_FOR_ANELASTIC")

    def status(state, **details):
        write_json(runtime / "status.json", dict(status=state, pid=os.getpid(), tasks=tasks,
                   updated_utc=datetime.now(timezone.utc).isoformat(), results=str(results), **details))

    def stop(signum, frame):
        status("CANCELLED")
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (runtime / "watcher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (runtime / "watcher.pid").write_text(str(os.getpid()) + "\n")
        try:
            if any(results.glob("*_weak_efficiency.png")):
                raise RuntimeError("Use a fresh output directory for another plot run")
            while True:
                verify_sources(root, hashes)
                try:
                    parent = json.loads(Path(plan["anelastic_status"]).read_text())
                    data_path = Path(parent["results"]) / "results.json"
                    data = json.loads(data_path.read_text()) if data_path.exists() else None
                except (OSError, ValueError, KeyError) as error:
                    status("WAITING_FOR_ANELASTIC_STATUS", error=str(error))
                    time.sleep(30)
                    continue
                gate = anelastic_gate(parent, data)
                if gate == "WAIT":
                    status("WAITING_FOR_ANELASTIC_SUCCESS", predecessor_status=parent.get("status"))
                    time.sleep(30)
                    continue
                if gate == "SKIP":
                    for name in ("breeze", "oceananigans"):
                        for suffix in (".png", ".svg"):
                            (results / f"{name}_weak_efficiency{suffix}").unlink(missing_ok=True)
                    tasks = dict.fromkeys(tasks, "SKIPPED_ANELASTIC_UNSUCCESSFUL")
                    outcome = "SKIPPED_ANELASTIC_UNSUCCESSFUL"
                    break
                if tasks["breeze"] != "COMPLETE":
                    tasks["breeze"] = "RUNNING"
                    status("RENDERING_BREEZE")
                    render_breeze(root, results, data)
                    tasks["breeze"] = "COMPLETE"
                series = {}
                coverage = {}
                for name, settings in plan["oceananigans_inputs"].items():
                    series[name], coverage[name] = discover_complete_series(Path(settings["root"]), settings["model_type"])
                if any(rows is None for rows in series.values()):
                    tasks["oceananigans"] = "WAITING_FOR_MATCHING_512_DATA"
                    write_report(results, runtime, tasks, "WAITING_FOR_OCEANANIGANS_DATA")
                    status("WAITING_FOR_OCEANANIGANS_DATA", available_gpu_counts=coverage)
                    time.sleep(30)
                    continue
                tasks["oceananigans"] = "RUNNING"
                status("RENDERING_OCEANANIGANS")
                render_ocean(root, results, series)
                tasks["oceananigans"] = "COMPLETE"
                outcome = "COMPLETE"
                break
            verify_sources(root, hashes)
            write_report(results, runtime, tasks, outcome)
            if args.publish_results:
                remote = subprocess.check_output(["git", "remote", "get-url", "--push", "origin"], cwd=root, text=True).strip()
                if remote != plan["origin_push_url"]:
                    raise RuntimeError("Push destination changed")
                if subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=root, text=True).strip():
                    raise RuntimeError("Unrelated staged changes prevent publication")
                subprocess.run(["git", "pull", "--ff-only", "origin", "benchmark-results"], cwd=root, check=True)
                for path in results.rglob("*"):
                    if path.is_file() and path.suffix != ".png":
                        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()).rstrip() + "\n")
                subprocess.run(["git", "add", "--", str(results.relative_to(root))], cwd=root, check=True)
                figures = [str(p.relative_to(root)) for pattern in ("*.png", "*.svg") for p in results.glob(pattern)]
                if figures:
                    subprocess.run(["git", "add", "-f", "--", *figures], cwd=root, check=True)
                subprocess.run(["git", "diff", "--cached", "--check"], cwd=root, check=True)
                subprocess.run(["git", "commit", "-m", f"Record conditional efficiency plots: {outcome}"], cwd=root, check=True)
                commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
                push = subprocess.run(["git", "push", "origin", "HEAD:refs/heads/benchmark-results"], cwd=root)
                if push.returncode:
                    status("RESULTS_READY_PUSH_FAILED", results_commit=commit)
                    return 1
            status(outcome)
            return 0
        except Exception as error:
            tasks = {name: "FAILED" if state == "RUNNING" else state for name, state in tasks.items()}
            status("ERROR", error=str(error))
            raise


if __name__ == "__main__":
    sys.exit(main())
