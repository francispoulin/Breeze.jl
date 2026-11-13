"""Report validated Fir measurements, including anelastic, without inventing pending results."""
import csv
import json
from pathlib import Path
import sys
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {"anelastic": "Anelastic", "compressible_explicit": "CE", "compressible_splitexplicit": "CSE"}


def read_json(path):
    return json.loads(path.read_text())


def report(study):
    plan = read_json(study / "plan.json")
    statuses, rows = [], []
    for case in plan:
        path = study / "cases" / case["id"] / "status.json"
        status = read_json(path) if path.exists() else {"state": "pending"}
        statuses.append((case, status))
        if status["state"] == "validated":
            rows.append(dict(status["measurement"], case=case["id"], time_step=case["time_step"]))
    baselines = {row["dynamics"]: float(row["seconds_per_step"]) for row in rows if row["communication"] == "SERIAL"}
    for row in rows:
        baseline = baselines.get(row["dynamics"])
        row["weak_efficiency_percent"] = 100 * baseline / float(row["seconds_per_step"]) if baseline else ""
    if rows:
        with (study / "comparison.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        metrics = ("seconds_per_step", "weak_efficiency_percent", "points_per_second_per_gpu")
        for dynamics, label in LABELS.items():
            selected = sorted((row for row in rows if row["dynamics"] == dynamics), key=lambda row: int(row["gpus"]))
            if not selected:
                continue
            for axis, metric in zip(axes, metrics):
                axis.plot([int(row["gpus"]) for row in selected],
                          [float(row[metric]) if row[metric] != "" else float("nan") for row in selected], "o-", label=label)
        for axis in axes:
            axis.set_xscale("log", base=2)
            axis.set_xticks([1, 2, 4, 8, 16], labels=[1, 2, 4, 8, 16])
            axis.set_xlabel("Active GPUs (N×1 partition)")
            axis.grid(alpha=0.25)
            axis.legend()
        axes[0].set_ylabel("Synchronized seconds / step")
        axes[1].set_ylabel("Weak-scaling efficiency (%)")
        axes[2].set_ylabel("Grid points / second / GPU")
        axes[1].axhline(100, color="gray", linewidth=0.8, linestyle="--")
        figure.suptitle("Fir H100 · MPI · 512³ cells per GPU · Float32")
        figure.savefig(study / "weak_scaling.png", dpi=180)
        figure.savefig(study / "weak_scaling.svg")
        plt.close(figure)
    else:
        for name in ("comparison.csv", "weak_scaling.png", "weak_scaling.svg"):
            (study / name).unlink(missing_ok=True)
    final = read_json(study / "final_status.json") if (study / "final_status.json").exists() else None
    state = final["state"].upper() if final else ("SUBMITTED" if (study / "submission.json").exists() else "PREPARED")
    lines = ["# Fir MPI weak scaling: anelastic, CE, CSE", "",
             f"Status: **{state}; {len(rows)}/{len(plan)} cases validated.** Pending, failed, and unrun cases are excluded from performance metrics.", "",
             "Full dry convective boundary layer, WENO5, Float32, deterministic smooth initial perturbations.",
             "Fixed 12 km × 12 km × 3 km domain; N×1 partitions and 512³ cells per GPU.",
             "All three dynamics halve Δt with each doubling of GPU count. CE is compressible explicit; CSE is compressible split-explicit.",
             "10 untimed warmup steps and 500 timed steps per case, one timed window; CSE uses 12 acoustic substeps.",
             "No Nsight or NVTX profiling. One Julia/BLAS/OpenMP thread per rank; CUDA memory pool disabled.", "",
             "| GPUs | Global grid | Partition | Backend | Δt (s) |", "|---:|---|---|---|---:|"]
    unique = {case["gpus"]: case for case in plan}
    for count, case in sorted(unique.items()):
        lines.append(f"| {count} | {case['size'].replace('x', '×')} | {case['partition'].replace('x', '×')} | {'SERIAL' if count == 1 else 'MPI'} | {case['time_step']} |")
    lines += ["", "GPU counts and per-GPU throughput refer to active MPI ranks, with one selected GPU per rank.",
              "The initial series used five independent Slurm allocations. Additional recovery reservations are documented below.",
              "Each runs anelastic, CE, and CSE sequentially on the same allocated GPUs, with eight CPU cores per rank and one Julia thread.",
              "Each allocation has a three-hour limit; each case has a 50-minute cap, including startup and compilation.",
              "An after-any CPU finalizer retains results and plots even if a GPU allocation fails."]
    if (study / "submission.json").exists():
        submission = read_json(study / "submission.json")
        lines += ["", "| GPUs | Slurm job |", "|---:|---|"]
        lines += [f"| {item['gpus']} | {item['job_id']} |" for item in submission["jobs"]]
        lines += ["", f"Finalizer job: {submission.get('finalizer_job_id', 'not yet submitted')}. Exact submission commands are in [submission.json](submission.json) and [allocations/](allocations/)."]
        if submission.get("retry_jobs"):
            lines += ["", "Recovery allocations: " + ", ".join(f"{item['gpus']} GPUs: job {item['job_id']}" for item in submission["retry_jobs"]) + ". See [recovery notes](recovery.md)."]
            for item in submission["retry_jobs"]:
                reserved = item.get("reserved_gpus_per_node", min(4, item["gpus"]))
                if reserved > min(4, item["gpus"]):
                    lines += ["", f"Job {item['job_id']} reserves {reserved} GPUs on one node and uses {item['gpus']} active GPUs; the unused GPUs do not contribute to the measured grid or throughput."]
    lines += ["", "| Dynamics | GPUs | State | s/step | Points/s/GPU | Weak efficiency (%) | Evidence / error |",
              "|---|---:|---|---:|---:|---:|---|"]
    metrics = {row["case"]: row for row in rows}
    for case, status in statuses:
        row = metrics.get(case["id"], {})
        detail = status.get("error", "[case files](cases/" + case["id"] + "/)").replace("|", "/").replace("\n", " ")
        lines.append(f"| {LABELS[case['dynamics']]} | {case['gpus']} | {status['state']} | {row.get('seconds_per_step', '')} | {row.get('points_per_second_per_gpu', '')} | {row.get('weak_efficiency_percent', '')} | {detail} |")
    lines += ["", "Timing includes final GPU synchronization and an MPI barrier and uses the slowest rank.",
              "Weak efficiency is 100 × serial seconds/step ÷ distributed seconds/step, using a separate serial baseline for each dynamics.",
              "The fixed physical domain becomes finer in x as GPU count increases. These are step-throughput measurements at constant cells per GPU; simulated durations decrease with Δt.",
              "Every accepted measurement requires the planned grid/dynamics/precision, distinct H100 GPU placement, the expected MPI communicator, positive finite timing, and finite prognostic and velocity fields on every rank.",
              "Finite fields do not establish full numerical agreement with serial solutions. One timed window supplies no repeat-to-repeat uncertainty estimate.", "",
              "[Source commit](provenance/source_commit.txt), [source archive](provenance/source.tar.gz), [frozen hashes](provenance/hashes.json),",
              "[Julia environment](provenance/Manifest.toml), [MPI preferences](provenance/LocalPreferences.toml), [setup log](provenance/setup.log), and [case plan](plan.json).",
              "Per-case directories retain exact commands, every rank's log and field validation, hardware details, final device-memory use, and five-second process-memory samples (which can miss peaks)."]
    if (study / "provenance/Manifest.toml").exists():
        manifest = tomllib.loads((study / "provenance/Manifest.toml").read_text())
        versions = ", ".join(f"{name} {manifest['deps'][name][0]['version']}" for name in ("Breeze", "Oceananigans", "CUDA", "MPI") if name in manifest["deps"])
        lines += ["", f"Julia {manifest['julia_version']}; {versions}."]
    if rows:
        lines += ["", "![MPI weak scaling](weak_scaling.png)", "", "[Comparison CSV](comparison.csv) · [Vector SVG](weak_scaling.svg)"]
    (study / "README.md").write_text("\n".join(lines) + "\n")
    progress = dict(state=final["state"] if final else state.lower(), expected=len(plan), validated=len(rows),
                    cases={case["id"]: status["state"] for case, status in statuses})
    (study / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")


if __name__ == "__main__":
    report(Path(sys.argv[1]).resolve())
