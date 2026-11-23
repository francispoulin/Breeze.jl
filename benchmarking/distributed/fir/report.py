"""Report validated Fir measurements without inventing pending results."""
import csv
import json
from pathlib import Path
import statistics
import sys
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {"anelastic": "Anelastic", "compressible_explicit": "CE", "compressible_splitexplicit": "CSE",
          "oceananigans_nonhydrostatic": "Oceananigans nonhydrostatic"}

# Workload description per series; the default reproduces the original Breeze MPI report.
DESCRIPTIONS = {
    "breeze_mpi_nx1": [
        "Full dry convective boundary layer, WENO5, Float32, deterministic smooth initial perturbations.",
        "Fixed 12 km × 12 km × 3 km domain; N×1 partitions and 512³ cells per GPU.",
        "All three dynamics halve Δt with each doubling of GPU count. CE is compressible explicit; CSE is compressible split-explicit.",
        "10 untimed warmup steps and 500 timed steps per case, one timed window; CSE uses 12 acoustic substeps.",
        "No Nsight or NVTX profiling. One Julia/BLAS/OpenMP thread per rank; CUDA memory pool disabled."],
    "oceananigans_nonhydrostatic_nccl": [
        "Oceananigans `NonhydrostaticModel` from the suite `rectilinear_buoyancy` workload, run by",
        "`distributed_cbl_benchmark.jl --dynamics=oceananigans_nonhydrostatic`: Float64 RectilinearGrid, periodic x/y,",
        "WENO5, one buoyancy tracer, RungeKutta3, distributed FFT pressure solver, Δt = 10 s, N×1 partitions, 512³ cells per GPU.",
        "Two GPUs exchange halos and transpose the FFT solve with NCCL; MPI only launches ranks. One GPU has no inter-rank communication.",
        "Each launch takes 10 untimed warmup steps and 100 timed steps; every case is launched three times.",
        "**Pure performance benchmark: no Nsight capture and no NVTX instrumentation** (`--profile=false --nvtx=false`).",
        "One Julia/BLAS/OpenMP thread per rank; CUDA memory pool disabled."],
}
TITLES = {"breeze_mpi_nx1": "Fir MPI weak scaling: anelastic, CE, CSE",
          "oceananigans_nonhydrostatic_nccl": "Fir NCCL weak scaling: Oceananigans nonhydrostatic, 1 and 2 GPUs"}


def read_json(path):
    return json.loads(path.read_text())


def median_rows(rows):
    """One row per (dynamics, GPU count) with the median seconds per step over repeats and its range."""
    groups = {}
    for row in rows:
        groups.setdefault((row["dynamics"], int(row["gpus"])), []).append(row)
    summary = []
    for (dynamics, gpus), group in sorted(groups.items()):
        times = sorted(float(row["seconds_per_step"]) for row in group)
        median = statistics.median(times)
        summary.append(dict(group[0], case=",".join(row["case"] for row in group), repeats=len(group),
                            seconds_per_step=median, seconds_per_step_min=times[0], seconds_per_step_max=times[-1],
                            points_per_second=float(group[0]["points_per_second"]) * float(group[0]["seconds_per_step"]) / median,
                            points_per_second_per_gpu=float(group[0]["points_per_second_per_gpu"]) * float(group[0]["seconds_per_step"]) / median))
    return summary


def report(study):
    plan = read_json(study / "plan.json")
    config = read_json(study / "config.json") if (study / "config.json").exists() else {}
    series = config.get("series", "breeze_mpi_nx1")
    repeated = config.get("repeats", 1) > 1
    statuses, rows = [], []
    for case in plan:
        path = study / "cases" / case["id"] / "status.json"
        status = read_json(path) if path.exists() else {"state": "pending"}
        statuses.append((case, status))
        if status["state"] == "validated":
            rows.append(dict(status["measurement"], case=case["id"], time_step=case["time_step"]))
    if repeated:
        rows = median_rows(rows)
    baselines = {row["dynamics"]: float(row["seconds_per_step"]) for row in rows if row["communication"] == "SERIAL"}
    for row in rows:
        baseline = baselines.get(row["dynamics"])
        row["weak_efficiency_percent"] = 100 * baseline / float(row["seconds_per_step"]) if baseline else ""
    gpu_counts = sorted({case["gpus"] for case in plan})
    backends = sorted({"SERIAL" if case["gpus"] == 1 else case["backend"] for case in plan})
    float_types = sorted({case.get("float_type", "Float32") for case in plan})
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
            axis.set_xticks(gpu_counts, labels=gpu_counts)
            axis.set_xlabel("Active GPUs (N×1 partition)")
            axis.grid(alpha=0.25)
            axis.legend()
        axes[0].set_ylabel("Synchronized seconds / step" + (" (median)" if repeated else ""))
        axes[1].set_ylabel("Weak-scaling efficiency (%)")
        axes[2].set_ylabel("Grid points / second / GPU")
        axes[1].axhline(100, color="gray", linewidth=0.8, linestyle="--")
        backend = "/".join(name for name in backends if name != "SERIAL")
        figure.suptitle(f"Fir H100 · {backend} · 512³ cells per GPU · {'/'.join(float_types)} · no Nsight or NVTX")
        figure.savefig(study / "weak_scaling.png", dpi=180)
        figure.savefig(study / "weak_scaling.svg")
        plt.close(figure)
    else:
        for name in ("comparison.csv", "weak_scaling.png", "weak_scaling.svg"):
            (study / name).unlink(missing_ok=True)
    final = read_json(study / "final_status.json") if (study / "final_status.json").exists() else None
    state = final["state"].upper() if final else ("SUBMITTED" if (study / "submission.json").exists() else "PREPARED")
    validated = sum(status["state"] == "validated" for _, status in statuses)
    lines = ["# " + TITLES[series], "",
             f"Status: **{state}; {validated}/{len(plan)} cases validated.** Pending, failed, and unrun cases are excluded from performance metrics.", "",
             *DESCRIPTIONS[series], "",
             "| GPUs | Global grid | Partition | Backend | Δt (s) |", "|---:|---|---|---|---:|"]
    unique = {case["gpus"]: case for case in plan}
    for count, case in sorted(unique.items()):
        lines.append(f"| {count} | {case['size'].replace('x', '×')} | {case['partition'].replace('x', '×')} | {'SERIAL' if count == 1 else case['backend']} | {case['time_step']} |")
    lines += ["", "GPU counts and per-GPU throughput refer to active MPI ranks, with one selected GPU per rank."]
    if series == "breeze_mpi_nx1":
        lines += ["The initial series used five independent Slurm allocations. Additional recovery reservations are documented below.",
                  "Each runs anelastic, CE, and CSE sequentially on the same allocated GPUs, with eight CPU cores per rank and one Julia thread."]
    else:
        lines += [f"One Slurm allocation per GPU count runs its cases sequentially, with eight CPU cores per rank and one Julia thread.",
                  "The two-GPU allocation reserves all four H100s of one node so both ranks see every device (GPU peer copies);",
                  "only two ranks and two GPUs run the benchmark, and the unused GPUs are excluded from throughput and efficiency."]
    lines += ["Each allocation has a three-hour limit; each case has a 50-minute cap, including startup and compilation.",
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
    if repeated and rows:
        lines += ["", "| Dynamics | GPUs | Repeats | Median s/step | Min–max s/step | Points/s/GPU | Weak efficiency (%) |",
                  "|---|---:|---:|---:|---|---:|---:|"]
        for row in rows:
            efficiency = row["weak_efficiency_percent"]
            efficiency = f"{efficiency:.1f}" if efficiency != "" else ""
            lines.append(f"| {LABELS[row['dynamics']]} | {row['gpus']} | {row['repeats']} | {row['seconds_per_step']:.4f} | "
                         f"{row['seconds_per_step_min']:.4f}–{row['seconds_per_step_max']:.4f} | {row['points_per_second_per_gpu']:.4g} | "
                         f"{efficiency} |")
    lines += ["", "| Case | GPUs | State | s/step | Evidence / error |" if repeated else
              "| Dynamics | GPUs | State | s/step | Points/s/GPU | Weak efficiency (%) | Evidence / error |",
              "|---|---:|---|---:|---|" if repeated else "|---|---:|---|---:|---:|---:|---|"]
    metrics = {row["case"]: row for row in rows}
    for case, status in statuses:
        row = metrics.get(case["id"], {})
        detail = status.get("error", "[case files](cases/" + case["id"] + "/)").replace("|", "/").replace("\n", " ")
        if repeated:
            seconds = status.get("measurement", {}).get("seconds_per_step", "") if status["state"] == "validated" else ""
            lines.append(f"| {case['id']} | {case['gpus']} | {status['state']} | {seconds} | {detail} |")
        else:
            lines.append(f"| {LABELS[case['dynamics']]} | {case['gpus']} | {status['state']} | {row.get('seconds_per_step', '')} | {row.get('points_per_second_per_gpu', '')} | {row.get('weak_efficiency_percent', '')} | {detail} |")
    lines += ["", "Timing includes final GPU synchronization and an MPI barrier and uses the slowest rank.",
              "Weak efficiency is 100 × serial seconds/step ÷ distributed seconds/step, using a separate serial baseline for each dynamics"
              + (" and the median over repeats." if repeated else "."),
              "The fixed physical domain becomes finer in x as GPU count increases. These are step-throughput measurements at constant cells per GPU; simulated durations decrease with Δt.",
              "Every accepted measurement requires the planned grid/dynamics/precision, distinct H100 GPU placement, the expected MPI communicator, positive finite timing, and finite prognostic and velocity fields on every rank.",
              "Finite fields do not establish full numerical agreement with serial solutions."
              + (" The min–max range spans the repeated launches." if repeated else " One timed window supplies no repeat-to-repeat uncertainty estimate."), "",
              "[Source commit](provenance/source_commit.txt), [source archive](provenance/source.tar.gz), [frozen hashes](provenance/hashes.json),",
              "[Julia environment](provenance/Manifest.toml), [MPI preferences](provenance/LocalPreferences.toml), [setup log](provenance/setup.log), and [case plan](plan.json).",
              "Per-case directories retain exact commands, every rank's log and field validation, hardware details, final device-memory use, and five-second process-memory samples (which can miss peaks)."]
    if (study / "provenance/Manifest.toml").exists():
        manifest = tomllib.loads((study / "provenance/Manifest.toml").read_text())
        versions = ", ".join(f"{name} {manifest['deps'][name][0]['version']}" for name in ("Breeze", "Oceananigans", "CUDA", "MPI") if name in manifest["deps"])
        lines += ["", f"Julia {manifest['julia_version']}; {versions}."]
    if rows:
        lines += ["", "![Weak scaling](weak_scaling.png)", "", "[Comparison CSV](comparison.csv) · [Vector SVG](weak_scaling.svg)"]
    (study / "README.md").write_text("\n".join(lines) + "\n")
    progress = dict(state=final["state"] if final else state.lower(), expected=len(plan), validated=validated,
                    cases={case["id"]: status["state"] for case, status in statuses})
    (study / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")


if __name__ == "__main__":
    report(Path(sys.argv[1]).resolve())
