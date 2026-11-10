"""Report only validated unprofiled measurements, with shared serial baselines."""
import csv
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def report(study):
    plan = json.loads((study / "plan.json").read_text())
    rows = []
    statuses = []
    for case in plan:
        path = study / "cases" / case["id"] / "status.json"
        status = json.loads(path.read_text()) if path.exists() else {"state": "pending"}
        statuses.append((case, status))
        if status["state"] == "validated":
            rows.append(dict(status["measurement"], case=case["id"]))
    baselines = {row["dynamics"]: float(row["seconds_per_step"]) for row in rows if row["communication"] == "SERIAL"}
    for row in rows:
        baseline = baselines.get(row["dynamics"])
        row["weak_efficiency_percent"] = 100 * baseline / float(row["seconds_per_step"]) if baseline else ""
    if rows:
        with (study / "comparison.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        for column, dynamics in enumerate(("compressible_explicit", "compressible_splitexplicit")):
            for backend in ("MPI", "NCCL"):
                selected = sorted((row for row in rows if row["dynamics"] == dynamics and row["communication"] in ("SERIAL", backend)), key=lambda row: int(row["gpus"]))
                counts = [int(row["gpus"]) for row in selected]
                axes[0, column].plot(counts, [float(row["seconds_per_step"]) for row in selected], "o-", label=backend)
                axes[1, column].plot(counts, [float(row["weak_efficiency_percent"]) if row["weak_efficiency_percent"] != "" else float("nan") for row in selected], "o-", label=backend)
            axes[0, column].set_title(dynamics.replace("compressible_", ""))
            for axis in axes[:, column]:
                axis.set_xscale("log", base=2)
                axis.set_xticks([1, 2, 4, 8, 16], labels=[1, 2, 4, 8, 16])
                axis.set_xlabel("GPUs")
                axis.grid(alpha=0.25)
                axis.legend()
            axes[0, column].set_ylabel("Synchronized seconds / step")
            axes[1, column].set_ylabel("Weak-scaling efficiency (%)")
        figure.savefig(study / "weak_scaling.png", dpi=180)
        figure.savefig(study / "weak_scaling.svg")
        plt.close(figure)
    lines = ["# Rorqual compressible MPI/NCCL weak scaling", "",
             f"**{len(rows)}/18 cases validated.** See [progress](progress.json) and [case plan](plan.json).", "",
             "Full dry CBL, Float32, WENO5, fixed 12 km × 12 km × 3 km domain, Δt = 0.005 s for both dynamics;",
             "10 warmup and 500 timed steps. Split-explicit uses 12 acoustic substeps. No Nsight runs.", "",
             "| GPUs | Global grid | Partition | Backends |", "|---|---|---|---|",
             "| 1 | 512×512×512 | 1×1×1 | Shared SERIAL |",
             "| 2 | 1024×512×512 | 2×1×1 | MPI, NCCL |",
             "| 4 | 1024×1024×512 | 4×1×1 | MPI, NCCL |",
             "| 8 | 4096×512×512 | 8×1×1 | MPI, NCCL |",
             "| 16 | 8192×512×512 | 16×1×1 | MPI, NCCL |", "",
             "One 16-GPU/four-node Slurm allocation runs cases sequentially, one Julia thread per rank,",
             "system CUDA-aware MPI and JULIA_CUDA_MEMORY_POOL=none. Each case has a 30-minute execution cap;",
             "the allocation requests three hours based on previous stepping times plus compilation allowance.", "",
             "Every accepted case requires finite prognostic and velocity fields on every rank, unique GPU placement,",
             "the requested communicator, and positive finite timing. Timing uses the slowest synchronized rank.",
             "Weak efficiency is 100 × serial seconds/step ÷ distributed seconds/step. Finiteness does not",
             "establish agreement with a serial solution. No four-GPU partition comparison was requested in this run.", "",
             "[Source revision](provenance/source_commit.txt), [working-tree patch](provenance/source_changes.patch),",
             "[frozen hashes](provenance/hashes.json), [environment setup](provenance/setup.log),",
             "[manifest](provenance/Manifest.toml), and [source archive](provenance/source.tar.gz).",
             "Per-case directories contain exact commands, environment records, per-rank logs, final device-memory",
             "snapshots and sampled per-process GPU memory (MiB, every five seconds; sampled maxima can miss peaks).", "",
             "| Case | State | s/step | Points/s | Weak efficiency |", "|---|---|---|---|---|"]
    metrics = {row["case"]: row for row in rows}
    for case, status in statuses:
        row = metrics.get(case["id"], {})
        efficiency = row.get("weak_efficiency_percent", "")
        lines.append(f"| {case['id']} | {status['state']} | {row.get('seconds_per_step', '')} | {row.get('points_per_second', '')} | {efficiency} |")
    if rows:
        lines += ["", "![Weak scaling](weak_scaling.png)"]
    (study / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    report(Path(sys.argv[1]).resolve())
