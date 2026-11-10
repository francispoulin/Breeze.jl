"""Validate finished cases and regenerate tables and plots, including partial studies."""

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STUDY = Path(__file__).resolve().parent.parent
REPO = STUDY.parents[3]
LABELS = {"compressible_explicit": "Explicit", "compressible_splitexplicit": "Split-explicit"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    state = json.loads((STUDY / "study.json").read_text())
    rows = []
    if (STUDY / "measurements.csv").exists():
        with (STUDY / "measurements.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    completed = [case for case in state["cases"] if case["state"] == "COMPLETE"]
    assert len(rows) == len(completed), "Only completed cases may enter analysis"
    keys = set()
    metrics = []
    for case in completed:
        key = (case["dynamics"], case["communication"], case["gpus"], *case["partition"])
        assert key not in keys, f"Duplicate case: {key}"
        keys.add(key)
        matches = [row for row in rows if (row["dynamics"], row["communication"], int(row["gpus"]),
                                          int(row["Rx"]), int(row["Ry"])) == key]
        assert len(matches) == 1
        row = matches[0]
        assert [int(row[name]) for name in ("Nx", "Ny", "Nz")] == case["size"]
        assert math.prod(case["size"]) == 512**3 * case["gpus"]
        assert row["float"] == "Float32" and int(row["steps"]) == 500 and row["machine"] == "rondeau"
        text = (STUDY / case["log"]).read_text()
        assert case["exit_code"] == 0 and case["driver_exit_zero"]
        assert "finished, exit status 0" in text and "Δt = 0.005" in text
        assert "TIMING_START" in text and "TIMING_END" in text and "RESULT machine=" in text
        elapsed = float(re.search(r"TIMING_END communication=\w+ elapsed_seconds=(\S+)", text)[1])
        seconds = float(row["seconds_per_step"])
        assert math.isfinite(seconds) and seconds > 0
        assert math.isclose(elapsed / 500, seconds, rel_tol=5e-7)
        assert math.isclose(math.prod(case["size"]) / seconds, float(row["points_per_second"]), rel_tol=1e-6)
        finite = row["fields_finite"] == "true" and math.isfinite(float(row["max_abs_w"]))
        assert case["fields_finite"] == finite
        workers = case["workers"]
        assert len(workers) == case["gpus"], f"Unexpected worker count: {case['id']}"
        assert len({worker["environment"]["CUDA_VISIBLE_DEVICES"] for worker in workers}) == case["gpus"]
        assert {int(worker["environment"]["CUDA_VISIBLE_DEVICES"]) for worker in workers} == set(case["gpu_ids"])
        assert all(worker["environment"]["JULIA_NUM_THREADS"] == "1" for worker in workers)
        assert all(worker["environment"]["JULIA_CUDA_MEMORY_POOL"] == "none" for worker in workers)
        assert all(any("/opt/uw/openmpi/4.1.7-with-cuda/" in name for name in worker["library_mappings"])
                   for worker in workers)
        if case["communication"] == "NCCL":
            assert "NCCLCommunicator" in text
            assert all(any("libnccl.so" in name for name in worker["library_mappings"]) for worker in workers)
        with (STUDY / "memory_samples.csv").open(newline="") as stream:
            samples = [sample for sample in csv.DictReader(stream) if sample["case"] == case["id"]]
        allowed = {worker["pid"] for worker in workers}
        for sample in samples:
            assert all(int(pid) in allowed for pid in sample["compute_pids"].split(";") if pid)
        assert {gpu: max(int(sample["used_mib"]) for sample in samples if sample["gpu"] == gpu)
                for gpu in case["peak_mib"]} == case["peak_mib"]
        baseline = next((float(reference["seconds_per_step"]) for reference in rows
                         if reference["dynamics"] == case["dynamics"] and int(reference["gpus"]) == 1
                         and reference["fields_finite"] == "true"), None)
        metrics.append({"case": case["id"], "dynamics": case["dynamics"], "communication": case["communication"],
                        "gpus": case["gpus"], "partition": "x".join(map(str, case["partition"])),
                        "seconds_per_step": seconds, "fields_finite": str(finite).lower(),
                        "weak_efficiency_percent": 100 * baseline / seconds if finite and baseline else "",
                        "max_abs_w": row["max_abs_w"], "log": case["log"]})
    hashes = json.loads((STUDY / "provenance/source_sha256.json").read_text())
    assert all(digest(Path(name)) == expected for name, expected in hashes.items())
    with (STUDY / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, lineterminator="\n", fieldnames=("case", "dynamics", "communication", "gpus", "partition",
                                                   "seconds_per_step", "fields_finite", "weak_efficiency_percent",
                                                   "max_abs_w", "log"))
        writer.writeheader()
        writer.writerows(metrics)
    validation = {"validated_utc": datetime.now(timezone.utc).isoformat(), "configured_cases": 14,
                  "completed_cases": len(completed), "valid_performance_cases": sum(case["fields_finite"] for case in completed),
                  "source_and_environment_hashes_unchanged": True,
                  "case_geometry_csv_log_timestep_and_timing_verified": True,
                  "sampled_gpu_placement_memory_and_no_foreign_compute_verified": True,
                  "one_julia_thread_per_rank_and_system_cuda_aware_mpi_verified": True,
                  "loaded_nccl_library_and_communicator_verified": True,
                  "numerical_scope": state["numerical_scope"],
                  "complete_case_coverage": len(completed) == 14}
    (STUDY / "validation/results.json").write_text(json.dumps(validation, indent=2) + "\n")
    figures = []
    for dynamics, label in LABELS.items():
        for partition in ("2x2", "4x1"):
            selected = [row for row in rows if row["dynamics"] == dynamics
                        and (int(row["gpus"]) < 4 or f'{row["Rx"]}x{row["Ry"]}' == partition)]
            if len(selected) != 5 or any(row["fields_finite"] != "true" for row in selected):
                continue
            csv_path = STUDY / f'{dynamics}_{partition}.csv'
            with csv_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=selected[0].keys(), lineterminator="\n")
                writer.writeheader()
                writer.writerows(selected)
            prefix = STUDY / (label.lower().replace("-", "_") + f"_weak_mpi_nccl_{partition}")
            subprocess.run([sys.executable, str(REPO / "benchmarking/distributed/plot_split_explicit_results.py"),
                            str(csv_path), "--dynamics=" + dynamics, "--compare-comm", "--time-step=0.005",
                            "--output-prefix=" + str(prefix)], check=True)
            figures.append((f'{label} weak scaling; four-GPU partition {partition}', prefix.name))
    four_gpu = [item for item in metrics if item["gpus"] == 4]
    if len(four_gpu) == 8 and all(item["fields_finite"] == "true" for item in four_gpu):
        figure, axes = plt.subplots(1, 2, figsize=(11, 5.2))
        for axis, (dynamics, label) in zip(axes, LABELS.items()):
            for offset, backend, color in ((-0.18, "MPI", "#2563eb"), (0.18, "NCCL", "#0891b2")):
                values = [next(item["seconds_per_step"] for item in four_gpu
                               if item["dynamics"] == dynamics and item["communication"] == backend
                               and item["partition"] == partition) for partition in ("2x2", "4x1")]
                bars = axis.bar([offset, 1 + offset], values, width=0.34, label=backend, color=color, zorder=3)
                axis.bar_label(bars, fmt="%.4f", padding=4)
            axis.set_xticks([0, 1], ["2 × 2", "4 × 1"])
            axis.set_title(label)
            axis.set_xlabel("Four-GPU partition (Rx × Ry)")
            axis.set_ylabel("Seconds per main step")
            axis.grid(axis="y", alpha=0.25, zorder=0)
            axis.set_ylim(0, axis.get_ylim()[1] * 1.15)
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(frameon=False)
        figure.suptitle("Rondeau · MPI/NCCL and four-GPU partition comparison", fontsize=14)
        figure.text(0.08, 0.02, "1024 × 1024 × 512 global cells · Float32 · Δt = 0.005 s · 10 warmup + 500 timed steps\n"
                    "12 acoustic substeps for split-explicit · One timing window per case; no error bars.", fontsize=9)
        figure.subplots_adjust(bottom=0.23, top=0.85, wspace=0.30)
        plt.rcParams["svg.fonttype"] = "none"
        for suffix in ("png", "svg"):
            figure.savefig(STUDY / f"four_gpu_partition_comparison.{suffix}", dpi=180,
                           metadata={"Date": None} if suffix == "svg" else {})
        plt.close(figure)
        figures.append(("Four-GPU partition comparison", "four_gpu_partition_comparison"))
    now = datetime.now(ZoneInfo("America/Toronto")).strftime("%Y-%m-%d %H:%M %Z")
    lines = ["# Rondeau MPI/NCCL weak scaling at Δt = 0.005 s", "",
             f'Updated {now}. **{len(completed)}/14 cases complete; {validation["valid_performance_cases"]} with finite fields.**', "",
             "Both compressible dynamics are rerun from the previous 0.02-second study. The only physics/timing change is",
             "Δt = 0.005 s. The added four-GPU cases compare partitions 2×2 and 4×1 on the same global grid.", "",
             "Float32, WENO5, full dry CBL setup, fixed 12 km × 12 km × 3 km domain, 10 warmup steps, and 500 timed main steps",
             "(2.5 simulated seconds) are retained. Split-explicit uses 12 acoustic substeps. Initial perturbations are unseeded.", "",
             "| GPUs | Global grid | Partition | Cells/GPU |", "|---|---|---|---|",
             "| 1 | 512×512×512 | 1×1 | 512³ |", "| 2 | 1024×512×512 | 2×1 | 512³ |",
             "| 4 | 1024×1024×512 | 2×2 or 4×1 | 512³ |", "",
             "The 4×1 layout has local grids 256×1024×512; 2×2 has 512×512×512. Both retain 512³ cells per rank.", "",
             "## Measurements", "", "| Dynamics | Backend | GPUs | Partition | s/step | Weak efficiency | Final fields |",
             "|---|---|---|---|---|---|---|"]
    for item in metrics:
        efficiency = item["weak_efficiency_percent"]
        lines.append(f'| {LABELS[item["dynamics"]]} | {item["communication"]} | {item["gpus"]} | {item["partition"]} | '
                     f'{item["seconds_per_step"]:.6f} | {f"{efficiency:.1f}%" if efficiency != "" else "excluded"} | '
                     f'{"Finite" if item["fields_finite"] == "true" else "Non-finite; excluded"} |')
    lines.extend(["", "Weak efficiency is 100 × serial time / distributed time. The shared one-GPU baseline uses plain GPU().",
                  "Times are the maximum rank elapsed time divided by 500, with CUDA synchronization and MPI barriers at",
                  "window boundaries. Construction/JIT/warmup and field validation are excluded. There is one timing window",
                  "per case on a shared server, without repeats or error bars; differences do not establish statistical significance.", ""])
    if len(four_gpu) == 8:
        lines.extend(["## Four-GPU comparison", "", "| Dynamics | Backend | 2×2 s/step | 4×1 s/step | 2×2 time / 4×1 time |",
                      "|---|---|---|---|---|"])
        for dynamics, label in LABELS.items():
            for backend in ("MPI", "NCCL"):
                pair = {item["partition"]: item for item in four_gpu
                        if item["dynamics"] == dynamics and item["communication"] == backend}
                ratio = (f'{pair["2x2"]["seconds_per_step"] / pair["4x1"]["seconds_per_step"]:.3f}×'
                         if all(item["fields_finite"] == "true" for item in pair.values()) else "excluded")
                lines.append(f'| {label} | {backend} | {pair["2x2"]["seconds_per_step"]:.6f} | '
                             f'{pair["4x1"]["seconds_per_step"]:.6f} | {ratio} |')
        lines.append("")
    for title, prefix in figures:
        lines.extend([f'![{title}]({prefix}.png)', "", f'[{title}: SVG]({prefix}.svg)', ""])
    waiting = [case for case in state["cases"] if case["state"] != "COMPLETE"]
    if waiting:
        lines.extend(["## Pending cases", "", *[f'- {case["id"]}: {case["state"]}' for case in waiting], "",
                      "The controller waits until the required number of devices has no compute processes before launching each case.", ""])
    lines.extend(["## Evidence and reproduction", "",
                  "- [Study status and per-case commands/placement](study.json), [measurements](measurements.csv),",
                  "  [validated metrics](comparison.csv), [validation](validation/results.json), [GPU/process samples](memory_samples.csv).",
                  "- [Exact commands](execution/commands.log), [controller log](execution/sweep.log), [controller](execution/run_series.py).",
                  "- [Automatic report/validation and final commit](execution/report_progress.py); this runs separately from the sweep.",
                  "- [Environment](provenance/environment.log), [manifest](provenance/Manifest.toml),",
                  "  [source hashes](provenance/source_sha256.json), [source archive](provenance/source.tar.gz).", "",
                  "Every prognostic and velocity field is checked for finite interior values on every rank after timing.",
                  "Non-finite cases are excluded from derived metrics and plots. This is a finiteness check; full-model numerical",
                  "agreement with a serial solution has not been tested. MPI still launches NCCL ranks and supplies scalar",
                  "reductions and barriers; NCCLDistributed/NCCLCommunicator handles GPU halo communication.", "",
                  "The resolved benchmark environment, scientific source, driver and platform launcher match the previous study.",
                  "Previous successful two/four-rank NCCL halo probes are retained in the [previous study](../weak_scaling_mpi_nccl_2026-11-08_dt0p02/validation/).",
                  "Worker snapshots verify actual NCCL library mappings, system CUDA-aware MPI, one Julia thread per rank,",
                  "distinct selected devices, and absence of other compute processes in samples taken every ten seconds.",
                  "Memory samples include the CUDA context and are lower bounds on peak device usage. Other host activity is uncontrolled.", "",
                  "To launch a fresh copy of this study, choose a new results directory in the environment/controller first:", "",
                  "```bash", f'source {STUDY}/execution/environment.sh',
                  f'python3 -u {STUDY}/execution/run_series.py > {STUDY}/execution/sweep.log 2>&1', "```", "",
                  "The controller refuses an existing study to prevent duplicate runs. From the repository root, regenerate this report:", "",
                  "```bash", f'python3 {Path(__file__).resolve().relative_to(REPO)}', "```", ""])
    (STUDY / "README.md").write_text("\n".join(lines).rstrip() + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
