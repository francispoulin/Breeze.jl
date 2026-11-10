"""Plot explicit or split-explicit timing and efficiency from a benchmark CSV."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def read_cases(path, dynamics="compressible_splitexplicit", communication=None):
    with path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["dynamics"] == dynamics]
    backends = {row.get("communication", "MPI") for row in rows if int(row["gpus"]) > 1}
    if communication is None and len(backends) > 1:
        raise ValueError("CSV contains multiple backends; use --comm or --compare-comm")
    if communication is not None:
        rows = [row for row in rows if int(row["gpus"]) == 1
                or row.get("communication", "MPI") == communication]
    if not rows:
        raise ValueError(f"No {dynamics} results in the CSV")
    for row in rows:
        for name in ("Nx", "Ny", "Nz", "Rx", "Ry", "gpus", "steps"):
            row[name] = int(row[name])
        row["seconds_per_step"] = float(row["seconds_per_step"])
        if row.get("fields_finite", "true") != "true" or not math.isfinite(float(row["max_abs_w"])):
            raise ValueError("Cannot plot a case with non-finite fields or vertical velocity")
        if not math.isfinite(row["seconds_per_step"]) or row["seconds_per_step"] <= 0:
            raise ValueError("Timing must be finite and positive")
        row["grid"] = (row["Nx"], row["Ny"], row["Nz"])
        row["cells"] = math.prod(row["grid"])
    baselines = [row for row in rows if row["gpus"] == 1]
    if len(baselines) != 1:
        raise ValueError("Expected one single-GPU reference case")
    baseline = baselines[0]
    strong = sorted([row for row in rows if row["grid"] == baseline["grid"]],
                    key=lambda row: row["gpus"])
    weak = [baseline] + sorted(
        [row for row in rows if row["grid"] != baseline["grid"]],
        key=lambda row: row["gpus"],
    )
    if any(row["cells"] != baseline["cells"] * row["gpus"] for row in weak):
        raise ValueError("Weak scaling must keep cells per GPU constant")
    if len({(row["machine"], row["float"], row["steps"]) for row in rows}) != 1:
        raise ValueError("Cases must share machine, precision, and timed step count")
    return baseline, strong, weak


def panel(axis, rows, values, labels, title, ylabel, ideal, efficiency=False):
    palette = {1: "#718096", 2: "#2563eb", 4: "#0891b2"}
    positions = list(range(len(rows)))
    bars = axis.bar(positions, values, width=0.66,
                    color=[palette.get(row["gpus"], "#7c3aed") for row in rows],
                    zorder=3)
    axis.plot(positions, ideal, "--o", color="#374151", linewidth=1.5,
              markersize=4, markerfacecolor="white", label="Ideal scaling", zorder=4)
    for bar, value in zip(bars, values):
        text = f"{value:.1f}%" if efficiency else f"{value:.2f}"
        axis.annotate(text, (bar.get_x() + bar.get_width() / 2, value),
                      xytext=(0, 5), textcoords="offset points", ha="center",
                      fontsize=9, color="#111827")
    axis.set_xticks(positions, labels)
    axis.tick_params(axis="x", length=0, pad=8)
    axis.set_title(title, loc="left", fontsize=12, fontweight="semibold", pad=12)
    axis.set_ylabel(ylabel)
    axis.set_ylim(0, 122 if efficiency else 1.28 * max(values + ideal))
    if efficiency:
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axis.grid(axis="y", color="#e5e7eb", linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#cbd5e1")
    axis.legend(loc="upper right", frameon=False, fontsize=9)


def save_figure(figure, prefix):
    for suffix in ("png", "svg"):
        path = prefix.with_suffix("." + suffix)
        figure.savefig(path, dpi=180, facecolor="white",
                       metadata={"Date": None} if suffix == "svg" else {})
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())
                            .rstrip("\n") + "\n")
        print(path)
    plt.close(figure)


def plot_communication_comparison(args):
    curves = {backend: read_cases(args.csv, args.dynamics, backend)[2]
              for backend in ("MPI", "NCCL")}
    if [row["gpus"] for row in curves["MPI"]] != [row["gpus"] for row in curves["NCCL"]]:
        raise ValueError("MPI and NCCL must cover the same GPU counts")
    baseline = curves["MPI"][0]
    reference = baseline["seconds_per_step"]
    dynamics_label = {"compressible_explicit": "explicit",
                      "compressible_splitexplicit": "split-explicit"}[args.dynamics]
    prefix = args.output_prefix or args.csv.parent / f"{dynamics_label.replace('-', '_')}_weak_mpi_nccl"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "svg.hashsalt": dynamics_label + "-mpi-nccl"})
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    positions = list(range(len(curves["MPI"])))
    labels = [f'{row["gpus"]} GPU' + ("s" if row["gpus"] > 1 else "")
              + "\n" + "×".join(map(str, row["grid"])) for row in curves["MPI"]]
    for backend, color, marker in (("MPI", "#2563eb", "o"),
                                   ("NCCL", "#0891b2", "s")):
        rows = curves[backend]
        times = [1000 * row["seconds_per_step"] for row in rows]
        efficiencies = [100 * reference / row["seconds_per_step"] for row in rows]
        for axis, values, efficiency in ((axes[0], times, False), (axes[1], efficiencies, True)):
            axis.plot(positions, values, marker=marker, color=color, linewidth=2,
                      markersize=6, label=backend, zorder=3)
            for index, value in enumerate(values):
                if index == 0 and backend == "NCCL":
                    continue
                label = f"{value:.1f}%" if efficiency else f"{value:.2f}"
                other_backend = "NCCL" if backend == "MPI" else "MPI"
                other_time = curves[other_backend][index]["seconds_per_step"]
                other_value = 100 * reference / other_time if efficiency else 1000 * other_time
                above = value > other_value or (value == other_value and backend == "MPI")
                offset = 9 if above else -17
                axis.annotate(label, (index, value), xytext=(0, offset),
                              textcoords="offset points", ha="center", fontsize=9, color=color,
                              bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1})
    axes[0].axhline(1000 * reference, color="#374151", linestyle="--", label="Ideal weak scaling")
    axes[1].axhline(100, color="#374151", linestyle="--", label="Ideal weak scaling")
    axes[0].set_title("Weak scaling · mean time per step", loc="left", fontweight="semibold")
    axes[1].set_title("Weak scaling efficiency · t(1) / t(N)", loc="left", fontweight="semibold")
    axes[0].set_ylabel("Time per step (ms)")
    axes[1].set_ylabel("Efficiency")
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axes[1].set_ylim(0, 122)
    axes[0].set_ylim(0, 1.25 * max(1000 * row["seconds_per_step"] for rows in curves.values() for row in rows))
    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.set_xlim(-0.22, len(positions) - 1 + 0.22)
        axis.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="lower left", frameon=False, fontsize=9)
    figure.suptitle(f'{baseline["machine"].capitalize()} · {dynamics_label} CBL · MPI vs NCCL',
                   fontsize=18, fontweight="semibold", x=0.07, ha="left", y=0.99)
    figure.text(0.07, 0.91, f'{baseline["cells"]:,} cells/GPU · {baseline["float"]} · shared one-GPU serial reference',
                color="#475569")
    details = (f'{args.warmup_steps} warmup + {baseline["steps"]} timed steps · Δt = {args.time_step:g} s'
               + (f' · {args.substeps} acoustic substeps' if args.dynamics == "compressible_splitexplicit" else ""))
    figure.text(0.07, 0.045, details + "\nOne timing window per case on a shared server; no error bars.",
                color="#475569", linespacing=1.6)
    figure.subplots_adjust(left=0.07, right=0.985, top=0.81, bottom=0.22, wspace=0.24)
    save_figure(figure, prefix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--dynamics", default="compressible_splitexplicit",
                        choices=("compressible_explicit", "compressible_splitexplicit"))
    parser.add_argument("--comm", choices=("MPI", "NCCL"))
    parser.add_argument("--compare-comm", action="store_true", help="Compare MPI and NCCL weak scaling")
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--time-step", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--substeps", type=int, default=12)
    args = parser.parse_args()
    if args.compare_comm:
        if args.comm:
            parser.error("--comm and --compare-comm are mutually exclusive")
        plot_communication_comparison(args)
        return
    baseline, strong, weak = read_cases(args.csv, args.dynamics, args.comm)
    reference = baseline["seconds_per_step"]
    dynamics_label = {"compressible_explicit": "explicit",
                      "compressible_splitexplicit": "split-explicit"}[args.dynamics]
    prefix = args.output_prefix or args.csv.parent / f"{dynamics_label.replace('-', '_')}_scaling"
    prefix.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "svg.hashsalt": dynamics_label})
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    strong_labels = [f'{row["gpus"]} GPU' + ("s" if row["gpus"] > 1 else "")
                     + f'\n{row["Rx"]}×{row["Ry"]}' for row in strong]
    weak_labels = [f'{row["gpus"]} GPU' + ("s" if row["gpus"] > 1 else "")
                   + "\n" + "×".join(map(str, row["grid"])) for row in weak]
    strong_time = [1000 * row["seconds_per_step"] for row in strong]
    weak_time = [1000 * row["seconds_per_step"] for row in weak]
    strong_efficiency = [100 * reference / (row["gpus"] * row["seconds_per_step"])
                         for row in strong]
    weak_efficiency = [100 * reference / row["seconds_per_step"] for row in weak]
    grid_label = "×".join(map(str, baseline["grid"]))
    panel(axes[0, 0], strong, strong_time, strong_labels,
          f"Strong scaling · fixed {grid_label}", "Mean time per step (ms)",
          [1000 * reference / row["gpus"] for row in strong])
    panel(axes[0, 1], strong, strong_efficiency, strong_labels,
          "Strong scaling efficiency · t(1) / [N × t(N)]", "Efficiency",
          [100] * len(strong), efficiency=True)
    panel(axes[1, 0], weak, weak_time, weak_labels,
          f'Weak scaling · {baseline["cells"]:,} cells/GPU', "Mean time per step (ms)",
          [1000 * reference] * len(weak))
    panel(axes[1, 1], weak, weak_efficiency, weak_labels,
          "Weak scaling efficiency · t(1) / t(N)", "Efficiency",
          [100] * len(weak), efficiency=True)

    figure.suptitle(f'{baseline["machine"].capitalize()} · {dynamics_label} CBL · {baseline["float"]}',
                   fontsize=19, fontweight="semibold", x=0.07, ha="left", y=0.985)
    figure.text(0.07, 0.941,
                "All recorded partitions; partition labels show Rx × Ry. Dashed lines reference the one-GPU timing.",
                fontsize=10, color="#475569")
    step_details = (f'Mean per main step · {args.warmup_steps} warmup + {baseline["steps"]} timed steps'
                    f' · Δt = {args.time_step:g} s')
    if args.dynamics == "compressible_splitexplicit":
        step_details += f" · {args.substeps} acoustic substeps"
    figure.text(0.07, 0.045, step_details + "\n"
                "One timing window per case on a shared server; no error bars.",
                fontsize=10, color="#475569", linespacing=1.6)
    figure.subplots_adjust(left=0.07, right=0.985, top=0.865, bottom=0.16,
                           hspace=0.57, wspace=0.22)
    save_figure(figure, prefix)


if __name__ == "__main__":
    main()
