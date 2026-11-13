"""Render unprofiled anelastic and three-model Breeze weak-scaling efficiencies."""

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def weak_efficiencies(measurements):
    if {int(row["gpus"]) for row in measurements} != {1, 2, 4} or len(measurements) != 3:
        raise ValueError("Exactly one measurement at each of 1, 2 and 4 GPUs is required")
    baseline = next(float(row["seconds_per_step"]) for row in measurements if int(row["gpus"]) == 1)
    output = []
    for row in sorted(measurements, key=lambda row: int(row["gpus"])):
        ranks = int(row["gpus"])
        if (int(row["Nx"]) * int(row["Ny"]) * int(row["Nz"]) != ranks * 512**3
                or row["fields_finite"] != "true"):
            raise ValueError("Expected finite solutions with exactly 512 cubed cells per GPU")
        seconds = float(row["seconds_per_step"])
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Timings must be positive and finite")
        output.append(dict(row, weak_efficiency_percent=100 * baseline / seconds))
    return output


def draw_curves(curves, title, footer, output):
    colors = ["#0072B2", "#009E73", "#D55E00"]
    with plt.rc_context({"font.size": 11, "axes.titlesize": 15, "svg.hashsalt": "breeze-weak-scaling"}):
        figure, axis = plt.subplots(figsize=(7.8, 5.2))
        maximum = 100
        for (label, rows), color in zip(curves, colors):
            x = [int(row["gpus"]) for row in rows]
            y = [float(row["weak_efficiency_percent"]) for row in rows]
            maximum = max(maximum, *y)
            axis.plot(x, y, marker="o", linewidth=2.2, markersize=7, label=label, color=color)
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 2, 4], labels=["1", "2", "4"])
        axis.set_xlim(0.9, 4.4)
        axis.set_ylim(0, max(110, math.ceil(maximum / 10) * 10 + 10))
        axis.set_xlabel("Number of A100 GPUs")
        axis.set_ylabel("Weak-scaling efficiency (%)")
        axis.set_title(title, pad=28)
        axis.text(0.5, 1.025, "512³ cells per GPU · MPI · each model normalized to its 1-GPU case",
                  transform=axis.transAxes, ha="center", fontsize=9)
        axis.grid(alpha=0.25)
        axis.legend(loc="lower left", frameon=False)
        figure.text(0.5, 0.025, footer, ha="center", va="bottom", fontsize=8.5)
        figure.tight_layout(rect=(0, 0.09, 1, 1))
        for suffix in (".png", ".svg"):
            path = output.with_suffix(suffix)
            figure.savefig(path, dpi=200, metadata={"Creator": "Breeze benchmark report"})
            if suffix == ".svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()).rstrip() + "\n")
        plt.close(figure)


def render(results, anelastic):
    anelastic = weak_efficiencies(anelastic)
    with (results / "weak_scaling.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=anelastic[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(anelastic)
    draw_curves([("Anelastic", anelastic)], "Anelastic weak-scaling efficiency",
                "Partitions: 1×1, 2×1, 4×1 · Unprofiled timings · Efficiency = 100 × T₁ / Tₙ",
                results / "anelastic_weak_efficiency")
    with (results / "provenance/compressible_measurements.csv").open() as stream:
        archived = list(csv.DictReader(stream))
    curves = [("Anelastic", anelastic)]
    for model, label in [("compressible_explicit", "Compressible explicit"),
                         ("compressible_splitexplicit", "Compressible split-explicit")]:
        rows = [row for row in archived if row["dynamics"] == model
                and row["communication"] in ("SERIAL", "MPI")
                and (int(row["gpus"]) < 4 or (row["Rx"], row["Ry"]) == ("2", "2"))]
        curves.append((label, weak_efficiencies(rows)))
    draw_curves(curves, "Breeze weak-scaling efficiency",
                "4 GPUs: anelastic 4×1; compressible models 2×2 (all local grids 512³).\n"
                "Compressible measurements: 9–10 Nov; anelastic: this study. Different source revisions.",
                results / "breeze_weak_efficiency")
    return anelastic
