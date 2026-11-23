"""Weak-scaling efficiency figures at 512³ cells per GPU for Breeze and Oceananigans.

Efficiency is 100 × T₁ / Tₙ, where T is seconds per step and each model is
normalized to its own one-GPU case. Ideal weak scaling is 100%.
Figures are written only for series that have a measurement at every GPU count.
"""

import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BREEZE_RESULTS = HERE.parent
# The Oceananigans checkout is /u/ajtse/Oceananigans.jl on Rondeau and ~/Oceananigans.jl on Fir.
OCEANANIGANS_RESULTS = Path(os.environ.get("OCEANANIGANS_RESULTS", next(
    (str(path) for path in (Path("/u/ajtse/Oceananigans.jl/benchmarking/results"),
                            Path.home() / "Oceananigans.jl/benchmarking/results") if path.is_dir()),
    "/u/ajtse/Oceananigans.jl/benchmarking/results")))
# Pure performance rerun (no Nsight, no NVTX, NCCL, median of three 100-step launches) of the Fir
# Oceananigans nonhydrostatic 1- and 2-GPU cases. Once validated it replaces those two points of the
# Fir suite series; the suite's 4-16 GPU points stay normalized to the suite's own 1-GPU time.
FIR_NONHYDROSTATIC_NCCL = BREEZE_RESULTS / "fir/oceananigans_nonhydrostatic_nccl_2026-11-23"
README_MARKERS = ("<!-- fir-nonhydrostatic-nccl -->", "<!-- /fir-nonhydrostatic-nccl -->")
# Oceananigans series and halo/transpose backend per machine. Rondeau uses the NCCL rerun;
# its earlier MPI series (gpu_weak_scaling_512_x, 1-2 GPUs only) stays recorded but is not plotted.
OCEANANIGANS_SERIES = dict(fir=("gpu_weak_scaling_512_x", "MPI"),
                           rondeau=("gpu_weak_scaling_512_x_nccl", "NCCL"))

# Categorical slots 1-3 of the reference palette, assigned in fixed order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

# Cross-code figures: color encodes the code (slot 1 Breeze, slot 2 Oceananigans) and
# line style plus marker encode the model, so every model keeps one look in every panel.
BREEZE_COLOR, OCEANANIGANS_COLOR = COLORS[:2]
MODEL_STYLES = {
    "Anelastic": dict(code="Breeze", color=BREEZE_COLOR, linestyle="-", marker="o"),
    "Compressible explicit": dict(code="Breeze", color=BREEZE_COLOR, linestyle=(0, (5, 2)), marker="s"),
    "Compressible split-explicit": dict(code="Breeze", color=BREEZE_COLOR, linestyle=(0, (1, 1.5)), marker="^"),
    "Hydrostatic": dict(code="Oceananigans", color=OCEANANIGANS_COLOR, linestyle="-", marker="D"),
    "Nonhydrostatic": dict(code="Oceananigans", color=OCEANANIGANS_COLOR, linestyle=(0, (5, 2)), marker="o"),
}
SHORT_LABELS = {"Compressible explicit": "CE", "Compressible split-explicit": "CSE",
                "Hydrostatic": "Hydrostatic", "Nonhydrostatic": "Nonhydrostatic", "Anelastic": "Anelastic"}
CELLS_PER_GPU = 512**3


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def breeze_point(row):
    cells = int(row["Nx"]) * int(row["Ny"]) * int(row["Nz"])
    if cells != int(row["gpus"]) * CELLS_PER_GPU or row["fields_finite"] != "true":
        raise ValueError(f"Expected finite fields and 512³ cells per GPU: {row}")
    return dict(gpus=int(row["gpus"]), seconds_per_step=float(row["seconds_per_step"]),
                partition=f"{row['Rx']}×{row['Ry']}", grid=f"{row['Nx']}×{row['Ny']}×{row['Nz']}")


def oceananigans_point(row):
    Nx, Ny, Nz = map(int, row["resolution"].split("x"))
    if row["state"] != "COMPLETED" or Nx * Ny * Nz != int(row["ranks"]) * CELLS_PER_GPU:
        raise ValueError(f"Expected a completed case with 512³ cells per GPU: {row}")
    return dict(gpus=int(row["ranks"]), seconds_per_step=float(row["median"]),
                partition=f"{row['ranks']}×1", grid=f"{Nx}×{Ny}×{Nz}")


def with_efficiency(points, gpu_counts):
    points = sorted(points, key=lambda point: point["gpus"])
    if [point["gpus"] for point in points] != list(gpu_counts):
        raise ValueError(f"Expected GPU counts {gpu_counts}, found {[p['gpus'] for p in points]}")
    if any(not math.isfinite(p["seconds_per_step"]) or p["seconds_per_step"] <= 0 for p in points):
        raise ValueError("Timings must be positive and finite")
    baseline = points[0]["seconds_per_step"]
    return [dict(point, efficiency_percent=100 * baseline / point["seconds_per_step"]) for point in points]


def breeze_fir():
    folder = BREEZE_RESULTS / "fir/weak_scaling_mpi_2026-11-12_nx1/cases"
    curves = []
    for dynamics, label in [("anelastic", "Anelastic"), ("compressible_explicit", "Compressible explicit"),
                            ("compressible_splitexplicit", "Compressible split-explicit")]:
        rows = [row for path in sorted(folder.glob(f"{dynamics}_*/measurement.csv"))
                for row in read_csv(path) if row["dynamics"] == dynamics]
        curves.append((label, with_efficiency(map(breeze_point, rows), (1, 2, 4, 8, 16))))
    return curves


def breeze_rondeau():
    anelastic = read_csv(BREEZE_RESULTS / "rondeau/anelastic_weak_scaling_512_2026-11-13/weak_scaling.csv")
    archived = read_csv(BREEZE_RESULTS / "rondeau/anelastic_weak_scaling_512_2026-11-13/provenance/compressible_measurements.csv")
    curves = [("Anelastic", with_efficiency(map(breeze_point, anelastic), (1, 2, 4)))]
    for dynamics, label in [("compressible_explicit", "Compressible explicit"),
                            ("compressible_splitexplicit", "Compressible split-explicit")]:
        # The four-GPU compressible case uses 2×2, keeping each local grid 512³.
        rows = [row for row in archived if row["dynamics"] == dynamics
                and row["communication"] in ("SERIAL", "MPI")
                and (int(row["gpus"]) < 4 or (row["Rx"], row["Ry"]) == ("2", "2"))]
        curves.append((label, with_efficiency(map(breeze_point, rows), (1, 2, 4))))
    return curves


def fir_nonhydrostatic_nccl():
    """Validated median timings at 1 and 2 GPUs from the Fir NCCL rerun, or None while pending."""
    status = FIR_NONHYDROSTATIC_NCCL / "final_status.json"
    if not status.is_file() or json.loads(status.read_text())["state"] != "validated":
        return None
    rows = [row for row in read_csv(FIR_NONHYDROSTATIC_NCCL / "comparison.csv")
            if row["dynamics"] == "oceananigans_nonhydrostatic"]
    points = with_efficiency(map(breeze_point, rows), (1, 2))
    for point, row in zip(points, sorted(rows, key=lambda row: int(row["gpus"]))):
        point.update(seconds_per_step_min=float(row["seconds_per_step_min"]),
                     seconds_per_step_max=float(row["seconds_per_step_max"]), source="NCCL rerun")
    return points


def with_fir_nonhydrostatic_nccl(points):
    """Replace the suite's 1- and 2-GPU points with the NCCL rerun when it is available."""
    rerun = fir_nonhydrostatic_nccl()
    if rerun is None:
        return points
    replaced = {point["gpus"]: point for point in rerun}
    return [replaced.get(point["gpus"], dict(point, source="MPI suite")) for point in points]


def update_readme(points):
    """Rewrite the README block that tabulates the Fir nonhydrostatic rerun."""
    path = HERE / "README.md"
    text = path.read_text()
    start, end = README_MARKERS
    if start not in text:
        return
    suite = {int(row["ranks"]): float(row["median"]) for row in read_csv(
        OCEANANIGANS_RESULTS / "non_hydrostatic_models/fir/baseline" / OCEANANIGANS_SERIES["fir"][0] / "results.csv")}
    if points is None:
        body = "Status: **PENDING**; the figures still use the MPI suite at every GPU count."
    else:
        lines = ["Status: **COMPLETE**; the Fir figures use these points at 1 and 2 GPUs.", "",
                 "| GPUs | NCCL rerun median s/step (min–max) | Efficiency | MPI suite median s/step | Suite efficiency |",
                 "|---:|---|---:|---:|---:|"]
        for point in points:
            gpus = point["gpus"]
            lines.append(f"| {gpus} | {point['seconds_per_step']:.4f} ({point['seconds_per_step_min']:.4f}–"
                         f"{point['seconds_per_step_max']:.4f}) | {point['efficiency_percent']:.1f}% | "
                         f"{suite[gpus]:.4f} | {100 * suite[1] / suite[gpus]:.1f}% |")
        body = "\n".join(lines)
    pattern = re.compile(re.escape(start) + ".*?" + re.escape(end), re.DOTALL)
    path.write_text(pattern.sub(lambda _: f"{start}\n\n{body}\n\n{end}", text))


def oceananigans(machine, gpu_counts, partial=False):
    """Hydrostatic and nonhydrostatic curves, or None if either is missing a GPU count.

    With `partial=True`, a series measured at a leading subset of `gpu_counts` (at least
    two counts) is accepted, so a cross-code figure can show it before the study finishes.
    """
    curves = []
    for folder, label in [("hydrostatic_free_surface_models", "Hydrostatic"),
                          ("non_hydrostatic_models", "Nonhydrostatic")]:
        series, _ = OCEANANIGANS_SERIES[machine]
        path = OCEANANIGANS_RESULTS / folder / machine / "baseline" / series / "results.csv"
        if not path.is_file():
            return None
        rows = [row for row in read_csv(path) if row["state"] == "COMPLETED"]
        measured = sorted(int(row["ranks"]) for row in rows)
        expected = list(gpu_counts[:len(measured)]) if partial and len(measured) >= 2 else list(gpu_counts)
        if measured != expected:
            return None
        points = with_efficiency(map(oceananigans_point, rows), expected)
        if machine == "fir" and label == "Nonhydrostatic":
            points = with_fir_nonhydrostatic_nccl(points)
        curves.append((label, points))
    return curves


def draw(curves, gpu_counts, gpu_name, title, subtitle, footer, stem):
    with plt.rc_context({"font.size": 11, "svg.hashsalt": "breeze-weak-scaling", "svg.fonttype": "none"}):
        figure, axis = plt.subplots(figsize=(8, 5.2))
        axis.axhline(100, color="#8a8984", linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        axis.text(gpu_counts[0], 101.5, "ideal", color="#52514e", fontsize=9, va="bottom")
        ends = []
        for (label, points), color in zip(curves, COLORS):
            x = [point["gpus"] for point in points]
            y = [point["efficiency_percent"] for point in points]
            axis.plot(x, y, color=color, linewidth=2, marker="o", markersize=7,
                      markeredgecolor="white", markeredgewidth=1.5, label=label, zorder=3)
            ends.append([y[-1], y[-1], x[-1]])
        # Spread end labels at least 5 percentage points apart so close curves stay legible.
        ends.sort()
        for lower, upper in zip(ends, ends[1:]):
            upper[1] = max(upper[1], lower[1] + 5)
        for value, position, gpus in ends:
            axis.annotate(f"{value:.0f}%", (gpus, value), xytext=(gpus * 1.06, position),
                          textcoords="data", va="center", fontsize=9.5, color="#0b0b0b")
        axis.set_xscale("log", base=2)
        axis.set_xticks(gpu_counts, labels=[str(n) for n in gpu_counts])
        axis.minorticks_off()
        axis.set_xlim(gpu_counts[0] / 1.15, gpu_counts[-1] * 1.35)
        axis.set_ylim(0, 115)
        axis.set_xlabel(f"Number of {gpu_name} GPUs")
        axis.set_ylabel("Weak-scaling efficiency (%)")
        axis.set_title(title, loc="left", fontsize=14, pad=26)
        axis.text(0, 1.03, subtitle, transform=axis.transAxes, fontsize=9.5, color="#52514e")
        axis.grid(axis="y", color="#e4e3df", linewidth=0.8)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#8a8984")
        axis.legend(loc="lower left", frameon=False)
        figure.text(0.01, 0.015, footer, ha="left", va="bottom", fontsize=8.5, color="#52514e")
        figure.tight_layout(rect=(0, 0.04 * (footer.count("\n") + 1) + 0.03, 1, 1))
        for suffix in (".png", ".svg"):
            path = HERE / f"{stem}{suffix}"
            figure.savefig(path, dpi=200, metadata={"Creator": "Breeze benchmark report"} if suffix == ".png" else {"Creator": "Breeze benchmark report", "Date": None})
            if suffix == ".svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()).rstrip() + "\n")
        plt.close(figure)


def style_axis(axis, gpu_counts, gpu_name, ylabel=True):
    axis.axhline(100, color="#8a8984", linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    axis.text(gpu_counts[0], 101.5, "ideal", color="#52514e", fontsize=9, va="bottom")
    axis.set_xscale("log", base=2)
    axis.set_xticks(gpu_counts, labels=[str(n) for n in gpu_counts])
    axis.minorticks_off()
    axis.set_xlim(gpu_counts[0] / 1.15, gpu_counts[-1] * 2.6)
    axis.set_ylim(0, 115)
    axis.set_xlabel(f"Number of {gpu_name} GPUs")
    if ylabel:
        axis.set_ylabel("Weak-scaling efficiency (%)")
    axis.grid(axis="y", color="#e4e3df", linewidth=0.8)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#8a8984")


def draw_models(axis, curves):
    """Plot curves in the code-by-color, model-by-style scheme with direct end labels."""
    ends = []
    for label, points in curves:
        style = MODEL_STYLES[label]
        x = [point["gpus"] for point in points]
        y = [point["efficiency_percent"] for point in points]
        axis.plot(x, y, color=style["color"], linestyle=style["linestyle"], linewidth=2,
                  marker=style["marker"], markersize=6.5, markeredgecolor="white", markeredgewidth=1.2,
                  label=f"{style['code']} {label.lower()}",
                  zorder=3)
        ends.append([y[-1], y[-1], x[-1], label])
    # Spread end labels that share a GPU count at least 5.5 percentage points apart
    # so close curves stay legible; labels at different counts do not collide.
    ends.sort(key=lambda end: (end[2], end[0]))
    for lower, upper in zip(ends, ends[1:]):
        if lower[2] == upper[2]:
            upper[1] = max(upper[1], lower[1] + 5.5)
    for value, position, gpus, label in ends:
        axis.annotate(f"{SHORT_LABELS[label]} {value:.0f}%", (gpus, value), xytext=(gpus * 1.08, position),
                      textcoords="data", va="center", fontsize=8.5, color=MODEL_STYLES[label]["color"],
                      fontweight="bold", zorder=4,
                      bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.9))


def draw_comparison(panels, gpu_counts, gpu_name, title, subtitle, footer, stem):
    """Draw one or more side-by-side panels sharing the efficiency axis."""
    width = 10 if len(panels) == 1 else 13
    with plt.rc_context({"font.size": 11, "svg.hashsalt": "breeze-weak-scaling", "svg.fonttype": "none"}):
        figure, axes = plt.subplots(1, len(panels), figsize=(width, 5.6), sharey=True, squeeze=False)
        for index, (axis, (panel_title, curves)) in enumerate(zip(axes[0], panels)):
            style_axis(axis, gpu_counts, gpu_name, ylabel=index == 0)
            draw_models(axis, curves)
            if panel_title:
                axis.set_title(panel_title, loc="left", fontsize=11.5, pad=8)
            axis.legend(loc="lower left", frameon=False, fontsize=9, handlelength=3)
        figure.text(0.01, 0.985, title, ha="left", va="top", fontsize=14)
        figure.text(0.01, 0.935, subtitle, ha="left", va="top", fontsize=9.5, color="#52514e")
        figure.text(0.01, 0.015, footer, ha="left", va="bottom", fontsize=8.5, color="#52514e")
        figure.tight_layout(rect=(0, 0.035 * (footer.count("\n") + 1), 1, 0.86))
        save(figure, stem)


def save(figure, stem):
    for suffix in (".png", ".svg"):
        path = HERE / f"{stem}{suffix}"
        figure.savefig(path, dpi=200, metadata={"Creator": "Breeze benchmark report"} if suffix == ".png" else {"Creator": "Breeze benchmark report", "Date": None})
        if suffix == ".svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()).rstrip() + "\n")
    plt.close(figure)


COMPARISON_NOTES = dict(
    fir=dict(partitions="N×1 partitions", communication="MPI",
             breeze="Breeze: Float32 WENO5 dry CBL, Δt halved per GPU doubling, 500 timed steps."),
    rondeau=dict(partitions="N×1 partitions except Breeze compressible 2×2 at 4 GPUs",
                 communication="Breeze MPI, Oceananigans NCCL",
                 breeze="Breeze: Float32 WENO5 dry CBL, Δt = 0.005 s; anelastic measured 13 Nov, "
                        "compressible 9–10 Nov (different source revisions)."),
)


def rerun_note(machine):
    if machine != "fir" or fir_nonhydrostatic_nccl() is None:
        return ""
    return ("\nNonhydrostatic 1–2 GPUs: NCCL rerun without Nsight or NVTX, median of three 100-step launches;"
            "\nnonhydrostatic 4–16 GPUs: MPI suite, normalized to the suite's own 1-GPU time.")


def comparison_figures(machine, breeze_curves, oceananigans_curves, gpu_counts, gpu_name):
    """Three cross-code figure sets for one machine; GPU types are never mixed in one figure."""
    curves = dict(breeze_curves + oceananigans_curves)
    pick = lambda *labels: [(label, curves[label]) for label in labels]
    name = machine.capitalize()
    notes = COMPARISON_NOTES[machine]
    measured = max(point["gpus"] for _, points in oceananigans_curves for point in points)
    pending = (f"\nOceananigans {gpu_counts[-1]}-GPU cases not yet run; its lines stop at {measured} GPUs."
               if measured < gpu_counts[-1] else "")
    subtitle = (f"512³ cells per GPU · {notes['communication']} · {notes['partitions']} · "
                "each model normalized to its own 1-GPU time\n"
                "Blue = Breeze, orange = Oceananigans · CE = compressible explicit, CSE = compressible split-explicit")
    footer = (notes["breeze"] + "\n"
              f"Oceananigans ({OCEANANIGANS_SERIES[machine][1]}): Float64 rectilinear buoyancy, WENO5, Δt = 10 s, "
              "median of five 10-step windows." + rerun_note(machine) + "\n"
              "Workloads differ, so compare scaling shapes, not absolute speed." + pending)
    return [
        (f"by_code_weak_efficiency_{machine}", [
            ("Oceananigans", pick("Hydrostatic", "Nonhydrostatic")),
            ("Breeze", pick("Anelastic", "Compressible explicit", "Compressible split-explicit"))],
         f"Weak scaling on {name}, by code", subtitle, footer),
        (f"by_solver_weak_efficiency_{machine}", [
            ("No global pressure solve",
             pick("Hydrostatic", "Compressible explicit", "Compressible split-explicit")),
            ("Global elliptic pressure solve",
             pick("Nonhydrostatic", "Anelastic"))],
         f"Weak scaling on {name}, by pressure treatment", subtitle, footer),
        (f"all_models_weak_efficiency_{machine}", [
            (None, pick("Hydrostatic", "Compressible explicit", "Compressible split-explicit",
                        "Nonhydrostatic", "Anelastic"))],
         f"Weak scaling on {name}, all five models", subtitle, footer),
    ]


def main():
    figures = []
    subtitle = "512³ cells per GPU · {} · each model normalized to its own 1-GPU time"
    figures.append(("breeze_weak_efficiency_fir", breeze_fir(), (1, 2, 4, 8, 16), "H100",
                    "Breeze weak scaling on Fir",
                    "Partitions N×1 · Float32, WENO5 dry CBL · Δt halved per GPU doubling · 500 timed steps", "MPI"))
    figures.append(("breeze_weak_efficiency_rondeau", breeze_rondeau(), (1, 2, 4), "A100",
                    "Breeze weak scaling on Rondeau",
                    "Float32, WENO5 dry CBL · Δt = 0.005 s · 4 GPUs: anelastic 4×1, compressible 2×2\n"
                    "Anelastic measured 13 Nov; compressible 9–10 Nov (different source revisions)", "MPI"))
    comparisons = []
    breeze_curves = dict(fir=figures[0][1], rondeau=figures[1][1])
    for machine, counts, gpu in [("fir", (1, 2, 4, 8, 16), "H100"), ("rondeau", (1, 2, 4), "A100")]:
        partial_curves = oceananigans(machine, counts, partial=True)
        if partial_curves is not None:
            comparisons += [(*comparison, counts, gpu) for comparison in
                            comparison_figures(machine, breeze_curves[machine], partial_curves, counts, gpu)]
        curves = oceananigans(machine, counts)
        if curves is None:
            print(f"Oceananigans {machine}: incomplete, standalone figure skipped", file=sys.stderr)
            continue
        figures.append((f"oceananigans_weak_efficiency_{machine}", curves, counts, gpu,
                        f"Oceananigans weak scaling on {machine.capitalize()}",
                        "Partitions N×1 · Float64 rectilinear buoyancy, WENO5 · Δt = 10 s · median of five 10-step windows"
                        + rerun_note(machine),
                        OCEANANIGANS_SERIES[machine][1]))
    rows = []
    for stem, curves, counts, gpu, title, footer, communication in figures:
        draw(curves, counts, gpu, title, subtitle.format(communication), footer, stem)
        rows += [dict(figure=stem, model=label, **point) for label, points in curves for point in points]
    for stem, panels, title, subtitle, footer, counts, gpu in comparisons:
        draw_comparison(panels, counts, gpu, title, subtitle, footer, stem)
    with (HERE / "weak_efficiency.csv").open("w") as stream:
        # Rerun points carry a timing range and source; other rows leave those columns empty.
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames, restval="", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    update_readme(fir_nonhydrostatic_nccl())
    for stem, *_ in figures + comparisons:
        print(stem)


if __name__ == "__main__":
    main()
