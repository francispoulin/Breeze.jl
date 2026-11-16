"""Weak-scaling efficiency figures at 512³ cells per GPU for Breeze and Oceananigans.

Efficiency is 100 × T₁ / Tₙ, where T is seconds per step and each model is
normalized to its own one-GPU case. Ideal weak scaling is 100%.
Figures are written only for series that have a measurement at every GPU count.
"""

import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BREEZE_RESULTS = HERE.parent
OCEANANIGANS_RESULTS = Path("/u/ajtse/Oceananigans.jl/benchmarking/results")

# Categorical slots 1-3 of the reference palette, assigned in fixed order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
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


def oceananigans(machine, gpu_counts):
    curves = []
    for folder, label in [("hydrostatic_free_surface_models", "Hydrostatic"),
                          ("non_hydrostatic_models", "Nonhydrostatic")]:
        path = OCEANANIGANS_RESULTS / folder / machine / "baseline/gpu_weak_scaling_512_x/results.csv"
        if not path.is_file():
            return None
        rows = [row for row in read_csv(path) if row["state"] == "COMPLETED"]
        if sorted(int(row["ranks"]) for row in rows) != list(gpu_counts):
            return None
        curves.append((label, with_efficiency(map(oceananigans_point, rows), gpu_counts)))
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
        figure.tight_layout(rect=(0, 0.08, 1, 1))
        for suffix in (".png", ".svg"):
            path = HERE / f"{stem}{suffix}"
            figure.savefig(path, dpi=200, metadata={"Creator": "Breeze benchmark report"} if suffix == ".png" else {"Creator": "Breeze benchmark report", "Date": None})
            if suffix == ".svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()).rstrip() + "\n")
        plt.close(figure)


def main():
    figures = []
    subtitle = "512³ cells per GPU · MPI · each model normalized to its own 1-GPU time"
    figures.append(("breeze_weak_efficiency_fir", breeze_fir(), (1, 2, 4, 8, 16), "H100",
                    "Breeze weak scaling on Fir",
                    "Partitions N×1 · Float32, WENO5 dry CBL · Δt halved per GPU doubling · 500 timed steps"))
    figures.append(("breeze_weak_efficiency_rondeau", breeze_rondeau(), (1, 2, 4), "A100",
                    "Breeze weak scaling on Rondeau",
                    "Float32, WENO5 dry CBL · Δt = 0.005 s · 4 GPUs: anelastic 4×1, compressible 2×2\n"
                    "Anelastic measured 13 Nov; compressible 9–10 Nov (different source revisions)"))
    for machine, counts, gpu in [("fir", (1, 2, 4, 8, 16), "H100"), ("rondeau", (1, 2, 4), "A100")]:
        curves = oceananigans(machine, counts)
        if curves is None:
            print(f"Oceananigans {machine}: incomplete, figure skipped", file=sys.stderr)
            continue
        figures.append((f"oceananigans_weak_efficiency_{machine}", curves, counts, gpu,
                        f"Oceananigans weak scaling on {machine.capitalize()}",
                        "Partitions N×1 · Float64 rectilinear buoyancy, WENO5 · Δt = 10 s · median of five 10-step windows"))
    rows = []
    for stem, curves, counts, gpu, title, footer in figures:
        draw(curves, counts, gpu, title, subtitle, footer, stem)
        rows += [dict(figure=stem, model=label, **point) for label, points in curves for point in points]
    with (HERE / "weak_efficiency.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for stem, *_ in figures:
        print(stem)


if __name__ == "__main__":
    main()
