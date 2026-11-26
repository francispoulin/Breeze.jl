# Weak-scaling efficiency: Trillium NCCL (this study) against the Fir MPI N×1 study.
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

here = Path(__file__).resolve().parent
fir = here.parents[1] / "fir" / "weak_scaling_mpi_2026-11-12_nx1" / "comparison.csv"

def efficiency(path):
    table = {}
    with open(path) as io:
        for row in csv.DictReader(io):
            table.setdefault(row["dynamics"], {})[int(row["gpus"])] = float(row["weak_efficiency_percent"])
    return table

trillium, fir_mpi = efficiency(here / "comparison.csv"), efficiency(fir)
series = {"anelastic": ("Anelastic", "#257de0"), "compressible_explicit": ("Compressible explicit", "#f47230")}
text, muted, grid = "#1f1f1f", "#6b6b6b", "#e4e4e4"

figure, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
axis.axvspan(5.6, 23, color="#f4f4f4", zorder=0)
axis.text(6.2, 4, "multi-node (4 GPUs per node)", color=muted, fontsize=9)

for dynamics, (label, color) in series.items():
    for table, style, source in ((trillium, "-", "Trillium NCCL"), (fir_mpi, "--", "Fir MPI")):
        gpus = sorted(table[dynamics])
        values = [table[dynamics][g] for g in gpus]
        axis.plot(gpus, values, style, color=color, linewidth=2, marker="o", markersize=7,
                  markeredgecolor="white", markeredgewidth=1.5, label=f"{label}, {source}", zorder=3)
        axis.annotate(f"{values[-1]:.0f}%", (gpus[-1], values[-1]), xytext=(8, 0),
                      textcoords="offset points", va="center", fontsize=9, color=text)

axis.set_xscale("log", base=2)
axis.set_xticks([1, 2, 4, 8, 16], labels=["1", "2", "4", "8", "16"])
axis.set_xlim(0.8, 23)
axis.set_ylim(0, 105)
axis.set_xlabel("GPUs (512³ cells per GPU, N×1 partition)", color=text)
axis.set_ylabel("Weak-scaling efficiency (%)", color=text)
axis.set_title("Weak-scaling efficiency on H100: Trillium NCCL vs Fir MPI", color=text, loc="left")
axis.grid(axis="y", color=grid, linewidth=0.8)
axis.set_axisbelow(True)
for side in ("top", "right"):
    axis.spines[side].set_visible(False)
for side in ("left", "bottom"):
    axis.spines[side].set_color(muted)
axis.tick_params(colors=muted)
handles = [Line2D([], [], color=color, linestyle=style, linewidth=2, label=f"{label}, {source}")
           for label, color in series.values() for style, source in (("-", "Trillium NCCL"), ("--", "Fir MPI"))]
axis.legend(handles=handles, frameon=False, fontsize=9, handlelength=3, loc="lower left", bbox_to_anchor=(0, 0.08))

figure.savefig(here / "weak_scaling_efficiency.png", dpi=180)
figure.savefig(here / "weak_scaling_efficiency.svg")
