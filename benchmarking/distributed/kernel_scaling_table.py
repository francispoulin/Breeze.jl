"""Compare GPU kernel time per time step between two NVTX/Nsight cases, such as 1 and 2 GPUs.

Usage: python3 kernel_scaling_table.py ANALYSIS[@CASE+CASE][,...] OUTPUT_PREFIX [--note=TEXT] TITLE=BASE_CASE:SCALED_CASE ...

Reads the per-rank kernel totals that analyze_nvtx.py wrote to analysis.json. Several
comma-separated analysis.json files put comparisons from different campaigns in one table file;
case names must be unique across them, and FILE@CASE+CASE keeps only the listed cases of a file. Time per step is
the summed duration of all launches of a kernel divided by the captured steps. For a case with
several ranks each kernel uses its slowest rank, as weak-scaling efficiency does. Columns:
difference = scaled - base (ms and % of base) and weak-scaling efficiency = 100 × base / scaled.
--note=TEXT is printed under the headings of the comparisons that follow it. When both cases use
the same number of GPUs, the columns are labeled by case. Writes OUTPUT_PREFIX.md and OUTPUT_PREFIX.csv.
"""

import csv
import json
from pathlib import Path
import sys


def per_step(records):
    """Slowest-rank milliseconds and launches per step for every kernel of one case."""
    steps = {record["steps"] for record in records}
    if len(steps) != 1:
        raise RuntimeError("Ranks captured different numbers of steps")
    steps = steps.pop()
    kernels = {}
    for record in records:
        for name, metric in record["kernels"].items():
            time, calls = kernels.get(name, (0.0, 0.0))
            kernels[name] = (max(time, 1e3 * metric["seconds"] / steps), max(calls, metric["calls"] / steps))
    total = max(1e3 * record["kernel_seconds"] / steps for record in records)
    launches = max(record["kernel_count"] / steps for record in records)
    return kernels, (total, launches), len(records)


def gpus(count):
    return f"{count} GPU" + ("s" if count > 1 else "")


def number(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def signed(value, digits=3, suffix=""):
    return "—" if value is None else f"{value:+.{digits}f}{suffix}"


def main():
    analysis = []
    for source in sys.argv[1].split(","):
        path, _, selection = source.partition("@")  # path@case+case keeps only those cases
        records = json.loads(Path(path).read_text())
        analysis += [r for r in records if not selection or r["case"] in selection.split("+")]
    keys = [(record["case"], record["rank"]) for record in analysis]
    duplicates = sorted({key[0] for key in keys if keys.count(key) > 1})
    if duplicates:
        raise RuntimeError(f"Cases appear in more than one analysis file: {duplicates}; select with path@case+case")
    prefix = Path(sys.argv[2])
    by_case = {}
    for record in analysis:
        by_case.setdefault(record["case"], []).append(record)

    lines = ["# Kernel time per time step: scaling comparison", "",
             "GPU execution time per time step, summed over all launches of each kernel and averaged",
             "over the captured steps. Multi-rank cases use each kernel's slowest rank.",
             "Difference = scaled − base. Efficiency = 100 × base / scaled (weak scaling: 100% means",
             "the kernel takes as long with the same work per GPU). A dash marks a kernel that only",
             "one case launches. Rows are sorted by Difference ms, greatest first (a missing kernel counts as 0 ms).", ""]
    rows = []
    note = None
    for argument in sys.argv[3:]:
        if argument.startswith("--note="):  # text printed under the following comparisons' headings
            note = argument.removeprefix("--note=")
            continue
        title, cases = argument.split("=", 1)
        base_case, scaled_case = cases.split(":")
        base, (base_total, base_launches), base_ranks = per_step(by_case[base_case])
        scaled, (scaled_total, scaled_launches), scaled_ranks = per_step(by_case[scaled_case])
        if base_ranks == scaled_ranks:  # same GPU count: label the columns by case
            base_label, scaled_label, launches_label = f"`{base_case}`", f"`{scaled_case}`", "base→compared"
        else:
            base_label, scaled_label, launches_label = f"{base_ranks} GPU", f"{scaled_ranks} GPU", f"{base_ranks}→{scaled_ranks} GPU"
        lines += [f"## {title}", "", f"Base `{base_case}` ({gpus(base_ranks)}), compared `{scaled_case}` ({gpus(scaled_ranks)}).", "",
                  *([note, ""] if note else []),
                  f"| Kernel | Launches/step {launches_label} | {base_label} ms/step | "
                  f"{scaled_label} ms/step | Difference ms | Difference % | Efficiency % |",
                  "|---|---|---|---|---|---|---|"]
        # Greatest difference first; a kernel only one case launches counts as 0 ms in the other.
        names = sorted(set(base) | set(scaled),
                       key=lambda n: (base.get(n, (0.0,))[0] - scaled.get(n, (0.0,))[0], -scaled.get(n, (0.0,))[0], n))
        entries = [(name, base.get(name), scaled.get(name)) for name in names]
        entries.append(("**All kernels**", (base_total, base_launches), (scaled_total, scaled_launches)))
        for name, one, two in entries:
            t1 = one[0] if one else None
            t2 = two[0] if two else None
            difference = t2 - t1 if one and two else None
            percentage = 100 * difference / t1 if difference is not None and t1 else None
            efficiency = 100 * t1 / t2 if one and two and t2 else None
            launches = f"{one[1]:g} → {two[1]:g}" if one and two else (f"{one[1]:g} → 0" if one else f"0 → {two[1]:g}")
            label = name if name.startswith("**") else f"`{name}`"
            lines.append(f"| {label} | {launches} | {number(t1)} | {number(t2)} | {signed(difference)} | "
                         f"{signed(percentage, 1, '%')} | {number(efficiency, 1)} |")
            rows.append(dict(comparison=title, kernel=name.strip("*"), base_case=base_case, scaled_case=scaled_case,
                             base_launches_per_step=one[1] if one else 0, scaled_launches_per_step=two[1] if two else 0,
                             base_milliseconds_per_step=t1, scaled_milliseconds_per_step=t2,
                             difference_milliseconds=difference, difference_percent=percentage,
                             efficiency_percent=efficiency))
        lines.append("")

    prefix.with_suffix(".md").write_text("\n".join(lines).rstrip() + "\n")
    with prefix.with_suffix(".csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
