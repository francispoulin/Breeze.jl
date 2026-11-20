"""Tabulate GPU time per time step for every kernel in NVTX/Nsight captures.

Usage: python3 kernel_step_table.py /path/to/results/analysis.json [case ...]

Reads the per-rank kernel totals that analyze_nvtx.py wrote to analysis.json. A kernel's
time per step is the summed duration of all its launches during the captured steps divided
by the number of steps, so several launches of one kernel within a step add up. Launches
with the same short name (for example, one kernel specialized for several fields) are one
row. Writes kernels_per_step.csv and kernels_per_step.md beside analysis.json.
"""

from collections import defaultdict
import csv
import json
from pathlib import Path
import sys


def main():
    analysis_path = Path(sys.argv[1])
    selected = set(sys.argv[2:])
    records = [r for r in json.loads(analysis_path.read_text()) if not selected or r["case"] in selected]
    if not records:
        raise RuntimeError("No matching records in analysis.json")
    cases = defaultdict(dict)
    for record in records:
        cases[record["case"]][record["rank"]] = record

    rows = []
    lines = ["# GPU kernel time per time step", "",
             "Mean over the captured steps of the summed duration of every launch of each kernel;",
             "launches with the same short name are one row. Times are GPU execution in",
             "milliseconds; kernels on one GPU can overlap, so the column sum can exceed the",
             "step's wall time. NCCL kernels include time spent waiting for the peer rank.", ""]
    for case, ranks in cases.items():
        steps = {record["steps"] for record in ranks.values()}
        if len(steps) != 1:
            raise RuntimeError(f"Ranks of {case} captured different numbers of steps")
        steps = steps.pop()
        rank_ids = sorted(ranks)
        names = {name for record in ranks.values() for name in record["kernels"]}
        table = []
        for name in names:
            per_rank = [ranks[rank]["kernels"].get(name, dict(seconds=0.0, calls=0)) for rank in rank_ids]
            calls = [metric["calls"] / steps for metric in per_rank]
            milliseconds = [1e3 * metric["seconds"] / steps for metric in per_rank]
            table.append((name, calls, milliseconds))
            for rank, call, time in zip(rank_ids, calls, milliseconds):
                rows.append(dict(case=case, rank=rank, kernel=name, steps=steps,
                                 launches_per_step=call, milliseconds_per_step=time))
        table.sort(key=lambda entry: -max(entry[2]))
        totals = [1e3 * ranks[rank]["kernel_seconds"] / steps for rank in rank_ids]
        launches = [ranks[rank]["kernel_count"] / steps for rank in rank_ids]
        lines += [f"## {case}", "", f"{len(rank_ids)} rank(s), {steps} captured steps.", ""]
        header = "| Kernel | Launches/step | " + " | ".join(f"Rank {rank} ms/step" for rank in rank_ids) + " |"
        lines += [header, "|" + "---|" * (2 + len(rank_ids))]
        for name, calls, milliseconds in table:
            call_text = " / ".join(sorted({f"{call:g}" for call in calls}))
            lines.append(f"| `{name}` | {call_text} | " + " | ".join(f"{time:.3f}" for time in milliseconds) + " |")
        call_text = " / ".join(sorted({f"{launch:g}" for launch in launches}))
        lines.append(f"| **All kernels** | {call_text} | " + " | ".join(f"**{t:.3f}**" for t in totals) + " |")
        lines.append("")

    with analysis_path.with_name("kernels_per_step.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case", "rank", "kernel", "steps",
                                                    "launches_per_step", "milliseconds_per_step"])
        writer.writeheader()
        writer.writerows(rows)
    analysis_path.with_name("kernels_per_step.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
