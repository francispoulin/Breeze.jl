"""Render host NVTX ranges, correlated GPU projections and CUDA execution.

Reads an existing Nsight SQLite export; never runs the simulation or profiler.
Requires Matplotlib for PNG/SVG; the generated HTML works offline.
"""

import argparse
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path
import sqlite3


COLORS = {
    "Outer step": "#b197fc", "Acoustic stage": "#4dabf7",
    "Acoustic work": "#69db7c", "Pressure solve": "#ffd43b",
    "Halo fill": "#63c5bc", "State / thermodynamics": "#22b8cf",
    "Tendencies": "#ff8787", "Other / copies": "#adb5bd", "GC": "#f783ac",
}
PREFIXES = ("TimeSteppers/", "CompressibleEquations/", "AtmosphereModels/",
            "AnelasticEquations/", "PotentialTemperatureFormulations/", "call/")


def category(label):
    if label.endswith("/time_step!"):
        return "Outer step"
    if label.endswith("/acoustic_rk3_substep!"):
        return "Acoustic stage"
    if "solve" in label:
        return "Pressure solve"
    if "halo" in label:
        return "Halo fill"
    if "tendenc" in label:
        return "Tendencies"
    if any(word in label for word in ("acoustic", "predictor", "recovery", "damping",
                                      "perturbation", "vertical_rhs", "horizontal_step")):
        return "Acoustic work"
    if any(word in label for word in ("state", "auxiliary", "temperature", "velocities", "density")):
        return "State / thermodynamics"
    return "GC" if label == "GC" else "Other / copies"


def read_timeline(database):
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        strings = dict(connection.execute("SELECT id, value FROM StringIds"))
        ranges = [dict(start=s, end=e, thread=t, name=strings.get(i, text))
                  for s, e, t, i, text in connection.execute(
                      "SELECT start, end, globalTid, textId, text FROM NVTX_EVENTS "
                      "WHERE end IS NOT NULL AND eventType IN (59,60,70,71)")]
        ranges = sorted((r for r in ranges if r["name"] and
                         (r["name"].startswith(PREFIXES) or r["name"] == "GC")),
                        key=lambda r: (r["start"], -r["end"]))
        kernels = [dict(start=s, end=e, correlation=i, name=strings[n],
                        device=d, stream=stream, kind="kernel")
                   for s, e, i, n, d, stream in connection.execute(
                       "SELECT start, end, correlationId, shortName, deviceId, streamId "
                       "FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")]
        tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        copies = []
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
            copies = [dict(start=s, end=e, correlation=i, name=f"CUDA memcpy ({size} bytes)",
                           device=d, stream=stream, kind="copy")
                      for s, e, i, size, d, stream in connection.execute(
                          "SELECT start,end,correlationId,bytes,deviceId,streamId "
                          "FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start")]
        launches = connection.execute(
            "SELECT start,end,globalTid,correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME "
            "WHERE correlationId IN (SELECT correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL) "
            + ("OR correlationId IN (SELECT correlationId FROM CUPTI_ACTIVITY_KIND_MEMCPY) "
               if copies else "") + "ORDER BY start").fetchall()
    outer = [r for r in ranges if r["name"] == "TimeSteppers/time_step!"]
    if not outer or not kernels:
        raise ValueError("Expected Breeze time_step! NVTX ranges and CUDA kernel events")
    if len({r["thread"] for r in outer}) != 1:
        raise ValueError("Render one benchmark rank per SQLite export")
    # Preserve the host nesting for the GPU projection, including async execution
    # after a host range ends. Every GPU operation is linked through its launch API.
    stacks = defaultdict(list)
    for r in ranges:
        stack = stacks[r["thread"]]
        while stack and stack[-1]["end"] <= r["start"]:
            stack.pop()
        if stack and r["end"] > stack[-1]["end"]:
            raise ValueError("Overlapping, non-nested NVTX ranges are not supported")
        r["depth"] = len(stack)
        r["projected_start"], r["projected_end"], r["operations"] = None, None, 0
        stack.append(r)
    operations = defaultdict(list)
    for operation in kernels + copies:
        operations[operation["correlation"]].append(operation)
    active, index = [], 0
    for start, end, thread, correlation in launches:
        while index < len(ranges) and ranges[index]["start"] <= start:
            active.append(ranges[index])
            index += 1
        active = [r for r in active if r["end"] >= start]
        for r in active:
            if r["thread"] != thread or r["end"] < end:
                continue
            for operation in operations[correlation]:
                r["projected_start"] = min(r["projected_start"] or operation["start"], operation["start"])
                r["projected_end"] = max(r["projected_end"] or operation["end"], operation["end"])
                r["operations"] += 1
    origin = outer[0]["start"]
    end = max(max(r["end"] for r in outer), max(o["end"] for o in kernels + copies))
    depth_count = 1 + max(r["depth"] for r in ranges)
    lanes = ([f"Host NVTX · level {d}" for d in range(depth_count)] +
             [f"GPU projection · level {d}" for d in range(depth_count)])
    gpu_lanes = {}
    for operation in kernels + copies:
        key = operation["kind"], operation["device"], operation["stream"]
        if key not in gpu_lanes:
            gpu_lanes[key] = len(lanes)
            kind = "kernels" if key[0] == "kernel" else "copies"
            lanes.append(f"GPU {key[1]} · stream {key[2]} · {kind}")
    events, labels, label_ids = [], [], {}

    def append(start, finish, lane, label, color_name=None):
        if label not in label_ids:
            label_ids[label] = len(labels)
            labels.append(label)
        events.append([round((start - origin) / 1e6, 6), round((finish - origin) / 1e6, 6),
                       lane, label_ids[label], list(COLORS).index(category(color_name or label))])

    for number, r in enumerate(outer, 1):
        r["step"] = number
    for r in ranges:
        label = r["name"] + (f" #{r['step']}" if "step" in r else "")
        append(r["start"], r["end"], r["depth"], label, r["name"])
        if r["projected_start"] is not None:
            append(r["projected_start"], r["projected_end"], depth_count + r["depth"], label, r["name"])
    for operation in kernels + copies:
        key = operation["kind"], operation["device"], operation["stream"]
        append(operation["start"], operation["end"], gpu_lanes[key], operation["name"])
    projections = defaultdict(lambda: dict(instances=0, total_ns=0, gpu_operations=0))
    for r in ranges:
        if r["projected_start"] is not None:
            metric = projections[r["name"]]
            metric["instances"] += 1
            metric["total_ns"] += r["projected_end"] - r["projected_start"]
            metric["gpu_operations"] += r["operations"]
    return dict(lanes=lanes, labels=labels, events=sorted(events), colors=list(COLORS.values()),
                categories=list(COLORS), extent=[0, (end - origin) / 1e6],
                first_step=[(outer[0]["projected_start"] - origin) / 1e6,
                            (outer[0]["projected_end"] - origin) / 1e6],
                steps=len(outer), kernels=len(kernels), copies=len(copies),
                host_ranges=len(ranges), projections=dict(projections))


def write_figure(data, prefix, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "svg.fonttype": "none"})
    background, foreground = "#181c24", "#edf2f7"
    figure, axes = plt.subplots(2, 1, figsize=(18, 12), facecolor=background)
    figure.subplots_adjust(left=0.17, right=0.98, top=0.83, bottom=0.11, hspace=0.36)
    figure.suptitle(title, color=foreground, fontsize=20, x=0.17, ha="left", y=0.97)
    figure.text(0.17, 0.935,
                f"{data['steps']} captured steps · {data['kernels']:,} CUDA kernels · "
                f"{data['extent'][1] / 1000:.3f} s capture window · times after warmup",
                color="#c5cedb", fontsize=11)
    figure.legend(handles=[Patch(color=color, label=name) for name, color in COLORS.items()],
                  loc="upper left", bbox_to_anchor=(0.165, 0.922), ncol=5,
                  frameon=False, labelcolor=foreground, fontsize=9)
    for axis, bounds, heading in zip(axes, [data["extent"], data["first_step"]],
                                     ["Full capture", "First captured step · GPU projection"]):
        axis.set_facecolor(background)
        start, end = bounds
        for lane in range(len(data["lanes"])):
            axis.axhspan(lane - 0.43, lane + 0.43, color="#242b36", zorder=0)
        grouped = defaultdict(list)
        for s, e, lane, label, color in data["events"]:
            if e < start or s > end:
                continue
            left, right = max(start, s), min(end, e)
            grouped[lane, color].append((left, right - left))
            text = data["labels"][label].split("/")[-1]
            width = (right - left) / (end - start)
            if width > max(0.075, len(text) * 0.0045):
                axis.text((left + right) / 2, lane, text, color="#11151c", fontsize=8,
                          ha="center", va="center", clip_on=True, zorder=3)
        for (lane, color), segments in grouped.items():
            axis.broken_barh(segments, (lane - 0.33, 0.66),
                             facecolors=data["colors"][color], linewidth=0, zorder=2)
        axis.set_xlim(start, end)
        axis.set_ylim(len(data["lanes"]) - 0.4, -0.8)
        axis.set_yticks(range(len(data["lanes"])))
        axis.set_yticklabels(data["lanes"], color=foreground)
        axis.tick_params(colors=foreground, length=0, pad=7)
        axis.grid(axis="x", color="#64748b", alpha=0.25)
        axis.set_xlabel("Elapsed time from first captured host step (ms)", color=foreground)
        axis.set_title(heading, color=foreground, loc="left", pad=12, fontsize=12)
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.text(0.17, 0.04,
                "Host NVTX = CPU submission / waiting. GPU projection = first to last correlated GPU operation; includes gaps.\n"
                "GPU kernel / copy rows show measured execution. Parent and child ranges overlap. "
                "CPU can submit later steps while the GPU runs an earlier step.",
                color="#c5cedb", fontsize=10, linespacing=1.7)
    figure.savefig(str(prefix) + ".png", dpi=150, facecolor=background)
    figure.savefig(str(prefix) + ".svg", facecolor=background)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--title", default="Breeze NVTX timeline")
    args = parser.parse_args()
    data = read_timeline(args.database)
    data["title"] = args.title
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_figure(data, prefix, args.title)
    template = Path(__file__).with_name("nvtx_timeline.html").read_text()
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    rendered = template.replace("__TITLE__", html.escape(args.title)).replace("__DATA__", payload)
    Path(str(prefix) + ".html").write_text(rendered)
    metadata = dict(database=str(args.database.resolve()),
                    database_sha256=hashlib.sha256(args.database.read_bytes()).hexdigest(),
                    steps=data["steps"], kernels=data["kernels"], copies=data["copies"],
                    host_ranges=data["host_ranges"], extent_ms=data["extent"],
                    first_step_ms=data["first_step"], projections=data["projections"])
    Path(str(prefix) + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Rendered {data['steps']} steps, {data['kernels']} kernels, {data['copies']} copies: {prefix}")


if __name__ == "__main__":
    main()
