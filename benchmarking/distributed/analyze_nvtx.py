"""Attribute GPU kernels to the innermost enclosing Breeze host-method NVTX range.

Usage: python3 analyze_nvtx.py /path/to/results/captures.json
Writes analysis.json, kernels.csv, phases.csv, rank_summary.csv beside the manifest.
GPU attribution follows CUDA launch correlation IDs; it does not intersect CPU
ranges with GPU timestamps. Phase kernel totals are exclusive. Host ranges and
the acoustic-substep inclusive metric overlap and must not be summed.
"""

from collections import defaultdict
import csv
import json
from pathlib import Path
import sqlite3
import sys


MODULES = ("AtmosphereModels/", "CompressibleEquations/", "TimeSteppers/",
           "AnelasticEquations/", "PotentialTemperatureFormulations/")


def union_duration(intervals):
    total = 0
    previous_end = -1
    for start, end in sorted(intervals):
        total += max(0, end - max(start, previous_end))
        previous_end = max(previous_end, end)
    return total


def analyze(database, steps):
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        strings = dict(connection.execute("SELECT id, value FROM StringIds"))
        ranges = [(start, end, tid, strings.get(text_id, text))
                  for start, end, tid, text_id, text in connection.execute(
                      "SELECT start, end, globalTid, textId, text FROM NVTX_EVENTS "
                      "WHERE end IS NOT NULL AND eventType IN (59, 60, 70, 71)")]
        ranges = sorted(r for r in ranges if r[3] and
                        (r[3].startswith(MODULES) or r[3].startswith("call/")))
        if not ranges:
            raise RuntimeError(f"No Breeze ranges found: {database}")
        # MPI progress can issue millions of cuEventQuery calls. Only launch APIs
        # with kernel correlation IDs are needed for attribution; aggregate all API
        # durations in SQL instead of materializing every polling event in Python.
        runtime = connection.execute("SELECT start, end, globalTid, correlationId, nameId "
                                     "FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId IN "
                                     "(SELECT correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL) ORDER BY start").fetchall()
        api_rows = connection.execute("SELECT nameId, sum(end-start), count(*) "
                                      "FROM CUPTI_ACTIVITY_KIND_RUNTIME GROUP BY nameId").fetchall()
        kernels = connection.execute("SELECT start, end, correlationId, shortName, "
                                     "registersPerThread, blockX * blockY * blockZ "
                                     "FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        copies = []
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
            copies = connection.execute("SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
        mpi = None
        if "MPI_P2P_EVENTS" in tables:
            mpi = connection.execute("SELECT start, end FROM MPI_P2P_EVENTS").fetchall()
    correlation = {}
    active = []
    index = 0
    api_totals = {strings[name_id]: [duration, count] for name_id, duration, count in api_rows}
    for start, end, tid, identifier, name_id in runtime:
        while index < len(ranges) and ranges[index][0] <= start:
            active.append(ranges[index])
            index += 1
        active = [r for r in active if r[1] >= start]
        enclosing = [r for r in active if r[2] == tid and r[1] >= end]
        functions = [r for r in enclosing if r[3].startswith(MODULES)]
        innermost = min(functions, key=lambda r: r[1] - r[0], default=None)
        correlation[identifier] = (innermost[3] if innermost else "unattributed",
                                  any(r[3].endswith("/acoustic_substep!") for r in functions),
                                  any(r[3] == "call/fill_halo_regions!" for r in enclosing))
    phase_totals = defaultdict(lambda: [0, 0])
    kernel_totals = defaultdict(lambda: [0, 0, 0, 0])
    acoustic = 0
    halo_gpu = 0
    for start, end, identifier, name_id, registers, threads in kernels:
        duration = end - start
        phase, in_acoustic, in_halo = correlation.get(identifier, ("unattributed", False, False))
        phase_totals[phase][0] += duration
        phase_totals[phase][1] += 1
        kernel_totals[strings[name_id]][0] += duration
        kernel_totals[strings[name_id]][1] += 1
        kernel_totals[strings[name_id]][2] = max(kernel_totals[strings[name_id]][2], registers)
        kernel_totals[strings[name_id]][3] = max(kernel_totals[strings[name_id]][3], threads)
        acoustic += duration if in_acoustic else 0
        halo_gpu += duration if in_halo else 0
    host_totals = defaultdict(lambda: [0, 0])
    for start, end, _, label in ranges:
        host_totals[label][0] += end - start
        host_totals[label][1] += 1
    outer = [r for r in ranges if r[3] == "TimeSteppers/time_step!"]
    if len(outer) != steps:
        raise RuntimeError(f"Expected {steps} time steps, found {len(outer)}: {database}")
    window_start = min(r[0] for r in outer)
    window_end = max(max(r[1] for r in outer), max(k[1] for k in kernels))
    busy_ns = union_duration([(k[0], k[1]) for k in kernels] + [(c[0], c[1]) for c in copies])
    kernel_ns = sum(k[1] - k[0] for k in kernels)
    assigned = kernel_ns - phase_totals["unattributed"][0]
    return dict(window_seconds=(window_end - window_start) / 1e9,
                kernel_seconds=kernel_ns / 1e9, kernel_count=len(kernels),
                gpu_active_fraction=busy_ns / (window_end - window_start),
                attributed_kernel_fraction=assigned / kernel_ns,
                acoustic_kernel_fraction=acoustic / kernel_ns,
                halo_kernel_fraction=halo_gpu / kernel_ns,
                memcpy_bytes=sum(c[2] for c in copies),
                mpi_p2p_seconds=None if mpi is None else sum(end - start for start, end in mpi) / 1e9,
                phases={k: dict(seconds=v[0] / 1e9, calls=v[1], fraction=v[0] / kernel_ns)
                        for k, v in sorted(phase_totals.items(), key=lambda kv: -kv[1][0])},
                kernels={k: dict(seconds=v[0] / 1e9, calls=v[1], fraction=v[0] / kernel_ns,
                                 registers=v[2], threads=v[3])
                         for k, v in sorted(kernel_totals.items(), key=lambda kv: -kv[1][0])},
                host_ranges={k: dict(seconds=v[0] / 1e9, calls=v[1]) for k, v in host_totals.items()},
                cuda_apis={k: dict(seconds=v[0] / 1e9, calls=v[1])
                           for k, v in sorted(api_totals.items(), key=lambda kv: -kv[1][0])})


def main():
    manifest = Path(sys.argv[1])
    output = []
    for case in json.loads(manifest.read_text()):
        if case["state"] != "COMPLETE":
            raise RuntimeError(f"Incomplete case: {case['case']}")
        for profile in case["profiles"]:
            record = dict(case=case["case"], rank=profile["rank"], steps=case["steps"],
                          **analyze(profile["sqlite"], case["steps"]))
            if record["attributed_kernel_fraction"] < 0.99:
                raise RuntimeError(f"Insufficient kernel attribution: {record['case']} rank {record['rank']}")
            output.append(record)
            print(record["case"], record["rank"], f"kernels={record['kernel_count']}",
                  f"active={record['gpu_active_fraction']:.3f}",
                  f"acoustic={record['acoustic_kernel_fraction']:.3f}",
                  f"halo={record['halo_kernel_fraction']:.3f}")
    manifest.with_name("analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    with manifest.with_name("rank_summary.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "rank", "steps", "trace_window_seconds", "trace_seconds_per_step",
                         "summed_kernel_seconds", "kernel_launches", "timeline_active_fraction",
                         "attributed_kernel_fraction", "acoustic_kernel_fraction", "halo_kernel_fraction"])
        for record in output:
            writer.writerow([record["case"], record["rank"], record["steps"], record["window_seconds"],
                             record["window_seconds"] / record["steps"], record["kernel_seconds"],
                             record["kernel_count"], record["gpu_active_fraction"],
                             record["attributed_kernel_fraction"], record["acoustic_kernel_fraction"],
                             record["halo_kernel_fraction"]])
    for category in ("phases", "kernels"):
        with manifest.with_name(f"{category}.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["case", "rank", "name", "seconds", "kernel_launches", "fraction_of_kernel_time"])
            for record in output:
                for name, metric in record[category].items():
                    writer.writerow([record["case"], record["rank"], name,
                                     metric["seconds"], metric["calls"], metric["fraction"]])


if __name__ == "__main__":
    main()
