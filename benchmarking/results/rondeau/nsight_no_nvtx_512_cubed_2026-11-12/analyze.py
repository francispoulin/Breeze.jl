"""Verify and summarize this CUDA-only capture without needing NVTX ranges."""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3


def main():
    results = Path(__file__).resolve().parent
    root = results.parents[3]
    raw = root / ".runtime/nsight_no_nvtx_512_cubed_2026-11-12/raw"
    report, database = raw / "profile.nsys-rep", raw / "profile.sqlite"
    reference = root / "benchmarking/results/rondeau/nvtx_demo_512_cubed_2026-11-10"
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        strings = dict(connection.execute("SELECT id,value FROM StringIds"))
        tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        nvtx_count = connection.execute("SELECT count(*) FROM NVTX_EVENTS").fetchone()[0] if "NVTX_EVENTS" in tables else 0
        kernels = connection.execute(
            "SELECT start,end,shortName,deviceId,streamId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
        copies = connection.execute("SELECT start,end,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
        capture = connection.execute("SELECT duration,startTime,stopTime FROM ANALYSIS_DETAILS").fetchall()
        capture_types = [r[0] for r in connection.execute(
            "SELECT value FROM META_DATA_CAPTURE WHERE name='CAPTURE_EVENT_TYPE'")]
    measured = list(csv.DictReader((results / "profiled_measurements.csv").open()))
    assert len(measured) == 1
    row = measured[0]
    assert all(row[key] == value for key, value in dict(
        dynamics="compressible_splitexplicit", float="Float32", Nx="512", Ny="512", Nz="512",
        Rx="1", Ry="1", gpus="1", steps="10", communication="SERIAL", fields_finite="true").items())
    assert nvtx_count == 0 and not any("nvtx" in value.lower() for value in capture_types)
    assert len(kernels) == 10300 and len(copies) == 60
    assert len(capture) == 1
    kernel_totals = defaultdict(lambda: [0, 0])
    for start, end, identifier, _, _ in kernels:
        kernel_totals[strings[identifier]][0] += end - start
        kernel_totals[strings[identifier]][1] += 1
    old_kernel_counts = {r["name"]: int(r["kernel_launches"]) for r in
                         csv.DictReader((reference / "kernels.csv").open())}
    assert {name: values[1] for name, values in kernel_totals.items()} == old_kernel_counts
    for name in ("Project.toml", "Manifest.toml", "LocalPreferences.toml"):
        assert (results / "provenance" / name).read_bytes() == (reference / "provenance" / name).read_bytes()
    hashes = json.loads((results / "provenance/source_sha256.json").read_text())
    assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
    log = (results / "run.log").read_text()
    assert "PROFILE_START rank=0 steps=10" in log and "PROFILE_END rank=0 steps=10" in log
    assert "NVTX host instrumentation enabled" not in log
    uuid = re.search(r"DEVICE rank=0 uuid=(\S+) visible=2", log).group(1)
    operations = [(s, e) for s, e, *_ in kernels + copies]
    window_start, window_end = min(s for s, _ in operations), max(e for _, e in operations)
    union_ns, previous_end = 0, window_start
    for start, end in sorted(operations):
        union_ns += max(0, end - max(start, previous_end))
        previous_end = max(previous_end, end)
    summary = dict(
        status="PASS", report=str(report), report_bytes=report.stat().st_size,
        report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(), sqlite=str(database),
        trace="cuda", nvtx_instrumentation=False, nvtx_event_count=nvtx_count,
        capture_event_types=capture_types, size=[512, 512, 512], steps=10, warmup_steps=10,
        time_step=0.02, nominal_acoustic_substeps=12, float_type="Float32", gpu_physical_index=2,
        gpu_uuid=uuid, fields_finite=True, max_abs_w=float(row["max_abs_w"]),
        post_run_launcher_messages=[line for line in log.splitlines() if "PMIX ERROR:" in line],
        original_max_abs_w=float(next(csv.DictReader((reference / "profiled_measurements.csv").open()))["max_abs_w"]),
        kernel_count=len(kernels), memcpy_count=len(copies), kernel_counts_match_nvtx_demo=True,
        source_hashes_match_nvtx_demo=len(hashes),
        capture_seconds=capture[0][0] / 1e9,
        gpu_operation_window_seconds=(window_end - window_start) / 1e9,
        gpu_operation_window_seconds_per_step=(window_end - window_start) / 1e10,
        summed_kernel_seconds=sum(e - s for s, e, *_ in kernels) / 1e9,
        timeline_active_fraction=union_ns / (window_end - window_start))
    (results / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    total_ns = sum(v[0] for v in kernel_totals.values())
    with (results / "kernels.csv").open("w") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["name", "kernel_launches", "total_ms", "fraction_of_summed_kernel_time"])
        for name, (duration, count) in sorted(kernel_totals.items(), key=lambda item: -item[1][0]):
            writer.writerow([name, count, duration / 1e6, duration / total_ns])
    lines = ["Breeze 512 cubed: Nsight Systems CUDA-only capture, NVTX disabled",
             "Split-explicit dry CBL; Float32; WENO5; dt=0.02; 10 warmup + 10 captured steps",
             f"GPU: physical 2, A100-SXM4-40GB, UUID {uuid}",
             f"Validation: fields finite; {len(hashes)} source hashes and all kernel counts match the NVTX demo",
             f"NVTX events: {nvtx_count}; kernels: {len(kernels)}; copies: {len(copies)}",
             f"Nsight capture duration: {summary['capture_seconds']:.6f} s",
             f"GPU operation span: {summary['gpu_operation_window_seconds']:.6f} s total; "
             f"{summary['gpu_operation_window_seconds_per_step']:.6f} s/step averaged over 10 steps",
             f"Summed kernel duration: {summary['summed_kernel_seconds']:.6f} s",
             "GPU operation span covers first to last captured GPU operation, including gaps.",
             "Driver seconds/step includes report generation; do not interpret it as execution time.",
             "Kernel shares are fractions of summed kernel durations, not hardware utilization.",
             "No NVTX ranges are available to attribute kernels to host methods.", "", "CUDA kernels:"]
    if summary["post_run_launcher_messages"]:
        lines[12:12] = ["Post-run launcher message (capture and export completed): " + message
                        for message in summary["post_run_launcher_messages"]]
    for name, (duration, count) in sorted(kernel_totals.items(), key=lambda item: -item[1][0]):
        lines.append(f"{name}: {count} launches, {duration / 1e6:.3f} ms, {duration / total_ns:.2%} of kernel time")
    (results / "summary.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: summary[k] for k in ("status", "nvtx_event_count", "kernel_count", "capture_seconds",
                                            "gpu_operation_window_seconds_per_step", "max_abs_w")}))


if __name__ == "__main__":
    main()
