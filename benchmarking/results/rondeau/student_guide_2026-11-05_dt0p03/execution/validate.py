"""Validate this rerun against its logs and compare with the previous time step."""

import csv
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


root = Path(__file__).resolve().parents[1]
previous = root.parent / "student_guide_2026-11-04"
repository = root.parents[3]
identity = ("dynamics", "Nx", "Ny", "Nz", "Rx", "Ry", "gpus")


def key(row):
    return tuple(row[name] for name in identity)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


rows = read_csv(root / "scaling.csv")
old_rows = {key(row): row for row in read_csv(previous / "scaling.csv")}
geometries = [
    (256, 256, 128, 1, 1, 1),
    (256, 256, 128, 2, 1, 2),
    (256, 256, 128, 1, 2, 2),
    (256, 256, 128, 2, 2, 4),
    (256, 256, 128, 4, 1, 4),
    (256, 256, 128, 1, 4, 4),
    (512, 256, 128, 2, 1, 2),
    (512, 512, 128, 2, 2, 4),
]
expected = {(dynamics, *map(str, geometry))
            for dynamics in ("anelastic", "compressible_explicit", "compressible_splitexplicit")
            for geometry in geometries if dynamics != "anelastic" or geometry[-1] == 1}
assert len(rows) == 17 and len({key(row) for row in rows}) == 17
assert {key(row) for row in rows} == expected == set(old_rows)
logs = {}
for path in sorted((root / "runs/logs").glob("*.out")):
    content = path.read_text()
    results = re.findall(r"^RESULT (.+)$", content, re.MULTILINE)
    assert len(results) == 1, f"Expected one result: {path}"
    assert content.rstrip().endswith("exit status 0"), path
    assert "--time_step=0.03 " in content, path
    assert re.search(r"Δt = 0\.03\s", content), path
    result = dict(item.split("=", 1) for item in results[0].split())
    result.update(zip(("Nx", "Ny", "Nz"), result["size"].split("x")))
    result.update(zip(("Rx", "Ry"), result["partition"].split("x")))
    assert key(result) not in logs, path
    logs[key(result)] = (path.relative_to(root).as_posix(), result)
assert set(logs) == expected

outcomes = []
comparison = []
for row in rows:
    log, emitted = logs[key(row)]
    assert all(row[name] == emitted[name] for name in (*identity, "machine", "float", "steps"))
    gpus = int(row["gpus"])
    size = [int(row[name]) for name in ("Nx", "Ny", "Nz")]
    partition = [int(row[name]) for name in ("Rx", "Ry")]
    assert math.prod(partition) == gpus
    assert all(size[index] % partition[index] == 0 for index in (0, 1))
    assert row["machine"] == "rondeau" and row["float"] == "Float32"
    assert row["steps"] == emitted["steps"] == "500"
    seconds = float(row["seconds_per_step"])
    assert math.isfinite(seconds) and seconds > 0
    assert math.isclose(seconds, float(emitted["seconds_per_step"]), rel_tol=6e-5)
    assert math.isclose(math.prod(size) / seconds, float(row["points_per_second"]), rel_tol=2e-6)
    maximum = float(row["max_abs_w"])
    finite = math.isfinite(maximum)
    emitted_maximum = float(emitted["max_abs_w"])
    assert (math.isclose(maximum, emitted_maximum, rel_tol=6e-4) if finite
            else not math.isfinite(emitted_maximum))
    outcomes.append({**{name: row[name] for name in identity},
                     "log": log, "execution_status": "EXIT_0",
                     "solution_status": "FINITE_W" if finite else "NONFINITE_W",
                     "performance_usable": finite, "seconds_per_step": seconds,
                     "max_abs_w": maximum if finite else str(maximum)})
    old = old_rows[key(row)]
    old_finite = math.isfinite(float(old["max_abs_w"]))
    comparison.append({**{name: row[name] for name in identity},
                       "finite_w_dt0p05": old_finite, "finite_w_dt0p03": finite,
                       "ms_per_step_dt0p05": float(old["seconds_per_step"]) * 1000 if old_finite else "",
                       "ms_per_step_dt0p03": seconds * 1000 if finite else "",
                       "max_abs_w_dt0p05": old["max_abs_w"],
                       "max_abs_w_dt0p03": row["max_abs_w"]})

for line in (root / "provenance/source_sha256.txt").read_text().splitlines():
    digest, filename = line.split(None, 1)
    assert hashlib.sha256((repository / filename).read_bytes()).hexdigest() == digest, filename
base_commit = (root / "provenance/source_commit.txt").read_text().strip()
subprocess.run(["git", "diff", "--exit-code", base_commit, "--", "src", "Project.toml",
                "benchmarking/src/convective_boundary_layer.jl",
                "benchmarking/distributed/distributed_cbl_benchmark.jl"],
               cwd=repository, check=True)
for filename in ("Project.toml", "Manifest.toml", "LocalPreferences.toml"):
    assert (root / "provenance" / filename).read_bytes() == (previous / "provenance/breeze" / filename).read_bytes()
workers = json.loads((root / "provenance/four_rank_observation.json").read_text())["workers"]
assert len(workers) == 4
assert {worker["environment"]["CUDA_VISIBLE_DEVICES"] for worker in workers} == {"0", "1", "2", "3"}
assert all(worker["exe"] == "/u/ajtse/software/julia-1.11.9/bin/julia"
           and worker["environment"]["JULIA_NUM_THREADS"] == "1"
           and worker["environment"]["JULIA_CUDA_MEMORY_POOL"] == "none"
           and worker["mpi_library_mappings"] == ["/opt/uw/openmpi/4.1.7-with-cuda/lib/libmpi.so.40.30.7"]
           and "--time_step=0.03" in worker["command"] for worker in workers)
wrapper = json.loads((root / "execution/wrapper_status.json").read_text())
assert wrapper["benchmark_case_exit_statuses"] == [0] * 17

validation = {
    "validated_utc": datetime.now(timezone.utc).isoformat(),
    "execution_status": "COMPLETE", "configured_cases": 17, "completed_cases": len(rows),
    "finite_w_cases": sum(item["performance_usable"] for item in outcomes),
    "nonfinite_w_cases": sum(not item["performance_usable"] for item in outcomes),
    "time_step_seconds": 0.03, "warmup_steps": 10, "timed_steps": 500,
    "simulated_seconds_timed": 15.0, "acoustic_substeps": 12,
    "all_case_logs_exit_zero": True, "complete_unique_case_coverage": True,
    "all_logs_confirm_dt0p03": True, "geometry_valid": True,
    "timing_and_throughput_consistent": True, "source_hashes_unchanged_during_run": True,
    "package_environment_matches_previous": True,
    "four_distinct_devices_and_rank_environment_verified": True,
    "wrapper_exit_status": wrapper["wrapper_exit_status"],
    "wrapper_error_after_all_benchmark_cases": True,
    "completion_time_source": wrapper["completion_time_source"],
    "validation_scope": "Maximum vertical velocity finiteness only; full-field finiteness and distributed/serial numerical agreement are not tested by this benchmark.",
    "comparison_scope": "Identical step counts at different time steps cover different simulated durations; this is not a comparison at equal physical time. Initial perturbations are random and no seed is fixed by the driver. Single timing window per case on a shared server.",
    "other_section_8_experiments_run": False, "outcomes": outcomes,
}
(root / "validation.json").write_text(json.dumps(validation, indent=2, allow_nan=False) + "\n")
with (root / "comparison.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(comparison[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows({name: f"{value:.7g}" if name.startswith("ms_per_step") and isinstance(value, float)
                     else value for name, value in row.items()} for row in comparison)

labels = {"anelastic": "anelastic", "compressible_explicit": "explicit",
          "compressible_splitexplicit": "split explicit"}
table = ["| Dynamics | GPUs | Global grid | Partition | ms/step at 0.05 | ms/step at 0.03 | max abs w at 0.03 (m/s) | Log |",
         "|---|---:|---|---|---:|---:|---:|---|"]
for row, outcome, change in zip(rows, outcomes, comparison):
    old_timing = f'{change["ms_per_step_dt0p05"]:.3f}' if change["finite_w_dt0p05"] else "invalid"
    new_timing = f'{change["ms_per_step_dt0p03"]:.3f}' if change["finite_w_dt0p03"] else "invalid"
    maximum = f'{outcome["max_abs_w"]:.5f}' if outcome["performance_usable"] else outcome["max_abs_w"]
    grid = "×".join(row[name] for name in ("Nx", "Ny", "Nz"))
    partition = "×".join(row[name] for name in ("Rx", "Ry"))
    table.append(f'| {labels[row["dynamics"]]} | {row["gpus"]} | {grid} | {partition} | {old_timing} | {new_timing} | {maximum} | [log]({outcome["log"]}) |')
finite_count = validation["finite_w_cases"]
report = f"""# Rondeau scaling rerun — Δt = 0.03 s — 2026-11-05

**17 of 17 cases completed with exit status 0; {finite_count} produced finite maximum vertical velocity and {17 - finite_count} produced non-finite values.**

[CSV](scaling.csv) · [Comparison CSV](comparison.csv) · [Validation](validation.json) · [Previous Δt = 0.05 results](../student_guide_2026-11-04/README.md)

The user requested the original 17-case sweep again with the time step changed from 0.05 to 0.03 seconds. Grid sizes, partitions, Float32 precision, 10 warmup steps, 500 timed steps, and 12 split-explicit acoustic substeps are unchanged. No other section 8 experiments were run.

## Measurement and comparison

Dry convective boundary layer with WENO5, no turbulence closure or microphysics. Each case measures one 500-step window after compilation warmup. The driver synchronizes the GPU, uses MPI barriers, and reports the maximum elapsed time across ranks divided by the number of steps. Model construction, warmup, and final checks are outside the timer. Timings are single-window observations on a shared server, with no error bars.

The timed window covers **15 simulated seconds** at 0.03, compared with **25 simulated seconds** at 0.05. Including warmup, final velocity values are measured at about 15.3 and 25.5 simulated seconds. The case uses random initial perturbations and the driver does not fix a seed, so reruns also do not preserve an identical initial realization. Finite maximum vertical velocity passes the driver's limited check; it does not establish full-field finiteness, numerical accuracy, or long-duration stability. Non-finite cases are excluded from performance comparisons; their emitted timing numbers remain in the raw CSV.

Rondeau: four NVIDIA A100-SXM4-40GB GPUs, Julia 1.11.9, Breeze 0.14.0, Oceananigans 0.116.1, CUDA.jl 6.4.2, MPI.jl 0.20.29, system CUDA-aware Open MPI 4.1.7. Project, Manifest, and MPI preferences match the previous run byte for byte. The scientific model and Julia benchmark driver are unchanged. The sweep script now accepts `--time_step`, retaining its default of 0.05.

## Results

""" + "\n".join(table) + """

## Reproduction and evidence

The complete environment and invocation are recorded in [execution/run.sh](execution/run.sh). From the repository root, after sourcing the environment in that file, the sweep command is:

```bash
bash benchmarking/distributed/pelee_rondeau/run_scaling.sh \\
    --time_steps=500 --time_step=0.03 --max_gpus=4 \\
    --results_file="$OCEAN_RESULTS_ROOT/scaling.csv"
```

Use a fresh results directory and CSV for a new run. Logs record `--time_step=0.03`, the driver's actual Δt, and terminal exit status for every case. [Validation script](execution/validate.py) checks unique case coverage, geometry, CSV/log agreement, timing/throughput consistency, source hashes, and package environment identity. Provenance snapshots are under `provenance/`. The run uses the primary `/u/ajtse/Breeze.jl` repository on `distributed-benchmark`.

All 17 benchmark commands exited with status 0. The outer recording wrapper exited with status 2 after the sweep: its script was edited while Bash was waiting for the sweep, shifting the remaining input position and causing a parse error on the final timestamp command. The retained script passes syntax validation. The completion timestamp was recovered from the last case's exit-status footer. [Wrapper status](execution/wrapper_status.json) and the original sweep log preserve this bookkeeping failure separately from benchmark outcomes.

Text logs normalize trailing whitespace. [raw_logs.zip](raw_logs.zip) retains the original log bytes, with sizes and SHA-256 hashes in [log_originals.json](log_originals.json). CUDA library-path warnings and PMIx cleanup messages are retained in the logs. No additional MPI numerical-validation tests were run for this timestep-only rerun.
"""
if all(item["performance_usable"] for item in outcomes if item["dynamics"] == "compressible_splitexplicit"):
    report += """
## Split-explicit scaling

![Split-explicit timing and efficiency at Δt = 0.03](split_explicit_scaling.png)

The top panels use a fixed 256×256×128 global grid; the bottom panels keep 8,388,608 cells per GPU. Strong efficiency is t(1)/(N × t(N)); weak efficiency is t(1)/t(N). All recorded partitions are shown. [Vector SVG](split_explicit_scaling.svg).

```bash
python3 benchmarking/distributed/plot_split_explicit_results.py \\
    benchmarking/results/rondeau/student_guide_2026-11-05_dt0p03/scaling.csv \\
    --time-step=0.03
```
"""
(root / "README.md").write_text(report.rstrip() + "\n")
print(json.dumps({name: value for name, value in validation.items() if name != "outcomes"}, indent=2))
