"""Find complete, validated, unprofiled Oceananigans A100 weak-scaling series."""

import hashlib
import json
import math
from pathlib import Path
from statistics import median


def measurement_from_case(path, records, suite):
    if not isinstance(records, list) or not records:
        return None
    config = records[0].get("configuration", {})
    ranks = config.get("ranks")
    shape = config.get("global_resolution", [])
    if (type(ranks) is not int or ranks not in (1, 2, 4) or len(records) != ranks or len(shape) != 3
            or any(type(n) is not int or n <= 0 for n in shape)
            or math.prod(shape) != ranks * 512**3 or config.get("device") != "GPU"
            or config.get("profile") is not False or "weak" not in config.get("series", "")
            or config.get("execution_mode", "baseline") != "baseline"):
        return None
    if suite.get("server", {}).get("machine") != "rondeau":
        return None
    case = next((case for case in suite.get("cases", []) if case.get("id") == path.parent.name), None)
    if case is None or case.get("state") != "COMPLETED":
        return None
    if {record.get("rank") for record in records} != set(range(ranks)):
        return None
    if any(record.get("configuration") != config or record.get("finite_state") is not True
           or record.get("pressure_solver_converged") is False
           or "A100" not in record.get("metadata", {}).get("gpu_name", "")
           or not record.get("metadata", {}).get("hostname", "").startswith("rondeau")
           or record.get("metadata", {}).get("execution_mode", "baseline") != "baseline"
           for record in records):
        return None
    steps = config.get("time_steps", 0)
    windows = [record.get("window_seconds", []) for record in records]
    if (not isinstance(steps, int) or steps <= 0 or not windows[0]
            or any(len(w) != len(windows[0]) for w in windows)
            or any(not math.isfinite(float(t)) or float(t) <= 0 for w in windows for t in w)):
        return None
    # Form the slowest-rank time for each synchronized window before taking a median.
    global_windows = [max(float(w[index]) for w in windows) for index in range(len(windows[0]))]
    seconds = median(global_windows) / steps
    version = records[0].get("metadata", {}).get("oceananigans_version")
    if not version or any(record.get("metadata", {}).get("oceananigans_version") != version for record in records):
        return None
    varying = {"ranks", "global_resolution", "partition", "x_sizes", "y_sizes", "z_sizes", "grid_name"}
    fixed_config = {key: value for key, value in config.items() if key not in varying}
    cohort = (str(path.parents[1]), version, json.dumps(fixed_config, sort_keys=True))
    return dict(gpus=ranks, Nx=shape[0], Ny=shape[1], Nz=shape[2], seconds_per_step=seconds,
                fields_finite="true", source=str(path), timing_statistic="median_of_slowest_rank_windows",
                global_window_seconds=global_windows, time_steps=steps, configuration=config,
                oceananigans_version=version, cohort=cohort)


def discover_complete_series(root, model_type):
    groups = {}
    available = set()
    for path in root.rglob("results.json"):
        if any(part in {"dagger", "archive", "raw_profiles", "provenance"} for part in path.relative_to(root).parts):
            continue
        suite_path = path.parents[1] / "suite.json"
        if not suite_path.is_file():
            continue
        try:
            raw = path.read_bytes()
            suite_raw = suite_path.read_bytes()
            row = measurement_from_case(path, json.loads(raw), json.loads(suite_raw))
            if row is None or row["configuration"].get("model_type") != model_type:
                continue
            row["source_sha256"] = hashlib.sha256(raw).hexdigest()
            row["suite_sha256"] = hashlib.sha256(suite_raw).hexdigest()
            row["source_mtime"] = path.stat().st_mtime
            available.add(row["gpus"])
            group = groups.setdefault(row.pop("cohort"), {})
            previous = group.get(row["gpus"])
            if previous is None or previous["source_mtime"] < row["source_mtime"]:
                group[row["gpus"]] = row
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    complete = [group for group in groups.values() if set(group) == {1, 2, 4}]
    if not complete:
        return None, sorted(available)
    selected = max(complete, key=lambda group: min(row["source_mtime"] for row in group.values()))
    return [selected[n] for n in (1, 2, 4)], sorted(available)
