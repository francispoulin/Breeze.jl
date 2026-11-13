"""Reject failed/incomplete dependencies and incompatible Oceananigans plotting inputs."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ocean_inputs import discover_complete_series, measurement_from_case
from queue_plots import anelastic_gate, main


def successful_anelastic():
    cases = []
    for n in (1, 2, 4):
        cases.append(dict(ranks=n, state="COMPLETE", exit_code=0, source_commit="same revision",
            measurement=dict(dynamics="anelastic", gpus=str(n), Nx=str(512*n), Ny="512", Nz="512",
                Rx=str(n), Ry="1", steps="500", float="Float32", fields_finite="true",
                communication="SERIAL" if n == 1 else "MPI", seconds_per_step="1", max_abs_w="0.001"),
            placements=[[str(rank), "rondeau", f"GPU-id{rank}", "none" if n == 1 else "Comm"] for rank in range(n)],
            rank_validation=[[str(rank), "true"] for rank in range(n)]))
    return dict(status="COMPLETE", cases=cases)


def ocean_case(n, model="HydrostaticFreeSurfaceModel"):
    config = dict(model_type=model, ranks=n, global_resolution=[512*n, 512, 512], partition=[n, 1, 1],
                  device="GPU", profile=False, series="gpu_weak_scaling_512", execution_mode="baseline",
                  time_steps=1, samples=3, dt=0.005)
    records = [dict(rank=rank, configuration=deepcopy(config), finite_state=True,
                    metadata=dict(gpu_name="NVIDIA A100", hostname="rondeau", oceananigans_version="0.117.2"),
                    window_seconds=[1, 1, 1]) for rank in range(n)]
    suite = dict(server=dict(machine="rondeau"), cases=[dict(id=f"{i}_ranks", state="COMPLETED") for i in (1, 2, 4)])
    return records, suite


class ConditionalPlotChecks(unittest.TestCase):
    def test_failed_dependency_never_invokes_either_renderer(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            results = base / "source/benchmarking/results/rondeau/plots"
            (results / "provenance").mkdir(parents=True)
            runtime = base / "runtime"; runtime.mkdir()
            parent_results = base / "anelastic"; parent_results.mkdir()
            parent_status = base / "anelastic_status.json"
            parent_status.write_text(json.dumps(dict(status="FAILED", results=str(parent_results))))
            (parent_results / "results.json").write_text(json.dumps(successful_anelastic()))
            (results / "plan.json").write_text(json.dumps(dict(anelastic_status=str(parent_status))))
            (results / "provenance/source_sha256.json").write_text("{}\n")
            with patch("queue_plots.__file__", str(results / "queue_plots.py")), \
                 patch("queue_plots.sys.argv", ["queue_plots.py", "--runtime", str(runtime)]), \
                 patch("queue_plots.signal.signal"), \
                 patch("queue_plots.render_breeze") as breeze, \
                 patch("queue_plots.render_ocean") as ocean:
                self.assertEqual(main(), 0)
                breeze.assert_not_called()
                ocean.assert_not_called()
            status = json.loads((runtime / "status.json").read_text())
            self.assertEqual(status["status"], "SKIPPED_ANELASTIC_UNSUCCESSFUL")
            self.assertEqual(set(status["tasks"].values()), {"SKIPPED_ANELASTIC_UNSUCCESSFUL"})

    def test_no_early_plotting_even_if_data_look_complete(self):
        for state in ["RUNNING", "RENDERING_PLOTS", "WAITING_FOR_PRECEDING_NVTX_CAMPAIGN"]:
            self.assertEqual(anelastic_gate({"status": state}, successful_anelastic()), "WAIT")

    def test_failure_cancel_and_missing_output_skip_both_jobs(self):
        for state in ["FAILED", "ERROR", "CANCELLED"]:
            self.assertEqual(anelastic_gate({"status": state}, successful_anelastic()), "SKIP")
        self.assertEqual(anelastic_gate({"status": "COMPLETE"}, None), "SKIP")

    def test_all_three_validated_cases_are_required(self):
        data = successful_anelastic()
        self.assertEqual(anelastic_gate({"status": "COMPLETE"}, data), "READY")
        self.assertEqual(anelastic_gate({"status": "RESULTS_READY_PUSH_FAILED"}, data), "READY")
        invalid = deepcopy(data); invalid["cases"].pop()
        self.assertEqual(anelastic_gate({"status": "COMPLETE"}, invalid), "SKIP")
        for field, value in [("exit_code", 1), ("state", "FAILED"), ("source_commit", "different revision")]:
            invalid = deepcopy(data); invalid["cases"][1][field] = value
            self.assertEqual(anelastic_gate({"status": "COMPLETE"}, invalid), "SKIP")

    def test_nonfinite_fields_wrong_grid_and_duplicate_devices_reject_success(self):
        data = successful_anelastic()
        for field, value in [("fields_finite", "false"), ("seconds_per_step", "nan"), ("Nx", "512")]:
            invalid = deepcopy(data); invalid["cases"][1]["measurement"][field] = value
            self.assertEqual(anelastic_gate({"status": "COMPLETE"}, invalid), "SKIP")
        invalid = deepcopy(data); invalid["cases"][1]["placements"][1][2] = "GPU-id0"
        self.assertEqual(anelastic_gate({"status": "COMPLETE"}, invalid), "SKIP")
        invalid = deepcopy(data); invalid["cases"][1]["rank_validation"][0][1] = "false"
        self.assertEqual(anelastic_gate({"status": "COMPLETE"}, invalid), "SKIP")

    def test_ocean_uses_median_of_slowest_rank_windows(self):
        records, suite = ocean_case(2)
        records[0]["window_seconds"] = [1, 1, 9]
        records[1]["window_seconds"] = [1, 9, 1]
        row = measurement_from_case(Path("/u/ajtse/series/2_ranks/results.json"), records, suite)
        self.assertEqual(row["seconds_per_step"], 9)

    def test_ocean_rejects_profiled_wrong_size_nonfinite_and_non_a100(self):
        path = Path("/u/ajtse/series/1_ranks/results.json")
        records, suite = ocean_case(1)
        self.assertIsNotNone(measurement_from_case(path, records, suite))
        for change in ["profile", "size", "finite", "hardware", "unfinished"]:
            records, suite = ocean_case(1)
            if change == "profile": records[0]["configuration"]["profile"] = True
            if change == "size": records[0]["configuration"]["global_resolution"] = [32, 32, 32]
            if change == "finite": records[0]["finite_state"] = False
            if change == "hardware": records[0]["metadata"]["gpu_name"] = "H100"
            if change == "unfinished": suite["cases"][0]["state"] = "RUNNING"
            self.assertIsNone(measurement_from_case(path, records, suite), change)

    def test_missing_or_mixed_ocean_series_do_not_unlock_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(discover_complete_series(root, "HydrostaticFreeSurfaceModel"), (None, []))
            series = root / "series"; series.mkdir()
            for n in (1, 2, 4):
                records, suite = ocean_case(n)
                if n == 2:
                    for record in records: record["configuration"]["dt"] = 0.01
                case = series / f"{n}_ranks"; case.mkdir()
                (case / "results.json").write_text(json.dumps(records))
            (series / "suite.json").write_text(json.dumps(suite))
            self.assertIsNone(discover_complete_series(root, "HydrostaticFreeSurfaceModel")[0])
            records, _ = ocean_case(2)
            (series / "2_ranks/results.json").write_text(json.dumps(records))
            rows, available = discover_complete_series(root, "HydrostaticFreeSurfaceModel")
            self.assertEqual(available, [1, 2, 4])
            self.assertEqual([row["gpus"] for row in rows], [1, 2, 4])


if __name__ == "__main__":
    unittest.main()
