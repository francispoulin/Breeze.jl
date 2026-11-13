"""Check the agreed workload and rejection of incomplete or mismatched evidence."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from study import build_plan, case_command, check_sources, finalize, validate_measurement, validate_preflight_placement
from report import report


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.case = next(case for case in build_plan() if case["gpus"] == 2 and case["dynamics"] == "anelastic")
        for rank in range(2):
            (self.directory / f"rank_{rank}.log").write_text(
                f"PLACEMENT rank={rank} host=node gpu_uuid=gpu{rank} communicator=Comm threads=1 pool=none\n"
                f"VALIDATION rank={rank} fields_finite=true gpu_used_bytes=100\n")
        (self.directory / "hardware_rank_0.txt").write_text("Product Name: NVIDIA H100 80GB HBM3\n")
        self.row = dict(dynamics="anelastic", float="Float32", Nx=1024, Ny=512, Nz=512,
                        Rx=2, Ry=1, fields_finite="true", communication="MPI", steps=500,
                        gpus=2, seconds_per_step=0.5, points_per_second=100, max_abs_w=0.01)
        self.write_measurement()

    def write_measurement(self):
        with (self.directory / "measurement.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.row))
            writer.writeheader()
            writer.writerow(self.row)

    def test_agreed_fifteen_mpi_workloads(self):
        plan = build_plan()
        self.assertEqual(len(plan), 15)
        self.assertEqual(len({case["id"] for case in plan}), 15)
        self.assertEqual({case["dynamics"] for case in plan}, {"anelastic", "compressible_explicit", "compressible_splitexplicit"})
        for case in plan:
            count = case["gpus"]
            self.assertEqual(case["backend"], "MPI")
            self.assertEqual(case["partition"], f"{count}x1")
            self.assertEqual(case["size"], f"{512 * count}x512x512")
            self.assertEqual(case["time_step"], 0.01 / count)

    def test_command_uses_deterministic_unprofiled_configuration(self):
        (self.directory / "config.json").write_text(json.dumps(dict(source="/source", environment="/environment")))
        case = next(case for case in build_plan() if case["gpus"] == 16)
        command = case_command(case, self.directory)
        for argument in ("--nodes=4", "--ntasks=16", "--partition=16x1", "--size=8192x512x512",
                         "--comm=MPI", "--random_perturbation=false", "--float_type=Float32", "--nvtx=false",
                         "--profile=false", "--time_step=0.000625", "--warmup_steps=10", "--time_steps=500",
                         "--gpus-per-node=h100:4", "--gpu-bind=none"):
            self.assertIn(argument, command)

    def test_preflight_rejects_shared_or_missing_devices(self):
        output = "PREFLIGHT rank=0 host=node gpu_uuid=gpu0 devices=2\nPREFLIGHT rank=1 host=node gpu_uuid=gpu1 devices=2\n"
        self.assertEqual(len(validate_preflight_placement(output, 2)), 2)
        for invalid in (output.replace("gpu1", "gpu0"), output.splitlines()[0], output + output, output.replace("devices=2", "devices=1")):
            with self.assertRaises(RuntimeError):
                validate_preflight_placement(invalid, 2)

    def test_two_active_gpus_in_full_node_allocation(self):
        (self.directory / "config.json").write_text(json.dumps(dict(source="/source", environment="/environment")))
        command = case_command(self.case, self.directory, node_gpus=4)
        for argument in ("--ntasks=2", "--size=1024x512x512", "--partition=2x1", "--gpus-per-node=h100:4"):
            self.assertIn(argument, command)
        output = "PREFLIGHT rank=0 host=node gpu_uuid=gpu0 devices=4\nPREFLIGHT rank=1 host=node gpu_uuid=gpu1 devices=4\n"
        self.assertEqual(len(validate_preflight_placement(output, 2, node_gpus=4)), 2)

    def test_accept_complete_evidence(self):
        self.assertEqual(validate_measurement(self.case, self.directory, 0)["dynamics"], "anelastic")

    def test_reject_wrong_workload(self):
        for key, value in (("Nx", 512), ("Ry", 2), ("float", "Float64"), ("dynamics", "compressible_explicit"),
                           ("max_abs_w", "nan"), ("communication", "NCCL"), ("steps", 499)):
            with self.subTest(key=key):
                previous = self.row[key]
                self.row[key] = value
                self.write_measurement()
                with self.assertRaises(RuntimeError):
                    validate_measurement(self.case, self.directory, 0)
                self.row[key] = previous
        self.write_measurement()

    def test_reject_missing_or_nonfinite_rank(self):
        path = self.directory / "rank_1.log"
        original = path.read_text()
        for replacement in ("", original.replace("fields_finite=true", "fields_finite=false"), original.replace("gpu1", "gpu0")):
            path.write_text(replacement)
            with self.assertRaises(RuntimeError):
                validate_measurement(self.case, self.directory, 0)

    def test_reject_non_h100_or_missing_hardware(self):
        path = self.directory / "hardware_rank_0.txt"
        path.write_text("NVIDIA A100\n")
        with self.assertRaises(RuntimeError):
            validate_measurement(self.case, self.directory, 0)
        path.unlink()
        with self.assertRaises(RuntimeError):
            validate_measurement(self.case, self.directory, 0)

    def test_frozen_source_change_is_detected(self):
        (self.directory / "provenance").mkdir()
        source = self.directory / "source.jl"
        source.write_text("original\n")
        (self.directory / "provenance/hashes.json").write_text(json.dumps({str(source): hashlib.sha256(source.read_bytes()).hexdigest()}))
        check_sources(self.directory)
        source.write_text("changed\n")
        with self.assertRaises(RuntimeError):
            check_sources(self.directory)

    def test_finalizer_refuses_queued_or_running_allocations(self):
        (self.directory / "submission.json").write_text(json.dumps(dict(jobs=[dict(job_id="123")])))
        with patch("study.run_text", return_value="123|RUNNING\n"):
            with self.assertRaisesRegex(RuntimeError, "Refusing to finalize"):
                finalize(self.directory, False)

    def test_report_excludes_failed_case_and_includes_anelastic(self):
        plan = build_plan()
        (self.directory / "plan.json").write_text(json.dumps(plan))
        for case, state in ((plan[0], "validated"), (plan[1], "failed")):
            directory = self.directory / "cases" / case["id"]
            directory.mkdir(parents=True)
            row = dict(self.row, gpus=1, communication="SERIAL", points_per_second_per_gpu=100)
            (directory / "status.json").write_text(json.dumps(dict(state=state, measurement=row)))
        report(self.directory)
        with (self.directory / "comparison.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]["weak_efficiency_percent"]), 100)
        self.assertNotIn(b"\r", (self.directory / "comparison.csv").read_bytes())
        self.assertIn("1/15 cases validated", (self.directory / "README.md").read_text())
        self.assertGreater((self.directory / "weak_scaling.png").stat().st_size, 1000)
        self.assertTrue((self.directory / "weak_scaling.svg").exists())


if __name__ == "__main__":
    unittest.main()
