"""Verify that migration cannot start while measurements or captures remain active."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


specification = importlib.util.spec_from_file_location("migration", Path(__file__).with_name("run_migration.py"))
migration = importlib.util.module_from_spec(specification)
specification.loader.exec_module(migration)


class MigrationGateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.states = []
        self.validations = []
        for name, count in (("weak_scaling_mpi_nccl_2026-11-09_dt0p005", 14),
                            ("nsight_mpi_nccl_2026-11-09_dt0p005", 6)):
            study = self.root / "benchmarking/results/rondeau" / name
            (study / "execution").mkdir(parents=True)
            (study / "validation").mkdir()
            state = {"status": "COMPLETE", "cases": [{"state": "COMPLETE", "fields_finite": True} for _ in range(count)]}
            validation = {"finalized_utc": "2026-11-09T23:00:00+00:00", "verified_rank_captures": 18,
                          "all_capture_sizes_hashes_kernel_counts_and_readme_paths_verified": True}
            self.states.append((study, state))
            self.validations.append((study, validation))
            (study / "execution/controller_process.json").write_text(json.dumps({"pid": 42}))
        self.write()

    def write(self):
        for study, state in self.states:
            (study / "study.json").write_text(json.dumps(state))
        for study, validation in self.validations:
            (study / "validation/results.json").write_text(json.dumps(validation))

    def test_ready_only_after_both_workflows_and_finalizers_finish(self):
        self.assertTrue(migration.workflows_ready(self.root, alive=lambda pid: False)[0])

    def test_waits_for_controller_exit(self):
        self.assertFalse(migration.workflows_ready(self.root, alive=lambda pid: True)[0])

    def test_waits_for_nsight_completion(self):
        self.states[1][1]["status"] = "WAITING_FOR_BENCHMARK"
        self.write()
        self.assertFalse(migration.workflows_ready(self.root, alive=lambda pid: False)[0])

    def test_rejects_nonfinite_completed_case(self):
        self.states[0][1]["cases"][0]["fields_finite"] = False
        self.write()
        self.assertFalse(migration.workflows_ready(self.root, alive=lambda pid: False)[0])

    def test_waits_for_final_validation_and_all_rank_reports(self):
        for invalid in ({}, {"finalized_utc": "done", "verified_rank_captures": 17}):
            with self.subTest(validation=invalid):
                self.validations[1] = (self.validations[1][0], invalid)
                self.write()
                self.assertFalse(migration.workflows_ready(self.root, alive=lambda pid: False)[0])


if __name__ == "__main__":
    unittest.main()
