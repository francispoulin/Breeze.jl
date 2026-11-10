"""Verify that Nsight selects complete, finite partition measurements per backend."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


specification = importlib.util.spec_from_file_location("profiles", Path(__file__).with_name("run_profiles.py"))
profiles = importlib.util.module_from_spec(specification)
specification.loader.exec_module(profiles)


class PartitionSelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.original_study = profiles.STUDY
        profiles.STUDY = Path(self.directory.name)
        self.addCleanup(setattr, profiles, "STUDY", self.original_study)
        self.state = {"cases": []}
        self.benchmark = {"cases": []}
        self.expected = {}
        for index, dynamics in enumerate(("compressible_explicit", "compressible_splitexplicit")):
            self.state["cases"].append({"dynamics": dynamics, "communication": "SERIAL", "gpus": 1, "partition": [1, 1]})
            for offset, backend in enumerate(("MPI", "NCCL")):
                winner = [2, 2] if (index + offset) % 2 == 0 else [4, 1]
                self.expected[dynamics, backend] = winner
                self.state["cases"].append({"dynamics": dynamics, "communication": backend, "gpus": 4, "partition": None})
                for partition in ([2, 2], [4, 1]):
                    self.benchmark["cases"].append({"id": f"{dynamics}_{backend}_{partition}", "dynamics": dynamics,
                                                   "communication": backend, "gpus": 4, "partition": partition,
                                                   "state": "COMPLETE", "fields_finite": True, "performance_valid": True,
                                                   "seconds_per_step": 1.0 if partition == winner else 2.0})

    def test_selects_independently_for_each_dynamics_and_backend(self):
        profiles.choose_partitions(self.state, self.benchmark)
        self.assertEqual(len(self.state["partition_selection"]), 4)
        for case in self.state["cases"]:
            expected = [1, 1] if case["gpus"] == 1 else self.expected[case["dynamics"], case["communication"]]
            self.assertEqual(case["partition"], expected)
        stored = json.loads((profiles.STUDY / "partition_selection.json").read_text())
        self.assertEqual(stored, self.state["partition_selection"])

    def test_requires_both_layouts_to_finish(self):
        self.benchmark["cases"][0]["state"] = "RUNNING"
        with self.assertRaises(AssertionError):
            profiles.choose_partitions(self.state, self.benchmark)

    def test_rejects_nonfinite_fields(self):
        self.benchmark["cases"][0]["fields_finite"] = False
        with self.assertRaises(AssertionError):
            profiles.choose_partitions(self.state, self.benchmark)

    def test_rejects_missing_layout(self):
        self.benchmark["cases"].pop(0)
        with self.assertRaises(AssertionError):
            profiles.choose_partitions(self.state, self.benchmark)

    def test_rejects_invalid_timing(self):
        for value in (float("nan"), float("inf"), 0.0, -1.0):
            with self.subTest(value=value):
                benchmark = copy.deepcopy(self.benchmark)
                benchmark["cases"][0]["seconds_per_step"] = value
                with self.assertRaises(AssertionError):
                    profiles.choose_partitions(self.state, benchmark)


if __name__ == "__main__":
    unittest.main()
