"""Validation must reject successful processes without complete per-rank evidence."""
import csv
from pathlib import Path
import tempfile
import unittest
from run_case import validate


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.case = dict(gpus=2, backend="NCCL")
        for rank in range(2):
            (self.directory / f"rank_{rank}.log").write_text(
                f"PLACEMENT rank={rank} host=node gpu_uuid=gpu{rank} communicator=NCCLCommunicator threads=1 pool=none\n"
                f"VALIDATION rank={rank} fields_finite=true gpu_used_bytes=100\n")
        with (self.directory / "measurement.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=["fields_finite", "communication", "steps", "gpus", "seconds_per_step", "points_per_second"])
            writer.writeheader()
            writer.writerow(dict(fields_finite="true", communication="NCCL", steps=500, gpus=2, seconds_per_step=0.5, points_per_second=100))

    def test_valid_evidence(self):
        self.assertEqual(validate(self.case, self.directory, 0)["communication"], "NCCL")

    def test_missing_rank(self):
        (self.directory / "rank_1.log").write_text("")
        with self.assertRaises(RuntimeError):
            validate(self.case, self.directory, 0)

    def test_shared_gpu(self):
        path = self.directory / "rank_1.log"
        path.write_text(path.read_text().replace("gpu1", "gpu0"))
        with self.assertRaises(RuntimeError):
            validate(self.case, self.directory, 0)

    def test_wrong_communicator(self):
        path = self.directory / "rank_1.log"
        path.write_text(path.read_text().replace("NCCLCommunicator", "Comm"))
        with self.assertRaises(RuntimeError):
            validate(self.case, self.directory, 0)

    def test_nonfinite_rank(self):
        path = self.directory / "rank_1.log"
        path.write_text(path.read_text().replace("fields_finite=true", "fields_finite=false"))
        with self.assertRaises(RuntimeError):
            validate(self.case, self.directory, 0)

    def test_failed_process(self):
        with self.assertRaises(RuntimeError):
            validate(self.case, self.directory, 1)


if __name__ == "__main__":
    unittest.main()
