"""CPU checks of workload selection and guards against invalid distributed results."""

import csv
from pathlib import Path
import tempfile
import unittest

from run_campaign import command_for, gpus_available, predecessor_ready, validate_run, verify_sources


class CampaignChecks(unittest.TestCase):
    def test_requested_workload_and_capture_boundary(self):
        for backend in ("NCCL", "MPI"):
            for profiling in (False, True):
                command = command_for(Path("/u/ajtse/source"), Path("/u/ajtse/results"),
                                      Path("/u/ajtse/runtime"), {"julia": "julia"}, backend, profiling)
                self.assertIn("--size=2048x512x512", command)
                self.assertIn("--partition=4x1", command)
                self.assertIn("--comm=" + backend, command)
                self.assertIn("--nvtx=true", command)
                self.assertIn("--time_step=0.0025", command)
                self.assertIn("--profile=" + str(profiling).lower(), command)
                self.assertEqual(any("profile_rank.sh" in part for part in command), profiling)

    def test_busy_missing_and_low_memory_gpus_block_launch(self):
        inventory = "\n".join(f"{rank}, 39000, 0" for rank in range(4))
        self.assertTrue(gpus_available("", inventory))
        self.assertFalse(gpus_available("12345\n", inventory))
        self.assertFalse(gpus_available("", inventory.replace("3, 39000, 0", "")))
        self.assertFalse(gpus_available("", inventory.replace("39000", "34000", 1)))
        self.assertFalse(gpus_available("", "NVIDIA-SMI unavailable"))

    def test_preceding_checks_run_first(self):
        self.assertFalse(predecessor_ready({"status": "WAITING_FOR_FOUR_FREE_GPUS"}))
        self.assertFalse(predecessor_ready({"status": "RUNNING"}))
        self.assertTrue(predecessor_ready({"status": "COMPLETE"}))
        self.assertTrue(predecessor_ready(dict(status="RESULTS_READY_PUSH_FAILED", tests_status="PASS",
                                               completed_cases=6, controller_exit_code=0)))
        with self.assertRaises(RuntimeError):
            predecessor_ready({"status": "FAILED"})

    def test_changed_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.jl").write_text("changed source\n")
            with self.assertRaises(RuntimeError):
                verify_sources(root, {"source.jl": "unexpected hash"})

    def fixture(self, directory, backend):
        communicator = "NCCLCommunicator" if backend == "NCCL" else "Comm"
        log = "\n".join(f"PLACEMENT rank={rank} host=rondeau gpu_uuid=GPU-device{rank} communicator={communicator}\n"
                        f"VALIDATION rank={rank} fields_finite=true" for rank in range(4))
        row = dict(machine="rondeau", dynamics="compressible_splitexplicit", float="Float32", Nx="2048",
                   Ny="512", Nz="512", Rx="4", Ry="1", gpus="4", steps="10", communication=backend,
                   fields_finite="true", max_abs_w="0.001", seconds_per_step="1.4")
        path = Path(directory) / "measurement.csv"
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=row.keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerow(row)
        return log, path

    def test_four_rank_success_requires_correct_communicator(self):
        with tempfile.TemporaryDirectory() as directory:
            for backend in ("MPI", "NCCL"):
                log, path = self.fixture(directory, backend)
                self.assertEqual(len(validate_run(log, path, backend)["placements"]), 4)
                if backend == "NCCL":
                    with self.assertRaises(RuntimeError):
                        validate_run(log.replace("NCCLCommunicator", "Comm"), path, backend)

    def test_duplicate_gpu_missing_rank_and_nonfinite_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            log, path = self.fixture(directory, "MPI")
            for invalid in (log.replace("GPU-device3", "GPU-device0"),
                            log.replace("VALIDATION rank=3 fields_finite=true", ""),
                            log.replace("rank=3 fields_finite=true", "rank=3 fields_finite=false")):
                with self.assertRaises(RuntimeError):
                    validate_run(invalid, path, "MPI")
            path.write_text(path.read_text().replace("0.001", "NaN"))
            with self.assertRaises(RuntimeError):
                validate_run(log, path, "MPI")


if __name__ == "__main__":
    unittest.main()
