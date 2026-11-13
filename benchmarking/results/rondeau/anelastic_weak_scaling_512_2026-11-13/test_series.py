"""CPU checks of weak-scaling normalization, GPU selection and distributed result validation."""

import csv
from pathlib import Path
import tempfile
import unittest

from plot_efficiency import weak_efficiencies
from run_series import available_gpus, command_for, predecessor_finished, validate_run, verify_sources


def measurement(ranks, seconds=1):
    return dict(machine="rondeau", dynamics="anelastic", float="Float32", Nx=str(512*ranks),
                Ny="512", Nz="512", Rx=str(ranks), Ry="1", gpus=str(ranks), steps="500",
                seconds_per_step=str(seconds), points_per_second="1", max_abs_w="0.001",
                communication="SERIAL" if ranks == 1 else "MPI", fields_finite="true")


class WeakScalingChecks(unittest.TestCase):
    def test_weak_efficiency_has_no_gpu_count_divisor(self):
        rows = weak_efficiencies([measurement(1, 1), measurement(2, 2), measurement(4, 4)])
        self.assertEqual([r["weak_efficiency_percent"] for r in rows], [100, 50, 25])
        rows = weak_efficiencies([measurement(n, 1) for n in [1, 2, 4]])
        self.assertEqual([r["weak_efficiency_percent"] for r in rows], [100, 100, 100])

    def test_rejects_strong_scaling_missing_points_and_nonfinite_fields(self):
        rows = [measurement(n) for n in [1, 2, 4]]
        for invalid in [rows[:2], [rows[0], rows[0], rows[2]],
                        [rows[0], dict(rows[1], Nx="512"), rows[2]],
                        [rows[0], dict(rows[1], fields_finite="false"), rows[2]],
                        [rows[0], dict(rows[1], seconds_per_step="nan"), rows[2]]]:
            with self.assertRaises(ValueError):
                weak_efficiencies(invalid)

    def test_all_three_commands_preserve_local_cube_and_disable_profiling(self):
        for ranks in [1, 2, 4]:
            command = command_for(Path("/u/ajtse/source"), Path("/u/ajtse/results"),
                                Path("/u/ajtse/runtime"), {"julia": "julia"}, ranks)
            for option in ["--dynamics=anelastic", f"--size={512*ranks}x512x512", f"--partition={ranks}x1",
                           "--time_steps=500", "--float_type=Float32", "--profile=false", "--nvtx=false"]:
                self.assertIn(option, command)

    def test_idle_contexts_low_memory_and_malformed_status_block_selection(self):
        inventory = "\n".join(f"{i}, GPU-id{i}, 39000" for i in range(4))
        self.assertEqual(available_gpus(inventory, ""), [0, 1, 2, 3])
        self.assertEqual(available_gpus(inventory, "GPU-id0, 123\n"), [1, 2, 3])
        self.assertEqual(available_gpus(inventory.replace("0, GPU-id0, 39000", "0, GPU-id0, 29999"), ""), [1, 2, 3])
        self.assertEqual(available_gpus("bad inventory", ""), [])
        self.assertEqual(available_gpus(inventory, "unexpected response"), [])
        self.assertEqual(available_gpus(inventory, "GPU-unknown, 456"), [])
        self.assertEqual(available_gpus(inventory.replace("GPU-id1", "GPU-id0"), ""), [])

    def test_preceding_campaign_must_finish_before_launch(self):
        self.assertFalse(predecessor_finished({"status": "RUNNING"}))
        self.assertFalse(predecessor_finished({"status": "WAITING_FOR_FOUR_FREE_GPUS"}))
        for state in ["COMPLETE", "FAILED", "CANCELLED", "RESULTS_READY_PUSH_FAILED"]:
            self.assertTrue(predecessor_finished({"status": state}))

    def test_validation_requires_finite_rank_results_unique_devices_and_expected_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.csv"
            for ranks in [1, 2, 4]:
                row = measurement(ranks)
                with path.open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=row.keys(), lineterminator="\n")
                    writer.writeheader()
                    writer.writerow(row)
                communicator = "none" if ranks == 1 else "Comm"
                log = "\n".join(f"PLACEMENT rank={i} host=rondeau gpu_uuid=GPU-id{i} communicator={communicator}\n"
                              f"VALIDATION rank={i} fields_finite=true" for i in range(ranks))
                self.assertEqual(validate_run(log,path,ranks)["measurement"]["gpus"],str(ranks))
                with self.assertRaises(RuntimeError):
                    validate_run(log.replace("fields_finite=true", "fields_finite=false", 1),path,ranks)
                if ranks > 1:
                    with self.assertRaises(RuntimeError):
                        validate_run(log.replace("GPU-id1", "GPU-id0"),path,ranks)

    def test_modified_frozen_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jl"
            path.write_text("changed\n")
            with self.assertRaises(RuntimeError):
                verify_sources(Path(directory),{"source.jl":"incorrect digest"})


if __name__ == "__main__":
    unittest.main()
