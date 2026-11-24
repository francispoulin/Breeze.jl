"""Contract checks for the Rorqual NVTX/Nsight study plan and launch command."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rorqual import nsight


class PlanTests(unittest.TestCase):
    def test_cases_cover_both_dynamics_on_1_4_8_gpus_with_512_cubed_cells_per_gpu(self):
        plan = nsight.build_plan()
        self.assertEqual({(case["dynamics"], case["gpus"]) for case in plan},
                         {(dynamics, gpus) for dynamics in ("anelastic", "compressible_explicit") for gpus in (1, 4, 8)})
        for case in plan:
            Nx, Ny, Nz = map(int, case["size"].split("x"))
            Rx, Ry = map(int, case["partition"].split("x"))
            self.assertEqual((Rx * Ry, Ry), (case["gpus"], 1))
            # 512³ cells per GPU; the four-GPU 4×1 partition gives 256×1024×512 local grids.
            self.assertEqual(Nx * Ny * Nz // case["gpus"], 512**3)
            self.assertEqual(case["backend"], "NCCL")
            self.assertEqual(case["timed_steps"], 10)

    def test_time_step_halves_with_each_doubling_of_resolution(self):
        for case in nsight.build_plan():
            finest = max(map(int, case["size"].split("x")))
            self.assertAlmostEqual(case["time_step"], 0.01 * 512 / finest)

    def test_each_multi_gpu_case_is_compared_with_its_one_gpu_case(self):
        pairs = nsight.comparisons(nsight.build_plan())
        self.assertEqual(len(pairs), 4)
        self.assertIn("Anelastic 1 vs 8 GPUs=anelastic_SERIAL_1g_1x1:anelastic_NCCL_8g_8x1", pairs)
        self.assertIn("CE 1 vs 4 GPUs=compressible_explicit_SERIAL_1g_1x1:compressible_explicit_NCCL_4g_4x1", pairs)


class CommandTests(unittest.TestCase):
    def test_eight_gpu_command_spans_two_nodes_and_profiles_each_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            (study / "config.json").write_text(json.dumps(dict(source="/frozen", environment="/environment",
                                                               raw="/scratch/raw", nsight_trace="cuda,nvtx,mpi")))
            case = next(case for case in nsight.build_plan() if case["gpus"] == 8 and case["dynamics"] == "anelastic")
            command = nsight.case_command(study, case)
        for argument in ("--nodes=2", "--ntasks=8", "--ntasks-per-node=4", "--gpus-per-node=h100:4",
                         "--capture-range=cudaProfilerApi", "--comm=NCCL", "--nvtx=true", "--profile=true",
                         "--size=4096x512x512", "--partition=8x1", "--time_step=0.00125", "--time_steps=10",
                         "--output=/scratch/raw/anelastic_NCCL_8g_8x1/rank_%q{SLURM_PROCID}/profile"):
            self.assertIn(argument, command)
        self.assertLess(command.index("nsys"), command.index("julia"))


if __name__ == "__main__":
    unittest.main()
