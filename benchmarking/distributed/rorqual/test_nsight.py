"""Contract checks for the NVTX/Nsight study plans (Rorqual 1/4/8, Trillium 4/8) and launch commands."""
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


class TrilliumTests(unittest.TestCase):
    def test_trillium_series_compares_four_with_eight_gpus_on_the_nx1_grids(self):
        plan = nsight.build_plan("trillium_4_8")
        self.assertEqual({(case["id"], case["size"], case["time_step"]) for case in plan},
                         {(f"{dynamics}_NCCL_{gpus}g_{gpus}x1", size, time_step)
                          for dynamics in ("anelastic", "compressible_explicit")
                          for gpus, size, time_step in ((4, "2048x512x512", 0.0025), (8, "4096x512x512", 0.00125))})
        self.assertEqual(sorted(nsight.comparisons(plan)),
                         ["Anelastic 4 vs 8 GPUs=anelastic_NCCL_4g_4x1:anelastic_NCCL_8g_8x1",
                          "CE 4 vs 8 GPUs=compressible_explicit_NCCL_4g_4x1:compressible_explicit_NCCL_8g_8x1"])

    def test_trillium_allocations_take_whole_nodes_without_memory_requests(self):
        config = dict(machine="trillium", series="trillium_4_8", allocation_limit="01:30:00", runtime="/scratch/runtime")
        command = nsight.allocation_command(config, 8, "breeze_nsight_8g", Path("/scratch/study/allocations/8g"))
        for argument in ("--account=def-fpoulin", "--nodes=2", "--ntasks-per-node=4", "--gpus-per-node=h100:4",
                         "--chdir=/scratch/runtime"):
            self.assertIn(argument, command)
        self.assertFalse(any(argument.startswith("--mem") for argument in command))
        self.assertEqual(nsight.series_title(config), "Trillium NVTX Nsight: anelastic and CE on 4 and 8 GPUs (NCCL)")
        rorqual = dict(config, machine="rorqual", series="rorqual_1_4_8")
        self.assertIn("--mem=256G", nsight.allocation_command(rorqual, 8, "job", Path("/study")))
        self.assertEqual(nsight.series_title(rorqual), "Rorqual NVTX Nsight: anelastic and CE on 1, 4, and 8 GPUs (NCCL)")


if __name__ == "__main__":
    unittest.main()
