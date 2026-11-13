"""The queue must never treat an occupied GPU as free, even at zero utilization."""

import hashlib
from pathlib import Path
import tempfile
import unittest

from wait_for_gpus import can_launch, verify_sources


class GPUAvailabilityTests(unittest.TestCase):
    inventory = "0, 39000, 0\n1, 39000, 0\n2, 39000, 0\n3, 39000, 0\n"

    def test_four_free_gpus(self):
        self.assertTrue(can_launch("", self.inventory))

    def test_idle_process_still_blocks_launch(self):
        self.assertFalse(can_launch("1234\n", self.inventory))

    def test_missing_gpu_blocks_launch(self):
        self.assertFalse(can_launch("", "0, 39000, 0\n1, 39000, 0\n2, 39000, 0\n"))

    def test_insufficient_memory_blocks_launch(self):
        self.assertFalse(can_launch("", self.inventory.replace("3, 39000", "3, 4096")))

    def test_invalid_inventory_blocks_launch(self):
        self.assertFalse(can_launch("", self.inventory.replace("3, 39000", "3, unknown")))

    def test_changed_or_missing_source_blocks_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "driver.jl"
            path.write_text("original\n")
            hashes = {"driver.jl": hashlib.sha256(path.read_bytes()).hexdigest()}
            verify_sources(root, hashes)
            path.write_text("changed\n")
            with self.assertRaises(RuntimeError):
                verify_sources(root, hashes)
            path.unlink()
            with self.assertRaises(RuntimeError):
                verify_sources(root, hashes)


if __name__ == "__main__":
    unittest.main()
