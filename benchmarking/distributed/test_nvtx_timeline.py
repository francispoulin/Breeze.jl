"""Regression checks for asynchronous NVTX-to-CUDA projection."""

from pathlib import Path
import sqlite3
import tempfile
import unittest

from render_nvtx_timeline import read_timeline


class TimelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Path(self.directory.name) / "profile.sqlite"
        with sqlite3.connect(self.database) as connection:
            connection.executescript("""
                CREATE TABLE StringIds (id INTEGER, value TEXT);
                INSERT INTO StringIds VALUES (1, 'TimeSteppers/time_step!');
                INSERT INTO StringIds VALUES (2, 'call/solve!');
                INSERT INTO StringIds VALUES (3, 'call/fill_halo_regions!');
                INSERT INTO StringIds VALUES (4, 'pressure_kernel');
                CREATE TABLE NVTX_EVENTS
                    (start INTEGER, end INTEGER, globalTid INTEGER, textId INTEGER,
                     text TEXT, eventType INTEGER);
                INSERT INTO NVTX_EVENTS VALUES (100, 200, 7, 1, NULL, 60);
                INSERT INTO NVTX_EVENTS VALUES (110, 140, 7, 2, NULL, 60);
                INSERT INTO NVTX_EVENTS VALUES (145, 155, 7, 3, NULL, 60);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME
                    (start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER);
                INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (115, 120, 7, 42);
                INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (125, 130, 8, 43);
                INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (160, 170, 7, 44);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL
                    (start INTEGER, end INTEGER, correlationId INTEGER, shortName INTEGER,
                     deviceId INTEGER, streamId INTEGER);
                INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (300, 400, 42, 4, 0, 1);
                INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (410, 420, 43, 4, 0, 2);
                CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY
                    (start INTEGER, end INTEGER, correlationId INTEGER, bytes INTEGER,
                     deviceId INTEGER, streamId INTEGER);
                INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (430, 450, 44, 64, 0, 1);
            """)

    def test_async_projection_nesting_threads_and_copies(self):
        data = read_timeline(self.database)
        outer = data["projections"]["TimeSteppers/time_step!"]
        self.assertEqual(outer, dict(instances=1, total_ns=150, gpu_operations=2))
        self.assertEqual(data["projections"]["call/solve!"],
                         dict(instances=1, total_ns=100, gpu_operations=1))
        self.assertNotIn("call/fill_halo_regions!", data["projections"])
        self.assertEqual(data["first_step"], [0.0002, 0.00035])
        self.assertEqual(data["extent"], [0, 0.00035])
        self.assertEqual((data["kernels"], data["copies"]), (2, 1))
        self.assertEqual(len(data["lanes"]), 7)

    def test_reject_crossing_ranges(self):
        with sqlite3.connect(self.database) as connection:
            connection.execute("UPDATE NVTX_EVENTS SET end=210 WHERE textId=2")
        with self.assertRaisesRegex(ValueError, "non-nested"):
            read_timeline(self.database)

    def test_read_only_missing_database(self):
        missing = self.database.with_name("missing.sqlite")
        with self.assertRaises(sqlite3.OperationalError):
            read_timeline(missing)
        self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
