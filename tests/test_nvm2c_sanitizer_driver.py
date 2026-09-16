"""I check sanitizer output isolation and failure propagation."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_nvm2c_sanitizers as driver


class SanitizerDriver(unittest.TestCase):
    def test_each_run_uses_fresh_owned_outputs(self):
        roots = []

        def build(command, **kwargs):
            values = dict(arg.split("=", 1) for arg in command if "=" in arg)
            obj = Path(values["OBJ_DIR"])
            root = obj.parent
            roots.append(root)
            self.assertTrue(root.is_dir())
            self.assertFalse(obj.exists())
            self.assertEqual(Path(values["BIN_DIR"]), root / "bin")
            self.assertEqual(Path(values["NVM2C_TEST_BINARY"]), root / "bin/test_nvm2c")
            self.assertIn(driver.FLAGS, values["CC"])
            self.assertEqual(command[-1], "test-nvm2c")
            self.assertEqual(kwargs["cwd"], driver.ROOT)
            return SimpleNamespace(returncode=0)

        with patch.object(driver.subprocess, "run", side_effect=build), \
             patch.object(driver.subprocess, "check_output", return_value="___asan_init\n___ubsan_handle_type_mismatch_v1\n") as nm:
            self.assertEqual(driver.run_sanitizers("make", "cc"), 0)
            self.assertEqual(driver.run_sanitizers("make", "cc"), 0)
            self.assertEqual(nm.call_count, 4)
        self.assertNotEqual(roots[0], roots[1])
        self.assertTrue(all(not root.exists() for root in roots))

    def test_build_failure_is_not_hidden(self):
        with patch.object(driver.subprocess, "run", return_value=SimpleNamespace(returncode=23)), \
             patch.object(driver.subprocess, "check_output") as nm:
            self.assertEqual(driver.run_sanitizers("make", "cc"), 23)
            nm.assert_not_called()

    def test_missing_instrumentation_fails(self):
        for symbols in ("", "__asan_init", "__ubsan_handle_type_mismatch_v1"):
            with self.subTest(symbols=symbols), \
                 patch.object(driver.subprocess, "run", return_value=SimpleNamespace(returncode=0)), \
                 patch.object(driver.subprocess, "check_output", return_value=symbols):
                self.assertEqual(driver.run_sanitizers("make", "cc"), 1)


if __name__ == "__main__":
    unittest.main()
