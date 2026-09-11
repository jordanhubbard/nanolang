"""I exercise reporting branches with fake compilers, not compiler semantics."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = '''#!/usr/bin/env python3
import pathlib, sys
out = pathlib.Path(sys.argv[sys.argv.index("-o") + 1])
out.write_text("#!/bin/sh\\nexit 0\\n")
out.chmod(0o755)
'''


class BootstrapMessages(unittest.TestCase):
    def run_branch(self, identical, deterministic=False):
        with tempfile.TemporaryDirectory(prefix="nanolang-bootstrap-messages-") as directory:
            root = Path(directory)
            shutil.copyfile(ROOT / "Makefile.gnu", root / "Makefile.gnu")
            (root / "bin").mkdir()
            (root / ".bootstrap2.built").touch()
            for name, suffix in [("nanoc_stage1", ""),
                                 ("nanoc_stage2", "" if identical else "# Different artifact\n")]:
                target = root / "bin" / name
                target.write_text(COMPILER + suffix)
                target.chmod(0o755)
            result = subprocess.run(
                [os.environ.get("MAKE_BIN", "make"), "-s", "-f", "Makefile.gnu",
                 "-o", ".bootstrap2.built", ".bootstrap3.built", "UNAME_S=Linux",
                 "VERIFY_SCRIPT=/usr/bin/true", f"BOOTSTRAP_TMPDIR={root}",
                 f"BOOTSTRAP_DETERMINISTIC={int(deterministic)}"],
                cwd=root, capture_output=True, text=True, timeout=30)
            return result.returncode, result.stdout + result.stderr, (root / ".bootstrap3.built").exists()

    def test_identical_artifacts_are_not_a_correctness_proof(self):
        status, output, stamped = self.run_branch(True)
        self.assertEqual(status, 0, output)
        self.assertTrue(stamped)
        self.assertIn("byte-identical in this build", output)
        self.assertIn("not established reproducibility across clean environments", output)
        self.assertNotIn("This proves reproducible builds", output)

    def test_different_artifacts_report_hypotheses(self):
        status, output, stamped = self.run_branch(False)
        self.assertEqual(status, 0, output)
        self.assertTrue(stamped)
        self.assertIn("have not diagnosed the difference", output)
        self.assertIn("not a correctness proof", output)
        self.assertIn("Canonical NanoISA artifact equality remains", output)
        self.assertNotIn("Both compilers work correctly", output)

    def test_deterministic_mode_still_rejects_differences(self):
        status, output, stamped = self.run_branch(False, deterministic=True)
        self.assertNotEqual(status, 0, output)
        self.assertFalse(stamped)
        self.assertIn("Expected identical binaries", output)


if __name__ == "__main__":
    unittest.main()
