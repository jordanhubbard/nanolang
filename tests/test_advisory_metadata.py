"""I execute an advisory-bearing artifact through the real VM and C translator."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]

class AdvisoryMetadata(unittest.TestCase):
    def test_public_artifact_and_native_execution(self):
        with tempfile.TemporaryDirectory(prefix="nano-advisory-") as name:
            prefix = Path(name) / "case"
            def run(argv, expected=0):
                p = subprocess.run(argv, capture_output=True, text=True, timeout=60, cwd=ROOT)
                self.assertEqual(p.returncode, expected, p.stdout + p.stderr)
                return p
            run([ROOT / "obj/test_advisory_metadata", prefix])
            run([ROOT / "bin/nano_vm", "--verify-only", prefix.with_suffix(".nvm")])
            run([ROOT / "bin/nano_vm", prefix.with_suffix(".nvm")], 42)
            rebuilt = Path(name) / "rebuilt.c"
            run([ROOT / "bin/nvm2c", prefix.with_suffix(".nvm"), "-o", rebuilt])
            self.assertEqual(rebuilt.read_bytes(), prefix.with_suffix(".c").read_bytes())
            run(shlex.split(os.environ.get("CC", "cc")) + ["-std=c11", "-Wall", "-Wextra", "-Werror",
                "-fsanitize=address,undefined", "-fno-omit-frame-pointer", str(rebuilt), "-lm", "-o", str(prefix)])
            run([prefix], 42)

if __name__ == "__main__":
    unittest.main()
