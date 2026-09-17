"""I preserve layout facts while executing one ordinary nested-record artifact."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RetainedLayouts(unittest.TestCase):
    def test_roundtrip_and_same_artifact_execution(self):
        with tempfile.TemporaryDirectory(prefix="nano-retained-layouts-") as tmp:
            module = Path(tmp) / "layouts.nvm"
            generated = Path(tmp) / "layouts.c"
            native = Path(tmp) / "layouts"
            commands = [
                ([ROOT / "obj/test_retained_layouts", module], None),
                ([ROOT / "bin/nano_vm", module], b"42\n"),
                ([ROOT / "bin/nvm2c", module, "-o", generated], None),
                ([os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra",
                  "-Werror", generated, "-lm", "-o", native], None),
                ([native], b"42\n"),
            ]
            for command, expected in commands:
                with self.subTest(command=str(command[0])):
                    result = subprocess.run(command, cwd=ROOT, capture_output=True,
                                            timeout=90, check=False)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    if expected is not None:
                        self.assertEqual(result.stdout, expected)


if __name__ == "__main__":
    unittest.main()
