"""I exercise the production coverage loop with known compiler outcomes."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class VmExampleReporting(unittest.TestCase):
    def test_exit_status_and_modern_diagnostics(self):
        source = (ROOT / "tests/test_vm_examples_coverage.sh").read_text()
        loop = source.split('skipped_list=""', 1)[1].split('if [ "$skipped" -ne 0 ]; then', 1)[0]
        with tempfile.TemporaryDirectory(prefix="nano-example-reporting-") as tmp:
            directory = Path(tmp)
            compiler = directory / "compiler"
            compiler.write_text('#!/bin/sh\ncase "$1" in\n'
                                '*/silent.nano) exit 7;;\n'
                                '*/structured.nano) echo "-- E024 UNDEFINED VARIABLE" >&2; exit 42;;\n'
                                '*/legacy.nano) echo "error: legacy rejection" >&2; exit 3;;\n'
                                '*/passed.nano) exit 0;;\nesac\n')
            compiler.chmod(0o700)
            (directory / "eligible.txt").write_text("silent.nano\nstructured.nano\nlegacy.nano\npassed.nano\n")
            result = subprocess.run(["bash", "-c", 'skipped_list=""\n' + loop +
                                     '\nprintf "count=%s%s\\n" "$skipped" "$skipped_list"'],
                                    env=dict(os.environ, REPO_ROOT=tmp, VM_COMPILER="compiler",
                                             EXAMPLES_DIR=tmp, work_dir=tmp),
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("count=3", result.stdout)
            self.assertIn("silent.nano: nano_virt exited 7 with no diagnostic", result.stdout)
            self.assertIn("structured.nano: exit 42: -- E024 UNDEFINED VARIABLE", result.stdout)
            self.assertIn("legacy.nano: exit 3: legacy rejection", result.stdout)
            self.assertNotIn("passed.nano:", result.stdout)


if __name__ == "__main__":
    unittest.main()
