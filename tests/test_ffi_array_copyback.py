"""I test native array copy-back and allocation failures with real heap code."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FfiArrayCopyback(unittest.TestCase):
    def test_frame(self):
        with tempfile.TemporaryDirectory(prefix="nano-ffi-array-") as tmp:
            output = Path(tmp) / "frame"
            built = subprocess.run([*shlex.split(os.environ.get("NANO_NATIVE_TEST_CC") or os.environ.get("CC", "cc")),
                                    "-std=c99", "-g", "-Wall", "-Wextra", "-Werror", "-Isrc",
                                    "tests/nanovm/test_ffi_arrays.c", "src/nanovm/heap.c",
                                    "src/nanovm/value.c", "src/nanovm/heap_cycles.c", "src/nanoisa/isa.c",
                                    "src/runtime/dyn_array.c", "src/runtime/gc.c",
                                    "src/runtime/gc_struct.c", "src/utf8.c", "-lm", "-o", str(output)],
                                   cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            result = subprocess.run([str(output)], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for phase in (built, result):
                for marker in ("AddressSanitizer", "LeakSanitizer", "runtime error:"):
                    self.assertNotIn(marker, phase.stdout + phase.stderr)


if __name__ == "__main__":
    unittest.main()
