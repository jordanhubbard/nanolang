"""I retain floating values and unary provenance across native compiler stages."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILERS = Path(os.environ.get("NANOLANG_FLOAT_COMPILER_ROOT", ROOT / "bin"))


class SelfhostFloatValues(unittest.TestCase):
    def test_precision_unary_zero_and_binary_subtraction(self):
        source = ROOT / "tests/nanoisa/fixtures/scalar_floats.nano"
        for compiler in ("nanoc_c", "nanoc_stage1", "nanoc_stage2"):
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix="nano-native-float-") as tmp:
                output = Path(tmp) / "program"
                result = subprocess.run([COMPILERS / compiler, source, "-o", output], cwd=ROOT,
                                        capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                result = subprocess.run([output], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_exact_format_precision_and_suffix(self):
        source = ROOT / "tests/nanoisa/fixtures/float_format.nano"
        expected_output = None
        for compiler in ("nanoc_c", "nanoc_stage1", "nanoc_stage2"):
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix="nano-float-format-") as tmp:
                output = Path(tmp) / "program"
                result = subprocess.run([COMPILERS / compiler, source, "-o", output], cwd=ROOT,
                                        capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                result = subprocess.run([output], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                if expected_output is None:
                    expected_output = result.stdout
                self.assertEqual(result.stdout, expected_output)


if __name__ == "__main__":
    unittest.main()
