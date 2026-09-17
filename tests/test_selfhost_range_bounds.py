"""I preserve range bounds and loop scopes through both native compiler stages."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILERS = Path(os.environ.get("NANOLANG_RANGE_COMPILER_ROOT", ROOT / "bin"))


class SelfhostRangeBounds(unittest.TestCase):
    def test_order_scope_exits_and_temporary_names(self):
        source = ROOT / "tests/nanoisa/fixtures/native_range_bounds.nano"
        for compiler in ("nanoc_c", "nanoc_stage1", "nanoc_stage2"):
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix="nano-range-native-") as tmp:
                output = Path(tmp) / "program"
                result = subprocess.run([COMPILERS / compiler, source, "-o", output], cwd=ROOT,
                                        capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                result = subprocess.run([output], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_failed_bound_shadow_preserves_output(self):
        source = (ROOT / "tests/nanoisa/fixtures/native_range_bounds.nano").read_text()
        source = source.replace("assert (== order 12)", "assert (== order 13)")
        for compiler in ("nanoc_c", "nanoc_stage1", "nanoc_stage2"):
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix="nano-range-shadow-") as tmp:
                path, output = Path(tmp) / "input.nano", Path(tmp) / "program"
                path.write_text(source)
                output.write_text("previous accepted output")
                result = subprocess.run([COMPILERS / compiler, path, "-o", output], cwd=ROOT,
                                        capture_output=True, text=True, timeout=120)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("shadow", (result.stdout + result.stderr).lower())
                self.assertEqual(output.read_text(), "previous accepted output")


if __name__ == "__main__":
    unittest.main()
