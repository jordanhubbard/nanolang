"""I retain nested union specializations through both native compilers."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class NativeNestedGenerics(unittest.TestCase):
    def test_nested_union_copy_parameter_and_payloads(self):
        source = ROOT / "tests/unit/test_native_nested_generics.nano"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "program"
            for compiler in ("nanoc_c", "nanoc_stage1", "nanoc_stage2"):
                with self.subTest(compiler=compiler):
                    result = subprocess.run([ROOT / "bin" / compiler, source, "-o", output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    result = subprocess.run([output], capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

if __name__ == "__main__": unittest.main()
