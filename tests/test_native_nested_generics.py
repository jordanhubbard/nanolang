"""I retain nested union specializations through both native compilers."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class NativeNestedGenerics(unittest.TestCase):
    def test_nested_union_copy_parameter_and_payloads(self):
        self.check_fixture("test_native_nested_generics.nano")

    def test_constructor_contexts_preserve_values(self):
        self.check_fixture("test_native_generic_contexts.nano", ("nanoc_c",))

    def check_fixture(self, name, compilers=("nanoc_c", "nanoc_stage1", "nanoc_stage2")):
        source = ROOT / "tests/unit" / name
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "program"
            for compiler in compilers:
                with self.subTest(compiler=compiler):
                    result = subprocess.run([ROOT / "bin" / compiler, source, "-o", output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    result = subprocess.run([output], capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

if __name__ == "__main__": unittest.main()
