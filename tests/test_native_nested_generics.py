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

    def test_selected_array_rejects_different_record(self):
        baseline = (ROOT / "tests/unit/test_native_generic_contexts.nano").read_text()
        source = "struct Other { value: int }\n" + baseline.replace(
            "let grown: array<Plain> = (array_push payload.values Plain { value: 16 })",
            "let grown: array<Other> = (array_push payload.values Other { value: 16 })")
        with tempfile.TemporaryDirectory() as tmp:
            path, output = Path(tmp) / "wrong.nano", Path(tmp) / "prior"
            path.write_text(source)
            for compiler in ("nanoc_c", "nano_virt"):
                with self.subTest(compiler=compiler):
                    output.write_text("prior artifact")
                    command = [ROOT / "bin" / compiler, path, "-o", output]
                    if compiler == "nano_virt": command.append("--emit-nvm")
                    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("declared nominal record type", result.stderr)
                    self.assertEqual(output.read_text(), "prior artifact")

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
