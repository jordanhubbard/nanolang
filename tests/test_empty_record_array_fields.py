"""I check empty record-field array representations on both backends."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class EmptyRecordArrayFields(unittest.TestCase):
    def test_compile_shadows_and_execute(self):
        for compiler, bytecode in (("nanoc_c", False), ("nano_virt", True)):
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / ("fields.nvm" if bytecode else "fields")
                command = [str(ROOT / "bin" / compiler),
                           "tests/unit/test_empty_record_array_fields.nano", "-o", str(output)]
                if bytecode:
                    command.append("--emit-nvm")
                result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=40)
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
                command = [str(ROOT / "bin/nano_vm"), str(output)] if bytecode else [str(output)]
                result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))


if __name__ == "__main__":
    unittest.main()
