"""I keep C-seed nested-array literals aligned with my VM representation."""
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/unit/test_cseed_nested_array_literals.nano"
NATIVE_REFILL = """struct Box { rows: array<array<string>> }
fn main() -> int {
    let box: Box = Box { rows: [["before"]] }
    (array_set box.rows 0 [])
    let refilled: array<string> = (array_push (at box.rows 0) "after")
    assert (str_equals (at refilled 0) "after")
    return 0
}
shadow main { assert (== (main) 0) }
"""


class CSeedNestedArrayLiterals(unittest.TestCase):
    def run_checked(self, command):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_native_and_vm_match(self):
        with tempfile.TemporaryDirectory(prefix="nano-cseed-nested-array-") as tmp:
            directory = Path(tmp)
            native = directory / "native"
            module = directory / "program.nvm"

            self.run_checked([ROOT / "bin/nanoc_c", SOURCE, "-o", native])
            native_run = self.run_checked([native])

            self.run_checked([ROOT / "bin/nano_virt", SOURCE, "--emit-nvm", "-o", module])
            vm_run = self.run_checked([ROOT / "bin/nano_vm", module])

            self.assertEqual(native_run.stdout, b"nested-array-literals-ok\n")
            self.assertEqual(vm_run.stdout, native_run.stdout)

    def test_native_empty_replacement_preserves_inner_tag(self):
        with tempfile.TemporaryDirectory(prefix="nano-cseed-nested-empty-") as tmp:
            directory = Path(tmp)
            source = directory / "refill.nano"
            native = directory / "native"
            source.write_text(NATIVE_REFILL)
            self.run_checked([ROOT / "bin/nanoc_c", source, "-o", native])
            self.run_checked([native])


if __name__ == "__main__":
    unittest.main()
