"""I retain reference declarations without admitting unimplemented execution."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OwnershipContracts(unittest.TestCase):
    def run_checked(self, command):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_transport_execution_and_reference_refusal(self):
        with tempfile.TemporaryDirectory(prefix="nano-ownership-contracts-") as tmp:
            ordinary = Path(tmp) / "ordinary.nvm"
            borrowed = Path(tmp) / "borrowed.nvm"
            owned_union = Path(tmp) / "owned-union.nvm"
            generated = Path(tmp) / "ordinary.c"
            native = Path(tmp) / "ordinary"
            self.run_checked([ROOT / "obj/test_ownership_contracts", ordinary, borrowed, owned_union])
            self.assertEqual(self.run_checked([ROOT / "bin/nano_vm", ordinary]).stdout, b"42\n")
            self.run_checked([ROOT / "bin/nvm2c", ordinary, "-o", generated])
            self.run_checked([os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra",
                              "-Werror", generated, "-lm", "-o", native])
            self.assertEqual(self.run_checked([native]).stdout, b"42\n")
            generated.write_text("previous output\n")
            for command in ([ROOT / "bin/nano_vm", borrowed],
                            [ROOT / "bin/nvm2c", borrowed, "-o", generated]):
                result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=90)
                self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(b"reference lifetime and ownership", result.stdout + result.stderr)
            self.assertEqual(generated.read_text(), "previous output\n")
            for command, diagnostic in (
                    ([ROOT / "bin/nano_vm", owned_union], b"explicit owned entry execution"),
                    ([ROOT / "bin/nvm2c", owned_union, "-o", generated],
                     b"reference lifetime and ownership instruction verification")):
                result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=90)
                self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(diagnostic, result.stdout + result.stderr)
            self.assertEqual(generated.read_text(), "previous output\n")


if __name__ == "__main__":
    unittest.main()
