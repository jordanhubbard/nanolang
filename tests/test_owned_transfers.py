"""I retain verified transfer dataflow without publishing runtime execution."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OwnedTransfers(unittest.TestCase):
    def test_vm_native_refusal_preserves_output(self):
        with tempfile.TemporaryDirectory(prefix="nano-owned-transfer-") as tmp:
            artifact = Path(tmp) / "owned.nvm"
            output = Path(tmp) / "previous.c"
            generated = subprocess.run([ROOT / "obj/test_owned_transfers", artifact],
                                       cwd=ROOT, capture_output=True, timeout=90)
            self.assertEqual(generated.returncode, 0, generated.stdout + generated.stderr)
            output.write_text("previous output\n")
            for command in ([ROOT / "bin/nano_vm", artifact],
                            [ROOT / "bin/nvm2c", artifact, "-o", output]):
                refused = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=90)
                self.assertNotEqual(refused.returncode, 0, refused.stdout + refused.stderr)
                self.assertIn(b"ownership instruction", refused.stdout + refused.stderr)
            self.assertEqual(output.read_text(), "previous output\n")


if __name__ == "__main__":
    unittest.main()
