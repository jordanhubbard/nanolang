"""I roundtrip and execute the admitted standalone transfer contract."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OwnedTransfers(unittest.TestCase):
    def test_roundtrip_executes_vm_and_native(self):
        with tempfile.TemporaryDirectory(prefix="nano-owned-transfer-") as tmp:
            artifact = Path(tmp) / "owned.nvm"
            output = Path(tmp) / "owned.c"
            binary = Path(tmp) / "owned"
            generated = subprocess.run([ROOT / "obj/test_owned_transfers", artifact],
                                       cwd=ROOT, capture_output=True, timeout=90)
            self.assertEqual(generated.returncode, 0, generated.stdout + generated.stderr)
            result = subprocess.run([ROOT / "bin/nano_vm", artifact], capture_output=True, timeout=90)
            self.assertEqual(result.returncode, 42, result.stdout + result.stderr)
            subprocess.run([ROOT / "bin/nvm2c", artifact, "-o", output], check=True, capture_output=True)
            subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", output, "-o", binary],
                           check=True, capture_output=True)
            self.assertEqual(subprocess.run([binary], timeout=90).returncode, 42)


if __name__ == "__main__":
    unittest.main()
