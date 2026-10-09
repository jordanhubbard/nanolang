"""I exercise the actual module ABI, copied views and independent ownership."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FileSourceInputs(unittest.TestCase):
    def checked(self, command):
        result = subprocess.run(list(map(str, command)), cwd=ROOT,
                                capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_native_and_vm_module_consumers(self):
        with tempfile.TemporaryDirectory(prefix="nano-input-bridge-") as directory:
            work = Path(directory)
            origin = work / "binding.nano"
            origin.write_text("# original source identity\n")
            shutil.copyfile(ROOT / "tests/fixtures/nsi_file_plan.json", work / "interface.nsi.json")
            for compiler in os.environ.get("NANO_INPUT_COMPILERS", "nanoc_c,nano_virt,nanoc_stage1,nanoc_stage2").split(","):
                with self.subTest(compiler=compiler):
                    native = compiler == "nanoc_c"
                    product = work / (compiler + (".exe" if native else ".nvm"))
                    command = [ROOT / "bin" / compiler, ROOT / "tests/file_source_inputs.nano", "-o", product]
                    if not native:
                        command.append("--emit-nvm")
                    self.checked(command)
                    if native:
                        output = self.checked([product, origin])
                    else:
                        self.checked([ROOT / "bin/nano_vm", "--verify-only", product])
                        output = self.checked([ROOT / "bin/nano_vm", product, "--", origin])
                    self.assertIn("PASS source input bridge", output)


if __name__ == "__main__":
    unittest.main()
