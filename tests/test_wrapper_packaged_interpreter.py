"""I keep wrapper_gen explicitly a packaged interpreter, never native AOT."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
NANO_VIRT = ROOT / "bin" / "nano_virt"


class WrapperPackagedInterpreter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="nano-wrapper-package-")
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.env = os.environ.copy()
        for key in ("NANO_CC", "CC"):
            self.env.pop(key, None)
        self.env["NANO_VIRT_LIB"] = str(ROOT / "obj")

    def run_nano_virt(self, args):
        return subprocess.run([str(NANO_VIRT), *args], cwd=ROOT, env=self.env,
                              capture_output=True, text=True, timeout=120)

    def program(self):
        source = self.directory / "program.nano"
        source.write_text("fn main() -> int { return 42 }\n"
                          "shadow main { assert (== (main) 42) }\n")
        return source

    def test_usage_names_packaged_interpreter_and_not_native(self):
        result = self.run_nano_virt([])
        self.assertNotEqual(result.returncode, 0)
        text = result.stdout + result.stderr
        # I bind the default -o path to wrapper_gen and to the packaged
        # interpreter, and I point native AOT at nvm2c instead.
        self.assertIn("Packaged interpreter", text)
        self.assertIn("wrapper_gen", text)
        self.assertIn("nvm2c", text)
        self.assertNotIn("native executable", text.lower())

    def test_default_output_is_a_packaged_interpreter(self):
        output = self.directory / "program"
        result = self.run_nano_virt(["-v", str(self.program()), "-o", str(output)])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        text = (result.stdout + result.stderr).lower()
        self.assertIn("packaged interpreter", text)
        self.assertNotIn("native executable", text)
        executed = subprocess.run([str(output)], cwd=self.directory, env=self.env,
                                  capture_output=True, text=True, timeout=10)
        self.assertEqual(executed.returncode, 42, executed.stdout + executed.stderr)
        # A packaged interpreter links the VM rather than translating to C.
        self.assertIn(b"vm_execute", output.read_bytes())

    def test_documentation_keeps_wrapper_gen_packaged_only(self):
        nanoisa = (ROOT / "docs" / "NANOISA.md").read_text()
        self.assertIn("Packaged interpreter only (wrapper_gen embeds `nano_vm`)", nanoisa)
        self.assertIn("`wrapper_gen` is never native AOT", nanoisa)
        self.assertIn("`bin/nvm2c`", nanoisa)
        nanoisa_only = (ROOT / "docs" / "NANOISA_ONLY.md").read_text()
        self.assertIn("That is a **packaged interpreter**.", nanoisa_only)
        self.assertIn("It is not a native\nbinary in the 5.1 sense.", nanoisa_only)


if __name__ == "__main__":
    unittest.main()
