"""I exercise the actual module ABI, copied views and independent ownership."""
from pathlib import Path
import os
import shlex
import shutil
import sys
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
NVM2C = Path(os.environ.get("NANO_INPUT_NVM2C", str(ROOT / "bin/nvm2c")))


class FileSourceInputs(unittest.TestCase):
    def checked(self, command):
        result = subprocess.run(list(map(str, command)), cwd=ROOT,
                                capture_output=True, text=True, timeout=180,
                                env={**os.environ, "ASAN_OPTIONS": "detect_leaks=1"})
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
                        source = product.with_suffix(".c")
                        binary = product.with_suffix(".aot")
                        self.checked([NVM2C, product, "-o", source])
                        cc = shlex.split(os.environ.get("NANO_NATIVE_TEST_CC", "cc"))
                        libraries = ["-ldl"] if sys.platform.startswith("linux") else []
                        self.checked([*cc, "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                                      "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                                      source, "-o", binary, "-lm", *libraries])
                        self.assertEqual(self.checked([binary, origin]), output)
                    self.assertIn("PASS source input bridge", output)


    def test_aot_refuses_wrong_context_signatures(self):
        with tempfile.TemporaryDirectory(prefix="nano-input-abi-") as directory:
            work = Path(directory)
            for name in ("new", "valid", "count", "open", "text", "free"):
                with self.subTest(function=name):
                    assembly, module, output = (work / item for item in ("bad.nasm", "bad.nvm", "bad.c"))
                    assembly.write_text(
                        f'.import "/missing/never-opened.so" "nl_source_inputs_{name}" bool\n'
                        '.import_kind 0 artifact\n.entry main\n.function main 0 0 0 int 1\n'
                        'CALL_EXTERN 0\nPOP\nPUSH_I64 0\nRET\n.end\n')
                    self.checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                    output.write_text("prior output")
                    result = subprocess.run([NVM2C, module, "-o", output],
                                            capture_output=True, text=True, timeout=60)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("requires an exact library binding", result.stderr)
                    self.assertEqual(output.read_text(), "prior output")


if __name__ == "__main__":
    unittest.main()
