"""I require a native compiler built from bytecode to compile a real program."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OneIrCompiler(unittest.TestCase):
    def run_checked(self, args, timeout=180):
        env = dict(os.environ, NANO_MODULE_PATH=str(ROOT / "modules"))
        process = subprocess.Popen([str(arg) for arg in args], cwd=ROOT, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate(timeout=10)
            self.fail(f"I exceeded {timeout}s running {args[0]}")
        self.assertEqual(process.returncode, 0,
                         f"I failed {args[0]}\n" + (stdout + stderr).decode(errors="replace")[-6000:])
        return stdout

    def test_compiler_bytecode_to_native_to_program(self):
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        with tempfile.TemporaryDirectory(prefix="nano-one-ir-compiler-") as tmp:
            work = Path(tmp)
            module, source, compiler = work / "compiler.nvm", work / "compiler.c", work / "compiler"
            self.run_checked([ROOT / "bin/nano_virt", ROOT / "src_nano/nanoc_v06.nano",
                              "--emit-nvm", "--strip-debug", "-o", module], timeout=600)
            self.assertGreater(module.stat().st_size, 0)
            self.run_checked([ROOT / "bin/nvm2c", module, "-o", source], timeout=240)
            self.assertNotIn("nano_vm", source.read_text())
            self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-O0",
                              source, "-o", compiler], timeout=240)
            help_output = self.run_checked([compiler, "--help"], timeout=10)
            self.assertIn(b"Compiler", help_output)
            hello = work / "hello"
            self.run_checked([compiler, ROOT / "examples/language/nl_hello.nano", "-o", hello])
            self.assertEqual(self.run_checked([hello], timeout=10), b"Hello from NanoLang!\n")


if __name__ == "__main__":
    unittest.main()
