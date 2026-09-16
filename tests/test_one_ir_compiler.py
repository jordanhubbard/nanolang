"""I require a native compiler built from bytecode to compile a real program."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
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

    def test_declared_empty_array_returns_reach_native(self):
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        for element_type, value in (("int", "42"), ("string", '"answer"'),
                                    ("Point", "Point { x: 42 }")):
            with self.subTest(element_type=element_type), tempfile.TemporaryDirectory(prefix="nano-empty-return-") as tmp:
                work = Path(tmp)
                source, module, native_c, binary = (work / name for name in ("input.nano", "input.nvm", "input.c", "input"))
                source.write_text(
                    "struct Point { x: int }\n"
                    f"fn make(empty: bool) -> array<{element_type}> {{\n"
                    "  if empty { return [] }\n"
                    f"  return [{value}]\n}}\n"
                    "shadow make { assert (== (array_length (make true)) 0) }\n"
                    "fn main() -> int {\n"
                    "  assert (== (array_length (make true)) 0)\n"
                    "  assert (== (array_length (make false)) 1)\n"
                    "  return 0\n}\n"
                    "shadow main { assert (== (main) 0) }\n"
                )
                self.run_checked([ROOT / "bin/nano_virt", source, "--emit-nvm", "-o", module])
                self.run_checked([ROOT / "bin/nvm2c", module, "-o", native_c])
                self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", native_c, "-o", binary])
                self.run_checked([binary])

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
                              source, "-o", compiler,
                              *(["-ldl"] if sys.platform.startswith("linux") else [])], timeout=240)
            help_output = self.run_checked([compiler, "--help"], timeout=10)
            self.assertIn(b"Compiler", help_output)
            hello = work / "hello"
            self.run_checked([compiler, ROOT / "examples/language/nl_hello.nano", "-o", hello])
            self.assertEqual(self.run_checked([hello], timeout=10), b"Hello from NanoLang!\n")

    def test_projected_optional_arguments(self):
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        fixture = (ROOT / "tests/nanoisa/fixtures/nested_optional_returns.nasm").read_text()
        header, functions = fixture.split(".function main", 1)
        producers = ".function choose" + functions.split(".function choose", 1)[1]
        for reverse in (False, True):
            for tail in (False, True):
                for incompatible in (False, True):
                    with self.subTest(reverse=reverse, tail=tail, incompatible=incompatible), tempfile.TemporaryDirectory(prefix="nano-projected-arg-") as tmp:
                        work = Path(tmp)
                        assembly, module, source, binary = (work / name for name in ("input.nasm", "input.nvm", "input.c", "input"))
                        main = (
                            ".function main 0 0 0 int 1\n"
                            "PUSH_STR text\nCALL absent\nBOOL_NOT\nASSERT\n"
                            "LOAD_GLOBAL 0\nCALL absent\nASSERT\n"
                            "CALL present\nCALL project\nBOOL_NOT\nASSERT\n"
                            "PUSH_BOOL 1\nCALL choose\nCALL project\nBOOL_NOT\nASSERT\n"
                            "CALL missing\nCALL project\nASSERT\nPUSH_I64 0\nRET\n.end\n"
                        )
                        helpers = (
                            ".function absent 1 1 0 bool 1\nLOAD_LOCAL 0\nTYPE_CHECK 0\nRET\n.end\n"
                            ".function project 1 1 0 bool 1\nLOAD_LOCAL 0\nAGG_GET 0\nAGG_GET 0\n"
                            + ("TAIL_CALL absent\n" if tail else "CALL absent\nRET\n") + ".end\n"
                        )
                        text = header + main + helpers + producers
                        if incompatible:
                            text = text.replace(".function present 0 0 0 struct 1\n  PUSH_STR text",
                                                ".function present 0 0 0 struct 1\n  PUSH_I64 42")
                        if reverse:
                            prefix, *blocks = text.split(".function ")
                            text = prefix + "".join(".function " + block for block in reversed(blocks))
                        assembly.write_text(text)
                        self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                        if incompatible:
                            result = subprocess.run([ROOT / "bin/nvm2c", module, "-o", source], capture_output=True, timeout=30)
                            self.assertNotEqual(result.returncode, 0)
                            self.assertIn(b"shape", result.stderr)
                        else:
                            self.run_checked([ROOT / "bin/nano_vm", module])
                            self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
                            self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary])
                            self.run_checked([binary])

    def test_record_local_storage_joins(self):
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        fixture = (ROOT / "tests/nanoisa/fixtures/nested_optional_returns.nasm").read_text()
        header, functions = fixture.split(".function main", 1)
        producers = ".function choose" + functions.split(".function choose", 1)[1]
        for reverse in (False, True):
            for branch in (None, False, True):
                for incompatible in (False, True):
                    with self.subTest(reverse=reverse, branch=branch, incompatible=incompatible), tempfile.TemporaryDirectory(prefix="nano-local-join-") as tmp:
                        work = Path(tmp)
                        assembly, module, source, binary = (work / name for name in ("input.nasm", "input.nvm", "input.c", "input"))
                        present = "CALL present"
                        missing = "PUSH_BOOL 0\nCALL choose"
                        if incompatible:
                            present = "PUSH_I64 42\nAGG_PACK 0 0 0 1\nAGG_PACK 0 1 0 1"
                        first, second = (missing, present) if reverse else (present, missing)
                        body = first + "\nSTORE_LOCAL 0\n"
                        if branch is not None:
                            body += f"PUSH_BOOL {int(branch)}\nJMP_FALSE done\n"
                        body += second + "\nSTORE_LOCAL 0\ndone:\n"
                        body += "LOAD_LOCAL 0\nAGG_GET 0\nAGG_GET 0\n"
                        ends_present = reverse if branch is not False else not reverse
                        body += "PUSH_STR text\nEQ\nASSERT\n" if ends_present else "TYPE_CHECK 0\nASSERT\n"
                        # A destination join must not rewrite the producer's string.
                        body += "CALL present\nAGG_GET 0\nAGG_GET 0\nPUSH_STR text\nEQ\nASSERT\nPUSH_I64 0\nRET\n"
                        main = ".function main 0 1 0 int 1\n" + body + ".end\n"
                        assembly.write_text(header + (producers + main if reverse else main + producers))
                        self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                        if incompatible:
                            result = subprocess.run([ROOT / "bin/nvm2c", module, "-o", source], capture_output=True, timeout=30)
                            self.assertNotEqual(result.returncode, 0)
                            self.assertIn(b"shape", result.stderr)
                        else:
                            self.run_checked([ROOT / "bin/nano_vm", module])
                            self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
                            self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary])
                            self.run_checked([binary])

    def test_nested_optional_returns_reach_native(self):
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        fixture = (ROOT / "tests/nanoisa/fixtures/nested_optional_returns.nasm").read_text()
        for tail in (False, True):
            for reverse in (False, True):
                with self.subTest(tail=tail, reverse=reverse), tempfile.TemporaryDirectory(prefix="nano-nested-optional-") as tmp:
                    work = Path(tmp)
                    assembly, module, source, binary = (work / name for name in ("input.nasm", "input.nvm", "input.c", "input"))
                    text = fixture
                    if not tail:
                        text = text.replace("TAIL_CALL present", "CALL present\n  RET")
                        text = text.replace("TAIL_CALL missing", "CALL missing\n  RET")
                    if reverse:
                        header, *functions = text.split(".function ")
                        text = header + "".join(".function " + block for block in reversed(functions))
                    assembly.write_text(text)
                    self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                    self.run_checked([ROOT / "bin/nano_vm", module])
                    self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
                    self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary])
                    self.run_checked([binary])


if __name__ == "__main__":
    unittest.main()
