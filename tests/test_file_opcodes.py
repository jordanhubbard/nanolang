"""I qualify File encoding and checked refusal; I dispatch no service."""
try:
    from tests.sanitizer_options import asan_options
except ModuleNotFoundError:
    from sanitizer_options import asan_options
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FileOpcodes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix="nano-file-opcodes-"))
        print(f"I retain File encoding artifacts at {cls.artifacts}", flush=True)
        cls.compiler = shlex.split(os.environ.get("NANO_FILE_OPCODE_TEST_CC", "cc"))
        cls.flags = ["-std=c11", "-D_DEFAULT_SOURCE", "-g", "-O1", "-Wall", "-Wextra",
                     "-Werror", "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                     "-Isrc", "-Isrc/nanoisa", "-Imodules/nanoisa"]
        cls.flags += shlex.split(os.environ.get("NANO_FILE_OPCODE_TEST_CFLAGS", ""))
        cls.objects = shlex.split(os.environ["FILE_OPCODE_OBJECTS"])
        cls.linkflags = shlex.split(os.environ.get("FILE_OPCODE_LDFLAGS", ""))

    def command(self, name, command, *, expected=0, extra=None):
        env = dict(os.environ, ASAN_OPTIONS=asan_options("halt_on_error=1"),
                   UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
        env.update(extra or {})
        (self.artifacts / f"{name}-command.txt").write_text(shlex.join(command) + "\n")
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
        (self.artifacts / f"{name}.log").write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, expected, (command, result.stdout, result.stderr))
        return result.stdout

    def qualify(self, instrumented):
        name = "instrumented" if instrumented else "linked"
        objects = list(self.objects)
        if instrumented:
            for directory, source in (("nanoisa", "isa"), ("nanoisa", "nvm_format"),
                    ("nanoisa", "service_bindings_module"), ("nanoisa", "disassembler"),
                    ("nanovm", "vm_decode"), ("nanovirt", "wrapper_gen")):
                old = next(p for p in objects if p.endswith(f"/{directory}/{source}.o"))
                objects.remove(old)
                output = self.artifacts / f"{source}-instrumented.o"
                self.command(f"{source}-compile", [*self.compiler, *self.flags,
                    "-c", f"src/{directory}/{source}.c", "-o", str(output)])
                objects.append(str(output))
        exe = self.artifacts / name
        self.command(f"{name}-build", [*self.compiler, *self.flags,
            "tests/nanoisa/test_file_opcodes.c", *objects, *self.linkflags, "-o", str(exe)])
        wire = self.artifacts / f"{name}.nvm"
        wrapper_dir = self.artifacts / f"{name}-wrapper"
        wrapper_dir.mkdir()
        destination = wrapper_dir / "preserved"
        output = self.command(f"{name}-run", [str(exe)], extra={
            "FILE_OPCODE_WIRE": str(wire), "FILE_OPCODE_WRAPPER_OUTPUT": str(destination)})
        self.assertIn("File encoding/refusal checks; no service handler or CFG certificate", output)
        self.assertEqual(list(wrapper_dir.iterdir()), [destination])
        self.assertEqual(destination.read_bytes(), b"preserved")
        print(output.strip(), flush=True)
        if not instrumented:
            commands = [
                ("facts", [str(ROOT / "bin/nanoisa_hl_facts"), str(wire)], False),
                ("native-c", [str(ROOT / "bin/nvm2c"), str(wire)], True),
                ("llvm", [str(ROOT / "bin/nvm2llvm"), str(wire)], True),
                ("wasm", ["python3", "scripts/nvm2wasm.py", str(wire)], True),
                ("recover-c", ["python3", "scripts/nvm2hl.py", "--language", "c", str(wire)], True),
                ("recover-nano", ["python3", "scripts/nvm2hl.py", "--language", "nano", str(wire)], True),
            ]
            for label, command, has_output in commands:
                destination = self.artifacts / f"{label}-preserved"
                destination.write_bytes(b"prior output\0")
                if has_output:
                    command = [*command, "-o", str(destination)]
                stdout = self.command(f"cli-{label}", command, expected=1,
                    extra={"NANO_NVM2LLVM": str(ROOT / "bin/nvm2llvm")})
                self.assertEqual(stdout, "")
                self.assertEqual(destination.read_bytes(), b"prior output\0")

    def test_linked_encoding_transport_and_refusal(self):
        self.qualify(False)

    def test_instrumented_encoding_transport_and_refusal(self):
        self.qualify(True)
