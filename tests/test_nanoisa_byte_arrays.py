"""I execute the self-hosted emitter's byte-array storage and slice contracts."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests.test_byte_array_literals import BYTE_CONTEXTS

ROOT = Path(__file__).resolve().parents[1]

class NanoISAByteArrays(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix="nanoisa-byte-arrays-"))
        print("I retain byte-array qualification at", cls.work, flush=True)
        cls.sequence = 0
        cls.driver = Path(os.environ.get("NANOLANG_BYTE_DRIVER", cls.work / "driver"))
        if "NANOLANG_BYTE_DRIVER" not in os.environ:
            source = cls.work / "driver.nano"
            source.write_text((ROOT / "tests/nanoisa/fixtures/program_closure_driver.nano.txt").read_text())
            cls.run_checked(ROOT / "bin/nanoc_c", source, "-o", cls.driver)

    @classmethod
    def run_checked(cls, *args, success=True):
        cls.sequence += 1
        result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True, text=True, timeout=180)
        (cls.work / f"command-{cls.sequence}.log").write_text(repr(list(map(str,args))) + "\n" + result.stdout + result.stderr)
        if (result.returncode == 0) != success:
            raise AssertionError(f"{args}: {result.returncode}\n{result.stdout}\n{result.stderr}")
        return result

    def execute(self, name, source):
        path = self.work / (name + ".nano")
        path.write_text(source + " shadow main { assert true }\n")
        assembly = path.with_suffix(".nasm")
        assembly.write_text(self.run_checked(self.driver, path, "program").stdout)
        module = path.with_suffix(".nvm")
        self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", module)
        self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
        self.run_checked(ROOT / "bin/nano_vm", module)

    def test_contextual_destinations(self):
        for name, source in BYTE_CONTEXTS.items():
            with self.subTest(name=name):
                self.execute(name, source)

    def test_mutation_slice_and_outer_independence(self):
        self.execute("slice", """fn main()->int {
 let mut bytes:array<u8> = [300, 255]
 let value:int = 511
 set bytes (array_push bytes value)
 (array_set bytes 1 257)
 let copy:array<u8> = (array_slice bytes 0 3)
 (array_set bytes 0 99)
 assert (== (at copy 0) 44)
 assert (== (at copy 1) 1)
 assert (== (at copy 2) 255)
 assert (== (array_length copy) 3)
 assert (== (+ (at copy 0) (at copy 1)) 45)
 assert (< (at copy 0) (at copy 2))
 assert (> (at copy 2) 128)
 return 0
}""")

    def test_wrong_literal_elements_refused(self):
        for value in ("true", "1.5", '\"text\"'):
            with self.subTest(value=value):
                path = self.work / "wrong.nano"
                path.write_text("fn main()->int { let values:array<u8> = [" + value + "] return 0 }")
                self.run_checked(self.driver, path, "program", success=False)
