"""I qualify slice lowering through the actual emitter, VM and native translator."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile
import unittest
from tests.test_selfhost_native_array_slice import CASES

ROOT = Path(__file__).resolve().parents[1]

class NanoISASlice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix="nanoisa-slice-"))
        print("I retain slice qualification at", cls.work, flush=True)
        cls.sequence = 0
        cls.driver = Path(os.environ.get("NANOLANG_SLICE_DRIVER", str(cls.work / "driver")))
        if "NANOLANG_SLICE_DRIVER" not in os.environ:
            source = cls.work / "driver.nano"
            source.write_text((ROOT / "tests/nanoisa/fixtures/program_closure_driver.nano.txt").read_text())
            cls.run_checked(ROOT / "bin/nanoc_c", source, "-o", cls.driver)
        cls.cc = shlex.split(os.environ.get("CC", "clang"))

    @classmethod
    def run_checked(cls, *args):
        cls.sequence += 1
        command = list(map(str, args))
        p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=900)
        (cls.work / f"command-{cls.sequence}.log").write_text(repr(command)+"\n"+p.stdout+"\n"+p.stderr)
        if p.returncode:
            raise AssertionError(f"{command}: {p.returncode}\n{p.stdout}\n{p.stderr}")
        return p

    def qualify(self, name, text):
        source = self.work / (name + ".nano")
        source.write_text(text)
        for producer in ("seed", "selfhost"):
            with self.subTest(case=name, producer=producer):
                module = self.work / (name + "-" + producer + ".nvm")
                if producer == "seed":
                    self.run_checked(ROOT / "bin/nano_virt", source, "--emit-nvm", "-o", module)
                else:
                    assembly = module.with_suffix(".nasm")
                    assembly.write_text(self.run_checked(self.driver, source, "program").stdout)
                    self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", module)
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.run_checked(ROOT / "bin/nano_vm", module)
                native_c, native = module.with_suffix(".c"), module.with_suffix(".native")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", native_c)
                self.run_checked(*self.cc, "-std=c11", "-Wall", "-Wextra", "-Werror",
                                 "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                                 native_c, "-lm", "-o", native)
                self.run_checked(native)

    def test_source_shapes(self):
        for name, body in CASES.items():
            declaration = "struct Item { value: int }\n" if name == "record" else ""
            self.qualify(name, declaration + "fn main() -> int {\n" + body +
                         "\nreturn 0\n}\nshadow main { assert (== (main) 0) }\n")

    def test_bounds_bits_and_record_children(self):
        self.qualify("bits", (ROOT / "tests/nanoisa/fixtures/array_slice_bits.nano.txt").read_text())

    def test_byte_slice(self):
        self.qualify("bytes", """fn main() -> int {
 let mut bytes: array<u8> = []
 set bytes (array_push bytes 255)
 assert (== (at (array_slice bytes 0 1) 0) 255)
 return 0
}
shadow main { assert (== (main) 0) }
""")

    def test_closure_environment_copy(self):
        self.qualify("closures", """fn make(n:int)->fn()->int { return fn()->int { return n } }
shadow make { let f:fn()->int = (make 7) assert (== (f) 7) }
fn main()->int {
 let values:array<fn()->int> = [(make 7), (make 9)]
 let copy:array<fn()->int> = (array_slice values 0 2)
 (array_set values 0 (make 1))
 let first:fn()->int = (at copy 0)
 let second:fn()->int = (at copy 1)
 assert (== (first) 7)
 assert (== (second) 9)
 return 0
}
shadow main { assert (== (main) 0) }
""")

    def test_argument_order_once(self):
        self.qualify("order", """let mut trace: int = 0
fn values() -> array<int> { set trace (+ (* trace 10) 1) return [7, 8, 9] }
shadow values { set trace 0 assert (== (array_length (values)) 3) assert (== trace 1) }
fn start() -> int { set trace (+ (* trace 10) 2) return 1 }
shadow start { set trace 0 assert (== (start) 1) assert (== trace 2) }
fn length() -> int { set trace (+ (* trace 10) 3) return 9223372036854775807 }
shadow length { set trace 0 assert (> (length) 0) assert (== trace 3) }
fn main() -> int {
 set trace 0
 let copy: array<int> = (array_slice (values) (start) (length))
 assert (== trace 123)
 assert (== (array_length copy) 2)
 assert (== (at copy 0) 8)
 return 0
}
shadow main { assert (== (main) 0) }
""")

    def test_unchecked_emitter_refuses_invalid_operands(self):
        for expression in ("(array_slice [1] 0)", "(array_slice [1] true 1)",
                           "(array_slice [1] 0 false)", '(array_slice "x" 0 1)'):
            with self.subTest(expression=expression):
                source = self.work / "invalid.nano"
                source.write_text("fn main() -> int { let a: array<int> = " + expression + " return 0 }")
                p = subprocess.run([str(self.driver), str(source), "program"], cwd=ROOT,
                                   capture_output=True, text=True, timeout=120)
                self.assertEqual(p.returncode, 1, p.stderr)
                self.assertEqual(p.stdout, "")

if __name__ == "__main__":
    unittest.main()
