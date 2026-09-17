"""I retain concrete union payload types before checking ownership boundaries."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER_ROOT = Path(os.environ.get("NANOLANG_AFFINE_COMPILER_ROOT", ROOT / "bin"))
COMPILERS = os.environ.get("NANOLANG_AFFINE_COMPILERS", "nanoc_c,nanoc_stage1,nanoc_stage2").split(",")


class GenericAffineIdentity(unittest.TestCase):
    def check(self, source, accepted):
        for compiler in COMPILERS:
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix="nano-generic-affine-") as directory:
                work = Path(directory)
                program = work / "main.nano"
                output = work / "program"
                program.write_text(source)
                output.write_bytes(b"prior artifact")
                result = subprocess.run([str(COMPILER_ROOT / compiler), str(program), "-o", str(output)], cwd=ROOT, capture_output=True, text=True, timeout=120)
                messages = result.stdout + result.stderr
                if accepted:
                    self.assertEqual(result.returncode, 0, messages)
                    run = subprocess.run([str(output)], capture_output=True, text=True, timeout=10)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                else:
                    self.assertGreater(result.returncode, 0, messages)
                    self.assertRegex(messages, r"(?i)(ownership|resource-bearing)")
                    self.assertEqual(output.read_bytes(), b"prior artifact")

    def test_integer_payload_copy_and_match(self):
        self.check('''union Box<T> { Some { value: T }, None {} }
fn read(value: Box<int>) -> int {
    match value { Some(v) => { return v.value } None(n) => { return 0 } }
}
shadow read { let boxed: Box<int> = Box.Some { value: 7 } assert (== (read boxed) 7) }
fn main() -> int { let boxed: Box<int> = Box.Some { value: 7 } let copy: Box<int> = boxed return (- (+ (read copy) (read boxed)) 14) }
shadow main { assert (== (main) 0) }
''', True)

    def test_string_payload_copy_and_match(self):
        self.check('''union Box<T> { Some { value: T }, None {} }
fn read(value: Box<string>) -> string {
    match value { Some(v) => { return v.value } None(n) => { return "empty" } }
}
shadow read { let boxed: Box<string> = Box.Some { value: "kept" } assert (== (read boxed) "kept") }
fn main() -> int { let boxed: Box<string> = Box.Some { value: "kept" } let copy: Box<string> = boxed assert (== (read copy) "kept") assert (== (read boxed) "kept") return 0 }
shadow main { assert (== (main) 0) }
''', True)

    def test_resource_parameter_requires_lowering(self):
        self.check('''resource struct Handle { fd: int }
union Box<T> { Some { value: T }, None {} }
fn abandon(value: Box<Handle>) -> void { }
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
''', False)

    def test_nested_record_resource_parameter_requires_lowering(self):
        self.check('''resource struct Handle { fd: int }
struct Envelope { inner: Handle }
union Box<T> { Some { value: T }, None {} }
fn abandon(value: Box<Envelope>) -> void { }
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
''', False)

    def test_array_of_generic_resources_is_rejected(self):
        self.check('''resource struct Handle { fd: int }
union Box<T> { Some { value: T }, None {} }
fn abandon(values: array<Box<Handle>>) -> void { }
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
''', False)


if __name__ == "__main__":
    unittest.main()
