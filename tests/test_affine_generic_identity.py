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
    def check(self, source, accepted, modules=None):
        for compiler in COMPILERS:
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix="nano-generic-affine-") as directory:
                work = Path(directory)
                program = work / "main.nano"
                output = work / "program"
                for name, contents in (modules or {}).items():
                    (work / name).write_text(contents)
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

    def test_formal_shadows_resource_record(self):
        self.check('''resource struct T { fd: int }
union Box<T> { Some { value: T }, None {} }
fn close_record(value: T) -> int { let T { fd } = value return fd }
shadow close_record { assert (== (close_record T { fd: 3 }) 3) }
fn read(value: Box<int>) -> int {
    match value { Some(v) => { return v.value } None(n) => { return 0 } }
}
shadow read { let boxed: Box<int> = Box.Some { value: 7 } assert (== (read boxed) 7) }
fn main() -> int { let boxed: Box<int> = Box.Some { value: 7 } return (- (read boxed) 7) }
shadow main { assert (== (main) 0) }
''', True)

    def test_guarded_first_success_preserves_lexical_payloads(self):
        self.check('''resource struct T { fd: int }
union Box<T> { Some { value: T }, None {} }
fn close_record(value: T) -> int { let T { fd } = value return fd }
shadow close_record { assert (== (close_record T { fd: 3 }) 3) }
fn identity(value: Box<int>) -> Box<int> { return value }
shadow identity { let boxed: Box<int> = Box.Some { value: 7 } let selected: Box<int> = (identity boxed) match selected { Some(v) => { assert (== v.value 7) } None(n) => { assert false } } }
fn choose(value: Box<int>, first: bool) -> int {
    let v: int = 40
    let selected: Box<int> = (identity value)
    match selected {
        Some(v) if first => { return (+ v.value 1) }
        Some(v) if true => { return v.value }
        None(n) => { return v }
    }
    return 0
}
shadow choose { let boxed: Box<int> = Box.Some { value: 7 } assert (== (choose boxed false) 7) assert (== (choose boxed true) 8) }
fn main() -> int { let boxed: Box<int> = Box.Some { value: 7 } return (- (+ (choose boxed false) (close_record T { fd: 3 })) 10) }
shadow main { assert (== (main) 0) }
''', True)

    def test_concrete_same_named_resource_still_rejected(self):
        self.check('''resource struct T { fd: int }
union Box<T> { Some { value: T }, None {} }
fn abandon(value: Box<T>) -> void { }
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
''', False)

    def test_formal_survives_module_record_binding(self):
        modules = {
            "generic.nano": '''resource struct T { fd: int }
union Box<T> { Some { value: T }, None {} }
fn read(value: Box<int>) -> int {
    match value { Some(v) => { return v.value } None(n) => { return 0 } }
}
shadow read { let boxed: Box<int> = Box.Some { value: 7 } assert (== (read boxed) 7) }
pub fn result() -> int { let boxed: Box<int> = Box.Some { value: 7 } return (read boxed) }
shadow result { assert (== (result) 7) }
''',
            "plain.nano": '''struct T { value: int }
pub fn result() -> int { let value: T = T { value: 3 } let copy: T = value return (+ value.value copy.value) }
shadow result { assert (== (result) 6) }
''',
        }
        self.check('''module "generic.nano" as generic
module "plain.nano" as plain
fn main() -> int { return (- (+ (generic.result) (plain.result)) 13) }
shadow main { assert (== (main) 0) }
''', True, modules)

    def test_module_concrete_resource_still_rejected(self):
        modules = {
            "generic.nano": '''resource struct T { fd: int }
union Box<T> { Some { value: T }, None {} }
fn abandon(value: Box<T>) -> void { }
pub fn result() -> int { return 0 }
shadow result { assert (== (result) 0) }
''',
            "plain.nano": '''struct T { value: int }
pub fn result() -> int { return 0 }
shadow result { assert (== (result) 0) }
''',
        }
        self.check('''module "generic.nano" as generic
module "plain.nano" as plain
fn main() -> int { return (+ (generic.result) (plain.result)) }
shadow main { assert (== (main) 0) }
''', False, modules)

    def test_ordinary_match_closes_outer_owner_in_each_arm(self):
        self.check('''resource struct Handle { fd: int }
union Box<T> { Some { value: T }, None {} }
fn close_handle(owner: Handle) -> int { let Handle { fd } = owner return fd }
shadow close_handle { assert (== (close_handle Handle { fd: 4 }) 4) }
fn choose(value: Box<int>, owner: Handle) -> int {
    match value {
        Some(v) => { return (+ v.value (close_handle owner)) }
        None(n) => { return (close_handle owner) }
    }
}
shadow choose { let some: Box<int> = Box.Some { value: 3 } let none: Box<int> = Box.None {} assert (== (choose some Handle { fd: 4 }) 7) assert (== (choose none Handle { fd: 4 }) 4) }
fn main() -> int { let some: Box<int> = Box.Some { value: 3 } return (- (choose some Handle { fd: 4 }) 7) }
shadow main { assert (== (main) 0) }
''', True)

    def test_ordinary_match_rejects_unresolved_return_arm(self):
        self.check('''resource struct Handle { fd: int }
union Box<T> { Some { value: T }, None {} }
fn close_handle(owner: Handle) -> int { let Handle { fd } = owner return fd }
shadow close_handle { assert (== (close_handle Handle { fd: 4 }) 4) }
fn choose(value: Box<int>, owner: Handle) -> int {
    match value { Some(v) => { return (close_handle owner) } None(n) => { return 0 } }
}
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
''', False)

    def test_ordinary_match_rejects_disagreeing_join(self):
        self.check('''resource struct Handle { fd: int }
union Box<T> { Some { value: T }, None {} }
fn close_handle(owner: Handle) -> int { let Handle { fd } = owner return fd }
shadow close_handle { assert (== (close_handle Handle { fd: 4 }) 4) }
fn choose(value: Box<int>, owner: Handle) -> int {
    match value { Some(v) => { let used: int = (close_handle owner) } None(n) => { } }
    return (close_handle owner)
}
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
''', False)

    def test_inline_resource_payload_match_stays_rejected(self):
        self.check('''resource struct Handle { fd: int }
union Choice { Some { value: Handle }, None {} }
fn inspect(owner: Handle) -> int {
    match Choice.Some { value: owner } { Some(v) => { return v.value.fd } None(n) => { return 0 } }
}
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
''', False)

    def test_string_payload_copy_and_match(self):
        self.check('''union Box<T> { Some { value: T }, None {} }
fn read(value: Box<string>) -> string {
    match value { Some(v) => { return v.value } None(n) => { return "empty" } }
}
shadow read { let boxed: Box<string> = Box.Some { value: "kept" } assert (== (read boxed) "kept") }
fn main() -> int { let boxed: Box<string> = Box.Some { value: "kept" } let copy: Box<string> = boxed assert (== (read copy) "kept") assert (== (read boxed) "kept") return 0 }
shadow main { assert (== (main) 0) }
''', True)

    def test_array_payload_copy_and_match(self):
        self.check('''union Box<T> { Some { value: T }, None {} }
fn read(value: Box<array<int>>) -> array<int> {
    match value { Some(v) => { return v.value } None(n) => { return [] } }
}
shadow read { let boxed: Box<array<int>> = Box.Some { value: [7, 8] } let result: array<int> = (read boxed) assert (== (at result 1) 8) }
fn main() -> int { let boxed: Box<array<int>> = Box.Some { value: [7, 8] } let copy: Box<array<int>> = boxed let result: array<int> = (read copy) let again: array<int> = (read boxed) return (- (+ (at result 0) (at again 1)) 15) }
shadow main { assert (== (main) 0) }
''', True)

    def test_second_payload_parameter(self):
        self.check('''union Choice<T, E> { Left { value: T }, Right { value: E } }
fn read(value: Choice<int,string>) -> string {
    match value { Left(v) => { return (int_to_string v.value) } Right(v) => { return v.value } }
}
shadow read { let left: Choice<int,string> = Choice.Left { value: 7 } let right: Choice<int,string> = Choice.Right { value: "kept" } assert (== (read left) "7") assert (== (read right) "kept") }
fn main() -> int { let right: Choice<int,string> = Choice.Right { value: "kept" } assert (== (read right) "kept") return 0 }
shadow main { assert (== (main) 0) }
''', True)

    def test_second_resource_parameter_requires_lowering(self):
        self.check('''resource struct Handle { fd: int }
union Choice<T, E> { Left { value: T }, Right { value: E } }
fn abandon(value: Choice<int,Handle>) -> void { }
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
''', False)

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
