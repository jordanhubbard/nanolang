"""I execute side-effecting call arguments in source order across backends."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
NANOC = Path(os.environ.get("NANOLANG_COMPILER", ROOT / "bin/nanoc_c")).resolve()
NANO = Path(os.environ.get("NANOLANG_INTERPRETER", ROOT / "bin/nano")).resolve()
NANO_VIRT = Path(os.environ.get("NANOLANG_VM_COMPILER", ROOT / "bin/nano_virt")).resolve()


MODULE = """pub fn combine(a: int, b: int, c: int) -> int {
    return (+ (+ (* a 100) (* b 10)) c)
}

shadow combine {
    assert (== (combine 1 2 3) 123)
}
"""

PROGRAM = """module "order_helper.nano" as helper

let mut trace: int = 0

fn first() -> int {
    set trace (+ (* trace 10) 1)
    return 1
}

shadow first { assert (== (first) 1) }

fn second() -> int {
    set trace (+ (* trace 10) 2)
    return 2
}

shadow second { assert (== (second) 2) }

fn third() -> int {
    set trace (+ (* trace 10) 3)
    return 3
}

shadow third { assert (== (third) 3) }

fn local_combine(a: int, b: int, c: int) -> int {
    return (+ (+ (* a 100) (* b 10)) c)
}

shadow local_combine { assert (== (local_combine 1 2 3) 123) }

fn main() -> int {
    set trace 0
    assert (== (local_combine (first) (second) (third)) 123)
    assert (== trace 123)
    set trace 0
    assert (== (helper.combine (first) (second) (third)) 123)
    assert (== trace 123)
    set trace 0
    assert (== (local_combine trace (first) (second)) 12)
    assert (== trace 12)
    set trace 0
    assert (== (helper.combine (local_combine (first) (second) (third)) (first) (second)) 12312)
    assert (== trace 12312)
    set trace 0
    assert (== (local_combine (helper.combine (first) (second) (third)) (first) (second)) 12312)
    assert (== trace 12312)
    return 0
}

shadow main { assert true }
"""


CALLEE_MUTATION = 'union Choice { Some { value: int }, None { } }\nfn first(x: int) -> int { return (+ x 100) }\nshadow first { assert (== (first 1) 101) }\nfn second(x: int) -> int { return (+ x 200) }\nshadow second { assert (== (second 1) 201) }\nfn main() -> int {\n let mut target: fn(int) -> int = first\n let choice: Choice = Choice.Some { value: 1 }\n let result: int = (target (match choice { Some(item) => { set target second 1 } None(empty) => { 0 } }))\n assert (== result 101)\n assert (== (target 1) 201)\n return 0\n}\nshadow main { assert true }\n'

OPAQUE_NULL = """opaque type LocalHandle

fn is_null(value: LocalHandle) -> bool {
    return (== value 0)
}

shadow is_null { assert (is_null 0) }

fn main() -> int {
    assert (is_null 0)
    return 0
}

shadow main { assert (== (main) 0) }
"""

class NativeCallArgumentOrder(unittest.TestCase):
    def test_interpreter_vm_and_native_agree(self):
        with tempfile.TemporaryDirectory(prefix="nano-call-order-") as tmp:
            directory = Path(tmp)
            source = directory / "main.nano"
            source.write_text(PROGRAM)
            (directory / "order_helper.nano").write_text(MODULE)

            self._run([str(NANO), str(source)], directory)
            self._run([str(NANO_VIRT), str(source), "--run"], directory)

            executable = directory / "native"
            self._run([str(NANOC), str(source), "-o", str(executable)], directory)
            self._run([str(executable)], directory)

    def test_native_captures_mutable_callee_before_argument(self):
        with tempfile.TemporaryDirectory(prefix="nano-callee-order-") as tmp:
            directory = Path(tmp)
            source = directory / "callee.nano"
            source.write_text(CALLEE_MUTATION)
            executable = directory / "native"
            self._run([str(NANOC), str(source), "-o", str(executable)], directory)
            self._run([str(executable)], directory)

    def test_native_function_variable_shadows_global_function(self):
        program = CALLEE_MUTATION.replace(
            "fn main() -> int {",
            "fn target(x: int) -> int { return (+ x 900) }\n"
            "shadow target { assert (== (target 1) 901) }\nfn main() -> int {")
        with tempfile.TemporaryDirectory(prefix="nano-shadow-callee-") as tmp:
            directory = Path(tmp)
            source = directory / "shadow.nano"
            source.write_text(program)
            executable = directory / "native"
            self._run([str(NANOC), str(source), "-o", str(executable)], directory)
            self._run([str(executable)], directory)

    def test_native_temporaries_do_not_shadow_source_variables(self):
        names = [f"__nl_arg_{i}_0" for i in range(32)] + [f"__nl_callee_{i}" for i in range(32)]
        declarations = "\n".join(f"let {name}: int = {i + 1}" for i, name in enumerate(names))
        total = names[0]
        for name in names[1:]:
            total = f"(+ {total} {name})"
        program = ("fn keep(a: int, b: int) -> int { return b }\nshadow keep { assert true }\n"
                   "fn main() -> int {\n" + declarations +
                   f"\nassert (== (keep (keep 1 2) {total}) 2080)\nreturn 0\n}}\n"
                   "shadow main { assert true }\n")
        with tempfile.TemporaryDirectory(prefix="nano-call-names-") as tmp:
            directory = Path(tmp)
            source = directory / "names.nano"
            source.write_text(program)
            executable = directory / "native"
            self._run([str(NANOC), str(source), "-o", str(executable)], directory)
            self._run([str(executable)], directory)

    def test_native_snapshot_restores_opaque_null_pointer_type(self):
        with tempfile.TemporaryDirectory(prefix="nano-opaque-null-") as tmp:
            directory = Path(tmp)
            source = directory / "opaque_null.nano"
            source.write_text(OPAQUE_NULL)
            executable = directory / "native"
            self._run([str(NANOC), str(source), "--keep-c", "-o", str(executable)], directory)
            self._run([str(executable)], directory)
            generated = executable.with_suffix(".c").read_text()
            self.assertRegex(generated, r"is_null\(\(void\*\)__nl_arg_\d+_0\)")

    def test_native_rejects_nonzero_integer_for_opaque_parameter(self):
        with tempfile.TemporaryDirectory(prefix="nano-opaque-nonzero-") as tmp:
            directory = Path(tmp)
            source = directory / "opaque_nonzero.nano"
            source.write_text(OPAQUE_NULL.replace("assert (is_null 0)\n    return 0",
                                                  "assert (is_null 1)\n    return 0"))
            executable = directory / "preserved"
            executable.write_bytes(b"preserve")
            result = subprocess.run([str(NANOC), str(source), "-o", str(executable)],
                                    cwd=directory, capture_output=True, timeout=90)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(executable.read_bytes(), b"preserve")
            self.assertIn(b"or 0 (null)", result.stderr)

    def _run(self, command, directory):
        result = subprocess.run(command, cwd=directory, capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
