"""I execute side-effecting call arguments in source order across backends."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
NANOC = Path(os.environ.get("NANOLANG_COMPILER", ROOT / "bin/nanoc")).resolve()
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
    return 0
}

shadow main { assert true }
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

    def _run(self, command, directory):
        result = subprocess.run(command, cwd=directory, capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
