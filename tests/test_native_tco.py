"""I require native TCO to preserve results, not merely change an AST."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get("NANO_TCO_COMPILER", ROOT / "bin/nanoc_c")).resolve()


class NativeTCO(unittest.TestCase):
    def compile_run(self, source, *, optimize):
        with tempfile.TemporaryDirectory(prefix="nano-native-tco-") as tmp:
            path = Path(tmp) / "input.nano"
            path.write_text(source)
            output = Path(tmp) / "program"
            command = [str(COMPILER), *(["--tco"] if optimize else []),
                       str(path), "-o", str(output)]
            compiled = subprocess.run(command, cwd=ROOT, capture_output=True,
                                      text=True, timeout=60)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            executed = subprocess.run([str(output)], cwd=ROOT, capture_output=True,
                                      text=True, timeout=15)
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)
            return executed.stdout

    def check_both(self, source):
        baseline = self.compile_run(source, optimize=False)
        self.assertEqual(self.compile_run(source, optimize=True), baseline)

    def test_argument_swap(self):
        self.check_both('''
fn rotate(n: int, a: int, b: int) -> int {
    if (== n 0) { return (+ (* a 10) b) }
    return (rotate (- n 1) b a)
}
shadow rotate { assert (== (rotate 3 2 7) 72) }
fn main() -> int { assert (== (rotate 4 2 7) 27) return 0 }
shadow main { assert (== (main) 0) }
''')

    def test_early_return(self):
        self.check_both('''
fn descend(n: int) -> int {
    if (== n 2) { return 42 }
    if (== n 0) { return -1 }
    return (descend (- n 1))
}
shadow descend { assert (== (descend 5) 42) assert (== (descend 0) -1) }
fn main() -> int { assert (== (descend 10) 42) return 0 }
shadow main { assert (== (main) 0) }
''')

    def test_string_result(self):
        self.check_both('''
fn repeat(n: int, text: string) -> string {
    if (== n 0) { return text }
    return (repeat (- n 1) (str_concat text "x"))
}
shadow repeat { assert (== (repeat 3 "a") "axxx") }
fn main() -> int { assert (== (repeat 2 "b") "bxx") return 0 }
shadow main { assert (== (main) 0) }
''')

    def test_float_result(self):
        self.check_both('''
fn step(n: int, value: float) -> float {
    if (== n 0) { return value }
    return (step (- n 1) (+ value 0.5))
}
shadow step { assert (== (step 3 1.0) 2.5) }
fn main() -> int { assert (== (step 2 1.0) 2.0) return 0 }
shadow main { assert (== (main) 0) }
''')

    def test_generated_name_collision(self):
        self.check_both('''
fn collide(n: int, __tco_n: int) -> int {
    let __tco_result: int = 40
    if (== n 0) { return (+ __tco_result __tco_n) }
    return (collide (- n 1) (+ __tco_n 1))
}
shadow collide { assert (== (collide 2 0) 42) }
fn main() -> int { assert (== (collide 3 0) 43) return 0 }
shadow main { assert (== (main) 0) }
''')

    def test_million_tail_calls(self):
        self.assertEqual(self.compile_run((ROOT / "tests/tco_test.nano").read_text(),
                                         optimize=True),
                         "I completed one million tail calls.\n")


if __name__ == "__main__":
    unittest.main()
