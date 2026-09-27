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
    let __tco_0_p0: int = 0
    let __tco_1_a0: int = 0
    if (== n 0) { return (+ __tco_result __tco_n) }
    return (collide (- n 1) (+ __tco_n 1))
}
shadow collide { assert (== (collide 2 0) 42) }
fn main() -> int { assert (== (collide 3 0) 43) return 0 }
shadow main { assert (== (main) 0) }
''')

    def test_void_fallthrough(self):
        self.check_both('''
fn descend(n: int) -> void {
    if (> n 0) { return (descend (- n 1)) }
}
shadow descend { (descend 3) }
fn main() -> int { (descend 10) return 0 }
shadow main { assert (== (main) 0) }
''')

    def test_array_parameter_metadata_and_deep_recursion(self):
        source = '''
fn rotate(n: int, left: array<int>, right: array<int>) -> int {
    if (== n 0) {
        return (+ (* (array_get left 0) 10) (array_get right 0))
    }
    return (rotate (- n 1) right left)
}
shadow rotate {
    assert (== (rotate 3 [2] [7]) 72)
}
fn main() -> int {
    assert (== (rotate 200000 [2] [7]) 27)
    return 0
}
shadow main { assert (== (main) 0) }
'''
        self.assertEqual(self.compile_run(source, optimize=True), "")

    def test_record_and_tuple_parameters(self):
        self.check_both('''
struct Pair { left: int, right: int }
fn records(n: int, first: Pair, second: Pair) -> int {
    if (== n 0) { return (+ (* first.left 10) second.right) }
    return (records (- n 1) second first)
}
fn tuples(n: int, first: (int, int), second: (int, int)) -> int {
    if (== n 0) { return (+ (* first.0 10) second.1) }
    return (tuples (- n 1) second first)
}
shadow records {
    assert (== (records 1 Pair { left: 2, right: 3 }
                              Pair { left: 7, right: 8 }) 73)
}
shadow tuples { assert (== (tuples 1 (2, 3) (7, 8)) 73) }
fn main() -> int {
    assert (== (records 2 Pair { left: 2, right: 3 }
                              Pair { left: 7, right: 8 }) 28)
    assert (== (tuples 2 (2, 3) (7, 8)) 28)
    return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_reused_record_tuple_scalar_callback_parameter_names(self):
        self.check_both('''
struct Earlier { value: int }
fn record(value: Earlier) -> int { return value.value }
shadow record { assert (== (record Earlier { value: 6 }) 6) }
fn pair(value: (int, int)) -> int { return (+ value.0 value.1) }
shadow pair { assert (== (pair (2, 3)) 5) }
fn scalar(value: int) -> int { return (+ value 1) }
shadow scalar { assert (== (scalar 4) 5) }
fn callback(value: fn(int) -> int) -> int { return (value 8) }
shadow callback { assert (== (callback scalar) 9) }
fn main() -> int {
    assert (== (record Earlier { value: 7 }) 7)
    assert (== (pair (3, 4)) 7)
    assert (== (scalar 6) 7)
    assert (== (callback scalar) 9)
    return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_function_parameter_and_callable_expression(self):
        self.check_both('''
fn increment(value: int) -> int { return (+ value 1) }
shadow increment { assert (== (increment 3) 4) }
fn double(value: int) -> int { return (* value 2) }
shadow double { assert (== (double 3) 6) }
fn apply_tail(n: int, operation: fn(int) -> int,
              next_operation: fn(int) -> int, value: int) -> int {
    if (== n 0) { return value }
    return (apply_tail (- n 1) next_operation operation (operation value))
}
shadow apply_tail { assert (== (apply_tail 4 increment double 1) 10) }
fn main() -> int {
    assert (== (apply_tail 20 increment double 1) 3070)
    return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_parameter_shadowing_is_lexical(self):
        self.check_both('''
fn shadowed(n: int, value: int) -> int {
    if (== n 0) {
        let outer_value: int = value
        let value: int = (+ outer_value 5)
        if true { let n: int = (+ value 1) assert (== n (+ value 1)) }
        return value
    }
    if true {
        let outer_value: int = value
        let value: int = (+ outer_value 100)
        assert (> value 100)
    }
    return (shadowed (- n 1) (+ value 2))
}
shadow shadowed { assert (== (shadowed 3 1) 12) }
fn main() -> int { assert (== (shadowed 4 2) 15) return 0 }
shadow main { assert (== (main) 0) }
''')

    def test_tail_return_inside_while_preserves_loop_control(self):
        self.check_both('''
fn through_while(n: int, total: int) -> int {
    if (== n 0) { return total }
    let mut i: int = 0
    while (< i 4) {
        set i (+ i 1)
        if (== i 1) { continue }
        if (== i 2) { return (through_while (- n 1) (+ total i)) }
        break
    }
    return -999
}
shadow through_while { assert (== (through_while 4 1) 9) }
fn main() -> int { assert (== (through_while 20 0) 40) return 0 }
shadow main { assert (== (main) 0) }
''')

    def test_tail_return_inside_for_preserves_loop_control(self):
        self.check_both('''
fn through_for(n: int, total: int) -> int {
    if (== n 0) { return total }
    for i in (range 0 5) {
        if (== i 0) { continue }
        if (== i 2) { return (through_for (- n 1) (+ total i)) }
    }
    return -999
}
shadow through_for { assert (== (through_for 4 1) 9) }
fn main() -> int { assert (== (through_for 20 0) 40) return 0 }
shadow main { assert (== (main) 0) }
''')

    def test_tail_return_propagates_through_nested_loops(self):
        self.check_both('''
fn nested(n: int, total: int) -> int {
    if (== n 0) { return total }
    for outer in (range 0 3) {
        if (== outer 0) { continue }
        let mut inner: int = 0
        while (< inner 3) {
            set inner (+ inner 1)
            if (== inner 1) { continue }
            if (== inner 2) {
                return (nested (- n 1) (+ total (+ outer inner)))
            }
        }
        break
    }
    return -999
}
shadow nested { assert (== (nested 3 1) 10) }
fn main() -> int { assert (== (nested 12 0) 36) return 0 }
shadow main { assert (== (main) 0) }
''')

    def test_million_tail_calls(self):
        self.assertEqual(self.compile_run((ROOT / "tests/tco_test.nano").read_text(),
                                         optimize=True),
                         "I completed one million tail calls.\n")


if __name__ == "__main__":
    unittest.main()
