"""I keep retained C-frontend symbols inside their source scopes."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LexicalScopeBoundaries(unittest.TestCase):
    def compile_case(self, source, accepted, native=False):
        with tempfile.TemporaryDirectory(prefix="nano-lexical-scope-") as tmp:
            path = Path(tmp) / "case.nano"
            output = Path(tmp) / ("program" if native else "case.c")
            path.write_text(source)
            output.write_bytes(b"prior artifact")
            command = [str(ROOT / "bin/nanoc_c"), str(path), "-o", str(output)]
            if not native:
                command += ["--target", "c"]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
            diagnostic = result.stdout + result.stderr
            if accepted:
                self.assertEqual(result.returncode, 0, diagnostic)
                if native:
                    run = subprocess.run([str(output)], capture_output=True, text=True, timeout=10)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            else:
                self.assertGreater(result.returncode, 0, diagnostic)
                self.assertRegex(diagnostic, r"(?i)(undefined|not defined|unknown variable)")
                self.assertEqual(output.read_bytes(), b"prior artifact")

    def test_shadow_restores_outer_type_and_value(self):
        for separator in (" ", "\n"):
            with self.subTest(separator=repr(separator)):
                self.compile_case(separator.join([
                    "fn probe() -> float {",
                    "let value: float = 2.5",
                    'if true { let value: string = "inner" assert (== value "inner") }',
                    "return (+ value 0.5)",
                    "}",
                    "shadow probe { assert (== (probe) 3.0) }",
                    "fn main() -> int { assert (== (probe) 3.0) return 0 }",
                    "shadow main { assert (== (main) 0) }",
                ]), True, native=True)

    def test_exited_bindings_are_not_visible(self):
        for inner in (
            "if true { let hidden: int = 7 }",
            "unsafe { let hidden: int = 7 }",
            "while false { let hidden: int = 7 }",
            "for hidden in (range 0 1) { assert (== hidden 0) }",
        ):
            with self.subTest(inner=inner):
                self.compile_case("fn main() -> int { " + inner + " return hidden }", False)

    def test_loop_exits_and_string_return(self):
        self.compile_case('''fn probe() -> string {
    let mut count: int = 0
    let mut result: string = "outer"
    while (< count 3) {
        let local: string = "retained"
        set result local
        set count (+ count 1)
        if (< count 3) { continue }
        break
    }
    unsafe { let result: int = 9 assert (== result 9) }
    return result
}
shadow probe { assert (== (probe) "retained") }
fn local_return() -> string {
    if true { let text: string = "returned" return text }
    return "unused"
}
shadow local_return { assert (== (local_return) "returned") }
fn main() -> int {
    assert (== (probe) "retained")
    assert (== (local_return) "returned")
    return 0
}
shadow main { assert (== (main) 0) }
''', True, native=True)

    def test_array_metadata_survives_shadow_scope(self):
        self.compile_case('''fn probe() -> float {
    let values: array<float> = [2.5]
    if true { let values: array<int> = [9] assert (== (at values 0) 9) }
    return (at values 0)
}
shadow probe { assert (== (probe) 2.5) }
fn main() -> int { assert (== (probe) 2.5) return 0 }
shadow main { assert (== (main) 0) }
''', True, native=True)

    def test_same_named_array_parameters_keep_function_type(self):
        self.compile_case('''fn read_int(values: array<int>) -> int { return (at values 0) }
shadow read_int { assert (== (read_int [9]) 9) }
fn read_float(values: array<float>) -> float { return (at values 0) }
shadow read_float { assert (== (read_float [2.5]) 2.5) }
fn main() -> int {
    assert (== (read_int [9]) 9)
    assert (== (read_float [2.5]) 2.5)
    return 0
}
shadow main { assert (== (main) 0) }
''', True, native=True)

    def test_hashmap_parameter_metadata_survives_emission(self):
        self.compile_case('''fn read_value(values: HashMap<string, string>) -> string {
    return (map_get values "key")
}
shadow read_value {
    let values: HashMap<string, string> = (map_new)
    (map_put values "key" "value")
    assert (== (read_value values) "value")
}
fn main() -> int {
    let values: HashMap<string, string> = (map_new)
    (map_put values "key" "value")
    assert (== (read_value values) "value")
    return 0
}
shadow main { assert (== (main) 0) }
''', True, native=True)

    def test_sibling_branch_cannot_read_other_branch_local(self):
        self.compile_case("""fn main() -> int {
    if true { let hidden: int = 7 } else { return hidden }
    return 0
}""", False)

    def test_function_parameter_is_not_a_later_function_local(self):
        self.compile_case("""fn first(hidden: int) -> int { return hidden }
shadow first { assert (== (first 7) 7) }
fn main() -> int { return hidden }
""", False)


if __name__ == "__main__":
    unittest.main()
