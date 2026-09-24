"""I retain reference formatting results after scratch buffers are released."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AggregateFormatting(unittest.TestCase):
    def execute(self, source):
        with tempfile.TemporaryDirectory(prefix="nano-cseed-format-") as tmp:
            src, product = Path(tmp) / "case.nano", Path(tmp) / "product"
            src.write_text(source)
            for command in ([ROOT / "bin/nanoc_c", src, "-o", product], [product]):
                result = subprocess.run(command, cwd=ROOT, capture_output=True,
                                        text=True, timeout=120)
                self.assertEqual(result.returncode, 0, str(command) + "\n" + result.stdout + result.stderr)

    def test_retained_shadow_native_disagreements(self):
        evidence = ROOT / "docs/evidence/pr522-implementation-2026-09-23/aggregate-format-ownership"
        for name in ("enum_field", "whole_float_array", "nested_array"):
            with self.subTest(case=name):
                self.execute((evidence / (name + "-shadow.nano")).read_text())

    def test_enum_union_fields_and_nested_float_arrays(self):
        self.execute(r'''enum Mode { Quiet = 3, Loud = 7 }
union Choice { Selected { mode: Mode }, Empty {} }
struct State { mode: Mode, choice: Choice, values: array<array<float>> }
fn main() -> int {
 let xs: array<float> = [2.0, (float_from_bits -9223372036854775808)]
 let ys: array<float> = (array_push xs 3.5)
 let nested: array<array<float>> = [ys, [], [4.0]]
 let state: State = State { mode: Mode.Loud, choice: Choice.Selected { mode: Mode.Quiet }, values: nested }
 let saved: string = (to_string state)
 assert (== saved "State { mode: Mode.Loud, choice: Choice.Selected { mode: Mode.Quiet }, values: [[2.0, -0.0, 3.5], [], [4.0]] }")
 assert (== (cast_string state) saved)
 assert (== (to_string Mode.Loud) "7")
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_union_enum_payload_keeps_exact_identity(self):
        with tempfile.TemporaryDirectory(prefix="nano-enum-payload-refusal-") as tmp:
            src, product = Path(tmp) / "case.nano", Path(tmp) / "product"
            for value in ("Other.On", "7"):
                with self.subTest(value=value):
                    src.write_text('enum Mode { On = 7 }\nenum Other { On = 7 }\n'
                                   'union Choice { Selected { mode: Mode } }\n'
                                   'fn main()->int { let x: Choice = Choice.Selected { mode: '
                                   + value + ' } return 0 }\nshadow main { assert true }\n')
                    product.write_text("prior output")
                    result = subprocess.run([ROOT / "bin/nanoc_c", src, "-o", product],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("declared enum payload type", result.stderr)
                    self.assertEqual(product.read_text(), "prior output")

    def test_nested_values_and_retained_results(self):
        self.execute(r'''struct Point { number: int, label: string, values: array<float> }
union Choice { Some { value: Point }, Empty {} }
fn main() -> int {
 let point: Point = Point { number: 7, label: "saved", values: [2.5, -1.5] }
 let choice: Choice = Choice.Some { value: point }
 let saved: string = (to_string choice)
 let alias: string = saved
 assert (== saved "Choice.Some { value: Point { number: 7, label: saved, values: [2.5, -1.5] } }")
 assert (== (to_string Choice.Empty {}) "Choice.Empty")
 let empty: array<int> = []
 assert (== (cast_string empty) "[]")
 assert (== (to_string [true, false]) "[true, false]")
 assert (== (to_string ["a\"b", "line\nnext", "λ"]) "[\"a\"b\", \"line\nnext\", \"λ\"]")
 let mut i: int = 0
 while (< i 100) {
  let next: string = (to_string [i, (+ i 1)])
  assert (> (str_length next) 0)
  set i (+ i 1)
 }
 assert (== saved alias)
 assert (str_contains saved "label: saved")
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_growth_and_single_evaluation(self):
        self.execute(r'''let mut calls: int = 0
fn values() -> array<int> { set calls (+ calls 1) return [calls] }
shadow values { set calls 0 assert (== (at (values) 0) 1) }
struct Label { text: string }
fn render(text: string) -> string { return (to_string Label { text: text }) }
shadow render { assert (== (render "test") "Label { text: test }") }
fn main() -> int {
 set calls 0
 assert (== (to_string (values)) "[1]")
 assert (== calls 1)
 let mut expanded: string = ""
 let mut i: int = 0
 while (< i 1024) { set expanded (+ expanded "x") set i (+ i 1) }
 let result: string = (render expanded)
 assert (== (str_length result) 1040)
 assert (== result (+ "Label { text: " (+ expanded " }")))
 let later: string = (render "short")
 assert (== later "Label { text: short }")
 assert (== (str_length result) 1040)
 return 0
}
shadow main { assert (== (main) 0) }
''')
