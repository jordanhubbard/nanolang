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

    def test_nested_values_and_retained_results(self):
        self.execute(r'''enum Mode { Quiet = 3, Loud = 7 }
struct Point { number: int, label: string, mode: Mode, values: array<float> }
union Choice { Some { value: Point }, Empty {} }
fn main() -> int {
 let point: Point = Point { number: 7, label: "saved", mode: Mode.Loud, values: [2.0, -1.5] }
 let choice: Choice = Choice.Some { value: point }
 let saved: string = (to_string choice)
 let alias: string = saved
 assert (== saved "Choice.Some { value: Point { number: 7, label: saved, mode: Mode.Loud, values: [2.0, -1.5] } }")
 assert (== (to_string Choice.Empty {}) "Choice.Empty")
 let empty: array<int> = []
 assert (== (cast_string empty) "[]")
 assert (== (to_string [[1, 2], [], [3]]) "[[1, 2], [], [3]]")
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
 let mut long: string = ""
 let mut i: int = 0
 while (< i 1024) { set long (+ long "x") set i (+ i 1) }
 let result: string = (render long)
 assert (== (str_length result) 1040)
 assert (== result (+ "Label { text: " (+ long " }")))
 let later: string = (render "short")
 assert (== later "Label { text: short }")
 assert (== (str_length result) 1040)
 return 0
}
shadow main { assert (== (main) 0) }
''')
