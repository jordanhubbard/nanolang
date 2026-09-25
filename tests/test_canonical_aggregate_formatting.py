"""I preserve aggregate formatting through the reference and both canonical routes."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AggregateFormatting(unittest.TestCase):
    def execute(self, text):
        with tempfile.TemporaryDirectory(prefix="nano-canonical-format-") as tmp:
            source, product = Path(tmp) / "case.nano", Path(tmp) / "product"
            source.write_text(text)
            for compiler in ("nanoc_c", "nanoc_stage1", "nanoc_stage2"):
                for vm in (False, True) if compiler != "nanoc_c" else (False,):
                    with self.subTest(compiler=compiler, vm=vm):
                        commands = ([ROOT / "bin" / compiler, source,
                                     *(["--emit-nvm"] if vm else []), "-o", product],
                                    [ROOT / "bin/nano_vm", product] if vm else [product])
                        for command in commands:
                            result = subprocess.run(command, cwd=ROOT, capture_output=True,
                                                    text=True, timeout=120)
                            self.assertEqual(result.returncode, 0, str(command) + "\n" + result.stdout + result.stderr)

    def test_scalar_and_nested_arrays(self):
        self.execute(r'''fn main() -> int {
 let empty: array<int> = []
 assert (== (to_string empty) "[]")
 assert (== (cast_string [1, -2, 3]) "[1, -2, 3]")
 assert (== (to_string [true, false]) "[true, false]")
 assert (== (to_string [2.0, -1.5, (float_from_bits -9223372036854775808)]) "[2.0, -1.5, -0.0]")
 assert (== (to_string ["a\"b", "line\nnext", "λ"]) "[\"a\"b\", \"line\nnext\", \"λ\"]")
 let nested: array<array<int>> = [[1, 2], [], [3]]
 assert (== (to_string nested) "[[1, 2], [], [3]]")
 assert (== f"values {nested}" "values [[1, 2], [], [3]]")
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_records_and_alternating_union_payloads(self):
        self.execute(r'''enum Mode { Quiet = 3, Loud = 7 }
struct Point { number: int, label: string, mode: Mode, values: array<float> }
union Choice { Some { value: Point }, Text { value: string }, Empty {} }
fn render(value: Choice) -> string { return (to_string value) }
shadow render { assert (== (render Choice.Empty {}) "Choice.Empty") }
fn main() -> int {
 let point: Point = Point { number: 7, label: "saved", mode: Mode.Loud, values: [2.0, -1.5] }
 let mut choice: Choice = Choice.Some { value: point }
 let saved: string = (render choice)
 set choice Choice.Text { value: "new" }
 assert (== (render choice) "Choice.Text { value: new }")
 assert (== saved "Choice.Some { value: Point { number: 7, label: saved, mode: Mode.Loud, values: [2.0, -1.5] } }")
 set choice Choice.Empty {}
 assert (== (cast_string choice) "Choice.Empty")
 assert (== (to_string Mode.Loud) "7")
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_single_evaluation_and_retained_strings(self):
        self.execute(r'''let mut calls: int = 0
struct Sample { values: array<int> }
fn next() -> Sample { set calls (+ calls 1) return Sample { values: [calls] } }
shadow next { set calls 0 assert (== (at (next).values 0) 1) }
fn main() -> int {
 set calls 0
 let saved: string = (to_string (next))
 assert (== calls 1)
 assert (== saved "Sample { values: [1] }")
 let mut i: int = 0
 while (< i 300) {
  let text: string = (to_string (next))
  assert (== text (+ "Sample { values: [" (+ (to_string calls) "] }")))
  set i (+ i 1)
 }
 assert (== calls 301)
 assert (== saved "Sample { values: [1] }")
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_union_storage_without_formatting(self):
        self.execute(r'''struct Point { number: int }
union Choice { Some { value: Point }, Text { value: string }, Empty {} }
fn main()->int {
 let mut choice: Choice = Choice.Some { value: Point { number: 7 } }
 match choice { Some(p) => { assert (== p.value.number 7) } Text(t) => { assert false } Empty(e) => { assert false } }
 set choice Choice.Text { value: "saved" }
 match choice { Some(p) => { assert false } Text(t) => { assert (== t.value "saved") } Empty(e) => { assert false } }
 set choice Choice.Empty {}
 match choice { Some(p) => { assert false } Text(t) => { assert false } Empty(e) => { assert true } }
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_distinct_record_variant_layouts(self):
        self.execute(r'''struct Number { value: int }
struct Words { value: string, items: array<string> }
union Choice { Numbered { value: Number }, Named { value: Words }, Empty {} }
fn identity(value: Choice) -> Choice { return value }
shadow identity {
 let value: Choice = (identity Choice.Empty {})
 match value { Numbered(n) => { assert false } Named(n) => { assert false } Empty(e) => { assert true } }
}
fn main() -> int {
 let mut choice: Choice = Choice.Numbered { value: Number { value: 7 } }
 let saved: Choice = (identity choice)
 set choice Choice.Named { value: Words { value: "saved", items: ["one", "two"] } }
 let named: Choice = (identity choice)
 set choice Choice.Empty {}
 match saved { Numbered(n) => { assert (== n.value.value 7) } Named(n) => { assert false } Empty(e) => { assert false } }
 match named { Numbered(n) => { assert false } Named(n) => { assert (== n.value.value "saved") assert (== (at n.value.items 1) "two") } Empty(e) => { assert false } }
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_global_union_snapshots_survive_churn(self):
        self.execute(r'''struct Item { label: string, items: array<string> }
union Choice { Box { value: Item }, Text { value: string }, Empty {} }
let mut current: Choice = Choice.Empty {}
fn identity(value: Choice) -> Choice { return value }
shadow identity {
 let value: Choice = (identity Choice.Empty {})
 match value { Box(b) => { assert false } Text(t) => { assert false } Empty(e) => { assert true } }
}
fn main() -> int {
 set current (identity Choice.Box { value: Item { label: (+ "saved" "-label"), items: [(+ "saved" "-array")] } })
 let saved: Choice = current
 let mut i: int = 0
 while (< i 300) {
  set current (identity Choice.Text { value: (+ "temporary-" (to_string i)) })
  set i (+ i 1)
 }
 set current Choice.Empty {}
 match saved {
  Box(b) => { assert (== b.value.label "saved-label") assert (== (at b.value.items 0) "saved-array") }
  Text(t) => { assert false }
  Empty(e) => { assert false }
 }
 assert (== (to_string saved) "Choice.Box { value: Item { label: saved-label, items: [\"saved-array\"] } }")
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_nested_union_record_transport(self):
        self.execute(r'''struct Item { label: string }
union Choice { Box { value: Item }, Text { value: string }, Empty {} }
struct Envelope { choice: Choice }
fn identity(value: Envelope) -> Envelope { return value }
shadow identity {
 let envelope: Envelope = (identity Envelope { choice: Choice.Empty {} })
 match envelope.choice { Box(b) => { assert false } Text(t) => { assert false } Empty(e) => { assert true } }
}
fn read(value: Envelope) -> string {
 match value.choice {
  Box(b) => { return b.value.label }
  Text(t) => { return t.value }
  Empty(e) => { return "empty" }
 }
}
shadow read { assert (== (read Envelope { choice: Choice.Empty {} }) "empty") }
fn main() -> int {
 let saved: Envelope = (identity Envelope { choice: Choice.Box { value: Item { label: (+ "saved" "-label") } } })
 let other: Envelope = (identity Envelope { choice: Choice.Text { value: (+ "text" "-label") } })
 assert (== (read other) "text-label")
 assert (== (read saved) "saved-label")
 assert (== (read (identity Envelope { choice: Choice.Empty {} })) "empty")
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_union_record_array_literal(self):
        self.execute(r'''struct Item { label: string }
union Choice { Box { value: Item }, Text { value: string }, Empty {} }
struct Envelope { choice: Choice }
fn read(value: Choice) -> string {
 match value {
  Box(b) => { return b.value.label }
  Text(t) => { return t.value }
  Empty(e) => { return "empty" }
 }
}
shadow read { assert (== (read Choice.Empty {}) "empty") }
fn main() -> int {
 let values: array<Envelope> = [Envelope { choice: Choice.Box { value: Item { label: (+ "saved" "-label") } } }, Envelope { choice: Choice.Text { value: (+ "text" "-label") } }, Envelope { choice: Choice.Empty {} }]
 assert (== (read (at values 0).choice) "saved-label")
 assert (== (read (at values 1).choice) "text-label")
 assert (== (read (at values 2).choice) "empty")
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_union_record_array_aliases(self):
        self.execute(r'''struct Item { label: string }
union Choice { Box { value: Item }, Text { value: string }, Empty {} }
struct Envelope { choice: Choice }
fn read(value: Choice) -> string {
 match value {
  Box(b) => { return b.value.label }
  Text(t) => { return t.value }
  Empty(e) => { return "empty" }
 }
}
shadow read { assert (== (read Choice.Empty {}) "empty") }
let mut current: array<Envelope> = []
fn identity(values: array<Envelope>) -> array<Envelope> { return values }
shadow identity { assert (== (array_length (identity [])) 0) }
fn write(values: array<Envelope>, value: Envelope) -> void { (array_set values 0 value) }
shadow write {
 let values: array<Envelope> = [Envelope { choice: Choice.Empty {} }]
 (write values Envelope { choice: Choice.Text { value: "shadow" } })
 assert (== (read (at values 0).choice) "shadow")
}
fn main() -> int {
 let values: array<Envelope> = [Envelope { choice: Choice.Box { value: Item { label: (+ "saved" "-label") } } }]
 assert (== (read (at values 0).choice) "saved-label")
 set current (identity values)
 let alias: array<Envelope> = (identity current)
 let saved: Envelope = (at values 0)
 (write alias Envelope { choice: Choice.Text { value: (+ "changed" "-label") } })
 assert (== (read (at current 0).choice) "changed-label")
 assert (== (read saved.choice) "saved-label")
 let grown: array<Envelope> = (array_push alias Envelope { choice: Choice.Box { value: Item { label: "appended" } } })
 assert (== (read (at grown 1).choice) "appended")
 (write current Envelope { choice: Choice.Empty {} })
 assert (== (read (at values 0).choice) "empty")
 return 0
}
shadow main { assert (== (main) 0) }
''')
