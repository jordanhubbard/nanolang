"""I retain aggregate global values, managed fields and local copies."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RECORD = r'''struct Inner { label: string }
struct State { number: int, inner: Inner, items: array<string>, cache: HashMap<string,string> }
fn make(label: string, number: int) -> State {
 let cache: HashMap<string,string> = (map_new)
 (map_put cache "label" label)
 return State { number: number, inner: Inner { label: (str_concat label "!") }, items: [(str_concat label " item")], cache: cache }
}
shadow make { let value = (make "test" 3) assert (== value.number 3) }
let mut state: State = (make "old" 7)
fn read() -> State { return state }
shadow read { assert (== (read).number 7) }
fn replace(label: string, number: int) -> void { set state (make label number) }
shadow replace { (replace "old" 7) assert (== state.number 7) }
fn main() -> int {
 let saved: State = (read)
 let appended: array<string> = (array_push saved.items "shared")
 assert (== (array_length state.items) 2)
 assert (== (array_length appended) 2)
 (replace "new" 9)
 let mut i: int = 0
 while (< i 300) { let temporary = (make "discarded" i) assert (== temporary.number i) set i (+ i 1) }
 assert (== (map_get saved.cache "label") "old")
 assert (== (map_get state.cache "label") "new")
 assert (== saved.number 7)
 assert (== saved.inner.label "old!")
 assert (== (at saved.items 0) "old item")
 assert (== (at saved.items 1) "shared")
 assert (== state.number 9)
 assert (== state.inner.label "new!")
 assert (== (at state.items 0) "new item")
 return 0
}
shadow main { assert (== (main) 0) }
'''
VARIANT = r'''union Choice { Number { value: int }, Text { value: string }, Empty {} }
let mut chosen: Choice = Choice.Number { value: 7 }
fn select(number: bool) -> void {
 if number { set chosen Choice.Number { value: 9 } }
 else { set chosen Choice.Text { value: (str_concat "saved" " text") } }
}
shadow select { (select true) match chosen { Number(p) => { assert (== p.value 9) } Text(t) => { assert false } Empty(e) => { assert false } } }
fn main() -> int {
 (select false)
 let saved: Choice = chosen
 (select true)
 match saved { Text(t) => { assert (== t.value "saved text") } Number(n) => { assert false } Empty(e) => { assert false } }
 match chosen { Number(n) => { assert (== n.value 9) } Text(t) => { assert false } Empty(e) => { assert false } }
 set chosen Choice.Empty {}
 match chosen { Empty(e) => { return 0 } Number(n) => { return 1 } Text(t) => { return 1 } }
}
shadow main { assert (== (main) 0) }
'''

class AggregateGlobals(unittest.TestCase):
    def execute(self, text, compilers=("nanoc_c", "nanoc_stage1", "nanoc_stage2")):
        with tempfile.TemporaryDirectory(prefix="nano-aggregate-global-") as tmp:
            source, output = Path(tmp)/"case.nano", Path(tmp)/"product"
            source.write_text(text)
            for compiler in compilers:
                for vm in (False, True) if compiler != "nanoc_c" else (False,):
                    with self.subTest(compiler=compiler, vm=vm):
                        commands = [[ROOT/"bin"/compiler, source, *(["--emit-nvm"] if vm else []), "-o", output],
                                    [ROOT/"bin/nano_vm", output] if vm else [output]]
                        for command in commands:
                            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
                            self.assertEqual(result.returncode, 0, str(command)+"\n"+result.stdout+result.stderr)

    def test_nested_record_global_copy_and_replacement(self):
        self.execute(RECORD)

    def test_variant_global_copy_and_alternating_payloads(self):
        self.execute(VARIANT)

    def test_existing_string_map_global(self):
        self.execute((ROOT/"tests/nanoisa/fixtures/string_map_global.nano").read_text())

    def test_cseed_integer_key_map_string_survives_replace_and_remove(self):
        self.execute(r'''let cache: HashMap<int,string> = (map_new)
fn main() -> int {
 (map_put cache 1 (str_concat "old" " value"))
 let saved: string = (map_get cache 1)
 (map_put cache 1 saved)
 assert (== (map_get cache 1) "old value")
 (map_put cache 1 "new value")
 (map_remove cache 1)
 assert (== saved "old value")
 assert (not (map_has cache 1))
 return 0
}
shadow main { assert (== (main) 0) }
''', compilers=("nanoc_c",))

    def test_wrong_global_union_preserves_output(self):
        text = "union Wanted { Some { value: int } }\nunion Other { Some { value: int } }\n" \
               "let chosen: Wanted = Other.Some { value: 7 }\n" \
               "fn main() -> int { return 0 }\nshadow main { assert true }\n"
        with tempfile.TemporaryDirectory(prefix="nano-global-union-refusal-") as tmp:
            source, output = Path(tmp)/"case.nano", Path(tmp)/"product"
            source.write_text(text)
            for compiler in ("nanoc_c", "nanoc_stage1", "nanoc_stage2"):
                with self.subTest(compiler=compiler):
                    output.write_text("prior output")
                    result = subprocess.run([ROOT/"bin"/compiler, source, "-o", output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.assertEqual(output.read_text(), "prior output")

    def test_nominal_enum_global(self):
        self.execute(r'''enum Shade { Light = 3, Dark = 7 }
let mut shade: Shade = Shade.Dark
fn main() -> int {
 assert (== shade Shade.Dark)
 set shade Shade.Light
 assert (== shade Shade.Light)
 return 0
}
shadow main { assert (== (main) 0) }
''')
