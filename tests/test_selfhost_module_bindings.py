"""I check function identity through the actual self-hosted driver and output."""
import os
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get("NANOLANG_SELFHOST_COMPILER", ROOT / "bin/nanoc_stage2")).resolve()


class ModuleBindings(unittest.TestCase):
    def check(self, modules, source):
        with tempfile.TemporaryDirectory(prefix="nano-bindings-") as tmp:
            directory = Path(tmp)
            for name, text in modules.items():
                path = directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text.replace("@ROOT@", str(directory)))
            path = directory / "main.nano"
            path.write_text(source.replace("@ROOT@", str(directory)))
            output = directory / "program"
            compiled = subprocess.run([str(COMPILER), str(path), "-o", str(output)],
                                      cwd=ROOT, capture_output=True, timeout=90,
                                      env=dict(os.environ, TMPDIR=tmp))
            self.assertEqual(compiled.returncode, 0, (compiled.stdout + compiled.stderr)[-6000:])
            executed = subprocess.run([str(output)], capture_output=True, timeout=10)
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)

    def test_reused_aliases_and_same_basename(self):
        self.check({
            "left/value.nano": "module LeftValue\npub fn answer() -> int { return 11 }\nshadow answer { assert (== (answer) 11) }\n",
            "right/value.nano": "module RightValue\npub fn answer() -> int { return 22 }\nshadow answer { assert (== (answer) 22) }\n",
            "left/wrapper.nano": 'module LeftWrapper\nmodule "@ROOT@/left/value.nano" as lib\npub fn answer() -> int { return (lib.answer) }\nshadow answer { assert (== (answer) 11) }\n',
            "right/wrapper.nano": 'module RightWrapper\nmodule "@ROOT@/right/value.nano" as lib\npub fn answer() -> int { return (lib.answer) }\nshadow answer { assert (== (answer) 22) }\n',
        }, '''module "@ROOT@/left/wrapper.nano" as left
module "@ROOT@/right/wrapper.nano" as right
fn main() -> int { assert (== (left.answer) 11) assert (== (right.answer) 22) return 0 }
shadow main { assert (== (main) 0) }
''')

    def test_multiline_literal_preserves_next_module_owner(self):
        self.check({
            "first.nano": 'pub fn text() -> string { return "one\ntwo\nthree\nfour\nfive" }\nshadow text { assert (> (str_length (text)) 0) }\n',
            "second.nano": 'module "@ROOT@/first.nano" as first\npub fn answer() -> int { return (str_length (first.text)) }\nshadow answer { assert (> (answer) 0) }\n',
        }, '''module "@ROOT@/second.nano" as second
let banner: string = "a\nb"
fn main() -> int {
    assert (== (second.answer) 23)
    assert (== (str_length banner) 3)
    assert (== (char_at banner 1) 10)
    return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_lexer_positions_after_physical_and_escaped_newlines(self):
        self.check({}, f'''module "{ROOT}/src_nano/compiler/lexer.nano" as lexer
fn main() -> int {{
    let diags: List<CompilerDiagnostic> = (list_CompilerDiagnostic_new)
    let toks: List<LexerToken> = (lexer.tokenize_string "\\\"a\\nb\\\" next\\n\\\"a\\\\nb\\\" last" "positions.nano" diags)
    let next: LexerToken = (list_LexerToken_get toks 1)
    let last: LexerToken = (list_LexerToken_get toks 3)
    assert (== next.line 2)
    assert (== next.column 4)
    assert (== last.line 3)
    assert (== last.column 8)
    return 0
}}
shadow main {{ assert (== (main) 0) }}
''')

    def test_unknown_qualified_name_preserves_output(self):
        for called in ("return (lib.answer)", "return (missing.answer)",
                       "let box: Box = Box { value: (lib.answer) } return box.value"):
            with self.subTest(called=called), tempfile.TemporaryDirectory(prefix="nano-bindings-") as tmp:
                directory = Path(tmp)
                library = directory / "lib.nano"
                library.write_text("pub fn present() -> int { return 1 }\nshadow present { assert (== (present) 1) }\n")
                source = directory / "main.nano"
                source.write_text(f'''module "{library}" as lib
struct Box {{ value: int }}
fn answer() -> int {{ return 7 }}
shadow answer {{ assert (== (answer) 7) }}
fn main() -> int {{ {called} }}
shadow main {{ assert (== (main) 7) }}
''')
                for target in ("native", "c"):
                    output = directory / "preserved"
                    diagnostics = directory / "diagnostics.json"
                    output.write_bytes(b"prior artifact")
                    result = subprocess.run([str(COMPILER), str(source), "--target", target, "-o", str(output),
                                             "--llm-diags-json", str(diagnostics)],
                                            cwd=ROOT, capture_output=True, timeout=90,
                                            env=dict(os.environ, TMPDIR=tmp))
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(output.read_bytes(), b"prior artifact")
                    self.assertIn("M0002", json.dumps(json.loads(diagnostics.read_text())))

    def test_selective_alias_preserves_local_string_and_field(self):
        self.check({"lib.nano": "pub fn answer() -> int { return 6 }\nshadow answer { assert (== (answer) 6) }\n"}, '''from "@ROOT@/lib.nano" import answer as selected
struct Box { selected: int }
fn take(selected: int) -> int { return selected }
shadow take { assert (== (take 8) 8) }
fn main() -> int {
    assert (== (selected) 6)
    let box: Box = Box { selected: (selected) }
    assert (== box.selected 6)
    assert (== (str_length "selected") 8)
    assert (== (take 9) 9)
    if true { let selected: int = 7 assert (== selected 7) }
    assert (== (selected) 6)
    return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_distinct_signatures_and_aggregate_calls(self):
        self.check({
            "number.nano": "pub fn answer(value: int) -> int { return (+ value 1) }\nshadow answer { assert (== (answer 1) 2) }\n",
            "text.nano": 'pub fn answer(value: string) -> string { return (+ value "!") }\nshadow answer { assert (== (answer "x") "x!") }\n',
        }, '''module "@ROOT@/number.nano" as number
module "@ROOT@/text.nano" as text
struct Pair { count: int, label: string }
fn main() -> int {
    let pair: Pair = Pair { count: (number.answer 41), label: (text.answer "ok") }
    assert (== pair.count 42)
    assert (== pair.label "ok!")
    return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_generated_name_does_not_capture_user_names(self):
        self.check({"lib.nano": "pub fn answer() -> int { return 7 }\nshadow answer { assert (== (answer) 7) }\n"}, '''module "@ROOT@/lib.nano" as lib
fn __nano_module_0_answer() -> int { return 99 }
shadow __nano_module_0_answer { assert (== (__nano_module_0_answer) 99) }
fn main() -> int {
    let __nano_module__0_answer: int = 3
    assert (== (__nano_module_0_answer) 99)
    assert (== __nano_module__0_answer 3)
    assert (== (lib.answer) 7)
    return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_function_value_alias_and_callable_parameter(self):
        self.check({"lib.nano": "pub fn answer(value: int) -> int { return (+ value 1) }\nshadow answer { assert (== (answer 1) 2) }\n"}, '''from "@ROOT@/lib.nano" import answer as selected
fn apply(selected: fn(int) -> int, value: int) -> int { return (selected value) }
shadow apply { assert (== (apply selected 4) 5) }
fn main() -> int {
    let operation: fn(int) -> int = selected
    assert (== (operation 41) 42)
    assert (== (apply selected 4) 5)
    return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_loop_and_match_binders_do_not_escape(self):
        self.check({"lib.nano": "pub fn answer() -> int { return 6 }\nshadow answer { assert (== (answer) 6) }\n"}, '''from "@ROOT@/lib.nano" import answer as selected
union Choice { Some { value: int }, None { } }
fn main() -> int {
    let choice: Choice = Choice.Some { value: 9 }
    for selected in (range 0 2) { assert (< selected 2) }
    assert (== (selected) 6)
    match choice {
        Some(selected) => { assert (== selected.value 9) }
        None(empty) => { assert false }
    }
    assert (== (selected) 6)
    let value: int = (match choice {
        Some(selected) => { selected.value }
        None(empty) => { 0 }
    })
    assert (== value 9)
    assert (== (selected) 6)
    return 0
}
shadow main { assert (== (main) 0) }
''')
