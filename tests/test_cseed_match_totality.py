"""I check the C-seed match domain, guard and totality boundary."""
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
COMPILER = ROOT / "bin/nanoc_c"
PREFIX = """union Choice { Some { number: int }, None {} }
"""


class CSeedMatchTotality(unittest.TestCase):
    def compile(self, source_text, output_bytes=None):
        directory = tempfile.TemporaryDirectory(prefix="nanolang-cseed-match-totality-")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        source = root / "case.nano"
        output = root / "case"
        source.write_text(source_text)
        if output_bytes is not None:
            output.write_bytes(output_bytes)
        result = subprocess.run(
            [str(COMPILER), str(source), "-o", str(output)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result, output

    def refusal_source(self, body):
        return PREFIX + """fn main() -> int {
 let choice: Choice = Choice.Some { number: 7 }
""" + body + """
 return 0
}
shadow main { assert true }
"""

    def assert_refused(self, body, diagnostic):
        prior = b"prior artifact"
        result, output = self.compile(self.refusal_source(body), prior)
        combined = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, combined)
        self.assertRegex(combined, diagnostic)
        self.assertNotRegex(
            combined,
            r"(?i)(parse error|parsing failed|unexpected token|error: incompatible)",
        )
        self.assertEqual(output.read_bytes(), prior)

    def test_complete_domains_execute_and_integer_scrutinees_run_once(self):
        source = PREFIX + """let mut calls: int = 0
fn mark(value: int) -> int { set calls (+ calls 1) return value }
shadow mark { set calls 0 assert (== (mark 3) 3) assert (== calls 1) set calls 0 }

fn wildcard_value() -> int {
 return match (mark 7) { _ => 9 }
}
shadow wildcard_value { set calls 0 assert (== (wildcard_value) 9) assert (== calls 1) set calls 0 }

fn wildcard_statement() -> int {
 let mut selected: int = 0
 match (mark 8) { _ => { set selected 11 } }
 return selected
}
shadow wildcard_statement { set calls 0 assert (== (wildcard_statement) 11) assert (== calls 1) set calls 0 }

fn guarded_value(value: Choice) -> int {
 return match value { Some(payload) if true => payload.number None(empty) => 0 }
}
shadow guarded_value {
 assert (== (guarded_value Choice.Some { number: 13 }) 13)
 assert (== (guarded_value Choice.None {}) 0)
}

fn complete_statement(value: Choice) -> int {
 let mut selected: int = 0
 match value {
  Some(payload) => { set selected payload.number }
  None(empty) => { set selected -1 }
 }
 return selected
}
shadow complete_statement {
 assert (== (complete_statement Choice.Some { number: 17 }) 17)
 assert (== (complete_statement Choice.None {}) -1)
}

fn main() -> int {
 set calls 0
 assert (== (wildcard_value) 9)
 assert (== calls 1)
 set calls 0
 assert (== (wildcard_statement) 11)
 assert (== calls 1)
 assert (== (guarded_value Choice.Some { number: 13 }) 13)
 assert (== (complete_statement Choice.None {}) -1)
 return 0
}
shadow main { assert true }
"""
        result, output = self.compile(source)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        executed = subprocess.run(
            [str(output)], capture_output=True, text=True, timeout=10
        )
        self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)

    def test_generic_union_calls_keep_exact_payloads_and_run_once(self):
        source = """union Result<T, E> { Ok { value: T }, Err { error: E } }
union Option<T> { Some { value: T }, None {} }
let mut calls: int = 0
fn text_result(ok: bool) -> Result<string, string> {
 set calls (+ calls 1)
 if ok { return Result.Ok { value: "kept" } }
 return Result.Err { error: "error" }
}
shadow text_result {
 let value: Result<string, string> = (text_result true)
 match value { Ok(item) => { assert (== item.value "kept") } Err(item) => { assert false } }
}
fn int_result(ok: bool) -> Result<int, string> {
 set calls (+ calls 1)
 if ok { return Result.Ok { value: 31 } }
 return Result.Err { error: "error" }
}
shadow int_result {
 let value: Result<int, string> = (int_result true)
 match value { Ok(item) => { assert (== item.value 31) } Err(item) => { assert false } }
}
fn bool_option(present: bool) -> Option<bool> {
 set calls (+ calls 1)
 if present { return Option.Some { value: true } }
 return Option.None {}
}
shadow bool_option {
 let value: Option<bool> = (bool_option true)
 match value { Some(item) => { assert item.value } None(item) => { assert false } }
}
fn probe() -> int {
 set calls 0
 let mut text: string = ""
 match (text_result true) {
  Ok(item) if false => { assert false }
  Ok(item) => { set text item.value }
  Err(item) => { set text item.error }
 }
 assert (== text "kept") assert (== calls 1)
 let error: string = match (text_result false) {
  Ok(item) => item.value Err(item) => item.error
 }
 assert (== error "error") assert (== calls 2)
 let number: int = match (int_result true) {
  Ok(item) => (+ item.value 1) Err(item) => 0
 }
 assert (== number 32) assert (== calls 3)
 match (int_result false) {
  Ok(item) => { assert false }
  Err(item) => { assert (== item.error "error") }
 }
 assert (== calls 4)
 let selected: bool = match (bool_option true) {
  Some(item) => item.value None(item) => false
 }
 assert selected assert (== calls 5)
 match (bool_option false) {
  Some(item) => { assert false }
  None(item) => { assert true }
 }
 assert (== calls 6)
 return 0
}
shadow probe { assert (== (probe) 0) }
fn main() -> int { return (probe) }
shadow main { assert (== (main) 0) }
"""
        result, output = self.compile(source)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        executed = subprocess.run([str(output)], capture_output=True, text=True, timeout=10)
        self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)

    def test_generic_union_call_refusals_preserve_output(self):
        prefix = """union Option<T> { Some { value: T }, None {} }
fn source() -> Option<string> { return Option.Some { value: "kept" } }
shadow source { assert true }
"""
        cases = (
            ('let value: string = match (source) { Some(item) => item.value }', r'(?is)E035.*missing: None'),
            ('match (source) { Some(item) => { assert true } }', r'(?is)E035.*missing: None'),
            ('let value: int = match (source) { Some(item) => item.value None(item) => "" }', r'(?is)(type mismatch|expected.*int|E001)'),
            ('match (source) { Some(item) => { let value: int = item.value } None(item) => { assert true } }', r'(?is)(type mismatch|expected.*int|E001)'),
        )
        for body, diagnostic in cases:
            with self.subTest(body=body):
                prior = b"prior artifact"
                result, output = self.compile(prefix + 'fn main() -> int { ' + body + ' return 0 }\nshadow main { assert true }\n', prior)
                combined = result.stdout + result.stderr
                self.assertNotEqual(result.returncode, 0, combined)
                self.assertRegex(combined, diagnostic)
                self.assertNotRegex(combined, r'(?i)(parse error|parsing failed|unexpected token|error: incompatible)')
                self.assertEqual(output.read_bytes(), prior)

    def test_incomplete_union_value_and_statement_are_refused(self):
        cases = (
            " let selected: int = match choice { Some(payload) => payload.number }",
            " match choice { Some(payload) => { assert (== payload.number 7) } }",
        )
        for body in cases:
            with self.subTest(body=body):
                self.assert_refused(body, r"(?is)E035.*missing: None")

    def test_conditional_guards_do_not_establish_union_coverage(self):
        cases = (
            " let selected: int = match choice { Some(payload) if (> payload.number 0) => payload.number None(empty) => 0 }",
            " match choice { Some(payload) if (> payload.number 0) => { assert true } None(empty) => { assert true } }",
        )
        for body in cases:
            with self.subTest(body=body):
                self.assert_refused(body, r"(?is)E035.*missing: Some")

    def test_integer_value_and_statement_require_unconditional_wildcard(self):
        cases = (
            " let selected: int = match 7 { 7 => 1 }",
            " match 7 { 7 => { assert true } }",
        )
        for body in cases:
            with self.subTest(body=body):
                self.assert_refused(body, r"(?is)E035.*unconditional wildcard")

    def test_arms_after_unconditional_wildcard_are_refused(self):
        cases = (
            " let selected: int = match choice { _ => 1 Some(payload) => payload.number }",
            " match choice { _ => { assert true } None(empty) => { assert true } }",
            " let selected: int = match choice { _ if true => 1 Some(payload) => payload.number }",
            " match choice { _ if true => { assert true } None(empty) => { assert true } }",
        )
        for body in cases:
            with self.subTest(body=body):
                self.assert_refused(body, r"(?is)E036.*unreachable match arm")

    def test_guards_require_exact_bool_in_value_and_statement_positions(self):
        cases = (
            " let selected: int = match 7 { 7 if 1 => 1 _ => 0 }",
            " match choice { Some(payload) if 1 => { assert true } None(empty) => { assert true } }",
        )
        for body in cases:
            with self.subTest(body=body):
                self.assert_refused(body, r"(?is)E001.*match guard.*bool")

    def test_wrong_scalar_and_mixed_pattern_domains_are_refused(self):
        cases = (
            (
                ' let text: string = "kept" let selected: int = match text { _ => 1 }',
                r"(?is)require a match to inspect an int or a known union",
            ),
            (
                " let real: float = 1.5 match real { _ => { assert true } }",
                r"(?is)require a match to inspect an int or a known union",
            ),
            (
                " let selected: int = match 7 { 7 => 1 Some(payload) => 2 _ => 0 }",
                r"(?is)do not mix integer and union-variant patterns",
            ),
            (
                " match choice { 7 => { assert true } _ => { assert true } }",
                r"(?is)require integer match patterns to inspect an int",
            ),
            (
                " match 7 { Some(payload) => { assert true } _ => { assert true } }",
                r"(?is)require named and or-pattern match arms to inspect a known union",
            ),
        )
        for body, diagnostic in cases:
            with self.subTest(body=body):
                self.assert_refused(body, diagnostic)

    def test_unresolved_union_identity_is_refused_before_coverage(self):
        body = """ let selected: int = match
  (match choice { Some(payload) => Choice.Some { number: payload.number } None(empty) => Choice.None {} })
  { _ => 1 }
"""
        self.assert_refused(body, r"(?is)require an exact known union identity")


if __name__ == "__main__":
    unittest.main()
