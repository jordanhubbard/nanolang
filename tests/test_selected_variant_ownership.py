"""I transfer selected nongeneric payloads and resolve their resource fields."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests import test_affine_generic_identity as generic

PREFIX = '''resource struct Handle { fd: int }
union Choice { Some { left: Handle, right: Handle, label: string }, Plain { number: int }, None {} }
fn close_handle(owner: Handle) -> int { let Handle { fd } = owner return fd }
shadow close_handle { assert (== (close_handle Handle { fd: 3 }) 3) }
'''

class SelectedVariantOwnership(unittest.TestCase):
    def check(self, some, accepted, plain='return payload.number', after=''):
        source = PREFIX + '''fn consume(value: Choice) -> int { match value {
Some(payload) => { ''' + some + ''' }
Plain(payload) => { ''' + plain + ''' }
None(payload) => { return 0 }
} ''' + after + ''' }
shadow consume {
 let value: Choice = Choice.Some { left: Handle { fd: 3 }, right: Handle { fd: 4 }, label: "kept" }
 assert (== (consume value) 7)
 let empty: Choice = Choice.None {} assert (== (consume empty) 0)
 let plain: Choice = Choice.Plain { number: 7 } assert (== (consume plain) 7)
}
fn main() -> int {
 let value: Choice = Choice.Some { left: Handle { fd: 3 }, right: Handle { fd: 4 }, label: "kept" }
 return (- (consume value) 7)
}
shadow main { assert (== (main) 0) }
'''
        self.program(source, accepted)

    def program(self, source, accepted):
        for compiler in generic.COMPILERS:
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix='nano-selected-owner-') as tmp:
                root = Path(tmp)
                program, output = root/'main.nano', root/'program'
                program.write_text(source)
                output.write_bytes(b'prior artifact')
                result = subprocess.run([str(generic.COMPILER_ROOT/compiler), str(program), '-o', str(output)], cwd=generic.ROOT, capture_output=True, text=True, timeout=120)
                if accepted:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    run = subprocess.run([str(output)], capture_output=True, text=True, timeout=10)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                else:
                    self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                    self.assertRegex(result.stdout + result.stderr, '(?i)(ownership|resource|moved)')
                    self.assertEqual(output.read_bytes(), b'prior artifact')

    def test_two_resources_and_ordinary_sibling_arms(self):
        self.check('let Choice.Some { right, label, left } = payload assert (== label "kept") return (+ (close_handle left) (close_handle right))', True)

    def test_ordinary_call_match_expression_and_statement(self):
        self.program('''union Value { Some { number: int }, None {} }
let mut calls: int = 0
fn make_value() -> Value { set calls (+ calls 1) return Value.Some { number: 7 } }
shadow make_value { let value: Value = (make_value) match value { Some(payload) => { assert (== payload.number 7) } None(payload) => { assert false } } }
fn expression() -> int { return match (make_value) { Some(payload) => payload.number None(payload) => 0 } }
shadow expression { set calls 0 assert (== (expression) 7) assert (== calls 1) }
fn statement() -> int { match (make_value) { Some(payload) => { return payload.number } None(payload) => { return 0 } } }
shadow statement { set calls 0 assert (== (statement) 7) assert (== calls 1) }
fn main() -> int { set calls 0 assert (== (expression) 7) assert (== calls 1) set calls 0 assert (== (statement) 7) assert (== calls 1) return 0 }
shadow main { assert (== (main) 0) }
''', True)

    def test_ignored_selected_payload(self):
        self.check('return 7', False)

    def test_unresolved_selected_field(self):
        self.check('let Choice.Some { left, right, label } = payload return (+ (close_handle left) 4)', False)

    def test_duplicate_selected_field_consumption(self):
        self.check('let Choice.Some { left, right, label } = payload let a: int = (close_handle left) let b: int = (close_handle right) return (+ a (close_handle left))', False)

    def test_use_payload_after_destructure(self):
        self.check('let Choice.Some { left, right, label } = payload let a: int = (close_handle left) let b: int = (close_handle right) return payload.left.fd', False)

    def test_partial_resource_field_move(self):
        self.check('return (+ (close_handle payload.left) (close_handle payload.right))', False)

    def test_original_scrutinee_after_match(self):
        self.check('let Choice.Some { left, right, label } = payload let a: int = (close_handle left) let b: int = (close_handle right)', False, plain='let ignored: int = payload.number', after='return (consume value)')

    def test_nested_resource_record(self):
        self.program('''resource struct Handle { fd: int }
struct Pair { owner: Handle, label: string }
union Choice { Some { pair: Pair }, None {} }
fn close_handle(owner: Handle) -> int { let Handle { fd } = owner return fd }
shadow close_handle { assert (== (close_handle Handle { fd: 7 }) 7) }
fn consume(value: Choice) -> int { match value {
Some(payload) => { let Choice.Some { pair } = payload let Pair { owner, label } = pair assert (== label "kept") return (close_handle owner) }
None(payload) => { return 0 }
} }
shadow consume { let value: Choice = Choice.Some { pair: Pair { owner: Handle { fd: 7 }, label: "kept" } } assert (== (consume value) 7) }
fn main() -> int { let value: Choice = Choice.Some { pair: Pair { owner: Handle { fd: 7 }, label: "kept" } } return (- (consume value) 7) }
shadow main { assert (== (main) 0) }
''', True)

    def test_incompatible_outer_join(self):
        self.program(PREFIX + '''fn consume(value: Choice, extra: Handle) -> int {
 match value {
 Some(payload) => { let Choice.Some { left, right, label } = payload let a: int = (close_handle left) let b: int = (close_handle right) let c: int = (close_handle extra) }
 Plain(payload) => { }
 None(payload) => { }
 }
 return (close_handle extra)
}
shadow consume { assert true }
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
''', False)

    def test_scrutinee_call_evaluates_once(self):
        self.program('''resource struct Handle { fd: int }
union Choice { Some { owner: Handle }, None {} }
let mut calls: int = 0
fn make_value() -> Choice { set calls (+ calls 1) return Choice.Some { owner: Handle { fd: 7 } } }
shadow make_value { set calls 0 let value: Choice = (make_value) match value { Some(payload) => { let Choice.Some { owner } = payload let Handle { fd } = owner assert (== fd 7) } None(payload) => { assert false } } assert (== calls 1) }
fn consume() -> int { match (make_value) { Some(payload) => { let Choice.Some { owner } = payload let Handle { fd } = owner return fd } None(payload) => { return 0 } } }
shadow consume { set calls 0 assert (== (consume) 7) assert (== calls 1) }
fn main() -> int { set calls 0 assert (== (consume) 7) assert (== calls 1) return 0 }
shadow main { assert (== (main) 0) }
''', True)

    def test_wildcard_cannot_hide_owned_payload(self):
        self.program(PREFIX + '''fn consume(value: Choice) -> int { match value { _ => { return 0 } } }
shadow consume { assert true }
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
''', False)

    def test_ignored_resource_binding(self):
        self.program(PREFIX + '''fn consume(value: Choice) -> int { match value {
 Some(_) => { return 0 } Plain(payload) => { return payload.number } None(payload) => { return 0 }
} }
shadow consume { assert true }
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
''', False)

    def test_loop_cannot_reconsume_outer_scrutinee(self):
        self.program(PREFIX + '''fn consume(value: Choice) -> int {
 let mut i: int = 0
 while (< i 1) {
  match value {
   Some(payload) => { let Choice.Some { left, right, label } = payload let a: int = (close_handle left) let b: int = (close_handle right) }
   Plain(payload) => { } None(payload) => { }
  }
  set i (+ i 1)
 }
 return (consume value)
}
shadow consume { assert true }
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
''', False)

    def test_compatible_outer_join(self):
        self.program(PREFIX + '''fn consume(value: Choice, extra: Handle) -> int {
 let mut total: int = 0
 match value {
  Some(payload) => { let Choice.Some { left, right, label } = payload set total (+ (close_handle left) (close_handle right)) }
  Plain(payload) => { set total payload.number }
  None(payload) => { }
 }
 return (+ total (close_handle extra))
}
shadow consume { let value: Choice = Choice.None {} assert (== (consume value Handle { fd: 2 }) 2) }
fn main() -> int { let value: Choice = Choice.Some { left: Handle { fd: 3 }, right: Handle { fd: 4 }, label: "kept" } return (- (consume value Handle { fd: 2 }) 9) }
shadow main { assert (== (main) 0) }
''', True)

if __name__ == '__main__': unittest.main()
