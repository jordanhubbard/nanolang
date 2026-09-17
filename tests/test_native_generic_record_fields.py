"""I execute concrete union fields without confusing classification with layout."""
import unittest
import subprocess
import tempfile
from pathlib import Path
from tests import test_affine_generic_identity as generic

BOX = 'union Box<T> { Some { value: T }, None {} }\n'
SHADOW = '\nshadow main { assert (== (main) 0) }\n'

class NativeGenericRecordLayout(unittest.TestCase):
    def check(self, source, accepted=True, modules=None):
        generic.GenericAffineIdentity.check(self, source, accepted, modules)

    def reject_type(self, source):
        for compiler in generic.COMPILERS:
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix='nano-record-field-type-') as directory:
                path, output = Path(directory)/'input.nano', Path(directory)/'output'
                path.write_text(source)
                output.write_text('prior artifact')
                result = subprocess.run([generic.COMPILER_ROOT/compiler, path, '-o', output], cwd=generic.ROOT, capture_output=True, text=True, timeout=120)
                diagnostic = result.stdout + result.stderr
                self.assertNotEqual(result.returncode, 0, diagnostic)
                self.assertEqual(output.read_text(), 'prior artifact')
                self.assertNotIn('C compilation failed', diagnostic)
                self.assertRegex(diagnostic, '(?i)(type mismatch|expected.*type)')

    def test_direct_field_match_executes(self):
        self.check(BOX + """struct Outer { boxed: Box<int> }
fn main() -> int { let outer: Outer = Outer { boxed: Box.Some { value: 42 } }
 match outer.boxed { Some(p) => { return (- p.value 42) } None(n) => { return 1 } }
}""" + SHADOW)

    def test_direct_field_match_expression_executes(self):
        self.check(BOX + """struct Outer { boxed: Box<int> }
fn main() -> int { let outer: Outer = Outer { boxed: Box.Some { value: 42 } }
 return match outer.boxed { Some(p) => (- p.value 42), None(n) => 1 }
}""" + SHADOW)

    def test_wrong_concrete_field_argument_rejected(self):
        self.reject_type(BOX + """struct Outer { boxed: Box<int> }
fn main() -> int { let outer: Outer = Outer { boxed: Box.Some { value: 42 } }
 let wrong: Box<string> = outer.boxed return 0
}""" + SHADOW)

    def test_wrong_union_field_declaration_rejected(self):
        self.reject_type(BOX + """union Other<T> { Some { value: T }, None {} }
struct Outer { boxed: Box<int> }
fn main() -> int { let outer: Outer = Outer { boxed: Box.Some { value: 42 } }
 let wrong: Other<int> = outer.boxed return 0
}""" + SHADOW)

    def test_empty_inline_field_executes(self):
        self.check(BOX + 'struct Outer { boxed: Box<int> }\nfn main() -> int { let outer: Outer = Outer { boxed: Box.None {} } return 0 }' + SHADOW)

    def test_nonempty_field_payload_executes(self):
        self.check(BOX + '''struct Outer { boxed: Box<int> }
fn main() -> int {
 let outer: Outer = Outer { boxed: Box.Some { value: 42 } }
 let boxed: Box<int> = outer.boxed
 match boxed { Some(p) => { return (- p.value 42) } None(n) => { return 1 } }
}''' + SHADOW)

    def test_nested_generic_field_executes(self):
        self.check(BOX + '''struct Outer { boxed: Box<Box<int>> }
fn main() -> int {
 let outer: Outer = Outer { boxed: Box.Some { value: Box.Some { value: 42 } } }
 let boxed: Box<Box<int>> = outer.boxed
 match boxed { Some(p) => {
  let inner: Box<int> = p.value
  match inner { Some(q) => { return (- q.value 42) } None(n) => { return 1 } }
 } None(n) => { return 2 } }
}''' + SHADOW)

    def test_field_only_instance_is_defined(self):
        self.check(BOX + 'struct Outer { boxed: Box<int> }\nfn main() -> int { return 0 }' + SHADOW)

    def test_forward_union_declaration_executes(self):
        self.check('struct Outer { boxed: Box<int> }\n' + BOX + 'fn main() -> int { let outer: Outer = Outer { boxed: Box.None {} } return 0 }' + SHADOW)

    def test_distinct_instances_execute(self):
        self.check(BOX + '''struct Outer { number: Box<int>, label: Box<string> }
fn main() -> int {
 let outer: Outer = Outer { number: Box.Some { value: 42 }, label: Box.Some { value: "kept" } }
 let label: Box<string> = outer.label
 match label { Some(p) => { assert (== p.value "kept") } None(n) => { return 1 } }
 let number: Box<int> = outer.number
 match number { Some(p) => { return (- p.value 42) } None(n) => { return 2 } }
}''' + SHADOW)

    def test_two_argument_result_field_executes(self):
        self.check("""union Result<T,E> { Ok { value: T }, Err { error: E } }
struct State { result: Result<int,string> }
fn main() -> int {
 let state: State = State { result: Result.Err { error: "kept" } }
 let result: Result<int,string> = state.result
 match result { Ok(p) => { return 1 } Err(p) => { assert (== p.error "kept") return 0 } }
}""" + SHADOW)

    def test_nested_record_executes(self):
        self.check(BOX + '''struct Inner { boxed: Box<int> }
struct Outer { inner: Inner }
fn main() -> int {
 let outer: Outer = Outer { inner: Inner { boxed: Box.Some { value: 42 } } }
 let boxed: Box<int> = outer.inner.boxed
 match boxed { Some(p) => { return (- p.value 42) } None(n) => { return 1 } }
}''' + SHADOW)

    def test_imported_field_layout_executes(self):
        module = BOX + """struct Outer { boxed: Box<int> }
pub fn answer() -> int {
 let outer: Outer = Outer { boxed: Box.Some { value: 42 } }
 let boxed: Box<int> = outer.boxed
 match boxed { Some(p) => { return p.value } None(n) => { return 0 } }
}
shadow answer { assert (== (answer) 42) }
"""
        self.check('module "owner.nano" as owner\nfn main() -> int { return (- (owner.answer) 42) }' + SHADOW, modules={'owner.nano': module})

    def test_global_resource_field_remains_rejected(self):
        self.check('resource struct Handle { fd: int }\n' + BOX + 'struct Outer { boxed: Box<Handle> }\nlet owner: Outer = Outer { boxed: Box.None {} }\nfn main() -> int { return 0 }' + SHADOW, False)

    def test_global_resource_collection_remains_rejected(self):
        self.check('resource struct Handle { fd: int }\n' + BOX + 'struct Outer { boxed: Box<array<Handle>> }\nlet owner: Outer = Outer { boxed: Box.None {} }\nfn main() -> int { return 0 }' + SHADOW, False)
