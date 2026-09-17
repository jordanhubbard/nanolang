"""I execute concrete union fields without confusing classification with layout."""
import unittest
from tests import test_affine_generic_identity as generic

BOX = 'union Box<T> { Some { value: T }, None {} }\n'
SHADOW = '\nshadow main { assert (== (main) 0) }\n'

class NativeGenericRecordLayout(unittest.TestCase):
    def check(self, source, accepted=True, modules=None):
        generic.GenericAffineIdentity.check(self, source, accepted, modules)

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
