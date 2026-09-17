"""I reject hidden resource collections without rejecting ordinary union arrays."""
import unittest
from tests import test_affine_generic_identity as generic

class UnionResourceCollections(unittest.TestCase):
    def check(self, source, accepted, modules=None):
        generic.GenericAffineIdentity().check(source, accepted, modules)

    def test_fixed_resource_array_parameter_rejected(self):
        self.check('''resource struct Handle { fd: int }
union Owners { Some { values: array<Handle> }, None {} }
fn abandon(value: Owners) -> void { }
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
''', False)

    def test_fixed_resource_array_passthrough_rejected(self):
        self.check('''resource struct Handle { fd: int }
union Owners { Some { values: array<Handle> }, None {} }
fn transfer(value: Owners) -> Owners { return value }
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
''', False)

    def test_nested_resource_array_parameter_rejected(self):
        self.check('''resource struct Handle { fd: int }
struct Inner { owner: Handle }
union Owners { Some { values: array<array<Inner>> }, None {} }
fn abandon(value: Owners) -> void { }
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
''', False)

    def test_nested_union_passthrough_rejected(self):
        self.check('''resource struct Handle { fd: int }
union Owners { Some { values: array<Handle> }, None {} }
union Envelope { Some { owners: Owners }, None {} }
fn transfer(value: Envelope) -> Envelope { return value }
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
''', False)

    def test_enclosing_record_passthrough_rejected(self):
        self.check('''resource struct Handle { fd: int }
union Owners { Some { values: array<Handle> }, None {} }
struct Envelope { owners: Owners }
fn transfer(value: Envelope) -> Envelope { return value }
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
''', False)

    def test_ordinary_record_array_payload(self):
        self.check('''struct Plain { value: int }
union Values { Some { values: array<Plain> }, None {} }
fn read(value: Values) -> int { match value { Some(v) => { return (array_length v.values) } None(n) => { return 0 } } }
shadow read { let value: Values = Values.Some { values: [Plain { value: 7 }] } assert (== (read value) 1) }
fn main() -> int { let value: Values = Values.Some { values: [Plain { value: 7 }] } return (- (read value) 1) }
shadow main { assert (== (main) 0) }
''', True)

    def test_ordinary_array_payload_copy_and_match(self):
        self.check('''struct Plain { value: int }
union Values { Some { values: array<int> }, None {} }
fn read(value: Values) -> int { match value { Some(v) => { return (at v.values 0) } None(n) => { return 0 } } }
shadow read { let value: Values = Values.Some { values: [7] } assert (== (read value) 7) }
fn main() -> int { let value: Values = Values.Some { values: [7] } let copy: Values = value return (- (+ (read value) (read copy)) 14) }
shadow main { assert (== (main) 0) }
''', True)

if __name__ == '__main__': unittest.main()
