"""I preserve nested union payload types before admitting ownership transfer."""
import unittest
from tests import test_affine_generic_identity as generic

class UnionPayloadMetadata(unittest.TestCase):
    def check(self, source, accepted, modules=None):
        generic.GenericAffineIdentity().check(source, accepted, modules)

    def test_generic_array_string_payload(self):
        self.check('''union Box<T> { Some { values: array<T> }, None {} }
fn read(value: Box<string>) -> string { match value { Some(v) => { return (at v.values 0) } None(n) => { return "" } } }
shadow read { let boxed: Box<string> = Box.Some { values: ["kept"] } assert (== (read boxed) "kept") }
fn main() -> int { let boxed: Box<string> = Box.Some { values: ["kept"] } assert (== (read boxed) "kept") return 0 }
shadow main { assert (== (main) 0) }
''', True)

    def test_array_formal_shadows_resource_record(self):
        self.check('''resource struct T { fd: int }
union Box<T> { Some { values: array<T> }, None {} }
fn read(value: Box<string>) -> string { match value { Some(v) => { return (at v.values 0) } None(n) => { return "" } } }
shadow read { let boxed: Box<string> = Box.Some { values: ["kept"] } assert (== (read boxed) "kept") }
fn main() -> int { let boxed: Box<string> = Box.Some { values: ["kept"] } assert (== (read boxed) "kept") return 0 }
shadow main { assert (== (main) 0) }
''', True)

    def test_imported_array_payload(self):
        module = '''union Box<T> { Some { values: array<T> }, None {} }
fn read(value: Box<string>) -> string { match value { Some(v) => { return (at v.values 0) } None(n) => { return "" } } }
shadow read { let boxed: Box<string> = Box.Some { values: ["kept"] } assert (== (read boxed) "kept") }
pub fn check() -> int { let boxed: Box<string> = Box.Some { values: ["kept"] } assert (== (read boxed) "kept") return 0 }
shadow check { assert (== (check) 0) }
'''
        self.check('''module "payload.nano" as payload
fn main() -> int { return (payload.check) }
shadow main { assert (== (main) 0) }
''', True, {"payload.nano": module})

    def test_nested_array_payload(self):
        self.check('''union Box<T> { Some { values: array<array<T>> }, None {} }
fn read(value: Box<int>) -> int { match value { Some(v) => { let row: array<int> = (at v.values 0) return (at row 0) } None(n) => { return 0 } } }
shadow read { let rows: array<array<int>> = [[7]] let boxed: Box<int> = Box.Some { values: rows } assert (== (read boxed) 7) }
fn main() -> int { let rows: array<array<int>> = [[7]] let boxed: Box<int> = Box.Some { values: rows } return (- (read boxed) 7) }
shadow main { assert (== (main) 0) }
''', True)

    def test_generic_array_resource_payload_rejected(self):
        self.check('''resource struct Handle { fd: int }
union Box<T> { Some { values: array<T> }, None {} }
fn abandon(value: Box<Handle>) -> void { }
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
''', False)

if __name__ == "__main__": unittest.main()
