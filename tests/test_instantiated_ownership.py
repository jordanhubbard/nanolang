"""I classify concrete payloads while generic owned transfer stays rejected."""
import unittest
from tests import test_affine_generic_identity as generic

PREFIX = '''resource struct Handle { fd: int }
union Box<T> { Some { value: T }, None {} }
union Result<T,E> { Ok { value: T }, Err { error: E } }
'''
ENDING = '''fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
'''

class InstantiatedOwnership(unittest.TestCase):
    check = generic.GenericAffineIdentity.check

    def test_unused_resource_argument_has_no_payload_owner(self):
        self.check(PREFIX + '''union Marker<T> { Mark { number: int } }
fn read(value: Marker<Handle>) -> int { match value { Mark(payload) => { return payload.number } } }
shadow read { let value: Marker<Handle> = Marker.Mark { number: 7 } assert (== (read value) 7) }
fn main() -> int { let value: Marker<Handle> = Marker.Mark { number: 7 } let copy: Marker<Handle> = value return (- (+ (read copy) (read value)) 14) }
shadow main { assert (== (main) 0) }
''', True)

    def test_nested_unused_resource_argument_has_no_payload_owner(self):
        self.check(PREFIX + '''union Marker<T> { Mark { number: int } }
fn read(value: Box<Marker<Handle>>) -> int { match value { Some(payload) => { let marker: Marker<Handle> = payload.value match marker { Mark(inner) => { return inner.number } } } None(payload) => { return 0 } } }
shadow read { let inner: Marker<Handle> = Marker.Mark { number: 7 } let outer: Box<Marker<Handle>> = Box.Some { value: inner } assert (== (read outer) 7) }
fn main() -> int { let inner: Marker<Handle> = Marker.Mark { number: 7 } let outer: Box<Marker<Handle>> = Box.Some { value: inner } let copy: Box<Marker<Handle>> = outer return (- (+ (read outer) (read copy)) 14) }
shadow main { assert (== (main) 0) }
''', True)

    def test_result_arguments_survive_alias_and_return(self):
        self.check(PREFIX + '''fn identity(value: Result<int,string>) -> Result<int,string> { let copy: Result<int,string> = value return copy }
shadow identity { let value: Result<int,string> = Result.Ok { value: 7 } let result: Result<int,string> = (identity value) match result { Ok(payload) => { assert (== payload.value 7) } Err(payload) => { assert false } } }
fn read(value: Result<int,string>) -> int { match value { Ok(payload) => { return payload.value } Err(payload) => { return (str_length payload.error) } } }
shadow read { let value: Result<int,string> = Result.Err { error: "failure" } assert (== (read value) 7) }
fn main() -> int { let value: Result<int,string> = Result.Err { error: "failure" } let copy: Result<int,string> = (identity value) return (- (+ (read value) (read copy)) 14) }
shadow main { assert (== (main) 0) }
''', True)

    def test_nested_ordinary_instantiation(self):
        self.check(PREFIX + '''fn read(value: Box<Result<int,string>>) -> int { match value { Some(payload) => { let result: Result<int,string> = payload.value match result { Ok(inner) => { return inner.value } Err(inner) => { return (str_length inner.error) } } } None(payload) => { return 0 } } }
shadow read { let inner: Result<int,string> = Result.Ok { value: 7 } let outer: Box<Result<int,string>> = Box.Some { value: inner } assert (== (read outer) 7) }
fn main() -> int { let inner: Result<int,string> = Result.Err { error: "failure" } let outer: Box<Result<int,string>> = Box.Some { value: inner } let copy: Box<Result<int,string>> = outer return (- (+ (read outer) (read copy)) 14) }
shadow main { assert (== (main) 0) }
''', True)

    def test_unsubstituted_tuple_resource_payload_rejected(self):
        self.check(PREFIX + 'union Bundle<T> { Some { value: (T,int) }, None {} }\nfn abandon(value: Bundle<Handle>) -> void { }\n' + ENDING, False)

    def test_nested_unsubstituted_tuple_resource_payload_rejected(self):
        self.check(PREFIX + 'union Bundle<T> { Some { value: (T,int) }, None {} }\nfn abandon(value: Box<Bundle<Handle>>) -> void { }\n' + ENDING, False)

    def test_nested_resource_parameter_rejected(self):
        self.check(PREFIX + 'fn abandon(value: Box<Result<int,Handle>>) -> void { }\n' + ENDING, False)

    def test_resource_return_metadata_transfers(self):
        self.check(PREFIX + 'fn identity(value: Result<Handle,string>) -> Result<Handle,string> { return value }\n' + ENDING, True)

    def test_generic_selected_transfer(self):
        self.check(PREFIX + '''fn consume(value: Box<Handle>) -> int { match value { Some(payload) => { let Box.Some { value } = payload let Handle { fd } = value return fd } None(payload) => { return 0 } } }
''' + ENDING, True)

    def test_generic_resource_collection_payload_rejected(self):
        self.check(PREFIX + 'fn abandon(value: Box<array<Handle>>) -> void { }\n' + ENDING, False)

    def test_collection_of_generic_resources_rejected(self):
        self.check(PREFIX + 'fn abandon(value: array<Result<int,Handle>>) -> void { }\n' + ENDING, False)

    def test_nested_concrete_resource_alias_rejected(self):
        self.check(PREFIX + 'fn abandon(value: Box<Result<int,Handle>>) -> void { let copy: Box<Result<int,Handle>> = value }\n' + ENDING, False)

if __name__ == '__main__': unittest.main()
