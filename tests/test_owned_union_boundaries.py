"""I distinguish payload ownership from generic arguments and nested copyable arms."""
import unittest

from tests.test_owned_global_producer import _ProducerRoute
from tests.test_owned_union_c_source import _SourceRoute


class _BoundarySources:
    def test_copyable_union_inside_ordinary_arm_of_owned_union(self):
        self.program('''resource struct Handle { fd: int }
union Box<T> { Some { item: T }, None {} }
union Choice { Owned { owner: Handle }, Plain { inner: Box<int> } }
fn close_handle(owner: Handle) -> int { let Handle { fd } = owner return fd }
shadow close_handle { assert (== (close_handle Handle { fd: 7 }) 7) }
fn consume(value: Choice) -> int {
    return match value {
        Owned(payload) => { let Choice.Owned { owner } = payload (close_handle owner) }
        Plain(payload) => {
            let Choice.Plain { inner } = payload
            match inner { Some(item) => item.item None(empty) => 0 }
        }
    }
}
shadow consume {
    assert (== (consume Choice.Owned { owner: Handle { fd: 7 } }) 7)
    let inner: Box<int> = Box.Some { item: 9 }
    assert (== (consume Choice.Plain { inner: inner }) 9)
    let empty: Box<int> = Box.None {}
    assert (== (consume Choice.Plain { inner: empty }) 0)
}
fn main() -> int {
    let inner: Box<int> = Box.Some { item: 9 }
    assert (== (consume Choice.Plain { inner: inner }) 9)
    assert (== (consume Choice.Owned { owner: Handle { fd: 7 } }) 7)
    return 0
}
shadow main { assert (== (main) 0) }
''', True)

    def test_phantom_resource_argument_does_not_make_payload_owned(self):
        self.program('''resource struct Handle { fd: int }
union Marker<T> { Value { number: int } }
fn close_handle(owner: Handle) -> int { let Handle { fd } = owner return fd }
shadow close_handle { assert (== (close_handle Handle { fd: 7 }) 7) }
fn read(value: Marker<Handle>) -> int {
    return match value { Value(payload) => payload.number }
}
shadow read {
    let marker: Marker<Handle> = Marker.Value { number: 4 }
    assert (== (read marker) 4)
    assert (== (read marker) 4)
}
fn main() -> int {
    let marker: Marker<Handle> = Marker.Value { number: 4 }
    let copied: Marker<Handle> = marker
    assert (== (+ (read marker) (read copied)) 8)
    assert (== (close_handle Handle { fd: 7 }) 7)
    return 0
}
shadow main { assert (== (main) 0) }
''', True)


class UnionBoundaryProducer(_ProducerRoute, _BoundarySources, unittest.TestCase):
    pass


class UnionBoundaryC(_SourceRoute, _BoundarySources, unittest.TestCase):
    pass


class UnionBoundaryStage1(_SourceRoute, _BoundarySources, unittest.TestCase):
    source_compiler = 'nanoc_stage1'


class UnionBoundaryStage2(_SourceRoute, _BoundarySources, unittest.TestCase):
    source_compiler = 'nanoc_stage2'
