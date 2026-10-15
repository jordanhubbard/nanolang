"""I exercise global source semantics through checked shadows, VM and native C."""
import unittest
from tests.test_owned_union_c_source import _SourceRoute


class OwnedGlobalSource(_SourceRoute, unittest.TestCase):
    def source(self, declarations, body, helpers="", accepted=True, diagnostic=None):
        program = f'''
resource struct Handle {{ fd: int }}
{declarations}
{helpers}
fn main() -> int {{
    let local_owner: Handle = Handle {{ fd: 7 }}
    let Handle {{ fd }} = local_owner
    assert (== fd 7)
    {body}
    return 0
}}
shadow main {{ assert (== (main) 0) }}
'''
        self.source_route(program, accepted, {"nanoc_c": diagnostic} if diagnostic else None)

    def test_scalar_kinds_and_mutation(self):
        self.source('''let mut counter: int = 3
let mut enabled: bool = false
let mut ratio: float = 1.25
let mut label: string = "first"
''', '''assert (== counter 3)
set counter (+ counter 1)
set enabled (not enabled)
set ratio (+ ratio 0.75)
set label "second"
assert (== counter 4)
assert enabled
assert (== ratio 2.0)
assert (== label "second")''')

    def test_initializer_order_and_once_only_helper_effects(self):
        self.source('''let mut calls: int = 0
let first: int = (next)
let second: int = (next)
''', '''assert (== first 1)
assert (== second 2)
assert (== calls 2)''', '''fn next() -> int { set calls (+ calls 1) return calls }
shadow next { set calls 1 assert (== (next) 2) }
''')

    def test_local_and_parameter_shadowing(self):
        self.source('let mut counter: int = 4', '''assert (== (identity 9) 9)
if true {
    let mut counter: int = 100
    set counter 101
    assert (== counter 101)
}
assert (== counter 4)
set counter 5
assert (== (read) 5)''', '''fn identity(counter: int) -> int { return counter }
shadow identity { assert (== (identity 9) 9) }
fn read() -> int { return counter }
shadow read { assert (== (read) 4) }
''')

    def test_copyable_union_global(self):
        self.source('''union Choice { Some { value: int }, None {} }
let mut selected: Choice = Choice.Some { value: 7 }
''', '''match selected {
    Some(payload) => { assert (== payload.value 7) }
    None(payload) => { assert false }
}
set selected Choice.None {}
match selected {
    Some(payload) => { assert false }
    None(payload) => { assert true }
}''')

    def test_immutable_assignment_refused(self):
        self.source('let counter: int = 0', 'set counter 1', accepted=False,
                    diagnostic='(?i)(immutable|mutable|constant)')

    def test_wrong_global_assignment_type_refused(self):
        self.source('let mut counter: int = 0', 'set counter true', accepted=False,
                    diagnostic='Assignment expects int but got bool')

    def test_uninitialized_global_read_in_initializer_refused(self):
        self.source('''let first: int = (read)
let later: int = 7
''', 'assert (== first 7)', '''fn read() -> int { return later }
shadow read { assert (== (read) 7) }
''', accepted=False, diagnostic='global initialization')

    def test_same_shape_wrong_union_initializer_refused(self):
        self.source('''union First { Some { value: int } }
union Second { Some { value: int } }
let selected: First = Second.Some { value: 7 }
''', '', accepted=False, diagnostic='exact declared global union')

    def test_same_shape_wrong_union_assignment_refused(self):
        self.source('''union First { Some { value: int } }
union Second { Some { value: int } }
let mut selected: First = First.Some { value: 7 }
''', 'set selected Second.Some { value: 7 }', accepted=False,
                    diagnostic='exact declared global union')

    def test_generic_union_global_and_alias(self):
        self.source('''union Box<T> { Some { value: T }, None {} }
let selected: Box<int> = Box.Some { value: 7 }
let alias: Box<int> = selected
''', '''match alias {
    Some(payload) => { assert (== payload.value 7) }
    None(payload) => { assert false }
}''')

    def test_wrong_generic_union_global_alias_refused(self):
        self.source('''union Box<T> { Some { value: T }, None {} }
let selected: Box<int> = Box.Some { value: 7 }
let alias: Box<bool> = selected
''', '', accepted=False, diagnostic='exact declared global union')

    def test_resource_global_remains_refused(self):
        self.source('let owner: Handle = Handle { fd: 9 }', '', accepted=False,
                    diagnostic='(?i)(global|resource|ownership|copyable)')


if __name__ == "__main__":
    unittest.main()
