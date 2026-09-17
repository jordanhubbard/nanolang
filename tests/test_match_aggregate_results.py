"""I execute scalar and aggregate match results with retained concrete types."""
import unittest
from tests import test_affine_generic_identity as generic

class MatchAggregateResults(unittest.TestCase):
    def test_generic_union_result(self):
        generic.GenericAffineIdentity().check('''union Box<T> { Some { value: T }, None {} }
fn remap(boxed: Box<int>) -> Box<int> { return match boxed {
 Some(payload) => Box.Some { value: (+ payload.value 1) }
 None(payload) => Box.None {}
} }
shadow remap { let input: Box<int> = Box.Some { value: 6 } let output: Box<int> = (remap input) match output { Some(p) => { assert (== p.value 7) } None(n) => { assert false } } }
fn main() -> int { let input: Box<int> = Box.None {} let output: Box<int> = (remap input) match output { Some(p) => { return 1 } None(n) => { return 0 } } }
shadow main { assert (== (main) 0) }
''', True)

    def test_integer_selector_union_result(self):
        generic.GenericAffineIdentity().check('''union Answer { Value { value: int }, Empty {} }
fn choose(value: int) -> Answer { return match value { 1 => Answer.Value { value: 7 }, _ => Answer.Empty {} } }
shadow choose { let result: Answer = (choose 1) match result { Value(p) => { assert (== p.value 7) } Empty(n) => { assert false } } }
fn main() -> int { let result: Answer = (choose 2) match result { Value(p) => { return 1 } Empty(n) => { return 0 } } }
shadow main { assert (== (main) 0) }
''', True)

    def test_scalar_match_result(self):
        generic.GenericAffineIdentity().check('''fn choose(value: int) -> int { return match value { 1 => 7, _ => 8 } }
shadow choose { assert (== (choose 1) 7) assert (== (choose 0) 8) }
fn main() -> int { return (- (choose 0) 8) }
shadow main { assert (== (main) 0) }
''', True)

if __name__ == '__main__':
    unittest.main()
