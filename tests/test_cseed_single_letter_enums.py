"""I distinguish exact declared enum names from unbound generic variables."""
import unittest
from tests import test_cseed_single_letter_nominals as fixtures

class SingleLetterEnums(unittest.TestCase):
    check = fixtures.SingleLetterNominals.check

    def test_enum_values_and_local(self):
        self.check('enum T { First, Second } fn main() -> int { let value: T = T.Second assert (== value 1) assert (== T.First 0) return 0 }')

    def test_declared_enum_function_annotation(self):
        self.check('enum T { First, Second } fn value(item: T) -> int { return item } shadow value { assert (== (value T.Second) 1) } fn main() -> int { assert (== (value T.Second) 1) return 0 }')

    def test_declared_enum_is_not_implicit_function_variable(self):
        self.check('enum T { First, Second } fn value(item: T) -> int { return item } shadow value { assert (== (value T.Second) 1) } fn main() -> int { return (value true) }', reject=True)

    def test_unbound_other_function_variable(self):
        self.check('enum T { First, Second } fn identity(value: U) -> U { return value } shadow identity { assert (== (identity 7) 7) } fn main() -> int { assert (== (identity 9) 9) assert (== (identity true) true) assert (== T.Second 1) return 0 }')

    def test_payload_formal_is_not_declared_enum(self):
        self.check('''enum T { First, Second }
union Box<T> { Some { value: T } }
fn read(value: Box<int>) -> int { match value { Some(payload) => { return payload.value } } }
shadow read { let value: Box<int> = Box.Some { value: 7 } assert (== (read value) 7) }
fn main() -> int { let value: Box<int> = Box.Some { value: 9 } assert (== (read value) 9) assert (== T.Second 1) return 0 }''')

    def test_imported_enum_definition(self):
        self.check('import "choice.nano" as Provider\nfn main() -> int { assert (== (Provider.number) 1) return 0 }', module='enum T { First, Second } pub fn number() -> int { return T.Second } shadow number { assert (== (number) 1) }')

    def test_unknown_variant_preserves_output(self):
        self.check('enum T { First, Second } fn main() -> int { return T.Missing }', reject=True)

    def test_wrong_scalar_preserves_output(self):
        self.check('enum T { First, Second } fn main() -> int { let value: T = "wrong" return 0 }', reject=True)

if __name__ == '__main__':
    unittest.main()
