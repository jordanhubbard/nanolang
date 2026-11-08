"""I retain whole dynamic-array fixtures and source removal identity."""
import hashlib
import unittest

from tests import test_numeric_builtin_selection as numeric

ROOT = numeric.ROOT


class DynamicArrayContracts(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command
    exercise = numeric.NumericBuiltinSelection.exercise

    @classmethod
    def setUpClass(cls):
        numeric.NumericBuiltinSelection.setUpClass()

    def test_original_complete_dynamic_array_fixture(self):
        path = ROOT/'tests/test_dynamic_arrays.nano'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = ('=== Dynamic Array Tests ===\n\n'
            '✓ test_push_pop_int passed\n✓ test_remove_at passed\n'
            '✓ test_float_array passed\n✓ test_string_array passed\n'
            '✓ test_bool_array passed\n✓ test_large_array passed\n\n'
            '=== All tests passed! ===\n')
        try:
            self.exercise(path.read_text(), expected, source_path=path,
                          expected_opcodes=('ARR_POP', 'ARR_REMOVE'))
        finally:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_exact_scalar_bits_and_mutation_aliases(self):
        self.exercise('''fn main()->int{
 let values:array<float> = [(float_from_bits -9223372036854775808),(float_from_bits 9221120237041090626)]
 let alias:array<float> = values
 let result:float=(array_pop values)
 assert (== (float_to_bits result) 9221120237041090626)
 assert (== (array_length alias) 1)
 assert (== (float_to_bits (at alias 0)) -9223372036854775808)
 let words:array<string> = ["é","kept","末"]
 let same:array<string> = (array_remove_at words 1)
 assert (== (array_length words) 2)
 assert (== (at same 1) "末")
 let popped:string=(array_pop same)
 assert (== popped "末")
 assert (== (array_length words) 1)
 let flags:array<bool> = [true,false]
 assert (== (array_pop flags) false)
 assert (== (array_length flags) 1)
 let bytes:array<u8> = [7,9]
 let byte_value:u8=(array_pop bytes)
 assert (== byte_value 9)
 assert (== (array_length bytes) 1)
 return 0
}
shadow main{assert (== (main) 0)}
''', expected_opcodes=('ARR_POP', 'ARR_REMOVE'))

    def test_receiver_then_index_once_with_alias_growth(self):
        self.exercise('''let mut trace:int=0
let mut values:array<int> = [10,20]
fn receiver()->array<int>{set trace (+ (* trace 10) 1) return values}
shadow receiver{let old:int=trace set trace 0 let same:array<int> = (receiver) assert (== trace 1) assert (== (array_length same) (array_length values)) set trace old}
fn index()->int{set trace (+ (* trace 10) 2) set values (array_push values 30) return 1}
shadow index{let old_trace:int=trace let old:array<int> = values set values [10,20] set trace 0 assert (== (index) 1) assert (== trace 2) assert (== (array_length values) 3) set values old set trace old_trace}
fn main()->int{
 set values [10,20] set trace 0
 let same:array<int> = (array_remove_at (receiver) (index))
 assert (== trace 12)
 assert (== (array_length values) 2)
 assert (== (at same 0) 10)
 assert (== (at same 1) 30)
 set trace 0
 let popped:int=(array_pop (receiver))
 assert (== trace 1)
 assert (== popped 30)
 assert (== (array_length same) 1)
 return 0
}
shadow main{let old_trace:int=trace let old:array<int> = values assert (== (main) 0) set values old set trace old_trace}
''')

    def test_lexical_bindings_keep_declared_signatures(self):
        self.exercise('''fn word(value:int)->string{return "kept"}
shadow word{assert (== (word 7) "kept")}
fn choose(left:bool,right:string)->string{if left{return right} return "other"}
shadow choose{assert (== (choose true "kept") "kept")}
fn main()->int{
 let array_pop:fn(int)->string=word
 let array_remove_at:fn(bool,string)->string=choose
 assert (== (array_pop 7) "kept")
 assert (== (array_remove_at true "kept") "kept")
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_declared_functions_keep_their_signatures(self):
        self.exercise('''fn array_pop(value:int)->string{return "kept"}
shadow array_pop{assert (== (array_pop 7) "kept")}
fn array_remove_at(value:bool,word:string)->string{if value{return word} return "other"}
shadow array_remove_at{assert (== (array_remove_at true "kept") "kept")}
fn main()->int{
 assert (== (array_pop 7) "kept")
 assert (== (array_remove_at true "kept") "kept")
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_existing_negative_and_past_end_removal(self):
        self.exercise('''fn main()->int{
 let values:array<int> = [10,20]
 let same:array<int> = (array_remove_at values -1)
 assert (== (array_length same) 2)
 let unchanged:array<int> = (array_remove_at same 99)
 assert (== (at unchanged 0) 10)
 assert (== (at unchanged 1) 20)
 let empty:array<string> = []
 let still_empty:array<string> = (array_remove_at empty 0)
 assert (== (array_length still_empty) 0)
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_malformed_operands_preserve_prior_output(self):
        for expression in ['(array_pop)', '(array_pop [1] 0)', '(array_pop 1)',
                           '(array_remove_at [1])', '(array_remove_at [1] 0 1)',
                           '(array_remove_at true 0)', '(array_remove_at [1] true)']:
            with self.subTest(expression=expression):
                self.exercise('fn main()->int{'+expression+' return 0}', refused=True)
