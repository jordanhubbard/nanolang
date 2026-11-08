"""I preserve the existing scalar sorting contract and the whole original fixture."""
import hashlib
import unittest

from tests import test_numeric_builtin_selection as numeric
from tests.test_math_host_contracts import refuse_host_imports
from tests import test_string_stdlib_contracts as string_stdlib

ROOT = numeric.ROOT


class ArraySortContracts(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command
    exercise = numeric.NumericBuiltinSelection.exercise
    exercise_raw = string_stdlib.StringStdlibContracts.exercise_raw

    @classmethod
    def setUpClass(cls):
        numeric.NumericBuiltinSelection.setUpClass()

    def test_original_complete_array_stdlib_fixture(self):
        path = ROOT/'tests/test_stdlib_arrays.nano'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            self.exercise(path.read_text(), 'OK: test_array_sort passed\n'
                'OK: test_array_reverse passed\nOK: test_array_contains passed\n'
                'OK: test_array_index_of passed\n', source_path=path)
        finally:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_five_scalar_kinds_typed_empty_and_detached_storage(self):
        self.exercise('''fn main()->int{
 let integers:array<int> = [9223372036854775807,-9223372036854775808,0,7,-1]
 let sorted:array<int> = (array_sort integers)
 assert (== (at sorted 0) -9223372036854775808)
 assert (== (at sorted 4) 9223372036854775807)
 (array_set integers 0 3)
 assert (== (at sorted 4) 9223372036854775807)
 let high:u8=255 let low:u8=0 let middle:u8=128
 let bytes:array<u8> = [high,low,middle]
 let byte_order:array<u8> = (array_sort bytes)
 assert (== (at byte_order 0) low) assert (== (at byte_order 2) high)
 let direct_bytes:array<u8> = (array_sort [high,low,middle])
 assert (== (at direct_bytes 0) low) assert (== (at direct_bytes 2) high)
 let floats:array<float> = [2.5,-1.5,0.0]
 let float_order:array<float> = (array_sort floats)
 assert (== (at float_order 0) -1.5) assert (== (at floats 0) 2.5)
 let flags:array<bool> = (array_sort [true,false,true])
 assert (not (at flags 0)) assert (at flags 2)
 let unicode_words:array<string> = (array_sort ["é","茶","z","Ä","a"])
 assert (== (at unicode_words 0) "a") assert (== (at unicode_words 1) "z")
 assert (== (at unicode_words 2) "Ä") assert (== (at unicode_words 3) "é") assert (== (at unicode_words 4) "茶")
 let words:array<string> = ["z","café","a"]
 let word_order:array<string> = (array_sort words)
 assert (== (at word_order 0) "a") assert (== (at word_order 2) "z")
 (array_set words 0 "changed") assert (== (at word_order 2) "z")
 let empty_int:array<int> = [] let empty_byte:array<u8> = []
 let empty_float:array<float> = [] let empty_bool:array<bool> = [] let empty_string:array<string> = []
 assert (== (array_length (array_sort empty_int)) 0)
 assert (== (array_length (array_sort empty_byte)) 0)
 assert (== (array_length (array_sort empty_float)) 0)
 assert (== (array_length (array_sort empty_bool)) 0)
 assert (== (array_length (array_sort empty_string)) 0)
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_float_payload_multiset_and_nan_last_without_tie_order(self):
        self.exercise('''fn bit_count(values:array<float>,bits:int)->int{
 let mut count:int=0
 for value in values {if (== (float_to_bits value) bits) {set count (+ count 1)}}
 return count
}
shadow bit_count{assert (== (bit_count [0.0,0.0] 0) 2)}
fn main()->int{
 let nan_a:float=(float_from_bits 9221120237041090561)
 let nan_b:float=(float_from_bits -2251799813685245)
 let negative_zero:float=(float_from_bits -9223372036854775808)
 let positive_inf:float=(float_from_bits 9218868437227405312)
 let negative_inf:float=(float_from_bits -4503599627370496)
 let original:array<float> = [nan_a,positive_inf,negative_zero,nan_b,negative_inf,0.0]
 let sorted:array<float> = (array_sort original)
 assert (== (float_to_bits (at sorted 0)) -4503599627370496)
 assert (== (float_to_bits (at sorted 3)) 9218868437227405312)
 assert (not (== (at sorted 4) (at sorted 4)))
 assert (not (== (at sorted 5) (at sorted 5)))
 assert (== (bit_count sorted 9221120237041090561) 1)
 assert (== (bit_count sorted -2251799813685245) 1)
 assert (== (bit_count sorted -9223372036854775808) 1)
 assert (== (bit_count sorted 0) 1)
 (array_set original 0 0.0)
 assert (== (bit_count sorted 9221120237041090561) 1)
 return 0
}
shadow main{assert (== (main) 0)}
''', native_optimizations=('-O0', '-O2'))

    def test_once_only_lexical_identity_and_surviving_string_owners(self):
        self.exercise('''let mut calls:int=0
let mut saved:array<string> = []
fn build()->array<string>{
 set calls (+ calls 1)
 return [(str_concat "z" "ebra"),(str_concat "a" "pple"),(str_concat "c" "afé")]
}
shadow build{let old:int=calls let values:array<string> = (build) assert (== (at values 1) "apple") set calls old}
fn sorted_words()->array<string>{return (array_sort (build))}
shadow sorted_words{let old:int=calls let values:array<string> = (sorted_words) assert (== (at values 0) "apple") set calls old}
fn keep(value:int)->int{return (+ value 100)}
shadow keep{assert (== (keep 7) 107)}
fn main()->int{
 set calls 0
 set saved (sorted_words)
 assert (== calls 1)
 let original:array<string> = (build)
 let sorted:array<string> = (array_sort original)
 assert (== calls 2)
 (array_push original "aardvark")
 (array_set original 1 "changed")
 assert (== (array_length sorted) 3) assert (== (at sorted 0) "apple")
 for index in (range 0 5000) {let trash:string=(str_replace "temporary temporary" "temporary" "discarded")}
 assert (== (at saved 0) "apple") assert (== (at saved 2) "zebra")
 let array_sort:fn(int)->int=keep
 assert (== (array_sort 7) 107)
 return 0
}
shadow main{let old_calls:int=calls let old_saved:array<string> = saved assert (== (main) 0) set calls old_calls set saved old_saved}
''', native_optimizations=('-O0', '-O2'))

    def test_wrong_operands_and_arities_preserve_prior_output(self):
        for call in ('(array_sort)', '(array_sort true)', '(array_sort [1] [2])',
                     '(array_sort [[1]])'):
            with self.subTest(call=call):
                self.exercise('fn main()->int{'+call+' return 0}\nshadow main{assert (== (main) 0)}\n',
                    refused=True, refusal_pattern=r'(?i)(array|operand|type|argument|arity)')

    def test_wrong_native_signatures_and_array_families_preserve_output(self):
        refuse_host_imports(self, [
            ('other.so', 'vm_array_sort', 'array', 'array', 'PUSH_I64 1\nARR_LITERAL 1 1'),
            ('', 'vm_array_sort', 'int', 'array', 'PUSH_I64 1\nARR_LITERAL 1 1'),
            ('', 'vm_array_sort', 'array', 'int', 'PUSH_I64 1'),
            ('', 'vm_array_sort', 'array', 'array array', 'PUSH_I64 1\nARR_LITERAL 1 1\nDUP'),
            ('', 'vm_array_sort', 'array', 'array', 'PUSH_I64 1\nARR_LITERAL 1 1\nARR_LITERAL 6 1'),
            ('', 'vm_array_sort', 'array', 'array', 'AGG_PACK 0 0 0 0\nARR_LITERAL 8 1'),
            ('', 'vm_array_sort', 'array', 'array', 'FUNCREF 0\nARR_LITERAL 11 1'),
        ], 'array-sort-import-refusal-')

    def test_raw_string_sort_keeps_utf8_array_boundary(self):
        self.exercise_raw('.string bad "\\xc3"\n.import "" "vm_array_sort" array array\n.entry main\n'
            '.function main 0 0 0 int 1\nPUSH_STR bad\nARR_LITERAL 5 1\n'
            'CALL_EXTERN 0\nPOP\nPUSH_I64 0\nRET\n.end\n',
            invalid_utf8=True, utf8_invariant='nhost_array_sort', work_prefix='array-sort-raw-')

    def test_raw_five_storage_families_preserve_result_shape(self):
        cases = [
            ('PUSH_I64 7\nPUSH_I64 -1\nARR_LITERAL 1 2', 'PUSH_I64 -1\nI64_EQ\nASSERT'),
            ('PUSH_U8 255\nPUSH_U8 0\nARR_LITERAL 2 2', 'PUSH_U8 0\nEQ\nASSERT'),
            ('PUSH_F64 1.5\nPUSH_F64 -2.5\nARR_LITERAL 3 2', 'PUSH_F64 -2.5\nF64_EQ\nASSERT'),
            ('PUSH_BOOL 1\nPUSH_BOOL 0\nARR_LITERAL 4 2', 'BOOL_NOT\nASSERT'),
            ('PUSH_STR z\nPUSH_STR a\nARR_LITERAL 5 2', 'PUSH_STR a\nSTR_EQ\nASSERT'),
        ]
        for values, check in cases:
            with self.subTest(values=values):
                self.exercise_raw('.string a "a"\n.string z "z"\n'
                    '.import "" "vm_array_sort" array array\n.entry main\n'
                    '.function main 0 0 0 int 1\n'+values+'\nCALL_EXTERN 0\n'
                    'PUSH_I64 0\nARR_GET\n'+check+'\nPUSH_I64 0\nRET\n.end\n',
                    work_prefix='array-sort-raw-')

    def test_nominal_enum_array_keeps_existing_integer_storage(self):
        self.exercise('''enum Rank{Low=1,High=9}
fn sort_ranks(values:array<Rank>)->array<Rank>{return (array_sort values)}
shadow sort_ranks{let result:array<Rank> = (sort_ranks [Rank.High,Rank.Low]) assert (== (at result 0) Rank.Low)}
fn main()->int{
 let original:array<Rank> = [Rank.High,Rank.Low]
 let result:array<Rank> = (sort_ranks original)
 assert (== (at result 0) Rank.Low) assert (== (at result 1) Rank.High)
 assert (== (at original 0) Rank.High)
 return 0
}
shadow main{assert (== (main) 0)}
''', native_optimizations=('-O0', '-O2'))
