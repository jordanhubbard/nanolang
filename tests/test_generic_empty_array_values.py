"""I keep each generic empty-array value in its concrete storage instance."""
from pathlib import Path
import unittest
from tests import test_nominal_record_array_mutation as nominal

ROOT = Path(__file__).resolve().parents[1]
DECL = 'struct First { a:int }\nstruct Second { a:int }\n'
INTS = '''let ints:array<int> = [100]
let int_values:array<int> = (array_push (choose true ints) 7)
(array_set int_values 0 8) assert (== (at int_values 0) 8)
assert (== (at ints 0) 100)
'''
RECORDS = '''let records:array<First> = [First{a:100}]
let record_values:array<First> = (array_push (choose true records) First{a:7})
(array_set record_values 0 First{a:42}) assert (== (at record_values 0).a 42)
assert (== (at records 0).a 100)
'''
END = '\nshadow main { assert (== (main) 0) }\n'
CHOICES = ['(if flag { [] } else { items })',
           '(cond (flag []) (else items))',
           '(match (if flag { 1 } else { 0 }) { 1 => { [] } _ => { items } })']


class GenericEmptyArrayValues(unittest.TestCase):
    routes = nominal.NominalRecordArrayMutation.routes
    exercise = nominal.NominalRecordArrayMutation.exercise

    def program(self, body, main):
        return (DECL+'fn choose(flag:bool,items:array<T>)->array<T>{'+body+'}\n'
            'shadow choose{let xs:array<int> = [1] assert (== (array_length (choose true xs)) 0)}\n'
            'fn main()->int{'+main+' return 0}'+END)

    def test_original_conditional_record_witness(self):
        self.exercise((ROOT/'tests/nanoisa/fixtures/generic_empty_array_values.nano').read_text(), True)

    def test_return_choices_separate_scalar_and_record_instances(self):
        for target in ['[]', *CHOICES]:
            for main in (INTS+RECORDS, RECORDS+INTS):
                with self.subTest(target=target, record_first=main.startswith('let records')):
                    self.exercise(self.program('return '+target, main), True)

    def test_local_and_block_context_does_not_replace_other_bindings(self):
        choices = [*CHOICES,
            '(if flag { let floats:array<float> = [] '
            'let grown:array<float> = (array_push floats 3.5) '
            'assert (== (at grown 0) 3.5) [] } else { items })']
        for target in choices:
            with self.subTest(target=target):
                self.exercise(self.program('let chosen:array<T> = '+target+' return chosen', RECORDS+INTS), True)

    def test_nested_callable_return_context_stays_separate(self):
        target = ('(if flag { '
            'let build:fn()->array<float> = fn()->array<float>{return []} '
            'let floats:array<float> = (array_push (build) 3.5) '
            'assert (== (at floats 0) 3.5) [] } else { items })')
        self.exercise(self.program('return '+target, RECORDS+INTS), True)

    def test_nested_array_values_keep_the_inner_record_kind(self):
        self.exercise(DECL+'''fn rows(items:array<T>)->array<array<T>>{return [[],items]}
shadow rows{let xs:array<int> = [1] assert (== (array_length (at (rows xs) 0)) 0)}
fn main()->int{
 let items:array<First> = [First{a:1}]
 let values:array<array<First>> = (rows items)
 let fresh:array<First> = (array_push (at values 0) First{a:7})
 (array_set fresh 0 First{a:42}) assert (== (at fresh 0).a 42)
 (array_set (at values 1) 0 First{a:9}) assert (== (at items 0).a 9)
 let numbers:array<array<int>> = (rows [3])
 let expanded:array<int> = (array_push (at numbers 0) 8)
 assert (== (at expanded 0) 8) assert (== (at (at numbers 1) 0) 3)
 return 0
}'''+END, True)

    def test_scalar_kinds_preserve_values_after_empty_growth(self):
        self.exercise(self.program('return '+CHOICES[0], INTS+RECORDS+'''
let bools:array<bool> = [false]
let booleans:array<bool> = (array_push (choose true bools) true)
assert (at booleans 0)
let floats:array<float> = [1.5]
let decimals:array<float> = (array_push (choose true floats) 3.5)
assert (== (at decimals 0) 3.5)
let strings:array<string> = ["old"]
let words:array<string> = (array_push (choose true strings) "kept")
assert (== (at words 0) "kept")
let octet_value:u8 = 255
let bytes:array<u8> = [octet_value]
let octets:array<u8> = (array_push (choose true bytes) octet_value)
assert (== (at octets 0) octet_value)
'''), True)

    def test_concrete_nominal_and_scalar_refusals_preserve_output(self):
        for value in ('Second{a:7}', '7'):
            with self.subTest(value=value):
                self.exercise(self.program('return '+CHOICES[0],
                    'let items:array<First> = [First{a:1}] '
                    '(array_set (choose true items) 0 '+value+')'), False,
                    refusal=None, diagnostic=r'E001|E0001')


if __name__ == '__main__':
    unittest.main()
