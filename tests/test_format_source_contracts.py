"""I retain the whole original runtime-template formatting fixture."""
import hashlib
import unittest

from tests import test_numeric_builtin_selection as numeric

ROOT = numeric.ROOT


class FormatSourceContracts(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command
    exercise = numeric.NumericBuiltinSelection.exercise

    @classmethod
    def setUpClass(cls):
        numeric.NumericBuiltinSelection.setUpClass()

    def test_original_complete_format_fixture(self):
        path = ROOT/'tests/test_format.nano'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = ('PASS: test_format_string\nPASS: test_format_int\n'
            'PASS: test_format_float\nPASS: test_format_multiple\n'
            'PASS: test_format_no_args\nResults: 5 passed, 0 failed\n')
        try:
            self.exercise(path.read_text(), expected, source_path=path)
        finally:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_runtime_template_order_and_unused_arguments(self):
        self.exercise('''let mut trace:int=0
fn template()->string{set trace (+ (* trace 10) 1) return "%s %d"}
shadow template{let old:int=trace set trace 0 assert (== (template) "%s %d") assert (== trace 1) set trace old}
fn first()->string{set trace (+ (* trace 10) 2) return "é"}
shadow first{let old:int=trace set trace 0 assert (== (first) "é") assert (== trace 2) set trace old}
fn second()->int{set trace (+ (* trace 10) 3) return 7}
shadow second{let old:int=trace set trace 0 assert (== (second) 7) assert (== trace 3) set trace old}
fn extra()->float{set trace (+ (* trace 10) 4) return -0.0}
shadow extra{let old:int=trace set trace 0 assert (== (float_to_bits (extra)) -9223372036854775808) assert (== trace 4) set trace old}
fn main()->int{
 set trace 0
 let result:string=(format (template) (first) (second) (extra))
 assert (== result "é 7")
 assert (== trace 1234)
 return 0
}
shadow main{let old:int=trace assert (== (main) 0) set trace old}
''', expected_opcodes=('STR_CHAR_AT', 'STR_SUBSTR', 'CAST_STRING'))

    def test_raw_conversion_policy_and_runtime_literal_spans(self):
        self.exercise('''fn main()->int{
 let markers:string="%g/%f/%s/%d"
 assert (== (format markers 2.0 -0.0 true -9223372036854775808) "2/-0/true/-9223372036854775808")
 assert (== (format "%g" (float_from_bits -2251799813685182)) "-nan")
 assert (== (format "%s/%d" "é🙂") "é🙂/%d")
 assert (== (format "%%s %q %" "value") "%value %q %")
 assert (== (format "%q %g %s" 7) "%q 7 %s")
 let mut literal:string="é"
 while (< (str_length literal) 8192){set literal (str_concat literal literal)}
 assert (== (format literal 7) literal)
 let decorated:string=(str_concat literal "%s%")
 assert (== (format decorated "末") (str_concat literal "末%"))
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_lexical_binding_keeps_its_signature(self):
        self.exercise('''fn truth(value:int)->bool{return (== value 7)}
shadow truth{assert (truth 7)}
fn main()->int{
 let format:fn(int)->bool=truth
 assert (format 7)
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_declared_function_keeps_its_signature(self):
        self.exercise('''fn format(value:int)->bool{return (== value 7)}
shadow format{assert (format 7)}
fn main()->int{assert (format 7) return 0}
shadow main{assert (== (main) 0)}
''')

    def test_template_arity_and_type_preserve_prior_output(self):
        for expression in ['(format)', '(format 7)', '(format true "value")']:
            with self.subTest(expression=expression):
                self.exercise('fn main()->int{'+expression+' return 0}', refused=True)
