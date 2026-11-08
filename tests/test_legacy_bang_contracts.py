"""I retain standalone bang spelling without replacing the original scoping fixture."""
import hashlib
import unittest

from tests import test_numeric_builtin_selection as numeric

ROOT = numeric.ROOT


class LegacyBangContracts(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command
    exercise = numeric.NumericBuiltinSelection.exercise

    @classmethod
    def setUpClass(cls):
        numeric.NumericBuiltinSelection.setUpClass()

    def test_original_complete_symbol_scoping_fixture(self):
        path = ROOT/'tests/test_symbol_scoping.nano'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            self.exercise(path.read_text(), 'symbol scoping tests passed\n', source_path=path)
        finally:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_negation_forms_once_only_operand_and_inequality(self):
        self.exercise('''let mut calls:int=0
fn mark(value:bool)->bool{set calls (+ calls 1) return value}
shadow mark{let old:int=calls set calls 0 assert (mark true) assert (== calls 1) set calls old}
fn main()->int{
 set calls 0
 assert (== (! true) false) assert (== (!false) true)
 assert (== (not true) false) assert (== (not false) true)
 assert (== (!(mark true)) false) assert (== calls 1)
 assert (== (!(!(mark false))) false) assert (== calls 2)
 assert (!= 7 8) assert (not (!= 7 7))
 return 0
}
shadow main{let old:int=calls assert (== (main) 0) set calls old}
''', expected_opcodes=('BOOL_NOT',))

    def test_wrong_operands_and_malformed_forms_preserve_prior_output(self):
        for expression in ('(! 1)', '!1', '(! "wrong")', '!(+ 1 2)', '(!)', '(! true false)'):
            with self.subTest(expression=expression):
                self.exercise('fn main()->int{let value:bool='+expression+' return 0}\n'
                    'shadow main{assert (== (main) 0)}\n', refused=True,
                    refusal_pattern=r'(?i)(bool|operand|type|argument|arity|parse|expected|unexpected)')
