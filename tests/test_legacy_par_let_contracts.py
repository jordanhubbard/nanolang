"""I preserve legacy par-let values, void bodies and binding scope."""
import hashlib
import unittest

from tests import test_numeric_builtin_selection as numeric

ROOT = numeric.ROOT


class LegacyParLetContracts(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command
    exercise = numeric.NumericBuiltinSelection.exercise

    @classmethod
    def setUpClass(cls):
        numeric.NumericBuiltinSelection.setUpClass()

    def test_original_complete_par_let_fixture(self):
        path = ROOT/'tests/test_par_let.nano'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            self.exercise(path.read_text(), '3\n55\n42\n20\npar-let tests PASSED\n',
                          source_path=path, expected_opcodes=('PRINTLN',))
        finally:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_void_and_value_bodies_preserve_scope_and_effects(self):
        self.exercise('''let mut trace:int=0
fn note(value:int)->void{set trace (+ (* trace 10) value)}
shadow note{let old:int=trace set trace 0 (note 7) assert (== trace 7) set trace old}
fn main()->int{
 set trace 0
 par-let first=2 second=3 in (note first)
 assert (== (+ first second) 5)
 assert (== trace 2)
 par-let third=4 fourth=5 in (+ third fourth)
 assert (== (+ third fourth) 9)
 par-let left=6 right=7 in (print left)
 par-let last=8 end=9 in (println last)
 assert (== (+ left right) 13)
 assert (== (+ last end) 17)
 assert (== trace 2)
 return 0
}
shadow main{let old:int=trace assert (== (main) 0) set trace old}
''', '68\n')

    def test_bound_print_result_is_dropped_once(self):
        self.exercise('''let mut trace:int=0
fn value(number:int)->int{set trace (+ trace number) return (+ number 100)}
shadow value{let old:int=trace set trace 0 assert (== (value 7) 107) assert (== trace 7) set trace old}
fn main()->int{
 set trace 0
 let println:fn(int)->int=value
 par-let left=7 right=9 in (println left)
 assert (== trace 7)
 assert (== (+ left right) 16)
 return 0
}
shadow main{let old:int=trace assert (== (main) 0) set trace old}
''')
