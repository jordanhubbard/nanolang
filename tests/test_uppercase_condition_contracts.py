"""I preserve my existing uppercase condition and statement-block distinction."""
import hashlib
import unittest

from tests import test_numeric_builtin_selection as numeric

ROOT = numeric.ROOT


class UppercaseConditionContracts(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command
    exercise = numeric.NumericBuiltinSelection.exercise

    @classmethod
    def setUpClass(cls):
        numeric.NumericBuiltinSelection.setUpClass()

    def test_original_complete_top_level_constants_fixture(self):
        path = ROOT/'tests/test_top_level_constants.nano'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            self.exercise(path.read_text(),
                '=== Testing Top-Level Constants ===\n\nPI = 3.14159\nMAX_SCORE = 100\n'
                'GAME_TITLE = Nanolang Game\nDEBUG_MODE = false\n\n'
                'Circle area (radius=10): 314.159\n\n✅ All top-level constant tests passed!\n', source_path=path)
        finally:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_each_existing_statement_head_keeps_identifier_condition(self):
        self.exercise('''fn choose(FLAG:bool)->int{if FLAG {return 7} else {return 2}}
shadow choose{assert (== (choose true) 7) assert (== (choose false) 2)}
fn local(FLAG:bool)->int{if FLAG {let value:int=8 return value} return 2}
shadow local{assert (== (local true) 8) assert (== (local false) 2)}
fn nested(FLAG:bool)->int{if FLAG {if true {return 9}} return 2}
shadow nested{assert (== (nested true) 9) assert (== (nested false) 2)}
fn loop_while(FLAG:bool)->int{let mut value:int=0 if FLAG {while (< value 3) {set value (+ value 1)}} return value}
shadow loop_while{assert (== (loop_while true) 3) assert (== (loop_while false) 0)}
fn loop_for(FLAG:bool)->int{let mut value:int=0 if FLAG {for index in (range 0 4) {set value (+ value 1)}} return value}
shadow loop_for{assert (== (loop_for true) 4) assert (== (loop_for false) 0)}
fn main()->int{
 assert (== (choose true) 7) assert (== (choose false) 2)
 assert (== (local true) 8) assert (== (local false) 2)
 assert (== (nested true) 9) assert (== (nested false) 2)
 assert (== (loop_while true) 3) assert (== (loop_while false) 0)
 assert (== (loop_for true) 4) assert (== (loop_for false) 0)
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_malformed_literal_and_condition_preserve_prior_output(self):
        for source in (
            'struct Point{value:int} fn main()->int{let point:Point=Point{return 1} return 0}',
            'fn main()->int{let FLAG:bool=true if FLAG {let } return 0}',
            'fn main()->int{let FLAG:int=1 if FLAG {return 1} return 0}',
        ):
            with self.subTest(source=source):
                self.exercise(source+'\nshadow main{assert (== (main) 0)}\n', refused=True,
                    refusal_pattern=r'(?i)(parse|expected|unexpected|type|statement|identifier)')
