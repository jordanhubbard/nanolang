"""I retain exact scalar libm imports and unchanged whole math fixtures."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from tests import test_numeric_builtin_selection as numeric

ROOT = numeric.ROOT


class MathHostContracts(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command
    exercise = numeric.NumericBuiltinSelection.exercise

    @classmethod
    def setUpClass(cls):
        numeric.NumericBuiltinSelection.setUpClass()

    def original(self, name):
        path = ROOT/'tests'/name
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        self.exercise(path.read_text(), source_path=path)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_original_extended_math(self):
        self.original('test_std_math_extended.nano')

    def test_original_complete_extended_math(self):
        self.original('test_std_math_extended_full.nano')

    def test_original_vector2d(self):
        self.original('test_vector2d.nano')

    def test_original_complex(self):
        self.original('test_complex.nano')

    def test_core_values_once_order_and_function_bindings(self):
        self.exercise('''let mut trace:int=0
fn next(value:float, digit:int)->float{set trace (+ (* trace 10) digit) return value}
shadow next{set trace 0 assert (== (next 2.0 1) 2.0) assert (== trace 1) set trace 0}
fn kept(value:string)->string{return value}
shadow kept{assert (== (kept "kept") "kept")}
fn atan(value:string)->string{return value}
shadow atan{assert (== (atan "declared") "declared")}
fn main()->int{
 set trace 0 assert (== (pow (next 2.0 1) (next 3.0 2)) 8.0) assert (== trace 12)
 assert (== (sqrt 16.0) 4.0) assert (== (floor 2.8) 2.0)
 assert (== (ceil 2.2) 3.0) assert (== (round -2.5) -3.0)
 assert (== (sin 0.0) 0.0) assert (== (cos 0.0) 1.0)
 assert (== (tan 0.0) 0.0) assert (== (atan2 0.0 1.0) 0.0)
 let sqrt:fn(string)->string=kept assert (== (sqrt "lexical") "lexical")
 assert (== (atan "declared") "declared")
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_wrong_math_source_operands_preserve_output(self):
        for expression in ['(sqrt)', '(pow 2.0)', '(sqrt true)', '(pow "bad" "bad")']:
            with self.subTest(expression=expression):
                self.exercise('fn main()->int{let result = '+expression+
                              ' return 0} shadow main{assert true}', refused=True)

    def test_wrong_import_signatures_and_libraries_preserve_output(self):
        cases = [
            ('', 'sqrt', 'float', 'int', 'PUSH_I64 4'),
            ('', 'pow', 'float', 'float int', 'PUSH_F64 2.0\nPUSH_I64 3'),
            ('', 'pow', 'int', 'float float', 'PUSH_F64 2.0\nPUSH_F64 3.0'),
            ('libm.so', 'sqrt', 'float', 'float', 'PUSH_F64 4.0'),
            ('', 'sqrt_other', 'float', 'float', 'PUSH_F64 4.0'),
        ]
        output = ROOT/'.test_output/numeric-builtin-selection'
        output.mkdir(parents=True, exist_ok=True)
        for library, name, result_type, parameters, body in cases:
            with self.subTest(library=library, symbol=name, parameters=parameters):
                work = Path(tempfile.mkdtemp(prefix='math-import-refusal-', dir=output))
                assembly, module, source = work/'main.nasm', work/'module.nvm', work/'prior.c'
                assembly.write_text('.entry main\n.import "'+library+'" "'+name+'" '+
                    result_type+' '+parameters+'\n.function main 0 0 0 int 1\n'+body+
                    '\nCALL_EXTERN 0\nPOP\nPUSH_I64 0\nRET\n.end\n')
                self.command(work, 'assemble', [ROOT/'bin/nanoisa', 'asm', assembly, '-o', module])
                self.command(work, 'verify', [ROOT/'bin/nano_vm', '--verify-only', module])
                prior = b'I preserve my prior math translation.\n'
                source.write_bytes(prior)
                result = self.command(work, 'refusal', [ROOT/'bin/nvm2c', module,
                    '-o', source], accepted=False)
                self.assertGreater(result.returncode, 0, result.stdout+result.stderr)
                self.assertTrue(result.stderr, 'I require the actual refusal diagnostic.')
                self.assertEqual(source.read_bytes(), prior)
