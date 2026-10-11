"""I retain complete global declaration types in native C output."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get('NANO_CONTEXTUAL_NATIVE_COMPILER', ROOT / 'bin/nanoc_c'))


class NativeGlobalTypes(unittest.TestCase):
    def execute(self, body):
        with tempfile.TemporaryDirectory(prefix='nano-native-global-types-') as tmp:
            source, binary = Path(tmp) / 'main.nano', Path(tmp) / 'main'
            source.write_text(body + '\nshadow main {assert (== (main) 0)}\n')
            for command in ([COMPILER, source, '-o', binary], [binary]):
                result = subprocess.run(list(map(str, command)), cwd=ROOT,
                                        capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_record_global_and_mutation(self):
        self.execute('''struct Box {value:int}
let mut global:Box=Box {value:7}
fn main()->int {assert (== global.value 7) set global Box {value:9}
assert (== global.value 9) set global Box {value:7} return 0}''')

    def test_generic_record_initializer(self):
        self.execute('''struct Box {value:int}
fn identity(value:T)->T {return value}
shadow identity {assert (== (identity 1) 1)}
let global:Box=(identity Box {value:7})
fn main()->int {assert (== global.value 7) return 0}''')

    def test_tuple_global(self):
        self.execute('''let pair:(int,bool)=(7,true)
fn main()->int {assert (== pair.0 7) assert pair.1 return 0}''')

    def test_callable_global(self):
        self.execute('''fn identity(value:T)->T {return value}
shadow identity {assert (== (identity 1) 1)}
let callback:fn(int)->int=identity
fn main()->int {assert (== (callback 7) 7) return 0}''')

    def test_enum_global(self):
        self.execute('''enum Color {Red, Blue}
let color:Color=Color.Blue
fn main()->int {assert (== color Color.Blue) return 0}''')

    def test_incompatible_globals_preserve_output(self):
        cases = [
            'enum First {One} enum Second {One} let value:First=Second.One',
            'fn identity(value:T)->T{return value} shadow identity {assert (== (identity 1) 1)} '
            'let callback:fn(int)->bool=identity',
        ]
        for declaration in cases:
            with self.subTest(declaration=declaration), tempfile.TemporaryDirectory() as tmp:
                source, binary = Path(tmp) / 'main.nano', Path(tmp) / 'main'
                source.write_text(declaration + '\nfn main()->int{return 0}\nshadow main {assert true}\n')
                binary.write_bytes(b'prior executable')
                result = subprocess.run(list(map(str, [COMPILER, source, '-o', binary])),
                                        cwd=ROOT, capture_output=True, text=True, timeout=120)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(binary.read_bytes(), b'prior executable')
