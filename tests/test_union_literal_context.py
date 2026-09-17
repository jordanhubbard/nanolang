"""I retain concrete union payload types during recursive native literal emission."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests import test_affine_generic_identity as generic

class UnionLiteralContext(unittest.TestCase):
    def check(self, source):
        generic.GenericAffineIdentity().check(source, True)

    def test_nongeneric_record_array(self):
        self.check('''struct Plain { value: int }
union Values { None {}, Other { unused: int }, Some { values: array<Plain> } }
fn read(value: Values) -> int { match value { Some(v) => { let items: array<Plain> = v.values let first: Plain = (at items 0) return first.value } Other(o) => { return 0 } None(n) => { return 0 } } }
shadow read { let value: Values = Values.Some { values: [Plain { value: 7 }] } assert (== (read value) 7) }
fn main() -> int { let value: Values = Values.Some { values: [Plain { value: 9 }] } return (- (read value) 9) }
shadow main { assert (== (main) 0) }
''')

    def test_generic_record_array(self):
        self.check('''struct Plain { value: int }
union Box<T> { None {}, Some { values: array<T> } }
fn read(value: Box<Plain>) -> int { match value { Some(v) => { let items: array<Plain> = v.values let first: Plain = (at items 0) return first.value } None(n) => { return 0 } } }
shadow read { let value: Box<Plain> = Box.Some { values: [Plain { value: 7 }] } assert (== (read value) 7) }
fn main() -> int { let value: Box<Plain> = Box.Some { values: [Plain { value: 9 }] } return (- (read value) 9) }
shadow main { assert (== (main) 0) }
''')

    def test_nested_generic_record_array(self):
        self.check('''struct Plain { value: int }
union Box<T> { None {}, Some { values: array<T> } }
fn read(value: Box<array<Plain>>) -> int { match value { Some(v) => { let rows: array<array<Plain>> = v.values let items: array<Plain> = (at rows 0) let first: Plain = (at items 0) return first.value } None(n) => { return 0 } } }
shadow read { let value: Box<array<Plain>> = Box.Some { values: [[Plain { value: 7 }]] } assert (== (read value) 7) }
fn main() -> int { let value: Box<array<Plain>> = Box.Some { values: [[Plain { value: 9 }]] } return (- (read value) 9) }
shadow main { assert (== (main) 0) }
''')

    def test_wrong_record_retains_prior_output(self):
        source = '''struct Plain { value: int }
struct Other { different: int }
union Values { Some { values: array<Plain> } }
fn main() -> int { let value: Values = Values.Some { values: [Other { different: 7 }] } return 0 }
shadow main { assert (== (main) 0) }
'''
        self.reject(source)

    def test_generic_wrong_record_retains_prior_output(self):
        self.reject('''struct Plain { value: int }
struct Other { different: int }
union Box<T> { Some { values: array<T> } }
fn main() -> int { let value: Box<Plain> = Box.Some { values: [Other { different: 7 }] } return 0 }
shadow main { assert (== (main) 0) }
''')

    def reject(self, source):
        for compiler in generic.COMPILERS:
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix='nano-union-context-') as tmp:
                root = Path(tmp)
                program, output = root / 'main.nano', root / 'program'
                program.write_text(source)
                output.write_bytes(b'prior artifact')
                result = subprocess.run([str(generic.COMPILER_ROOT / compiler), str(program), '-o', str(output)], cwd=generic.ROOT, capture_output=True, text=True, timeout=120)
                self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(output.read_bytes(), b'prior artifact')

if __name__ == '__main__': unittest.main()
