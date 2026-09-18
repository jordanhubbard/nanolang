"""I preserve declared one-letter nominal names and unbound generic variables."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from tests import test_cseed_imported_unions as imported

ROOT = Path(__file__).resolve().parents[1]
COMPILERS = os.environ.get('NANO_SINGLE_LETTER_COMPILERS', 'nanoc_c').split(',')
UNION = '''union T { Some { n: int }, Empty {} }
fn read(value: T) -> int { match value { Some(payload) => { return payload.n } Empty(payload) => { return 0 } } }
shadow read { let value: T = T.Some { n: 7 } assert (== (read value) 7) }
fn make() -> T { return T.Some { n: 11 } }
shadow make { assert (== (read (make)) 11) }
'''

class SingleLetterNominals(unittest.TestCase):
    def check(self, source, reject=False, module=None):
        for compiler in COMPILERS:
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix='nano-single-letter-') as directory:
                root = Path(directory)
                if module is not None:
                    (root/'choice.nano').write_text(module)
                path, output = root/'input.nano', root/'program'
                path.write_text(source + '\nshadow main { assert (== (main) 0) }\n')
                output.write_text('previous accepted artifact')
                result = subprocess.run([ROOT/'bin'/compiler, path, '-o', output], cwd=ROOT, capture_output=True, text=True, timeout=120)
                diagnostic = result.stdout + result.stderr
                if reject:
                    self.assertGreater(result.returncode, 0, diagnostic)
                    self.assertNotIn('C compilation failed', diagnostic)
                    self.assertEqual(output.read_text(), 'previous accepted artifact')
                else:
                    self.assertEqual(result.returncode, 0, diagnostic)
                    run = subprocess.run([output], capture_output=True, text=True, timeout=15)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_local_union_callback_and_result(self):
        self.check(UNION + 'fn main() -> int { let f: fn(T) -> int = read let value: T = (make) assert (== (f value) 11) assert (== (f (T.Empty {})) 0) return 0 }')

    def test_imported_union(self):
        self.check('import "choice.nano" as Provider\nfn main() -> int { let value: T = (Provider.make) assert (== (Provider.read value) 11) return 0 }', module=imported.MODULE.replace('Choice', 'T'))

    def test_declared_union_does_not_capture_payload_formal(self):
        self.check(UNION + '''union Box<T> { Some { value: T } }
fn read_box(value: Box<int>) -> int { match value { Some(payload) => { return payload.value } } }
shadow read_box { let value: Box<int> = Box.Some { value: 7 } assert (== (read_box value) 7) }
fn main() -> int { let value: Box<int> = Box.Some { value: 9 } assert (== (read_box value) 9) assert (== (read (make)) 11) return 0 }''')

    def test_single_letter_record(self):
        self.check('''struct S { value: int }
fn read(value: S) -> int { return value.value }
shadow read { assert (== (read S { value: 7 }) 7) }
fn main() -> int { let value: S = S { value: 9 } assert (== (read value) 9) return 0 }''')

    def test_unbound_generic_function(self):
        self.check('fn identity(value: T) -> T { return value } shadow identity { assert (== (identity 7) 7) } fn main() -> int { assert (== (identity 9) 9) assert (== (identity true) true) return 0 }')

    def test_distinct_union_refusal(self):
        self.check(UNION + 'union U { Some { n: int } } fn main() -> int { let value: U = U.Some { n: 9 } return (read value) }', reject=True)

    def test_payload_refusal(self):
        self.check(UNION + 'fn main() -> int { let value: T = T.Some { n: true } return (read value) }', reject=True)

if __name__ == '__main__':
    unittest.main()
