"""I retain exact declared union identity in C-seed callback annotations."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PRELUDE = '''
union Choice { Some { n: int }, Empty {} }
union Other { Some { n: int } }
union Box<T> { Some { value: T } }
fn read_choice(value: Choice) -> int { match value { Some(item) => { return item.n } Empty(item) => { return 0 } } }
shadow read_choice { let v: Choice = Choice.Some { n: 7 } assert (== (read_choice v) 7) }
fn choose_reader() -> fn(Choice) -> int { return read_choice }
shadow choose_reader { assert true }
fn read_box(value: Box<int>) -> int { match value { Some(item) => { return item.value } } }
shadow read_box { let v: Box<int> = Box.Some { value: 9 } assert (== (read_box v) 9) }
'''

class CseedUnionSignatures(unittest.TestCase):
    def check(self, body, reject=False, shadow_result=0):
        with tempfile.TemporaryDirectory(prefix='nano-cseed-union-signature-') as directory:
            source, output = Path(directory)/'input.nano', Path(directory)/'output'
            source.write_text(PRELUDE + body + '\nshadow main { assert (== (main) ' + str(shadow_result) + ') }\n')
            output.write_text('previous accepted artifact')
            result = subprocess.run([ROOT/'bin/nanoc_c', source, '-o', output], cwd=ROOT, capture_output=True, text=True, timeout=120)
            diagnostic = result.stdout + result.stderr
            if reject:
                self.assertGreater(result.returncode, 0, diagnostic)
                self.assertNotIn('C compilation failed', diagnostic)
                self.assertEqual(output.read_text(), 'previous accepted artifact')
            else:
                self.assertEqual(result.returncode, 0, diagnostic)
                run = subprocess.run([output], capture_output=True, text=True, timeout=15)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_local_and_computed_constructor_arguments(self):
        self.check('fn main() -> int { let f: fn(Choice) -> int = read_choice assert (== (f (Choice.Some { n: 8 })) 8) assert (== ((choose_reader) (Choice.Some { n: 9 })) 9) return 0 }')

    def test_empty_and_stored_arguments(self):
        self.check('fn main() -> int { let f: fn(Choice) -> int = read_choice let v: Choice = Choice.Some { n: 8 } assert (== (f v) 8) assert (== (f (Choice.Empty {})) 0) return 0 }')

    def test_union_return_signature(self):
        self.check('fn make_choice() -> Choice { return Choice.Some { n: 11 } } shadow make_choice { assert (== (read_choice (make_choice)) 11) } fn main() -> int { let f: fn() -> Choice = make_choice let v: Choice = (f) assert (== (read_choice v) 11) return 0 }')

    def test_callback_parameter(self):
        self.check('fn apply(f: fn(Choice) -> int, v: Choice) -> int { return (f v) } shadow apply { let v: Choice = Choice.Some { n: 7 } assert (== (apply read_choice v) 7) } fn main() -> int { let v: Choice = Choice.Some { n: 8 } assert (== (apply read_choice v) 8) return 0 }')

    def test_wrong_nominal_signature(self):
        self.check('fn main() -> int { let f: fn(Other) -> int = read_choice return 0 }', reject=True)

    def test_wrong_nominal_argument(self):
        self.check('fn main() -> int { let f: fn(Choice) -> int = read_choice let v: Other = Other.Some { n: 8 } return (f v) }', reject=True)

    def test_wrong_payload(self):
        self.check('fn main() -> int { let f: fn(Choice) -> int = read_choice return (f (Choice.Some { n: true })) }', reject=True)

    def test_concrete_generic_identity(self):
        self.check('fn main() -> int { let f: fn(Box<int>) -> int = read_box assert (== (f (Box.Some { value: 9 })) 9) return 0 }')
        self.check('fn main() -> int { let f: fn(Box<string>) -> int = read_box return 0 }', reject=True)

    def test_shadow_preserves_previous_output(self):
        self.check('fn main() -> int { let f: fn(Choice) -> int = read_choice return (- (f (Choice.Some { n: 7 })) 7) }', reject=True, shadow_result=1)

if __name__ == '__main__':
    unittest.main()
