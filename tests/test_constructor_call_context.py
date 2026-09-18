"""I preserve declared union construction at call argument boundaries."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
# I exercise both selfhost stages by default; retained C-seed refusal records
# and the explicit shared C control command are in the evidence document.
COMPILERS = os.environ.get('NANO_CONSTRUCTOR_COMPILERS', 'nanoc_stage1,nanoc_stage2').split(',')
PRELUDE = '''union Choice { Some { n: int }, Empty {} }
union Other { Some { n: int } }
union Box<T> { Some { value: T } }
fn read_choice(value: Choice) -> int { match value { Some(item) => { return item.n } Empty(item) => { return 0 } } }
shadow read_choice { let v: Choice = Choice.Some { n: 7 } assert (== (read_choice v) 7) }
fn read_box(value: Box<int>) -> int { match value { Some(item) => { return item.value } } }
shadow read_box { let v: Box<int> = Box.Some { value: 9 } assert (== (read_box v) 9) }
fn choose_reader() -> fn(Choice) -> int { return read_choice }
shadow choose_reader { assert true }
'''

class ConstructorCallContext(unittest.TestCase):
    def check(self, body, reject=False, failing_shadow=False):
        source = PRELUDE + body + '\nshadow main { assert (== (main) ' + ('1' if failing_shadow else '0') + ') }\n'
        for compiler in COMPILERS:
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix='nano-constructor-call-') as directory:
                source_path, output = Path(directory)/'input.nano', Path(directory)/'output'
                source_path.write_text(source)
                output.write_text('previous accepted artifact')
                result = subprocess.run([ROOT/'bin'/compiler, source_path, '-o', output], cwd=ROOT, capture_output=True, text=True, timeout=150)
                diagnostic = result.stdout + result.stderr
                if reject or failing_shadow:
                    self.assertGreater(result.returncode, 0, diagnostic)
                    self.assertEqual(output.read_text(), 'previous accepted artifact')
                    self.assertNotIn('C compilation failed', diagnostic)
                    if failing_shadow:
                        self.assertIn('shadow', diagnostic.lower())
                else:
                    self.assertEqual(result.returncode, 0, diagnostic)
                    run = subprocess.run([output], capture_output=True, text=True, timeout=15)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_direct_and_empty(self):
        self.check('fn main() -> int { assert (== (read_choice (Choice.Some { n: 7 })) 7) assert (== (read_choice (Choice.Empty {})) 0) return 0 }')

    def test_parenthesized_let_and_return(self):
        self.check('fn make_choice() -> Choice { return (Choice.Some { n: 7 }) } shadow make_choice { assert (== (read_choice (make_choice)) 7) } fn main() -> int { let value: Choice = (Choice.Some { n: 8 }) assert (== (read_choice value) 8) assert (== (read_choice (make_choice)) 7) return 0 }')

    def test_function_variable_and_computed(self):
        self.check('fn main() -> int { let f: fn(Choice) -> int = read_choice assert (== (f (Choice.Some { n: 8 })) 8) assert (== ((choose_reader) (Choice.Some { n: 9 })) 9) return 0 }')

    def test_function_value_payload_refusals(self):
        self.check('fn main() -> int { let f: fn(Box<int>) -> int = read_box return (f (Box.Some { value: true })) }', reject=True)
        self.check('fn main() -> int { return ((choose_reader) (Other.Some { n: 7 })) }', reject=True)

    def test_concrete_generic(self):
        self.check('fn main() -> int { assert (== (read_box (Box.Some { value: 9 })) 9) return 0 }')

    def test_generic_function_variable_and_computed(self):
        self.check('fn choose_box_reader() -> fn(Box<int>) -> int { return read_box } shadow choose_box_reader { assert true } fn main() -> int { let f: fn(Box<int>) -> int = read_box assert (== (f (Box.Some { value: 8 })) 8) assert (== ((choose_box_reader) (Box.Some { value: 9 })) 9) return 0 }')

    def test_qualified_call(self):
        for compiler in COMPILERS:
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix='nano-qualified-constructor-') as directory:
                root = Path(directory)
                module = root/'choice.nano'
                module.write_text('union Choice { Some { n: int } } pub fn read(value: Choice) -> int { match value { Some(payload) => { return payload.n } } } shadow read { let v: Choice = Choice.Some { n: 7 } assert (== (read v) 7) }')
                source = root/'main.nano'
                for payload, reject in (('7', False), ('true', True)):
                    source.write_text('import "' + str(module) + '" as Provider\nfn main() -> int { return (- (Provider.read (Choice.Some { n: ' + payload + ' })) 7) } shadow main { assert (== (main) 0) }')
                    output = root/'program'
                    output.write_text('previous accepted artifact')
                    result = subprocess.run([ROOT/'bin'/compiler, source, '-o', output], cwd=ROOT, capture_output=True, text=True, timeout=150)
                    diagnostic = result.stdout + result.stderr
                    if reject:
                        self.assertGreater(result.returncode, 0, diagnostic)
                        self.assertNotIn('C compilation failed', diagnostic)
                        self.assertEqual(output.read_text(), 'previous accepted artifact')
                    else:
                        self.assertEqual(result.returncode, 0, diagnostic)
                        self.assertEqual(subprocess.run([output], timeout=15).returncode, 0)

    def test_reject_wrong_union(self):
        self.check('fn main() -> int { return (read_choice (Other.Some { n: 7 })) }', reject=True)

    def test_reject_unknown_variant(self):
        self.check('fn main() -> int { return (read_choice (Choice.Missing { n: 7 })) }', reject=True)

    def test_reject_missing_extra_duplicate_fields(self):
        for payload in ('', 'n: 7, extra: 1', 'n: 7, n: 8'):
            with self.subTest(payload=payload):
                self.check('fn main() -> int { return (read_choice (Choice.Some {' + payload + '})) }', reject=True)

    def test_reject_payload_type(self):
        self.check('fn main() -> int { return (read_choice (Choice.Some { n: "wrong" })) }', reject=True)
        self.check('fn main() -> int { return (read_box (Box.Some { value: true })) }', reject=True)

    def test_reject_selected_variable_coercion(self):
        self.check('fn main() -> int { let v: Choice = Choice.Some { n: 7 } match v { Some(payload) => { return (read_choice payload) } Empty(payload) => { return 0 } } }', reject=True)

    def test_shadow_failure_preserves_output(self):
        self.check('fn main() -> int { return (- (read_choice (Choice.Some { n: 7 })) 7) }', failing_shadow=True)

if __name__ == '__main__':
    unittest.main()
