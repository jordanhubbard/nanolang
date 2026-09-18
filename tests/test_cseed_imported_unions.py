"""I bind ordinary union declarations before importing function metadata."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = '''union Choice { Some { n: int }, Empty {} }
pub fn read(value: Choice) -> int { match value { Some(payload) => { return payload.n } Empty(payload) => { return 0 } } }
shadow read { let v: Choice = Choice.Some { n: 7 } assert (== (read v) 7) }
pub fn make() -> Choice { return Choice.Some { n: 11 } }
shadow make { assert (== (read (make)) 11) }
pub fn apply(f: fn(Choice) -> int, value: Choice) -> int { return (f value) }
shadow apply { let v: Choice = Choice.Some { n: 7 } assert (== (apply read v) 7) }
'''

class CseedImportedUnions(unittest.TestCase):
    def check(self, body, reject=False, module=MODULE, imported='import "choice.nano" as Provider'):
        with tempfile.TemporaryDirectory(prefix='nano-imported-union-') as directory:
            root = Path(directory)
            (root/'choice.nano').write_text(module)
            source, output = root/'main.nano', root/'program'
            source.write_text(imported + '\n' + body + '\nshadow main { assert (== (main) 0) }\n')
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

    def test_qualified_parameter_and_empty_variant(self):
        self.check('fn main() -> int { assert (== (Provider.read (Choice.Some { n: 7 })) 7) assert (== (Provider.read (Choice.Empty {})) 0) return 0 }')

    def test_imported_result_and_typed_local(self):
        self.check('fn main() -> int { let value: Choice = (Provider.make) assert (== (Provider.read value) 11) return 0 }')

    def test_from_import_calls(self):
        self.check('fn main() -> int { let value: Choice = (make) assert (== (read value) 11) return 0 }', imported='from "choice.nano" import read, make')

    def test_module_callback_signature(self):
        self.check('fn main() -> int { let value: Choice = Choice.Some { n: 9 } assert (== (Provider.apply read value) 9) return 0 }', imported='from "choice.nano" import read\nimport "choice.nano" as Provider')

    def test_wrong_payload_preserves_output(self):
        self.check('fn main() -> int { return (Provider.read (Choice.Some { n: true })) }', reject=True)

    def test_wrong_union_preserves_output(self):
        self.check('union Other { Some { n: int } } fn main() -> int { let value: Other = Other.Some { n: 7 } return (Provider.read value) }', reject=True)

    def test_module_shadow_preserves_output(self):
        self.check('fn main() -> int { return 0 }', reject=True, module=MODULE.replace('(read v) 7)', '(read v) 8)'))

if __name__ == '__main__':
    unittest.main()
