"""I check complete qualified variant patterns before enabling owned matching."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests import test_affine_generic_identity as generic

DECL = '''union Choice { Some { number: int, text: string }, Other { number: int, text: string }, None {} }
'''

class SelectedVariantPatterns(unittest.TestCase):
    def compile_case(self, body, accepted):
        source = DECL + '''fn read(value: Choice) -> int { match value {
Some(payload) => { ''' + body + ''' }
Other(payload) => { return 0 }
None(payload) => { let Choice.None {} = payload return 0 }
} }
shadow read { let value: Choice = Choice.Some { number: 7, text: "kept" } assert (== (read value) 7) }
fn main() -> int { let value: Choice = Choice.Some { number: 9, text: "kept" } return (- (read value) 9) }
shadow main { assert (== (main) 0) }
'''
        for compiler in generic.COMPILERS:
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix='nano-variant-pattern-') as tmp:
                root = Path(tmp)
                program, output = root / 'main.nano', root / 'program'
                program.write_text(source)
                output.write_bytes(b'prior artifact')
                p = subprocess.run([str(generic.COMPILER_ROOT / compiler), str(program), '-o', str(output)], cwd=generic.ROOT, capture_output=True, text=True, timeout=120)
                if accepted:
                    self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                    run = subprocess.run([str(output)], capture_output=True, text=True, timeout=10)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                else:
                    self.assertGreater(p.returncode, 0, p.stdout + p.stderr)
                    self.assertEqual(output.read_bytes(), b'prior artifact')

    def test_complete_selected_variant(self):
        self.compile_case('let Choice.Some { text, number } = payload assert (== text "kept") return number', True)

    def test_incomplete_variant(self):
        self.compile_case('let Choice.Some { number } = payload return number', False)

    def test_duplicate_variant_field(self):
        self.compile_case('let Choice.Some { number, number } = payload return number', False)

    def test_wrong_selected_variant(self):
        self.compile_case('let Choice.Other { number, text } = payload return number', False)

    def test_owned_union_guard_remains(self):
        generic.GenericAffineIdentity().check('''resource struct Handle { fd: int }
union Choice { Some { owner: Handle }, None {} }
fn close_handle(owner: Handle) -> int { let Handle { fd } = owner return fd }
shadow close_handle { assert (== (close_handle Handle { fd: 3 }) 3) }
fn consume(value: Choice) -> int { match value {
 Some(payload) => { let Choice.Some { owner } = payload return (close_handle owner) }
 None(payload) => { return 0 }
} }
shadow consume { let value: Choice = Choice.Some { owner: Handle { fd: 3 } } assert (== (consume value) 3) }
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
''', False)

if __name__ == '__main__': unittest.main()
