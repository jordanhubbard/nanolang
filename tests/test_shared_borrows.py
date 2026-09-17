"""I share a fixed owner by reference and reject moves while its call borrows it."""
from pathlib import Path
import os
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILERS = os.environ.get('NANO_BORROW_COMPILERS', 'nanoc_c,nanoc_stage1,nanoc_stage2').split(',')
PRELUDE = '''resource struct Handle { fd: int }
fn read(view: &Handle) -> int { return view.fd }
shadow read { assert true }
fn consume(value: Handle) -> int { let Handle { fd } = value return fd }
shadow consume { assert (== (consume Handle { fd: 7 }) 7) }
'''

class SharedBorrows(unittest.TestCase):
    def check(self, source, reject=False, fixtures=None):
        for compiler in COMPILERS:
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix='nano-shared-') as directory:
                path, output = Path(directory)/'input.nano', Path(directory)/'output'
                path.write_text(source)
                for name, text in (fixtures or {}).items():
                    (Path(directory)/name).write_text(text)
                output.write_text('prior artifact')
                result = subprocess.run([ROOT/'bin'/compiler, path, '--keep-c', '-o', output], cwd=ROOT,
                                        capture_output=True, text=True, timeout=180)
                diagnostic = result.stdout + result.stderr
                if reject:
                    self.assertNotEqual(result.returncode, 0, diagnostic)
                    self.assertEqual(output.read_text(), 'prior artifact')
                    self.assertNotIn('C compilation failed', diagnostic)
                    self.assertNotIn('Parsing error', diagnostic)
                    self.assertNotIn('Failed to parse', diagnostic)
                    self.assertNotIn('Segmentation fault', diagnostic)
                else:
                    self.assertEqual(result.returncode, 0, diagnostic)
                    generated = output.with_suffix('.c')
                    if not generated.exists():
                        match = re.search(r'I kept generated C in (.+)', diagnostic)
                        self.assertIsNotNone(match, diagnostic)
                        generated = Path(match.group(1))
                    native = generated.read_text()
                    self.assertRegex(native, r'nl_read\(const nl_Handle\*')
                    self.assertRegex(native, r'&\(\(?\*?nl_?h|&\(h\)')
                    ran = subprocess.run([output], capture_output=True, text=True, timeout=30)
                    self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)

    def test_repeated_shared_aliases_and_forwarding(self):
        self.check(PRELUDE + '''fn sum(a: &Handle, b: &Handle) -> int { return (+ a.fd b.fd) }
shadow sum { assert true }
fn forward(view: &Handle) -> int { return (sum &view &view) }
shadow forward { assert true }
fn main() -> int {
 let h: Handle = Handle { fd: 11 }
 assert (== (read &h) 11)
 assert (== (sum &h &h) 22)
 assert (== (forward &h) 22)
 assert (== (consume h) 11)
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_borrowed_parameter_cannot_move_escape_or_overwrite(self):
        for body, result in [
            ('return (consume view)', 'int'),
            ('return view', 'Handle'),
            ('let alias: Handle = view return (consume alias)', 'int'),
            ('let Handle { fd } = view return fd', 'int'),
            ('set view Handle { fd: 2 } return 0', 'int'),
        ]:
            with self.subTest(body=body):
                self.check(PRELUDE + f'fn invalid(view: &Handle) -> {result} {{ {body} }}\n'
                           'shadow invalid { assert true }\nfn main() -> int { return 0 }\nshadow main { assert true }\n', True)

    def test_argument_order_holds_owner_until_call(self):
        for args in ('&h (consume h)', '&h (if true { set h Handle { fd: 2 } 0 } else { 0 })'):
            with self.subTest(args=args):
                self.check(PRELUDE + '''fn sample(view: &Handle, other: int) -> int { return (+ view.fd other) }
shadow sample { assert true }
''' + f'fn main() -> int {{ let mut h: Handle = Handle {{ fd: 1 }} return (sample {args}) }}\nshadow main {{ assert true }}\n', True)
        self.check(PRELUDE + '''fn sample(first: int, view: &Handle) -> int { return (+ first view.fd) }
shadow sample { assert true }
fn main() -> int { let h: Handle = Handle { fd: 1 } return (sample (consume h) &h) }
shadow main { assert true }
''', True)

    def test_borrow_requires_live_explicit_named_owner(self):
        for statements in (
            'let h: Handle = Handle { fd: 1 } return (read h)',
            'return (read &Handle { fd: 1 })',
            'let h: Handle = Handle { fd: 1 } let alias = &h return (consume h)',
            'let h: Handle = Handle { fd: 1 } let used: int = (consume h) return (read &h)',
            'let h: Handle = Handle { fd: 1 } return (read &mut h)',
            'let h: Handle = Handle { fd: 1 } let callback = read return (callback &h)',
        ):
            with self.subTest(statements=statements):
                self.check(PRELUDE + f'fn main() -> int {{ {statements} }}\nshadow main {{ assert true }}\n', True)

    def test_bytecode_boundary_preserves_prior_artifact(self):
        source = PRELUDE + 'fn main() -> int { let h: Handle = Handle { fd: 7 } assert (== (read &h) 7) return (consume h) }\nshadow main { assert true }\n'
        for compiler, flags in [('nano_virt', ['--emit-nvm']), ('nanoc_stage1', ['--emit-nvm']), ('nanoc_stage2', ['--emit-nvm'])]:
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix='nano-borrow-ir-') as directory:
                path, output = Path(directory)/'input.nano', Path(directory)/'prior.nvm'
                path.write_text(source)
                output.write_text('prior artifact')
                result = subprocess.run([ROOT/'bin'/compiler, path, *flags, '-o', output], cwd=ROOT,
                                        capture_output=True, text=True, timeout=180)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(output.read_text(), 'prior artifact')

    def test_imported_same_spelling_does_not_alias_local_owner(self):
        self.check('module "other.nano" as other\n' + PRELUDE +
                   'fn main() -> int { let h: other.Handle = other.Handle { fd: 1 } return (read &h) }\n'
                   'shadow main { assert true }\n', True,
                   {'other.nano': 'pub struct Handle { fd: int }\n'})

    def test_nominal_identity_not_field_shape(self):
        self.check(PRELUDE + '''resource struct Other { fd: int }
fn main() -> int { let h: Other = Other { fd: 1 } return (read &h) }
shadow main { assert true }
''', True)

if __name__ == '__main__':
    unittest.main()
