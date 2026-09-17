"""I mutate a caller-owned scalar field only through an available exclusive reference."""
from pathlib import Path
import os
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILERS = os.environ.get('NANO_BORROW_COMPILERS', 'nanoc_c,nanoc_stage1,nanoc_stage2').split(',')
PRELUDE = '''resource struct Handle { fd: int }
union Choice { Some { value: int }, None {} }
fn consume(value: Handle) -> int { let Handle { fd } = value return fd }
shadow consume { assert (== (consume Handle { fd: 7 }) 7) }
fn read(view: &Handle) -> int { return view.fd }
shadow read { let h: Handle = Handle { fd: 7 } assert (== (read &h) 7) assert (== (consume h) 7) }
fn bump(view: &mut Handle) -> int { set view.fd (+ view.fd 1) return view.fd }
shadow bump { let mut h: Handle = Handle { fd: 7 } assert (== (bump &mut h) 8) assert (== (consume h) 8) }
'''

class ExclusiveBorrows(unittest.TestCase):
    def check(self, source, reject=False):
        for compiler in COMPILERS:
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix='nano-exclusive-') as directory:
                path, output = Path(directory)/'input.nano', Path(directory)/'output'
                path.write_text(source)
                output.write_text('prior artifact')
                result = subprocess.run([ROOT/'bin'/compiler, path, '--keep-c', '-o', output], cwd=ROOT,
                                        capture_output=True, text=True, timeout=180)
                diagnostic = result.stdout + result.stderr
                if reject:
                    self.assertNotEqual(result.returncode, 0, diagnostic)
                    self.assertEqual(output.read_text(), 'prior artifact')
                    for marker in ('C compilation failed', 'Parsing error', 'Parse error', 'Failed to parse', 'Segmentation fault'):
                        self.assertNotIn(marker, diagnostic)
                else:
                    self.assertEqual(result.returncode, 0, diagnostic)
                    generated = output.with_suffix('.c')
                    if not generated.exists():
                        match = re.search(r'I kept generated C in (.+)', diagnostic)
                        self.assertIsNotNone(match, diagnostic)
                        generated = Path(match.group(1))
                    self.assertRegex(generated.read_text(), r'nl_bump\(nl_Handle\*')
                    ran = subprocess.run([output], capture_output=True, text=True, timeout=30)
                    self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)

    def test_raw_nanoisa_field_target_refusal_preserves_output(self):
        with tempfile.TemporaryDirectory(prefix='nano-raw-field-place-') as directory:
            work = Path(directory)
            driver, binary = work/'driver.nano', work/'driver'
            driver.write_text('''import "src_nano/compiler/nanoisa_codegen.nano"
extern fn get_argc() -> int
extern fn get_argv(index: int) -> string
fn main() -> int {
 if (!= (get_argc) 3) { return 0 }
 let assembly: string = (nanoisa_emit_nasm (file_read (get_argv 1)))
 if (== assembly "") { (println (nanoisa_last_lowering_error)) return 1 }
 let result: int = (file_write (get_argv 2) assembly)
 return 0
}
shadow main { assert (== (main) 0) }
''')
            built = subprocess.run([ROOT/'bin/nanoc_c', driver, '-o', binary], cwd=ROOT,
                                   capture_output=True, text=True, timeout=180)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            source, output = work/'input.nano', work/'output.nasm'
            source.write_text('struct Item { value: int } fn main() -> int { '
                              'let mut item: Item = Item { value: 1 } set item.value 2 return 0 }')
            output.write_text('prior artifact')
            rejected = subprocess.run([binary, source, output], cwd=ROOT,
                                      capture_output=True, text=True, timeout=30)
            self.assertEqual(rejected.returncode, 1, rejected.stdout + rejected.stderr)
            self.assertIn('reference IR', rejected.stdout)
            self.assertEqual(output.read_text(), 'prior artifact')
            source.write_text('fn main() -> int { let mut value: int = 1 set value 2 return value }')
            accepted = subprocess.run([binary, source, output], cwd=ROOT,
                                      capture_output=True, text=True, timeout=30)
            self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
            self.assertIn('STORE_LOCAL', output.read_text())

    def test_mutation_is_visible_to_caller_and_forwarded_reference(self):
        self.check(PRELUDE + '''fn forward(view: &mut Handle) -> int { let value: int = (bump &mut view) assert (== (read &view) value) return value }
shadow forward { let mut h: Handle = Handle { fd: 1 } assert (== (forward &mut h) 2) assert (== (consume h) 2) }
fn main() -> int {
 let mut h: Handle = Handle { fd: 11 }
 assert (== (bump &mut h) 12)
 assert (== (read &h) 12)
 assert (== (forward &mut h) 13)
 assert (== h.fd 13)
 assert (== (consume h) 13)
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_distinct_owners_and_scalar_field_kinds(self):
        self.check(PRELUDE + '''resource struct Settings { active: bool, ratio: float }
fn tune(view: &mut Settings) -> void { set view.active (not view.active) set view.ratio (+ view.ratio 0.5) }
shadow tune { let mut value: Settings = Settings { active: false, ratio: 1.0 } (tune &mut value) let Settings { active, ratio } = value assert active assert (== ratio 1.5) }
fn pair(a: &mut Handle, b: &mut Handle) -> int { set a.fd (+ a.fd 1) set b.fd (+ b.fd 2) return (+ a.fd b.fd) }
shadow pair { let mut a: Handle = Handle { fd: 1 } let mut b: Handle = Handle { fd: 2 } assert (== (pair &mut a &mut b) 6) assert (== (+ (consume a) (consume b)) 6) }
fn main() -> int {
 let mut a: Handle = Handle { fd: 1 }
 let mut b: Handle = Handle { fd: 2 }
 assert (== (pair &mut a &mut b) 6)
 assert (== (bump &mut a) 3)
 assert (== (+ (consume a) (consume b)) 7)
 let mut value: Settings = Settings { active: false, ratio: 1.0 }
 (tune &mut value)
 let Settings { active, ratio } = value
 assert active
 assert (== ratio 1.5)
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_overlapping_arguments_and_later_reads_are_rejected(self):
        cases = [
            ('a: &mut Handle, b: &mut Handle', '&mut h &mut h'),
            ('a: &Handle, b: &mut Handle', '&h &mut h'),
            ('a: &mut Handle, b: &Handle', '&mut h &h'),
            ('a: &mut Handle, b: int', '&mut h h.fd'),
            ('a: &mut Handle, b: int', '&mut h (consume h)'),
            ('a: &mut Handle, b: int', '&mut h (read &h)'),
        ]
        for params, args in cases:
            with self.subTest(args=args):
                self.check(PRELUDE + f'fn sample({params}) -> int {{ return a.fd }}\nshadow sample {{ assert true }}\n'
                           f'fn main() -> int {{ let mut h: Handle = Handle {{ fd: 1 }} return (sample {args}) }}\nshadow main {{ assert true }}\n', True)

    def test_mutation_needs_an_available_exclusive_capability(self):
        cases = [
            ('view: &Handle', 'set view.fd 2 return 0'),
            ('view: &mut Handle', 'set view.missing 2 return 0'),
            ('view: &mut Handle', 'set view.fd true return 0'),
            ('view: &Handle', 'return (bump &mut view)'),
            ('view: &mut Handle', 'return (consume view)'),
            ('view: &mut Handle', 'let alias: Handle = view return (consume alias)'),
            ('view: &mut Handle', 'let Handle { fd } = view return fd'),
            ('view: &mut Handle', 'set view Handle { fd: 2 } return 0'),
        ]
        for params, body in cases:
            with self.subTest(body=body, params=params):
                self.check(PRELUDE + f'fn invalid({params}) -> int {{ {body} }}\nshadow invalid {{ assert true }}\n'
                           'fn main() -> int { return 0 }\nshadow main { assert true }\n', True)

    def test_immutable_moved_or_stored_exclusive_reference_is_rejected(self):
        for body in (
            'let h: Handle = Handle { fd: 1 } return (bump &mut h)',
            'let mut h: Handle = Handle { fd: 1 } let used: int = (consume h) return (bump &mut h)',
            'let mut h: Handle = Handle { fd: 1 } let alias = &mut h return (consume h)',
            'return (bump &mut Handle { fd: 1 })',
            'let mut h: Handle = Handle { fd: 1 } set h.fd 2 return (consume h)',
        ):
            with self.subTest(body=body):
                self.check(PRELUDE + f'fn main() -> int {{ {body} }}\nshadow main {{ assert true }}\n', True)

    def test_borrowed_callbacks_are_rejected_before_publication(self):
        for name in ('read', 'bump'):
            for body in (f'let callback = {name} return 0', f'{name} return 0'):
                with self.subTest(name=name, body=body):
                    self.check(PRELUDE + f'fn main() -> int {{ {body} }}\nshadow main {{ assert true }}\n', True)
            self.check(PRELUDE + f'let callback = {name}\nfn main() -> int {{ return 0 }}\nshadow main {{ assert true }}\n', True)
            self.check(PRELUDE + f'fn invalid() -> fn(int)->int {{ return {name} }}\nshadow invalid {{ assert true }}\n'
                       'fn main() -> int { return 0 }\nshadow main { assert true }\n', True)
            self.check(PRELUDE + 'fn invoke(callback: fn(int)->int) -> int { return (callback 1) }\nshadow invoke { assert true }\n'
                       + f'fn main() -> int {{ return (invoke {name}) }}\nshadow main {{ assert true }}\n', True)

    def test_ordinary_and_owned_callbacks_remain_supported(self):
        self.check(PRELUDE + '''fn increment(value: int) -> int { return (+ value 1) }
shadow increment { assert (== (increment 7) 8) }
fn main() -> int {
 let callback: fn(int)->int = increment
 assert (== (callback 7) 8)
 let owned_callback: fn(Handle)->int = consume
 let mut h: Handle = Handle { fd: 7 }
 assert (== (bump &mut h) 8)
 assert (== (owned_callback h) 8)
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_bytecode_boundary_preserves_prior_artifact(self):
        source = PRELUDE + 'fn main() -> int { let mut h: Handle = Handle { fd: 7 } assert (== (bump &mut h) 8) return (consume h) }\nshadow main { assert true }\n'
        for compiler in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix='nano-exclusive-ir-') as directory:
                path, output = Path(directory)/'input.nano', Path(directory)/'prior.nvm'
                path.write_text(source)
                output.write_text('prior artifact')
                result = subprocess.run([ROOT/'bin'/compiler, path, '--emit-nvm', '-o', output], cwd=ROOT,
                                        capture_output=True, text=True, timeout=180)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(output.read_text(), 'prior artifact')

    def test_mutable_reference_cannot_escape_as_an_owned_return(self):
        self.check(PRELUDE + 'fn invalid(view: &mut Handle) -> Handle { return view }\nshadow invalid { assert true }\n'
                   'fn main() -> int { return 0 }\nshadow main { assert true }\n', True)

    def test_shared_hold_prevents_field_mutation_in_later_argument(self):
        self.check(PRELUDE + '''fn sample(view: &Handle, ignored: int) -> int { return view.fd }
shadow sample { assert true }
fn invalid(view: &mut Handle) -> int { let choice: Choice = Choice.Some { value: 0 } return (sample &view (match choice { Some(payload) => { set view.fd 8 0 } None(empty) => { 0 } })) }
shadow invalid { assert true }
fn main() -> int { return 0 }
shadow main { assert true }
''', True)

if __name__ == '__main__':
    unittest.main()
