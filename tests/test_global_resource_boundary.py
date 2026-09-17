"""I reject unsupported global owners without weakening ordinary globals."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests import test_affine_generic_identity as generic

RESOURCE = 'resource struct Handle { fd: int }\n'
ENDING = 'fn main() -> int { return 0 }\nshadow main { assert (== (main) 0) }\n'

class GlobalResourceBoundary(unittest.TestCase):
    def check(self, source, accepted=False, modules=None):
        for compiler in generic.COMPILERS:
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix='nano-global-owner-') as directory:
                work = Path(directory)
                for name, contents in (modules or {}).items():
                    (work / name).write_text(contents)
                program, output = work / 'main.nano', work / 'program'
                program.write_text(source)
                output.write_bytes(b'prior artifact')
                result = subprocess.run([str(generic.COMPILER_ROOT / compiler), str(program), '-o', str(output)], cwd=generic.ROOT, capture_output=True, text=True, timeout=120)
                messages = result.stdout + result.stderr
                if accepted:
                    self.assertEqual(result.returncode, 0, messages)
                    run = subprocess.run([str(output)], capture_output=True, text=True, timeout=10)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                else:
                    self.assertGreater(result.returncode, 0, messages)
                    self.assertIn('global resource ownership is not supported', messages)
                    self.assertEqual(output.read_bytes(), b'prior artifact')

    def test_immutable_record_rejected(self):
        self.check(RESOURCE + 'let owner: Handle = Handle { fd: 7 }\n' + ENDING)

    def test_mutable_record_rejected(self):
        self.check(RESOURCE + 'let mut owner: Handle = Handle { fd: 7 }\n' + ENDING)

    def test_inferred_resource_rejected(self):
        self.check(RESOURCE + 'let owner = Handle { fd: 7 }\n' + ENDING)

    def test_nested_record_rejected(self):
        self.check(RESOURCE + 'struct Outer { owner: Handle }\nlet owner: Outer = Outer { owner: Handle { fd: 7 } }\n' + ENDING)

    def test_record_with_generic_resource_payload_rejected(self):
        self.check(RESOURCE + 'union Box<T> { Some { value: T }, None {} }\nstruct Outer { boxed: Box<Handle> }\nlet owner: Outer = Outer { boxed: Box.None {} }\n' + ENDING)

    def test_nested_record_generic_chain_rejected(self):
        self.check(RESOURCE + 'union Box<T> { Some { value: T }, None {} }\nstruct Inner { boxed: Box<Handle> }\nstruct Outer { inner: Inner }\nlet owner: Outer = Outer { inner: Inner { boxed: Box.None {} } }\n' + ENDING)

    def test_record_generic_collection_rejected(self):
        self.check(RESOURCE + 'union Box<T> { Some { value: T }, None {} }\nstruct Outer { boxed: Box<array<Handle>> }\nlet owner: Outer = Outer { boxed: Box.None {} }\n' + ENDING)

    def test_fixed_union_rejected(self):
        self.check(RESOURCE + 'union Choice { Some { owner: Handle }, None {} }\nlet owner: Choice = Choice.Some { owner: Handle { fd: 7 } }\n' + ENDING)

    def test_generic_union_rejected(self):
        self.check(RESOURCE + 'union Box<T> { Some { value: T }, None {} }\nlet owner: Box<Handle> = Box.Some { value: Handle { fd: 7 } }\n' + ENDING)

    def test_empty_resource_union_rejected(self):
        self.check(RESOURCE + 'union Box<T> { Some { value: T }, None {} }\nlet owner: Box<Handle> = Box.None {}\n' + ENDING)

    def test_resource_collection_rejected(self):
        self.check(RESOURCE + 'let owners: array<Handle> = []\n' + ENDING)

    def test_imported_resource_global_rejected(self):
        module = RESOURCE + 'let owner: Handle = Handle { fd: 7 }\npub fn answer() -> int { return 0 }\nshadow answer { assert (== (answer) 0) }\n'
        self.check('module "owner.nano" as owner\nfn main() -> int { return (owner.answer) }\nshadow main { assert (== (main) 0) }\n', modules={'owner.nano': module})

    def test_ordinary_globals_execute(self):
        self.check('''union Box<T> { Some { value: T }, None {} }
let boxed: Box<int> = Box.Some { value: 7 }
let label: string = "kept"
let mut count: int = 0
fn main() -> int { set count 1 assert (== label "kept") match boxed { Some(p) => { return (- (+ p.value count) 8) } None(n) => { return 1 } } }
shadow main { assert (== (main) 0) }
''', True)

    def test_unused_resource_argument_is_ordinary(self):
        self.check(RESOURCE + '''union Marker<T> { Mark { number: int } }
let marker: Marker<Handle> = Marker.Mark { number: 7 }
fn main() -> int { match marker { Mark(p) => { return (- p.number 7) } } }
shadow main { assert (== (main) 0) }
''', True)

    def test_local_owner_is_not_global(self):
        self.check(RESOURCE + '''let owner: int = 7
fn consume(owner: Handle) -> int { let Handle { fd } = owner return fd }
shadow consume { assert (== (consume Handle { fd: 3 }) 3) }
fn main() -> int { let owner: Handle = Handle { fd: 7 } return (- (consume owner) 7) }
shadow main { assert (== (main) 0) }
''', True)

if __name__ == '__main__':
    unittest.main()
