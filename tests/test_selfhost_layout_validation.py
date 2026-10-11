"""I reject impossible by-value declarations before native or NanoISA publication."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get('NANOLANG_SELFHOST_COMPILER', ROOT/'bin/nanoc_stage2'))
MAIN = '\nfn main()->int{return 0}\nshadow main {assert (== (main) 0)}\n'
CYCLES = [
    'struct Loop { next:Loop }',
    'struct Left { right:Right } struct Right { left:Left }',
    'union Loop { Some { next:Loop }, None {} }',
    'struct Loop { value:Choice } union Choice { Some { value:Loop } }',
    'struct Loop { pair:(int,Loop) }',
    'union Box<T> { Some { value:T }, None {} } struct Loop { box:Box<Loop> }',
]


class SelfhostLayoutValidation(unittest.TestCase):
    def compile(self, source, mode, output, work):
        path = work/'main.nano'
        path.write_text(source+MAIN)
        return subprocess.run([str(COMPILER), str(path), *mode, '-o', str(output)],
                              cwd=ROOT, text=True, capture_output=True, timeout=120)

    def refuse(self, source, diagnostic, files=None):
        for mode in ([], ['--emit-nvm']):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                for name, contents in (files or {}).items():
                    (work/name).write_text(contents)
                output = work/'output'
                output.write_bytes(b'prior artifact')
                result = self.compile(source, mode, output, work)
                self.assertEqual(result.returncode, 1, result.stdout+result.stderr)
                self.assertIn(diagnostic, result.stdout+result.stderr)
                self.assertNotIn('C compilation failed', result.stdout+result.stderr)
                self.assertEqual(output.read_bytes(), b'prior artifact')

    def test_cycles_preserve_both_artifact_kinds(self):
        for source in CYCLES:
            with self.subTest(source=source):
                self.refuse(source, 'cyclic by-value')

    def test_expanding_generic_layout_has_a_distinct_diagnostic(self):
        self.refuse('union Grow<T> { Some { value:Grow<array<T>> } }',
                    'layout expansion beyond 128')

    def test_indirect_fields_and_unused_generic_parameters_are_legal(self):
        sources = [
            'struct Node { children:array<Node>, callback:fn(Node)->Node }',
            'struct Node { children:List<Node> }',
            'union Box<T> { Some { values:array<T> }, None {} } struct Node { value:Box<Node> }',
            'union Phantom<T> { None {} } struct Node { value:Phantom<Node> }',
            'union Box<T> { Some { value:T } } struct Plain { nested:Box<Box<int>> }',
        ]
        for source in sources:
            for mode in ([], ['--emit-nvm']):
                with self.subTest(source=source, mode=mode), tempfile.TemporaryDirectory() as tmp:
                    work = Path(tmp)
                    output = work/'output'
                    result = self.compile(source, mode, output, work)
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                    command = [str(output)] if not mode else [str(ROOT/'bin/nano_vm'), str(output)]
                    run = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=15)
                    self.assertEqual(run.returncode, 0, run.stdout+run.stderr)

    def test_imported_equal_name_owners_keep_the_cycle_local(self):
        self.refuse('module "a.nano" as a\nmodule "b.nano" as b', 'cyclic by-value',
                    files={'a.nano': 'pub struct Loop { value:int }',
                           'b.nano': 'pub struct Loop { next:Loop }'})

    def test_deep_acyclic_records_do_not_consume_generic_expansion_budget(self):
        source = 'struct Item0 { value:int }\n'+''.join(
            f'struct Item{i} {{ value:Item{i-1} }}\n' for i in range(1, 145))
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            output = work/'output'
            result = self.compile(source, ['--emit-nvm'], output, work)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            run = subprocess.run([str(ROOT/'bin/nano_vm'), str(output)], cwd=ROOT,
                                 text=True, capture_output=True, timeout=15)
            self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
