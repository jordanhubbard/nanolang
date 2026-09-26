"""I preserve match result context, selection effects and enclosing control flow."""
from pathlib import Path
import os
import tempfile
import unittest
from tests import test_match_aggregate_results as original
from tests import test_source_record_unions as source

ROOT = source.ROOT
EMITTER = Path(os.environ.get('NANO_MATCH_EMITTER', ROOT / 'bin/nanoisa_emit'))


def original_cases():
    cases = []
    old = original.generic.GenericAffineIdentity.check
    try:
        original.generic.GenericAffineIdentity.check = lambda self, text, accepted, modules=None: cases.append(text)
        case = original.MatchAggregateResults()
        case.test_generic_union_result()
        case.test_integer_selector_union_result()
        case.test_scalar_match_result()
    finally:
        original.generic.GenericAffineIdentity.check = old
    return cases


class CanonicalMatchResults(unittest.TestCase):
    command = source.SourceRecordUnions.command
    checked = source.SourceRecordUnions.checked
    execute = source.SourceRecordUnions.execute

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='nano-match-shadow-driver-')
        work = Path(cls.temporary.name)
        override = os.environ.get('NANO_MATCH_SHADOW_EMITTER')
        if override:
            cls.shadows = Path(override)
        else:
            driver = work / 'driver.nano'
            driver.write_text((ROOT / 'tests/nanoisa/fixtures/shadow_module_driver.nano.txt').read_text())
            cls.shadows = work / 'driver'
            cls().checked([ROOT / 'bin/nanoc_c', driver, '-o', cls.shadows])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def paired(self, text):
        with tempfile.TemporaryDirectory(prefix='nano-match-results-') as d:
            work = Path(d)
            program, module = work / 'source.nano', work / 'module.nvm'
            program.write_text(text)
            for emitter in (source.C_EMITTER, EMITTER):
                with self.subTest(emitter=emitter):
                    self.checked([emitter, program, '--emit-nvm', '-o', module])
                    self.execute(work, module)
            assembly = work / 'shadows.nasm'
            assembly.write_text(self.checked([self.shadows, program, '0', 'raw']).stdout)
            self.checked([ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module])
            self.execute(work, module)

    def test_original_aggregate_and_integer_results(self):
        cases = original_cases()
        self.assertEqual(len(cases), 3)
        for i, text in enumerate(cases):
            with self.subTest(case=i):
                self.paired(text)

    def test_previously_admitted_nested_union_fixture(self):
        self.paired('union Inner { Empty {} } union Box { Some { value: Inner } } fn main() -> int { let x: Box = Box.Some { value: Inner.Empty {} } return 0 } shadow main { assert true }')

    def test_wrong_nested_union_preserves_prior_output(self):
        declarations = 'union Inner { Empty {} } union Other { Empty {} } union Box { Some { value: Inner } } '
        for body in ('let x: Box = Box.Some { value: Other.Empty {} }',
                     'let other: Other = Other.Empty {} let x: Box = Box.Some { value: other }'):
            with self.subTest(body=body), tempfile.TemporaryDirectory(prefix='nano-nested-refusal-') as d:
                work = Path(d)
                program, output = work / 'source.nano', work / 'prior.nvm'
                program.write_text(declarations + 'fn main() -> int { ' + body + ' return 0 } shadow main { assert true }')
                for emitter in (source.C_EMITTER, EMITTER):
                    with self.subTest(emitter=emitter):
                        output.write_bytes(b'prior artifact')
                        result = self.command([emitter, program, '--emit-nvm', '-o', output])
                        self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                        self.assertEqual(output.read_bytes(), b'prior artifact')

    def test_integer_guards_and_wildcards_keep_written_effect_order(self):
        self.paired('''let mut trace: int = 0
fn mark(value: int) -> int { set trace (+ (* trace 10) value) return value }
shadow mark { set trace 0 assert (== (mark 1) 1) assert (== trace 1) }
fn guard(digit: int, value: bool) -> bool { let ignored: int = (mark digit) return value }
shadow guard { set trace 0 assert (guard 2 true) assert (== trace 2) }
fn choose(early: bool, later: bool) -> int { return match (mark 1) {
 _ if (guard 2 early) => (mark 3)
 2 if (guard 9 true) => (mark 9)
 1 if (guard 4 false) => (mark 9)
 _ if (guard 5 later) => (mark 6)
 1 if true => (mark 7)
 _ => (mark 8)
} }
shadow choose { set trace 0 assert (== (choose false false) 7) assert (== trace 12457) }
fn main() -> int {
 set trace 0 assert (== (choose true true) 3) assert (== trace 123)
 set trace 0 assert (== (choose false true) 6) assert (== trace 12456)
 set trace 0 assert (== (choose false false) 7) assert (== trace 12457)
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_union_wildcard_scope_and_nested_block_tail(self):
        self.paired('''union Choice { Some { value: int }, None {} }
fn choose(value: Choice) -> int {
 let payload: int = 3
 let result = match value {
  _ if false => { 99 }
  Some(payload) if false => { payload.value }
  _ if (!= payload 3) => { 999 }
  Some(payload) => { let local: int = match 1 { 1 => { 2 } _ => { 0 } } (+ payload.value local) }
  _ if true => { payload }
 }
 return (+ result payload)
}
shadow choose { assert (== (choose Choice.Some { value: 7 }) 12) assert (== (choose Choice.None {}) 6) }
fn main() -> int { assert (== (choose Choice.Some { value: 7 }) 12) assert (== (choose Choice.None {}) 6) return 0 }
shadow main { assert (== (main) 0) }
''')

    def test_block_returns_exit_enclosing_function(self):
        self.paired('''fn choose(value: int) -> int {
 let result: int = match value {
  1 => { if (== value 1) { return 7 } 99 }
  2 => { return 8 }
  _ => { let local: int = 3 (+ local 1) }
 }
 return (+ result 10)
}
shadow choose { assert (== (choose 1) 7) assert (== (choose 2) 8) assert (== (choose 0) 14) }
fn main() -> int { assert (== (choose 1) 7) assert (== (choose 2) 8) assert (== (choose 0) 14) return 0 }
shadow main { assert (== (main) 0) }
''')

    def test_statement_match_preserves_loop_control(self):
        self.paired('''fn main() -> int {
 let mut count: int = 0
 let mut total: int = 0
 while (< count 4) {
  set count (+ count 1)
  match count { 1 => { continue } 3 => { break } _ => { set total (+ total count) } }
 }
 assert (== count 3) assert (== total 2)
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_record_and_nested_array_results(self):
        self.paired('''struct Point { x: int }
fn point(value: int) -> Point { return match value { 1 => { Point { x: 7 } } _ => Point { x: 3 } } }
shadow point { assert (== (point 1).x 7) }
fn rows(value: int) -> array<array<Point>> { return match value { 1 => [[Point { x: 7 }], []] _ => [] } }
shadow rows { let values = (rows 1) let p: Point = (at (at values 0) 0) assert (== p.x 7) }
fn main() -> int { let p: Point = (point 0) assert (== p.x 3) let values = (rows 1) let first: Point = (at (at values 0) 0) assert (== first.x 7) assert (== (array_length (at values 1)) 0) assert (== (array_length (rows 0)) 0) return 0 }
shadow main { assert (== (main) 0) }
''')

    def test_late_join_facts_keep_checked_heap_boundaries(self):
        choose = ('.function choose 1 2 0 int 1\n'
                  'LOAD_LOCAL 0\nAGG_TAG\nPUSH_I64 0\nEQ\nJMP_FALSE empty\n'
                  'LOAD_LOCAL 0\nAGG_GET 0\nJMP joined\nempty:\nPUSH_I64 7\n'
                  'joined:\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nRET\n.end\n'
                  '.parameters choose union\n')
        for heap in (False, True):
            for caller_first in (False, True):
                with self.subTest(heap=heap, caller_first=caller_first), tempfile.TemporaryDirectory(prefix='nano-match-join-') as d:
                    work = Path(d)
                    assembly, module, output = (work / n for n in ('input.nasm', 'input.nvm', 'prior.c'))
                    entry = ('.function main 0 0 0 int 1\nPUSH_I64 7\n'
                             + ('ARR_LITERAL 1 1\n' if heap else '')
                             + 'AGG_PACK 1 0 0 1\nCALL choose\nPUSH_I64 7\nEQ\nASSERT\n'
                             'AGG_PACK 1 0 1 0\nCALL choose\nPUSH_I64 7\nEQ\nASSERT\n'
                             'PUSH_I64 0\nRET\n.end\n')
                    assembly.write_text('.types 0 0 1\n.entry main\n' +
                                        (entry + choose if caller_first else choose + entry))
                    self.checked([ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module])
                    if heap:
                        self.checked([source.TRANSLATOR, module, '-o', output])
                        vm = self.command([ROOT / 'bin/nano_vm', module])
                        self.assertNotEqual(vm.returncode, 0, vm.stdout + vm.stderr)
                        native = work / 'program'
                        compiler = os.environ.get('NANO_NATIVE_TEST_CC') or 'cc'
                        self.checked([compiler, '-x', 'c', '-std=c11', '-O1',
                                      '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                                      output, '-o', native])
                        result = self.command([native])
                        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertIn('native invariant', result.stderr)
                        self.assertNotIn('ERROR: AddressSanitizer', result.stderr)
                        self.assertNotIn('runtime error:', result.stderr)
                    else:
                        self.execute(work, module)

    def test_refusals_preserve_prior_output(self):
        for name, expression in (
            ('non_total', 'match 1 { 1 => 7 }'),
            ('conditional_total', 'match 1 { 1 => 7 _ if false => 8 }'),
            ('unreachable', 'match 1 { _ => 7 1 => 8 }'),
            ('literal_true_unreachable', 'match 1 { _ if true => 7 1 => 8 }'),
            ('non_bool_guard', 'match 1 { 1 if 7 => 7 _ => 8 }'),
            ('wrong_arm_type', 'match 1 { 1 => 7 _ => false }'),
            ('wrong_block_tail', 'match 1 { 1 => { 7 } _ => { false } }'),
            ('empty_block', 'match 1 { 1 => { } _ => 8 }'),
        ):
            with self.subTest(case=name), tempfile.TemporaryDirectory(prefix='nano-match-refusal-') as d:
                work = Path(d)
                program, output = work / 'source.nano', work / 'prior.nvm'
                program.write_text('fn main() -> int { return ' + expression + ' }\nshadow main { assert true }\n')
                for emitter in (source.C_EMITTER, EMITTER):
                    with self.subTest(emitter=emitter):
                        output.write_bytes(b'prior artifact')
                        result = self.command([emitter, program, '--emit-nvm', '-o', output])
                        self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                        self.assertEqual(output.read_bytes(), b'prior artifact')


if __name__ == '__main__':
    unittest.main()
