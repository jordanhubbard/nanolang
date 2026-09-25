"""I preserve finite record/union source shapes and declared constructor context."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile
import unittest
from tests import test_union_literal_context as arrays
from tests import test_native_generic_record_fields as records

ROOT = Path(__file__).resolve().parents[1]
EMITTER = Path(os.environ.get('NANO_RECORD_SOURCE_EMITTER', ROOT / 'bin/nanoisa_emit'))
TRANSLATOR = Path(os.environ.get('NANO_RECORD_SOURCE_TRANSLATOR', ROOT / 'bin/nvm2c'))
C_EMITTER = Path(os.environ.get('NANO_RECORD_C_EMITTER', ROOT / 'bin/nano_virt'))


def original_cases():
    cases = []
    array_case = arrays.UnionLiteralContext()
    for name in ('test_nongeneric_record_array', 'test_generic_record_array', 'test_nested_generic_record_array'):
        array_case.check = lambda source, name=name: cases.append((name, source))
        getattr(array_case, name)()
    record_case = records.NativeGenericRecordLayout()
    for name in ('test_direct_field_match_executes', 'test_direct_field_match_expression_executes',
                 'test_empty_inline_field_executes', 'test_nonempty_field_payload_executes',
                 'test_nested_generic_field_executes', 'test_forward_union_declaration_executes',
                 'test_distinct_instances_execute', 'test_two_argument_result_field_executes',
                 'test_nested_record_executes'):
        record_case.check = lambda source, accepted=True, modules=None, name=name: cases.append((name, source))
        getattr(record_case, name)()
    return cases


class SourceRecordUnions(unittest.TestCase):
    def command(self, args):
        return subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True, text=True,
                              timeout=120, env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1',
                                                'UBSAN_OPTIONS': 'halt_on_error=1'})

    def checked(self, args):
        result = self.command(args)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def execute(self, work, module):
        self.checked([ROOT / 'bin/nano_vm', '--verify-only', module])
        self.checked([ROOT / 'bin/nano_vm', module])
        native, binary = work / 'output.c', work / 'program'
        self.checked([TRANSLATOR, module, '-o', native])
        compiler = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC') or os.environ.get('CC') or 'cc')
        self.checked([*compiler, '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror',
                      '-fsanitize=address,undefined', '-fno-sanitize-recover=all', native, '-o', binary])
        self.checked([binary])

    def test_original_standalone_sources(self):
        cases = original_cases()
        self.assertEqual(len(cases), 12)
        for name, source in cases:
            for emitter in (C_EMITTER, EMITTER):
                with self.subTest(case=name, emitter=emitter), tempfile.TemporaryDirectory(prefix='nano-record-unions-') as d:
                    work = Path(d)
                    program, module = work / 'source.nano', work / 'source.nvm'
                    program.write_text(source)
                    self.checked([emitter, program, '--emit-nvm', '-o', module])
                    self.execute(work, module)

    def test_c_compound_union_field_context(self):
        declarations = 'union Box<T> { Some { value: T }, None {} } struct Outer { boxed: Box<int> } '
        expressions = (
            'match 1 { 1 => Box.Some { value: 7 }, _ => Box.None {} }',
            'if true { Box.Some { value: 7 } } else { Box.None {} }',
            'if false { Box.None {} } else { Box.Some { value: 7 } }',
            'cond ((== 1 1) Box.Some { value: 7 }) (else Box.None {})',
            'match Box.Some { value: 7 } { Some(p) => Box.Some { value: p.value }, None(n) => Box.None {} }',
        )
        for expression in expressions:
            for wrong in (False, True):
                with self.subTest(expression=expression, wrong=wrong), tempfile.TemporaryDirectory(prefix='nano-record-compound-') as d:
                    work = Path(d)
                    program, module = work / 'source.nano', work / 'source.nvm'
                    value = expression.replace('Box.None {}', 'Box.Some { value: "wrong" }') if wrong else expression
                    program.write_text(declarations + 'fn main() -> int { let outer: Outer = Outer { boxed: ' + value + ' } match outer.boxed { Some(p) => { assert (== p.value 7) } None(n) => { assert false } } return 0 }\nshadow main { assert (== (main) 0) }\n')
                    module.write_bytes(b'prior artifact')
                    result = self.command([C_EMITTER, program, '--emit-nvm', '-o', module])
                    if wrong:
                        self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                        self.assertEqual(module.read_bytes(), b'prior artifact')
                    else:
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.execute(work, module)

    def test_original_record_array_refusals(self):
        cases = []
        original = arrays.UnionLiteralContext()
        original.reject = cases.append
        original.test_wrong_record_retains_prior_output()
        original.test_generic_wrong_record_retains_prior_output()
        for source in cases:
            for emitter in (C_EMITTER, EMITTER):
                with self.subTest(source=source, emitter=emitter), tempfile.TemporaryDirectory(prefix='nano-record-array-refusal-') as d:
                    work = Path(d)
                    program, module = work / 'source.nano', work / 'source.nvm'
                    program.write_text(source)
                    module.write_bytes(b'prior artifact')
                    result = self.command([emitter, program, '--emit-nvm', '-o', module])
                    self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(module.read_bytes(), b'prior artifact')

    def test_constructor_context_preserves_rejections(self):
        declarations = ('union Box<T> { Some { value: T }, None {} } '
                        'union Other<T> { Some { value: T }, None {} } '
                        'struct Outer { boxed: Box<int> } ')
        for name, body in (
            ('other_declaration', 'let outer: Outer = Outer { boxed: Other.Some { value: 7 } }'),
            ('wrong_payload', 'let outer: Outer = Outer { boxed: Box.Some { value: "wrong" } }'),
            ('unknown_field', 'let outer: Outer = Outer { boxed: Box.Some { unknown: 7 } }'),
            ('missing_field', 'let outer: Outer = Outer { boxed: Box.Some {} }'),
            ('wrong_instance', 'let text: Box<string> = Box.Some { value: "wrong" } let outer: Outer = Outer { boxed: text }'),
        ):
            for emitter in (C_EMITTER, EMITTER):
                with self.subTest(case=name, emitter=emitter), tempfile.TemporaryDirectory(prefix='nano-record-union-refusal-') as d:
                    work = Path(d)
                    program, module = work / 'source.nano', work / 'source.nvm'
                    program.write_text(declarations + 'fn main() -> int { ' + body + ' return 0 }\nshadow main { assert (== (main) 0) }\n')
                    module.write_bytes(b'prior artifact')
                    result = self.command([emitter, program, '--emit-nvm', '-o', module])
                    self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(module.read_bytes(), b'prior artifact')


class SourceRecordUnionShadows(unittest.TestCase):
    command = SourceRecordUnions.command
    checked = SourceRecordUnions.checked
    execute = SourceRecordUnions.execute

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='nano-record-shadow-driver-')
        work = Path(cls.temporary.name)
        override = os.environ.get('NANO_RECORD_SHADOW_EMITTER')
        if override:
            cls.driver = Path(override)
        else:
            source = work / 'driver.nano'
            source.write_text((ROOT / 'tests/nanoisa/fixtures/shadow_module_driver.nano.txt').read_text())
            cls.driver = work / 'driver'
            cls().checked([ROOT / 'bin/nanoc_c', source, '-o', cls.driver])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_original_standalone_shadows(self):
        for name, source in original_cases():
            with self.subTest(case=name), tempfile.TemporaryDirectory(prefix='nano-record-union-shadow-') as d:
                work = Path(d)
                program, assembly, module = (work / n for n in ('source.nano', 'shadow.nasm', 'shadow.nvm'))
                program.write_text(source)
                result = self.checked([self.driver, program, '0', 'raw'])
                assembly.write_text(result.stdout)
                self.checked([ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module])
                self.execute(work, module)


if __name__ == '__main__':
    unittest.main()
