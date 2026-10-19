"""I keep conditional array-read storage separate from exact shared record fields."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest

from tests.native_process import native_link_flags, run_captured

ROOT = Path(__file__).resolve().parents[1]


def source_text(values, checked, index=0, guarded=True):
    condition = ('(and (== checked 1) (== (array_length f.owner) 1))'
                 if guarded else '(== checked 1)')
    result = 'result.completed' if guarded else '(+ result.completed 1)'
    return f'''struct Fact {{ owner: array<int>, completed: int }}
struct Parser {{ functions: List<Fact> }}
fn update(p: Parser, index: int, checked: int) -> void {{
    let f: Fact = (list_Fact_get p.functions index)
    (list_Fact_set p.functions index Fact {{ owner: f.owner,
        completed: (cond ({condition} (at f.owner {index})) (else 0)) }})
}}
shadow update {{
    let functions: List<Fact> = (list_Fact_new)
    (list_Fact_push functions Fact {{ owner: [7], completed: 12 }})
    (update Parser {{ functions: functions }} 0 0)
    assert (== (list_Fact_get functions 0).completed 0)
}}
fn main() -> int {{
    let functions: List<Fact> = (list_Fact_new)
    (list_Fact_push functions Fact {{ owner: {values}, completed: 0 }})
    let p: Parser = Parser {{ functions: functions }}
    (update p 0 {checked})
    let result: Fact = (list_Fact_get functions 0)
    return {result}
}}
'''


def scalar_source_text(kind, value, zero, case):
    values = '[]' if case == 'empty' else '[' + value + ']'
    checked = 0 if case == 'unchecked' else 1
    index = 1 if case == 'raw-missing' else 0
    condition = ('(== checked 1)' if case == 'raw-missing' else
                 '(and (== checked 1) (== (array_length f.owner) 1))')
    expected = zero if case in ('empty', 'unchecked') else value
    return f'''struct Fact {{ owner: array<{kind}>, completed: {kind} }}
struct Parser {{ functions: List<Fact> }}
fn update(p: Parser, index: int, checked: int) -> void {{
    let f: Fact = (list_Fact_get p.functions index)
    (list_Fact_set p.functions index Fact {{ owner: f.owner,
        completed: (cond ({condition} (at f.owner {index})) (else {zero})) }})
}}
shadow update {{
    let functions: List<Fact> = (list_Fact_new)
    (list_Fact_push functions Fact {{ owner: [{value}], completed: {value} }})
    (update Parser {{ functions: functions }} 0 0)
    assert (== (list_Fact_get functions 0).completed {zero})
}}
fn main() -> int {{
    let functions: List<Fact> = (list_Fact_new)
    (list_Fact_push functions Fact {{ owner: {values}, completed: {zero} }})
    (update Parser {{ functions: functions }} 0 {checked})
    assert (== (list_Fact_get functions 0).completed {expected})
    return 0
}}
# I exercise main in the VM/native matrix; raw-missing intentionally fails.
shadow main {{ assert true }}
'''


class OptionalRecordJoin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='nano-optional-record-join-',
            dir=os.environ.get('NANO_OPTIONAL_RECORD_JOIN_REPORT_DIR')))
        print(f'I retain conditional record-join evidence at {cls.work}', flush=True)
        configured = os.environ.get('NANO_OPTIONAL_RECORD_JOIN_COMPILERS')
        cls.compilers = ([Path(path).resolve() for path in shlex.split(configured)] if configured else
                         [ROOT/'bin'/name for name in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2')])
        if not cls.compilers:
            raise ValueError('I require actual source producers for conditional record joins.')
        cls.cc = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', os.environ.get('CC', 'cc')))
        paths = cls.compilers + [ROOT/'bin/nano_vm', ROOT/'bin/nvm2c', ROOT/'bin/nano_aot_runtime.o']
        (cls.work/'providers.json').write_text(json.dumps([
            {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in paths], indent=2)+'\n')

    def run_retained(self, name, args, expected=0, build=False):
        args = list(map(str, args))
        (self.work/(name+'-command.json')).write_text(json.dumps(args, indent=2)+'\n')
        result = run_captured(args, cwd=ROOT, timeout=90, build=build, text=True,
            env={**os.environ, 'ASAN_OPTIONS': 'halt_on_error=1',
                 'UBSAN_OPTIONS': 'halt_on_error=1:print_stacktrace=1'})
        (self.work/(name+'.stdout')).write_text(result.stdout)
        (self.work/(name+'.stderr')).write_text(result.stderr)
        (self.work/(name+'-status.json')).write_text(json.dumps({'returncode': result.returncode})+'\n')
        if expected is not None:
            self.assertEqual(result.returncode, expected, result.stdout+result.stderr)
        return result

    def paired(self, case, text, expected=None, missing_error="requires two integers"):
        source = self.work/(case+'.nano')
        source.write_text(text)
        for number, compiler in enumerate(self.compilers):
            label = f'{case}-{number}-{compiler.name}'
            with self.subTest(case=case, compiler=str(compiler)):
                module, native_source = self.work/(label+'.nvm'), self.work/(label+'.c')
                self.run_retained(label+'-produce', [compiler, source, '--emit-nvm', '-o', module])
                self.run_retained(label+'-verify', [ROOT/'bin/nano_vm', '--verify-only', module])
                vm = self.run_retained(label+'-vm', [ROOT/'bin/nano_vm', module], expected)
                if expected is None:
                    self.assertNotEqual(vm.returncode, 0)
                    self.assertIn(missing_error, vm.stderr)
                self.run_retained(label+'-translate', [ROOT/'bin/nvm2c', module, '-o', native_source])
                for optimization in ('-O0', '-O2'):
                    name = label+optimization[1:]
                    binary = self.work/name
                    self.run_retained(name+'-build', [*self.cc, '-std=c11', optimization,
                        '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                        '-fno-sanitize-recover=all', native_source, ROOT/'bin/nano_aot_runtime.o',
                        '-lm', *native_link_flags(), '-o', binary], build=True)
                    native = self.run_retained(name+'-native', [binary], expected)
                    if expected is None:
                        self.assertNotEqual(native.returncode, 0)
                        self.assertIn('native invariant', native.stderr)
                        self.assertNotIn('ERROR: AddressSanitizer', native.stderr)
                        self.assertNotIn('runtime error:', native.stderr)

    def test_actual_conditional_reads_preserve_shared_record_fields(self):
        for case, values, checked, expected in (
            ('present', '[7]', 1, 7),
            ('empty', '[]', 1, 0),
            ('unchecked', '[7]', 0, 0),
        ):
            self.paired(case, source_text(values, checked), expected)

    def test_float_bool_string_conditional_record_fields(self):
        for kind, value, zero in (('float', '7.5', '0.0'),
                                  ('bool', 'true', 'false'),
                                  ('string', '"yes"', '""')):
            for case in ('present', 'empty', 'unchecked', 'raw-missing'):
                self.paired(kind + '-' + case, scalar_source_text(kind, value, zero, case),
                            None if case == 'raw-missing' else 0,
                            'requires two floats' if kind == 'float' else 'Assertion failed')

    def test_missing_raw_reads_do_not_become_integer_zero(self):
        for index in (-1, 1):
            self.paired('raw-missing-'+str(index), source_text('[7]', 1, index, guarded=False))
        self.paired('raw-unchecked', source_text('[7]', 0, 1, guarded=False), 1)


if __name__ == '__main__':
    unittest.main()
