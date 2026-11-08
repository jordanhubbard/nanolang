"""I execute numeric selection without replacing my original corpus."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest

from tests.native_process import native_link_flags, run_captured

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = os.environ.get('NANO_NUMERIC_PROVIDERS', 'nano_virt,nanoc_stage1,nanoc_stage2').split(',')


class NumericBuiltinSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        allowed = {'nano_virt', 'nanoc_stage1', 'nanoc_stage2'}
        if not PROVIDERS or len(set(PROVIDERS)) != len(PROVIDERS) or set(PROVIDERS)-allowed:
            raise AssertionError('I require explicit unique NanoISA source providers.')
        missing = [name for name in PROVIDERS if not (ROOT/'bin'/name).is_file()]
        if missing:
            raise AssertionError('I require actual source providers: '+', '.join(missing))

    def command(self, work, label, argv, *, build=False, accepted=True):
        result = run_captured(argv, cwd=ROOT, text=True, timeout=60, build=build)
        (work/(label+'.stdout')).write_text(result.stdout)
        (work/(label+'.stderr')).write_text(result.stderr)
        with (work/'commands.jsonl').open('a') as log:
            log.write(json.dumps({'label': label, 'argv': list(map(str, argv)),
                                  'returncode': result.returncode})+'\n')
        if accepted:
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        return result

    def exercise(self, source, expected='', *, refused=False, source_path=None):
        output = ROOT/'.test_output/numeric-builtin-selection'
        output.mkdir(parents=True, exist_ok=True)
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                work = Path(tempfile.mkdtemp(prefix=self._testMethodName+'-'+provider+'-', dir=output))
                path = work/'main.nano'
                path.write_text(source)
                actual_source = path if source_path is None else Path(source_path)
                self.assertEqual(actual_source.read_text(), source)
                source_before = hashlib.sha256(actual_source.read_bytes()).hexdigest()
                compiler = ROOT/'bin'/provider
                before = hashlib.sha256(compiler.read_bytes()).hexdigest()
                module, native, generated = work/'module.nvm', work/'native', work/'native.c'
                argv = [compiler, actual_source, '--emit-nvm', '-o', module]
                if refused:
                    original = b'I preserve my prior numeric program.\n'
                    module.write_bytes(original)
                    result = self.command(work, 'refusal', argv, build=True, accepted=False)
                    self.assertGreater(result.returncode, 0, result.stdout+result.stderr)
                    self.assertRegex(result.stdout+result.stderr, r'(?i)(type|operand|argument|arity)')
                    self.assertEqual(module.read_bytes(), original)
                else:
                    self.command(work, 'source-build', argv, build=True)
                    self.command(work, 'verify', [ROOT/'bin/nano_vm', '--verify-only', module])
                    result = self.command(work, 'vm', [ROOT/'bin/nano_vm', module])
                    self.assertEqual(result.stdout, expected)
                    self.command(work, 'aot', [ROOT/'bin/nvm2c', module, '-o', generated], build=True)
                    self.command(work, 'native-build', [
                        *shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', 'cc')),
                        '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror',
                        '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                        generated, ROOT/'bin/nano_aot_runtime.o', '-lm', *native_link_flags(),
                        *(['-Wl,--export-dynamic', '-ldl'] if sys.platform.startswith('linux') else []),
                        '-o', native], build=True)
                    result = self.command(work, 'native', [native])
                    self.assertEqual(result.stdout, expected)
                after = hashlib.sha256(compiler.read_bytes()).hexdigest()
                (work/'inputs.json').write_text(json.dumps({'provider': str(compiler),
                    'provider_before': before, 'provider_after': after,
                    'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'actual_source': str(actual_source), 'source_before': source_before,
                    'source_after': hashlib.sha256(actual_source.read_bytes()).hexdigest()}, indent=2)+'\n')
                self.assertEqual(before, after)
                self.assertEqual(source_before, hashlib.sha256(actual_source.read_bytes()).hexdigest())

    def test_original_complete_builtin_fixture(self):
        path = ROOT/'tests/test_builtins_phase1.nano'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        self.exercise(path.read_text(), 'Testing built-in functions...\nhello world\n'
            'String test result: 11\nConversion test result: 42\nMath test result: 30\n', source_path=path)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_numeric_values_and_original_float_bits(self):
        self.exercise('''fn main()->int{
 assert (== (min 2 7) 2) assert (== (min 7 2) 2)
 assert (== (max 2 7) 7) assert (== (max 7 2) 7)
 assert (== (min -7 -2) -7) assert (== (max -7 -2) -2)
 assert (== (min 7 7) 7) assert (== (max -7 -7) -7)
 let low:int = (- -9223372036854775807 1)
 assert (== (min low 9223372036854775807) low)
 assert (== (max low 9223372036854775807) 9223372036854775807)
 assert (== (min 2.5 7.8) 2.5) assert (== (min 7.8 2.5) 2.5)
 assert (== (max 2.5 7.8) 7.8) assert (== (max 7.8 2.5) 7.8)
 assert (== (min -7.0 -2.0) -7.0) assert (== (max -7.0 -2.0) -2.0)
 let magnitude:float = (abs -2.5) assert (== magnitude 2.5)
 let negative:float = (float_from_bits low)
 assert (== (float_to_bits (min 0.0 negative)) low)
 assert (== (float_to_bits (max negative 0.0)) 0)
 let nan:float = (float_from_bits 9221120237041090626)
 assert (== (float_to_bits (min 2.0 nan)) 9221120237041090626)
 assert (== (float_to_bits (max nan 2.0)) (float_to_bits 2.0))
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_once_order_and_lexical_function_identity(self):
        self.exercise('''let mut trace:int=0
fn next(value:int)->int{set trace (+ (* trace 10) value) return value}
shadow next{set trace 0 assert (== (next 2) 2) assert (== trace 2) set trace 0}
fn hundred(value:int)->int{return (+ value 100)}
shadow hundred{assert (== (hundred 2) 102)}
fn main()->int{
 set trace 0 assert (== (min (next 2) (next 7)) 2) assert (== trace 27)
 set trace 0 assert (== (max (next 2) (next 7)) 7) assert (== trace 27)
 let min:fn(int)->int=hundred let max:fn(int)->int=hundred
 let abs:fn(int)->int=hundred
 assert (== (min 2) 102) assert (== (max 3) 103) assert (== (abs 4) 104)
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_raw_pick_retains_scalar_record_and_callable_slots(self):
        cases = {
            'scalars': ('.string "hello"\n', 'PUSH_I64 7\nPUSH_F64 2.5\nPUSH_STR 0\nPUSH_BOOL 1\n'
                'PICK 3\nPUSH_I64 7\nI64_EQ\nASSERT\n'
                'PICK 2\nPUSH_F64 2.5\nF64_EQ\nASSERT\n'
                'PICK 1\nPUSH_STR 0\nSTR_EQ\nASSERT\nPICK 0\nASSERT\nPOP\nPOP\nPOP\nPOP\n', ''),
            'record': ('', 'PUSH_I64 7\nAGG_PACK 0 0 0 1\nPUSH_I64 99\nPICK 1\n'
                'AGG_GET 0\nPUSH_I64 7\nI64_EQ\nASSERT\nPOP\n'
                'AGG_GET 0\nPUSH_I64 7\nI64_EQ\nASSERT\n', ''),
            'callable': ('', 'FUNCREF identity\nPUSH_I64 99\nPICK 1\nPUSH_I64 7\nSWAP\n'
                'CALL_INDIRECT 1 1\nPUSH_I64 7\nI64_EQ\nASSERT\nPOP\nPOP\n',
                '.function identity 1 1 0 int 1\nLOAD_LOCAL 0\nRET\n.end\n.parameters identity int\n'),
        }
        output = ROOT/'.test_output/numeric-builtin-selection'
        output.mkdir(parents=True, exist_ok=True)
        for label, (constants, body, helpers) in cases.items():
            with self.subTest(case=label):
                work = Path(tempfile.mkdtemp(prefix='raw-pick-'+label+'-', dir=output))
                assembly, module, generated, native = [work/name for name in
                    ['main.nasm', 'module.nvm', 'native.c', 'native']]
                assembly.write_text('.entry main\n'+constants+'.function main 0 0 0 int 1\n'+body+
                    'PUSH_I64 0\nRET\n.end\n'+helpers)
                self.command(work, 'assemble', [ROOT/'bin/nanoisa', 'asm', assembly, '-o', module])
                self.command(work, 'verify', [ROOT/'bin/nano_vm', '--verify-only', module])
                self.command(work, 'vm', [ROOT/'bin/nano_vm', module])
                self.command(work, 'aot', [ROOT/'bin/nvm2c', module, '-o', generated], build=True)
                self.command(work, 'native-build', [
                    *shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', 'cc')),
                    '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror',
                    '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                    generated, ROOT/'bin/nano_aot_runtime.o', '-lm', *native_link_flags(),
                    *(['-Wl,--export-dynamic', '-ldl'] if sys.platform.startswith('linux') else []),
                    '-o', native], build=True)
                self.command(work, 'native', [native])

    def test_pick_invalid_depth_preserves_prior_output(self):
        output = ROOT/'.test_output/numeric-builtin-selection'
        output.mkdir(parents=True, exist_ok=True)
        for body in ['PICK 0', 'PUSH_I64 7\nPICK 1', 'PUSH_I64 7\nPICK 65535']:
            with self.subTest(body=body):
                work = Path(tempfile.mkdtemp(prefix='invalid-pick-', dir=output))
                assembly, module = work/'main.nasm', work/'prior.nvm'
                assembly.write_text('.entry main\n.function main 0 0 0 int 1\n'+body+'\nRET\n.end\n')
                module.write_bytes(b'I retain prior output.\n')
                result = self.command(work, 'refusal', [ROOT/'bin/nanoisa', 'asm', assembly,
                    '-o', module], accepted=False)
                self.assertGreater(result.returncode, 0)
                self.assertIn('underflow', result.stderr)
                self.assertEqual(module.read_bytes(), b'I retain prior output.\n')

    def test_numeric_refusals_preserve_output(self):
        for expression in ['(min)', '(max 1)', '(min 1 2 3)', '(min true false)',
                           '(max "one" "two")', '(min 2.0 7)', '(abs true)']:
            with self.subTest(expression=expression):
                self.exercise('fn main()->int{let result = '+expression+
                    ' return 0} shadow main{assert true}', refused=True)
