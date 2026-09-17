"""I preserve typed float bytecode and checked scalar transport in native code."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
OPS = ('ADD', 'SUB', 'MUL', 'DIV', 'NEG', 'EQ', 'NE', 'LT', 'LE', 'GT', 'GE')


class NativeFloats(unittest.TestCase):
    def run_command(self, args):
        leak_detection = '0' if sys.platform == 'darwin' else '1'
        return subprocess.run([str(x) for x in args], text=True, capture_output=True,
                              timeout=90,
                              env={**os.environ,
                                   'ASAN_OPTIONS': f'detect_leaks={leak_detection}'})

    def checked(self, args):
        result = self.run_command(args)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def assemble(self, work, body, helpers=''):
        source = work / 'input.nasm'
        source.write_text('.entry main\n.string zero "0"\n.string negative "-0"\n'
                          + helpers + '.function main 0 2 0 int 1\n' + body +
                          'PUSH_I64 0\nRET\n.end\n')
        module = work / 'input.nvm'
        self.checked([ROOT / 'bin/nanoisa', 'asm', source, '-o', module])
        return module

    def native(self, work, module, sanitize=False):
        source, binary = work / 'input.c', work / 'program'
        self.checked([ROOT / 'bin/nvm2c', module, '-o', source])
        flags = ['-fsanitize=address,undefined', '-fno-sanitize-recover=all'] if sanitize else []
        self.checked(['cc', '-std=c11', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                      *flags, source, '-o', binary])
        return binary

    def test_operations_tags_signed_zero_and_transport(self):
        body = ''
        for op, a, b, result in [('ADD', 1.5, 2.5, 4), ('SUB', 5.5, 2, 3.5),
                                 ('MUL', 1.5, 2, 3), ('DIV', 5, 2, 2.5)]:
            body += f'PUSH_F64 {a}\nPUSH_F64 {b}\nF64_{op}\nDUP\nTYPE_CHECK 3\nASSERT\nPUSH_F64 {result}\nF64_EQ\nASSERT\n'
        for op, a, b in [('EQ', 2, 2), ('NE', 2, 3), ('LT', 2, 3),
                          ('LE', 2, 2), ('GT', 3, 2), ('GE', 2, 2)]:
            body += f'PUSH_F64 {a}\nPUSH_F64 {b}\nF64_{op}\nDUP\nTYPE_CHECK 4\nASSERT\nASSERT\n'
        # I retain all bits through calls, locals, tagged globals and returns.
        helpers = ('.function identity 1 1 0 float 1\nLOAD_LOCAL 0\nRET\n.end\n'
                   '.function tail 1 1 0 float 1\nLOAD_LOCAL 0\nTAIL_CALL identity\n.end\n'
                   '.function global 0 0 0 float 1\nLOAD_GLOBAL 0\nRET\n.end\n'
                   '.function loop 2 2 0 float 1\nLOAD_LOCAL 1\nPUSH_I64 0\nEQ\nJMP_FALSE again\n'
                   'LOAD_LOCAL 0\nRET\nagain:\nLOAD_LOCAL 0\nF64_NEG\nLOAD_LOCAL 1\nPUSH_I64 1\nI64_SUB\nTAIL_CALL loop\n.end\n')
        body += ('PUSH_F64 0\nF64_NEG\nCALL tail\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nSTORE_GLOBAL 0\n'
                 'LOAD_GLOBAL 0\nTYPE_CHECK 3\nASSERT\nCALL global\nCAST_STRING\nPUSH_STR negative\nEQ\nASSERT\n'
                 'LOAD_GLOBAL 0\nCAST_STRING\nPUSH_STR negative\nEQ\nASSERT\n'
                 'PUSH_F64 0\nPUSH_I64 3\nCALL loop\nCAST_STRING\nPUSH_STR negative\nEQ\nASSERT\n'
                 'PUSH_F64 2.5\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPUSH_F64 2.5\nEQ\nASSERT\n'
                 'LOAD_GLOBAL 0\nPUSH_I64 3\nLT\nASSERT\nLOAD_GLOBAL 0\nASSERT\n'
                 'PUSH_F64 -0\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nJMP_FALSE zero_branch\nPUSH_BOOL 0\nASSERT\nzero_branch:\n'
                 'PUSH_F64 2.5\nPRINTLN\nLOAD_GLOBAL 0\nPRINTLN\n')
        for divisor in ('0', '-0'):
            body += f'PUSH_F64 -5\nPUSH_F64 {divisor}\nF64_DIV\nCAST_STRING\nPUSH_STR zero\nEQ\nASSERT\n'
        # NaNs remain unordered under typed comparisons; infinities are preserved.
        body += 'PUSH_F64 inf\nPUSH_F64 inf\nF64_EQ\nASSERT\n'
        body += 'PUSH_F64 nan\nPUSH_F64 nan\nF64_NE\nASSERT\n'
        body += 'PUSH_F64 nan\nPUSH_F64 1\nF64_LE\nBOOL_NOT\nASSERT\n'
        # Generic ordering uses val_compare: unordered NaN compares as zero.
        # Typed comparisons deliberately preserve IEEE unordered behavior instead.
        for tagged in (False, True):
            operand = ('PUSH_F64 nan\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\n'
                       if tagged else 'PUSH_F64 nan\n')
            for op, truth in [('EQ', False), ('NE', True), ('LT', False),
                              ('LE', True), ('GT', False), ('GE', True)]:
                body += operand + f'PUSH_F64 1\n{op}\n'
                if not truth:
                    body += 'BOOL_NOT\n'
                body += 'ASSERT\n'
        with tempfile.TemporaryDirectory(prefix='nano-native-float-') as tmp:
            work = Path(tmp)
            module = self.assemble(work, body, helpers)
            vm = self.checked([ROOT / 'bin/nano_vm', module])
            native = self.checked([self.native(work, module, sanitize=True)])
            self.assertEqual(native.stdout, vm.stdout)

    def test_typed_operations_reject_dynamic_nonfloat_tags(self):
        with tempfile.TemporaryDirectory(prefix='nano-native-float-tags-') as tmp:
            work = Path(tmp)
            for op in OPS:
                for wrong in ('PUSH_I64 1', 'PUSH_BOOL 1', 'PUSH_STR zero'):
                    with self.subTest(op=op, wrong=wrong):
                        # STORE/LOAD keeps the tag unknown to static verification.
                        body = wrong + '\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\n'
                        if op != 'NEG':
                            body += 'PUSH_F64 2\n'
                        body += f'F64_{op}\nPOP\n'
                        module = self.assemble(work, body)
                        vm = self.run_command([ROOT / 'bin/nano_vm', module])
                        self.assertNotEqual(vm.returncode, 0)
                        self.assertIn('float', vm.stderr)
                        binary = self.native(work, module)
                        self.assertNotEqual(self.run_command([binary]).returncode, 0)

    def test_direct_wrong_tag_refusal_preserves_output(self):
        with tempfile.TemporaryDirectory(prefix='nano-native-float-static-') as tmp:
            work = Path(tmp)
            source = work / 'invalid.nasm'
            source.write_text('.entry main\n.function main 0 0 0 int 1\n'
                              'PUSH_I64 1\nF64_NEG\nRET\n.end\n')
            output = work / 'existing.nvm'
            output.write_text('retained')
            result = self.run_command([ROOT / 'bin/nanoisa', 'asm', source, '-o', output])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('expects float', result.stderr)
            self.assertEqual(output.read_text(), 'retained')

    def test_tagged_float_integer_conversion_is_explicitly_refused(self):
        with tempfile.TemporaryDirectory(prefix='nano-native-float-conversion-') as tmp:
            work = Path(tmp)
            module = self.assemble(work, 'PUSH_F64 2.5\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nCAST_INT\nPUSH_I64 2\nEQ\nASSERT\n')
            self.checked([ROOT / 'bin/nano_vm', module])
            # I retain the native gap instead of accepting the old silent zero.
            binary = self.native(work, module)
            self.assertNotEqual(self.run_command([binary]).returncode, 0)

    def test_cast_float_matches_vm_scalar_conversions(self):
        strings = {
            'decimal': '  -12.5 trailing', 'invalid': 'not a number',
            'minus_zero': '-0', 'infinity': 'inf', 'not_number': 'nan',
        }
        helpers = ''.join(f'.string {name} "{text}"\n' for name, text in strings.items())
        cases = [(f'PUSH_I64 {n}', str(float(n))) for n in
                 (0, 1, -1, 9007199254740991, 9007199254740993,
                  -9223372036854775808, 9223372036854775807)]
        cases += [('PUSH_BOOL 0', '0'), ('PUSH_BOOL 1', '1'),
                  ('PUSH_F64 1.25', '1.25'), ('PUSH_F64 inf', 'inf'),
                  ('PUSH_STR decimal', '-12.5'), ('PUSH_STR invalid', '0')]
        body = ''
        for tagged in (False, True):
            for operand, expected in cases:
                body += operand + '\n'
                if tagged:
                    body += 'STORE_GLOBAL 0\nLOAD_GLOBAL 0\n'
                body += 'CAST_FLOAT\nDUP\nTYPE_CHECK 3\nASSERT\n'
                body += f'PUSH_F64 {expected}\nF64_EQ\nASSERT\n'
            for operand in ('PUSH_F64 -0', 'PUSH_STR minus_zero'):
                body += operand + '\n'
                if tagged:
                    body += 'STORE_GLOBAL 0\nLOAD_GLOBAL 0\n'
                body += 'CAST_FLOAT\nCAST_STRING\nPUSH_STR negative\nEQ\nASSERT\n'
            for operand in ('PUSH_F64 nan', 'PUSH_STR not_number'):
                body += operand + '\n'
                if tagged:
                    body += 'STORE_GLOBAL 0\nLOAD_GLOBAL 0\n'
                body += 'CAST_FLOAT\nDUP\nF64_NE\nASSERT\n'
        # VM defaults other tags to positive zero, including missing map values.
        body += 'HM_NEW 5 1\nCAST_FLOAT\nPUSH_F64 0\nF64_EQ\nASSERT\n'
        body += 'HM_NEW 5 1\nPUSH_STR zero\nHM_GET\nCAST_FLOAT\nPUSH_F64 0\nF64_EQ\nASSERT\n'
        body += 'PUSH_I64 42\nCAST_FLOAT\nPRINTLN\n'
        with tempfile.TemporaryDirectory(prefix='nano-native-cast-float-') as tmp:
            work = Path(tmp)
            module = self.assemble(work, body, helpers)
            vm = self.checked([ROOT / 'bin/nano_vm', module])
            native = self.checked([self.native(work, module, sanitize=True)])
            self.assertEqual(native.stdout, vm.stdout)
            self.assertEqual(vm.stdout, '42.0\n')

    def test_cast_float_through_polymorphic_calls_and_local_loop(self):
        helpers = ('.string fractional "2.5"\n'
                   '.function convert 1 1 0 float 1\nLOAD_LOCAL 0\nCAST_FLOAT\nRET\n.end\n'
                   '.function tailconvert 1 1 0 float 1\nLOAD_LOCAL 0\nTAIL_CALL convert\n.end\n'
                   '.function sum 1 3 0 float 1\nPUSH_I64 0\nSTORE_LOCAL 1\n'
                   'PUSH_F64 0\nSTORE_LOCAL 2\nloop:\nLOAD_LOCAL 1\nLOAD_LOCAL 0\n'
                   'I64_LT_S\nJMP_FALSE done\nLOAD_LOCAL 2\nLOAD_LOCAL 1\nCAST_FLOAT\n'
                   'F64_ADD\nSTORE_LOCAL 2\nLOAD_LOCAL 1\nPUSH_I64 1\nI64_ADD\n'
                   'STORE_LOCAL 1\nJMP loop\ndone:\nLOAD_LOCAL 2\nRET\n.end\n')
        body = ''
        for operand, expected in [('PUSH_I64 42', '42'), ('PUSH_BOOL 1', '1'),
                                  ('PUSH_STR fractional', '2.5'), ('PUSH_F64 1.25', '1.25')]:
            # I use the existing tagged transport contract for varying argument kinds.
            body += f'{operand}\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nCALL tailconvert\nPUSH_F64 {expected}\nF64_EQ\nASSERT\n'
        body += 'PUSH_I64 5\nCALL sum\nPRINTLN\n'
        with tempfile.TemporaryDirectory(prefix='nano-native-cast-float-calls-') as tmp:
            work = Path(tmp)
            module = self.assemble(work, body, helpers)
            vm = self.checked([ROOT / 'bin/nano_vm', module])
            native = self.checked([self.native(work, module, sanitize=True)])
            self.assertEqual(native.stdout, vm.stdout)
            self.assertEqual(vm.stdout, '10.0\n')

    def test_optional_float_helpers_remain_warning_clean(self):
        fixtures = {
            'tagged_runtime_without_float_transport': 'HM_NEW 5 1\nPOP\n',
            'integer_only_print': 'PUSH_I64 42\nPRINTLN\n',
        }
        with tempfile.TemporaryDirectory(prefix='nano-native-float-helpers-') as tmp:
            work = Path(tmp)
            for name, body in fixtures.items():
                with self.subTest(name=name):
                    module = self.assemble(work, body)
                    vm = self.checked([ROOT / 'bin/nano_vm', module])
                    native = self.checked([self.native(work, module)])
                    self.assertEqual(native.stdout, vm.stdout)
