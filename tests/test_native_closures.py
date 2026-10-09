"""I check native capture identity, mutations and managed environment lifetimes."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile
import unittest
from tests import test_native_callables as callables
ROOT = callables.ROOT


CHURN = ('.function churn 0 1 0 void 0\nPUSH_I64 0\nSTORE_LOCAL 0\nloop:\n'
         'LOAD_LOCAL 0\nPUSH_I64 8000\nLT\nJMP_FALSE done\n'
         'PUSH_STR a\nPUSH_STR b\nSTR_CONCAT\nPOP\n'
         'LOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 0\nJMP loop\ndone:\nRET\n.end\n')


class NativeClosures(unittest.TestCase):
    checked = callables.NativeCallables.checked
    emit = callables.NativeCallables.emit
    sanitized = callables.NativeCallables.sanitized

    def run_module(self, text, stdout=None, collections=False):
        with tempfile.TemporaryDirectory(prefix='nano-native-closures-') as tmp:
            work = Path(tmp)
            source = self.emit(work, text, stdout)
            if collections:
                generated = source.read_text()
                marker = 'static void nmap_collect(void) {\n'
                self.assertIn(marker, generated)
                generated = generated.replace(marker, 'static size_t collections;\n' + marker + '    ++collections;\n')
                source.write_text('#define main fixture_main\n' + generated +
                    '\n#undef main\nint main(void) { int result = fixture_main();\n'
                    '    if (collections < 2) { abort(); }\n'
                    '    return result;\n}\n')
            result = self.sanitized(source, work / 'program')
            if stdout is not None:
                self.assertEqual(result.stdout, stdout)

    def test_retained_canonical_returned_chain(self):
        with tempfile.TemporaryDirectory(prefix='nano-native-capture-source-') as tmp:
            work = Path(tmp)
            module = ROOT / 'docs/evidence/captured-closure-baseline-20261008/capture.nvm'
            self.assertEqual(self.checked([ROOT / 'bin/nano_vm', module]).stdout, '42\n')
            source = work / 'capture.c'
            self.checked([ROOT / 'bin/nvm2c', module, '-o', source])
            self.assertEqual(self.sanitized(source, work / 'program').stdout, '42\n')

    def test_instances_aliases_mutation_and_record_global_roots(self):
        text = ('.string a \"a\"\n.string b \"b\"\n.entry main\n'
                '.function make 1 1 0 function 1\nLOAD_LOCAL 0\nCLOSURE_NEW bump 1\nRET\n.end\n'
                '.function bump 0 0 1 int 1\nLOAD_UPVALUE 0 0\nPUSH_I64 1\nI64_ADD\n'
                'STORE_UPVALUE 0 0\nLOAD_UPVALUE 0 0\nRET\n.end\n'
                '.function main 0 3 0 int 1\nPUSH_I64 40\nCALL make\nSTORE_LOCAL 0\n'
                'LOAD_LOCAL 0\nSTORE_LOCAL 1\nPUSH_I64 40\nCALL make\nSTORE_LOCAL 2\n'
                'LOAD_LOCAL 0\nLOAD_LOCAL 1\nEQ\nASSERT\n'
                'LOAD_LOCAL 0\nLOAD_LOCAL 2\nNE\nASSERT\n'
                'LOAD_LOCAL 0\nCALL_INDIRECT 0 1\nPUSH_I64 41\nEQ\nASSERT\n'
                'LOAD_LOCAL 1\nCALL_INDIRECT 0 1\nPUSH_I64 42\nEQ\nASSERT\n'
                'LOAD_LOCAL 2\nCALL_INDIRECT 0 1\nPUSH_I64 41\nEQ\nASSERT\n'
                'LOAD_LOCAL 0\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\nSTORE_GLOBAL 0\n'
                'CALL churn\nLOAD_GLOBAL 0\nPUSH_I64 0\nARR_GET\nAGG_GET 0\n'
                'CALL_INDIRECT 0 1\nPUSH_I64 43\nEQ\nASSERT\n'
                'CALL churn\nLOAD_LOCAL 1\nCALL_INDIRECT 0 1\nPUSH_I64 44\nEQ\nASSERT\n'
                'LOAD_LOCAL 0\nPRINTLN\nPUSH_I64 0\nRET\n.end\n' + CHURN)
        self.run_module(text, 'closure(1)\n', collections=True)

    def test_managed_captures_and_active_environment_roots(self):
        text = ('.string a \"a\"\n.string b \"b\"\n.entry main\n'
                '.function make 0 0 0 function 1\nPUSH_STR a\nPUSH_STR b\nSTR_CONCAT\n'
                'PUSH_I64 41\nARR_LITERAL 1 1\nAGG_PACK 0 0 0 2\n'
                'CLOSURE_NEW read 1\nRET\n.end\n'
                '.function read 0 0 1 int 1\nCALL churn\nLOAD_UPVALUE 0 0\nAGG_GET 0\n'
                'PUSH_STR a\nPUSH_STR b\nSTR_CONCAT\nSTR_EQ\nASSERT\n'
                'LOAD_UPVALUE 0 0\nAGG_GET 1\nPUSH_I64 0\nARR_GET\nPUSH_I64 1\nI64_ADD\nRET\n.end\n'
                '.function main 0 0 0 int 1\nCALL make\nCALL_INDIRECT 0 1\nPUSH_I64 42\nEQ\nASSERT\n'
                'CALL make\nCALL_INDIRECT 0 1\nPUSH_I64 42\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n' + CHURN)
        self.run_module(text, collections=True)

    def test_nested_callable_and_scalar_capture_tags(self):
        text = ('.string a "a"\n.string b "b"\n.entry main\n'
                '.function make 0 0 0 function 1\nPUSH_STR a\nPUSH_STR b\nSTR_CONCAT\n'
                'CLOSURE_NEW inner 1\nPUSH_F64 3.5\nPUSH_BOOL 1\nCLOSURE_NEW outer 3\nRET\n.end\n'
                '.function inner 0 0 1 string 1\nCALL churn\nLOAD_UPVALUE 0 0\nRET\n.end\n'
                '.function outer 0 0 3 string 1\nCALL churn\n'
                'LOAD_UPVALUE 0 1\nPUSH_F64 3.5\nF64_EQ\nASSERT\n'
                'LOAD_UPVALUE 0 2\nASSERT\nLOAD_UPVALUE 0 0\nCALL_INDIRECT 0 1\nRET\n.end\n'
                '.function main 0 0 0 int 1\nCALL make\nCALL_INDIRECT 0 1\nPRINTLN\n'
                'PUSH_I64 0\nRET\n.end\n' + CHURN)
        self.run_module(text, 'ab\n', collections=True)

    def test_zero_capture_instances_and_exact_tags(self):
        text = ('.entry main\n.function leaf 0 0 0 int 1\nPUSH_I64 42\nRET\n.end\n'
                '.function main 0 1 0 int 1\nCLOSURE_NEW leaf 0\nSTORE_LOCAL 0\n'
                'LOAD_LOCAL 0\nTYPE_CHECK 15\nASSERT\n'
                'LOAD_LOCAL 0\nCLOSURE_NEW leaf 0\nNE\nASSERT\n'
                'LOAD_LOCAL 0\nFUNCREF leaf\nNE\nASSERT\n'
                'LOAD_LOCAL 0\nCALL_INDIRECT 0 1\nPUSH_I64 42\nEQ\nASSERT\n'
                'LOAD_LOCAL 0\nPRINTLN\nFUNCREF leaf\nPRINTLN\nPUSH_I64 0\nRET\n.end\n')
        self.run_module(text, 'closure(0)\nfn(0)\n')

    def test_captured_call_checks_tag_target_and_environment(self):
        text = ('.entry main\n.function make 0 0 0 function 1\nPUSH_I64 42\nCLOSURE_NEW read 1\nRET\n.end\n'
                '.function read 0 0 1 int 1\nLOAD_UPVALUE 0 0\nRET\n.end\n'
                '.function apply 1 1 0 int 1\nLOAD_LOCAL 0\nCALL_INDIRECT 0 1\nRET\n.end\n'
                '.function main 0 0 0 int 1\nCALL make\nCALL apply\nPUSH_I64 42\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n')
        with tempfile.TemporaryDirectory(prefix='nano-closure-guards-') as tmp:
            work = Path(tmp)
            source = self.emit(work, text)
            source.write_text('#define main fixture_main\n' + source.read_text() +
                '\n#undef main\nint main(int argc, char **argv) {\n'
                '    if (argc != 2) return 2;\n'
                '    nmap_value value = nl_make(); int mode = atoi(argv[1]);\n'
                '    if (mode == 1) value.kind = 11;\n'
                '    if (mode == 2) value.text = NULL;\n'
                '    if (mode == 3) value.integer = 99;\n'
                '    if (mode == 4) ((nrec_t *)value.text)->n = 0;\n'
                '    if (mode == 5) ((nrec_t *)value.text)->callable_target = 99;\n'
                '    if (nl_apply(value) != 42) return 3;\n'
                '    return fixture_main();\n}\n')
            binary = work / 'program'
            self.checked([*shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', os.environ.get('CC', 'cc'))), '-std=c11', '-O0', '-g', '-Wall', '-Wextra', '-Werror',
                          '-fsanitize=address,undefined', '-fno-sanitize-recover=all', source, '-o', binary])
            env = {**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1'}
            self.checked([binary, 0], env=env)
            for mode in range(1, 6):
                with self.subTest(mode=mode):
                    result = subprocess.run([str(binary), str(mode)], capture_output=True, text=True, timeout=10, env=env)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('native invariant', result.stderr)
                    self.assertNotIn('ERROR: AddressSanitizer', result.stderr)
                    self.assertNotIn('runtime error:', result.stderr)

    def test_implicit_roots_cannot_supply_captures(self):
        for name in ('main', '__init__'):
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix='nano-closure-entry-') as tmp:
                work = Path(tmp)
                text = '.entry main\n'
                if name == '__init__':
                    text += '.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n'
                text += f'.function {name} 0 0 1 int 1\nPUSH_I64 0\nRET\n.end\n'
                assembly, module, source = work / 'input.nasm', work / 'input.nvm', work / 'prior.c'
                assembly.write_text(text)
                self.checked([ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module])
                source.write_text('prior output\n')
                result = subprocess.run([str(ROOT / 'bin/nvm2c'), str(module), '-o', str(source)],
                                        capture_output=True, text=True, timeout=15)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('capture-free native entry', result.stderr)
                self.assertEqual(source.read_text(), 'prior output\n')


if __name__ == '__main__':
    unittest.main()
