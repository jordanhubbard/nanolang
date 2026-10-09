"""I retain record-global owners and reject invalid record projections."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from tests import test_native_closures as closures


class NativeRecordGlobals(unittest.TestCase):
    checked = closures.NativeClosures.checked
    emit = closures.NativeClosures.emit
    sanitized = closures.NativeClosures.sanitized
    run_module = closures.NativeClosures.run_module

    def test_alias_retention_overwrite_and_collection(self):
        self.run_module('.string before "before"\n.string after "after"\n.string suffix "-owned"\n.entry main\n.function main 0 2 0 int 1\nCALL make_before\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nSTORE_LOCAL 0\nPUSH_I64 0\nSTORE_LOCAL 1\nloop:\nLOAD_LOCAL 1\nPUSH_I64 4096\nI64_LT_S\nJMP_FALSE done\nCALL make_after\nSTORE_GLOBAL 0\nLOAD_LOCAL 1\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 1\nJMP loop\ndone:\nLOAD_LOCAL 0\nAGG_GET 0\nPUSH_STR before\nPUSH_STR suffix\nSTR_CONCAT\nSTR_EQ\nASSERT\nLOAD_LOCAL 0\nAGG_GET 1\nPUSH_I64 0\nARR_GET\nPUSH_STR before\nPUSH_STR suffix\nSTR_CONCAT\nSTR_EQ\nASSERT\nLOAD_GLOBAL 0\nAGG_GET 0\nPUSH_STR after\nPUSH_STR suffix\nSTR_CONCAT\nSTR_EQ\nASSERT\nPUSH_VOID\nSTORE_GLOBAL 0\nLOAD_LOCAL 0\nAGG_GET 0\nPRINTLN\nPUSH_I64 0\nRET\n.end\n.function make_before 0 0 0 struct 1\nPUSH_STR before\nPUSH_STR suffix\nSTR_CONCAT\nDUP\nARR_LITERAL 5 1\nAGG_PACK 0 0 0 2\nRET\n.end\n.function make_after 0 0 0 struct 1\nPUSH_STR after\nPUSH_STR suffix\nSTR_CONCAT\nDUP\nARR_LITERAL 5 1\nAGG_PACK 0 0 0 2\nRET\n.end\n', 'before-owned\n', collections=True)

    def test_scalar_overwrite_refuses_record_projection(self):
        text = ('.entry main\n.function main 0 0 0 int 1\n'
                'PUSH_I64 7\nAGG_PACK 0 0 0 1\nSTORE_GLOBAL 0\n'
                'PUSH_I64 99\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nAGG_GET 0\nRET\n.end\n')
        with tempfile.TemporaryDirectory(prefix='nano-record-global-refusal-') as tmp:
            work = Path(tmp)
            assembly, module, source, binary = [work/name for name in ('input.nasm','input.nvm','input.c','program')]
            assembly.write_text(text)
            self.checked([closures.ROOT/'bin/nanoisa','asm',assembly,'-o',module])
            self.checked([closures.ROOT/'bin/nano_vm','--verify-only',module])
            self.checked([closures.ROOT/'bin/nvm2c',module,'-o',source])
            cc = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC') or os.environ.get('CC') or 'cc')
            self.checked([*cc,'-std=c11','-Wall','-Wextra','-Werror','-fsanitize=address,undefined',
                          '-fno-sanitize-recover=all',source,'-lm','-o',binary])
            for label, command in [('vm',[str(closures.ROOT/'bin/nano_vm'),str(module)]),('native',[str(binary)])]:
                with self.subTest(route=label):
                    result = subprocess.run(command,capture_output=True,text=True,timeout=90,
                                            env={**os.environ,'ASAN_OPTIONS':'detect_leaks=1'})
                    self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
                    if label == 'native':
                        self.assertIn('native invariant',result.stderr)
                        self.assertNotIn('ERROR: AddressSanitizer',result.stderr)

    def test_record_arguments_from_direct_global_and_tail_calls(self):
        for global_first in (False, True):
            with self.subTest(global_first=global_first):
                direct = 'CALL make\nCALL identity\nCALL read\n'
                boxed = ('CALL make\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nCALL read\n'
                         'CALL make\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nCALL forward\nCALL read\n')
                self.run_module(
                    '.string a "a"\n.string b "b"\n.entry main\n'
                    '.function main 0 0 0 int 1\n' +
                    (boxed + direct if global_first else direct + boxed) +
                    'PUSH_I64 0\nRET\n.end\n'
                    '.function make 0 0 0 struct 1\n'
                    'PUSH_STR a\nPUSH_STR b\nSTR_CONCAT\nDUP\nARR_LITERAL 5 1\n'
                    'AGG_PACK 0 0 0 2\nAGG_PACK 0 0 0 1\nRET\n.end\n'
                    '.function forward 1 1 0 struct 1\nLOAD_LOCAL 0\nTAIL_CALL identity\n.end\n'
                    '.function identity 1 1 0 struct 1\nLOAD_LOCAL 0\nRET\n.end\n'
                    '.function read 1 1 0 void 0\nPUSH_VOID\nSTORE_GLOBAL 0\nCALL churn\n'
                    'LOAD_LOCAL 0\nAGG_GET 0\nAGG_GET 0\nPRINTLN\n'
                    'LOAD_LOCAL 0\nAGG_GET 0\nAGG_GET 1\nPUSH_I64 0\nARR_GET\n'
                    'PUSH_STR a\nPUSH_STR b\nSTR_CONCAT\nSTR_EQ\nASSERT\nRET\n.end\n' +
                    closures.CHURN, 'ab\nab\nab\n', collections=True)

    def test_invalid_boxed_record_argument_traps(self):
        for value in ('PUSH_VOID', 'PUSH_I64 99', 'PUSH_BOOL 1\nAGG_PACK 0 0 0 1', 'AGG_PACK 0 0 0 0'):
            with self.subTest(value=value), tempfile.TemporaryDirectory(prefix='nano-record-call-') as tmp:
                work = Path(tmp)
                assembly, module, source, binary = [work/name for name in ('input.nasm','input.nvm','input.c','program')]
                assembly.write_text(
                    '.entry main\n.function main 0 0 0 int 1\n'
                    'PUSH_I64 7\nAGG_PACK 0 0 0 1\nCALL read\nPOP\n' + value +
                    '\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nCALL read\nRET\n.end\n'
                    '.function read 1 1 0 int 1\nLOAD_LOCAL 0\nAGG_GET 0\nPUSH_I64 1\nI64_ADD\nRET\n.end\n')
                self.checked([closures.ROOT/'bin/nanoisa','asm',assembly,'-o',module])
                self.checked([closures.ROOT/'bin/nano_vm','--verify-only',module])
                self.checked([closures.ROOT/'bin/nvm2c',module,'-o',source])
                cc = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC') or os.environ.get('CC') or 'cc')
                self.checked([*cc,'-std=c11','-Wall','-Wextra','-Werror','-fsanitize=address,undefined',
                              '-fno-sanitize-recover=all',source,'-lm','-o',binary])
                for label, command in [('vm',[closures.ROOT/'bin/nano_vm',module]),('native',[binary])]:
                    result = subprocess.run(list(map(str,command)),capture_output=True,text=True,timeout=90)
                    self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
                    if label == 'native':
                        self.assertIn('native invariant',result.stderr)
                        self.assertNotIn('ERROR: AddressSanitizer',result.stderr)


if __name__ == '__main__':
    unittest.main()
