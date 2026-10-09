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


if __name__ == '__main__':
    unittest.main()
