"""I preserve typed root cleanup without hiding observable void assignments."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RECORD = "PUSH_I64 7\nAGG_PACK 0 0 0 1\nSTORE_LOCAL 0\n"
CLEAR = "PUSH_VOID\nSTORE_LOCAL 0\n"
READ = "LOAD_LOCAL 0\nAGG_GET 0\nPRINTLN\n"
END = "PUSH_I64 0\nRET\n"

class NativeLocalClear(unittest.TestCase):
    def checked(self, *args):
        p = subprocess.run(list(map(str,args)), cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(p.returncode, 0, p.stdout+p.stderr)
        return p

    def module(self, work, body):
        assembly, module = work/'input.nasm', work/'input.nvm'
        assembly.write_text('.entry 0\n.function main 0 2 0 int 1\n'+body+'.end\n')
        self.checked(ROOT/'bin/nanoisa', 'asm', assembly, '-o', module)
        self.checked(ROOT/'bin/nano_vm', '--verify-only', module)
        return module

    def test_dead_clear_and_reassignment_across_control_flow(self):
        cases = {
            'terminal': RECORD+READ+CLEAR+END,
            'branch': RECORD+CLEAR+'PUSH_BOOL 1\nJMP_FALSE other\n'+RECORD+'JMP join\nother:\n'+RECORD+'join:\n'+READ+END,
            'loop': 'PUSH_I64 0\nSTORE_LOCAL 1\ntop:\n'+RECORD+READ+CLEAR+
                    'LOAD_LOCAL 1\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nPUSH_I64 3\nI64_LT_S\nJMP_TRUE top\n'+END,
        }
        for name, body in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory(prefix='native-local-clear-') as tmp:
                work=Path(tmp);module=self.module(work,body)
                expected=self.checked(ROOT/'bin/nano_vm',module).stdout
                self.assertEqual(expected, '7\n'*(3 if name=='loop' else 1))
                c,native=work/'native.c',work/'native'
                self.checked(ROOT/'bin/nvm2c',module,'-o',c)
                self.checked(*shlex.split(os.environ.get('CC','clang')),'-std=c11','-Wall','-Wextra','-Werror',
                             '-fsanitize=address,undefined','-fno-sanitize-recover=all',c,'-lm','-o',native)
                self.assertEqual(self.checked(native).stdout,expected)

    def test_observable_clear_is_not_erased(self):
        cases = {
            'direct': RECORD+CLEAR+READ+END,
            'branch': RECORD+CLEAR+'PUSH_BOOL 1\nJMP_TRUE join\n'+RECORD+'join:\n'+READ+END,
            'backedge': RECORD+'top:\n'+READ+CLEAR+'JMP top\n',
            'join_store': RECORD+'PUSH_BOOL 1\nJMP_FALSE clear\nPUSH_I64 9\nAGG_PACK 0 0 0 1\nJMP store\nclear:\nPUSH_VOID\nstore:\nSTORE_LOCAL 0\n'+READ+END,
        }
        for name,body in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory(prefix='native-observed-clear-') as tmp:
                work=Path(tmp);module=self.module(work,body);c=work/'native.c'
                c.write_text('previous output\n')
                p=subprocess.run([str(ROOT/'bin/nvm2c'),str(module),'-o',str(c)],cwd=ROOT,capture_output=True,text=True,timeout=120)
                self.assertNotEqual(p.returncode,0)
                self.assertEqual(c.read_text(),'previous output\n')
                self.assertIn('cannot',p.stderr)

if __name__=='__main__':
    unittest.main()
