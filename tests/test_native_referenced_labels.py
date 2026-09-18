"""I retain only referenced PC labels without changing native flow or joins."""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]

class NativeReferencedLabels(unittest.TestCase):
    def checked(self,*args):
        p=subprocess.run(list(map(str,args)),cwd=ROOT,capture_output=True,text=True,timeout=120)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        self.assertNotIn('runtime error:',p.stderr)
        return p.stdout

    def paired(self,body,locals=1):
        with tempfile.TemporaryDirectory(prefix='native-labels-') as temp:
            d=Path(temp);asm=d/'ordinary.nasm';module=d/'ordinary.nvm';source=d/'ordinary.c';binary=d/'ordinary'
            asm.write_text(f'.entry main\n.function main 0 {locals} 0 int 1\n'+body+'\n.end\n')
            self.checked(ROOT/'bin/nanoisa','asm',asm,'-o',module)
            self.checked(ROOT/'bin/nano_vm',module)
            self.checked(ROOT/'bin/nvm2c',module,'-o',source)
            text=source.read_text()
            definitions=set(re.findall(r'^L_(\d+):',text,re.M))
            references=set(re.findall(r'goto L_(\d+);',text))
            self.assertEqual(definitions,references)
            self.assertNotRegex(text,r'if\s*\(0\)\s*goto L_\d+')
            for optimize in ('-O0','-O2'):
                self.checked(os.environ.get('CC','cc'),'-std=c11',optimize,'-Wall','-Wextra','-Werror',
                             '-fsanitize=address,undefined','-fno-sanitize-recover=all',source,'-o',binary)
                self.checked(binary)
            return text

    def test_zero_trip_false_assert_skips_dead_backedge(self):
        text=self.paired('''PUSH_I64 0
STORE_LOCAL 0
loop:
LOAD_LOCAL 0
PUSH_I64 0
I64_LT_S
JMP_FALSE done
PUSH_BOOL 0
ASSERT
JMP loop
done:
PUSH_I64 0
RET''')
        self.assertEqual(len(re.findall(r'^L_\d+:',text,re.M)),1)

    def test_conditional_forward_and_live_backward_joins(self):
        self.paired('''PUSH_I64 0
STORE_LOCAL 0
loop:
LOAD_LOCAL 0
PUSH_I64 3
I64_GE_S
JMP_TRUE done
LOAD_LOCAL 0
PUSH_I64 1
I64_ADD
STORE_LOCAL 0
JMP loop
done:
LOAD_LOCAL 0
PUSH_I64 3
I64_EQ
ASSERT
PUSH_I64 0
RET''')

    def test_skipped_return_jump_and_buffer_growth(self):
        prefix='''PUSH_BOOL 1
JMP_FALSE other
PUSH_I64 7
STORE_LOCAL 0
JMP live
other:
PUSH_I64 0
RET
JMP ghost
live:
NOP
ghost:
'''
        growth=''.join(f'PUSH_I64 {i}\nPOP\n' for i in range(400))
        text=self.paired(prefix+growth+'LOAD_LOCAL 0\nPUSH_I64 7\nI64_EQ\nASSERT\nPUSH_I64 0\nRET')
        self.assertGreater(len(text),8192)
        self.assertEqual(len(re.findall(r'^L_\d+:',text,re.M)),2)

    def test_end_of_body_target(self):
        text=self.paired('PUSH_I64 0\nJMP done\ndone:',locals=0)
        self.assertEqual(len(re.findall(r'^L_\d+:',text,re.M)),1)

if __name__=='__main__':unittest.main()
