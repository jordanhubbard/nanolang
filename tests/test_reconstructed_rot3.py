"""I preserve scalar ROT3 order without repeating evaluated operands."""
from pathlib import Path
import re
import os
import subprocess
import tempfile
import unittest
from tests import test_reconstructed_integer_addition as addition
from tests import test_reconstructed_wide_multiply as wide

ROOT=Path(__file__).resolve().parents[1]
SHADOW='shadow nlr_f0_main { assert (== (nlr_f0_main) 0) }'

class ScalarRot3(unittest.TestCase):
    check_once=False
    inspect_call=False
    paired=addition.IntegerReconstruction.paired
    def assemble(self,directory,text):
        module=addition.IntegerReconstruction.assemble(self,directory,text)
        source,binary=directory/'native.c',directory/'native'
        self.checked([os.environ.get('NVM2C',str(ROOT/'bin/nvm2c')),module,'-o',source])
        self.checked([os.environ.get('CC','cc'),'-std=c11','-O1','-Wall','-Wextra','-Werror',
                      '-fsanitize=address,undefined','-fno-sanitize-recover=all',source,'-o',binary])
        self.checked([binary])
        return module

    def checked(self,args,expected=0):
        result=wide.WideMultiply.checked(self,args,expected)
        if self.inspect_call and Path(args[0]).name=='nvm2hl':
            source=Path(args[-1]).read_text()
            calls=re.findall(r'nlr_f1_identity\(nlr_t\d+\)|\(nlr_f1_identity nlr_t\d+\)',source)
            self.assertEqual(len(calls),1)
        return result

    def test_distinct_values_and_mixed_tags(self):
        for values in ((11,22,33),(True,22,False),(11,False,True),(True,False,True)):
            with self.subTest(values=values):
                body=''.join(f'PUSH_BOOL {int(v)}\n' if type(v)==bool else f'PUSH_I64 {v}\n' for v in values)
                body+='ROT3\nSTORE_LOCAL 2\nSTORE_LOCAL 1\nSTORE_LOCAL 0\n'
                for slot,value in enumerate((values[2],values[0],values[1])):
                    body+=f'LOAD_LOCAL {slot}\nCAST_INT\nPUSH_I64 {int(value)}\nI64_EQ\nJMP_FALSE bad\n'
                self.paired('.entry main\n.function main 0 3 0 int 1\n'+body+
                    'PUSH_I64 0\nRET\nbad:\nPUSH_I64 1\nRET\n.end\n',0,SHADOW)

    def test_call_local_snapshot_and_pure_loop(self):
        self.inspect_call=True
        try:
            self.paired('''.entry main
.function main 0 4 0 int 1
PUSH_I64 7
STORE_LOCAL 0
PUSH_I64 11
CALL identity
LOAD_LOCAL 0
PUSH_I64 19
PUSH_I64 99
STORE_LOCAL 0
ROT3
STORE_LOCAL 3
STORE_LOCAL 2
STORE_LOCAL 1
LOAD_LOCAL 1
PUSH_I64 19
I64_EQ
JMP_FALSE bad
LOAD_LOCAL 2
PUSH_I64 11
I64_EQ
JMP_FALSE bad
LOAD_LOCAL 3
PUSH_I64 7
I64_EQ
JMP_FALSE bad
PUSH_I64 0
STORE_LOCAL 0
loop:
PUSH_I64 0
LOAD_LOCAL 0
PUSH_I64 3
ROT3
POP
POP
LOAD_LOCAL 0
I64_GT_S
JMP_FALSE done
LOAD_LOCAL 0
PUSH_I64 1
I64_ADD
STORE_LOCAL 0
JMP loop
done:
LOAD_LOCAL 0
PUSH_I64 3
I64_EQ
JMP_FALSE bad
PUSH_I64 0
RET
bad:
PUSH_I64 1
RET
.end
.function identity 1 1 0 int 1
.parameters identity int
LOAD_LOCAL 0
RET
.end
''',0,SHADOW+'\nshadow nlr_f1_identity { assert (== (nlr_f1_identity 11) 11) }')
        finally: self.inspect_call=False

    def test_underflow_refused_without_execution(self):
        for count in range(3):
            with self.subTest(count=count),tempfile.TemporaryDirectory() as temp:
                directory=Path(temp);source=directory/'underflow.nasm';output=directory/'previous.nvm'
                source.write_text('.entry main\n.function main 0 0 0 int 1\n'+'PUSH_I64 1\n'*count+'ROT3\n'+'POP\n'*count+'PUSH_I64 0\nRET\n.end\n')
                output.write_bytes(b'previous')
                result=subprocess.run([ROOT/'bin/nanoisa','asm',source,'-o',output],capture_output=True,text=True)
                self.assertEqual(result.returncode,1,result.stdout+result.stderr)
                self.assertIn('underflow',result.stderr.lower())
                self.assertEqual(output.read_bytes(),b'previous')

if __name__=='__main__': unittest.main()
