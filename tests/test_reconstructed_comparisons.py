"""I retain generic equality separately from exact INT/BOOL ordering."""
from pathlib import Path
import re
import tempfile
import unittest
from tests import test_reconstructed_integer_addition as addition
from tests import test_reconstructed_wide_multiply as wide

ROOT=Path(__file__).resolve().parents[1]
SHADOW='shadow nlr_f0_main { assert (== (nlr_f0_main) 0) }'

def push(value):
    return f'PUSH_BOOL {int(value)}\n' if type(value)==bool else f'PUSH_I64 {value}\n'

def comparison(op,a,b):
    same=type(a)==type(b)
    order=(int(a)>int(b))-(int(a)<int(b)) if same else (1 if type(a)==bool else -1)
    equal=same and a==b
    return {'EQ':equal,'NE':not equal,'LT':order<0,'LE':order<=0,'GT':order>0,'GE':order>=0}[op]

class ScalarComparisons(unittest.TestCase):
    check_once=False
    inspect_calls=False
    paired=addition.IntegerReconstruction.paired
    assemble=wide.WideMultiply.assemble

    def checked(self,args,expected=0):
        result=wide.WideMultiply.checked(self,args,expected)
        if self.inspect_calls and Path(args[0]).name=='nvm2hl':
            source=Path(args[-1]).read_text()
            for index,name in ((1,'integer'),(2,'boolean')):
                calls=re.findall(rf'nlr_f{index}_{name}\(nlr_t\d+\)|\(nlr_f{index}_{name} nlr_t\d+\)',source)
                self.assertEqual(len(calls),1)
        return result

    def test_each_operator_and_both_tag_orders(self):
        pairs=(-(1<<63),(1<<63)-1),((1<<63)-1,-(1<<63)),(7,7),(False,True),(True,False),(False,False),(True,True),(0,False),(0,True),(99,False),(False,0),(True,-7)
        for op in ('EQ','NE','LT','LE','GT','GE'):
            with self.subTest(op=op):
                body=''
                for a,b in pairs:
                    body+=push(a)+push(b)+op+'\n'
                    if not comparison(op,a,b): body+='BOOL_NOT\n'
                    body+='JMP_FALSE bad\n'
                self.paired('.entry main\n.function main 0 0 0 int 1\n'+body+
                    'PUSH_I64 0\nRET\nbad:\nPUSH_I64 1\nRET\n.end\n',0,SHADOW)

    def test_mixed_constant_retains_calls_and_local_snapshot(self):
        self.inspect_calls=True
        try:
            self.paired('''.entry main
.function main 0 1 0 int 1
PUSH_I64 0
CALL integer
PUSH_BOOL 0
CALL boolean
EQ
BOOL_NOT
JMP_FALSE bad
PUSH_I64 3
STORE_LOCAL 0
LOAD_LOCAL 0
PUSH_I64 9
STORE_LOCAL 0
PUSH_I64 3
EQ
JMP_FALSE bad
PUSH_I64 0
RET
bad:
PUSH_I64 1
RET
.end
.function integer 1 1 0 int 1
.parameters integer int
LOAD_LOCAL 0
RET
.end
.function boolean 1 1 0 bool 1
.parameters boolean bool
LOAD_LOCAL 0
RET
.end
''',0,SHADOW+'\nshadow nlr_f1_integer { assert (== (nlr_f1_integer 0) 0) }\nshadow nlr_f2_boolean { assert (not (nlr_f2_boolean false)) }')
        finally:self.inspect_calls=False

    def test_pure_loop_comparisons(self):
        self.paired('''.entry main
.function main 0 1 0 int 1
PUSH_I64 0
STORE_LOCAL 0
loop:
LOAD_LOCAL 0
PUSH_I64 3
LT
PUSH_BOOL 1
EQ
JMP_FALSE done
LOAD_LOCAL 0
PUSH_I64 1
I64_ADD
STORE_LOCAL 0
JMP loop
done:
LOAD_LOCAL 0
PUSH_I64 3
EQ
JMP_FALSE bad
PUSH_I64 0
RET
bad:
PUSH_I64 1
RET
.end
''',0,SHADOW)

    def test_other_tags_preserve_output(self):
        for op in ('EQ','NE','LT','LE','GT','GE'):
            for value in ('PUSH_U8 0','PUSH_F64 1.5','PUSH_VOID'):
                with self.subTest(op=op,value=value),tempfile.TemporaryDirectory() as temp:
                    d=Path(temp)
                    module=addition.IntegerReconstruction.assemble(self,d,'.entry main\n.function main 0 0 0 int 1\n'+value+'\nPUSH_I64 0\n'+op+'\nPOP\nPUSH_I64 0\nRET\n.end\n')
                    for language in ('c','nano'):
                        out=d/('previous.'+language);out.write_text('previous')
                        result=self.checked([ROOT/'bin/nvm2hl','--language',language,module,'-o',out],1)
                        self.assertIn('I ',result.stderr)
                        self.assertEqual(out.read_text(),'previous')

if __name__=='__main__':unittest.main()
