"""I rotate only exact INT/BOOL native values with unchanged snapshots."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests import test_integer_pair_verification as pairs

ROOT=Path(__file__).resolve().parents[1]

class NativeScalarRot3(unittest.TestCase):
    checked=pairs.IntegerPairVerification.checked
    paired=pairs.IntegerPairVerification.paired

    def test_mixed_tags_and_three_rotations(self):
        self.paired('PUSH_I64 11\nPUSH_BOOL 1\nPUSH_I64 33\nROT3\n'
                    'DUP\nTYPE_CHECK 4\nASSERT\nASSERT\n'
                    'PUSH_I64 11\nI64_EQ\nASSERT\nPUSH_I64 33\nI64_EQ\nASSERT\n'
                    'PUSH_I64 4\nPUSH_I64 5\nPUSH_I64 6\nROT3\nROT3\nROT3\n'
                    'PUSH_I64 6\nI64_EQ\nASSERT\nPUSH_I64 5\nI64_EQ\nASSERT\nPUSH_I64 4\nI64_EQ\nASSERT\n')

    def test_calls_local_snapshots_and_loop(self):
        self.paired('PUSH_I64 7\nSTORE_LOCAL 0\nPUSH_I64 11\nCALL identity\nLOAD_LOCAL 0\n'
                    'PUSH_BOOL 0\nPUSH_I64 99\nSTORE_LOCAL 0\nROT3\n'
                    'PUSH_I64 7\nI64_EQ\nASSERT\nPUSH_I64 11\nI64_EQ\nASSERT\nBOOL_NOT\nASSERT\n'
                    'PUSH_I64 0\nSTORE_LOCAL 0\nloop:\nPUSH_I64 0\nLOAD_LOCAL 0\nPUSH_I64 3\n'
                    'ROT3\nPOP\nPOP\nLOAD_LOCAL 0\nI64_GT_S\nJMP_FALSE done\n'
                    'LOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 0\nJMP loop\ndone:\n'
                    'LOAD_LOCAL 0\nPUSH_I64 3\nI64_EQ\nASSERT\n',
                    '.function identity 1 1 0 int 1\n.parameters identity int\nLOAD_LOCAL 0\nRET\n.end\n')

    def test_other_kinds_preserve_previous_output(self):
        for operand in ('PUSH_F64 1.5','PUSH_U8 7','PUSH_STR 0','PUSH_VOID'):
            with self.subTest(operand=operand),tempfile.TemporaryDirectory() as temp:
                d=Path(temp);asm=d/'ordinary.nasm';module=d/'ordinary.nvm';out=d/'previous.c'
                asm.write_text('.string 0 "text"\n.entry main\n.function main 0 0 0 int 1\nPUSH_I64 1\n'+operand+'\nPUSH_BOOL 1\nROT3\nPOP\nPOP\nPOP\nPUSH_I64 0\nRET\n.end\n')
                self.checked(ROOT/'bin/nanoisa','asm',asm,'-o',module)
                out.write_text('previous')
                result=subprocess.run([ROOT/'bin/nvm2c',module,'-o',out],capture_output=True,text=True)
                self.assertEqual(result.returncode,1,result.stdout+result.stderr)
                self.assertIn('exact int/bool operands for native ROT3',result.stderr)
                self.assertEqual(out.read_text(),'previous')

    def test_underflow_preserves_previous_module(self):
        for count in range(3):
            with self.subTest(count=count),tempfile.TemporaryDirectory() as temp:
                d=Path(temp);asm=d/'short.nasm';out=d/'previous.nvm'
                asm.write_text('.entry main\n.function main 0 0 0 int 1\n'+'PUSH_I64 1\n'*count+'ROT3\n'+'POP\n'*count+'PUSH_I64 0\nRET\n.end\n')
                out.write_bytes(b'previous')
                result=subprocess.run([ROOT/'bin/nanoisa','asm',asm,'-o',out],capture_output=True,text=True)
                self.assertEqual(result.returncode,1,result.stdout+result.stderr)
                self.assertIn('underflow',result.stderr.lower())
                self.assertEqual(out.read_bytes(),b'previous')

if __name__=='__main__': unittest.main()
