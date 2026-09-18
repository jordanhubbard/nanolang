"""I compare exact bit identities across ordinary scalar backend routes."""
import random
import subprocess
import unittest
from tests import test_nvm2wasm as wasm
from tests.test_canonical_f64_bits import PATTERNS

class Binary64BitTransport(unittest.TestCase):
    setUp=wasm.ScalarWasm.setUp
    run_cmd=wasm.ScalarWasm.run_cmd
    module=wasm.ScalarWasm.module
    compare=wasm.ScalarWasm.compare

    def test_all_bit_classes_and_two_inverse_directions(self):
        rng=random.Random(0x8d8e)
        values=list(dict.fromkeys(PATTERNS+tuple(1<<i for i in range(64))+
                                 tuple(rng.getrandbits(64) for _ in range(32))))
        # Small independent chunks stay below existing native temporary limits.
        for start in range(0,len(values),12):
            with self.subTest(chunk=start):
                body=''
                for bits in values[start:start+12]:
                    signed=bits if bits<(1<<63) else bits-(1<<64)
                    body+=f'PUSH_I64 {signed}\nF64_FROM_BITS\nF64_TO_BITS\nPUSH_I64 {signed}\nI64_EQ\nASSERT\n'
                    body+=f'PUSH_F64 bits:{bits:016x}\nF64_TO_BITS\nF64_FROM_BITS\nF64_TO_BITS\nPUSH_I64 {signed}\nI64_EQ\nASSERT\n'
                self.compare('.entry main\n.function main 0 0 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n')

    def test_transport_calls_globals_locals_and_join(self):
        bits=0xfff0000000000042;signed=bits-(1<<64)
        text=('.entry main\n.function relay 1 1 0 float 1\n.parameters relay float\nLOAD_LOCAL 0\nRET\n.end\n'
              '.function main 0 1 0 int 1\n'+f'PUSH_I64 {signed}\nF64_FROM_BITS\nCALL relay\nSTORE_GLOBAL 0\n'
              'LOAD_GLOBAL 0\nSTORE_LOCAL 0\nPUSH_BOOL 0\nJMP_FALSE right\nLOAD_LOCAL 0\nF64_TO_BITS\nF64_FROM_BITS\nJMP joined\nright:\n'
              f'PUSH_I64 {signed}\nF64_FROM_BITS\njoined:\nF64_TO_BITS\nPUSH_I64 {signed}\nI64_EQ\nASSERT\n'
              'PUSH_I64 0\nRET\n.end\n')
        self.compare(text)
        self.compare(text.replace("PUSH_BOOL 0", "PUSH_BOOL 1"))

    def test_wrong_exact_tags_are_verified_refusals(self):
        for op,producer in [('F64_FROM_BITS','PUSH_BOOL 1'),('F64_FROM_BITS','PUSH_U8 7'),
                            ('F64_FROM_BITS','PUSH_F64 1.0'),('F64_TO_BITS','PUSH_I64 1'),
                            ('F64_TO_BITS','PUSH_BOOL 1')]:
            with self.subTest(op=op,producer=producer):
                assembly=self.work/'refuse.nasm';out=self.work/'retained.nvm'
                assembly.write_text('.entry main\n.function main 0 0 0 int 1\n'+producer+'\n'+op+'\nPOP\nPUSH_I64 0\nRET\n.end\n')
                out.write_bytes(b'retained')
                result=subprocess.run([wasm.ROOT/'bin/nanoisa','asm',assembly,'-o',out],capture_output=True,text=True,timeout=30)
                self.assertGreater(result.returncode,0)
                self.assertEqual(out.read_bytes(),b'retained')

if __name__=='__main__':unittest.main()
