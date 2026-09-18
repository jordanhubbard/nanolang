"""I observe exact floating array results across existing source routes."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
SIGN=1<<63
ONE=0x3ff0000000000000
INF=0x7ff0000000000000
NAN=0x7ff8000000000000
NANS=[0x7ff0000000000001,0xfff0000000000042,0x7ff8123456789abc,0xfff8000000001234]
CASES=[]
for op in range(4):
    for value in NANS:
        CASES.extend([(op,value,ONE,NAN),(op,ONE,value,NAN)])
for value in [0,SIGN,ONE,INF,INF|SIGN,*NANS]:
    CASES.extend([(3,value,0,0),(3,value,SIGN,0)])
CASES.extend([(0,INF,INF|SIGN,NAN),(1,INF,INF,NAN),(2,0,INF,NAN),
              (2,INF,SIGN,NAN),(3,INF,INF,NAN),(0,SIGN,SIGN,SIGN),
              (0,SIGN,0,0),(1,SIGN,0,SIGN),(2,SIGN,ONE,SIGN),
              (3,SIGN,ONE,SIGN),(0,ONE,0x3ca0000000000000,ONE),
              (0,ONE+1,0x3ca0000000000000,ONE+2),
              (2,1,0x3fe0000000000000,0),(2,3,0x3fe0000000000000,2),
              (2,SIGN|1,0x3fe0000000000000,SIGN),(0,1,1,2),
              (0,0x7fefffffffffffff,0x7fefffffffffffff,INF),
              (3,ONE,0x4008000000000000,0x3fd5555555555555)])
def signed(value):return value if value<SIGN else value-(1<<64)
def arithmetic_source():
    parts=[]
    for i,op in enumerate('+-*/'):
        parts.append(f'''fn check{i}(xb:int,yb:int,want:int)->void {{
 let x:float=(float_from_bits xb)
 let y:float=(float_from_bits yb)
 let a: array<float> = [x]
 let b: array<float> = [y]
 let pair: array<float> = ({op} a b)
 let left: array<float> = ({op} x b)
 let right: array<float> = ({op} a y)
 assert (== (float_to_bits (at pair 0)) want)
 assert (== (float_to_bits (at left 0)) want)
 assert (== (float_to_bits (at right 0)) want)
 assert (== (float_to_bits (at a 0)) xb)
 assert (== (float_to_bits (at b 0)) yb)
}}
shadow check{i} {{ (check{i} {ONE} {ONE} {[0x4000000000000000,0,ONE,ONE][i]}) }}
''')
    parts.append('fn main()->int {\n')
    parts.extend(f' (check{op} {signed(x)} {signed(y)} {signed(want)})\n' for op,x,y,want in CASES)
    parts.append(' return 0\n}\nshadow main { assert (== (main) 0) }\n')
    return ''.join(parts)
ORDER='''let mut trace:int=0
fn scalar()->float {set trace (+ (* trace 10) 1) return 8.0}
shadow scalar {let old:int=trace set trace 0 assert (== (scalar) 8.0) assert (== trace 1) set trace old}
fn source(values:array<float>)->array<float> {
 set trace (+ (* trace 10) 2)
 let grown:array<float> = (array_push values 2.0)
 return grown
}
shadow source {
 let old:int=trace set trace 0
 let values:array<float> = [1.0]
 let result:array<float> = (source values)
 assert (== (array_length result) 2) assert (== trace 2) set trace old
}
fn main()->int {
 set trace 0
 let values:array<float> = [1.0]
 let alias:array<float> = values
 let result:array<float> = (- (scalar) (source values))
 assert (== trace 12) assert (== (array_length result) 2)
 assert (== (at result 0) 7.0) assert (== (at result 1) 6.0)
 assert (== (array_length alias) 2) assert (== (at alias 0) 1.0)
 let empty:array<float> = []
 let empty_result:array<float> = (/ empty -0.0)
 assert (== (array_length empty_result) 0)
 let ints:array<int> = [2,3]
 let integer_result:array<int> = (- 9 ints)
 assert (== (at integer_result 0) 7) assert (== (at integer_result 1) 6)
 return 0
}
shadow main {let old:int=trace assert (== (main) 0) set trace old}
'''
class AggregateBinary64(unittest.TestCase):
    def setUp(self):
        self.work=Path(tempfile.mkdtemp(prefix='nano-aggregate-policy-'))
        self.serial=0
        print('I retain aggregate policy observations at',self.work,flush=True)
    def command(self,*args):
        args=list(map(str,args));self.serial+=1
        result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=240)
        (self.work/f'{self.serial}.log').write_text(json.dumps(args)+'\n'+result.stdout+result.stderr)
        self.assertEqual(result.returncode,0,str(args)+'\n'+result.stdout+result.stderr)
        return result
    def source(self,text,name):
        path=self.work/(name+'.nano');path.write_text(text);return path
    def legacy(self,source):
        self.command(ROOT/'bin/nano',source)
        for compiler in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
            with self.subTest(compiler=compiler):
                exe=self.work/compiler
                self.command(ROOT/'bin'/compiler,source,'-o',exe)
                self.command(exe)
    def canonical(self,source):
        for compiler in ('nano_virt','nanoc_stage1','nanoc_stage2'):
            with self.subTest(compiler=compiler):
                module=self.work/(compiler+'.nvm')
                self.command(ROOT/'bin'/compiler,source,'--emit-nvm','-o',module)
                self.command(ROOT/'bin/nano_vm','--verify-only',module)
                self.command(ROOT/'bin/nano_vm',module)
    def test_exact_arithmetic_existing_legacy_routes(self):
        self.legacy(self.source(arithmetic_source(),'exact-bits'))
    def test_exact_arithmetic_canonical_vm_routes(self):
        self.canonical(self.source(arithmetic_source(),'canonical-bits'))
    def test_order_alias_empty_and_neighbors_legacy(self):
        self.legacy(self.source(ORDER,'ordered-broadcast'))
    def test_order_alias_empty_and_neighbors_canonical_vm(self):
        self.canonical(self.source(ORDER,'canonical-order'))
    def test_selfhost_mixed_numeric_types_preserve_output(self):
        cases = [('array<float>', '[1.0]', 'int', '1'),
                 ('array<int>', '[1]', 'float', '1.0'),
                 ('array<float>', '[1.0]', 'array<int>', '[1]')]
        for left_type, left, right_type, right in cases:
            source = self.source(f'fn main()->int {{let a:{left_type} = {left} let b:{right_type} = {right} let result:{left_type} = (+ a b) return 0}}', 'mismatch')
            for compiler in ('nanoc_stage1', 'nanoc_stage2'):
                for flags in (('--target', 'c'), ('--emit-nvm',)):
                    with self.subTest(left=left_type, right=right_type, compiler=compiler, flags=flags):
                        output = self.work/'previous'
                        output.write_bytes(b'previous output')
                        args = [str(ROOT/'bin'/compiler), str(source), *flags, '-o', str(output)]
                        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=240)
                        self.serial += 1
                        (self.work/f'{self.serial}.log').write_text(json.dumps(args)+'\n'+result.stdout+result.stderr)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn('matching flat numeric', result.stdout+result.stderr)
                        self.assertEqual(output.read_bytes(), b'previous output')
if __name__=='__main__':unittest.main()
