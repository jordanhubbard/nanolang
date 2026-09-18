"""I retain exact nonfinite scalar observations across legacy source routes."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
CASES=[(0x7ff8000000000001,'nan'),(0xfff8000000000001,'-nan'),
       (0x7ff0000000000001,'nan'),(0xfff0000000000001,'-nan'),
       (0x7ff0000000000000,'inf'),(0xfff0000000000000,'-inf'),
       (0,'0.0'),(1<<63,'-0.0'),(0x3ff0000000000000,'1.0'),
       (0x7fefffffffffffff,'1.79769e+308'),(1,'4.94066e-324')]

def signed(bits):return bits if bits<1<<63 else bits-(1<<64)
def source():
    calls='\n'.join(f'    (check {signed(bits)} {json.dumps(text)})' for bits,text in CASES)
    return '''fn check(bits:int, expected:string)->void {
    let value:float=(float_from_bits bits)
    assert (== (float_to_string value) expected)
    assert (== (float_to_bits value) bits)
    (print value)
    (println "|")
}
shadow check {
    let value:float=(float_from_bits -2251799813685247)
    assert (== (float_to_string value) "-nan")
    assert (== (float_to_bits value) -2251799813685247)
}
fn main()->int {
'''+calls+'''
    return 0
}
shadow main { assert (== (float_to_string -0.0) "-0.0") }
'''

class LegacySignedNan(unittest.TestCase):
    def setUp(self):
        self.work=Path(tempfile.mkdtemp(prefix='nano-legacy-nan-'))
        print('I retain legacy format artifacts at',self.work,flush=True)
        self.sequence=0
    def command(self,*args):
        args=list(map(str,args));self.sequence+=1
        result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=240)
        (self.work/f'{self.sequence}.log').write_text(json.dumps(args)+'\n'+result.stdout+result.stderr)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr+str(self.work))
        return result
    def test_exact_bits_and_captured_scalar_output(self):
        path=self.work/'scalar.nano';path.write_text(source())
        expected=''.join(({'0.0':'0','-0.0':'-0','1.0':'1'}.get(text,text))+'|\n' for _,text in CASES)
        self.assertEqual(self.command(ROOT/'bin/nano',path).stdout,expected)
        for compiler in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
            with self.subTest(compiler=compiler):
                executable=self.work/compiler
                self.command(ROOT/'bin'/compiler,path,'-o',executable)
                route_expected = expected if compiler == "nanoc_c" else "".join(text+"|\n" for _,text in CASES)
                self.assertEqual(self.command(executable).stdout,route_expected)
    def test_interpreter_array_and_generic_format(self):
        calls=[]
        for bits,text in CASES:
            printed={'0.0':'0','-0.0':'-0','1.0':'1'}.get(text,text)
            calls.append(f'    (check {signed(bits)} {json.dumps(text)} {json.dumps(printed)})')
        program='''fn check(bits:int, scalar:string, raw:string)->void {
    let value:float=(float_from_bits bits)
    let values: array<float> = [value]
    assert (== (to_string value) scalar)
    assert (== (format "%g" value) raw)
    assert (== (to_string values) (str_concat "[" (str_concat raw "]")))
    assert (== (float_to_bits (array_get values 0)) bits)
    (print values)
    (println "|")
}
shadow check {
    let value:float=(float_from_bits -2251799813685247)
    assert (== (format "%g" value) "-nan")
    assert (== (to_string value) "-nan")
}
fn main()->int {
'''+ '\n'.join(calls)+'''
    return 0
}
shadow main { assert (== (format "%g" -0.0) "-0") }
'''
        path=self.work/'array-format.nano';path.write_text(program)
        expected=''.join('['+{'0.0':'0','-0.0':'-0','1.0':'1'}.get(text,text)+']|\n' for _,text in CASES)
        self.assertEqual(self.command(ROOT/'bin/nano',path).stdout,expected)

    def test_c_seed_array_print(self):
        body=[]
        for i,(bits,text) in enumerate(CASES):
            body.extend([f'    let value{i}: float = (float_from_bits {signed(bits)})',
                         f'    let values{i}: array<float> = [value{i}]',
                         f'    assert (== (float_to_bits (array_get values{i} 0)) {signed(bits)})',
                         f'    (print values{i})', '    (println "|")'])
        program='fn main()->int {\n'+'\n'.join(body)+'\n    return 0\n}\nshadow main { assert (== (float_to_string -0.0) "-0.0") }\n'
        path=self.work/'c-seed-array.nano';path.write_text(program)
        executable=self.work/'c-seed-array'
        self.command(ROOT/'bin/nanoc_c',path,'-o',executable)
        expected=''.join('['+text+']|\n' for _,text in CASES)
        self.assertEqual(self.command(executable).stdout,expected)

    def test_generated_provider_identity(self):
        self.command('python3',ROOT/'scripts/embed_binary64_format.py','--check')
        path=self.work/'scalar.nano';path.write_text(source())
        header=(ROOT/'src/binary64_format.h').read_text()
        for compiler in ('nanoc_stage1','nanoc_stage2'):
            output=self.work/(compiler+'.c')
            self.command(ROOT/'bin'/compiler,path,'--target','c','-o',output)
            self.assertIn(header,output.read_text())
            self.assertIn('nano_rt_f64_print(stdout, v)',output.read_text())

if __name__=='__main__':unittest.main()
