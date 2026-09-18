"""I qualify the public standalone C route with integer binary64 observations."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
class PublicCBinary64(unittest.TestCase):
    def setUp(self):
        self.work=Path(tempfile.mkdtemp(prefix='nano-public-c-binary64-'))
        print('I retain public C evidence at',self.work,flush=True)
    def command(self,*args,success=True):
        args=list(map(str,args))
        result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=240)
        message=f'tool={args[0]} command={args!r}\n{result.stdout}\n{result.stderr}'
        (self.assertEqual if success else self.assertNotEqual)(result.returncode,0,message)
        return result
    def source(self,text):
        path=self.work/'fresh.nano';path.write_text(text);return path
    def portable(self,text):
        source=self.source(text);output=self.work/'public.c'
        self.command(ROOT/'bin/nanoc_c','--target','c',source,'-o',output)
        emitted=output.read_text()
        self.assertNotIn('({',emitted)
        self.assertNotIn('__auto_type',emitted)
        for standard in ('c99','c11'):
            for optimization in ('-O0','-O2'):
                exe=self.work/(standard+optimization)
                self.command(os.environ.get('CC','cc'),'-std='+standard,'-pedantic-errors','-Werror=implicit-function-declaration','-Werror=return-type','-Werror=unused-local-typedefs',optimization,'-ffp-contract=fast','-fsanitize=undefined','-fno-sanitize-recover=all',output,'-lm','-o',exe)
                self.command(exe)
        return source,emitted
    def test_exact_transport_arithmetic_and_rounding(self):
        source,emitted=self.portable(ARITHMETIC)
        self.assertIn('volatile double rounded',emitted)
        self.command(ROOT/'bin/nano',source)
        exe=self.work/'legacy';self.command(ROOT/'bin/nanoc_c',source,'-o',exe);self.command(exe)
        module=self.work/'ordinary.nvm'
        self.command(ROOT/'bin/nano_virt',source,'--emit-nvm','-o',module)
        self.command(ROOT/'bin/nano_vm','--verify-only',module);self.command(ROOT/'bin/nano_vm',module)
        native=self.work/'native.c';self.command(ROOT/'bin/nvm2c',module,'-o',native)
        exe=self.work/'canonical-native';self.command(os.environ.get('CC','cc'),'-std=c11','-O2','-fsanitize=address,undefined',native,'-lm','-o',exe);self.command(exe)
    def test_operands_loops_branches_and_namespace(self):
        self.portable(ORDER)
    def test_global_order_and_main_reentry(self):
        self.portable(GLOBALS)
        self.portable(INITIALIZER_REENTRY)
    def test_exact_decimal_literal(self):
        self.portable('fn main()->int{assert (== (float_to_bits 1.0000000000000002) 4607182418800017409) return 0} shadow main{assert (== (main) 0)}')
    def test_wrong_transport_type_preserves_previous_source(self):
        source=self.source('fn main()->int{return (float_to_bits true)} shadow main{assert true}')
        output=self.work/'previous.c';output.write_text('I retain previous output.\n')
        result=self.command(ROOT/'bin/nanoc_c','--target','c',source,'-o',output,success=False)
        self.assertIn('exactly typed operand',result.stdout+result.stderr)
        self.assertEqual(output.read_text(),'I retain previous output.\n')
        self.assertEqual(list(self.work.glob('previous.c.tmp.*')),[])
    def test_c_api_failure_recovery_and_type_resolution(self):
        exe=self.work/'api'
        self.command(os.environ.get('CC','cc'),'-std=c99','-D_POSIX_C_SOURCE=200809L','-Wall','-Wextra','-Werror','-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all','-I',ROOT/'src',ROOT/'tests/test_public_c_api.c','-o',exe)
        self.command(exe,self.work/'api-output.c')
    def test_shared_provider_identity(self):
        self.command('python3',ROOT/'scripts/embed_binary64_arithmetic.py','--check')

ARITHMETIC='''fn add(a:float,b:float)->float{return (+ a b)}
shadow add{assert (== (add 1.0 2.0) 3.0)}
fn main()->int{
 let sn:float=(float_from_bits 9218868437227405313)
 let nq:float=(float_from_bits -2251799813685247)
 let inf:float=(float_from_bits 9218868437227405312)
 let z:float=(float_from_bits -9223372036854775808)
 assert (== (float_to_bits sn) 9218868437227405313)
 assert (== (float_to_bits nq) -2251799813685247)
 assert (== (float_to_bits (+ sn 1.0)) 9221120237041090560)
 assert (== (float_to_bits (- inf inf)) 9221120237041090560)
 assert (== (float_to_bits (* nq 0.0)) 9221120237041090560)
 assert (== (float_to_bits (/ inf inf)) 9221120237041090560)
 assert (== (float_to_bits (/ sn z)) 0)
 assert (== (float_to_bits (/ -1.0 0.0)) 0)
 assert (== (float_to_bits (* z 1.0)) -9223372036854775808)
 assert (== (float_to_bits (+ (float_from_bits 1) (float_from_bits 1))) 2)
 assert (== (float_to_bits (+ 1.0 (float_from_bits 4368491638549381120))) 4607182418800017408)
 let a:float=(float_from_bits 4607182418800017409)
 let b:float=(float_from_bits 4607182418800017407)
 assert (== (float_to_bits (- (* a b) 1.0)) 0)
 assert (== (float_to_bits (add 1.0 2.0)) 4613937818241073152)
 assert (== (float_to_bits sn) 9218868437227405313)
 return 0
}
shadow main{assert (== (main) 0)}
'''
ORDER='''let mut sequence:int=0
fn operand(id:int,x:float)->float{set sequence (+ (* sequence 10) id) return x}
shadow operand{let saved:int=sequence assert (== (operand 1 2.0) 2.0) set sequence saved}
fn nano_cb_0_f64_add(x:float)->float{return (+ x 1.0)}
shadow nano_cb_0_f64_add{assert (== (nano_cb_0_f64_add 1.0) 2.0)}
fn f64_add(x:float)->float{return (+ x 2.0)}
shadow f64_add{assert (== (f64_add 1.0) 3.0)}
fn main()->int{
 set sequence 0
 let nano_cb_1_l:float=1.0
 let main:int=7
 assert (== main 7)
 let x:float=(+ (* (operand 1 2.0) (operand 2 3.0)) (- (operand 3 8.0) (operand 4 4.0)))
 assert (== (float_to_bits x) 4621819117588971520)
 assert (== sequence 1234)
 set sequence 0
 let mut i:int=0
 let mut total:float=0.0
 while (< i 2){set total (+ total (operand 5 nano_cb_1_l)) set i (+ i 1)}
 assert (== sequence 55)
 if (== i 2){set total (+ total (operand 6 2.0))} else {set total (+ total (operand 7 20.0))}
 if false {set total (+ total (operand 8 20.0))} else {set total (+ total (operand 9 1.0))}
 assert (== sequence 5569)
 assert (== (float_to_bits total) 4617315517961601024)
 assert (== (float_to_bits (f64_add (nano_cb_0_f64_add 1.0))) 4616189618054758400)
 return 0
}
shadow main{let saved:int=sequence assert (== (main) 0) set sequence saved}
'''
GLOBALS='''let mut sequence:int=0
fn operand(id:int,x:float)->float{set sequence (+ (* sequence 10) id) return x}
shadow operand{let saved:int=sequence assert (== (operand 1 2.0) 2.0) set sequence saved}
let first:float=(+ (operand 1 1.0) (operand 2 2.0))
let mut second:float=(/ (+ first 1.0) -0.0)
let third:float=(- (+ first second))
let mut entered:int=0
fn main()->int{
 if (== entered 1){assert (== sequence 12) return 0}
 set entered 1
 assert (== (main) 0)
 assert (== sequence 12)
 assert (== (float_to_bits first) 4613937818241073152)
 assert (== (float_to_bits second) 0)
 assert (== (float_to_bits third) -4609434218613702656)
 set second (+ first 2.0)
 assert (== (float_to_bits second) 4617315517961601024)
 return 0
}
shadow main{assert true}
'''
INITIALIZER_REENTRY='''let mut entered:int=0
let trigger:int=(main)
let late:float=(+ 1.0 2.0)
fn main()->int{
 if (== entered 0){set entered 1 return 1099511627776}
 assert (== trigger 1099511627776)
 assert (== (float_to_bits late) 4613937818241073152)
 return 0
}
shadow main{assert true}
'''
if __name__=='__main__':unittest.main()
