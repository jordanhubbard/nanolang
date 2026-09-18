"""I compare exact public C strings without changing their storage contract."""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
class PublicCStringEquality(unittest.TestCase):
    def setUp(self):
        self.work=Path(tempfile.mkdtemp(prefix='nano-public-c-equality-'))
        print('I retain fresh equality evidence at',self.work,flush=True)
    def run_cmd(self,args):
        r=subprocess.run(list(map(str,args)),cwd=ROOT,capture_output=True,text=True,timeout=240)
        self.assertEqual(r.returncode,0,f'command={args!r}\n{r.stdout}\n{r.stderr}')
        self.assertNotIn('Sanitizer',r.stderr);self.assertNotIn('runtime error:',r.stderr)
        return r
    def emit(self,text):
        source=self.work/'ordinary.nano';source.write_text(text)
        output=self.work/'ordinary.c';self.run_cmd([ROOT/'bin/nanoc_c','--target','c',source,'-o',output])
        code=output.read_text();self.assertNotIn('({',code)
        return source,output,code
    def compile(self,source,output,standard='c99',optimization='-O2'):
        self.run_cmd([os.environ.get('CC','cc'),'-std='+standard,'-pedantic-errors',
                      '-Werror=implicit-function-declaration','-Werror=return-type',
                      '-Werror=unused-local-typedefs',optimization,'-fsanitize=address,undefined',
                      '-fno-sanitize-recover=all',source,'-lm','-o',output])
    def test_order_nested_slots_scopes_globals_and_loops(self):
        _,output,code=self.emit(ORDER)
        self.assertIn('string_equal(',code)
        self.assertNotIn('static int nano_cb_0_string_equal(const',code)
        self.assertNotIn('const char *nano_cb_1_sl[',code)
        for standard in ('c99','c11'):
            for optimization in ('-O0','-O2'):
                exe=self.work/(standard+optimization);self.compile(output,exe,standard,optimization);self.run_cmd([exe])
    def test_distinct_allocations_fields_and_existing_source_routes(self):
        source,output,_=self.emit(VALUES)
        exe=self.work/'public';self.compile(output,exe);self.run_cmd([exe])
        self.run_cmd([ROOT/'bin/nano',source])
        module=self.work/'ordinary.nvm';self.run_cmd([ROOT/'bin/nano_virt',source,'--emit-nvm','-o',module])
        self.run_cmd([ROOT/'bin/nano_vm','--verify-only',module]);self.run_cmd([ROOT/'bin/nano_vm',module])
        generated=self.work/'native.c';self.run_cmd([ROOT/'bin/nvm2c',module,'-o',generated])
        exe=self.work/'native';self.compile(generated,exe);self.run_cmd([exe])
    def test_private_helper_pointer_and_null_boundary(self):
        _,output,code=self.emit('fn main()->int{return 0} shadow main{assert (== (main) 0)}')
        prefix=re.search(r'static int (nano_cb_\d+_)string_equal',code).group(1)
        harness=self.work/'helper.c';harness.write_text('#include <assert.h>\n#define main generated_main\n#include "'+str(output)+'"\n#undef main\nint main(void){char a[]={\'q\',0},b[]={\'q\',0};assert('+prefix+'string_equal(a,b));assert('+prefix+'string_equal(a,a));assert(!'+prefix+'string_equal(a,"r"));assert('+prefix+'string_equal(NULL,NULL));assert(!'+prefix+'string_equal(NULL,a));assert(!'+prefix+'string_equal(a,NULL));return 0;}\n')
        exe=self.work/'helper';self.compile(harness,exe);self.run_cmd([exe])
    def test_exact_refusal_prior_output_and_same_process_recovery(self):
        exe=self.work/'api';self.run_cmd([os.environ.get('CC','cc'),'-std=c99','-D_POSIX_C_SOURCE=200809L',
            '-Wall','-Wextra','-Werror','-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all',
            '-I',ROOT/'src',ROOT/'tests/test_public_c_string_api.c','-o',exe]);self.run_cmd([exe,self.work/'previous.c'])

ORDER='''let mut trace:int=0
fn step(id:int,value:string)->string{set trace (+ (* trace 10) id) return value}
shadow step{let saved:int=trace assert (== (step 1 "elm") "elm") set trace saved}
fn choose(test:bool,yes:string,no:string)->string{if test{return yes} else{return no}}
shadow choose{assert (== (choose true "elm" "ash") "elm")}
fn nano_cb_0_string_equal(x:int)->int{return x}
shadow nano_cb_0_string_equal{assert (== (nano_cb_0_string_equal 2) 2)}
let initialized:bool=(== (step 1 "seed") (step 2 "seed"))
fn main()->int{
 let nano_cb_1_sl:string="leaf"
 assert initialized assert (== trace 12)
 set trace 0
 assert (== (choose (== (step 1 "nested") (step 2 "nested")) nano_cb_1_sl "other") (step 3 "leaf"))
 assert (== trace 123)
 set trace 0
 assert (!= (step 4 "left") (step 5 "right"))
 assert (== trace 45)
 set trace 0
 if false {assert (== (step 9 "unused") "unused")} else {assert (== (step 6 "branch") "branch")}
 assert (== trace 6)
 let mut i:int=0
 while (< i 3){set trace 0 assert (== (step 7 "loop") (step 8 "loop")) assert (== trace 78) set i (+ i 1)}
 assert (== (== "same" "same") (!= "same" "different"))
 return 0
}
shadow main{assert true}
'''
VALUES='''struct Label { value:string }
fn convert(value:float)->string{return (float_to_string value)}
shadow convert{assert (== (convert 42.25) "42.25")}
fn main()->int{
 let first:string=(convert 42.25)
 let second:string=(convert 42.25)
 let alias:string=first
 let record:Label=Label {value:second}
 assert (== first second) assert (== alias first) assert (== record.value "42.25")
 assert (!= first "42.5") assert (== "" "") assert (!= first "")
 return 0
}
shadow main{assert (== (main) 0)}
'''
if __name__=='__main__':unittest.main()
