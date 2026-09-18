"""I qualify public C nonfinite bytes, stable snapshots and checked cleanup."""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
CASES=[(0x7ff8000000000001,'nan'),(0xfff8000000000001,'-nan'),
       (0x7ff0000000000001,'nan'),(0xfff0000000000001,'-nan'),
       (0x7ff0000000000000,'inf'),(0xfff0000000000000,'-inf'),
       (0,'0'),(1<<63,'-0'),(0x7fefffffffffffff,'1.79769e+308'),
       (0xffefffffffffffff,'-1.79769e+308'),(1,'4.94066e-324'),
       (0x3ff4000000000000,'1.25'),(0x400921fb54442d18,'3.14159')]
def signed(bits):return bits if bits<1<<63 else bits-(1<<64)

class PublicCFormat(unittest.TestCase):
    def setUp(self):
        self.work=Path(tempfile.mkdtemp(prefix='nano-public-c-format-'))
        print('I retain public C formatting evidence at',self.work,flush=True)
    def run_cmd(self,args,expected=0):
        result=subprocess.run(list(map(str,args)),cwd=ROOT,capture_output=True,text=True,timeout=240)
        self.assertEqual(result.returncode,expected,f'command={args!r}\n{result.stdout}\n{result.stderr}')
        self.assertNotIn('Sanitizer',result.stderr)
        self.assertNotIn('runtime error:',result.stderr)
        return result
    def emit(self,source):
        path=self.work/'ordinary.nano';path.write_text(source)
        output=self.work/'ordinary.c'
        self.run_cmd([ROOT/'bin/nanoc_c','--target','c',path,'-o',output])
        text=output.read_text()
        self.assertNotIn('({',text)
        self.assertNotIn('__auto_type',text)
        return output,text
    def compile(self,source,exe,standard='c99',optimization='-O2'):
        self.run_cmd([os.environ.get('CC','cc'),'-std='+standard,'-pedantic-errors',
                      '-Werror=implicit-function-declaration','-Werror=return-type',
                      '-Werror=unused-local-typedefs',optimization,'-fsanitize=address,undefined',
                      '-fno-sanitize-recover=all',source,'-lm','-o',exe])
    def test_exact_output_input_bits_and_once_evaluation(self):
        calls='\n'.join(f'assert (== (show {signed(bits)}) 0)' for bits,text in CASES)
        output,code=self.emit(SOURCE.replace('CASE_CALLS',calls))
        expected=''.join(text+'\n'+text+'|'+text+'\n' for _,text in CASES)
        self.assertNotIn('nano_cb_0_float_text_new(double',code)
        for standard in ('c99','c11'):
            for optimization in ('-O0','-O2'):
                exe=self.work/(standard+optimization)
                self.compile(output,exe,standard,optimization)
                self.assertEqual(self.run_cmd([exe]).stdout,expected)
    def test_aliases_globals_returns_and_loop_snapshots(self):
        output,_=self.emit(ALIASES)
        for standard in ('c99','c11'):
            exe=self.work/standard;self.compile(output,exe,standard)
            self.assertEqual(self.run_cmd([exe]).stdout,'1\n2\n'+'3\n'*300+'-nan\n-nan\n1\n')
    def test_generated_cleanup_and_allocation_registration_refusals(self):
        output,code=self.emit('fn main()->int{return 0} shadow main{assert (== (main) 0)}')
        prefix=re.search(r'static const char \*(nano_cb_\d+_)float_text_new',code).group(1)
        harness=self.work/'ownership.c'
        harness.write_text(FAILURE_HARNESS.replace('GENERATED',str(output)).replace('PREFIX',prefix))
        exe=self.work/'ownership';self.compile(harness,exe)
        self.run_cmd([exe,'normal'])
        for mode,diagnostic in [('allocation','I could not allocate my C float result.'),
                                ('registration','I could not register my C float cleanup.')]:
            result=self.run_cmd([exe,mode],expected=1)
            self.assertIn(diagnostic,result.stderr)
    def test_scoped_builtin_resolution_and_previous_output(self):
        exe=self.work/'api'
        self.run_cmd([os.environ.get('CC','cc'),'-std=c99','-D_POSIX_C_SOURCE=200809L',
                      '-Wall','-Wextra','-Werror','-O1','-fsanitize=address,undefined',
                      '-fno-sanitize-recover=all','-I',ROOT/'src',ROOT/'tests/test_public_c_format_api.c','-o',exe])
        self.run_cmd([exe,self.work/'previous.c'])

SOURCE='''let mut count:int=0
fn operand(bits:int)->float{set count (+ count 1) return (float_from_bits bits)}
shadow operand{assert (== (float_to_bits (operand 0)) 0)}
fn nano_cb_0_float_text_new(x:int)->int{return x}
shadow nano_cb_0_float_text_new{assert (== (nano_cb_0_float_text_new 1) 1)}
fn show(bits:int)->int{
 let value:float=(float_from_bits bits)
 let before:int=count
 let text:string=(float_to_string (operand bits))
 assert (== count (+ before 1))
 (println text)
 (print (operand bits))
 (print "|")
 (println (operand bits))
 assert (== count (+ before 3))
 assert (== (float_to_bits value) bits)
 return 0
}
shadow show{assert (== (show 0) 0)}
fn main()->int{CASE_CALLS return 0}
shadow main{assert (== (main) 0)}
'''
ALIASES='''fn text(value:float)->string{return (float_to_string value)}
shadow text{assert (== (text 1.0) "1")}
let first:string=(text 1.0)
let second:string=(text 2.0)
fn main()->int{
 (println first) (println second)
 let negative:string=(text (float_from_bits -2251799813685247))
 let alias:string=negative
 let mut i:int=0
 while (< i 300){let temporary:string=(text 3.0) (println temporary) set i (+ i 1)}
 (println negative) (println alias) (println first)
 return 0
}
shadow main{assert (== (main) 0)}
'''
FAILURE_HARNESS=r'''#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
static int allocations, releases, registrations, fail_allocate, fail_register;
static void *fixture_malloc(size_t bytes){++allocations;return fail_allocate?NULL:malloc(bytes);}
static void fixture_free(void *p){if(p)++releases;free(p);}
static int fixture_atexit(void (*callback)(void)){++registrations;return fail_register?-1:atexit(callback);}
#define malloc fixture_malloc
#define free fixture_free
#define atexit fixture_atexit
#define main generated_main
#include "GENERATED"
#undef main
#undef atexit
#undef free
#undef malloc
static int registration_mode;
static void verify_exit(void){
 if(registration_mode){assert(allocations==1&&releases==1&&registrations==1);assert(PREFIXfloat_text_registered==0);}
 else {assert(allocations==2&&releases==1&&registrations==1);}
 assert(PREFIXfloat_text_head==NULL);
}
int main(int argc,char **argv){
 assert(argc==2);
 if(strcmp(argv[1],"registration")==0){registration_mode=1;fail_register=1;assert(atexit(verify_exit)==0);PREFIXfloat_text_new(1.0);return 9;}
 if(strcmp(argv[1],"allocation")==0){assert(atexit(verify_exit)==0);PREFIXfloat_text_new(1.0);fail_allocate=1;PREFIXfloat_text_new(2.0);return 9;}
 const char *a=PREFIXfloat_text_new(1.0);const char *b=PREFIXfloat_text_new(2.0);
 assert(strcmp(a,"1")==0&&strcmp(b,"2")==0);assert(allocations==2&&registrations==1);
 PREFIXfloat_text_cleanup();assert(releases==2&&PREFIXfloat_text_head==NULL);
 PREFIXfloat_text_cleanup();assert(releases==2);return 0;
}
'''
if __name__=='__main__':unittest.main()
