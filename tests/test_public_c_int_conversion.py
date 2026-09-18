"""I qualify exact integer conversion, stable mixed snapshots and refusals."""
import os
from pathlib import Path
import re
import unittest
from tests import test_public_c_nonfinite_format as formatting

ROOT=Path(__file__).resolve().parents[1]

class PublicCInt(unittest.TestCase):
    setUp=formatting.PublicCFormat.setUp
    run_cmd=formatting.PublicCFormat.run_cmd
    emit=formatting.PublicCFormat.emit
    compile=formatting.PublicCFormat.compile

    def test_endpoints_once_evaluation_and_private_names(self):
        output,code=self.emit(ENDPOINTS)
        self.assertNotIn('static char _ibuf',code)
        self.assertNotIn('static const char *nano_cb_0_int_text_new(int64_t',code)
        expected='-9223372036854775808\n9223372036854775807\n-1\n0\n1\n'
        for standard in ('c99','c11'):
            for optimization in ('-O0','-O2'):
                exe=self.work/(standard+optimization)
                self.compile(output,exe,standard,optimization)
                self.assertEqual(self.run_cmd([exe]).stdout,expected)

    def test_mixed_aliases_globals_returns_and_loops(self):
        output,_=self.emit(ALIASES)
        for standard in ('c99','c11'):
            for optimization in ('-O0','-O2'):
                exe=self.work/(standard+optimization)
                self.compile(output,exe,standard,optimization)
                self.assertEqual(self.run_cmd([exe]).stdout,'-10\n20\n-nan\n30\n1.25\n-10\n')

    def test_shared_cleanup_and_controlled_refusals(self):
        output,code=self.emit('fn main()->int{return 0} shadow main{assert (== (main) 0)}')
        prefix=re.search(r'static const char \*(nano_cb_\d+_)int_text_new',code).group(1)
        harness=self.work/'ownership.c'
        harness.write_text(FAILURES.replace('GENERATED',str(output)).replace('PREFIX',prefix))
        exe=self.work/'ownership';self.compile(harness,exe)
        self.run_cmd([exe,'normal'])
        for mode,diagnostic in [('allocation','I could not allocate my C scalar result.'),
                                ('registration','I could not register my C scalar cleanup.')]:
            result=self.run_cmd([exe,mode],expected=1)
            self.assertIn(diagnostic,result.stderr)

    def test_binding_types_publication_and_recovery(self):
        exe=self.work/'api'
        self.run_cmd([os.environ.get('CC','cc'),'-std=c99','-D_POSIX_C_SOURCE=200809L',
                      '-Wall','-Wextra','-Werror','-O1','-fsanitize=address,undefined',
                      '-fno-sanitize-recover=all','-I',ROOT/'src',ROOT/'tests/test_public_c_int_api.c','-o',exe])
        self.run_cmd([exe,self.work/'previous.c'])

ENDPOINTS='''let mut calls:int=0
fn operand(value:int)->int{set calls (+ calls 1) return value}
shadow operand{assert (== (operand 4) 4)}
fn nano_cb_0_int_text_new(value:int)->int{return value}
shadow nano_cb_0_int_text_new{assert (== (nano_cb_0_int_text_new 3) 3)}
fn show(value:int)->int{
 let before:int=calls
 let text:string=(int_to_string (operand value))
 assert (== calls (+ before 1))
 (println text)
 return 0
}
shadow show{assert (== (show 0) 0)}
fn main()->int{
 assert (== (nano_cb_0_int_text_new 8) 8)
 assert (== (show -9223372036854775808) 0)
 assert (== (show 9223372036854775807) 0)
 assert (== (show -1) 0)
 assert (== (show 0) 0)
 assert (== (show 1) 0)
 return 0
}
shadow main{assert (== (main) 0)}
'''
ALIASES='''fn text(value:int)->string{return (int_to_string value)}
shadow text{assert (== (text 7) "7")}
let first:string=(text -10)
let second:string=(text 20)
let initial_float:string=(float_to_string (float_from_bits -2251799813685247))
fn main()->int{
 let held:string=(text 30)
 let held_float:string=(float_to_string 1.25)
 let alias:string=first
 let mut i:int=0
 while (< i 300){
  let fresh:string=(text i)
  let fresh_float:string=(float_to_string 2.0)
  assert (!= fresh "") assert (== fresh_float "2")
  set i (+ i 1)
 }
 assert (== first "-10") assert (== second "20")
 assert (== held "30") assert (== held_float "1.25")
 (println first) (println second) (println initial_float)
 (println held) (println held_float) (println alias)
 return 0
}
shadow main{assert (== (main) 0)}
'''
FAILURES=formatting.FAILURE_HARNESS.replace('PREFIXfloat_text_new(1.0)','PREFIXint_text_new(INT64_MIN)').replace('PREFIXfloat_text_new(2.0)','PREFIXint_text_new(INT64_MAX)').replace('strcmp(a,"1")==0&&strcmp(b,"2")==0','strcmp(a,"-9223372036854775808")==0&&strcmp(b,"9223372036854775807")==0')
# I retain a mixed float owner before an integer allocation failure.
FAILURES=FAILURES.replace('assert(atexit(verify_exit)==0);PREFIXint_text_new(INT64_MIN);fail_allocate=1;', 'assert(atexit(verify_exit)==0);PREFIXfloat_text_new(1.0);fail_allocate=1;')
if __name__=='__main__':unittest.main()
