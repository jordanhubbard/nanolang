"""I preserve private boolean text names, exact dispatch and stable literal results."""
import os
from pathlib import Path
import unittest
from tests import test_public_c_nonfinite_format as base
ROOT=Path(__file__).resolve().parents[1]
class BoolText(unittest.TestCase):
    setUp=base.PublicCFormat.setUp
    run_cmd=base.PublicCFormat.run_cmd
    compile=base.PublicCFormat.compile
    emit=base.PublicCFormat.emit
    def test_api_builtin_declarations_refusals_and_recovery(self):
        api=self.work/'api'
        self.run_cmd([os.environ.get('CC','cc'),'-std=c99','-D_POSIX_C_SOURCE=200809L',
            '-Wall','-Wextra','-Werror','-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all',
            '-I',ROOT/'src',ROOT/'tests/test_public_c_bool_text_api.c','-o',api])
        for mode in ('builtin','direct','signature','qualified'):
            output=self.work/(mode+'.c');self.run_cmd([api,output,mode])
            for standard in ('c99','c11'):
                for optimization in ('-O0','-O2'):
                    exe=self.work/(mode+standard+optimization);self.compile(output,exe,standard,optimization);self.run_cmd([exe])
    def test_helper_names_once_effects_and_stable_aliases(self):
        source='''let mut calls:int=0
fn next_bool()->bool{set calls (+ calls 1) return (== calls 1)}
shadow next_bool{let before:int=calls let value:bool=(next_bool) assert (== calls (+ before 1))}
fn nano_bool_to_string(x:int)->int{return (+ x 1)}
shadow nano_bool_to_string{assert (== (nano_bool_to_string 3) 4)}
fn nano_rt_bool_text(x:int)->int{return (+ x 2)}
shadow nano_rt_bool_text{assert (== (nano_rt_bool_text 3) 5)}
fn nano_cb_0_bool_text(x:int)->int{return (+ x 3)}
shadow nano_cb_0_bool_text{assert (== (nano_cb_0_bool_text 3) 6)}
fn main()->int{
let first:string=(bool_to_string (next_bool))
let second:string=(bool_to_string (next_bool))
assert (== calls 2) assert (== first "true") assert (== second "false")
assert (== (nano_bool_to_string 3) 4) assert (== (nano_rt_bool_text 3) 5) assert (== (nano_cb_0_bool_text 3) 6)
(println first) (println second) (println (bool_to_string true)) return 0}
shadow main{assert true}
'''
        output,_=self.emit(source)
        self.assertIn('static const char* nano_cb_1_bool_text(',output.read_text())
        for standard in ('c99','c11'):
            for optimization in ('-O0','-O2'):
                exe=self.work/(standard+optimization);self.compile(output,exe,standard,optimization);self.assertEqual(self.run_cmd([exe]).stdout,'true\nfalse\ntrue\n')
    def test_source_wrong_types_refuse_before_output(self):
        for index,argument in enumerate(('1','1.0','"true"')):
            path=self.work/f'wrong{index}.nano';path.write_text(f'fn main()->int{{let text:string=(bool_to_string {argument}) return 0}}\nshadow main{{assert true}}\n')
            output=self.work/'previous.c';output.write_text('previous')
            result=self.run_cmd([ROOT/'bin/nanoc_c','--target','c',path,'-o',output],expected=1)
            self.assertIn('[c_backend] I require one exact BOOL operand for C bool_to_string.',result.stderr)
            self.assertEqual(output.read_text(),'previous')
if __name__=='__main__':unittest.main()
