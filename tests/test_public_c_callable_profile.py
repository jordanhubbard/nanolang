"""I retain direct signatures and refuse callable storage or indirect execution."""
import os
from pathlib import Path
import unittest
from tests import test_public_c_nonfinite_format as base
ROOT=Path(__file__).resolve().parents[1]
class CallableProfile(unittest.TestCase):
    setUp=base.PublicCFormat.setUp
    run_cmd=base.PublicCFormat.run_cmd
    compile=base.PublicCFormat.compile
    emit=base.PublicCFormat.emit
    def test_api_callable_boundaries_qualified_identity_and_recovery(self):
        api=self.work/'api'
        self.run_cmd([os.environ.get('CC','cc'),'-std=c99','-D_POSIX_C_SOURCE=200809L',
            '-Wall','-Wextra','-Werror','-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all',
            '-I',ROOT/'src',ROOT/'tests/test_public_c_callable_profile_api.c','-o',api])
        for mode in ('direct','qualified','scalar'):
            output=self.work/(mode+'.c');self.run_cmd([api,output,mode])
            for standard in ('c99','c11'):
                for optimization in ('-O0','-O2'):
                    exe=self.work/(mode+standard+optimization);self.compile(output,exe,standard,optimization);self.run_cmd([exe])
    def test_ordinary_named_callable_source_refusals(self):
        prefix='fn increment(x:int)->int{return (+ x 1)} shadow increment{assert (== (increment 2) 3)}\n'
        cases=[
          'fn main()->int{let callback:fn(int)->int=increment return (callback 2)}',
          'let callback:fn(int)->int=increment fn main()->int{return 0}',
          'fn apply(callback:fn(int)->int,x:int)->int{return (callback x)} shadow apply{assert true} fn main()->int{return 0}',
          'fn choose()->fn(int)->int{return increment} shadow choose{assert true} fn main()->int{return 0}',
        ]
        for index,source in enumerate(cases):
            path=self.work/f'ordinary{index}.nano';path.write_text(prefix+source+'\nshadow main{assert true}\n')
            output=self.work/'previous.c';output.write_text('previous')
            result=self.run_cmd([ROOT/'bin/nanoc_c','--target','c',path,'-o',output],expected=1)
            self.assertIn('first-class callable values or indirect calls',result.stderr)
            self.assertEqual(output.read_text(),'previous')
    def test_direct_calls_recursion_and_lexical_scalar_shadow(self):
        source='''fn count(n:int)->int{if (== n 0){return 0} return (+ 1 (count (- n 1)))}
shadow count{assert (== (count 3) 3)}
fn label()->string{return "direct"}
shadow label{assert (== (label) "direct")}
fn main()->int{assert (== (count 4) 4) (println (label)) let count:int=7 assert (== count 7) return 0}
shadow main{assert true}
'''
        output,_=self.emit(source)
        for standard in ('c99','c11'):
            for optimization in ('-O0','-O2'):
                exe=self.work/(standard+optimization);self.compile(output,exe,standard,optimization);self.assertEqual(self.run_cmd([exe]).stdout,'direct\n')
if __name__=='__main__':unittest.main()
