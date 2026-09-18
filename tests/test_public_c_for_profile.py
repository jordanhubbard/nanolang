"""I refuse unqualified FOR profiles and retain ordinary WHILE/block behavior."""
import os
from pathlib import Path
import unittest
from tests import test_public_c_nonfinite_format as base
ROOT=Path(__file__).resolve().parents[1]
class ForProfile(unittest.TestCase):
    setUp=base.PublicCFormat.setUp
    run_cmd=base.PublicCFormat.run_cmd
    compile=base.PublicCFormat.compile
    emit=base.PublicCFormat.emit
    def test_api_refusal_preserves_bindings_output_and_recovery(self):
        api=self.work/'api'
        self.run_cmd([os.environ.get('CC','cc'),'-std=c99','-D_POSIX_C_SOURCE=200809L',
            '-Wall','-Wextra','-Werror','-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all',
            '-I',ROOT/'src',ROOT/'tests/test_public_c_for_profile_api.c','-o',api])
        output=self.work/'recovered.c';self.run_cmd([api,output])
        for standard in ('c99','c11'):
            for optimization in ('-O0','-O2'):
                exe=self.work/(standard+optimization);self.compile(output,exe,standard,optimization);self.run_cmd([exe])
    def test_canonical_range_and_unqualified_bare_count_refuse_in_backend(self):
        # Bare count is a refused non-normative input, never an execution oracle.
        for name,iterable in [('canonical_range','(range 0 3)'),('unqualified_count','3')]:
            source=f'fn main()->int{{for value in {iterable}{{(println value)}} return 0}}\nshadow main{{assert true}}\n'
            path=self.work/(name+'.nano');path.write_text(source)
            output=self.work/'previous.c';output.write_text('previous')
            result=self.run_cmd([ROOT/'bin/nanoc_c','--target','c',path,'-o',output],expected=1)
            self.assertIn('[c_backend] I do not provide for-loop lowering in this C profile.',result.stderr)
            self.assertEqual(output.read_text(),'previous')
    def test_while_block_scope_and_real_break_continue_remain_supported(self):
        source='''fn main()->int{
let value:string="outer"
(println value)
let mut count:int=0
while (< count 4){set count (+ count 1) let value:int=count
if (== value 2){continue}
if true {let value:string="inner" (println value)}
if (== value 3){break}}
(println value)
(println (int_to_string count))
let number:int=7
if true {let number:int=9 (println (int_to_string number))}
(println (int_to_string number))
return 0}
shadow main{assert true}
'''
        output,_=self.emit(source)
        for standard in ('c99','c11'):
            for optimization in ('-O0','-O2'):
                exe=self.work/(standard+optimization);self.compile(output,exe,standard,optimization)
                self.assertEqual(self.run_cmd([exe]).stdout,'outer\ninner\ninner\nouter\n3\n9\n7\n')
if __name__=='__main__':unittest.main()
