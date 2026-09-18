"""I qualify explicit hosted/library options and checked unsupported profiles."""
import os
from pathlib import Path
import unittest
from tests import test_public_c_nonfinite_format as base
ROOT=Path(__file__).resolve().parents[1]
class ProfileOptions(unittest.TestCase):
    setUp=base.PublicCFormat.setUp
    run_cmd=base.PublicCFormat.run_cmd
    compile=base.PublicCFormat.compile
    def test_options_library_link_and_same_process_publication_recovery(self):
        api=self.work/'api'
        self.run_cmd([os.environ.get('CC','cc'),'-std=c99','-D_POSIX_C_SOURCE=200809L',
            '-Wall','-Wextra','-Werror','-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all',
            '-I',ROOT/'src',ROOT/'tests/test_public_c_profile_options_api.c','-o',api])
        for mode in range(5):
            output=self.work/f'mode{mode}.c';self.run_cmd([api,output,str(mode)])
            text=output.read_text();self.assertNotIn('({',text);self.assertNotIn('setjmp',text)
            self.assertEqual('int main(void)' in text,mode<3)
            if mode>=3:
                with output.open('a') as f:f.write('\nint main(void){return probe()==42?0:1;}\n')
            for standard in ('c99','c11'):
                for optimization in ('-O0','-O2'):
                    exe=self.work/f'mode{mode}{standard}{optimization}'
                    self.compile(output,exe,standard,optimization)
                    self.run_cmd([exe],expected=42 if mode<3 else 0)
    def test_ordinary_anonymous_source_refuses_and_preserves_output(self):
        for capture in (False,True):
            source='fn main()->int{let n:int=7 let f:fn(int)->int=fn(x:int)->int{return (+ x '+('n' if capture else '7')+')} return (f 1)}\nshadow main{assert true}\n'
            path=self.work/'ordinary.nano';path.write_text(source)
            output=self.work/'previous.c';output.write_text('previous')
            result=self.run_cmd([ROOT/'bin/nanoc_c','--target','c',path,'-o',output],expected=1)
            self.assertIn('anonymous or captured callable C lowering',result.stderr)
            self.assertEqual(output.read_text(),'previous')
if __name__=='__main__':unittest.main()
