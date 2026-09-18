"""I refuse spread and retain exact explicit-field C records."""
import os
from pathlib import Path
import unittest
from tests import test_public_c_nonfinite_format as base
ROOT=Path(__file__).resolve().parents[1]
class RecordSpread(unittest.TestCase):
    setUp=base.PublicCFormat.setUp
    run_cmd=base.PublicCFormat.run_cmd
    compile=base.PublicCFormat.compile
    emit=base.PublicCFormat.emit
    def test_api_plain_nested_discarded_lifted_refusal_and_recovery(self):
        api=self.work/'api'
        self.run_cmd([os.environ.get('CC','cc'),'-std=c99','-D_POSIX_C_SOURCE=200809L',
            '-Wall','-Wextra','-Werror','-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all',
            '-I',ROOT/'src',ROOT/'tests/test_public_c_record_spread_api.c','-o',api])
        output=self.work/'recovered.c';self.run_cmd([api,output])
        for standard in ('c99','c11'):
            for optimization in ('-O0','-O2'):
                exe=self.work/(standard+optimization);self.compile(output,exe,standard,optimization);self.run_cmd([exe])
    def test_annotated_source_spread_refuses_without_publication(self):
        source='''struct Record { value:int, text:string }
fn main()->int{let base:Record=Record{value:7,text:"kept"}
let copy:Record={..base,value:9} (println copy.text) return 0}
shadow main{assert true}
'''
        path=self.work/'spread.nano';path.write_text(source)
        output=self.work/'previous.c';output.write_text('previous')
        result=self.run_cmd([ROOT/'bin/nanoc_c','--target','c',path,'-o',output],expected=1)
        self.assertIn('[c_backend] I do not provide record spread lowering in this C profile.',result.stderr)
        self.assertEqual(output.read_text(),'previous')
    def test_explicit_fields_remain_exact(self):
        source='''struct Record { value:int, text:string }
fn main()->int{let base:Record=Record{value:7,text:"kept"}
let copy:Record=Record{value:9,text:base.text}
(println (int_to_string base.value)) (println (int_to_string copy.value))
(println copy.text) return 0}
shadow main{assert true}
'''
        output,_=self.emit(source)
        for standard in ('c99','c11'):
            for optimization in ('-O0','-O2'):
                exe=self.work/(standard+optimization);self.compile(output,exe,standard,optimization)
                self.assertEqual(self.run_cmd([exe]).stdout,'7\n9\nkept\n')
if __name__=='__main__':unittest.main()
