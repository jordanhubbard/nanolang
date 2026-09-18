"""I qualify named standard-C union payload copies and exact match identity."""
import os
from pathlib import Path
import re
import unittest
from tests import test_public_c_nonfinite_format as formatting
ROOT=Path(__file__).resolve().parents[1]
class PublicCMatch(unittest.TestCase):
    setUp=formatting.PublicCFormat.setUp
    run_cmd=formatting.PublicCFormat.run_cmd
    emit=formatting.PublicCFormat.emit
    compile=formatting.PublicCFormat.compile
    def qualify(self,source,expected):
        output,code=self.emit(source)
        self.assertNotIn('__typeof__',code)
        self.assertNotIn('static char _ibuf',code)
        for standard in ('c99','c11'):
            for optimization in ('-O0','-O2'):
                exe=self.work/(standard+optimization)
                self.compile(output,exe,standard,optimization)
                self.assertEqual(self.run_cmd([exe]).stdout,expected)
        path=self.work/'ordinary.nano'
        self.assertEqual(self.run_cmd([ROOT/'bin/nano',path]).stdout,expected)
        module=self.work/'ordinary.nvm'
        self.run_cmd([ROOT/'bin/nano_virt',path,'--emit-nvm','-o',module])
        self.run_cmd([ROOT/'bin/nano_vm','--verify-only',module])
        self.assertEqual(self.run_cmd([ROOT/'bin/nano_vm',module]).stdout,expected)
        return code
    def test_scalar_payloads_nested_scopes_once_and_hygiene(self):
        code=self.qualify(SCALARS,'7\n-1\n9\n')
        self.assertNotIn('} nano_cb_0_payload_',code)
        self.assertRegex(code,r'} nano_cb_[1-9][0-9]*_payload_[0-9]+_[0-9]+;')
    def test_empty_discard_and_same_spelled_variants(self):
        self.qualify(EMPTY,'1\n2\n11\n22\n')
    def test_existing_match_programs_strict_c99_c11(self):
        for name in ('05_match','06_single_field_variant'):
            with self.subTest(name=name):
                self.qualify((ROOT/'tests/cross-backend'/f'{name}.nano').read_text(),
                             (ROOT/'tests/cross-backend'/f'{name}.expected').read_text())
    def test_exact_identity_refusal_and_publication_recovery(self):
        exe=self.work/'api'
        self.run_cmd([os.environ.get('CC','cc'),'-std=c99','-D_POSIX_C_SOURCE=200809L',
            '-Wall','-Wextra','-Werror','-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all',
            '-I',ROOT/'src',ROOT/'tests/test_public_c_match_api.c','-o',exe])
        self.run_cmd([exe,self.work/'previous.c'])
SCALARS='''union Value { Data { number:int,text:string,flag:bool,real:float }, Empty {} }
let mut calls:int=0
fn mark(n:int)->int{set calls (+ calls 1) return n}
shadow mark{assert (== (mark 5) 5)}
fn nano_cb_0_payload_0_0(n:int)->int{return n}
shadow nano_cb_0_payload_0_0{assert (== (nano_cb_0_payload_0_0 2) 2)}
fn read(_match_val:Value)->int{
 let outer:int=90
 match _match_val {
  Data(outer)=>{
   assert outer.flag assert (== outer.text "yes")
   assert (== (str_length outer.text) 3)
   assert (== (float_to_bits outer.real) 4609434218613702656)
   let inner:Value=Value.Data{number:2,text:"yes",flag:true,real:1.5}
   match inner {
    Data(p)=>{assert (== p.number 2) assert (== outer.number 7)}
    Empty(_)=>{assert false}
   }
   return outer.number
  }
  Empty(_)=>{assert (== outer 90) return -1}
 }
 return 0
}
shadow read{assert (== (read Value.Empty{}) -1)}
fn main()->int{
 let before:int=calls
 let _match_val:Value=Value.Data{number:(mark 7),text:"yes",flag:true,real:1.5}
 (println (int_to_string (read _match_val)))
 assert (== calls (+ before 1))
 (println (int_to_string (read Value.Empty{})))
 let _:int=9
 match _match_val { Data(_)=>{assert (== _ 9)} Empty(_)=>{assert false} }
 (println (int_to_string _))
 assert (== (nano_cb_0_payload_0_0 4) 4)
 return 0
}
shadow main{assert true}
'''
EMPTY='''union Flag{Yes{},No{}}
union First{Item{value:int}}
union Second{Item{value:int}}
fn flag(value:Flag)->int{
 match value{Yes(_)=>{return 1} No(_)=>{return 2}}
 return 0
}
shadow flag{assert (== (flag Flag.Yes{}) 1) assert (== (flag Flag.No{}) 2)}
fn first(value:First)->int{match value{Item(p)=>{return p.value}} return 0}
shadow first{assert (== (first First.Item{value:3}) 3)}
fn second(value:Second)->int{match value{Item(p)=>{return p.value}} return 0}
shadow second{assert (== (second Second.Item{value:4}) 4)}
fn main()->int{
 (println (int_to_string (flag Flag.Yes{})))
 (println (int_to_string (flag Flag.No{})))
 (println (int_to_string (first First.Item{value:11})))
 (println (int_to_string (second Second.Item{value:22})))
 return 0
}
shadow main{assert true}
'''
if __name__=='__main__':unittest.main()
