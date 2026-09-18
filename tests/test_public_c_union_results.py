"""I preserve exact declared union result identity and once evaluation."""
import os
from pathlib import Path
import unittest
from tests import test_public_c_nonfinite_format as formatting
ROOT=Path(__file__).resolve().parents[1]
class UnionResults(unittest.TestCase):
    setUp=formatting.PublicCFormat.setUp
    run_cmd=formatting.PublicCFormat.run_cmd
    emit=formatting.PublicCFormat.emit
    compile=formatting.PublicCFormat.compile
    def qualify(self, source, expected):
        output,code=self.emit(source)
        for standard in ('c99','c11'):
            for opt in ('-O0','-O2'):
                exe=self.work/(standard+opt);self.compile(output,exe,standard,opt)
                self.assertEqual(self.run_cmd([exe]).stdout,expected)
        path=self.work/'ordinary.nano'
        self.assertEqual(self.run_cmd([ROOT/'bin/nano',path]).stdout,expected)
        module=self.work/'ordinary.nvm'
        self.run_cmd([ROOT/'bin/nano_virt',path,'--emit-nvm','-o',module])
        self.run_cmd([ROOT/'bin/nano_vm','--verify-only',module])
        self.assertEqual(self.run_cmd([ROOT/'bin/nano_vm',module]).stdout,expected)
        return code
    def test_scalar_payload_alias_parameter_local_and_forward_returns(self):
        code=self.qualify(SCALARS,'payload\nempty\n')
        self.assertIn('NanoUnion_Result relay(',code)
        self.assertIn('NanoUnion_Result make(',code)
    def test_empty_union_and_distinct_nominal_results(self):
        self.qualify(EMPTY,'yes\nother\n')
    def test_api_nominal_refusal_output_and_recovery(self):
        exe=self.work/'api'
        self.run_cmd([os.environ.get('CC','cc'),'-std=c99','-D_POSIX_C_SOURCE=200809L',
            '-Wall','-Wextra','-Werror','-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all',
            '-I',ROOT/'src',ROOT/'tests/test_public_c_union_results_api.c','-o',exe])
        output=self.work/'previous.c';self.run_cmd([exe,output])
        binary=self.work/'recovered';self.compile(output,binary);self.run_cmd([binary])
SCALARS='''union Result{Data{n:int,flag:bool,real:float,text:string},Empty{}}
let mut calls:int=0
fn relay(value:Result)->Result{let alias:Result=value return (identity alias)}
shadow relay{let value:Result=(relay Result.Empty{}) match value{Empty(_)=>{assert true} Data(p)=>{assert false}}}
fn identity(value:Result)->Result{return value}
shadow identity{let value:Result=(identity Result.Empty{}) match value{Empty(_)=>{assert true} Data(p)=>{assert false}}}
fn make(n:int)->Result{set calls (+ calls 1) return Result.Data{n:n,flag:true,real:(float_from_bits -9223372036854775808),text:"payload"}}
shadow make{let value:Result=(make 1) match value{Data(p)=>{assert (== p.n 1)} Empty(_)=>{assert false}}}
fn vacant()->Result{return Result.Empty{}}
shadow vacant{let value:Result=(vacant) match value{Empty(_)=>{assert true} Data(p)=>{assert false}}}
fn main()->int{
 let before:int=calls
 let value:Result=(relay (make 17))
 match value{Data(p) if p.flag=>{assert (== p.n 17) assert (== (float_to_bits p.real) -9223372036854775808) assert (== p.text "payload") (println p.text)} _=>{assert false}}
 assert (== calls (+ before 1))
 match (vacant){Empty(_)=>{(println "empty")} Data(p)=>{assert false}}
 return 0
}
shadow main{assert true}
'''
EMPTY='''union Flag{Yes{},No{}}
union Other{Yes{},No{}}
fn nano_cb_0_entry()->Flag{return Flag.Yes{}}
shadow nano_cb_0_entry{let value:Flag=(nano_cb_0_entry) match value{Yes(_)=>{assert true} No(_)=>{assert false}}}
fn other()->Other{return Other.Yes{}}
shadow other{let value:Other=(other) match value{Yes(_)=>{assert true} No(_)=>{assert false}}}
fn main()->int{
 let first:Flag=(nano_cb_0_entry)
 match first{Yes(_)=>{(println "yes")} No(_)=>{assert false}}
 let second:Other=(other)
 match second{Yes(_)=>{(println "other")} No(_)=>{assert false}}
 return 0
}
shadow main{assert true}
'''
if __name__=='__main__':unittest.main()
