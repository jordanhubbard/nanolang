"""I preserve typed block/match values, ordering, laziness and enclosing exits."""
import os
from pathlib import Path
import unittest
from tests import test_public_c_nonfinite_format as base
ROOT=Path(__file__).resolve().parents[1]
class ExpressionValues(unittest.TestCase):
    setUp=base.PublicCFormat.setUp
    run_cmd=base.PublicCFormat.run_cmd
    emit=base.PublicCFormat.emit
    compile=base.PublicCFormat.compile
    def qualify(self,source,expected=''):
        output,code=self.emit(source)
        self.assertNotIn('__typeof__',code)
        for standard in ('c99','c11'):
            for optimization in ('-O0','-O2'):
                exe=self.work/(standard+optimization);self.compile(output,exe,standard,optimization)
                self.assertEqual(self.run_cmd([exe]).stdout,expected)
        path=self.work/'ordinary.nano'
        self.assertEqual(self.run_cmd([ROOT/'bin/nano',path]).stdout,expected)
        module=self.work/'ordinary.nvm';self.run_cmd([ROOT/'bin/nano_virt',path,'--emit-nvm','-o',module])
        self.run_cmd([ROOT/'bin/nano_vm','--verify-only',module])
        self.assertEqual(self.run_cmd([ROOT/'bin/nano_vm',module]).stdout,expected)
        return code
    def test_ordered_operands_calls_and_lazy_selected_values(self):self.qualify(ORDER)
    def test_scalar_union_results_nested_scope_and_function_return(self):self.qualify(VALUES,'selected\n')
    def test_loop_body_break_and_continue_preserve_outer_targets(self):self.qualify(LOOP)
    def test_nested_initializer_function_exit_has_no_later_consumer(self):self.qualify(EXIT)
    def test_api_typed_paths_refusal_recovery_and_completion(self):
        exe=self.work/'api'
        self.run_cmd([os.environ.get('CC','cc'),'-std=c99','-D_POSIX_C_SOURCE=200809L',
            '-Wall','-Wextra','-Werror','-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all',
            '-I',ROOT/'src',ROOT/'tests/test_public_c_expression_values_api.c','-o',exe])
        output=self.work/'previous.c';self.run_cmd([exe,output])
        binary=self.work/'recovered';self.compile(output,binary);self.run_cmd([binary])
PREFIX='union Choice{Some{number:int},None{}}\n'
ORDER=PREFIX+'''let mut trace:int=0
fn mark(n:int)->int{set trace (+ (* trace 10) n) return n}
shadow mark{set trace 0 assert (== (mark 2) 2) assert (== trace 2)}
fn pair(a:int,b:int)->int{return (+ (* a 10) b)}
shadow pair{assert (== (pair 2 3) 23)}
fn main()->int{
 let chosen:Choice=Choice.Some{number:5}
 set trace 0
 let value:int=(pair (mark 1) (match chosen{Some(p)=>{let ignored:int=(mark 2) (+ p.number (mark 3))} None(_)=>0}))
 assert (== value 18) assert (== trace 123)
 set trace 0
 let sum:int=(+ (mark 4) (match chosen{Some(p)=>(mark 5) None(_)=>(mark 9)}))
 assert (== sum 9) assert (== trace 45)
 set trace 0
 let no:bool=(and false (match chosen{Some(p)=>{let ignored:int=(mark 8) true} None(_)=>true}))
 let yes:bool=(or true (match chosen{Some(p)=>{let ignored:int=(mark 9) false} None(_)=>false}))
 assert (not no) assert yes assert (== trace 0)
 let selected:int=(cond ((== trace 0) (match chosen{Some(p)=>(mark 6) None(_)=>0})) (else (mark 9)))
 assert (== selected 6) assert (== trace 6)
 (pair (mark 7) (match chosen{Some(p)=>(mark 8) None(_)=>0}))
 assert (== trace 678)
 return 0
}
shadow main{assert true}
'''
VALUES=PREFIX+'''fn checked(value:Choice)->int{
 let result:int=match value{Some(p)=>{if (< p.number 0){return -7} let x:int=(* p.number 2) x} None(_)=>0}
 return (+ result 100)
}
shadow checked{assert (== (checked Choice.Some{number:3}) 106) assert (== (checked Choice.Some{number:-1}) -7)}
fn main()->int{
 let payload:int=13 let value:Choice=Choice.Some{number:4}
 let result:int=match value{Some(payload)=>{let inner:int=match Choice.Some{number:7}{Some(payload)=>payload.number None(_)=>0} (+ inner payload.number)} None(_)=>0}
 assert (== result 11) assert (== payload 13)
 let floating:float=match value{Some(p)=>{let f:float=(float_from_bits -9223372036854775808) f} None(_)=>0.0}
 assert (== (float_to_bits floating) -9223372036854775808)
 let text:string=match value{Some(p)=>{let s:string="selected" s} None(_)=>"empty"}
 assert (== text "selected") (println text)
 let flag:bool=match value{Some(p)=>(> p.number 0) None(_)=>false}
 assert flag
 let alias:Choice=match value{Some(p)=>{let copy:Choice=Choice.Some{number:p.number} copy} None(_)=>Choice.None{}}
 match alias{Some(p)=>{assert (== p.number 4)} None(_)=>{assert false}}
 assert (== (checked Choice.Some{number:-1}) -7) assert (== (checked Choice.Some{number:3}) 106)
 return 0
}
shadow main{assert true}
'''
LOOP=PREFIX+'''fn main()->int{
 let mut index:int=0 let mut total:int=0
 while (< index 5){
  set index (+ index 1)
  let value:Choice=Choice.Some{number:index}
  let delta:int=match value{Some(p)=>{if (== p.number 1){continue} if (== p.number 3){break} p.number} None(_)=>0}
  set total (+ total delta)
 }
 assert (== index 3) assert (== total 2)
 return 0
}
shadow main{assert true}
'''
EXIT=PREFIX+'''fn nested(value:Choice)->int{
 let result:int=match value{
  Some(p)=>{let absent:int=match value{Some(q)=>{return 7} None(_)=>{return 8}} absent}
  None(_)=>3
 }
 return (+ result 10)
}
shadow nested{assert (== (nested Choice.Some{number:1}) 7) assert (== (nested Choice.None{}) 13)}
fn main()->int{assert (== (nested Choice.Some{number:1}) 7) assert (== (nested Choice.None{}) 13) return 0}
shadow main{assert true}
'''
if __name__=='__main__':unittest.main()
