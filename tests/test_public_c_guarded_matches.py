"""I qualify the total guarded union statement profile without new global policy."""
import os
from pathlib import Path
import re
import unittest
from tests import test_public_c_nonfinite_format as formatting
ROOT=Path(__file__).resolve().parents[1]
class PublicCGuarded(unittest.TestCase):
    setUp=formatting.PublicCFormat.setUp
    run_cmd=formatting.PublicCFormat.run_cmd
    emit=formatting.PublicCFormat.emit
    compile=formatting.PublicCFormat.compile
    def qualify(self,source,expected):
        output,code=self.emit(source);self.assertNotIn('__typeof__',code)
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
    def test_payload_guards_first_success_and_final_wildcard(self):
        self.qualify(GUARDS,'high\nactive\nzero\nempty\n')
    def test_nested_labels_outer_loop_control_and_scopes(self):
        code=self.qualify(LOOPS,'2\n9\n')
        labels=re.findall(r'(nano_cb_\d+_match_end_\d+):',code)
        # All matches here are in main, so each label must be unique in that function.
        self.assertEqual(len(labels),len(set(labels)))
        self.assertGreaterEqual(len(labels),3)
        self.assertNotIn('nano_cb_0_match_end_0:',code)
    def test_guarded_final_wildcard_with_independent_coverage(self):
        self.qualify(COVERED,'1\n2\n')
    def test_profile_refusals_publication_and_recovery(self):
        exe=self.work/'api'
        self.run_cmd([os.environ.get('CC','cc'),'-std=c99','-D_POSIX_C_SOURCE=200809L',
            '-Wall','-Wextra','-Werror','-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all',
            '-I',ROOT/'src',ROOT/'tests/test_public_c_guarded_api.c','-o',exe])
        self.run_cmd([exe,self.work/'previous.c'])
        generated=self.work/'previous.c';binary=self.work/'api-generated'
        self.compile(generated,binary);self.run_cmd([binary])
GUARDS='''union Status{Active{score:int,text:string,flag:bool,real:float},Inactive{}}
let mut trace:int=0
let mut count:int=0
fn mark(n:int)->int{set count (+ count 1) return n}
shadow mark{assert (== (mark 1) 1)}
fn guard(id:int,yes:bool)->bool{set trace (+ (* trace 10) id) return yes}
shadow guard{assert (guard 1 true)}
fn describe(value:Status)->string{
 match value{
  Active(a) if (guard 1 (> a.score 100))=>{assert a.flag assert (== a.text "yes") assert (== (float_to_bits a.real) 4609434218613702656) return "high"}
  Active(a) if (guard 2 (> a.score 0))=>{return "active"}
  Active(a)=>{return "zero"}
  _=>{return "empty"}
 }
}
shadow describe{assert (== (describe Status.Inactive{}) "empty")}
fn main()->int{
 let before:int=count
 (println (describe Status.Active{score:(mark 150),text:"yes",flag:true,real:1.5}))
 assert (== count (+ before 1)) assert (== trace 1)
 set trace 0
 (println (describe Status.Active{score:50,text:"yes",flag:true,real:1.5}))
 assert (== trace 12)
 set trace 0
 (println (describe Status.Active{score:0,text:"yes",flag:true,real:1.5}))
 assert (== trace 12)
 set trace 0
 (println (describe Status.Inactive{})) assert (== trace 0)
 return 0
}
shadow main{assert true}
'''
LOOPS='''union Item{Some{n:int},None{}}
fn nano_cb_0_match_end_0(n:int)->int{return n}
shadow nano_cb_0_match_end_0{assert (== (nano_cb_0_match_end_0 2) 2)}
fn main()->int{
 let mut i:int=0 let mut total:int=0 let _:int=9
 while (< i 5){
  set i (+ i 1)
  let value:Item=Item.Some{n:i}
  match value{
   Some(p) if (== p.n 1)=>{continue}
   Some(p) if (== p.n 3)=>{break}
   _=>{set total (+ total i)}
  }
 }
 assert (== i 3) assert (== total 2)
 let p:int=8 let _match_val:Item=Item.Some{n:4}
 match _match_val{
  Some(p) if (> p.n 0)=>{
   match Item.Some{n:5}{
    Some(p) if (> p.n 0)=>{assert (== p.n 5)}
    _=>{assert false}
   }
   assert (== p.n 4)
  }
  _=>{assert false}
 }
 assert (== p 8)
 match _match_val{Some(_) if true=>{assert (== _ 9)} _=>{assert false}}
 assert (== (nano_cb_0_match_end_0 2) 2)
 (println (int_to_string total)) (println (int_to_string _)) return 0
}
shadow main{assert true}
'''
COVERED='''union Flag{Yes{},No{}}
let mut count:int=0
fn guarded()->bool{set count (+ count 1) return false}
shadow guarded{assert (not (guarded))}
fn choose(v:Flag)->int{
 match v{Yes(_) if true=>{return 1} No(_)=>{return 2} _ if (guarded)=>{return 3}}
}
shadow choose{assert (== (choose Flag.Yes{}) 1)}
fn main()->int{
 (println (int_to_string (choose Flag.Yes{})))
 (println (int_to_string (choose Flag.No{})))
 assert (== count 0) return 0
}
shadow main{assert true}
'''
if __name__=='__main__':unittest.main()
