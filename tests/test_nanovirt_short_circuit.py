"""I select source boolean operands while retaining eager ISA operations."""
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
class ShortCircuit(unittest.TestCase):
    def run_source(self,source):
        work=Path(tempfile.mkdtemp(prefix='nano-source-short-circuit-'))
        print('I retain source short-circuit evidence at',work,flush=True)
        path=work/'ordinary.nano';path.write_text(source);module=work/'ordinary.nvm'
        for args in ([ROOT/'bin/nano',path], [ROOT/'bin/nano_virt',path,'--emit-nvm','-o',module],
                     [ROOT/'bin/nano_vm','--verify-only',module], [ROOT/'bin/nano_vm',module]):
            result=subprocess.run(list(map(str,args)),cwd=ROOT,text=True,capture_output=True,timeout=30)
            self.assertEqual(result.returncode,0,str(args)+'\n'+result.stdout+result.stderr)
    def test_selected_skipped_nested_and_loop_operands(self):self.run_source(ORDER)
    def test_rhs_enclosing_return_break_and_continue(self):self.run_source(EXITS)
    def test_wrong_operand_type_preserves_prior_output(self):
        work=Path(tempfile.mkdtemp(prefix='nano-source-short-circuit-refusal-'))
        for expression in ['(and true 1)','(or 1 false)']:
            path=work/'ordinary.nano';path.write_text('fn main()->int{let value:bool='+expression+' return 0}\nshadow main{assert true}\n')
            module=work/'previous.nvm';module.write_bytes(b'previous')
            args=[ROOT/'bin/nano_virt',path,'--emit-nvm','-o',module]
            result=subprocess.run(list(map(str,args)),cwd=ROOT,text=True,capture_output=True,timeout=30)
            self.assertNotEqual(result.returncode,0,str(args));self.assertIn('bool operands',result.stderr)
            self.assertEqual(module.read_bytes(),b'previous')
ORDER='''let mut trace:int=0
fn mark(n:int,value:bool)->bool{set trace (+ (* trace 10) n) return value}
shadow mark{set trace 0 assert (mark 1 true) assert (== trace 1)}
fn main()->int{
 set trace 0 assert (not (and (mark 1 false) (mark 2 true))) assert (== trace 1)
 set trace 0 assert (and (mark 1 true) (mark 2 true)) assert (== trace 12)
 set trace 0 assert (or (mark 1 true) (mark 2 false)) assert (== trace 1)
 set trace 0 assert (not (or (mark 1 false) (mark 2 false))) assert (== trace 12)
 set trace 0 assert (and (mark 1 true) (or (mark 2 false) (mark 3 true))) assert (== trace 123)
 let mut count:int=0 set trace 0
 while (and (< count 2) (mark 4 true)){set count (+ count 1)}
 assert (== count 2) assert (== trace 44)
 return 0
}
shadow main{assert true}
'''
EXITS='''union Choice{Some{n:int},None{}}
fn leave(selected:bool)->bool{
 return (and selected (match Choice.Some{n:7}{Some(p)=>{if (> p.n 0){return false} true} None(_)=>true}))
}
shadow leave{assert (not (leave false)) assert (not (leave true))}
fn main()->int{
 assert (not (leave false)) assert (not (leave true))
 let mut index:int=0 let mut visits:int=0
 while (< index 5){
  set index (+ index 1)
  let chosen:bool=(and true (match Choice.Some{n:index}{Some(p)=>{if (== p.n 1){continue} if (== p.n 3){break} true} None(_)=>false}))
  if chosen{set visits (+ visits 1)}
 }
 assert (== index 3) assert (== visits 1)
 return 0
}
shadow main{assert true}
'''
if __name__=='__main__':unittest.main()
