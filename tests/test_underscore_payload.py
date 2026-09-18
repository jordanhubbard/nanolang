"""I discard an underscore payload without hiding ordinary outer names."""
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
SOURCE = '''union Choice { Some { n: int }, Other { a: int, b: int }, Empty {} }
fn choose(x: Choice) -> int {
 let _: int = 41
 match x { Some(_) => { return _ } Other(_) => { return _ } Empty(_) => { return _ } }
 return 0
}
shadow choose { let v: Choice = Choice.Some { n: 9 } assert (== (choose v) 41) }
fn expression(x: Choice) -> int {
 let _: int = 23
 let answer: int = match x { Some(_) => _ Other(_) => _ Empty(_) => _ }
 return answer
}
shadow expression { let v: Choice = Choice.Some { n: 9 } assert (== (expression v) 23) }
fn main() -> int {
 let a: Choice = Choice.Some { n: 9 }
 let b: Choice = Choice.Other { a: 1, b: 2 }
 let c: Choice = Choice.Empty {}
 assert (== (choose a) 41)
 assert (== (choose b) 41)
 assert (== (choose c) 41)
 assert (== (expression b) 23)
 return 0
}
'''
class UnderscorePayload(unittest.TestCase):
 def checked(self, args):
  r=subprocess.run(list(map(str,args)),cwd=ROOT,capture_output=True,text=True,timeout=120)
  self.assertEqual(r.returncode,0,r.stdout+r.stderr)
  return r
 def test_outer_name_all_stages(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);src=p/'input.nano';src.write_text(SOURCE)
   self.checked([ROOT/'bin/nano',src])
   for cc in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
    with self.subTest(compiler=cc):
     out=p/cc;self.checked([ROOT/'bin'/cc,src,'-o',out]);self.checked([out])
   mod=p/'module.nvm';self.checked([ROOT/'bin/nano_virt',src,'-o',mod]);self.checked([ROOT/'bin/nano_vm',mod])
 def test_guarded_discard_and_named_payload_controls(self):
  source = SOURCE.replace('Some(_) => { return _ }', 'Some(_) if (== _ 41) => { return _ } Some(payload) => { return payload.n }')
  source = source.replace('Some(_) => _', 'Some(_) if (== _ 23) => _ Some(payload) => payload.n')
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);src=p/'input.nano';src.write_text(source)
   self.checked([ROOT/'bin/nano',src])
   out=p/'native';self.checked([ROOT/'bin/nanoc_c',src,'-o',out]);self.checked([out])
   mod=p/'module.nvm';self.checked([ROOT/'bin/nano_virt',src,'-o',mod]);self.checked([ROOT/'bin/nano_vm',mod])
 def test_failed_shadow_preserves_output(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);src=p/'input.nano';src.write_text(SOURCE.replace('assert (== (choose v) 41)', 'assert (== (choose v) 42)'))
   for cc in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
    with self.subTest(compiler=cc):
     out=p/'output';out.write_text('previous accepted output')
     r=subprocess.run([ROOT/'bin'/cc,src,'-o',out],cwd=ROOT,capture_output=True,text=True,timeout=120)
     self.assertGreater(r.returncode,0,r.stdout+r.stderr)
     self.assertIn('shadow',(r.stdout+r.stderr).lower())
     self.assertEqual(out.read_text(),'previous accepted output')
 def test_absent_outer_name_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);src=p/'input.nano';src.write_text(SOURCE.replace(' let _: int = 41','').replace(' let _: int = 23',''))
   for cc in ('nanoc_c','nanoc_stage1','nanoc_stage2','nano_virt'):
    with self.subTest(compiler=cc):
     out=p/'output';out.write_text('previous accepted output')
     r=subprocess.run([ROOT/'bin'/cc,src,'-o',out],cwd=ROOT,capture_output=True,text=True,timeout=120)
     self.assertGreater(r.returncode,0,r.stdout+r.stderr)
     self.assertIn('cannot find',(r.stdout+r.stderr).lower())
     self.assertEqual(out.read_text(),'previous accepted output')
if __name__=='__main__': unittest.main()
