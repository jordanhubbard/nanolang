"""I qualify lexical File ownership through both independent source routes."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from tests.test_service_bodies import DECL, POSITIVE, ROOT

VALID = {
    'lifecycle': POSITIVE,
    'terminal-operator': 'fn exercise(file:File)->int {(+ 5 (match (close file) {Ok()=>{return 7} Error(e)=>{return 8}}))}',
    'branch-close': 'fn exercise(file:File,yes:bool)->void { if yes { let a:CloseResult=(close file) } else { let b:CloseResult=(close file) } }',
    'early-return': 'fn exercise(file:File,yes:bool)->int { if yes { let a:CloseResult=(close file) return 1 } let b:CloseResult=(close file) return 0 }',
    'loop-borrow': 'fn exercise(file:File)->void { let mut owned:File=file let mut i:int=0 while (< i 2) { let a:WriteResult=(write_byte &mut owned i) set i (+ i 1) } let b:CloseResult=(close owned) }',
    'loop-owner': 'fn exercise()->void { let mut i:int=0 while (< i 2) { let r:OpenResult=(temp) match r { Ok(f)=>{let c:CloseResult=(close f)} Error(e)=>{} } set i (+ i 1) } }',
    'reinitialize': 'fn exercise(file:File,other:File)->void { let mut owned:File=file let a:CloseResult=(close owned) set owned other let b:CloseResult=(close owned) }',
    'shared-call': 'fn observe(a:&File,b:&File)->int{return 0} fn exercise(file:File)->void { let n:int=(observe &file &file) let c:CloseResult=(close file) }',
    'lexical-shadow': 'fn exercise(file:File)->void { if true { let file:int=1 assert (== file 1) } let c:CloseResult=(close file) }',
    'branch-return': 'fn choose(a:File,b:File,yes:bool)->File { if yes { let c:CloseResult=(close b) return a } else { let c:CloseResult=(close a) return b } }',
    'error-first': 'fn exercise(result:OpenResult)->void { match result { Error(e)=>{} Ok(file)=>{let c:CloseResult=(close file)} } }',
    'terminal-argument': '''fn sink(stop:void, file:&mut File)->void { return }
fn exercise(file:File)->int {
    let mut owned:File=file
    (sink (match (rewind &mut owned) {
        Ok()=>{ let c:CloseResult=(close owned) return 1 }
        Error(e)=>{ let c:CloseResult=(close owned) return 2 }
    }) &mut owned)
    return 3
}''',
}
CLOSE_BOOL = 'fn consumed(file:File)->bool { let c:CloseResult=(close file) return true } '
INVALID = {
    'short-circuit-and': CLOSE_BOOL+'fn exercise(file:File)->void {let skipped:bool=(and false (consumed file))}',
    'short-circuit-or': CLOSE_BOOL+'fn exercise(file:File)->void {let skipped:bool=(or true (consumed file))}',
    'double-close': 'fn exercise(file:File)->void { let a:CloseResult=(close file) let b:CloseResult=(close file) }',
    'use-after-move': 'fn exercise(file:File)->File { let owned:File=file let a:CloseResult=(close file) return owned }',
    'parameter-leak': 'fn unused(file:File)->void { return }',
    'result-leak': 'fn exercise()->void { let opened:OpenResult=(temp) }',
    'unbound-result': 'fn exercise()->void { (temp) }',
    'overwrite': 'fn exercise(file:File,other:File)->void { let mut owned:File=file set owned other let c:CloseResult=(close owned) }',
    'branch-disagreement': 'fn exercise(file:File,yes:bool)->void { if yes { let c:CloseResult=(close file) } let c:CloseResult=(close file) }',
    'return-leak': 'fn exercise(file:File,yes:bool)->int { if yes { return 1 } let c:CloseResult=(close file) return 0 }',
    'loop-consume': 'fn exercise(file:File,yes:bool)->void { while yes { let c:CloseResult=(close file) } let c:CloseResult=(close file) }',
    'loop-condition': CLOSE_BOOL+'fn exercise(file:File)->void { while (consumed file) {} }',
    'duplicate-owner': 'fn pair(a:File,b:File)->void {let c:CloseResult=(close a) let d:CloseResult=(close b)} fn exercise(file:File)->void {(pair file file)}',
    'duplicate-exclusive': 'fn pair(a:&mut File,b:&mut File)->void {let c:WriteResult=(write_byte &mut a 1) let d:WriteResult=(write_byte &mut b 2)} fn exercise(file:File)->void {let mut owned:File=file (pair &mut owned &mut owned) let c:CloseResult=(close owned)}',
    'consume-during-borrow': CLOSE_BOOL+'fn pair(a:&mut File,b:bool)->void { let c:WriteResult=(write_byte &mut a 1) } fn exercise(file:File)->void { let mut owned:File=file (pair &mut owned (consumed owned)) let c:CloseResult=(close owned) }',
    'borrow-after-consume': CLOSE_BOOL+'fn pair(a:bool,b:&mut File)->void { let c:WriteResult=(write_byte &mut b 1) } fn exercise(file:File)->void { let mut owned:File=file (pair (consumed owned) &mut owned) let c:CloseResult=(close owned) }',
    'payload-leak': 'fn exercise()->void {let r:OpenResult=(temp) match r {Ok(f)=>{} Error(e)=>{}}}',
    'shadow-leak': 'shadow main {let opened:OpenResult=(temp)}',
}

def source(body):
    return DECL+body+('' if 'fn main(' in body else '\nfn main()->int{return 0}\n')


class ServiceOwnership(unittest.TestCase):
    @staticmethod
    def checked(command, timeout=90):
        run=subprocess.run(list(map(str,command)),cwd=ROOT,capture_output=True,text=True,timeout=timeout)
        if run.returncode:
            raise AssertionError(f'{command}\nexit={run.returncode}\n{run.stdout}\n{run.stderr}')
        return run.stdout

    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory(prefix='nano-service-ownership-')
        cls.addClassCleanup(cls.directory.cleanup)
        cls.work=Path(cls.directory.name).resolve()
        (cls.work/'interface.nsi.json').write_bytes((ROOT/'tests/fixtures/nsi_file_plan.json').read_bytes())
        driver=os.environ.get('NANO_SERVICE_OWNERSHIP_DRIVER_MODULE')
        producers=[[ROOT/'bin/nano_virt']]
        cls.drivers=[[ROOT/'bin/nanoc_c','--emit-nvm'],[ROOT/'bin/nano_virt','--emit-nvm']]
        if driver:
            producers.append([ROOT/'bin/nano_vm',driver,'--'])
            cls.drivers.append([ROOT/'bin/nano_vm',driver,'--','--emit-nvm'])
        cls.probes=[]
        for i,command in enumerate(producers):
            module=cls.work/f'probe-{i}.nvm'; native=module.with_suffix('.native'); generated=module.with_suffix('.c')
            cls.checked([*command,ROOT/'tests/service_ownership.nano','--emit-nvm','-o',module],300)
            cls.probes.append([ROOT/'bin/nano_vm',module,'--'])
            cls.checked([ROOT/'bin/nvm2c',module,'-o',generated],120)
            compiler=shlex.split(os.environ.get('NANO_NATIVE_TEST_CC') or os.environ.get('CC') or 'cc')
            cls.checked([*compiler,'-std=c11','-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all',
                         generated,'-o',native,'-lm',*(['-ldl'] if sys.platform.startswith('linux') else [])],300)
            cls.probes.append([native])

    def check(self, body, status, complete=False):
        path=self.work/'body.nano';path.write_text(body if complete else source(body))
        reports=[self.checked([os.environ.get('NANO_SERVICE_OWNERSHIP_C_RUNNER',ROOT/'obj/test_service_ownership'),path,status])]
        for command in self.probes: reports.append(self.checked([*command,path,status]))
        facts=[[line for line in report.splitlines() if line.startswith('FLOW ')] for report in reports]
        for actual in facts[1:]: self.assertEqual(facts[0],actual,reports)
        self.check_drivers(path,status)
        return reports

    def check_drivers(self,path,status):
        for command in self.drivers:
            output=self.work/'prior.nvm';output.write_bytes(b'prior-output')
            run=subprocess.run(list(map(str,[*command,path,'-o',output])),cwd=ROOT,capture_output=True,text=True,timeout=90)
            message=run.stdout+run.stderr
            if run.returncode==0:
                self.assertEqual(status,0)
                self.assertTrue(output.read_bytes().startswith(b'NVM'))
                continue
            self.assertNotIn('I cannot type-check File service bodies:',message)
            if status==1: self.assertIn('I cannot verify File service ownership:',message)
            else:
                self.assertRegex(message,r'I have not resolved File service declarations|I require --allow-temporary-files|I cannot lower File source')
                self.assertNotIn('I cannot verify File service ownership:',message)
            self.assertEqual(output.read_bytes(),b'prior-output')

    def test_moves_branches_loops_and_borrows(self):
        for name,body in VALID.items():
            with self.subTest(case=name):self.check(body,0)

    def test_invalid_ownership_in_called_uncalled_and_shadow_bodies(self):
        for name,body in INVALID.items():
            with self.subTest(case=name):self.check(body,1)

    def test_published_binding_keeps_all_five_shadow_bodies(self):
        destination=self.work/'generated'
        self.checked([ROOT/'bin/nsi-file-binding',ROOT/'tests/fixtures/nsi_file_plan.json','--file-binding-dir',destination])
        generated=(destination/'binding.nano').read_text()
        self.assertEqual(generated,(ROOT/'tests/fixtures/nsi_file_binding_expected.nano.txt').read_text())
        self.assertIn(DECL,generated)
        reports=self.check(generated+'\nfn main()->int{return 0}\n',0,complete=True)
        self.assertTrue(all('SHADOWS 5' in report for report in reports))

    def test_imported_owners_and_borrowed_aliases(self):
        library=self.work/'binding.nano'
        bridge=self.work/'bridge.nano';bridge.write_text('pub use "binding.nano" as files\n')
        for bad in (False,True):
            with self.subTest(bad=bad):
                library.write_text(DECL+'pub fn pass(file:File)->File{return file}\n'+
                                  ('fn invalid(file:File)->void{return}\n' if bad else ''))
                root=self.work/'root.nano'
                root.write_text('module "bridge.nano" as api\nfn exercise(file:api.files.File)->void {let mut owned:api.files.File=(api.files.pass file) let w:api.files.WriteResult=(api.files.write_byte &mut owned 1) let c:api.files.CloseResult=(api.files.close owned)}\nfn main()->int{return 0}\n')
                self.checked([os.environ.get('NANO_SERVICE_OWNERSHIP_C_RUNNER',ROOT/'obj/test_service_ownership'),root,int(bad)])
                self.check_drivers(root,int(bad))
