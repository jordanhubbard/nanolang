"""I qualify actual C File compiler publication and invocation boundaries."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
DECL='service "nsi:nanolang/filesystem" catalog 1 from "interface.nsi.json"\n'
MAIN='''fn main()->int {
 match (temp) {Error(e)=>{return -1} Ok(file)=>{
  let closed:CloseResult=(close file)
  match closed {Ok()=>{return 7} Error(e)=>{return -2}}
 }}
}
'''

class ServiceDrivers(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(prefix="nano-file-drivers-")
        self.addCleanup(self.directory.cleanup)
        self.work=Path(self.directory.name).resolve()
        self.source=self.work/'source.nano'
        self.catalog=self.work/'interface.nsi.json'
        self.catalog.write_bytes((ROOT/'tests/fixtures/nsi_file_plan.json').read_bytes())
        self.env=os.environ.copy()
        self.env['NANO_CC']=os.environ.get('NANO_NATIVE_TEST_CC','cc')
        self.env['NANO_CFLAGS']='-O1'
        self.drivers=[ROOT/'bin/nanoc_c',Path(os.environ.get('NANO_SERVICE_DRIVER_VIRT',ROOT/'bin/nano_virt'))]

    def run_command(self,args,expected=0,env=None):
        run=subprocess.run(list(map(str,args)),cwd=self.work,env=env or self.env,
                           capture_output=True,text=True,timeout=120)
        self.assertEqual(run.returncode,expected,run.stdout+run.stderr)
        self.assertEqual(list(self.work.glob('.nano-*')),[],run.stdout+run.stderr)
        return run

    def test_generated_shadows_byte_parity_and_native_runtime_grant(self):
        generated=(ROOT/'tests/fixtures/nsi_file_binding_expected.nano.txt').read_text()
        self.source.write_text(generated+MAIN)
        modules=[]
        for driver in self.drivers:
            bytecode=self.work/(driver.name+'.nvm')
            run=self.run_command([driver,self.source,'--allow-temporary-files','--emit-nvm','-o',bytecode])
            records=run.stderr.splitlines()
            selected=[x[7:] for x in records if x.startswith('SELECT ')]
            self.assertEqual(len(selected),5)
            self.assertEqual(selected,[x[6:] for x in records if x.startswith('START ')])
            self.assertEqual(selected,[x[5:] for x in records if x.startswith('DONE ')])
            modules.append(bytecode.read_bytes())
            self.run_command([ROOT/'bin/nano_vm','--allow-temporary-files','--file-cyclic',
                              '--file-instruction-limit','1000000',bytecode],7)
            native=self.work/driver.name
            self.run_command([driver,self.source,'--allow-temporary-files','-o',native])
            self.run_command([native],1)
            self.run_command([native,'--allow-temporary-files'],7)
            env=self.env.copy();env['PATH']=str(driver.parent)+os.pathsep+env.get('PATH','')
            self.run_command([driver.name,self.source,'--allow-temporary-files','--emit-nvm','-o',bytecode],env=env)
            symbols=subprocess.run(['nm',str(native)],capture_output=True,text=True,check=True).stdout
            self.assertNotIn('vm_execute',symbols)
            self.assertNotIn('nvm_file_execute_cyclic_bytes',symbols)
        self.assertEqual(modules[0],modules[1])
        if self.drivers[1].name.startswith('nano_virt'):
            self.run_command([self.drivers[1],self.source,'--allow-temporary-files','--run'],7)
            self.run_command([self.drivers[1],self.source,'--allow-temporary-files'])
            env=self.env.copy();env['PATH']=str(self.drivers[1].parent)+os.pathsep+env.get('PATH','')
            self.run_command([self.drivers[1].name,self.source,'--allow-temporary-files','--run'],7,env)
        else:
            for driver in self.drivers:
                self.run_command([driver,self.source,'--allow-temporary-files'])
                self.run_command([self.work/'a.out','--allow-temporary-files'],7)

    def test_cyclic_owners_helpers_and_selected_shadows(self):
        generated=(ROOT/'tests/fixtures/nsi_file_binding_expected.nano.txt').read_text()
        self.source.write_text(generated+"""
fn write_checked(file:&mut File,octet:int)->int {
 match (write_byte &mut file octet) {Ok(n)=>{return n} Error(e)=>{return -1}}
}
shadow write_checked {
 match (temp) {Error(e)=>{assert false} Ok(f)=>{
  let mut file:File=f
  assert (== (write_checked &mut file 23) 1)
  let c:CloseResult=(close file)
  match c {Ok()=>{} Error(e)=>{assert false}}
 }}
}
fn cycle()->int {
 let mut total:int=0
 let mut round:int=0
 while (< round 3) {
  match (temp) {Error(e)=>{return -1} Ok(f)=>{
   let mut file:File=f
   let mut index:int=0
   while (< index 2) {
    assert (== (write_checked &mut file (+ 20 index)) 1)
    set index (+ index 1)
   }
   match (rewind &mut file) {Ok()=>{} Error(e)=>{assert false}}
   match (read_byte &mut file) {
    Error(e)=>{let c:CloseResult=(close file) return -2}
    Ok(octet)=>{set total (+ total octet.value)}
   }
   match (close file) {Ok()=>{} Error(e)=>{return -3}}
  }}
  set round (+ round 1)
 }
 return total
}
shadow cycle {assert (== (cycle) 60)}
fn main()->int {return (cycle)}
""")
        modules=[]
        for driver in self.drivers:
            bytecode=self.work/(driver.name+'.cycle.nvm')
            run=self.run_command([driver,self.source,'--allow-temporary-files','--emit-nvm','-o',bytecode])
            records=run.stderr.splitlines()
            selected=[x[7:] for x in records if x.startswith('SELECT ')]
            self.assertEqual(len(selected),7)
            self.assertEqual(selected,[x[6:] for x in records if x.startswith('START ')])
            self.assertEqual(selected,[x[5:] for x in records if x.startswith('DONE ')])
            modules.append(bytecode.read_bytes())
            self.run_command([ROOT/'bin/nano_vm','--allow-temporary-files','--file-cyclic',
                              '--file-instruction-limit','1000000',bytecode],60)
            native=self.work/(driver.name+'.cycle')
            self.run_command([driver,self.source,'--allow-temporary-files','-o',native])
            self.run_command([native,'--allow-temporary-files'],60)
        self.assertEqual(modules[0],modules[1])

    def test_shared_aliases_and_forwarded_loans(self):
        self.check_multiple_borrows('file_shared_borrow.nano',10,43)

    def test_multiple_forwarded_reordered_file_borrows(self):
        self.check_multiple_borrows('file_multiborrow.nano',8)

    def test_multiple_borrows_with_owned_operand(self):
        self.check_multiple_borrows('file_multiborrow_owned.nano',9)

    def test_multiple_borrow_shadow_failure_preserves_output(self):
        body=(ROOT/'tests/fixtures/file_multiborrow.nano').read_text().replace('assert (== n 1)','assert false',1)
        self.source.write_text((ROOT/'tests/fixtures/nsi_file_binding_expected.nano.txt').read_text()+body)
        for driver in self.drivers:
            output=self.work/'prior.nvm';output.write_bytes(b'prior')
            run=self.run_command([driver,self.source,'--allow-temporary-files','--emit-nvm','-o',output],1)
            self.assertIn('File shadow failure',run.stderr)
            self.assertEqual(output.read_bytes(),b'prior')

    def test_shared_borrow_shadow_failure_preserves_output(self):
        body=(ROOT/'tests/fixtures/file_shared_borrow.nano').read_text().replace('return (+ n 1)','assert false return (+ n 1)')
        self.source.write_text((ROOT/'tests/fixtures/nsi_file_binding_expected.nano.txt').read_text()+body)
        for driver in self.drivers:
            output=self.work/'prior.nvm';output.write_bytes(b'prior')
            run=self.run_command([driver,self.source,'--allow-temporary-files','--emit-nvm','-o',output],1)
            self.assertIn('File shadow failure',run.stderr)
            self.assertEqual(output.read_bytes(),b'prior')

    def check_multiple_borrows(self,fixture,shadows,expected=129):
        self.source.write_text((ROOT/'tests/fixtures/nsi_file_binding_expected.nano.txt').read_text()+
                               (ROOT/'tests/fixtures'/fixture).read_text())
        modules=[]
        for driver in self.drivers:
            bytecode=self.work/(driver.name+'.borrows.nvm')
            run=self.run_command([driver,self.source,'--allow-temporary-files','--emit-nvm','-o',bytecode])
            records=run.stderr.splitlines()
            selected=[x[7:] for x in records if x.startswith('SELECT ')]
            self.assertEqual(len(selected),shadows)
            self.assertEqual(selected,[x[6:] for x in records if x.startswith('START ')])
            self.assertEqual(selected,[x[5:] for x in records if x.startswith('DONE ')])
            modules.append(bytecode.read_bytes())
            self.run_command([ROOT/'bin/nano_vm','--allow-temporary-files','--file-cyclic',
                              '--file-instruction-limit','1000000',bytecode],expected)
            native=self.work/(driver.name+'.borrows')
            self.run_command([driver,self.source,'--allow-temporary-files','-o',native])
            self.run_command([native,'--allow-temporary-files'],expected)
        self.assertEqual(modules[0],modules[1])

    def test_cyclic_and_overlapping_owner_refusals_preserve_output(self):
        cases=[
            'fn consume(file:File)->void {let mut i:int=0 while (< i 2) {'
            'let c:CloseResult=(close file) set i (+ i 1)}} fn main()->int{return 0}',
            'fn pair(a:&mut File,b:&mut File)->int{return 0} fn main()->int {'
            'match (temp) {Error(e)=>{return -1} Ok(f)=>{let mut file:File=f '
            'let n:int=(pair &mut file &mut file) let c:CloseResult=(close file) return n}}}',
            'fn pair(a:&File,b:&mut File)->int{return 0} fn main()->int {'
            'match (temp) {Error(e)=>{return -1} Ok(f)=>{let mut file:File=f '
            'let n:int=(pair &file &mut file) let c:CloseResult=(close file) return n}}}',
            'fn mutate(a:&File)->int{let result:PositionResult=(rewind &mut a) return 0} '
            'fn main()->int{return 0}',

        ]
        for body in cases:
            self.source.write_text(DECL+body)
            for driver in self.drivers:
                with self.subTest(driver=driver.name,body=body):
                    output=self.work/'prior.nvm';output.write_bytes(b'prior')
                    run=self.run_command([driver,self.source,'--allow-temporary-files','--emit-nvm','-o',output],1)
                    self.assertEqual(output.read_bytes(),b'prior')
                    self.assertNotIn('START ',run.stderr)
                    self.assertIn('mutable File root' if body.startswith('fn mutate') else 'ownership',run.stdout+run.stderr)

    def test_required_import_shadows_and_root_only_selection(self):
        (self.work/'binding.nano').write_text(DECL+'''pub fn broken()->void {}
shadow broken {assert false}
''')
        self.source.write_text('module "binding.nano" as files\nfn main()->int{return 0}\nshadow main {assert true}\n')
        for driver in self.drivers:
            output=self.work/'prior.nvm';output.write_bytes(b'prior')
            run=self.run_command([driver,self.source,'--emit-nvm','--allow-temporary-files','-o',output],1)
            self.assertIn('File shadow failure',run.stderr)
            self.assertEqual(output.read_bytes(),b'prior')
            run=self.run_command([driver,self.source,'--emit-nvm','--allow-temporary-files',
                                  '--root-shadows-only','-o',output])
            self.assertEqual(sum(x.startswith('SELECT ') for x in run.stderr.splitlines()),1)
            self.assertNotEqual(output.read_bytes(),b'prior')
        (self.work/'binding.nano').write_text(DECL+'pub fn broken()->void {}\nshadow broken {assert 1}\n')
        for driver in self.drivers:
            output=self.work/'prior.nvm';output.write_bytes(b'prior')
            self.run_command([driver,self.source,'--emit-nvm','--allow-temporary-files',
                              '--root-shadows-only','-o',output],1)
            self.assertEqual(output.read_bytes(),b'prior')

    def test_grant_failure_aliases_and_compiler_failure_preserve_output(self):
        self.source.write_text(DECL+MAIN+'shadow main {assert true}\n')
        for driver in self.drivers:
            output=self.work/'prior';output.write_bytes(b'prior')
            run=self.run_command([driver,self.source,'-o',output],1)
            self.assertIn('--allow-temporary-files',run.stderr)
            self.assertEqual(output.read_bytes(),b'prior')
            for protected in (self.source,self.catalog):
                before=protected.read_bytes()
                alias=self.work/'alias';alias.unlink(missing_ok=True);os.link(protected,alias)
                for path in (protected,alias):
                    self.run_command([driver,self.source,'--allow-temporary-files','--emit-nvm','-o',path],1)
                    self.assertEqual(protected.read_bytes(),before)
            env=self.env.copy();env['NANO_CC']='false'
            self.run_command([driver,self.source,'--allow-temporary-files','-o',output],1,env)
            self.assertEqual(output.read_bytes(),b'prior')
            unsupported=['--target','c'] if driver.name=='nanoc_c' else ['--daemon-wrapper']
            run=self.run_command([driver,self.source,'--allow-temporary-files',*unsupported,'-o',output],1)
            self.assertNotIn('START ',run.stderr)
            self.assertEqual(output.read_bytes(),b'prior')
            env=self.env.copy();env['NANO_SHADOW_TIMEOUT_SECONDS']='0'
            self.run_command([driver,self.source,'--emit-nvm','--allow-temporary-files','-o',output],1,env)
            self.assertEqual(output.read_bytes(),b'prior')
            destination=self.work/'destination';destination.mkdir(exist_ok=True)
            self.run_command([driver,self.source,'--emit-nvm','--allow-temporary-files','-o',destination],1)
            self.assertEqual(list(destination.iterdir()),[])

    def test_no_shadows_need_no_compile_grant_and_invalid_bodies_refuse(self):
        self.source.write_text(DECL+MAIN)
        for driver in self.drivers:
            output=self.work/'ungranted.nvm'
            self.run_command([driver,self.source,'--emit-nvm','-o',output])
            self.run_command([ROOT/'bin/nano_vm',output],1)
        self.source.write_text(DECL+'fn main()->int {let file:File=1 return 0}\n')
        for driver in self.drivers:
            output=self.work/'prior.nvm';output.write_bytes(b'prior')
            run=self.run_command([driver,self.source,'--allow-temporary-files','--emit-nvm','-o',output],1)
            self.assertIn('type-check File service bodies',run.stdout+run.stderr)
            self.assertEqual(output.read_bytes(),b'prior')

if __name__=='__main__':unittest.main()
