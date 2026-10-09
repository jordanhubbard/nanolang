"""I lower actual checked File sources and compare granted VM/native execution."""
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import tempfile
import time
import unittest
from tests.test_service_bodies import DECL, ROOT

WRAPPER = r'''
#include "nanoisa/file_cyclic_public.h"
#include <assert.h>
#include <stdio.h>
#include <fcntl.h>
extern NvmFileCyclicExecutionReport nvm_file_cyclic_program_source(NvmFileHostGrant *,const NvmFileCyclicOptions *,NvmFileScalar *);
int main(void) {
    NvmFileHostGrant *grant=NULL;
    assert(nvm_file_host_grant_create_temporary_files(&grant)==NVM_FILE_HOST_OK);
    NvmFileCyclicOptions options={1,100000};NvmFileScalar scalar={0};
    unsigned before=0,after=0;
    for(int i=0;i<1024;i++)before+=fcntl(i,F_GETFD)!=-1;
    NvmFileCyclicExecutionReport denied=nvm_file_cyclic_program_source(NULL,&options,&scalar);
    assert(denied.runtime.status==NVM_FILE_RUNTIME_INVALID && !denied.runtime.acquired);
    NvmFileCyclicExecutionReport report=nvm_file_cyclic_program_source(grant,&options,&scalar);
    for(int i=0;i<1024;i++)after+=fcntl(i,F_GETFD)!=-1;
    assert(before==after);
    printf("EXEC %u VALUE %lld\n",report.runtime.status,(long long)scalar.value);
    assert(!report.runtime.cleanup.cleanup_failures);
    assert(nvm_file_host_grant_destroy(&grant)==NVM_FILE_HOST_OK);
    return 0;
}
'''

class ServiceLowering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory(prefix='nano-service-lowering-')
        cls.addClassCleanup(cls.directory.cleanup)
        cls.work=Path(cls.directory.name).resolve()
        (cls.work/'interface.nsi.json').write_bytes((ROOT/'tests/fixtures/nsi_file_plan.json').read_bytes())
        (cls.work/'nanolang').mkdir()
        (cls.work/'nanolang/file').symlink_to(ROOT/'src',target_is_directory=True)
        (cls.work/'wrapper.c').write_text(WRAPPER)
        cls.compiler=shlex.split(os.environ.get('NANO_NATIVE_TEST_CC') or os.environ.get('CC') or 'cc')

    @staticmethod
    def command(args):
        started=time.monotonic()
        process=subprocess.Popen(list(map(str,args)),cwd=ROOT,stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,text=True,start_new_session=True)
        try:
            stdout,stderr=process.communicate(timeout=120)
        except BaseException:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.communicate(timeout=10)
            raise
        if process.returncode:raise AssertionError(f'{args}\n{process.returncode}\n{stdout}\n{stderr}')
        if '-std=c11' in args and any(str(arg).endswith('source.c') for arg in args):
            print(f'I finish native compilation in {time.monotonic()-started:.3f}s',flush=True)
        return stdout

    def check(self,body,selection='main',expected=0,status=0,complete=False):
        path=self.work/'source.nano';path.write_text(body if complete else DECL+body)
        native_c=self.work/'source.c';native=self.work/'source.native'
        vm=self.command([os.environ.get('NANO_SERVICE_LOWERING_RUNNER',ROOT/'obj/test_service_lowering'),path,selection,native_c,status])
        self.command([*self.compiler,'-std=c11','-g',os.environ.get('NANO_NATIVE_TEST_OPT','-O2'),'-fsanitize=address,undefined','-fno-sanitize-recover=all',
                      '-I'+str(self.work),'-Isrc',native_c,self.work/'wrapper.c',ROOT/'lib/libnano_file_runtime.a','-o',native,'-lm'])
        actual=self.command([native])
        pattern=r'EXEC (\d+) VALUE (-?\d+)'
        self.assertEqual(re.search(pattern,vm).groups(),re.search(pattern,actual).groups())
        self.assertEqual(re.search(pattern,actual).groups(),(str(status),str(expected)))

    def test_unchanged_generated_binding_and_all_shadows(self):
        generated=(ROOT/'tests/fixtures/nsi_file_binding_expected.nano.txt').read_text()
        # I use the exact publisher golden text, preserving every source body.
        for name in ('temp','write_byte','rewind','read_byte','close'):
            with self.subTest(shadow=name):
                self.check(generated+'\nfn main()->int{return 0}\n',name,complete=True)

    def test_helpers_loops_fields_and_result_values(self):
        self.check('''
fn pass(file:File)->File {return file}
fn write(file:&mut File,octet:int)->WriteResult {return (write_byte &mut file octet)}
fn main()->int {
 let opened:OpenResult=(temp)
 match opened {
  Error(error)=>{return error.status}
  Ok(file)=>{
   let mut owned:File=(pass file)
   let mut i:int=0
   while (< i 3) {
    let written:WriteResult=(write &mut owned (+ i 10))
    match written {Ok(count)=>{assert (== count 1)} Error(error)=>{assert false}}
    set i (+ i 1)
   }
   let positioned:PositionResult=(rewind &mut owned)
   match positioned {Ok()=>{} Error(error)=>{assert false}}
   let read:ReadResult=(read_byte &mut owned)
   let value:int=(match read {Ok(octet)=>{octet.value} Error(error)=>{error.status}})
   let closed:CloseResult=(close owned)
   match closed {Ok()=>{} Error(error)=>{assert false}}
   return value
  }
 }
}
''',expected=10)

    def test_nested_argument_borrows_keep_branch_state(self):
        self.check("""
fn pick(file:&mut File,condition:bool)->int {assert condition return 7}
fn main()->int {
 match (temp) {
  Error(e)=>{return -1}
  Ok(a)=>{
   let mut first:File=a
   match (temp) {
    Error(e)=>{let c:CloseResult=(close first) return -2}
    Ok(b)=>{
     let mut second:File=b
     let value:int=(pick &mut first (or false
      (match (write_byte &mut second 9) {Ok(n)=>{(== n 1)} Error(e)=>{false}})))
     let c:CloseResult=(close first)
     let d:CloseResult=(close second)
     return value
    }
   }
  }
 }
}
""",expected=7)

    def test_short_circuit_skips_effects(self):
        self.check('fn fail()->bool {assert false return true}\nfn main()->int {assert (not (and false (fail))) assert (or true (fail)) return 31}\n',expected=31)

    def test_terminal_operands_release_pending_values(self):
        scalar = """fn sink(first:int, stop:void)->void{return}
fn exercise(file:File)->int {
 let mut owned:File=file
 (sink 5 (match (rewind &mut owned) {
  Ok()=>{let c:CloseResult=(close owned) return 7}
  Error(e)=>{let c:CloseResult=(close owned) return 8}
 }))
 return 9
}
"""
        owned = """fn sink(file:File, stop:void)->void{let c:CloseResult=(close file)}
fn exercise(file:File)->int {
 (sink file (match (temp) {
  Ok(other)=>{let c:CloseResult=(close other) return 7}
  Error(e)=>{return 8}
 }))
 return 9
}
"""
        operator = """fn exercise(file:File)->int {
 (+ 5 (match (close file) {Ok()=>{return 7} Error(e)=>{return 8}}))
}
"""
        main = """fn main()->int {
 let opened:OpenResult=(temp)
 match opened {Ok(file)=>{return (exercise file)} Error(e)=>{return -1}}
}
"""
        for body in (scalar,owned,operator):
            with self.subTest(body=body):self.check(body+main,expected=7)

    def test_limits_and_unimplemented_reference_calls_preserve_outputs(self):
        from tests.test_service_ownership import VALID
        cases = [
            (VALID['shared-call']+'fn main()->int{return 0}',2),
            ('fn main()->int{'+''.join(f'let n{i}:int={i} ' for i in range(257))+'return 0}',3),
            (''.join(f'fn f{i}()->int{{return {i}}} ' for i in range(63))+'fn main()->int{return 0}',3),
        ]
        for body,status in cases:
            path=self.work/'refused.nano';path.write_text(DECL+body)
            output=self.work/'prior.c';output.write_bytes(b'prior-output')
            report=self.command([os.environ.get('NANO_SERVICE_LOWERING_RUNNER',ROOT/'obj/test_service_lowering'),path,'main',output,f'lower:{status}'])
            self.assertIn(f'REFUSAL {status}',report)
            self.assertEqual(output.read_bytes(),b'prior-output')

    def test_assertion_drains_live_file(self):
        self.check('''fn main()->int {
 let opened:OpenResult=(temp)
 match opened {Error(error)=>{return 0} Ok(file)=>{
  assert false
  let closed:CloseResult=(close file)
  return 1
 }}
}''',status=9)

    def test_qualified_reexported_helper(self):
        (self.work/'binding.nano').write_text(DECL+'pub fn pass(file:File)->File{return file}\n')
        (self.work/'bridge.nano').write_text('pub use "binding.nano" as files\n')
        self.check('''module "bridge.nano" as api
fn main()->int {
 let opened:api.files.OpenResult=(api.files.temp)
 match opened {
  Error(error)=>{return -1}
  Ok(file)=>{
   let moved:api.files.File=(api.files.pass file)
   let closed:api.files.CloseResult=(api.files.close moved)
   match closed {Ok()=>{return 23} Error(error)=>{return -2}}
  }
 }
}
''',expected=23,complete=True)
