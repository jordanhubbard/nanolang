"""I qualify descriptive identities independently from storage authority."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
class RecordPlan(unittest.TestCase):
    def run_command(self,args,success=True):
        p=subprocess.run(list(map(str,args)),capture_output=True,text=True,timeout=60)
        self.assertEqual(p.returncode==0,success,p.stdout+p.stderr);return p
    def test_owned_plans_limits_roundtrip_and_runtime_refusal(self):
        with tempfile.TemporaryDirectory(prefix='nano-record-plan-') as temp:
            work=Path(temp)
            for compiler,flags in [('cc',[]),('clang',shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS',''))+['-fsanitize=address,undefined','-fno-sanitize-recover=all'])]:
                executable=work/compiler;module=work/(compiler+'.nvm')
                common=[compiler,*flags,'-std=c11','-O1','-Wall','-Wextra','-Werror','-Isrc/nanoisa']
                objects=[]
                for name in ('retained_layouts','nvm_v2_layouts','nvm_v2_cursor'):
                    obj=work/(compiler+'-'+name+'.o');objects.append(obj)
                    self.run_command([*common,'-Dcalloc=nrp_test_calloc','-c','src/nanoisa/'+name+'.c','-o',obj])
                self.run_command([*common,'tests/nanoisa/test_managed_record_plan.c',*objects,
                    *shlex.split(os.environ['NRP_LINK_OBJECTS']),'-lm','-lcrypto','-o',executable])
                result=self.run_command([executable,module]);self.assertIn('record plan checks passed',result.stdout)
                self.run_command([ROOT/'bin/nano_vm',module])
                for tool in ('nvm2llvm','nvm2wasm'):
                    output=work/'prior';output.write_bytes(b'prior output')
                    self.run_command([ROOT/'bin'/tool,module,'-o',output],success=False)
                    self.assertEqual(output.read_bytes(),b'prior output')
if __name__=='__main__':unittest.main()
