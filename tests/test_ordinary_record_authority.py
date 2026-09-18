"""I qualify checked ordinary declarations independently of nominal admission."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
class OrdinaryAuthority(unittest.TestCase):
    def command(self,args,success=True):
        result=subprocess.run(list(map(str,args)),capture_output=True,text=True,timeout=90,cwd=ROOT)
        self.assertEqual(result.returncode==0,success,result.stdout+result.stderr)
        return result
    def test_declarations_transport_and_unchanged_profiles(self):
        with tempfile.TemporaryDirectory(prefix='nano-ordinary-authority-') as temp:
            work=Path(temp)
            for compiler,flags in [('cc',[]),('clang',shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS',''))+['-fsanitize=address,undefined','-fno-sanitize-recover=all'])]:
                exe=work/compiler;module=work/(compiler+'.nvm')
                self.command([compiler,*flags,'-std=c11','-O1','-Wall','-Wextra','-Werror','-Isrc/nanoisa',
                    'tests/nanoisa/test_ordinary_record_authority.c',*shlex.split(os.environ['NOA_LINK_OBJECTS']),'-lm','-lcrypto','-o',exe])
                self.assertIn('ordinary authority checks',self.command([exe,module]).stdout)
                self.command([ROOT/'bin/nano_vm',module])
                for tool in ('nvm2llvm','nvm2wasm'):
                    output=work/'prior';output.write_bytes(b'prior')
                    self.command([ROOT/'bin'/tool,module,'-o',output],False)
                    self.assertEqual(output.read_bytes(),b'prior')
if __name__=='__main__':unittest.main()
