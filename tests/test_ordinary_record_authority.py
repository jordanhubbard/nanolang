"""I qualify checked ordinary declarations independently of nominal admission."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from tests.native_toolchain import native_cc, native_link_flags
ROOT=Path(__file__).resolve().parents[1]
class OrdinaryAuthority(unittest.TestCase):
    def command(self,args,success=True,expected=None):
        result=subprocess.run(list(map(str,args)),capture_output=True,text=True,timeout=90,cwd=ROOT)
        if expected is None:
            self.assertEqual(result.returncode==0,success,result.stdout+result.stderr)
        else:
            self.assertEqual(result.returncode,expected,result.stdout+result.stderr)
        return result
    def test_declarations_transport_and_unchanged_profiles(self):
        with tempfile.TemporaryDirectory(prefix='nano-ordinary-authority-') as temp:
            work=Path(temp)
            clang=shlex.split(os.environ.get('NANOLANG_GUARD_SAN_CC') or 'clang')
            clang_flags=shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS',''))
            link_flags=native_link_flags()
            for label,compiler,flags in [('native',native_cc(),[]),('sanitized',clang,clang_flags+['-fsanitize=address,undefined','-fno-sanitize-recover=all'])]:
                exe=work/label;module=work/(label+'.nvm')
                self.command([*compiler,*flags,'-std=c11','-O1','-Wall','-Wextra','-Werror','-Isrc/nanoisa',
                    'tests/nanoisa/test_ordinary_record_authority.c',*shlex.split(os.environ['NOA_LINK_OBJECTS']),'-lm','-lcrypto',*link_flags,'-o',exe])
                forward=work/(label+'-forward.nvm')
                owned_forward=work/(label+'-owned-forward.nvm')
                owned_prior=work/(label+'-owned-prior.nvm')
                self.assertIn('ordinary authority checks',self.command([exe,module,forward,owned_forward,owned_prior]).stdout)
                for accepted,expected in ((forward,0),(owned_prior,7)):
                    self.command([ROOT/'bin/nano_vm','--verify-only',accepted])
                    self.command([ROOT/'bin/nano_vm',accepted],expected=expected)
                    translated=accepted.with_suffix('.c');binary=accepted.with_suffix('.out')
                    self.command([ROOT/'bin/nvm2c',accepted,'-o',translated])
                    self.command([*compiler,*flags,'-std=c11','-O1','-Wall','-Wextra','-Werror',translated,'-lm',*link_flags,'-o',binary])
                    self.command([binary],expected=expected)
                self.command([ROOT/'bin/nano_vm','--verify-only',owned_forward],False)
                self.command([ROOT/'bin/nano_vm',owned_forward],False)
                output=work/'owned-prior-output';output.write_bytes(b'prior')
                self.command([ROOT/'bin/nvm2c',owned_forward,'-o',output],False)
                self.assertEqual(output.read_bytes(),b'prior')
                for tool in ('nvm2llvm','nvm2wasm'):
                    output=work/('forward.ll' if tool=='nvm2llvm' else 'forward.wasm')
                    self.command([ROOT/'bin'/tool,forward,'-o',output])
                    if tool=='nvm2wasm':
                        self.assertEqual(self.command(['wasmtime','run','--invoke','nano_entry',output]).stdout,'0\n')
                    else:
                        binary=work/'forward-llvm'
                        self.command([*clang,*clang_flags,output,*link_flags,'-o',binary])
                        self.command([binary])
                self.command([ROOT/'bin/nano_vm',module])
                for tool in ('nvm2llvm','nvm2wasm'):
                    output=work/('ordinary.ll' if tool=='nvm2llvm' else 'ordinary.wasm')
                    self.command([ROOT/'bin'/tool,module,'-o',output])
                    if tool=='nvm2wasm':
                        self.assertEqual(self.command(['wasmtime','run','--invoke','nano_entry',output]).stdout,'0\n')
                    else:
                        binary=work/'ordinary-llvm'
                        self.command([*clang,*clang_flags,output,*link_flags,'-o',binary])
                        self.command([binary])
if __name__=='__main__':unittest.main()
