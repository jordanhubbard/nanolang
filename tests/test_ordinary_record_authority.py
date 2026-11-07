"""I qualify checked ordinary declarations independently of nominal admission."""
import os
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile
import unittest
import zlib
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
            for compiler,flags in [('cc',[]),('clang',shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS',''))+['-fsanitize=address,undefined','-fno-sanitize-recover=all'])]:
                exe=work/compiler;module=work/(compiler+'.nvm')
                self.command([compiler,*flags,'-std=c11','-O1','-Wall','-Wextra','-Werror','-Isrc/nanoisa',
                    'tests/nanoisa/test_ordinary_record_authority.c',*shlex.split(os.environ['NOA_LINK_OBJECTS']),'-lm','-lcrypto','-o',exe])
                forward=work/(compiler+'-forward.nvm')
                owned_forward=work/(compiler+'-owned-forward.nvm')
                owned_prior=work/(compiler+'-owned-prior.nvm')
                ordinary_forward=work/(compiler+'-ordinary-owned-forward.nvm')
                ordinary_prior=work/(compiler+'-ordinary-owned-prior.nvm')
                self.assertIn('ordinary authority checks',self.command([
                    exe,module,forward,owned_prior,ordinary_forward,ordinary_prior]).stdout)
                # The real producer refuses a resource-forward declaration before
                # serialization. I forge only its authority bits, retaining a valid
                # checksum, to exercise the public consumers' independent refusal.
                raw=bytearray(ordinary_forward.read_bytes())
                header,count=struct.unpack_from('<II',raw,20)
                for index in range(count):
                    kind,_,offset,_=struct.unpack_from('<IIQQ',raw,header+24*index)
                    if kind==13:
                        self.assertEqual(raw[offset+8:offset+10],b'\x01\x01')
                        raw[offset+8:offset+10]=b'\x03\x03'
                        break
                else:
                    self.fail('I require the emitted ownership section.')
                struct.pack_into('<I',raw,36,zlib.crc32(raw[header:]))
                owned_forward.write_bytes(raw)
                for accepted,expected in ((forward,0),(owned_prior,7)):
                    self.command([ROOT/'bin/nano_vm','--verify-only',accepted])
                    self.command([ROOT/'bin/nano_vm',accepted],expected=expected)
                    translated=accepted.with_suffix('.c');binary=accepted.with_suffix('.out')
                    self.command([ROOT/'bin/nvm2c',accepted,'-o',translated])
                    self.command([compiler,*flags,'-std=c11','-O1','-Wall','-Wextra','-Werror',translated,'-lm','-o',binary])
                    self.command([binary],expected=expected)
                for refused in (owned_forward,ordinary_forward,ordinary_prior):
                    for arguments in (['--verify-only',refused],[refused]):
                        result=self.command([ROOT/'bin/nano_vm',*arguments],False)
                        self.assertGreater(result.returncode,0,result.stdout+result.stderr)
                        self.assertNotIn('checksum',result.stderr.lower())
                    output=work/'owned-prior-output';output.write_bytes(b'prior')
                    result=self.command([ROOT/'bin/nvm2c',refused,'-o',output],False)
                    self.assertGreater(result.returncode,0,result.stdout+result.stderr)
                    self.assertNotIn('checksum',result.stderr.lower())
                    self.assertEqual(output.read_bytes(),b'prior')
                for tool in ('nvm2llvm','nvm2wasm'):
                    output=work/('forward.ll' if tool=='nvm2llvm' else 'forward.wasm')
                    self.command([ROOT/'bin'/tool,forward,'-o',output])
                    if tool=='nvm2wasm':
                        self.assertEqual(self.command(['wasmtime','run','--invoke','nano_entry',output]).stdout,'0\n')
                    else:
                        binary=work/'forward-llvm'
                        self.command(['clang',*shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS','')),output,'-o',binary])
                        self.command([binary])
                self.command([ROOT/'bin/nano_vm',module])
                for tool in ('nvm2llvm','nvm2wasm'):
                    output=work/('ordinary.ll' if tool=='nvm2llvm' else 'ordinary.wasm')
                    self.command([ROOT/'bin'/tool,module,'-o',output])
                    if tool=='nvm2wasm':
                        self.assertEqual(self.command(['wasmtime','run','--invoke','nano_entry',output]).stdout,'0\n')
                    else:
                        binary=work/'ordinary-llvm'
                        self.command(['clang',*shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS','')),output,'-o',binary])
                        self.command([binary])
if __name__=='__main__':unittest.main()
