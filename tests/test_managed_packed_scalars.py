"""I compare portable packed storage with ordinary VM scalar reference values."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
class PackedScalars(unittest.TestCase):
    def run_command(self,args):
        p=subprocess.run(list(map(str,args)),capture_output=True,text=True,timeout=60)
        self.assertEqual(p.returncode,0,str(args)+'\n'+p.stdout+p.stderr)
        return p
    def test_vm_native_llvm_and_wasm_core(self):
        with tempfile.TemporaryDirectory(prefix='nano-packed-core-') as temp:
            work=Path(temp);native_flags=shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS',''))
            reference=work/'reference'
            self.run_command(['clang',*native_flags,'-std=c11','-O1','-D_GNU_SOURCE',
                              '-fsanitize=address,undefined','-fno-sanitize-recover=all',
                              ROOT/'tests/nanovm/packed_scalar_reference.c',
                              *[ROOT/f'src/nanovm/{name}.c' for name in ('heap','heap_cycles','value')],ROOT/'src/nanoisa/isa.c','-o',reference])
            vectors=self.run_command([reference]).stdout
            self.assertEqual(len(vectors.splitlines()),563)
            (work/'packed_vectors.h').write_text(vectors)
            core=ROOT/'src/nanoisa/managed_strings.c';fixture=ROOT/'tests/nanoisa/test_managed_packed_scalars.c'
            for testing in (False,True):
                options=['-DNMS_TESTING'] if testing else []
                ir=work/'native.ll';exe=work/'native'
                self.run_command(['clang',*native_flags,'-std=c11','-O1','-Wall','-Wextra','-Werror',
                                  '-fsanitize=address,undefined','-fno-sanitize-recover=all',*options,
                                  '-S','-emit-llvm',core,'-o',ir])
                self.run_command(['opt','-passes=verify','-disable-output',ir])
                self.run_command(['clang',*native_flags,'-fsanitize=address,undefined','-fno-sanitize-recover=all',
                                  *options,'-I'+str(work),ir,fixture,'-o',exe]);self.run_command([exe])
                ir=work/'wasm.ll';wasm=work/'core.wasm'
                flags=['--target=wasm32-unknown-unknown','-std=c11','-O2','-ffreestanding','-fno-builtin',*options]
                self.run_command(['clang',*flags,'-S','-emit-llvm',core,'-o',ir]);self.run_command(['opt','-passes=verify','-disable-output',ir])
                names=['nms_packed_values','nms_packed_lifecycle']+(['nms_packed_failures','nms_packed_reuse'] if testing else [])
                self.run_command(['clang',*flags,'-nostdlib','-I'+str(work),ir,fixture,'-Wl,--no-entry',
                                  '-Wl,--max-memory=1048576',*['-Wl,--export='+n for n in names],'-o',wasm])
                script='''const fs=require('fs');const m=new WebAssembly.Module(fs.readFileSync(process.argv[1]));
if(WebAssembly.Module.imports(m).length)throw Error('imports');
for(let i=0;i<2;i++){const e=new WebAssembly.Instance(m).exports;
for(let r=0;r<3;r++)for(const n of JSON.parse(process.argv[2])){
const result=e[n]();if(result)throw Error(n+': line '+result);}}'''
                self.run_command(['node','-e',script,wasm,json.dumps(names)])
                for name in names:self.assertEqual(self.run_command(['wasmtime','run','--invoke',name,wasm]).stdout,'0\n')
if __name__=='__main__':unittest.main()
