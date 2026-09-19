"""I qualify private record module ownership and target ABI, without admission."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from scripts.embed_managed_runtime import generate
ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'tests/nanoisa/test_managed_record_adapters.c'
class RecordAdapters(unittest.TestCase):
    def command(self,args):
        result=subprocess.run(list(map(str,args)),cwd=ROOT,capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,str(args)+'\n'+result.stdout+result.stderr)
        return result.stdout
    def wasm_controls(self,path,names):
        for name in names:
            self.assertEqual(self.command(['wasmtime','run','--invoke',name,path]),'0\n')
        script="const fs=require('fs');const m=new WebAssembly.Module(fs.readFileSync(process.argv[1]));if(WebAssembly.Module.imports(m).length)throw Error('imports');for(let i=0;i<3;i++)for(const name of process.argv.slice(2)){const e=new WebAssembly.Instance(m).exports;const r=e[name]();if(r)throw Error(name+':'+r);}"
        self.command(['node','-e',script,path,*names])
    def test_private_adapters_native_and_wasm(self):
        clang=shlex.split(os.environ.get('NMS_RUNTIME_CLANG','clang'))
        native_flags=shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS',''))
        with tempfile.TemporaryDirectory(prefix='nano-record-adapters-') as temp:
            work=Path(temp)
            for testing in (False,True):
                flags=['-DNMS_TESTING'] if testing else []
                for compiler,extra in [(shlex.split(os.environ.get('CC','cc')),[]),(clang,native_flags)]:
                    exe=work/'native'
                    self.command([*compiler,*extra,'-std=c11','-O1','-Wall','-Wextra','-Werror',
                        '-fsanitize=address,undefined','-fno-sanitize-recover=all',*flags,FIXTURE,'-o',exe])
                    self.command([exe])
                names=['record_adapter_values','record_adapter_lifecycle']+(['record_adapter_failures'] if testing else [])
                wasm=work/'adapters.wasm'
                self.command([*clang,'--target=wasm32-unknown-unknown','-O2','-ffreestanding','-fno-builtin','-nostdlib',
                    *flags,FIXTURE,'-Wl,--no-entry','-Wl,--max-memory=4194304',
                    *['-Wl,--export='+n for n in names],'-o',wasm])
                self.wasm_controls(wasm,names)
    def test_packaged_production_scalar_abi(self):
        clang=shlex.split(os.environ.get('NMS_RUNTIME_CLANG','clang'))
        opt=shlex.split(os.environ.get('NMS_RUNTIME_OPT','opt'))
        header,manifest,variants=generate(clang,opt)
        self.assertNotIn('nms_test_',header)
        self.assertIn('src/nanoisa/managed_module.c',manifest['sources'])
        with tempfile.TemporaryDirectory(prefix='nano-record-adapter-link-') as temp:
            work=Path(temp)
            for target,variant in variants.items():
                ir=work/(target+'.ll');ir.write_text(variant['ir'])
                if target=='native':
                    exe=work/'native'
                    self.command([*clang,*shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS','')),
                        '-std=c11','-O1','-Wall','-Wextra','-Werror','-fsanitize=address,undefined',
                        '-fno-sanitize-recover=all','-DNMS_RECORD_ADAPTER_LINKED',FIXTURE,ir,'-o',exe])
                    self.command([exe])
                else:
                    wasm=work/'linked.wasm'
                    self.command([*clang,'--target=wasm32-unknown-unknown','-O2','-ffreestanding','-fno-builtin','-nostdlib',
                        '-DNMS_RECORD_ADAPTER_LINKED',FIXTURE,ir,'-Wl,--no-entry','-Wl,--max-memory=4194304',
                        '-Wl,--export=record_adapter_values','-o',wasm])
                    self.wasm_controls(wasm,['record_adapter_values'])
if __name__=='__main__':unittest.main()
