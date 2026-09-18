"""I qualify private record storage without granting nominal execution authority."""
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from tests import test_managed_array_copy_runtime as copies
ROOT=Path(__file__).resolve().parents[1]
class ManagedRecords(unittest.TestCase):
    run_command=copies.ArrayCopyRuntime.run_command
    def test_private_records_native_and_wasm(self):
        with tempfile.TemporaryDirectory(prefix='nano-record-core-') as temp:
            work=Path(temp);fixture=ROOT/'tests/nanoisa/test_managed_records.c'
            for testing in (False,True):
                flags=['-DNMS_TESTING'] if testing else []
                names=['record_values','record_graphs']+(['record_failures'] if testing else [])
                native=work/'native'
                self.run_command(['clang',*shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS','')),
                    '-O1','-std=c11','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',*flags,fixture,'-o',native])
                self.run_command([native])
                wasm=work/'records.wasm'
                self.run_command(['clang','--target=wasm32-unknown-unknown','-O2','-ffreestanding','-fno-builtin','-nostdlib',*flags,fixture,
                    '-Wl,--no-entry','-Wl,--max-memory=4194304',*['-Wl,--export='+name for name in names],'-o',wasm])
                for name in names:self.assertEqual(self.run_command(['wasmtime','run','--invoke',name,wasm]),'0\n')
                script="const fs=require('fs');const m=new WebAssembly.Module(fs.readFileSync(process.argv[1]));if(WebAssembly.Module.imports(m).length)throw Error('imports');for(let i=0;i<2;i++){const e=new WebAssembly.Instance(m).exports;for(let j=0;j<3;j++)for(const name of process.argv.slice(2)){const r=e[name]();if(r)throw Error(name+':'+r);}}"
                self.run_command(['node','-e',script,wasm,*names])
    def test_separate_target_ir_linkage(self):
        with tempfile.TemporaryDirectory(prefix='nano-record-link-') as temp:
            work=Path(temp);fixture=ROOT/'tests/nanoisa/test_managed_records.c'
            for wasm in (False,True):
                flags=['--target=wasm32-unknown-unknown','-ffreestanding','-fno-builtin','-nostdlib'] if wasm else [*shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS','')),'-fsanitize=address,undefined','-fno-sanitize-recover=all']
                ir=work/'runtime.ll';output=work/('linked.wasm' if wasm else 'native')
                self.run_command(['clang',*flags,'-O1','-S','-emit-llvm',ROOT/'src/nanoisa/managed_module.c','-o',ir])
                link=['-Wl,--no-entry','-Wl,--max-memory=4194304','-Wl,--export=record_values','-Wl,--export=record_graphs'] if wasm else []
                self.run_command(['clang',*flags,'-O1','-DNMS_RECORD_LINKED',fixture,ir,*link,'-o',output])
                if wasm:
                    for name in ('record_values','record_graphs'):self.assertEqual(self.run_command(['wasmtime','run','--invoke',name,output]),'0\n')
                else:self.run_command([output])
if __name__=='__main__':unittest.main()
