"""I test private graph owners and explicit collection before profile admission."""
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from tests import test_managed_array_copy_runtime as copies
ROOT=Path(__file__).resolve().parents[1]
class GraphSafepoints(unittest.TestCase):
    run_command=copies.ArrayCopyRuntime.run_command
    def test_prepared_collection_targets(self):
        with tempfile.TemporaryDirectory(prefix='nano-graph-core-') as temp:
            work=Path(temp);fixture=ROOT/'tests/nanoisa/test_managed_graph_safepoints.c'
            for testing in (False,True):
                flags=['-DNMS_TESTING'] if testing else []
                exe=work/'native'
                self.run_command(['clang',*shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS','')),
                    '-O1','-std=c11','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',*flags,fixture,'-o',exe])
                self.run_command([exe])
                wasm=work/'graph.wasm'
                names=['nms_prepared_roots','nms_prepared_module','nms_prepared_pressure']+(['nms_prepared_failures'] if testing else [])
                self.run_command(['clang','--target=wasm32-unknown-unknown','-O2','-ffreestanding','-fno-builtin','-nostdlib',*flags,fixture,
                    '-Wl,--no-entry','-Wl,--max-memory=4194304',*['-Wl,--export='+name for name in names],'-o',wasm])
                for name in names:self.assertEqual(self.run_command(['wasmtime','run','--invoke',name,wasm]),'0\n')
                script="const fs=require('fs');const m=new WebAssembly.Module(fs.readFileSync(process.argv[1]));if(WebAssembly.Module.imports(m).length)throw Error('imports');for(let i=0;i<2;i++){const e=new WebAssembly.Instance(m).exports;for(let j=0;j<4;j++)for(const name of process.argv.slice(2))if(e[name]())throw Error(name);}"
                self.run_command(['node','-e',script,wasm,*names])
    def test_separate_production_target_ir_linkage(self):
        with tempfile.TemporaryDirectory(prefix='nano-graph-link-') as temp:
            work=Path(temp);module=ROOT/'src/nanoisa/managed_module.c'
            fixture=ROOT/'tests/nanoisa/test_managed_graph_linkage.c'
            native_flags=shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS',''))
            sanitizer=['-fsanitize=address,undefined','-fno-sanitize-recover=all']
            ir=work/'runtime.ll';exe=work/'native'
            self.run_command(['clang',*native_flags,'-O1',*sanitizer,'-S','-emit-llvm',module,'-o',ir])
            self.run_command(['clang',*native_flags,'-O1',*sanitizer,ir,fixture,'-o',exe])
            self.run_command([exe])
            wasm_ir=work/'runtime-wasm.ll';wasm=work/'runtime.wasm'
            flags=['--target=wasm32-unknown-unknown','-O2','-ffreestanding','-fno-builtin','-nostdlib']
            self.run_command(['clang',*flags,'-S','-emit-llvm',module,'-o',wasm_ir])
            self.run_command(['clang',*flags,wasm_ir,fixture,'-Wl,--no-entry','-Wl,--max-memory=1048576',
                '-Wl,--export=graph_entry','-Wl,--export=graph_dispose','-o',wasm])
            self.assertEqual(self.run_command(['wasmtime','run','--invoke','graph_entry',wasm]),'0\n')
            script="const fs=require('fs');const m=new WebAssembly.Module(fs.readFileSync(process.argv[1]));if(WebAssembly.Module.imports(m).length)throw Error('imports');for(let i=0;i<2;i++){const e=new WebAssembly.Instance(m).exports;for(let j=0;j<3;j++){const r=e.graph_entry();if(r)throw Error('entry '+r);}if(e.graph_dispose())throw Error('dispose');}"
            self.run_command(['node','-e',script,wasm])
if __name__=='__main__':unittest.main()
