"""I test private graph owners before cycle collection or profile admission."""
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from tests import test_managed_array_copy_runtime as copies
ROOT=Path(__file__).resolve().parents[1]
class ArrayGraphs(unittest.TestCase):
    run_command=copies.ArrayCopyRuntime.run_command
    def test_graph_edges_and_iterative_release_targets(self):
        with tempfile.TemporaryDirectory(prefix='nano-graph-core-') as temp:
            work=Path(temp);fixture=ROOT/'tests/nanoisa/test_managed_array_graphs.c'
            for testing in (False,True):
                flags=['-DNMS_TESTING'] if testing else []
                exe=work/'native'
                self.run_command(['clang',*shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS','')),
                    '-O1','-std=c11','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',*flags,fixture,'-o',exe])
                self.run_command([exe])
                wasm=work/'graph.wasm'
                names=['nms_graph_edges','nms_graph_chain','nms_graph_cycles','nms_graph_promotion']+(['nms_graph_failures'] if testing else [])
                self.run_command(['clang','--target=wasm32-unknown-unknown','-O2','-ffreestanding','-fno-builtin','-nostdlib',*flags,fixture,
                    '-Wl,--no-entry','-Wl,--max-memory=4194304',*['-Wl,--export='+name for name in names],'-o',wasm])
                for name in names:self.assertEqual(self.run_command(['wasmtime','run','--invoke',name,wasm]),'0\n')
                script="const fs=require('fs');const m=new WebAssembly.Module(fs.readFileSync(process.argv[1]));if(WebAssembly.Module.imports(m).length)throw Error('imports');for(let i=0;i<2;i++){const e=new WebAssembly.Instance(m).exports;for(let j=0;j<4;j++)for(const name of process.argv.slice(2))if(e[name]())throw Error(name);}"
                self.run_command(['node','-e',script,wasm,*names])
if __name__=='__main__':unittest.main()
