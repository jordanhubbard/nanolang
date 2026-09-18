"""I test private graph owners and explicit collection before profile admission."""
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
                names=['nms_graph_edges','nms_graph_chain','nms_graph_cycles','nms_graph_promotion','nms_graph_collect','nms_graph_collect_reuse','nms_graph_collect_long_cycle']+(['nms_graph_failures','nms_graph_collect_failure'] if testing else [])
                self.run_command(['clang','--target=wasm32-unknown-unknown','-O2','-ffreestanding','-fno-builtin','-nostdlib',*flags,fixture,
                    '-Wl,--no-entry','-Wl,--max-memory=4194304',*['-Wl,--export='+name for name in names],'-o',wasm])
                for name in names:self.assertEqual(self.run_command(['wasmtime','run','--invoke',name,wasm]),'0\n')
                script="const fs=require('fs');const m=new WebAssembly.Module(fs.readFileSync(process.argv[1]));if(WebAssembly.Module.imports(m).length)throw Error('imports');for(let i=0;i<2;i++){const e=new WebAssembly.Instance(m).exports;for(let j=0;j<4;j++)for(const name of process.argv.slice(2))if(e[name]())throw Error(name);}"
                self.run_command(['node','-e',script,wasm,*names])
    def test_vm_nested_identity_and_managed_refusal(self):
        import subprocess
        with tempfile.TemporaryDirectory(prefix='nano-graph-vm-') as temp:
            work=Path(temp);source=work/'graph.nasm';module=work/'graph.nvm'
            source.write_text('.entry main\n.function main 0 3 0 int 1\n'
                'ARR_NEW 7\nDUP\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nARR_PUSH\nPOP\n'
                'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nLOAD_LOCAL 0\nEQ\nASSERT\n'
                'LOAD_LOCAL 0\nPUSH_I64 0\nPUSH_I64 1\nARR_SLICE\nSTORE_LOCAL 1\n'
                'LOAD_LOCAL 1\nLOAD_LOCAL 0\nEQ\nNOT\nASSERT\n'
                'LOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\nPUSH_I64 42\nARR_PUSH\nPOP\n'
                'LOAD_LOCAL 0\nARR_LEN\nPUSH_I64 2\nEQ\nASSERT\n'
                'PUSH_VOID\nSTORE_LOCAL 0\nPUSH_VOID\nSTORE_LOCAL 1\nPUSH_I64 0\nRET\n.end\n')
            self.run_command([ROOT/'bin/nanoisa','asm',source,'-o',module])
            self.run_command([ROOT/'bin/nano_vm',module])
            for tool in ('nvm2llvm','nvm2wasm'):
                output=work/(tool+'.old');output.write_bytes(b'prior output')
                p=subprocess.run([ROOT/'bin'/tool,module,'-o',output],capture_output=True,timeout=30)
                self.assertNotEqual(p.returncode,0);self.assertEqual(output.read_bytes(),b'prior output')
if __name__=='__main__':unittest.main()
