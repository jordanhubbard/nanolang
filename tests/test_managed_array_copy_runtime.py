"""I qualify the private array-copy checkpoint without changing instruction admission."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from scripts.embed_managed_runtime import generate
ROOT=Path(__file__).resolve().parents[1]
class ArrayCopyRuntime(unittest.TestCase):
    def run_command(self,args):
        p=subprocess.run(list(map(str,args)),capture_output=True,text=True,timeout=60)
        self.assertEqual(p.returncode,0,str(args)+'\n'+p.stdout+p.stderr);return p.stdout
    def test_core_policy_ownership_and_failure_targets(self):
        with tempfile.TemporaryDirectory(prefix='nano-mutable-runtime-') as temp:
            work=Path(temp);fixture=ROOT/'tests/nanoisa/test_managed_array_copy.c'
            for testing in (False,True):
                options=['-DNMS_TESTING'] if testing else []
                exe=work/'native'
                self.run_command(['clang',*shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS','')),
                    '-std=c11','-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all',*options,fixture,'-o',exe])
                self.run_command([exe])
                wasm=work/'runtime.wasm'
                names=['nms_copy_core','nms_copy_aliases','nms_copy_abi']+(['nms_copy_failures','nms_copy_pressure'] if testing else [])
                self.run_command(['clang','--target=wasm32-unknown-unknown','-O2','-ffreestanding','-fno-builtin',
                    '-nostdlib',*options,fixture,'-Wl,--no-entry','-Wl,--max-memory=1048576',
                    *['-Wl,--export='+n for n in names],'-o',wasm])
                for name in names:self.assertEqual(self.run_command(['wasmtime','run','--invoke',name,wasm]),'0\n')
                script="""const fs=require('fs');const m=new WebAssembly.Module(fs.readFileSync(process.argv[1]));
if(WebAssembly.Module.imports(m).length)throw Error('imports');
for(let i=0;i<2;i++){const e=new WebAssembly.Instance(m).exports;
for(let r=0;r<3;r++){if((e.nms_copy_core() || e.nms_copy_aliases()))throw Error('core');
if(e.nms_copy_failures && e.nms_copy_failures())throw Error('failure');
if(e.nms_copy_pressure && e.nms_copy_pressure())throw Error('pressure');
}
if(e.nms_copy_abi())throw Error('ABI');}"""
                self.run_command(['node','-e',script,wasm])
    def test_packaged_scalar_output_abi(self):
        _,_,variants=generate(['clang'],['opt'])
        app='''
@payloads = internal constant [3 x i64] [i64 10, i64 20, i64 30]
@tags = internal constant [3 x i32] [i32 1, i32 1, i32 1]
define i32 @adapter_check() {
 %bits = alloca i64, align 8
 %tag = alloca i32, align 4
 %begin = call i32 @nms_module_begin(ptr null, i32 0)
 %array = call i64 @nms_module_array_literal(i32 1, i32 3, ptr @payloads, ptr @tags)
 %copy = call i64 @nms_module_array_slice(i64 %array, i64 4294967297, i32 1, i64 -1, i32 1)
 call void @nms_module_release(i64 %array, i32 7)
 %get = call i32 @nms_module_array_get_value(i64 %copy, i64 0, ptr %bits, ptr %tag)
 %b = load i64, ptr %bits
 %t = load i32, ptr %tag
 %bv = icmp eq i64 %b, 20
 %tv = icmp eq i32 %t, 1
 %valid = and i1 %bv, %tv
 call void @nms_module_release(i64 %copy, i32 7)
 %finish = call i64 @nms_module_finish(i32 0)
 %expected = icmp eq i64 %finish, 0
 %dispose = call i32 @nms_module_dispose()
 %d = icmp eq i32 %dispose, 0
 %a = and i1 %valid, %expected
 %ok = and i1 %a, %d
 %answer = select i1 %ok, i32 0, i32 1
 ret i32 %answer
}
'''
        with tempfile.TemporaryDirectory(prefix='nano-mutable-abi-') as temp:
            work=Path(temp)
            for target in ('native','wasm32'):
                ir=work/(target+'.ll');ir.write_text(variants[target]['ir']+app+('define i32 @main() { %v = call i32 @adapter_check()\n ret i32 %v\n}\n' if target=='native' else ''))
                self.run_command(['opt','-passes=verify','-disable-output',ir])
                if target=='native':
                    exe=work/'native';self.run_command(['clang',*shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS','')),ir,'-o',exe]);self.run_command([exe])
                else:
                    wasm=work/'adapter.wasm';self.run_command(['clang','--target=wasm32-unknown-unknown','-nostdlib',ir,'-Wl,--no-entry','-Wl,--export=adapter_check','-o',wasm])
                    self.assertEqual(self.run_command(['wasmtime','run','--invoke','adapter_check',wasm]),'0\n')
                    self.run_command(['node','-e',"const fs=require('fs');const m=new WebAssembly.Module(fs.readFileSync(process.argv[1]));if(WebAssembly.Module.imports(m).length)throw Error('imports');for(let i=0;i<2;i++)if(new WebAssembly.Instance(m).exports.adapter_check())throw Error('ABI');",wasm])
if __name__=='__main__':unittest.main()
