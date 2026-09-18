"""I retain exact source arithmetic observations with explicit compiler identity."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
class ArithmeticSource(unittest.TestCase):
    def setUp(self):
        self.work=Path(tempfile.mkdtemp(prefix='nano-arithmetic-source-'))
        print('I retain source arithmetic artifacts at',self.work,flush=True)
    def command(self,*args):
        args=list(map(str,args))
        result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=240)
        self.assertEqual(result.returncode,0,f'compiler/tool={args[0]} command={args!r}\n{result.stdout}\n{result.stderr}')
        return result
    def routes(self,source,scalar=False,callback=False,legacy_names=('nanoc_c','nanoc_stage1','nanoc_stage2')):
        self.command(ROOT/'bin/nano',source)
        for name in legacy_names:
            with self.subTest(legacy=name):
                exe=self.work/(name+'-legacy')
                self.command(ROOT/'bin'/name,source,'-o',exe)
                self.command(exe)
        for name in ('nano_virt','nanoc_stage1','nanoc_stage2'):
            with self.subTest(canonical=name):
                module=self.work/(name+'.nvm')
                self.command(ROOT/'bin'/name,source,'--emit-nvm','-o',module)
                self.command(ROOT/'bin/nano_vm','--verify-only',module)
                self.command(ROOT/'bin/nano_vm',module)
                assembly=self.command(ROOT/'bin/nanoisa','dump',module).stdout
                self.assertIn('F64_',assembly)
                (self.work/(name+'.nasm')).write_text(assembly)
                c=self.work/(name+'.c');exe=self.work/(name+'-native')
                if callback and name=='nano_virt':
                    c.write_text('retained')
                    args=[ROOT/'bin/nvm2c',module,'-o',c]
                    refused=subprocess.run(list(map(str,args)),cwd=ROOT,capture_output=True,text=True,timeout=60)
                    self.assertGreater(refused.returncode,0,str(args))
                    self.assertIn('unsupported opcode FUNCREF',refused.stderr)
                    self.assertEqual(c.read_text(),'retained')
                    continue
                self.command(ROOT/'bin/nvm2c',module,'-o',c)
                self.command(os.environ.get('CC','cc'),'-std=c11','-O2','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',c,'-lm','-o',exe)
                self.command(exe)
                if scalar:
                    ir=self.work/(name+'.ll'); wasm=self.work/(name+'.wasm')
                    self.command(ROOT/'bin/nvm2llvm',module,'-o',ir)
                    self.command('lli',ir)
                    optimized=self.work/(name+'-opt.bc')
                    self.command('opt','-O2',ir,'-o',optimized)
                    self.command('lli',optimized)
                    self.command(ROOT/'bin/nvm2wasm',module,'-o',wasm)
                    self.assertEqual(self.command('wasmtime','run','--invoke','nano_entry',wasm).stdout,'0\n')
                    self.command('node','-e','const fs=require("fs");const m=new WebAssembly.Module(fs.readFileSync(process.argv[1]));if(WebAssembly.Module.imports(m).length)process.exit(2);if(new WebAssembly.Instance(m).exports.nano_entry()!==0)process.exit(3);',wasm)
    def test_generated_runtime_provider_identity(self):
        self.command('python3',ROOT/'scripts/embed_binary64_arithmetic.py','--check')
        header=(ROOT/'src/binary64_arithmetic.h').read_text()
        source=ROOT/'tests/nanoisa/fixtures/binary64_arithmetic.nano'
        for name in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
            output=self.work/(name+'-legacy.c')
            if name=='nanoc_c':
                executable=self.work/'seed-native'
                self.command(ROOT/'bin'/name,source,'--keep-c','-o',executable)
                output=Path(str(executable)+'.c')
            else:
                self.command(ROOT/'bin'/name,source,'--target','c','-o',output)
            self.assertIn(header,output.read_text(),name)
    def test_exact_scalar_arithmetic(self):
        self.routes(ROOT/'tests/nanoisa/fixtures/binary64_arithmetic.nano',scalar=True)
    def test_ordered_global_and_operand_evaluation(self):
        source=self.work/'globals.nano'
        source.write_text(GLOBALS)
        self.routes(source)
    def test_map_reduce_supported_routes(self):
        # I retain failed selfhost legacy reduce ABI acceptance under task_d0997.
        # This subset does not execute that route or claim full callback parity.
        source=self.work/'callbacks.nano'
        source.write_text(CALLBACKS)
        self.routes(source,callback=True,legacy_names=('nanoc_c',))
    def test_direct_reduce_observer_exact_result(self):
        # I use the newly qualified fixture; retained prior failed inputs stay untouched.
        from test_scalar_reduce_source import FLOAT
        source=self.work/'reduce-exact-result.nano'
        source.write_text(FLOAT)
        for name in ('nanoc_stage1','nanoc_stage2'):
            output=self.work/(name+'.nvm')
            self.command(ROOT/'bin'/name,source,'--emit-nvm','-o',output)
            self.command(ROOT/'bin/nano_vm','--verify-only',output)
            self.command(ROOT/'bin/nano_vm',output)
GLOBALS='''let mut calls:int = 0
fn operand(id:int,x:float)->float { set calls (+ (* calls 10) id) return x }
shadow operand { let saved:int=calls set calls 0 assert (== (operand 2 2.0) 2.0) assert (== calls 2) set calls saved }
let first:float = (+ (operand 1 1.0) (operand 2 2.0))
let mut second:float = (/ (+ first 1.0) 0.0)
let third:float = (+ first second)
let negative:float = (- third)
fn main()->int {
    assert (== first 3.0)
    assert (== third 3.0)
    assert (== negative -3.0)
    assert (== calls 12)
    set calls 0
    assert (== (+ (operand 4 4.0) (operand 5 5.0)) 9.0)
    assert (== calls 45)
    set second (+ third 2.0)
    assert (== second 5.0)
    return 0
}
shadow main { let saved:int=calls let value:float=second set calls 12 assert (== (main) 0) set calls saved set second value }
'''
CALLBACKS='''fn zero(x:float)->float { return (/ x 0.0) }
shadow zero { assert (== (float_to_bits (zero 1.0)) 0) }
fn combine(a:float,b:float)->float { return (+ a b) }
shadow combine { assert (== (combine 1.0 2.0) 3.0) }
fn main()->int {
    let nan:float=(float_from_bits -4503599627370430)
    let values:array<float> =[nan,1.0]
    let mapped:array<float> =(map values zero)
    assert (== (float_to_bits (at mapped 0)) 0)
    assert (== (float_to_bits (at mapped 1)) 0)
    let result:float = (reduce values 0.0 combine)
    assert (== (float_to_bits result) 9221120237041090560)
    return 0
}
shadow main { assert (== (main) 0) }
'''
if __name__=='__main__':unittest.main()
