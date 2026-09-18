"""I qualify paired declaration facts without admitting nominal LLVM execution."""
from pathlib import Path
import os
import re
import struct
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
POSITIVE='''struct Leaf { text: string }
struct Parent { child: Leaf }
struct Empty {}
struct Twin { text: string }
fn length(value: Parent) -> int { return (str_length value.child.text) }
shadow length { let p: Parent = Parent { child: Leaf { text: "abc" } } assert (== (length p) 3) }
fn main() -> int { let p: Parent = Parent { child: Leaf { text: "abc" } } assert (== (length p) 3) return 0 }
shadow main { assert true }
'''
class OrdinaryProducers(unittest.TestCase):
    @classmethod
    def command(cls,*args,ok=True,timeout=900):
        p=subprocess.run(list(map(str,args)),cwd=ROOT,capture_output=True,text=True,timeout=timeout)
        if (p.returncode==0)!=ok:raise AssertionError(f'{args}: {p.returncode}\n{p.stdout}\n{p.stderr}')
        return p.stdout
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='nano-ordinary-producers-')
        cls.work=Path(cls.temp.name);cls.emitters=[];cls.shadows=[]
        driver=cls.work/'shadows.nano'
        driver.write_text((ROOT/'tests/nanoisa/fixtures/shadow_module_driver.nano.txt').read_text())
        for compiler in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
            emitter=cls.work/(compiler+'-emit');shadow=cls.work/(compiler+'-shadow')
            cls.command(ROOT/'bin'/compiler,ROOT/'src_nano/nanoisa_emit.nano','-o',emitter)
            cls.command(ROOT/'bin'/compiler,driver,'-o',shadow)
            cls.emitters.append(emitter);cls.shadows.append(shadow)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def source(self,text):
        p=self.work/'case.nano';p.write_text(text);return p
    def dump(self,module):return self.command(ROOT/'bin/nanoisa','dump',module)
    def facts(self,text,key):
        return bytes.fromhex(''.join(re.findall(r'^\.'+key+r' "([0-9a-f]*)"$',text,re.M)))
    def qualify(self,module,ordinary=True):
        self.command(ROOT/'bin/nano_vm','--verify-only',module)
        self.command(ROOT/'bin/nano_vm',module)
        dump=self.dump(module);layouts=self.facts(dump,'layouts');ownership=self.facts(dump,'ownership')
        if ordinary:
            self.assertTrue(layouts);self.assertTrue(ownership)
            count=struct.unpack_from('<I',layouts)[0]
            self.assertEqual(struct.unpack_from('<II',ownership),(1,count))
            self.assertEqual(ownership[8:8+count],b'\x01'*count)
            at=8+((count+3)&~3);functions=struct.unpack_from('<I',ownership,at)[0];at+=4
            headers=re.findall(r'^\.function \S+ (\d+) (\d+) (\d+) (\w+) (\d+)',dump,re.M)
            self.assertEqual(functions,len(headers))
            tags={'void':0,'int':1,'u8':2,'float':3,'bool':4,'string':5,'struct':8}
            parameters={int(index):values.split() for index,values in re.findall(r'^\.parameters (\d+)(.*)$',dump,re.M)}
            for function,(params,locals_,_,result,results) in enumerate(headers):
                slots,arity=struct.unpack_from('<HH',ownership,at);at+=4
                self.assertEqual((slots,arity),(int(locals_),int(params)))
                for i in range(slots+1):
                    tag,mode,reserved,identity=struct.unpack_from('<BBHI',ownership,at);at+=8
                    self.assertEqual((mode,reserved,identity),(0,0,0xffffffff))
                    self.assertIn(tag,(0,1,2,3,4,5,8))
                    if i:self.assertNotEqual(tag,0)
                    else:self.assertEqual(tag,tags[result] if int(results) else 0)
                    if 0<i<=arity:self.assertEqual(tag,tags[parameters[function][i-1]])
            self.assertEqual(at,len(ownership))
            asm=self.work/'roundtrip.nasm';copy=self.work/'roundtrip.nvm';asm.write_text(dump)
            self.command(ROOT/'bin/nanoisa','asm',asm,'-o',copy)
            again=self.dump(copy)
            self.assertEqual(self.facts(again,'layouts'),layouts)
            self.assertEqual(self.facts(again,'ownership'),ownership)
            for tool in ('nvm2llvm','nvm2wasm'):
                prior=self.work/'prior.out';prior.write_bytes(b'prior')
                self.command(ROOT/'bin'/tool,module,'-o',prior,ok=False)
                self.assertEqual(prior.read_bytes(),b'prior')
        else:
            self.assertEqual(layouts,b'');self.assertEqual(ownership,b'')
        return layouts
    def modules(self,text):
        source=self.source(text);module=self.work/'seed.nvm'
        self.command(ROOT/'bin/nano_virt',source,'--emit-nvm','-o',module)
        yield module
        for index,emitter in enumerate(self.emitters):
            asm=self.work/f'emit{index}.nasm';module=self.work/f'emit{index}.nvm'
            self.command(emitter,source,'-o',asm)
            self.command(ROOT/'bin/nanoisa','asm',asm,'-o',module)
            yield module
    def test_nested_empty_distinct_records_and_all_stages(self):
        facts=[self.qualify(module) for module in self.modules(POSITIVE)]
        self.assertTrue(all(item==facts[0] for item in facts))
    def test_initializer_and_scalar_fields(self):
        source='''struct Scalars { i: int, f: float, b: bool }
let initial: Scalars = Scalars { i: 7, f: 2.5, b: true }
fn main() -> int { assert (== initial.i 7) assert (== initial.f 2.5) assert initial.b return 0 }
shadow main { assert true }
'''
        for module in self.modules(source):self.qualify(module)
    def test_all_selected_shadows_keep_authority(self):
        source=self.source(POSITIVE)
        text=self.command(ROOT/'obj/borrow_shadow_names',source)
        asm=self.work/'c-shadow.nasm';module=self.work/'c-shadow.nvm';asm.write_text(text)
        self.assertIn('$shadow_0_length',text);self.assertIn('$shadow_1_main',text)
        self.command(ROOT/'bin/nanoisa','asm',asm,'-o',module)
        self.qualify(module)
        for i,driver in enumerate(self.shadows):
            asm=self.work/f'shadow{i}.nasm';module=self.work/f'shadow{i}.nvm'
            text=self.command(driver,source,0,'raw');asm.write_text(text)
            self.assertIn('__nanoisa_shadow_0',text);self.assertIn('__nanoisa_shadow_1',text)
            self.command(ROOT/'bin/nanoisa','asm',asm,'-o',module)
            self.qualify(module)
    def test_nested_lexical_slots_keep_exact_tags(self):
        source='''struct Leaf { text: string }
fn main() -> int { let value: int = 1 if true { let value: string = "a" assert (== value "a") } assert (== value 1) return 0 }
shadow main { assert true }
'''
        for module in self.modules(source):self.qualify(module)
    def test_unsupported_array_field_keeps_ordinary_execution(self):
        source='''struct Data { values: array<int> }
fn main() -> int { let value: Data = Data { values: [1, 2] } assert (== (array_length value.values) 2) return 0 }
shadow main { assert true }
'''
        for module in self.modules(source):self.qualify(module,False)
    def test_extern_record_provenance_stays_unknown(self):
        source='extern struct Foreign { value: int } struct Local { value: int } fn main() -> int { return 0 } shadow main { assert true }'
        for module in self.modules(source):self.qualify(module,False)
if __name__=='__main__':unittest.main()
