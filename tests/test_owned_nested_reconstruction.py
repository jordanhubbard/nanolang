"""I retain exact child ownership through bounded depth-two resource transfers."""
import copy
import struct
import sys
import unittest
import zlib
from tests import test_owned_branch_reconstruction as support
ROOT=support.ROOT
sys.path.insert(0,str(ROOT/'scripts'))
from nanoisa_owned_reconstruction import reconstruct_owned
from nanoisa_reconstruction import Refusal

class OwnedNestedReconstruction(unittest.TestCase):
    setUpClass=classmethod(support.OwnedBranchReconstruction.setUpClass.__func__)
    command=support.OwnedBranchReconstruction.command
    facts=support.OwnedBranchReconstruction.facts
    native=support.OwnedBranchReconstruction.native

    def module(self,case='original',compiler=None):
        compiler=compiler or self.compilers[0]
        source=ROOT/'tests/nanoisa/fixtures/source_owned_value_nested.nano'
        if case=='distinct':
            source=self.work/'distinct.nano'
            source.write_text('''resource struct Left { value: int }
resource struct Right { value: int }
resource struct Pair { left: Left, right: Right }
fn pass(pair: Pair, choice: bool) -> Pair {
 if choice { return pair } else { return pair }
}
shadow pass { assert true }
fn consume(pair: Pair) -> int {
 let Pair { right, left } = pair
 let Left { value } = left
 let a: int = value
 let Right { value } = right
 assert (== a 1)
 assert (== value 2)
 return (+ a value)
}
shadow consume { assert true }
fn main() -> int {
 let left: Left = Left { value: 1 }
 let right: Right = Right { value: 2 }
 let pair: Pair = Pair { right: right, left: left }
 let first: int = (consume (pass pair true))
 let second_left: Left = Left { value: 1 }
 let second_right: Right = Right { value: 2 }
 let second: Pair = Pair { right: second_right, left: second_left }
 let mut again: bool = true
 while again { set again false }
 return (+ first (consume (pass second false)))
}
shadow main { assert (== (main) 6) }
''')
        module=self.work/(case+'-'+compiler.name+'.nvm')
        self.command([compiler,source,'--emit-nvm','-o',module])
        return module

    def test_actual_nested_pack_unpack_calls_returns_and_branches(self):
        for case,expected in [('original',0),('distinct',6)]:
            for compiler in self.compilers:
                with self.subTest(case=case,producer=str(compiler)):
                    module=self.module(case,compiler)
                    self.command([ROOT/'bin/nano_vm','--verify-only',module])
                    self.command([ROOT/'bin/nano_vm',module],expected)
                    facts=self.facts(module)
                    self.assertTrue(any(any(t[0]==8 for t in l['field_types']) for l in facts['layouts']))
                    for language in ('c','nano'):
                        output=module.with_suffix('.'+language)
                        self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output])
                        self.assertNotRegex(output.read_text(),r'\bgoto\b|nano_vm|nvm_blob')
                        if language=='c':self.native(output,expected)
                        else:
                            for producer in self.compilers:
                                rebuilt=output.with_suffix('.'+producer.name+'.nvm')
                                self.command([producer,output,'--emit-nvm','-o',rebuilt])
                                self.command([ROOT/'bin/nano_vm','--verify-only',rebuilt])
                                self.command([ROOT/'bin/nano_vm',rebuilt],expected)
                                rebuilt_facts=self.facts(rebuilt)
                                self.assertEqual(facts['layouts'],rebuilt_facts['layouts'])
                                native=rebuilt.with_suffix('.c')
                                self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native])
                                self.native(native,expected)

    def test_field_descriptors_and_child_tokens_refuse(self):
        original=self.facts(self.module('distinct'))
        for descriptor in (None,1,[],[[]],[[8]],[[8,0,1]],[[8,'0']],[[8,True]]):
            facts=copy.deepcopy(original);facts['layouts'][-1]['field_types']=descriptor
            with self.subTest(descriptor=descriptor),self.assertRaises(Refusal):reconstruct_owned(facts,'c')
        for case in ('wrong-child','cycle','mixed','deep','pop-child','cast-child','copy-child','scalar-store-child','duplicate-child','live-child'):
            facts=copy.deepcopy(original)
            if case=='wrong-child':facts['layouts'][-1]['field_types'][0][1]=1
            elif case=='cycle':facts['layouts'][-1]['field_types'][0][1]=2
            elif case=='mixed':facts['layouts'][-1]['fields'][0]=1;facts['layouts'][-1]['field_types'][0]=[1,4294967295]
            elif case=='deep':facts['layouts'].append(dict(identity=3,resource=True,fields=[8],field_types=[[8,2]]))
            else:
                fn=next(f for f in facts['functions'] if any(i['op']=='OWN_UNPACK_LOCAL' for i in f['code']))
                code=fn['code'];unpack=next(n for n,i in enumerate(code) if i['op']=='OWN_UNPACK_LOCAL')
                if case in ('pop-child','cast-child','copy-child','scalar-store-child'):
                    code[unpack+1]['op']={'pop-child':'POP','cast-child':'CAST_INT','copy-child':'DUP','scalar-store-child':'STORE_LOCAL'}[case]
                elif case=='duplicate-child':code[unpack+2].update(arg=code[unpack+1]['arg'],args=code[unpack+1]['args'][:])
                else:
                    last=max(n for n,i in enumerate(code) if i['op']=='OWN_UNPACK_LOCAL')
                    for i in code[last:last+2]:i.update(op='NOP',arg=0,args=[])
            with self.subTest(case=case),self.assertRaises(Refusal):reconstruct_owned(facts,'c')

    def test_crc_valid_child_identity_and_loss_refuse_original_admission(self):
        module=self.module('distinct');facts=self.facts(module);raw=module.read_bytes()
        header,count=struct.unpack_from('<II',raw,20)
        sections={struct.unpack_from('<IIQQ',raw,header+24*n)[0]:struct.unpack_from('<IIQQ',raw,header+24*n)[2] for n in range(count)}
        at=sections[4]+4;fields=[]
        for layout in facts['layouts']:
            size=struct.unpack_from('<H',raw,at+2)[0]
            self.assertEqual(size,len(layout['fields']))
            fields.append(at+8);at+=8+12*size
        fn_index=next(n for n,f in enumerate(facts['functions']) if any(i['op']=='OWN_UNPACK_LOCAL' for i in f['code']))
        code=facts['functions'][fn_index]['code'];base=sections[6]+sum(f['size'] for f in facts['functions'][:fn_index])
        unpack=next(n for n,i in enumerate(code) if i['op']=='OWN_UNPACK_LOCAL')
        store=code[unpack+1];self.assertEqual(store['op'],'OWN_STORE_LOCAL')
        cases={'wrong-child':(fields[2]+4,struct.pack('<I',1)),
               'cycle':(fields[2]+4,struct.pack('<I',2)),
               'no-child-layout':(fields[2]+4,struct.pack('<I',4294967295)),
               'child-pop':(base+store['pc'],b'\x08\x00\x00'),
               'child-authority':(sections[13]+8,b'\x00')}
        for case,(offset,payload) in cases.items():
            changed=bytearray(raw);changed[offset:offset+len(payload)]=payload
            self.assertNotEqual(changed,raw)
            struct.pack_into('<I',changed,36,zlib.crc32(changed[header:]))
            bad=self.work/(case+'.nvm');bad.write_bytes(changed)
            self.assertNotEqual(self.command([ROOT/'bin/nano_vm','--verify-only',bad],None).returncode,0)
            for language in ('c','nano'):
                output=self.work/('prior.'+language);output.write_text('prior checked source')
                result=self.command([ROOT/'bin/nvm2hl','--language',language,bad,'-o',output],None)
                self.assertNotEqual(result.returncode,0)
                self.assertNotIn('checksum',result.stderr.lower())
                self.assertEqual(output.read_text(),'prior checked source')

    def test_verified_deeper_and_borrowed_layouts_refuse_atomically(self):
        sources={
            'deep':'''resource struct Leaf { value: int }
resource struct Mid { leaf: Leaf }
resource struct Outer { mid: Mid }
fn main() -> int {
 let leaf: Leaf = Leaf { value: 7 }
 let mid: Mid = Mid { leaf: leaf }
 let outer: Outer = Outer { mid: mid }
 let Outer { mid } = outer
 let Mid { leaf } = mid
 let Leaf { value } = leaf
 return value
}
shadow main { assert true }
''',
            'borrowed':'''resource struct Leaf { value: int }
resource struct Box { leaf: Leaf }
fn read(view: &Leaf) -> int { return view.value }
shadow read { let leaf: Leaf = Leaf { value: 7 } assert (== (read &leaf) 7) let Leaf { value } = leaf assert (== value 7) }
fn main() -> int {
 let leaf: Leaf = Leaf { value: 7 }
 let box: Box = Box { leaf: leaf }
 let Box { leaf } = box
 assert (== (read &leaf) 7)
 let Leaf { value } = leaf
 return value
}
shadow main { assert true }
'''}
        for case,text in sources.items():
            for compiler in self.compilers:
                with self.subTest(case=case,producer=str(compiler)):
                    source=self.work/(case+'.nano');source.write_text(text)
                    module=self.work/(case+'-'+compiler.name+'.nvm')
                    self.command([compiler,source,'--emit-nvm','-o',module])
                    self.command([ROOT/'bin/nano_vm','--verify-only',module])
                    self.command([ROOT/'bin/nano_vm',module],7)
                    for language in ('c','nano'):
                        output=self.work/('prior.'+language);output.write_text('prior checked source')
                        result=self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output],None)
                        self.assertNotEqual(result.returncode,0)
                        self.assertEqual(output.read_text(),'prior checked source')
