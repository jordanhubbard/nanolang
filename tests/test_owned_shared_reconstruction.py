"""I retain exact shared helper authority through structured source reconstruction."""
import copy
from pathlib import Path
import struct
import sys
import unittest
import zlib
from tests import test_owned_branch_reconstruction as support
ROOT=support.ROOT
sys.path.insert(0,str(ROOT/'scripts'))
from nanoisa_owned_reconstruction import reconstruct_owned
from nanoisa_reconstruction import Refusal

class OwnedSharedReconstruction(unittest.TestCase):
    setUpClass=classmethod(support.OwnedBranchReconstruction.setUpClass.__func__)
    command=support.OwnedBranchReconstruction.command
    facts=support.OwnedBranchReconstruction.facts
    native=support.OwnedBranchReconstruction.native

    def module(self,case='original',compiler=None):
        compiler=compiler or self.compilers[0]
        source=ROOT/'tests/nanoisa/fixtures/source_borrow_shared.nano'
        if case=='bool':
            source=self.work/'shared-bool.nano'
            source.write_text('''resource struct Flag { active: bool, count: int }
fn read(view: &Flag) -> bool { let value: bool = view.active return (not (not value)) }
shadow read { let owner: Flag = Flag { active: true, count: 7 } assert (read &owner) let Flag { active, count } = owner assert active assert (== count 7) }
fn main() -> int {
 let owner: Flag = Flag { active: true, count: 7 }
 assert (read &owner)
 assert (read &owner)
 let Flag { active, count } = owner
 assert active
 return count
}
shadow main { assert true }
''')
        module=self.work/(case+'-'+compiler.name+'.nvm')
        self.command([compiler,source,'--emit-nvm','-o',module])
        return module

    def test_actual_source_and_structured_roundtrips(self):
        for case,expected in [('original',0),('bool',7)]:
            for compiler in self.compilers:
                with self.subTest(case=case,compiler=str(compiler)):
                    module=self.module(case,compiler);facts=self.facts(module)
                    self.assertEqual(facts['functions'][1]['param_modes'],[1])
                    self.command([ROOT/'bin/nano_vm','--verify-only',module])
                    self.command([ROOT/'bin/nano_vm',module],expected)
                    for language in ('c','nano'):
                        output=module.with_suffix('.'+language)
                        self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output])
                        self.assertNotRegex(output.read_text(),r'\bgoto\b|nano_vm|nvm_blob')
                        self.assertIn('const NlrOwned' if language=='c' else '&NlrOwned',output.read_text())
                        if language=='c':self.native(output,expected)
                        else:
                            for producer in self.compilers:
                                rebuilt=output.with_suffix('.'+producer.name+'.nvm')
                                self.command([producer,output,'--emit-nvm','-o',rebuilt])
                                self.command([ROOT/'bin/nano_vm','--verify-only',rebuilt])
                                self.command([ROOT/'bin/nano_vm',rebuilt],expected)
                                rebuilt_facts=self.facts(rebuilt)
                                self.assertEqual(rebuilt_facts['functions'][1]['param_modes'],[1])
                                self.assertEqual(facts['layouts'],rebuilt_facts['layouts'])
                                native=rebuilt.with_suffix('.c')
                                self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native])
                                self.native(native,expected)

    def test_original_verifier_rejects_malformed_references_atomically(self):
        module=self.module();facts=self.facts(module);raw=module.read_bytes()
        header,count=struct.unpack_from('<II',raw,20)
        sections={struct.unpack_from('<IIQQ',raw,header+24*n)[0]:struct.unpack_from('<IIQQ',raw,header+24*n)[2] for n in range(count)}
        code=facts['functions'][0]['code'];base=sections[6]
        call=next(i for i in code if i['op']=='CALL_REF')
        borrow=next(i for i in code if i['op']=='BORROW_LOCAL_SHARED')
        end=next(i for i in code if i['op']=='REGION_END')
        helper=base+facts['functions'][0]['size']
        at=sections[13];layouts=struct.unpack_from('<I',raw,at+4)[0]
        at+=((8+layouts+3)//4)*4+4
        at+=4+8*(1+facts['functions'][0]['locals'])
        parameter=at+4+8
        self.assertEqual(raw[parameter+1],1)
        cases={'parameter-exclusive':(parameter+1,b'\x02'),
               'parameter-nominal':(parameter+4,struct.pack('<I',1)),
               'borrowed-load':(helper,b'\x10\x00\x00\x00\x00'),
               'borrowed-move':(helper,b'\x0b\x00\x00\x00\x00'),
               'borrowed-write':(helper,b'\x1b'),
               'call-reference':(base+call['pc']+5,struct.pack('<H',255)),
               'call-target':(base+call['pc']+1,struct.pack('<I',0)),
               'absent-owner':(base+borrow['pc']+3,struct.pack('<H',0)),
               'missing-end':(base+end['pc'],b'\x00'),
               'helper-reference':(helper+1,struct.pack('<H',255)),
               'helper-field':(helper+3,struct.pack('<H',255)),
               'missing-authority':(sections[13]+8,b'\x00')}
        for name,(offset,payload) in cases.items():
            with self.subTest(name=name):
                changed=bytearray(raw);changed[offset:offset+len(payload)]=payload
                self.assertNotEqual(changed,raw)
                struct.pack_into('<I',changed,36,zlib.crc32(changed[header:]))
                bad=self.work/(name+'.nvm');bad.write_bytes(changed)
                self.assertNotEqual(self.command([ROOT/'bin/nano_vm','--verify-only',bad],None).returncode,0)
                for language in ('c','nano'):
                    output=self.work/('prior.'+language);output.write_text('prior checked source')
                    result=self.command([ROOT/'bin/nvm2hl','--language',language,bad,'-o',output],None)
                    self.assertNotEqual(result.returncode,0)
                    self.assertNotIn('checksum',result.stderr.lower())
                    self.assertEqual(output.read_text(),'prior checked source')

    def test_exact_shared_profile_refusals(self):
        original=self.facts(self.module())
        for name in ('exclusive','nominal','helper-write','helper-move','helper-load','helper-call','helper-wrongslot','mixed-call','mismatch-ref','incomplete-region'):
            facts=copy.deepcopy(original);helper=facts['functions'][1];code=facts['functions'][0]['code']
            call=next(i for i in code if i['op']=='CALL_REF')
            if name=='exclusive':helper['param_modes']=[2]
            elif name=='nominal':helper['params'][0][1]=1;helper['local_types'][0][1]=1
            elif name=='helper-write':helper['code'][0]['op']='REF_SET'
            elif name=='helper-move':helper['code'][0]['op']='OWN_MOVE_LOCAL'
            elif name=='helper-load':helper['code'][0]['op']='LOAD_LOCAL'
            elif name=='helper-call':helper['code'][0]['op']='CALL_REF'
            elif name=='helper-wrongslot':helper['code'][0].update(arg=1,args=[1,0])
            elif name=='mixed-call':call['op']='CALL'
            elif name=='mismatch-ref':call['args'][1]=1
            else:next(i for i in code if i['op']=='REGION_END')['op']='NOP'
            with self.subTest(name=name),self.assertRaises(Refusal):reconstruct_owned(facts,'c')
        module=self.work/'exclusive.nvm'
        self.command([self.compilers[0],ROOT/'tests/nanoisa/fixtures/source_borrow_exclusive.nano','--emit-nvm','-o',module])
        self.command([ROOT/'bin/nano_vm','--verify-only',module])
        for language in ('c','nano'):
            output=self.work/('exclusive.'+language);output.write_text('prior checked source')
            self.assertNotEqual(self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output],None).returncode,0)
            self.assertEqual(output.read_text(),'prior checked source')
