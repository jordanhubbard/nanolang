"""I retain exact consuming resource depth without widening borrowed layouts."""
import copy
import struct
import unittest
import zlib
from tests import test_owned_shared_reconstruction as support
from nanoisa_owned_reconstruction import OwnedAnalyze, reconstruct_owned
from nanoisa_reconstruction import Refusal
ROOT=support.ROOT
FIXTURES=ROOT/'tests/nanoisa/fixtures/owned_depth_three'

class OwnedDepthThreeReconstruction(unittest.TestCase):
    setUpClass=support.OwnedSharedReconstruction.__dict__['setUpClass']
    command=support.OwnedSharedReconstruction.command
    facts=support.OwnedSharedReconstruction.facts
    native=support.OwnedSharedReconstruction.native

    def module(self,case='basic',compiler=None):
        compiler=compiler or self.compilers[0]
        module=self.work/(case+'-'+compiler.name+'.nvm')
        self.command([compiler,FIXTURES/(case+'.nano'),'--emit-nvm','-o',module])
        self.command([ROOT/'bin/nano_vm','--verify-only',module])
        self.command([ROOT/'bin/nano_vm',module],31 if case=='composed' else 7)
        return module

    def test_actual_depth_three_and_composed_calls_control_flow(self):
        for case,expected in (('basic',7),('composed',31)):
            for compiler in self.compilers:
                with self.subTest(case=case,compiler=str(compiler)):
                    module=self.module(case,compiler);facts=self.facts(module)
                    self.assertEqual(max(OwnedAnalyze(facts).depths),3)
                    for language in ('c','nano'):
                        output=module.with_suffix('.'+language)
                        self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output])
                        if language=='c':self.native(output,expected)
                        else:
                            for producer in self.compilers:
                                rebuilt=output.with_suffix('.'+producer.name+'.nvm')
                                self.command([producer,output,'--emit-nvm','-o',rebuilt])
                                self.command([ROOT/'bin/nano_vm','--verify-only',rebuilt])
                                self.command([ROOT/'bin/nano_vm',rebuilt],expected)
                                regenerated=self.facts(rebuilt)
                                self.assertEqual(regenerated['layouts'],facts['layouts'])
                                signatures=lambda m:sorted(repr((f['params'],f['result_type'])) for f in m['functions'])
                                for signature in signatures(facts):self.assertIn(signature,signatures(regenerated))
                                native=rebuilt.with_suffix('.c')
                                self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native]);self.native(native,expected)

    def test_actual_depth_four_and_borrowed_depth_three_refuse_atomically(self):
        for case in ('depth-four','borrowed-depth-three'):
            for compiler in self.compilers:
                module=self.module(case,compiler)
                for language in ('c','nano'):
                    output=self.work/('prior.'+language);output.write_text('prior checked source')
                    result=self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output],None)
                    self.assertNotEqual(result.returncode,0)
                    self.assertEqual(output.read_text(),'prior checked source')

    def test_exact_depth_descriptors_and_nested_child_tokens(self):
        original=self.facts(self.module())
        for case in ('wrong-child','cycle','forward','depth-four','bad-scalar-layout','bad-tag','pop','copy','cast','store'):
            facts=copy.deepcopy(original)
            if case=='wrong-child':facts['layouts'][2]['field_types'][0][1]=0
            elif case=='cycle':facts['layouts'][2]['field_types'][0][1]=2
            elif case=='forward':facts['layouts'][1]['field_types'][0][1]=2
            elif case=='depth-four':facts['layouts'].append(dict(identity=3,resource=True,fields=[8],field_types=[[8,2]]))
            elif case=='bad-scalar-layout':facts['layouts'][0]['field_types'][0][1]=0
            elif case=='bad-tag':facts['layouts'][0]['fields'][0]=5;facts['layouts'][0]['field_types'][0][0]=5
            else:
                code=facts['functions'][0]['code'];at=next(n for n,i in enumerate(code) if i['op']=='OWN_UNPACK_LOCAL')
                self.assertEqual(code[at+1]['op'],'OWN_STORE_LOCAL')
                code[at+1]['op']={'pop':'POP','copy':'DUP','cast':'CAST_INT','store':'STORE_LOCAL'}[case]
            for language in ('c','nano'):
                with self.subTest(case=case,language=language),self.assertRaises(Refusal):reconstruct_owned(facts,language)
        # I distinguish a defensive facts mutation from original module admission.
        source=ROOT/'tests/nanoisa/fixtures/owned_extracted_leaf/shared.nano'
        module=self.work/'borrowed-two.nvm';self.command([self.compilers[0],source,'--emit-nvm','-o',module])
        facts=self.facts(module)
        facts['layouts'].append(dict(identity=2,resource=True,fields=[8],field_types=[[8,1]]))
        for language in ('c','nano'):
            with self.assertRaisesRegex(Refusal,'borrowed modules at depth two'):reconstruct_owned(facts,language)

    def test_crc_valid_child_identity_and_loss_refuse_original_admission(self):
        module=self.module('basic');facts=self.facts(module);raw=module.read_bytes()
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
        cases={'wrong-child':(fields[2]+4,struct.pack('<I',0)),
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

