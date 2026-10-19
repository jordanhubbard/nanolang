"""I preserve independently exact scalar/child fields without new transfer rules."""
import copy
import struct
import sys
import unittest
import zlib
from tests import test_owned_branch_reconstruction as support
from tests import test_owned_void_reconstruction as traps
ROOT=support.ROOT
sys.path.insert(0,str(ROOT/'scripts'))
from nanoisa_owned_reconstruction import reconstruct_owned
from nanoisa_reconstruction import Refusal

class OwnedMixedReconstruction(unittest.TestCase):
    setUpClass=classmethod(support.OwnedBranchReconstruction.setUpClass.__func__)
    command=support.OwnedBranchReconstruction.command
    facts=support.OwnedBranchReconstruction.facts
    native=support.OwnedBranchReconstruction.native
    vm_failure=traps.OwnedVoidReconstruction.vm_failure
    native_failure=traps.OwnedVoidReconstruction.native_failure
    native_assert_status=traps.OwnedVoidReconstruction.native_assert_status

    def module(self,case='order',compiler=None):
        compiler=compiler or self.compilers[0]
        source=ROOT/'tests/nanoisa/fixtures/owned_mixed'/(case+'.nano')
        module=self.work/(case+'-'+compiler.name+'.nvm')
        self.command([compiler,source,'--emit-nvm','-o',module])
        return module

    def test_mixed_order_calls_branches_and_loop(self):
        for case,expected in [('basic',7),('order',45)]:
            for compiler in self.compilers:
                with self.subTest(case=case,producer=str(compiler)):
                    module=self.module(case,compiler)
                    self.command([ROOT/'bin/nano_vm','--verify-only',module])
                    self.command([ROOT/'bin/nano_vm',module],expected)
                    facts=self.facts(module)
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
                                self.assertEqual(facts['layouts'],self.facts(rebuilt)['layouts'])
                                native=rebuilt.with_suffix('.c')
                                self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native])
                                self.native(native,expected)

    def test_written_assertion_precedes_declaration_order_division(self):
        for compiler in self.compilers:
            module=self.module('trap-order',compiler);self.vm_failure(module)
            facts=self.facts(module)
            calls=[i for i in facts['functions'][0]['code'] if i['op']=='CALL']
            self.assertEqual(len(calls),2)
            # I retain BOOL checking before INT-to-resource creation, independent of source function numbering.
            self.assertEqual(facts['functions'][calls[0]['arg']]['params'],[[4,4294967295]])
            self.assertEqual(facts['functions'][calls[1]['arg']]['result_type'],[8,0])
            for language in ('c','nano'):
                output=module.with_suffix('.'+language)
                self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output])
                for call in calls:
                    name=f"nlr_f{call['arg']}_owned_{call['arg']}"
                    # One declaration/definition in Nano and one call; C also has a prototype.
                    self.assertEqual(output.read_text().count(name+'(' if language=='c' else '('+name+' '),3 if language=='c' else 1)
                if language=='c':self.native_failure(output,'Assertion failed')
                else:
                    for producer in self.compilers:
                        rebuilt=output.with_suffix('.'+producer.name+'.nvm')
                        self.command([producer,output,'--emit-nvm','-o',rebuilt])
                        self.vm_failure(rebuilt)
                        native=rebuilt.with_suffix('.c')
                        self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native])
                        self.native_failure(native,None)
                        self.native_assert_status(native)

    def test_mixed_descriptors_and_child_scalar_operations_refuse(self):
        original=self.facts(self.module())
        for case in ('scalar-layout','wrong-child','pop','copy','cast','store','field-range'):
            facts=copy.deepcopy(original)
            if case=='scalar-layout':facts['layouts'][2]['field_types'][0][1]=0
            elif case=='wrong-child':facts['layouts'][2]['field_types'][1][1]=1
            elif case=='field-range':
                for i in facts['functions'][0]['code']:
                    if i['op']=='REF_GET':i['args'][1]=1;break
            else:
                fn=next(f for f in facts['functions'] if any(i['op']=='OWN_UNPACK_LOCAL' for i in f['code']))
                code=fn['code'];start=next(n for n,i in enumerate(code) if i['op']=='OWN_UNPACK_LOCAL')
                child=next(i for i in code[start+1:] if i['op']=='OWN_STORE_LOCAL')
                child['op']={'pop':'POP','copy':'DUP','cast':'CAST_BOOL','store':'STORE_LOCAL'}[case]
            with self.subTest(case=case),self.assertRaises(Refusal):reconstruct_owned(facts,'c')

    def test_raw_aggregate_read_and_scalar_child_confusion_refuse_atomically(self):
        module=self.module();facts=self.facts(module);raw=module.read_bytes()
        header,count=struct.unpack_from('<II',raw,20)
        sections={struct.unpack_from('<IIQQ',raw,header+24*n)[0]:struct.unpack_from('<IIQQ',raw,header+24*n)[2] for n in range(count)}
        at=sections[4]+4;fields=[]
        for layout in facts['layouts']:
            fields.append(at+8);at+=8+12*len(layout['fields'])
        read=next(i for i in facts['functions'][0]['code'] if i['op']=='REF_GET')
        cases={'scalar-layout':(fields[2]+4,struct.pack('<I',0)),
               'wrong-child':(fields[2]+12+4,struct.pack('<I',1)),
               'field-range':(sections[6]+read['pc']+3,struct.pack('<H',1))}
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
                self.assertEqual(output.read_text(),'prior checked source')

    def test_source_and_verifier_observation_boundaries_refuse(self):
        source=ROOT/'docs/evidence/owned-mixed-depth2-20261019/original-observation-source.nano.txt'
        for compiler in self.compilers:
            actual=self.work/(compiler.name+'-observation.nano');actual.write_text(source.read_text())
            result=self.command([compiler,actual,'--emit-nvm','-o',self.work/(compiler.name+'-observation.nvm')],None)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('scalar field of a leaf resource record',result.stdout+result.stderr)
        original=self.module('basic');facts=self.facts(original)
        store=next(i for i in facts['functions'][0]['code'] if i['op']=='OWN_STORE_LOCAL' and
                   facts['functions'][0]['local_types'][i['arg']]==[8,1])
        assembly=self.command([ROOT/'bin/nanoisa','dump',original]).stdout
        marker=f"  OWN_STORE_LOCAL {store['arg']}\n"
        self.assertEqual(assembly.count(marker),1)
        assembly=assembly.replace(marker,marker+f"  REGION_BEGIN\n  BORROW_LOCAL_SHARED 0 {store['arg']}\n  REF_GET 0 1\n  REGION_END\n  POP\n")
        asm=self.work/'observed-parent.nasm';asm.write_text(assembly)
        module=asm.with_suffix('.nvm')
        result=self.command([ROOT/'bin/nanoisa','asm',asm,'-o',module],None)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('available scalar-leaf resource owner',result.stderr)
        self.assertFalse(module.exists())
        # I test my defensive facts guard separately, without inventing verifier admission.
        fn=facts['functions'][0];index=fn['code'].index(store)+1
        sequence=[{'op':'REGION_BEGIN','args':[],'arg':0},
                  {'op':'BORROW_LOCAL_SHARED','args':[0,store['arg']],'arg':0},
                  {'op':'REF_GET','args':[0,1],'arg':0},
                  {'op':'REGION_END','args':[],'arg':0},
                  {'op':'POP','args':[],'arg':0}]
        fn['code'][index:index]=sequence
        for pc,instruction in enumerate(fn['code']):
            instruction['pc']=pc;instruction['target']=4294967295
        fn['size']=len(fn['code'])
        for language in ('c','nano'):
            with self.assertRaisesRegex(Refusal,'scalar-leaf owner for ephemeral field observation'):
                reconstruct_owned(facts,language)
