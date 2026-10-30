"""I retain ordered scalar-leaf observation statements on every loop test."""
import copy
import re
import struct
import sys
import unittest
import zlib
from tests import test_owned_branch_reconstruction as support
ROOT=support.ROOT
sys.path.insert(0,str(ROOT/'scripts'))
from nanoisa_owned_reconstruction import reconstruct_owned
from nanoisa_reconstruction import Refusal

class OwnedObservationReconstruction(unittest.TestCase):
    setUpClass=classmethod(support.OwnedBranchReconstruction.setUpClass.__func__)
    command=support.OwnedBranchReconstruction.command
    facts=support.OwnedBranchReconstruction.facts
    native=support.OwnedBranchReconstruction.native

    def module(self,case='multiple',compiler=None):
        compiler=compiler or self.compilers[0]
        source=ROOT/'tests/nanoisa/fixtures/owned_observation_loops'/(case+'.nano')
        module=self.work/(case+'-'+compiler.name+'.nvm')
        self.command([compiler,source,'--emit-nvm','-o',module])
        return module

    def roundtrip(self,module,expected):
        self.command([ROOT/'bin/nano_vm','--verify-only',module])
        self.command([ROOT/'bin/nano_vm',module],expected)
        facts=self.facts(module)
        for language in ('c','nano'):
            output=module.with_suffix('.'+language)
            self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output])
            text=output.read_text()
            self.assertIn('while (true)' if language=='c' else 'while true',text)
            self.assertIn('break;',text) if language=='c' else self.assertIn('{ break }',text)
            self.assertNotRegex(text,r'\bgoto\b|nano_vm|nvm_blob')
            if language=='c':self.native(output,expected)
            else:
                for producer in self.compilers:
                    rebuilt=output.with_suffix('.'+producer.name+'.nvm')
                    self.command([producer,output,'--emit-nvm','-o',rebuilt])
                    self.command([ROOT/'bin/nano_vm','--verify-only',rebuilt])
                    self.command([ROOT/'bin/nano_vm',rebuilt],expected)
                    regenerated=self.facts(rebuilt)
                    self.assertEqual(facts['layouts'],regenerated['layouts'])
                    for fn in facts['functions']:
                        self.assertIn((fn['params'],fn['result_type']),[(f['params'],f['result_type']) for f in regenerated['functions']])
                    native=rebuilt.with_suffix('.c')
                    self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native])
                    self.native(native,expected)

    def test_actual_multiple_nested_and_terminal_conditions(self):
        for compiler in self.compilers:
            for case,expected in [('multiple',16),('nested-return',26)]:
                with self.subTest(case=case,producer=str(compiler)):
                    module=self.module(case,compiler)
                    self.roundtrip(module,expected)
                    if case=='multiple':
                        # I count only generated pretest reads of the live parameter owner.
                        # Eight reached tests times two eager reads includes both false routes.
                        source=module.with_suffix('.c').read_text()
                        pattern=r'(?m)^(\s*(?:bool|int64_t) nlr_t\d+ = )(nlr_l0\.f[01]);$'
                        source,count=re.subn(pattern,r'\1(nlr_observation_count++, \2);',source)
                        self.assertEqual(count,2)
                        probe=module.with_suffix('.counted.c')
                        probe.write_text('#define main nlr_counted_main\nstatic int nlr_observation_count;\n'+source+
                            '\n#undef main\nint main(void) { int result=nlr_counted_main(); return nlr_observation_count==16 ? result : 99; }\n')
                        self.native(probe,expected)

    def test_original_verified_inverted_exit(self):
        original=self.module()
        text=self.command([ROOT/'bin/nanoisa','dump',original]).stdout
        self.assertEqual(text.count('  JMP_FALSE '),1)
        text=text.replace('  JMP_FALSE ','  NOT\n  JMP_TRUE ')
        asm=self.work/'inverted.nasm';asm.write_text(text)
        module=asm.with_suffix('.nvm')
        self.command([ROOT/'bin/nanoisa','asm',asm,'-o',module])
        self.roundtrip(module,16)

    def test_raw_reference_and_interior_edges_refuse_atomically(self):
        module=self.module();facts=self.facts(module);raw=module.read_bytes()
        header,count=struct.unpack_from('<II',raw,20)
        sections={struct.unpack_from('<IIQQ',raw,header+24*n)[0]:struct.unpack_from('<IIQQ',raw,header+24*n)[2] for n in range(count)}
        fn=facts['functions'][1];code=fn['code'];base=sections[6]+facts['functions'][0]['size']
        borrow=next(i for i in code if i['op']=='BORROW_LOCAL_SHARED')
        read=next(i for i in code if i['op']=='REF_GET')
        end=next(i for i in code if i['op']=='REGION_END')
        back=next(i for i in code if i['op']=='JMP')
        cases={'reference':(base+read['pc']+1,struct.pack('<H',255)),
               'field':(base+read['pc']+3,struct.pack('<H',255)),
               'owner':(base+borrow['pc']+3,struct.pack('<H',255)),
               'end':(base+end['pc'],b'\x00'),
               'interior':(base+back['pc']+1,struct.pack('<i',borrow['pc']-back['pc']))}
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

    def test_defensive_condition_effects_refuse(self):
        original=self.facts(self.module())
        for case in ('call','write','move','pop','dup','arithmetic','partial-region'):
            facts=copy.deepcopy(original);code=facts['functions'][1]['code']
            region=next(n for n,i in enumerate(code) if i['op']=='REGION_BEGIN')
            if case=='call':code[region+2].update(op='CALL_REF',args=[1,0],arg=1)
            elif case=='partial-region':code[region+3].update(op='NOP',args=[],arg=0)
            else:
                after=region+4
                code[after].update(op={'write':'STORE_LOCAL','move':'OWN_MOVE_LOCAL','pop':'POP','dup':'DUP','arithmetic':'DIV'}[case])
            for language in ('c','nano'):
                with self.subTest(case=case,language=language),self.assertRaises(Refusal):
                    reconstruct_owned(facts,language)
