"""I borrow exact extracted leaves without borrowing through a live parent."""
import copy
import struct
import unittest
import zlib
from tests import test_owned_shared_reconstruction as support
from nanoisa_owned_reconstruction import OwnedAnalyze, OwnedEmit, reconstruct_owned
from nanoisa_reconstruction import Refusal
ROOT=support.ROOT
FIXTURES=ROOT/'tests/nanoisa/fixtures/owned_extracted_leaf'

class OwnedExtractedReconstruction(unittest.TestCase):
    setUpClass=support.OwnedSharedReconstruction.__dict__['setUpClass']
    command=support.OwnedSharedReconstruction.command
    facts=support.OwnedSharedReconstruction.facts
    native=support.OwnedSharedReconstruction.native

    def module(self,case,compiler=None,source=None):
        compiler=compiler or self.compilers[0]
        source=source or FIXTURES/(case+'.nano')
        module=self.work/(case+'-'+compiler.name+'.nvm')
        self.command([compiler,source,'--emit-nvm','-o',module])
        self.command([ROOT/'bin/nano_vm','--verify-only',module])
        self.command([ROOT/'bin/nano_vm',module],{'shared':7,'exclusive-one':15,'exclusive-allchild':10}.get(case,35))
        return module

    def test_actual_extracted_leaf_roundtrips(self):
        for case,expected,mode in (('shared',7,1),('exclusive',35,2),('exclusive-one',15,2),('exclusive-allchild',10,2)):
            for compiler in self.compilers:
                with self.subTest(case=case,compiler=str(compiler)):
                    module=self.module(case,compiler);facts=self.facts(module)
                    self.assertEqual(facts['functions'][1]['param_modes'],[mode])
                    functions=[OwnedAnalyze(facts,i).run() for i in range(2)]
                    emitter=OwnedEmit(functions,'nano',facts['layouts'])
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
                                self.assertEqual(facts['layouts'],regenerated['layouts'])
                                self.assertEqual(regenerated['functions'][1]['param_modes'],[mode])
                                if mode==2:
                                    self.assertEqual([fn['locals'] for fn in regenerated['functions']],
                                                     [emitter.nano_local_cost(fn) for fn in functions])
                                native=rebuilt.with_suffix('.c')
                                self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native]);self.native(native,expected)

    def test_borrowed_parent_parameter_fails_closed(self):
        module=self.module('exclusive');facts=self.facts(module)
        parent=next(i for i,l in enumerate(facts['layouts']) if 8 in l['fields'])
        changed=copy.deepcopy(facts)
        changed['functions'][1]['params'][0]=[8,parent]
        changed['functions'][1]['local_types'][0]=[8,parent]
        with self.assertRaisesRegex(Refusal,'scalar-leaf borrowed helper parameter'):
            OwnedAnalyze(changed,1)
        raw=module.read_bytes();header,count=struct.unpack_from('<II',raw,20)
        sections={struct.unpack_from('<IIQQ',raw,header+24*n)[0]:struct.unpack_from('<IIQQ',raw,header+24*n)[2] for n in range(count)}
        at=sections[13];layouts=struct.unpack_from('<I',raw,at+4)[0]
        at+=((8+layouts+3)//4)*4+4
        at+=4+8*(1+facts['functions'][0]['locals'])
        parameter=at+4+8
        for layout in (1,parent):
            bad=bytearray(raw);struct.pack_into('<I',bad,parameter+4,layout)
            self.assertNotEqual(bad,raw)
            struct.pack_into('<I',bad,36,zlib.crc32(bad[header:]))
            source=self.work/('wrong-layout'+str(layout)+'.nvm');source.write_bytes(bad)
            self.assertNotEqual(self.command([ROOT/'bin/nano_vm','--verify-only',source],None).returncode,0)
            for language in ('c','nano'):
                output=self.work/('prior.'+language);output.write_text('prior checked source')
                self.assertNotEqual(self.command([ROOT/'bin/nvm2hl','--language',language,source,'-o',output],None).returncode,0)
                self.assertEqual(output.read_text(),'prior checked source')

    def test_live_parent_path_remains_outside_reconstruction(self):
        original=(FIXTURES/'exclusive.nano').read_text()
        text=original.replace('let parent: Pair','let mut parent: Pair')
        text=text.replace(' let Pair { stamp, first, enabled, second } = parent',
                          ' assert (== (update &mut parent.first) 10)\n let Pair { stamp, first, enabled, second } = parent')
        text=text.replace(' assert (== (update &mut current) 10)\n','')
        source=self.work/'live-path.nano';source.write_text(text)
        for compiler in self.compilers:
            module=self.module('live-path',compiler,source)
            for language in ('c','nano'):
                output=self.work/('path-prior.'+language);output.write_text('prior checked source')
                result=self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output],None)
                self.assertNotEqual(result.returncode,0)
                self.assertEqual(output.read_text(),'prior checked source')

    def test_consumed_parent_cannot_be_reused(self):
        original=(FIXTURES/'exclusive.nano').read_text()
        source=self.work/'reuse-parent.nano'
        source.write_text(original.replace(' let mut current: Leaf = first',
                                           ' let lost: Pair = parent\n let mut current: Leaf = first'))
        for compiler in self.compilers:
            output=self.work/('reuse-'+compiler.name+'.nvm');output.write_text('prior checked module')
            result=self.command([compiler,source,'--emit-nvm','-o',output],None)
            self.assertNotEqual(result.returncode,0)
            self.assertRegex(result.stdout+result.stderr,r'(?i)consum|mov|live|ownership|resource')
            self.assertEqual(output.read_text(),'prior checked module')

    def test_nested_generated_local_boundary_preserves_output(self):
        original=(FIXTURES/'exclusive.nano').read_text()
        for padding,total in ((2,125),(3,133)):
            declarations='\n'.join(f' let pad{i}: int = (+ 0 0)' for i in range(padding))
            source=self.work/('nested-local'+str(total)+'.nano')
            source.write_text(original.replace('fn main() -> int {','fn main() -> int {\n'+declarations))
            for compiler in self.compilers:
                module=self.module('nested-local'+str(total),compiler,source);facts=self.facts(module)
                self.assertLess(facts['functions'][0]['locals'],128)
                functions=[OwnedAnalyze(facts,i).run() for i in range(2)]
                emitter=OwnedEmit(functions,'nano',facts['layouts'])
                self.assertEqual(emitter.nano_local_cost(functions[0]),total)
                output=module.with_suffix('.nano')
                if total>128:
                    output.write_text('prior checked source')
                    result=self.command([ROOT/'bin/nvm2hl','--language','nano',module,'-o',output],None)
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn('128 local slots',result.stderr)
                    self.assertEqual(output.read_text(),'prior checked source')
                    emitter.nano_local_cost=lambda fn:0
                    output=module.with_suffix('.diagnostic.nano');output.write_text(emitter.run(0))
                else:
                    self.command([ROOT/'bin/nvm2hl','--language','nano',module,'-o',output])
                rebuilt=output.with_suffix('.nvm')
                self.command([compiler,output,'--emit-nvm','-o',rebuilt])
                self.command([ROOT/'bin/nano_vm','--verify-only',rebuilt])
                self.command([ROOT/'bin/nano_vm',rebuilt],35)
                dump=self.command([ROOT/'bin/nanoisa','dump',rebuilt]).stdout
                self.assertRegex(dump,rf'(?m)^\.function \S+ 0 {total} ')
                if total>128:
                    self.assertNotEqual(self.command([ROOT/'bin/nanoisa_hl_facts',rebuilt],None).returncode,0)
                else:
                    self.assertEqual(self.facts(rebuilt)['functions'][0]['locals'],total)
                    native=rebuilt.with_suffix('.c')
                    self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native]);self.native(native,35)

    def test_actual_multiple_reference_fixture_has_explicit_witness(self):
        source=ROOT/'tests/nanoisa/fixtures/source_borrow_multi.nano'
        for compiler in self.compilers:
            module=self.work/('multiple-'+compiler.name+'.nvm')
            self.command([compiler,source,'--emit-nvm','-o',module])
            self.command([ROOT/'bin/nano_vm','--verify-only',module])
            self.command([ROOT/'bin/nano_vm',module])
            for language in ('c','nano'):
                output=self.work/('multiple-witness.'+language)
                self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output])
                self.assertIn('NlrOwned',output.read_text())
                if language=='nano':
                    self.assertIn('owner_first_2',output.read_text())
                    self.assertIn('field_second_2_1',output.read_text())
