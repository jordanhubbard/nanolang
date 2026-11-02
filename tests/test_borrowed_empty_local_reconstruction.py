"""I preserve an independent empty local beside an exact nonempty borrow."""
import copy
import json
import re
import unittest
from tests import test_empty_owned_reconstruction as support
from nanoisa_owned_reconstruction import OwnedAnalyze, OwnedEmit, reconstruct_owned
from nanoisa_reconstruction import Refusal
ROOT=support.ROOT
BASE=ROOT/'docs/evidence/borrowed-empty-local-baseline-20261020'

class BorrowedEmptyLocal(unittest.TestCase):
    setUpClass=support.EmptyOwnedReconstruction.__dict__['setUpClass']
    command=support.EmptyOwnedReconstruction.command
    facts=support.EmptyOwnedReconstruction.facts
    native=support.EmptyOwnedReconstruction.native
    module=support.EmptyOwnedReconstruction.module
    def qualify(self,module):
        original=module.with_suffix('.original.c')
        self.command([ROOT/'bin/nvm2c',module,'-o',original]);self.native(original,0)
        return support.EmptyOwnedReconstruction.qualify(self,module)
    strict_c=support.EmptyOwnedReconstruction.strict_c

    def test_unchanged_and_swapped_layout_order(self):
        original=(BASE/'original.nano').read_text()
        swapped=original.replace('resource struct Empty {}\n','').replace('fn update','resource struct Empty {}\nfn update',1)
        for order,text in enumerate((original,swapped)):
            for compiler in self.compilers:
                module=self.module(compiler,text,'borrowed-empty-'+str(order));facts=self.qualify(module)
                self.assertEqual([f['locals'] for f in facts['functions']],[8,1])
                empty=next(i for i,l in enumerate(facts['layouts']) if not l['fields'])
                self.assertEqual(empty,order);self.assertEqual(facts['functions'][1]['params'],[[8,1-order]])
                self.strict_c(module.with_suffix('.c'))

    def test_original_and_generated_shadows(self):
        original=self.work/'original-shadow.nvm'
        self.command([ROOT/'bin/nanoisa','asm',BASE/'original-shadows.nasm','-o',original])
        self.command([ROOT/'bin/nano_vm','--verify-only',original]);self.command([ROOT/'bin/nano_vm',original])
        self.qualify(original);self.strict_c(original.with_suffix('.c'))
        facts=self.facts(self.module(self.compilers[0],(BASE/'original.nano').read_text(),'generated-shadow'))
        text=reconstruct_owned(facts,'nano')
        selected=support.support.OwnedNestedPathReconstruction.selected_shadow(text)
        for compiler in self.compilers:
            module=self.module(compiler,selected,'selected-shadow');selected_facts=self.facts(module)
            print('I retain empty-local selected-shadow counts '+json.dumps([f['locals'] for f in selected_facts['functions']]),flush=True)
            self.assertEqual(selected_facts['functions'][1]['params'],[[8,1]])
            c=module.with_suffix('.native.c');self.command([ROOT/'bin/nvm2c',module,'-o',c]);self.native(c,0)

    def test_empty_nominal_leak_and_double_move_refuse(self):
        module=self.module(self.compilers[0],(BASE/'original.nano').read_text(),'negative')
        text=self.command([ROOT/'bin/nanoisa','dump',module]).stdout
        move=re.search(r'^  OWN_MOVE_LOCAL 0\n',text,re.M);unpack=re.search(r'^  OWN_UNPACK_LOCAL 1\n',text,re.M)
        self.assertIsNotNone(move);self.assertIsNotNone(unpack)
        cases={'nominal':text.replace('OWN_PACK 0','OWN_PACK 1',1),
               'leak':text[:unpack.start()]+text[unpack.end():],
               'move-twice':text[:move.end()]+move.group(0)+text[move.end():]}
        for name,content in cases.items():
            assembly=self.work/(name+'.nasm');assembly.write_text(content);bad=assembly.with_suffix('.nvm')
            result=self.command([ROOT/'bin/nanoisa','asm',assembly,'-o',bad],None)
            if result.returncode:self.assertRegex(result.stderr,r'(?i)verif|owner|layout|type|stack|live|resource|consum')
            else:self.assertNotEqual(self.command([ROOT/'bin/nano_vm','--verify-only',bad],None).returncode,0)

    def test_actual128_estimated129_preserves_prior_output(self):
        original=(BASE/'original.nano').read_text()
        begin=original.index('fn main()');end=original.index('return 0',begin)
        for count in (98,99):
            text=original[:end]+''.join(f'let cap_{i}: int = {i}\n' for i in range(count))+original[end:]
            for compiler in self.compilers:
                module=self.module(compiler,text,'empty-cap-'+str(count));facts=self.facts(module)
                emit=OwnedEmit([OwnedAnalyze(facts,i).run() for i in range(2)],'nano',facts['layouts'])
                self.assertEqual(emit.nano_local_cost(emit.functions[0]),30+count)
                output=module.with_suffix('.nano');output.write_text('prior source')
                result=self.command([ROOT/'bin/nvm2hl','--language','nano',module,'-o',output],None)
                if count==99:
                    self.assertNotEqual(result.returncode,0);self.assertIn('128 local slots',result.stderr);self.assertEqual(output.read_text(),'prior source')
                else:
                    self.assertEqual(result.returncode,0,result.stderr)
                    rebuilt=self.module(compiler,output.read_text(),'empty128')
                    self.assertEqual(self.facts(rebuilt)['functions'][0]['locals'],128)

    def test_exact_profile_and_malformed_parameter_guards(self):
        facts=self.facts(self.module(self.compilers[0],(BASE/'original.nano').read_text(),'defensive'))
        for params in (None,[[]],[[8]],[[8,0]],[[8.0,1]],[[8,True]]):
            bad=copy.deepcopy(facts);bad['functions'][1]['params']=params
            for language in ('c','nano'):
                with self.subTest(params=params),self.assertRaises(Refusal):reconstruct_owned(bad,language)
        for change in ('shared','paths','assert','branch','third-layout'):
            bad=copy.deepcopy(facts)
            if change=='shared':bad['functions'][1]['param_modes']=[1]
            elif change=='paths':bad['ownership_version']=2
            elif change=='third-layout':bad['layouts'].append(dict(bad['layouts'][0],identity=2))
            else:bad['functions'][1]['code'][0]['op']='ASSERT' if change=='assert' else 'JMP'
            with self.subTest(change=change),self.assertRaises(Refusal):reconstruct_owned(bad,'nano')

    def test_canonical_and_advisory_independence(self):
        module=self.module(self.compilers[0],(BASE/'original.nano').read_text(),'transport');facts=self.facts(module)
        text=self.command([ROOT/'bin/nanoisa','dump',module]).stdout
        for name,content in (('canonical',text),('stripped','\n'.join(line for line in text.splitlines() if not line.startswith('.metadata'))+'\n')):
            assembly=self.work/(name+'.nasm');assembly.write_text(content);rebuilt=assembly.with_suffix('.nvm')
            self.command([ROOT/'bin/nanoisa','asm',assembly,'-o',rebuilt]);self.command([ROOT/'bin/nano_vm','--verify-only',rebuilt]);self.command([ROOT/'bin/nano_vm',rebuilt])
            self.assertEqual(self.facts(rebuilt),facts)
            for language in ('c','nano'):self.assertEqual(reconstruct_owned(self.facts(rebuilt),language),reconstruct_owned(facts,language))

if __name__=='__main__':unittest.main()
