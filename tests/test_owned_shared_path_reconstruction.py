"""I measure exact shared path aliases and the generated scalar MUL loop first."""
import copy
import hashlib
import struct
import zlib
import json
import re
import unittest
from tests import test_owned_exclusive_reconstruction as support
from tests import test_owned_nested_path_reconstruction as path_support
from nanoisa_owned_reconstruction import OwnedAnalyze, OwnedEmit, reconstruct_owned
from nanoisa_reconstruction import Refusal, Expr, INT
ROOT = support.ROOT

class OwnedSharedPathReconstruction(unittest.TestCase):
    setUpClass = support.OwnedExclusiveReconstruction.__dict__['setUpClass']
    command = support.OwnedExclusiveReconstruction.command
    facts = support.OwnedExclusiveReconstruction.facts
    native = support.OwnedExclusiveReconstruction.native

    def module(self, compiler, text=None, name='shared-tree', execute=True):
        source = ROOT/'tests/nanoisa/fixtures/source_borrow_nested_shared.nano'
        if text is not None:
            source = self.work/(name+'.nano');source.write_text(text)
        module = self.work/(name+'-'+compiler.name+'.nvm')
        self.command([compiler, source, '--emit-nvm', '-o', module])
        self.command([ROOT/'bin/nano_vm', '--verify-only', module])
        if execute:self.command([ROOT/'bin/nano_vm', module])
        return module

    def qualify(self, module):
        native=module.with_suffix('.original.c')
        self.command([ROOT/'bin/nvm2c',module,'-o',native]);self.native(native,0)
        return path_support.OwnedNestedPathReconstruction.qualify(self,module)

    @staticmethod
    def source():
        return (ROOT/'tests/nanoisa/fixtures/source_borrow_nested_shared.nano').read_text()

    def test_original_anchor_generated_loop_and_exact_storage(self):
        source = ROOT / 'tests/nanoisa/fixtures/source_borrow_nested_shared.nano'
        for compiler in self.compilers:
            module = self.work / ('shared-tree-' + compiler.name + '.nvm')
            self.command([compiler, source, '--emit-nvm', '-o', module])
            self.command([ROOT/'bin/nano_vm', '--verify-only', module])
            self.command([ROOT/'bin/nano_vm', module])
            facts = self.facts(module)
            self.assertEqual(facts['ownership_version'], 2)
            self.assertEqual(facts['functions'][1]['param_modes'], [1, 1])
            functions = [OwnedAnalyze(facts, i).run() for i in range(2)]
            emitter = OwnedEmit(functions, 'nano', facts['layouts'])
            costs = [emitter.nano_local_cost(fn) for fn in emitter.functions]
            shadow_cost = emitter.nano_local_cost(emitter.path_shadow_function())
            record = {'provider':str(compiler), 'original_locals':[f['locals'] for f in facts['functions']],
                      'estimated_locals':costs, 'estimated_shadow_locals':shadow_cost}
            print('I measure shared-path storage ' + json.dumps(record), flush=True)
            (self.work/('counts-' + compiler.name + '.json')).write_text(json.dumps(record, indent=2))
            output = module.with_suffix('.nano')
            self.command([ROOT/'bin/nvm2hl', '--language', 'nano', module, '-o', output])
            text = output.read_text()
            self.assertEqual(sum(line.startswith('fn ') for line in text.splitlines()), 2)
            self.assertNotIn('nlr_i64_mul', text)
            self.assertIn('_mul_remaining', text)
            for kind, content, expected in (
                ('entry', text, costs),
                ('shadow', path_support.OwnedNestedPathReconstruction.selected_shadow(text), [shadow_cost, costs[1]]),
            ):
                generated = self.work/(kind + '-' + compiler.name + '.nano')
                generated.write_text(content)
                rebuilt = generated.with_suffix('.nvm')
                self.command([compiler, generated, '--emit-nvm', '-o', rebuilt])
                self.command([ROOT/'bin/nano_vm', '--verify-only', rebuilt])
                self.command([ROOT/'bin/nano_vm', rebuilt])
                assembly = self.command([ROOT/'bin/nanoisa', 'dump', rebuilt]).stdout
                actual = [int(n) for n in re.findall(r'(?m)^\.function \S+ \d+ (\d+) ', assembly)]
                self.assertEqual(actual, expected)

    def test_unchanged_alias_and_swapped_paths_native_roundtrips(self):
        for compiler in self.compilers:self.qualify(self.module(compiler))

    def test_total_mul_endpoints_preserve_original_results(self):
        cases = (
            ('min-negative-one', 'let low: int = (+ -9223372036854775807 -1) return (* low -1)', '(+ -9223372036854775807 -1)'),
            ('min-two', 'let low: int = (+ -9223372036854775807 -1) return (* low 2)', '0'),
            ('max-square', 'return (* 9223372036854775807 9223372036854775807)', '1'),
            ('negative-odd', 'return (* -3 -7)', '21'),
            ('negative-positive', 'return (* -3 7)', '-21'),
            ('high-product', 'return (* 4294967296 4294967296)', '0'),
            ('zero-one', 'return (+ (* 0 1) (* 1 1))', '1'),
        )
        for name, body, expected in cases:
            text=self.source().replace('return (+ (* first.value 10) second.value)',body)
            for result in ('22','27','72'):text=text.replace(') '+result+')',') '+expected+')')
            for compiler in self.compilers:
                self.qualify(self.module(compiler,text,name))

    def test_either_shared_parameter_supplies_assertion_provenance(self):
        text=self.source().replace('assert second.enabled','assert first.enabled assert second.enabled')
        for compiler in self.compilers:self.qualify(self.module(compiler,text,'both-provenance'))

    def test_distinct_nominal_endpoints_and_unequal_leaf_fields(self):
        text=self.source().replace('resource struct Pair', 'resource struct Other { value: int, enabled: bool, stamp: int }\nresource struct Pair')
        text=text.replace('right: Leaf','right: Other').replace('second: &Leaf','second: &Other')
        text=text.replace('let two: Leaf = Leaf { value: 7, enabled: true }','let two: Other = Other { value: 7, enabled: true, stamp: 41 }')
        text=text.replace(' assert (== (read &root.branch.left &root.branch.left) 22)\n','').replace(' assert (== (read &root.branch.right &root.branch.left) 72)\n','')
        text=text.replace('let Leaf { value, enabled } = right','let Other { value, enabled, stamp } = right\n assert (== stamp 41)')
        for compiler in self.compilers:self.qualify(self.module(compiler,text,'nominal-fields'))

    def test_generated_whole_tree_shadow_native(self):
        module=self.module(self.compilers[0])
        text=reconstruct_owned(self.facts(module),'nano')
        selected=path_support.OwnedNestedPathReconstruction.selected_shadow(text)
        for compiler in self.compilers:
            rebuilt=self.module(compiler,selected,'selected-shared-shadow')
            self.assertEqual(self.facts(rebuilt)['functions'][0]['locals'],74)
            native=rebuilt.with_suffix('.c')
            self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native]);self.native(native,0)

    def test_repeated_mul_generated_local_boundary(self):
        for count in (4,5):
            body=' '.join(f'let v{i}: int = (* first.value 10)' for i in range(count))+' return (+ v'+str(count-1)+' second.value)'
            text=self.source().replace('return (+ (* first.value 10) second.value)',body)
            for compiler in self.compilers:
                module=self.module(compiler,text,'mul-bound'+str(count));facts=self.facts(module)
                functions=[OwnedAnalyze(facts,i).run() for i in range(2)]
                emitter=OwnedEmit(functions,'nano',facts['layouts'])
                costs=[emitter.nano_local_cost(fn) for fn in emitter.functions]
                self.assertLess(facts['functions'][1]['locals'],128)
                self.assertEqual(costs[1],12+29*count)
                output=module.with_suffix('.nano');output.write_text('prior checked source')
                result=self.command([ROOT/'bin/nvm2hl','--language','nano',module,'-o',output],None)
                if costs[1]>128:
                    self.assertNotEqual(result.returncode,0);self.assertIn('128 local slots',result.stderr)
                    self.assertEqual(output.read_text(),'prior checked source')
                else:self.assertEqual(result.returncode,0,result.stderr)
                emitter.nano_local_cost=lambda fn:0
                diagnostic=self.module(compiler,emitter.run(0),'diagnostic-mul-bound'+str(count))
                assembly=self.command([ROOT/'bin/nanoisa','dump',diagnostic]).stdout
                actual=[int(n) for n in re.findall(r'(?m)^\.function \S+ \d+ (\d+) ',assembly)]
                self.assertEqual(actual,costs)
                if costs[1]>128:self.assertNotEqual(self.command([ROOT/'bin/nanoisa_hl_facts',diagnostic],None).returncode,0)

    def test_false_first_and_later_shared_assertions(self):
        from tests.test_owned_void_reconstruction import OwnedVoidReconstruction
        for first in (True,False):
            head,main=self.source().split('fn main()',1)
            value='2' if first else '7'
            main=main.replace('value: '+value+', enabled: true','value: '+value+', enabled: false')
            for compiler in self.compilers:
                module=self.module(compiler,head+'fn main()'+main,'false-shared'+str(first),False)
                OwnedVoidReconstruction.vm_failure(self,module)
                for language in ('c','nano'):
                    output=module.with_suffix('.'+language);output.write_text('prior checked source')
                    result=self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output],None)
                    if first:
                        self.assertNotEqual(result.returncode,0);self.assertIn('shared-BOOL assertion witness',result.stderr)
                        self.assertEqual(output.read_text(),'prior checked source')
                    elif language=='c':
                        self.assertEqual(result.returncode,0,result.stderr)
                        OwnedVoidReconstruction.native_failure(self,output,'Assertion failed')
                    else:
                        self.assertEqual(result.returncode,0,result.stderr)
                        for producer in self.compilers:
                            rebuilt=output.with_suffix('.'+producer.name+'.nvm')
                            self.command([producer,output,'--emit-nvm','-o',rebuilt]);OwnedVoidReconstruction.vm_failure(self,rebuilt)
                            native=rebuilt.with_suffix('.c');self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native])
                            OwnedVoidReconstruction.native_failure(self,native,None)

    def test_v2_exclusive_outputs_remain_byte_identical(self):
        source=(ROOT/'tests/nanoisa/fixtures/source_borrow_nested.nano').read_text()
        module=self.module(self.compilers[0],source,'exclusive-golden');facts=self.facts(module)
        rows=json.loads((ROOT/'docs/evidence/owned-nested-paths-20261019/artifacts.json').read_text())
        for language in ('c','nano'):
            suffix='/nano-owned-branches-8w6fapim/nested-path-nano_virt.'+language
            expected=next(row['sha256'] for row in rows if row['path'].endswith(suffix))
            self.assertEqual(hashlib.sha256(reconstruct_owned(facts,language).encode()).hexdigest(),expected)

    def test_defensive_modes_paths_fields_and_inline_placement(self):
        facts=self.facts(self.module(self.compilers[0]))
        for case in ('mode','count','slot','root','nominal','write','missing-path','unknown-witness'):
            changed=copy.deepcopy(facts);entry,helper=changed['functions']
            borrows=[i for i in entry['code'] if i['op']=='BORROW_PATH_SHARED'][:2]
            if case=='mode':helper['param_modes']=[2,1]
            elif case=='count':helper['params'].pop()
            elif case=='slot':borrows[1]['args'][0]+=1
            elif case=='root':borrows[1]['args'][1]=0
            elif case=='nominal':helper['params'][0]=[8,2];helper['local_types'][0]=[8,2]
            elif case=='write':next(i for i in helper['code'] if i['op']=='REF_GET')['op']='REF_SET'
            elif case=='missing-path':del borrows[0]['path_fields']
            elif case=='unknown-witness':next(i for i in entry['code'] if i['op']=='PUSH_BOOL')['arg']=0
            for language in ('c','nano'):
                with self.subTest(case=case,language=language),self.assertRaises(Refusal):reconstruct_owned(changed,language)
        functions=[OwnedAnalyze(facts,i).run() for i in range(2)]
        for n,stmt in enumerate(functions[1].body):
            if stmt[0]=='let' and stmt[2].kind=='arithmetic' and stmt[2].value=='mul':
                inner=stmt[2]
                functions[1].body[n]=('let',stmt[1],Expr(INT,'arithmetic','add',(inner,Expr(INT,'constant',1))))
                break
        with self.assertRaisesRegex(Refusal,'explicit inline statement boundary'):OwnedEmit(functions,'nano',facts['layouts'])

    def test_original_verifier_rejects_reference_corruption(self):
        module=self.module(self.compilers[0]);facts=self.facts(module);raw=module.read_bytes()
        header,count=struct.unpack_from('<II',raw,20)
        sections={struct.unpack_from('<IIQQ',raw,header+24*n)[0]:struct.unpack_from('<IIQQ',raw,header+24*n)[2] for n in range(count)}
        borrows=[i for i in facts['functions'][0]['code'] if i['op']=='BORROW_PATH_SHARED'][:2]
        base=sections[6]
        cases={'index':(base+borrows[0]['pc']+5,struct.pack('<I',255)),
               'root':(base+borrows[0]['pc']+3,struct.pack('<H',0)),
               'slot':(base+borrows[1]['pc']+1,struct.pack('<H',9))}
        for name,(offset,payload) in cases.items():
            changed=bytearray(raw);changed[offset:offset+len(payload)]=payload
            self.assertNotEqual(changed,raw);struct.pack_into('<I',changed,36,zlib.crc32(changed[header:]))
            bad=self.work/(name+'.nvm');bad.write_bytes(changed)
            self.assertNotEqual(self.command([ROOT/'bin/nano_vm','--verify-only',bad],None).returncode,0)
            for language in ('c','nano'):
                output=self.work/('prior.'+language);output.write_text('prior checked source')
                self.assertNotEqual(self.command([ROOT/'bin/nvm2hl','--language',language,bad,'-o',output],None).returncode,0)
                self.assertEqual(output.read_text(),'prior checked source')

if __name__ == '__main__':
    unittest.main()
