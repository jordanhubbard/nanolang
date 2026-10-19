"""I retain one immutable shared path and forward scalar helper diamonds."""
import copy
import hashlib
import json
import re
import struct
import zlib
import unittest
from tests import test_owned_nested_path_reconstruction as nested
from nanoisa_owned_reconstruction import OwnedAnalyze, OwnedEmit, reconstruct_owned
from nanoisa_reconstruction import Refusal
ROOT=nested.ROOT


class OwnedSharedControlReconstruction(unittest.TestCase):
    setUpClass=nested.OwnedNestedPathReconstruction.__dict__['setUpClass']
    command=nested.OwnedNestedPathReconstruction.command
    facts=nested.OwnedNestedPathReconstruction.facts
    native=nested.OwnedNestedPathReconstruction.native
    module=nested.OwnedNestedPathReconstruction.module
    qualify=nested.OwnedNestedPathReconstruction.qualify
    selected_shadow=staticmethod(nested.OwnedNestedPathReconstruction.selected_shadow)

    def anchor(self):
        return (ROOT/'tests/nanoisa/fixtures/source_borrow_control_shared.nano').read_text()

    def test_unchanged_anchor_and_whole_tree_shadow_measurement(self):
        for compiler in self.compilers:
            module=self.module(compiler,self.anchor(),'shared-control-anchor')
            facts=self.facts(module)
            self.assertEqual(facts['ownership_version'],2)
            self.assertEqual(facts['functions'][1]['param_modes'],[1])
            functions=[OwnedAnalyze(facts,i).run() for i in range(2)]
            self.assertTrue(any(stmt[0]=='if' for stmt in functions[1].body))
            emitter=OwnedEmit(functions,'nano',facts['layouts'])
            costs=[emitter.nano_local_cost(fn) for fn in emitter.functions]
            shadow_cost=emitter.nano_local_cost(emitter.path_shadow_function())
            print('I measure shared control '+json.dumps({'provider':str(compiler),
                  'original':[fn['locals'] for fn in facts['functions']],
                  'generated':costs,'shadow':shadow_cost}),flush=True)
            native=module.with_suffix('.original.c')
            self.command([ROOT/'bin/nvm2c',module,'-o',native]);self.native(native,0)
            self.qualify(module)
            selected=self.selected_shadow(reconstruct_owned(facts,'nano'))
            for producer in self.compilers:
                rebuilt=self.module(producer,selected,'shared-control-shadow')
                actual=self.facts(rebuilt)
                self.assertEqual(len(actual['functions']),2)
                self.assertEqual([fn['locals'] for fn in actual['functions']],[shadow_cost,costs[1]])
                native=rebuilt.with_suffix('.c')
                self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native]);self.native(native,0)

    def test_constant_helper_keeps_exact_unused_shared_parameter(self):
        source=self.anchor().replace(
            " let mut result: int = 0\n if view.enabled { set result view.value } else { set result 1 }\n return result\n}",
            " return 1\n}",1).replace('(== result 9)','(== result 6)')
        self.assertNotEqual(source,self.anchor())
        for compiler in self.compilers:
            module=self.module(compiler,source,'shared-control-constant')
            facts=self.facts(module)
            self.assertEqual(facts['functions'][1]['param_modes'],[1])
            self.assertFalse(any(ins['op']=='REF_GET' for ins in facts['functions'][1]['code']))
            native=module.with_suffix('.original.c')
            self.command([ROOT/'bin/nvm2c',module,'-o',native]);self.native(native,0)
            # qualify retains strict -Wall/-Wextra/-Werror native builds of
            # both reconstructed routes, including the unused reference ABI.
            self.qualify(module)

    def test_total_bool_diamonds_preserve_both_paths(self):
        source=self.anchor().replace('fn read(view: &Leaf) -> int {','fn read(view: &Leaf) -> bool {').replace(
            ' let mut result: int = 0\n if view.enabled { set result view.value } else { set result 1 }\n return result',
            ' let mut result: bool = false\n if view.enabled { set result true } else { set result false }\n return result',1)
        source=source.replace('(< (read &root.left) 3)','(read &root.left)').replace(
            '(> (read &root.right) 1)','(read &root.right)').replace(
            '(+ result (read &root.left))','(+ result 2)').replace(
            '(+ result (read &root.right))','(+ result 1)')
        for compiler in self.compilers:
            module=self.module(compiler,source,'shared-control-bool')
            native=module.with_suffix('.original.c')
            self.command([ROOT/'bin/nvm2c',module,'-o',native]);self.native(native,0)
            self.qualify(module)

    def test_caller_inline_add_exact_endpoints(self):
        minimum='(+ -9223372036854775807 -1)'
        for case,expression,expected in (('high','(+ 9223372036854775807 1)',minimum),
                                         ('low',f'(+ {minimum} -1)','9223372036854775807')):
            source=self.anchor().split('fn main()')[0]+f"""fn main() -> int {{
 let left: Leaf = Leaf {{ value: 2, enabled: true }}
 let right: Leaf = Leaf {{ value: 4, enabled: false }}
 let root: Pair = Pair {{ left: left, right: right }}
 assert (== (read &root.left) 2)
 assert (== (read &root.right) 1)
 let endpoint: int = {expression}
 assert (== endpoint {expected})
 let Pair {{ left, right }} = root
 let Leaf {{ value, enabled }} = left
 assert (== value 2) assert enabled
 let Leaf {{ value, enabled }} = right
 assert (== value 4) assert (not enabled)
 return 0
}}
shadow main {{ assert true }}
"""
            for compiler in self.compilers:
                module=self.module(compiler,source,'shared-control-endpoint-'+case)
                native=module.with_suffix('.original.c')
                self.command([ROOT/'bin/nvm2c',module,'-o',native]);self.native(native,0)
                self.qualify(module)

    def test_actual_helper_effects_and_loops_remain_atomic_refusals(self):
        for case,statement in (('assert','assert true'),('loop','while false { set result 3 }')):
            source=self.anchor().replace(' if view.enabled',f' {statement}\n if view.enabled',1)
            for compiler in self.compilers:
                module=self.module(compiler,source,'shared-control-excluded-'+case)
                for language in ('c','nano'):
                    output=module.with_suffix('.'+language);output.write_text('prior checked source')
                    result=self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output],None)
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn('shared-helper diamonds',result.stderr)
                    self.assertEqual(output.read_text(),'prior checked source')

    def test_generated_caller_local_boundary_preserves_output(self):
        for count in (3,4):
            extra=' let mut extra: int = 0\n'+(' set extra (+ extra 1)\n'*count)+f' assert (== extra {count})\n'
            source=self.anchor().replace(' let mut index: int = 0',extra+' let mut index: int = 0')
            for compiler in self.compilers:
                module=self.module(compiler,source,'shared-control-bound'+str(count))
                facts=self.facts(module)
                emitter=OwnedEmit([OwnedAnalyze(facts,i).run() for i in range(2)],'nano',facts['layouts'])
                costs=[emitter.nano_local_cost(fn) for fn in emitter.functions]
                print('I measure shared control bound '+json.dumps({'count':count,'provider':str(compiler),'costs':costs}),flush=True)
                if count==3:
                    self.assertLessEqual(max(costs),128);self.qualify(module)
                else:
                    self.assertGreater(max(costs),128)
                    output=module.with_suffix('.nano');output.write_text('prior checked source')
                    result=self.command([ROOT/'bin/nvm2hl','--language','nano',module,'-o',output],None)
                    self.assertNotEqual(result.returncode,0);self.assertIn('128 local slots',result.stderr)
                    self.assertEqual(output.read_text(),'prior checked source')
                    # I measure excess storage diagnostically, never as ordinary CLI support.
                    emitter.nano_local_cost=lambda fn:0
                    diagnostic=self.module(compiler,emitter.run(0),'shared-control-bound-diagnostic')
                    assembly=self.command([ROOT/'bin/nanoisa','dump',diagnostic]).stdout
                    self.assertEqual([int(n) for n in re.findall(r'(?m)^\.function \S+ \d+ (\d+) ',assembly)],costs)

    def test_original_verifier_corruption_preserves_prior_outputs(self):
        module=self.module(self.compilers[0],self.anchor(),'shared-control-wire')
        facts=self.facts(module);raw=module.read_bytes()
        header,count=struct.unpack_from('<II',raw,20)
        sections={struct.unpack_from('<IIQQ',raw,header+24*n)[0]:struct.unpack_from('<IIQQ',raw,header+24*n)[2] for n in range(count)}
        helper=facts['functions'][1];base=sections[6]+facts['functions'][0]['size']
        read=next(ins for ins in helper['code'] if ins['op']=='REF_GET')
        branch=next(ins for ins in helper['code'] if ins['op']=='JMP_FALSE')
        for case,offset,payload in (
            ('reference-slot',base+read['pc']+1,struct.pack('<H',255)),
            ('interior-branch',base+branch['pc']+1,struct.pack('<I',branch['target']+1))):
            changed=bytearray(raw);changed[offset:offset+len(payload)]=payload
            self.assertNotEqual(changed,raw)
            struct.pack_into('<I',changed,36,zlib.crc32(changed[header:]))
            bad=self.work/(case+'.nvm');bad.write_bytes(changed)
            result=self.command([ROOT/'bin/nano_vm','--verify-only',bad],None)
            self.assertNotEqual(result.returncode,0)
            for language in ('c','nano'):
                output=bad.with_suffix('.'+language);output.write_text('prior checked source')
                result=self.command([ROOT/'bin/nvm2hl','--language',language,bad,'-o',output],None)
                self.assertNotEqual(result.returncode,0)
                self.assertEqual(output.read_text(),'prior checked source')

    def test_single_shared_call_pretest_stays_separate(self):
        source=self.anchor().replace(' while keep { set keep false assert (not keep) }',
                                    ' while (< (read &root.left) 0) { assert false }')
        for compiler in self.compilers:
            module=self.module(compiler,source,'shared-control-pretest-refusal')
            for language in ('c','nano'):
                output=module.with_suffix('.'+language);output.write_text('prior checked source')
                result=self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output],None)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('one exact three-path reference call per owned pretest',result.stderr)
                self.assertEqual(output.read_text(),'prior checked source')

    def test_old_two_shared_path_outputs_remain_byte_identical(self):
        source=(ROOT/'tests/nanoisa/fixtures/source_borrow_nested_shared.nano').read_text()
        module=self.module(self.compilers[0],source,'prior-two-shared')
        facts=self.facts(module)
        # I pin the final prior9b emitter, including its reviewed void-use fix.
        receipt=json.loads((ROOT/'docs/evidence/owned-shared-control-20261019/prior-profile-bytes.json').read_text())
        for language in ('c','nano'):
            expected=receipt['outputs'][language]['sha256']
            self.assertEqual(hashlib.sha256(reconstruct_owned(facts,language).encode()).hexdigest(),expected)

    def test_defensive_modes_effects_and_total_shadow_boundary(self):
        module=self.module(self.compilers[0],self.anchor(),'shared-control-defense')
        original=self.facts(module)
        for case in ('exclusive','wrong-slot','wrong-layout','helper-call','helper-assert','helper-loop','uninitialized-join','interior-target'):
            facts=copy.deepcopy(original);entry,helper=facts['functions']
            read=next(ins for ins in helper['code'] if ins['op']=='REF_GET')
            if case=='exclusive':helper['param_modes']=[2]
            elif case=='wrong-slot':read.update(arg=1,args=[1,read['args'][1]])
            elif case=='wrong-layout':helper['params'][0][1]=len(facts['layouts'])-1
            elif case=='helper-call':read['op']='CALL_REF'
            elif case=='helper-assert':read['op']='ASSERT'
            elif case=='uninitialized-join':
                helper['locals']+=1;helper['names'].append('');helper['local_types'].append([1,4294967295])
                stores=[ins for ins in helper['code'] if ins['op']=='STORE_LOCAL']
                stores[1].update(arg=2,args=[2])
                load=next(ins for ins in reversed(helper['code']) if ins['op']=='LOAD_LOCAL')
                load.update(arg=2,args=[2])
            elif case=='interior-target':
                branch=next(ins for ins in helper['code'] if ins['op']=='JMP_FALSE')
                branch['target']+=1
            else:
                branch=next(ins for ins in helper['code'] if ins['op']=='JMP')
                branch['target']=helper['code'][0]['pc']
            for language in ('c','nano'):
                with self.subTest(case=case,language=language):
                    with self.assertRaises(Refusal):reconstruct_owned(facts,language)
        functions=[OwnedAnalyze(original,i).run() for i in range(2)]
        functions[1].body.insert(0,('assert',functions[1].body[-1][1]))
        for language in ('c','nano'):
            with self.assertRaisesRegex(Refusal,'total observation-only'):
                OwnedEmit(functions,language,original['layouts'])


if __name__=='__main__':unittest.main()
