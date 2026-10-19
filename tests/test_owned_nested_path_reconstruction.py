"""I retain one exact live-tree path profile and measured shadow storage."""
import copy
import json
import struct
import zlib
import re
import unittest
from tests import test_owned_exclusive_reconstruction as support
from nanoisa_owned_reconstruction import OwnedAnalyze, OwnedEmit, reconstruct_owned
from nanoisa_reconstruction import Refusal, Expr, INT, BOOL
ROOT=support.ROOT

class OwnedNestedPathReconstruction(unittest.TestCase):
    setUpClass=support.OwnedExclusiveReconstruction.__dict__['setUpClass']
    command=support.OwnedExclusiveReconstruction.command
    facts=support.OwnedExclusiveReconstruction.facts
    native=support.OwnedExclusiveReconstruction.native

    def module(self,compiler,text=None,name='nested-path'):
        source=ROOT/'tests/nanoisa/fixtures/source_borrow_nested.nano'
        if text is not None:
            source=self.work/(name+'.nano');source.write_text(text)
        module=self.work/(name+'-'+compiler.name+'.nvm')
        self.command([compiler,source,'--emit-nvm','-o',module])
        self.command([ROOT/'bin/nano_vm','--verify-only',module])
        self.command([ROOT/'bin/nano_vm',module])
        return module

    @staticmethod
    def selected_shadow(text):
        def bounds(start):
            begin=text.index('{',start);end=begin+1;depth=1
            while depth:depth+=(text[end]=='{')-(text[end]=='}');end+=1
            return begin,end
        begin,end=bounds(text.index('shadow nlr_f1'))
        shadow=text[begin+1:end-1]
        begin,end=bounds(text.index('fn main()'))
        return text[:begin+1]+shadow+'\n return 0\n'+text[end-1:]

    def test_original_anchor_and_selected_shadow_local_measurement(self):
        for compiler in self.compilers:
            module=self.module(compiler);facts=self.facts(module)
            self.assertEqual(facts['ownership_version'],2)
            self.assertEqual(facts['functions'][1]['param_modes'],[2,2,1])
            functions=[OwnedAnalyze(facts,i).run() for i in range(2)]
            emitter=OwnedEmit(functions,'nano',facts['layouts'])
            original_costs=[emitter.nano_local_cost(fn) for fn in functions]
            costs=[emitter.nano_local_cost(fn) for fn in emitter.functions]
            self.assertEqual(original_costs,[140,31])
            self.assertEqual(costs,[127,31])
            self.assertEqual(sum(stmt[0]=='assert' for stmt in functions[0].body),sum(stmt[0]=='assert' for stmt in emitter.functions[0].body))
            shadow_cost=emitter.nano_local_cost(emitter.path_shadow_function())
            record={'provider':str(compiler),'original_locals':[f['locals'] for f in facts['functions']],
                    'estimated_locals':costs,'estimated_shadow_locals':shadow_cost}
            print('I measure nested-path storage '+json.dumps(record),flush=True)
            (self.work/('counts-'+compiler.name+'.json')).write_text(json.dumps(record,indent=2))
            c=module.with_suffix('.c');self.command([ROOT/'bin/nvm2hl','--language','c',module,'-o',c])
            nano=module.with_suffix('.nano');nano.write_text('prior checked source')
            if max(*costs,shadow_cost)>128:
                result=self.command([ROOT/'bin/nvm2hl','--language','nano',module,'-o',nano],None)
                self.assertNotEqual(result.returncode,0);self.assertIn('128 local slots',result.stderr)
                self.assertEqual(nano.read_text(),'prior checked source')
            else:self.command([ROOT/'bin/nvm2hl','--language','nano',module,'-o',nano])
            # I bypass only output preflight for diagnostic exact producer storage.
            emitter.nano_local_cost=lambda function:0
            text=emitter.run(0)
            for case,content,expected in (('entry',text,costs),('shadow',self.selected_shadow(text),[shadow_cost,costs[1]])):
                source=self.work/(case+'-diagnostic-'+compiler.name+'.nano');source.write_text(content)
                rebuilt=source.with_suffix('.nvm')
                self.command([compiler,source,'--emit-nvm','-o',rebuilt])
                self.command([ROOT/'bin/nano_vm','--verify-only',rebuilt]);self.command([ROOT/'bin/nano_vm',rebuilt])
                assembly=self.command([ROOT/'bin/nanoisa','dump',rebuilt]).stdout
                actual=[int(n) for n in re.findall(r'(?m)^\.function \S+ \d+ (\d+) ',assembly)]
                self.assertEqual(actual,expected)

    def qualify(self,module):
        facts=self.facts(module)
        functions=[OwnedAnalyze(facts,i).run() for i in range(2)]
        emitter=OwnedEmit(functions,'nano',facts['layouts'])
        costs=[emitter.nano_local_cost(fn) for fn in emitter.functions]
        for language in ('c','nano'):
            output=module.with_suffix('.'+language)
            self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output])
            if language=='c':self.native(output,0)
            else:
                for compiler in self.compilers:
                    rebuilt=output.with_suffix('.'+compiler.name+'.nvm')
                    self.command([compiler,output,'--emit-nvm','-o',rebuilt])
                    self.command([ROOT/'bin/nano_vm','--verify-only',rebuilt]);self.command([ROOT/'bin/nano_vm',rebuilt])
                    regenerated=self.facts(rebuilt)
                    self.assertEqual(regenerated['layouts'],facts['layouts'])
                    self.assertEqual([f['locals'] for f in regenerated['functions']],costs)
                    self.assertEqual([f['params'] for f in regenerated['functions']],[f['params'] for f in facts['functions']])
                    native=rebuilt.with_suffix('.c')
                    self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native]);self.native(native,0)
        return facts

    def test_unchanged_anchor_native_and_paired_roundtrips(self):
        for compiler in self.compilers:
            module=self.module(compiler)
            native=module.with_suffix('.original.c')
            self.command([ROOT/'bin/nvm2c',module,'-o',native]);self.native(native,0)
            self.qualify(module)

    def test_selected_full_tree_shadows_native(self):
        module=self.module(self.compilers[0]);text=reconstruct_owned(self.facts(module),'nano')
        source=self.selected_shadow(text)
        for compiler in self.compilers:
            selected=self.module(compiler,source,'selected-tree-shadow')
            self.assertEqual(self.facts(selected)['functions'][0]['locals'],108)
            native=selected.with_suffix('.c')
            self.command([ROOT/'bin/nvm2c',selected,'-o',native]);self.native(native,0)

    def test_adjacent_assertion_folding_is_narrow_and_cloned(self):
        target=Expr(BOOL,'temporary',100)
        value=Expr(BOOL,'binary','==',(Expr(INT,'local',1),Expr(INT,'constant',7)))
        pair=[('let',target,value),('assert',target)]
        self.assertEqual(OwnedEmit.fold_adjacent_assertions(pair),[('assert',value)])
        self.assertEqual(pair,[('let',target,value),('assert',target)])
        cases=[pair+[('assert',target)],
               [pair[0],('store',1,Expr(INT,'constant',8)),pair[1]],
               [pair[0],('if',Expr(BOOL,'constant',True),[pair[1]],[])],
               [('if',Expr(BOOL,'constant',True),[pair[0]],[]),pair[1]],
               [('let',target,Expr(BOOL,'call',1,())),pair[1]],
               [('let',target,Expr(BOOL,'arithmetic','add',(Expr(INT,'constant',1),Expr(INT,'constant',2)))),pair[1]],
               [('let',target,Expr(BOOL,'binary','==',(Expr(INT,'call',1,()),Expr(INT,'constant',7)))),pair[1]],
               [('path_read',target,0,(0,0),1),pair[1]]]
        for case in cases:self.assertEqual(OwnedEmit.fold_adjacent_assertions(case),case)
        module=self.module(self.compilers[0]);facts=self.facts(module)
        functions=[OwnedAnalyze(facts,i).run() for i in range(2)]
        original=copy.deepcopy(functions)
        c=OwnedEmit(functions,'c',facts['layouts']);before=c.run(0)
        nano=OwnedEmit(functions,'nano',facts['layouts'])
        self.assertEqual(functions,original)
        self.assertEqual(OwnedEmit(functions,'c',facts['layouts']).run(0),before)
        self.assertEqual(c.nano_local_cost(functions[0]),140)
        self.assertEqual(nano.nano_local_cost(nano.functions[0]),127)

    def test_shared_path_pretests_retain_zero_and_final_false_reads(self):
        text=(ROOT/'tests/nanoisa/fixtures/source_borrow_nested.nano').read_text()
        head,main=text.split('fn main()',1)
        for value in ('13','27','10','30'):
            name='left' if value in ('13','10') else 'right'
            main=main.replace(f' assert (== root.branch.{name}.value {value})\n','')
        call=' assert (== (change &mut root.branch.right &mut root.branch.left &root.other) 20)'
        main=main.replace(call,' while (< root.branch.left.value 13) {\n'+call+'\n }\n while (< root.branch.right.value 30) {\n'+call+'\n }')
        for compiler in self.compilers:
            module=self.module(compiler,head+'fn main()'+main,'path-pretests')
            facts=self.qualify(module)
            function=OwnedAnalyze(facts,0).run()
            loops=[stmt for stmt in function.body if stmt[0]=='while_observe']
            self.assertEqual(len(loops),2)
            self.assertTrue(all(sum(s[0]=='path_read' for s in stmt[2])==1 for stmt in loops))

    def test_actual_path_call_pretest_refuses_atomically(self):
        text=(ROOT/'tests/nanoisa/fixtures/source_borrow_nested.nano').read_text()
        head,main=text.split('fn main()',1)
        call='(change &mut root.branch.right &mut root.branch.left &root.other)'
        main=main.replace(' assert (== '+call+' 20)', ' while (== '+call+' 0) { assert false }')
        for compiler in self.compilers:
            module=self.module(compiler,head+'fn main()'+main,'path-call-pretest')
            for language in ('c','nano'):
                output=module.with_suffix('.'+language);output.write_text('prior checked source')
                result=self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output],None)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('complete three-reference caller region',result.stderr)
                self.assertEqual(output.read_text(),'prior checked source')

    def test_projected_calls_in_both_caller_branch_arms(self):
        text=(ROOT/'tests/nanoisa/fixtures/source_borrow_nested.nano').read_text()
        head,main=text.split('fn main()',1)
        for value in ('13','27','10','30'):
            name='left' if value in ('13','10') else 'right'
            main=main.replace(f' assert (== root.branch.{name}.value {value})\n','')
        call=' assert (== (change &mut root.branch.right &mut root.branch.left &root.other) 20)'
        for condition in ('root.branch.left.enabled','(not root.branch.left.enabled)'):
            branch=' if '+condition+' {\n'+call+'\n } else {\n'+call+'\n }'
            for compiler in self.compilers:
                module=self.module(compiler,head+'fn main()'+main.replace(call,branch),'path-branch')
                self.qualify(module)

    def test_later_false_projected_shared_flag_keeps_runtime_assertion(self):
        text=(ROOT/'tests/nanoisa/fixtures/source_borrow_nested.nano').read_text()
        text=text.replace('resource struct Other { value: int, enabled: bool }\n','').replace('Other','Leaf')
        head,main=text.split('fn main()',1)
        main=main.replace('(change &mut root.branch.right &mut root.branch.left &root.other)',
                          '(change &mut root.branch.left &mut root.other &root.branch.right)')
        from tests.test_owned_void_reconstruction import OwnedVoidReconstruction
        for compiler in self.compilers:
            source=self.work/'later-false-path.nano';source.write_text(head+'fn main()'+main)
            module=self.work/('later-false-path-'+compiler.name+'.nvm')
            self.command([compiler,source,'--emit-nvm','-o',module])
            OwnedVoidReconstruction.vm_failure(self,module)
            for language in ('c','nano'):
                output=module.with_suffix('.'+language)
                self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output])
                if language=='c':OwnedVoidReconstruction.native_failure(self,output,'Assertion failed')
                else:
                    for producer in self.compilers:
                        rebuilt=output.with_suffix('.'+producer.name+'.nvm')
                        self.command([producer,output,'--emit-nvm','-o',rebuilt])
                        OwnedVoidReconstruction.vm_failure(self,rebuilt)
                        native=rebuilt.with_suffix('.c')
                        self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native])
                        OwnedVoidReconstruction.native_failure(self,native,None)

    def test_full_tree_shadow_near_boundary_is_measured(self):
        for extra in (2,3):
            text=(ROOT/'tests/nanoisa/fixtures/source_borrow_nested.nano').read_text()
            declarations=', '.join(f'extra{i}: int' for i in range(extra))
            entries=', '.join(f'extra{i}: {i+41}' for i in range(extra))
            names=', '.join(f'extra{i}' for i in range(extra))
            text=text.replace('other: Other, count: int }','other: Other, count: int, '+declarations+' }')
            text=text.replace('count: 7, branch: pair }','count: 7, branch: pair, '+entries+' }')
            text=text.replace('Tree { count, other, branch }','Tree { count, other, branch, '+names+' }')
            for compiler in self.compilers:
                module=self.module(compiler,text,'tree-shadow-bound'+str(extra));facts=self.facts(module)
                functions=[OwnedAnalyze(facts,i).run() for i in range(2)]
                emitter=OwnedEmit(functions,'nano',facts['layouts'])
                cost=emitter.nano_local_cost(emitter.path_shadow_function())
                self.assertEqual(cost,108+8*extra)
                if cost>128:
                    with self.assertRaisesRegex(Refusal,'tree shadow'):emitter.path_shadow()
                    output=module.with_suffix('.nano');output.write_text('prior checked source')
                    self.assertNotEqual(self.command([ROOT/'bin/nvm2hl','--language','nano',module,'-o',output],None).returncode,0)
                    self.assertEqual(output.read_text(),'prior checked source')
                emitter.nano_local_cost=lambda function:0
                selected=self.module(compiler,self.selected_shadow(emitter.run(0)),'diagnostic-tree-shadow'+str(extra))
                assembly=self.command([ROOT/'bin/nanoisa','dump',selected]).stdout
                self.assertRegex(assembly,rf'(?m)^\.function \S+ 0 {cost} ')
                if cost>128:self.assertNotEqual(self.command([ROOT/'bin/nanoisa_hl_facts',selected],None).returncode,0)
                else:self.assertEqual(self.facts(selected)['functions'][0]['locals'],cost)

    def test_false_first_tree_witness_refuses_both_outputs(self):
        text=(ROOT/'tests/nanoisa/fixtures/source_borrow_nested.nano').read_text()
        head,main=text.split('fn main()',1)
        main=main.replace('Other { value: 3, enabled: true }','Other { value: 3, enabled: false }')
        from tests.test_owned_void_reconstruction import OwnedVoidReconstruction
        for compiler in self.compilers:
            source=self.work/'false-tree-witness.nano';source.write_text(head+'fn main()'+main)
            module=self.work/('false-tree-witness-'+compiler.name+'.nvm')
            self.command([compiler,source,'--emit-nvm','-o',module])
            OwnedVoidReconstruction.vm_failure(self,module)
            for language in ('c','nano'):
                output=self.work/('prior.'+language);output.write_text('prior checked source')
                result=self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output],None)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('true actual tree shared-BOOL assertion witness',result.stderr)
                self.assertEqual(output.read_text(),'prior checked source')

    def test_defensive_path_identity_overlap_and_scope_refusals(self):
        facts=self.facts(self.module(self.compilers[0]))
        for case in ('missing-fields','wrong-index','inconsistent','scalar-step','wrong-leaf','equal','prefix','root-local','deep','mode','v3','missing-layout'):
            changed=copy.deepcopy(facts);entry,helper=changed['functions']
            borrows=[i for i in entry['code'] if i['op'].startswith('BORROW_PATH')][:3]
            if case=='missing-fields':del borrows[0]['path_fields']
            elif case=='wrong-index':borrows[0]['args'][2]=256
            elif case=='inconsistent':borrows[0]['path_fields']=[1]
            elif case=='scalar-step':
                for ins in entry['code']:
                    if ins['op'].startswith('BORROW_PATH') and ins['args'][2]==borrows[0]['args'][2]:ins['path_fields']=[2,0]
            elif case=='wrong-leaf':helper['params'][0]=helper['params'][2][:];helper['local_types'][0]=helper['params'][2][:]
            elif case=='equal':borrows[1]['args'][2]=borrows[0]['args'][2];borrows[1]['path_fields']=borrows[0]['path_fields'][:]
            elif case=='prefix':borrows[1]['path_fields']=[0]
            elif case=='root-local':borrows[1]['args'][1]=0
            elif case=='deep':borrows[0]['path_fields']=[0,0,0]
            elif case=='mode':helper['param_modes']=[2,1,2]
            elif case=='v3':changed['ownership_version']=3
            elif case=='missing-layout':del changed['layouts'][0]['field_types']
            for language in ('c','nano'):
                with self.subTest(case=case,language=language),self.assertRaises(Refusal):reconstruct_owned(changed,language)

    def test_original_verifier_rejects_path_operand_corruption(self):
        module=self.module(self.compilers[0]);facts=self.facts(module);raw=module.read_bytes()
        header,count=struct.unpack_from('<II',raw,20)
        sections={struct.unpack_from('<IIQQ',raw,header+24*n)[0]:struct.unpack_from('<IIQQ',raw,header+24*n)[2] for n in range(count)}
        borrows=[i for i in facts['functions'][0]['code'] if i['op'].startswith('BORROW_PATH')][:3]
        base=sections[6]
        cases={'path-index':(base+borrows[0]['pc']+5,struct.pack('<I',255)),
               'equal-exclusive':(base+borrows[1]['pc']+5,struct.pack('<I',borrows[0]['args'][2])),
               'wrong-root':(base+borrows[0]['pc']+3,struct.pack('<H',0)),
               'reference-gap':(base+borrows[1]['pc']+1,struct.pack('<H',9))}
        for case,(offset,payload) in cases.items():
            changed=bytearray(raw);changed[offset:offset+len(payload)]=payload
            self.assertNotEqual(changed,raw);struct.pack_into('<I',changed,36,zlib.crc32(changed[header:]))
            bad=self.work/(case+'.nvm');bad.write_bytes(changed)
            self.assertNotEqual(self.command([ROOT/'bin/nano_vm','--verify-only',bad],None).returncode,0)
            for language in ('c','nano'):
                output=self.work/('prior.'+language);output.write_text('prior checked source')
                self.assertNotEqual(self.command([ROOT/'bin/nvm2hl','--language',language,bad,'-o',output],None).returncode,0)
                self.assertEqual(output.read_text(),'prior checked source')

if __name__=='__main__':unittest.main()
