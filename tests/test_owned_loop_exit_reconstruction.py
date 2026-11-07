"""I retain the unchanged structured break/continue/range owner fixtures."""
import copy
import json
import re
import struct
import zlib
import unittest
from tests import test_owned_nested_path_reconstruction as support
from nanoisa_owned_reconstruction import OwnedAnalyze, OwnedEmit, reconstruct_owned
from nanoisa_owned_exits import prove_exit_helper, prove_exit_origins, needs_loop_exits
from nanoisa_reconstruction import Expr, Function, INT, BOOL, Refusal
ROOT = support.ROOT



class LoopExitProof(unittest.TestCase):
    @staticmethod
    def ranked(body,bound=3):
        rank=Expr(INT,'local',1)
        return Function('helper',[('borrow',0,2)],INT,{1:INT},[],[
            ('store',1,Expr(INT,'constant',0)),
            ('while',Expr(BOOL,'binary','<',(rank,Expr(INT,'constant',bound))),body),
            ('return',rank)])

    @staticmethod
    def increment():
        rank=Expr(INT,'local',1);temp=Expr(INT,'temporary','inc')
        return [('let',temp,Expr(INT,'arithmetic','add',(rank,Expr(INT,'constant',1)))),('store',1,temp)]

    def test_every_continue_is_ranked_but_break_and_return_need_no_increment(self):
        for terminal in ([('break',)],[('return',Expr(INT,'constant',7))]):
            body=[('if',Expr(BOOL,'constant',True),self.increment()+[('continue',)],terminal)]
            fn=self.ranked(body);before=copy.deepcopy(fn)
            proof=prove_exit_helper(fn)
            self.assertEqual(fn,before);self.assertEqual(proof['loops'][0]['iterations'],3)
            self.assertLessEqual(proof['work'],4096)
        for body in ([('continue',)],self.increment()+self.increment()+[('continue',)],
                     [('if',Expr(BOOL,'constant',True),self.increment(),[]),('continue',)]):
            with self.assertRaises(Refusal):prove_exit_helper(self.ranked(body))

    def test_break_writes_invalidate_postloop_rank_and_work_stays_bounded(self):
        first=self.ranked([('store',1,Expr(INT,'constant',9)),('break',)])
        loop=first.body[1]
        first.body.insert(2,('while',loop[1],self.increment()+[('continue',)]))
        with self.assertRaisesRegex(Refusal,'dominating'):prove_exit_helper(first)
        with self.assertRaisesRegex(Refusal,'4096'):prove_exit_helper(self.ranked(self.increment()+[('continue',)],100))
        finite=prove_exit_helper(self.ranked([('break',)]))
        self.assertEqual(finite['loops'][0]['form'],'break-return-only')

    def test_stale_rank_snapshot_and_changed_bound_refuse(self):
        old=Expr(INT,'temporary','old');added=Expr(INT,'temporary','add')
        body=[('let',old,Expr(INT,'local',1)),('let',added,Expr(INT,'arithmetic','add',(old,Expr(INT,'constant',1)))),
              ('store',1,added),('store',1,added),('continue',)]
        with self.assertRaises(Refusal):prove_exit_helper(self.ranked(body))
        fn=self.ranked(self.increment()+[('store',2,Expr(INT,'constant',9)),('continue',)])
        fn.body.insert(0,('store',2,Expr(INT,'constant',3)))
        fn.body[2]=('while',Expr(BOOL,'binary','<',(Expr(INT,'local',1),Expr(INT,'local',2))),fn.body[2][2])
        with self.assertRaises(Refusal):prove_exit_helper(fn)

    def test_origin_cleanup_and_replacement_are_checked_on_all_exit_edges(self):
        owner=Expr(('owner',0),'temporary',1);value=Expr(INT,'temporary',2)
        pack=[('own_pack',owner,0,(Expr(INT,'constant',0),)),('own_store',0,0,owner)]
        for transfer in ('break','continue'):
            fn=Function('main',[],INT,{},[],[('while',Expr(BOOL,'constant',False),pack+[('own_unpack',0,0,(value,)),(transfer,)]),('return',Expr(INT,'constant',0))])
            before=copy.deepcopy(fn);prove_exit_origins(fn);self.assertEqual(fn,before)
            fn.body[0]=('while',Expr(BOOL,'constant',False),pack+[(transfer,)])
            with self.assertRaises(Refusal):prove_exit_origins(fn)


    def test_single_backedge_routing_requires_exact_internal_exit(self):
        def module(target,op='JMP'):
            return {'functions':[{'param_modes':[],'size':7,'code':[
                {'pc':0,'op':'LOAD_LOCAL'}, {'pc':1,'op':'JMP_FALSE','target':6},
                {'pc':2,'op':'NOP'}, {'pc':3,'op':op,'target':target},
                {'pc':4,'op':'NOP'}, {'pc':5,'op':'JMP','target':0}, {'pc':6,'op':'RET'}]},
                {'param_modes':[2],'size':1,'code':[{'pc':0,'op':'RET'}]}]}
        self.assertTrue(needs_loop_exits(module(6)))
        self.assertFalse(needs_loop_exits(module(4)))
        self.assertFalse(needs_loop_exits(module(6,'JMP_FALSE')))

    def test_break_only_bound_write_cannot_seed_a_later_rank_proof(self):
        rank,bound=Expr(INT,'local',1),Expr(INT,'local',2)
        cond=Expr(BOOL,'binary','<',(rank,bound))
        fn=Function('helper',[('borrow',0,2)],INT,{1:INT,2:INT},[],[
            ('store',1,Expr(INT,'constant',0)),('store',2,Expr(INT,'constant',3)),
            ('while',cond,[('store',2,Expr(INT,'constant',9)),('break',)]),
            ('while',cond,self.increment()+[('continue',)]),('return',rank)])
        with self.assertRaisesRegex(Refusal,'dominating'):prove_exit_helper(fn)

    def test_cached_facts_crossing_side_entry_and_zero_trip_initialization_refuse(self):
        facts=json.loads((ROOT/'docs/evidence/owned-loop-exits-baseline-20261019/source_borrow_loop_exits-nano_virt-facts.json').read_text())
        for case in ('crossing','side-entry','zero-trip-uninitialized'):
            bad=copy.deepcopy(facts);code={i['pc']:i for i in bad['functions'][0]['code']}
            # I use retained exact-PC facts for defensive API controls only;
            # these mutations are not claimed as original-verified executables.
            if case=='crossing':
                self.assertEqual(code[231]['op'],'JMP_FALSE');code[231]['target']=334
            elif case=='side-entry':
                self.assertEqual(code[40]['op'],'STORE_LOCAL');code[40].update(op='JMP',target=236)
            else:
                self.assertEqual(code[346]['op'],'LOAD_LOCAL');code[346].update(arg=10,args=[10])
            for language in ('c','nano'):
                with self.subTest(case=case,language=language),self.assertRaises(Refusal):reconstruct_owned(bad,language)


class OwnedLoopExitReconstruction(unittest.TestCase):
    setUpClass=support.OwnedNestedPathReconstruction.__dict__['setUpClass']
    command=support.OwnedNestedPathReconstruction.command
    facts=support.OwnedNestedPathReconstruction.facts
    native=support.OwnedNestedPathReconstruction.native
    module=support.OwnedNestedPathReconstruction.module
    qualify=support.OwnedNestedPathReconstruction.qualify
    selected_shadow=staticmethod(support.OwnedNestedPathReconstruction.selected_shadow)

    def test_unchanged_anchors_and_ordinary_local_measurement(self):
        for name,expected in (('source_borrow_loop_exits',[60,13]),('source_borrow_range_for',[108,14])):
            source=(ROOT/'tests/nanoisa/fixtures'/(name+'.nano')).read_text()
            for compiler in self.compilers:
                with self.subTest(fixture=name,provider=str(compiler)):
                    module=self.module(compiler,source,name)
                    facts=self.facts(module)
                    functions=[OwnedAnalyze(facts,index).run() for index in range(2)]
                    original=copy.deepcopy(functions)
                    emitter=OwnedEmit(functions,'nano',facts['layouts'])
                    self.assertEqual(functions,original)
                    self.assertEqual([emitter.nano_local_cost(fn) for fn in emitter.functions],expected)
                    shadow_cost=emitter.nano_local_cost(emitter.path_shadow_function())
                    for language in ('c','nano'):
                        self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',module.with_suffix('.'+language)])
                    output=module.with_suffix('.nano').read_text()
                    rebuilt=self.module(compiler,output,name+'-regenerated')
                    regenerated=self.facts(rebuilt)
                    actual=[fn['locals'] for fn in regenerated['functions']]
                    self.assertEqual(len(actual),2);self.assertEqual(actual,expected)
                    self.assertEqual(facts['layouts'],regenerated['layouts'])
                    selected=self.module(compiler,self.selected_shadow(output),name+'-shadow')
                    shadow=[fn['locals'] for fn in self.facts(selected)['functions']]
                    self.assertEqual(shadow,[shadow_cost,expected[1]])
                    print('I measure loop exits '+json.dumps({'fixture':name,'provider':str(compiler),
                        'original':[fn['locals'] for fn in facts['functions']], 'actual':actual,
                        'shadow':shadow,'proof':emitter.control_proof,'plan':emitter.reassignment_plans}),flush=True)


    def test_unchanged_native_roundtrips_and_selected_shadows(self):
        for name in ('source_borrow_loop_exits','source_borrow_range_for'):
            text=(ROOT/'tests/nanoisa/fixtures'/(name+'.nano')).read_text()
            for compiler in self.compilers:
                module=self.module(compiler,text,name+'-native')
                original=module.with_suffix('.original.c')
                self.command([ROOT/'bin/nvm2c',module,'-o',original]);self.native(original,0)
                self.qualify(module)
                output=module.with_suffix('.nano')
                self.command([ROOT/'bin/nvm2hl','--language','nano',module,'-o',output])
                shadow=self.module(compiler,self.selected_shadow(output.read_text()),name+'-native-shadow')
                native=shadow.with_suffix('.c');self.command([ROOT/'bin/nvm2c',shadow,'-o',native]);self.native(native,0)

    def test_total_division_subtraction_and_once_only_operand_calls(self):
        text=(ROOT/'tests/nanoisa/fixtures/source_borrow_loop_exits.nano').read_text()
        before,main=text.split('fn main()',1)
        marker=' let Counter { value, active } = owner'
        main=main.replace(marker,""" assert (== (/ (bump &mut owner) (bump &mut owner)) 0)
 let low: int = (+ -9223372036854775807 -1)
 let high: int = 9223372036854775807
 assert (== (/ low -1) low)
 assert (== (/ low 0) 0)
 assert (== (/ -7 3) -2)
 assert (== (/ 7 -3) -2)
 assert (== (- low 1) high)
"""+marker).replace('assert (== value 2) assert active','assert (== value 4) assert active')
        for compiler in self.compilers:
            module=self.module(compiler,before+'fn main()'+main,'exit-div-guards')
            self.qualify(module)

    def test_generated_128_and_129_are_measured_without_cap_changes(self):
        text=(ROOT/'tests/nanoisa/fixtures/source_borrow_range_for.nano').read_text()
        for count in (6,7):
            extra=''.join(f' let extra_{i}: int = 0\n assert (== extra_{i} 0)\n' for i in range(count))
            if count==6:extra += ' let spare_a: int = 0\n let spare_b: int = 0\n'
            source=text.replace(' for tail in (range 0 1)',extra+' for tail in (range 0 1)')
            for compiler in self.compilers:
                module=self.module(compiler,source,'exit-cap-'+str(count))
                facts=self.facts(module);emitter=OwnedEmit([OwnedAnalyze(facts,i).run() for i in range(2)],'nano',facts['layouts'])
                costs=[emitter.nano_local_cost(fn) for fn in emitter.functions]
                self.assertEqual(costs,[128 if count==6 else 129,14])
                output=module.with_suffix('.nano');output.write_text('prior checked source')
                command=self.command([ROOT/'bin/nvm2hl','--language','nano',module,'-o',output],None)
                if count==6:
                    self.assertEqual(command.returncode,0,command.stderr);generated=output.read_text()
                else:
                    self.assertNotEqual(command.returncode,0);self.assertIn('128 local slots',command.stderr)
                    self.assertEqual(output.read_text(),'prior checked source')
                    emitter.nano_local_cost=lambda fn:0;generated=emitter.run(0)
                regenerated=self.module(compiler,generated,'exit-cap-measured-'+str(count))
                dump=self.command([ROOT/'bin/nanoisa','dump',regenerated]).stdout
                actual=[int(n) for n in re.findall(r'(?m)^\.function \S+ \d+ (\d+) ',dump)]
                self.assertEqual(actual,costs)
                print('I measure exit local limit '+json.dumps({'provider':str(compiler),'ordinary':count==6,'actual':actual}),flush=True)

    def test_single_final_backedge_with_internal_break_uses_proved_profile(self):
        source=(ROOT/'tests/nanoisa/fixtures/owned_loop_single_break.nano').read_text()
        for compiler in self.compilers:
            module=self.module(compiler,source,'exit-single-backedge')
            facts=self.facts(module);self.assertTrue(needs_loop_exits(facts))
            emitter=OwnedEmit([OwnedAnalyze(facts,i).run() for i in range(2)],'nano',facts['layouts'])
            self.assertIsNotNone(emitter.control_proof)
            self.qualify(module)

    def test_flat_helper_still_gets_work_proof_and_scalar_operation_guards(self):
        text=(ROOT/'tests/nanoisa/fixtures/source_borrow_loop_exits.nano').read_text()
        start=text.index('fn bump(');end=text.index('shadow bump')
        flat='fn bump(view: &mut Counter) -> int { set view.value (+ view.value 1) return view.value }\n'
        for compiler in self.compilers:
            module=self.module(compiler,text[:start]+flat+text[end:],'exit-flat-helper')
            facts=self.facts(module);emitter=OwnedEmit([OwnedAnalyze(facts,i).run() for i in range(2)],'nano',facts['layouts'])
            self.assertIsNotNone(emitter.control_proof);self.assertEqual(emitter.control_proof['loops'],[])
            self.assertLessEqual(emitter.control_proof['work'],4096)
            self.qualify(module)
            bad=self.module(compiler,text[:start]+flat.replace('(+ view.value 1)','(- view.value -1)')+text[end:],'exit-flat-sub-helper')
            for language in ('c','nano'):
                output=bad.with_suffix('.'+language);output.write_text('prior checked source')
                self.assertNotEqual(self.command([ROOT/'bin/nvm2hl','--language',language,bad,'-o',output],None).returncode,0)
                self.assertEqual(output.read_text(),'prior checked source')

    def test_zero_trip_keeps_entry_initialization_and_live_owner(self):
        text=(ROOT/'tests/nanoisa/fixtures/source_borrow_loop_exits.nano').read_text()
        text=text.replace('while (< i 4)','while (< i 0)').replace('assert (== i 4)','assert (== i 0)').replace('assert (== value 2) assert active','assert (== value 0) assert active')
        for compiler in self.compilers:
            module=self.module(compiler,text,'exit-zero-trip')
            self.qualify(module)

    def test_original_reference_corruption_and_defensive_loop_edges_refuse(self):
        source=(ROOT/'tests/nanoisa/fixtures/source_borrow_loop_exits.nano').read_text()
        module=self.module(self.compilers[0],source,'exit-corrupt');facts=self.facts(module);raw=module.read_bytes()
        header,count=struct.unpack_from('<II',raw,20)
        sections={struct.unpack_from('<IIQQ',raw,header+24*n)[0]:struct.unpack_from('<IIQQ',raw,header+24*n)[2] for n in range(count)}
        ins=next(i for i in facts['functions'][0]['code'] if i['op']=='BORROW_LOCAL_EXCLUSIVE')
        changed=bytearray(raw);struct.pack_into('<H',changed,sections[6]+ins['pc']+3,255)
        struct.pack_into('<I',changed,36,zlib.crc32(changed[header:]));bad=self.work/'exit-bad-root.nvm';bad.write_bytes(changed)
        self.assertNotEqual(changed,raw)
        self.assertNotEqual(self.command([ROOT/'bin/nano_vm','--verify-only',bad],None).returncode,0)
        for language in ('c','nano'):
            output=bad.with_suffix('.'+language);output.write_text('prior checked source')
            self.assertNotEqual(self.command([ROOT/'bin/nvm2hl','--language',language,bad,'-o',output],None).returncode,0)
            self.assertEqual(output.read_text(),'prior checked source')
        for case in ('conditional-backedge','interior','outer-transfer','wrong-mode','lost-rank'):
            bad=copy.deepcopy(facts);code=bad['functions'][0]['code']
            backward=[i for i in code if i['op']=='JMP' and i['target']<i['pc']]
            if case=='conditional-backedge':backward[0]['op']='JMP_TRUE'
            elif case=='interior':backward[0]['target']=ins['pc']
            elif case=='outer-transfer':backward[1]['target']=backward[0]['target']
            elif case=='wrong-mode':bad['functions'][1]['param_modes']=[1]
            else:
                helper=bad['functions'][1]['code'];store=next(i for i in helper if i['op']=='STORE_LOCAL' and i['pc']>20);store['op']='POP'
            for language in ('c','nano'):
                with self.subTest(case=case,language=language),self.assertRaises(Refusal):reconstruct_owned(bad,language)


if __name__=='__main__':unittest.main()
