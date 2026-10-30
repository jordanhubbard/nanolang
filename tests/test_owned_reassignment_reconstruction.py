"""I retain exact owner generations and bounded original-slot rendering."""
import copy
import json
import re
import struct
import zlib
import unittest
from tests import test_owned_nested_path_reconstruction as support
from nanoisa_owned_reconstruction import OwnedAnalyze, OwnedEmit, reconstruct_owned
from nanoisa_owned_origins import prove_owner_reassignment
from nanoisa_owned_rendering import plan_reassignment_nano
from nanoisa_reconstruction import Expr, Function, INT, BOOL, Refusal
ROOT = support.ROOT
BASELINE = ROOT/'docs/evidence/owned-reassignment-baseline-20261019'


class OwnerRenderingProof(unittest.TestCase):
    def test_pruned_source_omits_only_unused_scalar_copies(self):
        module=json.loads((BASELINE/'true-nano_virt-facts.json').read_text())
        functions=[OwnedAnalyze(module,i).run() for i in range(2)]
        before=copy.deepcopy(functions)
        emitter=OwnedEmit(functions,'nano',module['layouts'])
        source=emitter.run(0)
        self.assertEqual(functions,before)
        self.assertEqual([emitter.nano_local_cost(fn) for fn in emitter.functions],[127,9])
        self.assertEqual(emitter.reassignment_plans[0]['covered_statements'],211)
        self.assertEqual(emitter.reassignment_plans[0]['original_statements'],211)
        emitter.function=emitter.functions[0]
        omitted=0
        for stmt in emitter.walk_statements(emitter.function.body):
            if stmt[0]=='own_unpack_pruned':
                for target,keep in zip(stmt[3],stmt[4]):
                    if not keep:
                        omitted+=1
                        self.assertIn(target.tag,(INT,BOOL))
                        self.assertNotRegex(source,r'\blet (?:mut )?'+re.escape(emitter.expression(target))+r'\b')
        self.assertEqual(omitted,18)
        self.assertEqual(source.count('let mut nlr_rs_a: int = 0'),2)
        self.assertEqual(source.count('let mut nlr_rs_boundary: int = 0'),2)
        self.assertEqual(sum(s[0]=='assert' for s in emitter.walk_statements(functions[0].body)),
                         sum(s[0]=='assert' for s in emitter.walk_statements(emitter.functions[0].body)))


    @staticmethod
    def origin_function(body):
        return Function('main',[],INT,{},[],body)

    @staticmethod
    def pack(pc, slot, value=0):
        target=Expr(('owner',0),'temporary',pc)
        return [('own_pack',target,0,(Expr(INT,'constant',value),)),('own_store',slot,0,target)]

    @staticmethod
    def unpack(pc, slot):
        return ('own_unpack',slot,0,(Expr(INT,'temporary',pc),))

    def test_origin_restoration_is_immutable_and_generation_exact(self):
        a=Expr(('owner',0),'temporary',20);b=Expr(('owner',0),'temporary',21)
        body=self.pack(1,0)+[('while',Expr(BOOL,'constant',False),[
            ('own_move',a,0),('own_store',1,0,a),('own_move',b,1),('own_assign',0,0,b)]),
            self.unpack(30,0),('return',Expr(INT,'constant',0))]
        function=self.origin_function(body);before=copy.deepcopy(function)
        self.assertEqual(prove_owner_reassignment(function),frozenset({0}))
        self.assertEqual(function,before)
        replacement=Expr(('owner',0),'temporary',22)
        changed=copy.deepcopy(body)
        changed[2]=('while',Expr(BOOL,'constant',False),[
            self.unpack(23,0),('own_pack',replacement,0,(Expr(INT,'constant',0),)),('own_assign',0,0,replacement)])
        with self.assertRaisesRegex(Refusal,'loop-body'):
            prove_owner_reassignment(self.origin_function(changed))

    def test_origin_branch_replacement_and_same_layout_swap_refuse(self):
        a=Expr(('owner',0),'temporary',20);b=Expr(('owner',0),'temporary',21);c=Expr(('owner',0),'temporary',22)
        prefix=self.pack(1,0)+self.pack(2,1)
        swap=[('own_move',a,0),('own_store',2,0,a),('own_move',b,1),('own_assign',0,0,b),
              ('own_move',c,2),('own_assign',1,0,c)]
        body=prefix+[('while',Expr(BOOL,'constant',False),swap),self.unpack(30,0),self.unpack(31,1),('return',Expr(INT,'constant',0))]
        with self.assertRaisesRegex(Refusal,'restoration'):
            prove_owner_reassignment(self.origin_function(body))
        replacement=Expr(('owner',0),'temporary',40)
        body=self.pack(1,0)+[('if',Expr(BOOL,'constant',True),[
            self.unpack(41,0),('own_pack',replacement,0,(Expr(INT,'constant',0),)),('own_assign',0,0,replacement)],
            [self.unpack(42,0),('return',Expr(INT,'constant',0))]),self.unpack(43,0),('return',Expr(INT,'constant',0))]
        with self.assertRaisesRegex(Refusal,'surviving incoming'):
            prove_owner_reassignment(self.origin_function(body))

    def test_origin_declared_slot_does_not_resurrect_consumed_operand(self):
        temp=Expr(('owner',0),'temporary',1)
        missing=[('own_pack',temp,0,(Expr(INT,'constant',0),)),('own_assign',0,0,temp)]
        with self.assertRaisesRegex(Refusal,'dominating'):
            prove_owner_reassignment(self.origin_function(missing))
        stale=self.pack(1,0)+[self.unpack(2,0),('own_assign',0,0,temp)]
        with self.assertRaisesRegex(Refusal,'unconsumed'):
            prove_owner_reassignment(self.origin_function(stale))
        first=self.pack(1,0)+[self.unpack(2,0)]
        later=Expr(('owner',0),'temporary',3)
        branch=[('if',Expr(BOOL,'constant',True),first,[]),('own_pack',later,0,(Expr(INT,'constant',0),)),('own_assign',0,0,later)]
        with self.assertRaisesRegex(Refusal,'dominating'):
            prove_owner_reassignment(self.origin_function(branch))

    def test_plan_retains_nonadjacent_multiple_use_and_same_slot_transfers(self):
        owner=Expr(('owner',0),'temporary',1);local=Expr(INT,'local',0);temp=Expr(INT,'temporary',2)
        cases=[
            [('own_move',owner,0),('own_assign',0,0,owner)],
            [('own_move',owner,0),('assert',Expr(BOOL,'constant',True)),('own_store',1,0,owner)],
            [('let',temp,local),('assert',Expr(BOOL,'constant',True)),('store',1,temp)],
            [('let',temp,local),('assert',Expr(BOOL,'binary','==',(temp,Expr(INT,'constant',0)))),('store',1,temp)],
            [('let',temp,local),('assert',Expr(BOOL,'binary','==',(Expr(INT,'arithmetic','add',(temp,Expr(INT,'constant',1))),Expr(INT,'constant',1))))],
        ]
        for body in cases:
            with self.subTest(body=body):
                fn=Function('main',[],INT,{0:INT,1:INT},[],body);before=copy.deepcopy(fn)
                result,receipt=plan_reassignment_nano(fn)
                self.assertEqual(result,fn);self.assertEqual(fn,before)
                self.assertEqual(receipt['patterns'],{})

    def test_constructor_plan_requires_exact_single_def_use_and_retained_rhs(self):
        a=Expr(INT,'temporary',10);b=Expr(BOOL,'temporary',11);owner=Expr(('owner',0),'temporary',12)
        bundle=[('store',0,Expr(INT,'constant',1)),('store',1,Expr(BOOL,'constant',True)),
                ('let',a,Expr(INT,'local',0)),('let',b,Expr(BOOL,'local',1)),('own_pack',owner,0,(a,b))]
        fn=Function('main',[],INT,{0:INT,1:BOOL},[],bundle)
        result,receipt=plan_reassignment_nano(fn)
        self.assertEqual(result.locals,{});self.assertEqual(receipt['patterns']['scalar_constructor_bundle'],1)
        for body in (bundle+[('store',0,Expr(INT,'constant',2))],
                     [('store',0,Expr(INT,'local',9)),*bundle[1:]],
                     bundle+[('store',2,Expr(INT,'local',0))]):
            result,receipt=plan_reassignment_nano(Function('main',[],INT,{0:INT,1:BOOL,9:INT},[],body))
            self.assertNotIn('scalar_constructor_bundle',receipt['patterns'])
            self.assertIn(0,result.locals)


class OwnedReassignmentReconstruction(unittest.TestCase):
    setUpClass=support.OwnedNestedPathReconstruction.__dict__['setUpClass']
    command=support.OwnedNestedPathReconstruction.command
    facts=support.OwnedNestedPathReconstruction.facts
    native=support.OwnedNestedPathReconstruction.native
    module=support.OwnedNestedPathReconstruction.module
    qualify=support.OwnedNestedPathReconstruction.qualify
    selected_shadow=staticmethod(support.OwnedNestedPathReconstruction.selected_shadow)

    def test_unchanged_variants_and_exact_generated_storage(self):
        for choice in ('true','false'):
            source=(BASELINE/f'owner-reassignment-{choice}.nano').read_text()
            for compiler in self.compilers:
                with self.subTest(choice=choice,compiler=str(compiler)):
                    module=self.module(compiler,source,'reassignment-'+choice)
                    facts=self.facts(module)
                    functions=[OwnedAnalyze(facts,i).run() for i in range(2)]
                    emitter=OwnedEmit(functions,'nano',facts['layouts'])
                    costs=[emitter.nano_local_cost(fn) for fn in emitter.functions]
                    shadow_cost=emitter.nano_local_cost(emitter.path_shadow_function())
                    self.assertEqual(costs,[127,9]);self.assertEqual(shadow_cost,60)
                    native=module.with_suffix('.original.c')
                    self.command([ROOT/'bin/nvm2c',module,'-o',native]);self.native(native,0)
                    self.qualify(module)
                    generated=module.with_suffix('.nano')
                    self.command([ROOT/'bin/nvm2hl','--language','nano',module,'-o',generated])
                    rebuilt=self.module(compiler,generated.read_text(),'reassignment-'+choice+'-measured')
                    actual=[fn['locals'] for fn in self.facts(rebuilt)['functions']]
                    self.assertEqual(actual,costs)
                    selected=self.module(compiler,self.selected_shadow(generated.read_text()),'reassignment-'+choice+'-shadow')
                    shadow_actual=[fn['locals'] for fn in self.facts(selected)['functions']]
                    self.assertEqual(shadow_actual,[shadow_cost,costs[1]])
                    native=selected.with_suffix('.c')
                    self.command([ROOT/'bin/nvm2c',selected,'-o',native]);self.native(native,0)
                    print('I measure reassignment '+json.dumps({'choice':choice,'provider':str(compiler),
                        'original':[fn['locals'] for fn in facts['functions']], 'actual':actual,
                        'shadow':shadow_actual,'plan':emitter.reassignment_plans}),flush=True)


    def test_same_value_fresh_loop_owner_is_conservatively_refused(self):
        text=(BASELINE/'owner-reassignment-false.nano').read_text()
        before='let carried: Leaf = owner\n  set owner carried'
        after='let carried: Leaf = owner\n  let Leaf { value, active } = carried\n  let renewed: Leaf = Leaf { value: value, active: active }\n  set owner renewed'
        self.assertIn(before,text)
        for compiler in self.compilers:
            module=self.module(compiler,text.replace(before,after),'fresh-loop-origin')
            for language in ('c','nano'):
                output=module.with_suffix('.'+language);output.write_text('prior checked source')
                result=self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output],None)
                self.assertNotEqual(result.returncode,0);self.assertIn('loop-body owner origins',result.stderr)
                self.assertEqual(output.read_text(),'prior checked source')

    def test_output_128_and_129_boundaries_use_actual_counts(self):
        text=(BASELINE/'owner-reassignment-true.nano').read_text()
        before,after=text.rsplit(' return 0',1)
        for count in (1,2):
            extra=''.join(f' let extra_{i}: int = 0\n assert (== extra_{i} 0)\n' for i in range(count))
            for compiler in self.compilers:
                module=self.module(compiler,before+extra+' return 0'+after,'reassignment-cap-'+str(count))
                facts=self.facts(module);emitter=OwnedEmit([OwnedAnalyze(facts,i).run() for i in range(2)],'nano',facts['layouts'])
                costs=[emitter.nano_local_cost(fn) for fn in emitter.functions]
                self.assertEqual(costs,[127+count,9])
                output=module.with_suffix('.nano');output.write_text('prior checked source')
                result=self.command([ROOT/'bin/nvm2hl','--language','nano',module,'-o',output],None)
                if count==1:
                    self.assertEqual(result.returncode,0,result.stderr)
                    content=output.read_text()
                else:
                    self.assertNotEqual(result.returncode,0);self.assertIn('128 local slots',result.stderr)
                    self.assertEqual(output.read_text(),'prior checked source')
                    emitter.nano_local_cost=lambda fn:0
                    content=emitter.run(0)
                rebuilt=self.module(compiler,content,'reassignment-cap-regenerated-'+str(count))
                dump=self.command([ROOT/'bin/nanoisa','dump',rebuilt]).stdout
                actual=[int(n) for n in re.findall(r'(?m)^\.function \S+ \d+ (\d+) ',dump)]
                self.assertEqual(actual,costs)
                if count==1:
                    native=rebuilt.with_suffix('.c');self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native]);self.native(native,0)
                print('I retain reassignment boundary '+json.dumps({'provider':str(compiler),'ordinary':count==1,'actual':actual}),flush=True)


    def test_original_verifier_and_defensive_mixed_transport_refusals(self):
        module=self.module(self.compilers[0],(BASELINE/'owner-reassignment-true.nano').read_text(),'reassignment-corrupt')
        facts=self.facts(module);raw=module.read_bytes()
        header,count=struct.unpack_from('<II',raw,20)
        sections={struct.unpack_from('<IIQQ',raw,header+24*n)[0]:struct.unpack_from('<IIQQ',raw,header+24*n)[2] for n in range(count)}
        code=facts['functions'][0]['code'];stores=[i for i in code if i['op']=='OWN_STORE_LOCAL']
        moves=[i for i in code if i['op']=='OWN_MOVE_LOCAL']
        path=next(i for i in code if i['op']=='BORROW_PATH_EXCLUSIVE')
        cases={'live-destination':(stores[1]['pc']+1,struct.pack('<H',stores[0]['arg'])),
               'consumed-source':(moves[2]['pc']+1,struct.pack('<H',moves[0]['arg'])),
               'nominal-path':(path['pc']+3,struct.pack('<H',stores[0]['arg']))}
        for case,(offset,payload) in cases.items():
            changed=bytearray(raw);offset+=sections[6];changed[offset:offset+len(payload)]=payload
            self.assertNotEqual(changed,raw);struct.pack_into('<I',changed,36,zlib.crc32(changed[header:]))
            bad=self.work/('reassignment-'+case+'.nvm');bad.write_bytes(changed)
            self.assertNotEqual(self.command([ROOT/'bin/nano_vm','--verify-only',bad],None).returncode,0)
            for language in ('c','nano'):
                output=bad.with_suffix('.'+language);output.write_text('prior checked source')
                self.assertNotEqual(self.command([ROOT/'bin/nvm2hl','--language',language,bad,'-o',output],None).returncode,0)
                self.assertEqual(output.read_text(),'prior checked source')
        for case in ('local-arity','path-arity','local-opcode','path-opcode','split','interior','endpoint'):
            bad=copy.deepcopy(facts);entry,helper=bad['functions']
            local=next(i for i in entry['code'] if i['op']=='BORROW_LOCAL_EXCLUSIVE')
            path=next(i for i in entry['code'] if i['op']=='BORROW_PATH_EXCLUSIVE')
            if case=='local-arity':local['args'].append(0)
            elif case=='path-arity':path['args'].pop()
            elif case=='local-opcode':local['op']='BORROW_PATH_EXCLUSIVE'
            elif case=='path-opcode':path['op']='BORROW_LOCAL_EXCLUSIVE'
            elif case=='split':next(i for i in entry['code'] if i['op']=='REGION_END')['op']='NOP'
            elif case=='interior':next(i for i in entry['code'] if i['op']=='JMP_FALSE')['target']=path['pc']
            else:helper['params'][0]=[helper['params'][0][0],1];helper['local_types'][0]=helper['params'][0][:]
            for language in ('c','nano'):
                with self.subTest(case=case,language=language),self.assertRaises(Refusal):reconstruct_owned(bad,language)

    def test_scratch_add_wraps_endpoints_and_retains_distinct_results(self):
        for initial,expected in ((9223372036854775807,-9223372036854775808),(-9223372036854775808,-9223372036854775807)):
            literal=str(initial) if initial>=0 else '(+ -9223372036854775807 -1)'
            source="""resource struct Leaf { value: int, active: bool }
fn bump(view: &mut Leaf) -> int { set view.value (+ view.value 1) return view.value }
shadow bump { let mut x: Leaf = Leaf { value: 0, active: true } assert (== (bump &mut x) 1) let Leaf { value, active } = x assert active assert (== value 1) }
fn main() -> int {
 let mut owner: Leaf = Leaf { value: 0, active: true }
 let carried: Leaf = owner
 set owner carried
 let high: int = 9223372036854775807
 let low: int = (+ -9223372036854775807 -1)
 let first: int = (+ high 1)
 let second: int = (+ low -1)
 assert (== first low) assert (== second high)
 set owner.value INITIAL
 assert (== (bump &mut owner) EXPECTED)
 let Leaf { value, active } = owner
 assert (== value EXPECTED) assert active
 return 0
}
shadow main { assert true }
"""
            # Direct owner writes are outside reconstruction; construct the
            # endpoint before the transfer instead of adding that operation.
            source=source.replace('value: 0, active: true }\n let carried','value: '+literal+', active: true }\n let carried')
            source=source.replace(' set owner.value INITIAL\n','').replace('EXPECTED',str(expected) if expected!=-9223372036854775808 else 'low')
            for compiler in self.compilers:
                module=self.module(compiler,source,'reassignment-endpoint-'+str(initial))
                self.qualify(module)


if __name__=='__main__':unittest.main()
