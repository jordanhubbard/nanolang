"""I retain exact exclusive helper control and bounded consistency work."""
import copy
import json
import re
import hashlib
import struct
import zlib
import unittest
from tests import test_owned_nested_path_reconstruction as support
from nanoisa_owned_reconstruction import OwnedAnalyze, OwnedEmit, reconstruct_owned
from nanoisa_owned_control import prove_exclusive_helper_control, CONTROL_ADD_WORK
from nanoisa_reconstruction import Expr, Function, INT, BOOL, Refusal
ROOT = support.ROOT


class ExclusiveHelperWorkProof(unittest.TestCase):
    @staticmethod
    def helper(body):
        return Function('helper', [('borrow', 0, 2)], INT, {1: INT}, ['', ''], body)

    @staticmethod
    def rank_body():
        rank = Expr(INT, 'local', 1)
        snapshot = Expr(INT, 'temporary', 'snapshot')
        added = Expr(INT, 'temporary', 'added')
        one = Expr(INT, 'constant', 1)
        return [('let', snapshot, rank),
                ('let', added, Expr(INT, 'arithmetic', 'add', (snapshot, one))),
                ('store', 1, added)]

    def ranked(self, body, initial=0, bound=2):
        return self.helper([('store', 1, Expr(INT, 'constant', initial)),
            ('while', Expr(BOOL, 'binary', '<', (Expr(INT, 'local', 1), Expr(INT, 'constant', bound))), body),
            ('return', Expr(INT, 'local', 1))])

    def test_rank_loop_work_and_unchanged_ir(self):
        function=self.ranked(self.rank_body());before=copy.deepcopy(function)
        proof=prove_exclusive_helper_control(function)
        self.assertEqual(function,before)
        self.assertEqual(proof['loops'][0]['iterations'],2)
        self.assertGreaterEqual(proof['work'],2*CONTROL_ADD_WORK)
        self.assertLessEqual(proof['work'],4096)

    def test_guarded_add_work_covers_full_emitted_syntax(self):
        emitter=object.__new__(OwnedEmit)
        emitter.language='nano';emitter.lines=[];emitter.helpers=set()
        emitter.inline_add_statement(Expr(INT,'temporary','cost'),
            Expr(INT,'arithmetic','add',(Expr(INT,'constant',7),Expr(INT,'constant',-3))),1)
        # I count all lexical tokens, even untaken branches and delimiters;
        # this conservatively dominates the expansion's logical operations.
        tokens=re.findall(r'[A-Za-z_][A-Za-z_0-9]*|-?\d+|[^\s]', '\n'.join(emitter.lines))
        self.assertLessEqual(len(tokens),CONTROL_ADD_WORK)

    def test_adjacent_add_snapshot_folding_keeps_effect_boundaries(self):
        local=Expr(INT,'local',1);snapshot=Expr(INT,'temporary','snapshot')
        result=Expr(INT,'temporary','result');one=Expr(INT,'constant',1)
        load=('let',snapshot,local)
        add=('let',result,Expr(INT,'arithmetic','add',(snapshot,one)))
        store=('store',1,result)
        original=[load,add,store];before=copy.deepcopy(original)
        self.assertEqual(OwnedEmit.fold_add_snapshots(original),
            [('let',result,Expr(INT,'arithmetic','add',(local,one))),store])
        self.assertEqual(original,before)
        call=('borrow_call',Expr(INT,'temporary','call'),9)
        cases=[
            [load,call,add,store],
            [load,('store',1,Expr(INT,'constant',9)),add,store],
            [load,('borrow_read',Expr(INT,'temporary','read'),0),add,store],
            [load,add,('assert',Expr(BOOL,'binary','==',(snapshot,one))),store],
            [load,('if',Expr(BOOL,'constant',True),[add,store],[])],
            [('if',Expr(BOOL,'constant',True),[load],[]),add,store],
            [('let',snapshot,Expr(INT,'arithmetic','add',(local,one))),add,store],
            [load,('let',result,Expr(INT,'arithmetic','add',(snapshot,Expr(INT,'arithmetic','add',(one,one))))),store],
        ]
        for index,body in enumerate(cases):
            with self.subTest(index=index):self.assertEqual(OwnedEmit.fold_add_snapshots(body),body)

    def test_every_continuing_backedge_is_checked(self):
        condition=Expr(BOOL,'constant',True)
        body=self.rank_body();snapshot=body[0][1]
        cases={
            'missing':[],
            'double':body+body,
            'stale-snapshot':body+[('store',1,snapshot)],
            'conditional-only':[('if',condition,body,[])],
            'late-reset':body+[('store',1,Expr(INT,'constant',0))],
            'wrong-increment':[('store',1,Expr(INT,'constant',1))],
        }
        for case,invalid in cases.items():
            with self.subTest(case=case),self.assertRaises(Refusal):
                prove_exclusive_helper_control(self.ranked(invalid))
        for initial,bound in ((2,1),(0,1<<63),(0,10000)):
            with self.subTest(initial=initial,bound=bound),self.assertRaises(Refusal):
                prove_exclusive_helper_control(self.ranked(body,initial,bound))

    def test_only_continuing_arm_update_counts_join(self):
        early=[('return',Expr(INT,'constant',7))]
        body=[('if',Expr(BOOL,'constant',True),early,self.rank_body())]
        proof=prove_exclusive_helper_control(self.ranked(body))
        self.assertEqual(proof['loops'][0]['iterations'],2)
        late=self.rank_body()+early
        body=[('if',Expr(BOOL,'constant',True),late,self.rank_body())]
        self.assertEqual(prove_exclusive_helper_control(self.ranked(body))['loops'][0]['iterations'],2)

    def test_return_only_loop_counts_pretest_and_preserves_false_path_facts(self):
        field=Expr(BOOL,'temporary','enabled')
        body=[('if',field,[('store',1,Expr(INT,'constant',99)),('return',Expr(INT,'constant',3))],
                         [('return',Expr(INT,'constant',4))])]
        function=self.helper([('store',1,Expr(INT,'constant',0)),
            ('while_observe',field,[('borrow_read',field,1)],body),
            self.ranked(self.rank_body()).body[1],('return',Expr(INT,'local',1))])
        proof=prove_exclusive_helper_control(function)
        self.assertEqual([loop['form'] for loop in proof['loops']],['return-only','rank'])
        self.assertGreater(proof['loops'][0]['work'],1)
        invalid=copy.deepcopy(function)
        invalid.body[1][2].append(('borrow_write',1,Expr(BOOL,'constant',False)))
        with self.assertRaises(Refusal):prove_exclusive_helper_control(invalid)


class OwnedExclusiveControlReconstruction(unittest.TestCase):
    setUpClass=support.OwnedNestedPathReconstruction.__dict__['setUpClass']
    command=support.OwnedNestedPathReconstruction.command
    facts=support.OwnedNestedPathReconstruction.facts
    native=support.OwnedNestedPathReconstruction.native
    module=support.OwnedNestedPathReconstruction.module
    qualify=support.OwnedNestedPathReconstruction.qualify
    selected_shadow=staticmethod(support.OwnedNestedPathReconstruction.selected_shadow)

    def minimal_control_source(self, helper, expected):
        body=f""" let left: Leaf = Leaf {{ value: 0, enabled: true }}
 let right: Leaf = Leaf {{ value: 100, enabled: false }}
 let mut root: Pair = Pair {{ left: left, right: right }}
 assert (== (step &mut root.left) {expected})
 let Pair {{ left, right }} = root
 let Leaf {{ value, enabled }} = left
 assert (== value {expected}) assert enabled
 let Leaf {{ value, enabled }} = right
 assert (== value 100) assert (not enabled)
"""
        return helper+'shadow step {\n'+body+'}\nfn main() -> int {\n'+body+' return 0\n}\nshadow main { assert true }\n'

    def control_helper(self):
        return (ROOT/'tests/nanoisa/fixtures/source_borrow_control_flow.nano').read_text().split('shadow step')[0]

    def test_rank_zero_one_three_and_returning_arm_sources(self):
        original=self.control_helper()
        cases={str(count):(original.replace('(< index 2)',f'(< index {count})'),count) for count in (0,1,3)}
        cases['returning-arm']=(original.replace('set index (+ index 1)',
            'if view.enabled { set index (+ index 1) } else { return -1 }'),2)
        for case,(helper,expected) in cases.items():
            for compiler in self.compilers:
                with self.subTest(case=case,compiler=str(compiler)):
                    module=self.module(compiler,self.minimal_control_source(helper,expected),'rank-'+case)
                    native=module.with_suffix('.original.c')
                    self.command([ROOT/'bin/nvm2c',module,'-o',native]);self.native(native,0)
                    self.qualify(module)

    def test_return_caller_variants_preserve_complete_owner_consumption(self):
        text=(ROOT/'tests/nanoisa/fixtures/source_borrow_returns.nano').read_text()
        ending='if true { let code: int = 0 return code } else { return 1 }'
        variants={'else':'if false { return 1 } else { let code: int = 0 return code }',
            'loop':'while true { let code: int = 0 return code } return 1',
            'zero':'while false { return 1 } return 0',
            'single':'if false { return 1 } return 0',
            'then-only':'if true { return 0 } return 1'}
        for case,replacement in variants.items():
            for compiler in self.compilers:
                with self.subTest(case=case,compiler=str(compiler)):
                    module=self.module(compiler,text.replace(ending,replacement),'returns-'+case)
                    native=module.with_suffix('.original.c')
                    self.command([ROOT/'bin/nvm2c',module,'-o',native]);self.native(native,0)
                    self.qualify(module)

    def test_actual_unproved_rank_and_work_limit_refuse_atomically(self):
        original=self.control_helper()
        variants={
            'unknown-init':(original.replace('let mut index: int = 0','let mut index: int = view.value'),2),
            'stride-two':(original.replace('set index (+ index 1)','set index (+ index 2)'),1),
            'conditional-only':(original.replace('set index (+ index 1)','if view.enabled { set index (+ index 1) }'),2),
            'wrong-guard':(original.replace('(< index 2)','(<= index 2)'),3),
            'work-limit':(original.replace('(< index 2)','(< index 8)'),8),
        }
        for case,(helper,expected) in variants.items():
            for compiler in self.compilers:
                with self.subTest(case=case,compiler=str(compiler)):
                    module=self.module(compiler,self.minimal_control_source(helper,expected),'refuse-'+case)
                    for language in ('c','nano'):
                        output=module.with_suffix('.'+language);output.write_text('prior checked source')
                        result=self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output],None)
                        self.assertNotEqual(result.returncode,0)
                        self.assertRegex(result.stderr,'rank|4096 logical')
                        self.assertEqual(output.read_text(),'prior checked source')

    def test_generated_local_cap_remains_atomic_after_folding(self):
        source=(ROOT/'tests/nanoisa/fixtures/source_borrow_control_flow.nano').read_text().replace(
            ' let mut i: int = 0',' let extra: int = 0\n assert (== extra 0)\n let mut i: int = 0')
        for compiler in self.compilers:
            module=self.module(compiler,source,'exclusive-control-cap')
            facts=self.facts(module)
            emitter=OwnedEmit([OwnedAnalyze(facts,i).run() for i in range(2)],'nano',facts['layouts'])
            costs=[emitter.nano_local_cost(fn) for fn in emitter.functions]
            self.assertGreater(costs[0],128)
            output=module.with_suffix('.nano');output.write_text('prior checked source')
            result=self.command([ROOT/'bin/nvm2hl','--language','nano',module,'-o',output],None)
            self.assertNotEqual(result.returncode,0);self.assertIn('128 local slots',result.stderr)
            self.assertEqual(output.read_text(),'prior checked source')
            emitter.nano_local_cost=lambda fn:0
            rebuilt=self.module(compiler,emitter.run(0),'exclusive-control-cap-diagnostic')
            dump=self.command([ROOT/'bin/nanoisa','dump',rebuilt]).stdout
            actual=[int(n) for n in re.findall(r'(?m)^\.function \S+ \d+ (\d+) ',dump)]
            self.assertEqual(actual,costs)
            print('I retain exclusive cap '+json.dumps({'provider':str(compiler),'actual':actual}),flush=True)

    def test_original_reference_verifier_and_defensive_control_refusals(self):
        text=(ROOT/'tests/nanoisa/fixtures/source_borrow_control_flow.nano').read_text()
        module=self.module(self.compilers[0],text,'exclusive-control-corrupt')
        facts=self.facts(module);raw=module.read_bytes()
        header,count=struct.unpack_from('<II',raw,20)
        sections={struct.unpack_from('<IIQQ',raw,header+24*n)[0]:struct.unpack_from('<IIQQ',raw,header+24*n)[2] for n in range(count)}
        base=sections[6]+facts['functions'][0]['size']
        read=next(ins for ins in facts['functions'][1]['code'] if ins['op']=='REF_GET')
        write=next(ins for ins in facts['functions'][1]['code'] if ins['op']=='REF_SET')
        for case,offset,payload in (('slot',base+read['pc']+1,struct.pack('<H',255)),
                                   ('field-type',base+write['pc']+3,struct.pack('<H',1))):
            changed=bytearray(raw);changed[offset:offset+len(payload)]=payload
            self.assertNotEqual(changed,raw);struct.pack_into('<I',changed,36,zlib.crc32(changed[header:]))
            bad=self.work/('wire-'+case+'.nvm');bad.write_bytes(changed)
            self.assertNotEqual(self.command([ROOT/'bin/nano_vm','--verify-only',bad],None).returncode,0)
            for language in ('c','nano'):
                output=bad.with_suffix('.'+language);output.write_text('prior checked source')
                self.assertNotEqual(self.command([ROOT/'bin/nvm2hl','--language',language,bad,'-o',output],None).returncode,0)
                self.assertEqual(output.read_text(),'prior checked source')
        for case in ('mode','write-slot','split-region','interior'):
            bad=copy.deepcopy(facts)
            if case=='mode':bad['functions'][1]['param_modes']=[1]
            elif case=='write-slot':
                ins=next(ins for ins in bad['functions'][1]['code'] if ins['op']=='REF_SET');ins.update(arg=1,args=[1,0])
            elif case=='split-region':
                ins=next(ins for ins in bad['functions'][0]['code'] if ins['op']=='REGION_END');ins['op']='NOP'
            else:
                ins=next(ins for ins in bad['functions'][1]['code'] if ins['op']=='JMP_FALSE');ins['target']+=1
            for language in ('c','nano'):
                with self.subTest(case=case,language=language),self.assertRaises(Refusal):reconstruct_owned(bad,language)

    def test_unchanged_anchors_and_generated_shadow_measurement(self):
        for fixture in ('source_borrow_control_flow.nano','source_borrow_returns.nano'):
            source=(ROOT/'tests/nanoisa/fixtures'/fixture).read_text()
            for compiler in self.compilers:
                with self.subTest(fixture=fixture,compiler=str(compiler)):
                    module=self.module(compiler,source,fixture.removesuffix('.nano'))
                    facts=self.facts(module)
                    self.assertEqual(facts['functions'][1]['param_modes'],[2])
                    functions=[OwnedAnalyze(facts,index).run() for index in range(2)]
                    original_ir=copy.deepcopy(functions)
                    emitter=OwnedEmit(functions,'nano',facts['layouts'])
                    self.assertEqual(functions,original_ir)
                    costs=[emitter.nano_local_cost(fn) for fn in emitter.functions]
                    if fixture=='source_borrow_control_flow.nano':
                        def assertions(body):return sum(stmt[0]=='assert' for stmt in OwnedEmit.walk_statements(body))
                        self.assertEqual(assertions(functions[0].body),assertions(emitter.functions[0].body))
                        def retained_before_calls(body):
                            result=0
                            for index,stmt in enumerate(body):
                                if index and stmt[0]=='path_borrow_call' and body[index-1][0]=='let' and body[index-1][2].kind=='local':result+=1
                                if stmt[0]=='if':result+=retained_before_calls(stmt[2])+retained_before_calls(stmt[3])
                                elif stmt[0]=='while':result+=retained_before_calls(stmt[2])
                                elif stmt[0]=='while_observe':result+=retained_before_calls(stmt[2])+retained_before_calls(stmt[3])
                            return result
                        self.assertEqual(retained_before_calls(emitter.functions[0].body),2)
                    shadow_cost=emitter.nano_local_cost(emitter.path_shadow_function())
                    print('I measure exclusive control '+json.dumps({'fixture':fixture,'provider':str(compiler),
                        'original':[fn['locals'] for fn in facts['functions']], 'generated':costs,
                        'shadow':shadow_cost,'proof':emitter.control_proof}),flush=True)
                    native=module.with_suffix('.original.c')
                    self.command([ROOT/'bin/nvm2c',module,'-o',native]);self.native(native,0)
                    if max(*costs,shadow_cost)>128:
                        output=module.with_suffix('.nano');output.write_text('prior checked source')
                        result=self.command([ROOT/'bin/nvm2hl','--language','nano',module,'-o',output],None)
                        self.assertNotEqual(result.returncode,0);self.assertIn('128 local slots',result.stderr)
                        self.assertEqual(output.read_text(),'prior checked source')
                        # Diagnostic producer admission cannot turn this into ordinary support.
                        emitter.nano_local_cost=lambda fn:0
                        text=emitter.run(0)
                        for label,content,expected in (('entry',text,costs),
                            ('shadow',self.selected_shadow(text),[shadow_cost,costs[1]])):
                            rebuilt=self.module(compiler,content,fixture+'-diagnostic-'+label)
                            dump=self.command([ROOT/'bin/nanoisa','dump',rebuilt]).stdout
                            self.assertEqual([int(n) for n in re.findall(r'(?m)^\.function \S+ \d+ (\d+) ',dump)],expected)
                        self.fail('I retain the unchanged anchor as an ordinary 128-local output refusal')
                    self.qualify(module)
                    selected=self.module(compiler,self.selected_shadow(reconstruct_owned(facts,'nano')),fixture+'-shadow')
                    rebuilt=self.facts(selected)
                    self.assertEqual([fn['locals'] for fn in rebuilt['functions']],[shadow_cost,costs[1]])
                    native=selected.with_suffix('.c')
                    self.command([ROOT/'bin/nvm2c',selected,'-o',native]);self.native(native,0)


if __name__=='__main__':unittest.main()
