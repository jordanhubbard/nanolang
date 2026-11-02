"""I retain exact exclusive helper control and bounded consistency work."""
import copy
import json
import re
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

    def test_unchanged_anchors_and_generated_shadow_measurement(self):
        for fixture in ('source_borrow_control_flow.nano','source_borrow_returns.nano'):
            source=(ROOT/'tests/nanoisa/fixtures'/fixture).read_text()
            for compiler in self.compilers:
                with self.subTest(fixture=fixture,compiler=str(compiler)):
                    module=self.module(compiler,source,fixture.removesuffix('.nano'))
                    facts=self.facts(module)
                    self.assertEqual(facts['functions'][1]['param_modes'],[2])
                    functions=[OwnedAnalyze(facts,index).run() for index in range(2)]
                    emitter=OwnedEmit(functions,'nano',facts['layouts'])
                    costs=[emitter.nano_local_cost(fn) for fn in emitter.functions]
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
