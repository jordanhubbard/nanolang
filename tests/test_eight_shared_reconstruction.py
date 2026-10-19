"""I preserve eight shared aliases and typed first-call evidence."""
import copy
import os
import unittest
from tests import test_owned_nested_path_reconstruction as support
from nanoisa_owned_reconstruction import OwnedAnalyze, OwnedEmit, reconstruct_owned
from nanoisa_owned_eight_shared import prove_eight_shared
from nanoisa_reconstruction import Refusal, Expr, INT, BOOL

ROOT = support.ROOT


def original_source():
    return (ROOT/'tests/nanoisa/fixtures/source_borrow_eight.nano').read_text()


def literal(value):
    return str(value)


def endpoint_source(values, weights):
    text = original_source()
    for i, (value, weight) in enumerate(zip(values, weights)):
        text = text.replace(f'value: {i+1}, active:', f'value: {literal(value)}, active:')
        text = text.replace(f'(* p{i}.value {i+1})', f'(* p{i}.value {literal(weight)})')
        text = text.replace(f'assert (== value {i+1})', f'assert (== value {literal(value)})')
    wrap = lambda x: (x + (1 << 63)) % (1 << 64) - (1 << 63)
    text = text.replace(') 204)', f') {literal(wrap(sum(v*w for v,w in zip(values,weights))))})')
    return text.replace(') 36)', f') {literal(wrap(values[0]*sum(weights)))})')


class EightSharedReconstruction(unittest.TestCase):
    setUpClass = support.OwnedNestedPathReconstruction.__dict__['setUpClass']
    command = support.OwnedNestedPathReconstruction.command
    facts = support.OwnedNestedPathReconstruction.facts
    native = support.OwnedNestedPathReconstruction.native
    module = support.OwnedNestedPathReconstruction.module
    qualify = support.OwnedNestedPathReconstruction.qualify
    selected_shadow = staticmethod(support.OwnedNestedPathReconstruction.selected_shadow)

    def test_unchanged_all_producers_and_native(self):
        for compiler in self.compilers:
            module = self.module(compiler, original_source(), 'eight-anchor')
            facts = self.facts(module)
            self.assertEqual([f['locals'] for f in facts['functions']], [48,9])
            self.assertEqual(facts['functions'][1]['param_modes'], [1]*8)
            native = module.with_suffix('.original.c')
            self.command([ROOT/'bin/nvm2c',module,'-o',native]); self.native(native,0)
            self.qualify(module)

    def test_original_and_generated_shadow(self):
        assembly = self.work/'eight-original-shadow.nasm'
        driver = os.environ.get('NANOLANG_EIGHT_SHARED_SHADOW_EMITTER')
        if driver:
            source = self.work/'eight-original.nano'; source.write_text(original_source())
            assembly.write_text(self.command([driver,source,'0','raw']).stdout)
        else:
            assembly.write_bytes((ROOT/'docs/evidence/eight-shared-baseline-20261019/original-shadows.nasm').read_bytes())
        module = assembly.with_suffix('.nvm')
        self.command([ROOT/'bin/nanoisa','asm',assembly,'-o',module])
        self.command([ROOT/'bin/nano_vm','--verify-only',module]); self.command([ROOT/'bin/nano_vm',module])
        self.qualify(module)
        text = reconstruct_owned(self.facts(module),'nano')
        for compiler in self.compilers:
            rebuilt = self.module(compiler,self.selected_shadow(text),'eight-selected')
            self.assertEqual([f['locals'] for f in self.facts(rebuilt)['functions']], [98,103])
            native = rebuilt.with_suffix('.c'); self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native]); self.native(native,0)

    def test_immutable_proof_and_exact_rendering(self):
        facts = self.facts(self.module(self.compilers[0],original_source()))
        functions = [OwnedAnalyze(facts,i).run() for i in range(2)]
        frozen = copy.deepcopy(functions)
        emit = OwnedEmit(functions,'nano',facts['layouts'])
        self.assertEqual(functions,frozen)
        self.assertEqual(emit.eight_shared_plan['covered_statements'],151)
        self.assertEqual(emit.eight_shared_plan['original_statements'],151)
        self.assertEqual(len(functions[1].body),28)
        self.assertEqual([sum(s[0]=='assert' for s in f.body) for f in functions],[18,1])
        self.assertEqual([sum(s[0]=='assert' for s in f.body) for f in emit.functions],[18,1])
        self.assertEqual([emit.nano_local_cost(f) for f in emit.functions],[92,103])
        self.assertEqual(emit.scratch_functions,{id(emit.functions[1])})
        self.assertEqual(emit.scratch_sub_functions,emit.scratch_functions)
        self.assertEqual(emit.eight_shared_proof['work'],437)
        self.assertEqual(emit.eight_shared_proof['calls'],2)

    def test_signed_multiplication_endpoints(self):
        low,high=-(1<<63),(1<<63)-1
        cases=[([low,high,-1,0,1,-7,12,low],[-1,-2,3,0,-7,8,-9,-1]),
               ([high,low,0,1,-1,high,low,2],[low,high,-1,-1,-2,0,1,-3])]
        for index,(values,weights) in enumerate(cases):
            for compiler in self.compilers:
                self.qualify(self.module(compiler,endpoint_source(values,weights),'eight-endpoint-'+str(index)))

    def test_typed_identity_and_first_snapshot_refusals(self):
        facts = self.facts(self.module(self.compilers[0],original_source()))
        corruptions=[]
        for mode in (2,True,1.0):
            bad=copy.deepcopy(facts);bad['functions'][1]['param_modes'][7]=mode;corruptions.append(bad)
        for value in (True,1.0):
            bad=copy.deepcopy(facts);bad['functions'][1]['result_type'][0]=value;corruptions.append(bad)
        bad=copy.deepcopy(facts)
        reads=[i for i in bad['functions'][1]['code'] if i['op']=='REF_GET']
        reads[-1]['arg']=8;reads[-1]['args'][0]=8;corruptions.append(bad)
        bad=copy.deepcopy(facts);bad['functions'][1]['params'][7][1]=1;corruptions.append(bad)
        bad=copy.deepcopy(facts)
        index=next(i for i,ins in enumerate(bad['functions'][0]['code']) if ins['op']=='OWN_UNPACK_LOCAL')
        del bad['functions'][0]['code'][index];corruptions.append(bad)
        bad=copy.deepcopy(facts)
        move=next(ins for ins in bad['functions'][0]['code'] if ins['op']=='OWN_MOVE_LOCAL')
        move['arg']=127;move['args'][0]=127;corruptions.append(bad)
        for bad in corruptions:
            for language in ('c','nano'):
                with self.assertRaises(Refusal):reconstruct_owned(bad,language)
        functions=[OwnedAnalyze(facts,i).run() for i in range(2)]
        bad=copy.deepcopy(functions)
        # Actual p7 BOOL provenance is required; a known false field must refuse.
        bool_stores=[i for i,s in enumerate(bad[0].body) if s[0]=='store' and s[2].tag==BOOL and s[2].kind=='constant']
        self.assertGreaterEqual(len(bool_stores),8)
        i=bool_stores[7];s=bad[0].body[i];bad[0].body[i]=(s[0],s[1],Expr(BOOL,'constant',False))
        with self.assertRaises(Refusal):prove_eight_shared(bad,facts['layouts'])
        boundary=copy.deepcopy(functions)
        prefix=[('let',Expr(INT,'temporary',f'budget{n}'),Expr(INT,'constant',0)) for n in range(1829)]
        prefix[0]=('let',Expr(BOOL,'temporary','budget0'),Expr(BOOL,'not',None,(Expr(BOOL,'constant',False),)))
        boundary[0].body[0:0]=prefix
        self.assertEqual(prove_eight_shared(boundary,facts['layouts'])[1]['work'],4096)
        bad=copy.deepcopy(boundary)
        target,value=bad[0].body[0][1:]
        bad[0].body[0]=('let',target,Expr(BOOL,'not',None,(value,)))
        with self.assertRaisesRegex(Refusal,'4096'):prove_eight_shared(bad,facts['layouts'])
        self.assertEqual(prove_eight_shared(functions,facts['layouts'])[1]['work'],437)

    def test_actual_128_129_publication(self):
        for count in (36,37):
            text=original_source().replace(' return 0\n}', ''.join(f' let boundary_{i}: int = {i}\n' for i in range(count))+' return 0\n}')
            for compiler in self.compilers:
                module=self.module(compiler,text,'eight-boundary-'+str(count))
                output=module.with_suffix('.nano');output.write_text('prior checked source')
                result=self.command([ROOT/'bin/nvm2hl','--language','nano',module,'-o',output],None)
                if count==37:
                    self.assertNotEqual(result.returncode,0);self.assertIn('128 local slots',result.stderr)
                    self.assertEqual(output.read_text(),'prior checked source')
                else:
                    self.assertEqual(result.returncode,0,result.stderr)
                    rebuilt=self.module(compiler,output.read_text(),'eight-128')
                    self.assertEqual(self.facts(rebuilt)['functions'][0]['locals'],128)

    def test_later_false_helper_assertion_stays_runtime_failure(self):
        text=original_source()
        prefix,main=text.split('fn main()',1)
        main=main.replace('owner0: Counter = Counter { value: 1, active: true }','owner0: Counter = Counter { value: 1, active: false }')
        text=prefix+'fn main()'+main
        for compiler in self.compilers:
            source=self.work/('eight-false-'+compiler.name+'.nano');source.write_text(text)
            module=source.with_suffix('.nvm');self.command([compiler,source,'--emit-nvm','-o',module])
            self.command([ROOT/'bin/nano_vm','--verify-only',module])
            self.assertNotEqual(self.command([ROOT/'bin/nano_vm',module],None).returncode,0)
            for language in ('c','nano'):
                output=module.with_suffix('.out.'+language)
                self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output])
                if language=='nano':
                    rebuilt=output.with_suffix('.nvm');self.command([compiler,output,'--emit-nvm','-o',rebuilt])
                    self.command([ROOT/'bin/nano_vm','--verify-only',rebuilt])
                    self.assertNotEqual(self.command([ROOT/'bin/nano_vm',rebuilt],None).returncode,0)
                else:
                    for optimization in ('-O0','-O2'):
                        exe=output.with_name(output.name+optimization)
                        self.command(['cc','-std=c11','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',optimization,output,'-o',exe])
                        self.assertNotEqual(self.command([exe],None).returncode,0)
