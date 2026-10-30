"""I materialize only exact source-qualified wrapping arithmetic in owned tests."""
import copy
import re
import sys
import unittest
from tests import test_owned_observation_reconstruction as observation
ROOT=observation.ROOT
sys.path.insert(0,str(ROOT/'scripts'))
from nanoisa_owned_reconstruction import reconstruct_owned
from nanoisa_reconstruction import Refusal

class OwnedWrappingReconstruction(unittest.TestCase):
    setUpClass=observation.OwnedObservationReconstruction.__dict__['setUpClass']
    command=observation.OwnedObservationReconstruction.command
    facts=observation.OwnedObservationReconstruction.facts
    native=observation.OwnedObservationReconstruction.native
    roundtrip=observation.OwnedObservationReconstruction.roundtrip

    def module(self,operator='*',left=-3,right=7,expected=-21,compiler=None):
        compiler=compiler or self.compilers[0]
        literal=lambda n:'(- -9223372036854775807 1)' if n==-(1<<63) else str(n)
        name={'*':'mul','+':'add','-':'sub'}[operator]+str(left)+'_'+str(right)+'_'+compiler.name
        source=self.work/(name+'.nano')
        source.write_text('''resource struct Box { value: int, flag: bool }
fn pick(owner: Box, left: int, right: int, expected: int) -> int {
 let mut i: int = 0
 while (and owner.flag (and (< i 3) (== (OP left right) expected))) { set i (+ i 1) }
 let Box { value, flag } = owner
 return (+ value i)
}
shadow pick { assert (== (main) 23) }
fn main() -> int {
 let enabled: int = (pick Box { value: 10, flag: true } LEFT RIGHT EXPECTED)
 let disabled: int = (pick Box { value: 10, flag: false } LEFT RIGHT EXPECTED)
 return (+ enabled disabled)
}
shadow main { assert (== (main) 23) }
'''.replace('OP',operator).replace('LEFT',literal(left)).replace('RIGHT',literal(right)).replace('EXPECTED',literal(expected)))
        source.with_suffix('.source.nano.txt').write_text(source.read_text())
        module=source.with_suffix('.nvm')
        self.command([compiler,source,'--emit-nvm','-o',module])
        return module

    def test_exact_wrapping_endpoints_through_all_routes(self):
        low=-(1<<63);high=(1<<63)-1
        cases=[('+',high,1,low),('+',low,-1,high),('-',low,1,high),('-',high,-1,low),
               ('*',low,-1,low),('*',high,high,1),('*',-3,7,-21),('*',-3,-7,21),('*',low,2,0)]
        for compiler in self.compilers:
            for op,left,right,expected in cases:
                with self.subTest(op=op,left=left,right=right,producer=str(compiler)):
                    module=self.module(op,left,right,expected,compiler)
                    facts=self.facts(module);code=facts['functions'][1]['code']
                    branch=next(n for n,i in enumerate(code) if i['op']=='JMP_FALSE')
                    self.assertEqual(sum(i['op']=={'+':'ADD','-':'SUB','*':'MUL'}[op] for i in code[:branch]),1)
                    self.roundtrip(module,23)

    def test_eager_false_operand_and_final_false_operation_count(self):
        module=self.module();self.roundtrip(module,23)
        source=module.with_suffix('.c').read_text()
        pattern=r'(?m)^(\s*int64_t nlr_t\d+ = )(nlr_i64_mul\([^;]+\));$'
        source,count=re.subn(pattern,r'\1(nlr_operation_count++, \2);',source)
        self.assertEqual(count,1)
        instrumented=module.with_suffix('.counted.c')
        instrumented.write_text('#define main nlr_counted_main\nstatic int nlr_operation_count;\n'+source+
            '\n#undef main\nint main(void) { int result=nlr_counted_main(); return nlr_operation_count==5 ? result : 99; }\n')
        self.native(instrumented,23)

    def test_exact_condition_types_and_other_operations_refuse(self):
        original=self.facts(self.module())
        for case in ('bool','div','rem','neg','typed-add','assert','call','store'):
            facts=copy.deepcopy(original);fn=facts['functions'][1];code=fn['code']
            index=next(n for n,i in enumerate(code) if i['op']=='MUL')
            if case=='bool':code[index-1].update(op='PUSH_BOOL',arg=1,args=[1])
            else:code[index]['op']={'div':'DIV','rem':'MOD','neg':'NEG','typed-add':'I64_ADD','assert':'ASSERT','call':'CALL','store':'STORE_LOCAL'}[case]
            for language in ('c','nano'):
                with self.subTest(case=case,language=language),self.assertRaises(Refusal):reconstruct_owned(facts,language)

    def test_generated_helper_budget_and_local_budget_refuse_atomically(self):
        source=self.work/'helper-limit.nano'
        original=(ROOT/'tests/nanoisa/fixtures/owned_loops/arithmetic-condition.nano').read_text()
        original=original.replace('(+ count 0)','(* (- (+ count 0) 0) 1)')
        original=original.replace('(pick Box','(third Box')
        original+='''
fn first(owner: Box, count: int) -> int { return (pick owner count) }
shadow first { assert true }
fn second(owner: Box, count: int) -> int { return (first owner count) }
shadow second { assert true }
fn third(owner: Box, count: int) -> int { return (second owner count) }
shadow third { assert true }
'''
        source.write_text(original)
        for compiler in self.compilers:
            module=source.with_suffix('.'+compiler.name+'.nvm')
            self.command([compiler,source,'--emit-nvm','-o',module])
            self.command([ROOT/'bin/nano_vm','--verify-only',module])
            self.command([ROOT/'bin/nano_vm',module],17)
            c=module.with_suffix('.c');self.command([ROOT/'bin/nvm2hl','--language','c',module,'-o',c]);self.native(c,17)
            nano=module.with_suffix('.nano');nano.write_text('prior checked source')
            result=self.command([ROOT/'bin/nvm2hl','--language','nano',module,'-o',nano],None)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('reserve one of eight producer functions',result.stderr)
            self.assertEqual(nano.read_text(),'prior checked source')
            # I retain the real earlier regenerated129-local source, not an invented declaration.
            local_source=self.work/('local-limit-'+compiler.name+'.nano')
            local_source.write_text((ROOT/'docs/evidence/owned-mixed-depth2-20261019/order-129-locals-reconstructed.nano.txt').read_text())
            local_module=local_source.with_suffix('.nvm')
            self.command([compiler,local_source,'--emit-nvm','-o',local_module])
            self.command([ROOT/'bin/nano_vm','--verify-only',local_module])
            self.command([ROOT/'bin/nano_vm',local_module],45)
            for language in ('c','nano'):
                output=self.work/('prior-local.'+language);output.write_text('prior checked source')
                result=self.command([ROOT/'bin/nvm2hl','--language',language,local_module,'-o',output],None)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('verified bounded resource module',result.stderr)
                self.assertEqual(output.read_text(),'prior checked source')
