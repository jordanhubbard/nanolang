"""I preserve complete reference-call lifetimes on every loop condition."""
import copy
import unittest
from tests import test_owned_nested_path_reconstruction as nested
from nanoisa_owned_reconstruction import OwnedAnalyze, reconstruct_owned
from nanoisa_reconstruction import Refusal
ROOT = nested.ROOT


class OwnedCallPretestReconstruction(unittest.TestCase):
    setUpClass = nested.OwnedNestedPathReconstruction.__dict__['setUpClass']
    command = nested.OwnedNestedPathReconstruction.command
    facts = nested.OwnedNestedPathReconstruction.facts
    native = nested.OwnedNestedPathReconstruction.native
    module = nested.OwnedNestedPathReconstruction.module
    qualify = nested.OwnedNestedPathReconstruction.qualify

    @staticmethod
    def source(iterations=0, double=False, first_only=False):
        text=(ROOT/'tests/nanoisa/fixtures/source_borrow_nested.nano').read_text()
        head,main=text.split('fn main()',1)
        first=' assert (== (change &mut root.branch.left &mut root.branch.right &root.other) -14)'
        second='(change &mut root.branch.right &mut root.branch.left &root.other)'
        for value,name in ((13,'left'),(27,'right'),(10,'left'),(30,'right')):
            main=main.replace(f' assert (== root.branch.{name}.value {value})\n','')
        old=' assert (== '+second+' 20)'
        # Both occurrences remain eagerly evaluated in the two-call source.
        condition=f'(< {second} {20+6*iterations})'
        if double: condition=f'(== (+ {second} {second}) 0)'
        replacement=f' while {condition} {{ assert true }}'
        main=main.replace(old,replacement)
        main=main.replace('assert (== value 10)',f'assert (== value {10-3*iterations})')
        main=main.replace('assert (== value 30)',f'assert (== value {30+3*iterations})')
        if double or first_only:
            main=main.replace('assert (== value 10)','assert (== value 7)')
            main=main.replace('assert (== value 30)','assert (== value 33)')
        if first_only:
            main=main.replace(first,'')
        return head+'fn main()'+main

    def test_zero_one_multiple_iterations_retain_last_false_update(self):
        for iterations in (0,1,3):
            for compiler in self.compilers:
                with self.subTest(iterations=iterations,compiler=str(compiler)):
                    module=self.module(compiler,self.source(iterations),f'call-pretest-{iterations}')
                    native=module.with_suffix('.original.c')
                    self.command([ROOT/'bin/nvm2c',module,'-o',native]);self.native(native,0)
                    facts=self.qualify(module)
                    loops=[stmt for stmt in OwnedAnalyze(facts,0).run().body if stmt[0]=='while_observe']
                    self.assertEqual(len(loops),1)
                    self.assertEqual(sum(stmt[0]=='path_borrow_call' for stmt in loops[0][2]),1)

    def test_pretest_helper_assertion_remains_runtime_failure(self):
        from tests import test_owned_void_reconstruction as void
        text=self.source().replace('resource struct Other { value: int, enabled: bool }\n','').replace('Other','Leaf')
        head,main=text.split('fn main()',1)
        main=main.replace('(change &mut root.branch.right &mut root.branch.left &root.other)',
                          '(change &mut root.branch.left &mut root.other &root.branch.right)')
        for compiler in self.compilers:
            source=self.work/'call-pretest-false.nano';source.write_text(head+'fn main()'+main)
            module=self.work/('call-pretest-false-'+compiler.name+'.nvm')
            self.command([compiler,source,'--emit-nvm','-o',module])
            self.command([ROOT/'bin/nano_vm','--verify-only',module])
            void.OwnedVoidReconstruction.vm_failure(self,module)
            for language in ('c','nano'):
                output=module.with_suffix('.'+language)
                self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output])
                if language=='c': void.OwnedVoidReconstruction.native_failure(self,output,'Assertion failed')
                else:
                    for producer in self.compilers:
                        rebuilt=output.with_suffix('.'+producer.name+'.nvm')
                        self.command([producer,output,'--emit-nvm','-o',rebuilt])
                        void.OwnedVoidReconstruction.vm_failure(self,rebuilt)
                        native=rebuilt.with_suffix('.c')
                        self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native])
                        void.OwnedVoidReconstruction.native_failure(self,native,None)

    def test_original_verified_inverted_exit(self):
        module=self.module(self.compilers[0],self.source(1),'call-pretest-invert')
        assembly=self.command([ROOT/'bin/nanoisa','dump',module]).stdout
        self.assertIn('  JMP_FALSE ',assembly)
        assembly=assembly.replace('  JMP_FALSE ','  NOT\n  JMP_TRUE ')
        source=self.work/'call-pretest-inverted.nasm';source.write_text(assembly)
        inverted=source.with_suffix('.nvm')
        self.command([ROOT/'bin/nanoisa','asm',source,'-o',inverted])
        self.command([ROOT/'bin/nano_vm','--verify-only',inverted])
        self.command([ROOT/'bin/nano_vm',inverted])
        self.qualify(inverted)

    def test_assertion_before_loop_first_witness_refuses_atomically(self):
        for name,options,message in (
            ('prefix-assert',{'first_only':True},'literal'),):
            for compiler in self.compilers:
                module=self.module(compiler,self.source(**options).replace('\n while ','\n assert true\n while ',1),name)
                for language in ('c','nano'):
                    target=module.with_suffix('.'+language);target.write_text('prior checked source')
                    result=self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',target],None)
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn(message,result.stderr)
                    self.assertEqual(target.read_text(),'prior checked source')

    def test_defensive_split_region_and_reference_slot_refuse(self):
        module=self.module(self.compilers[0],self.source(),'call-pretest-defense')
        facts=self.facts(module)
        code=facts['functions'][0]['code']
        # First call supplies the shadow witness; the second belongs to the loop.
        begins=[i for i,ins in enumerate(code) if ins['op']=='REGION_BEGIN' and
                i+4<len(code) and code[i+4]['op']=='CALL_REF']
        self.assertEqual(len(begins),2)
        start=begins[1]
        malformed=copy.deepcopy(facts)
        malformed['functions'][0]['code'][start+5]['op']='NOP'
        for language in ('c','nano'):
            with self.assertRaises(Refusal): reconstruct_owned(malformed,language)
        # I alter only defensive facts here; I do not claim original admission.
        malformed=copy.deepcopy(facts)
        backedge=next(ins for ins in reversed(malformed['functions'][0]['code'])
                      if ins['op']=='JMP' and ins['target']<=ins['pc'])
        backedge['target']=code[start+1]['pc']
        for language in ('c','nano'):
            with self.assertRaisesRegex(Refusal,'interiors'):
                reconstruct_owned(malformed,language)
        malformed=copy.deepcopy(facts)
        malformed['functions'][0]['code'][start+1]['args'][0]+=1
        for language in ('c','nano'):
            with self.assertRaises(Refusal): reconstruct_owned(malformed,language)

    def test_output_local_limit_is_measured_and_atomic(self):
        text=(ROOT/'tests/nanoisa/fixtures/source_borrow_nested.nano').read_text()
        head,main=text.split('fn main()',1)
        call='(change &mut root.branch.right &mut root.branch.left &root.other)'
        main=main.replace(' assert (== '+call+' 20)',
            ' while (== '+call+' 0) { assert false }\n let marker: int = 41\n assert (== marker 41)')
        from nanoisa_owned_reconstruction import OwnedEmit
        for compiler in self.compilers:
            module=self.module(compiler,head+'fn main()'+main,'call-pretest-cap')
            facts=self.facts(module)
            self.assertLess(facts['functions'][0]['locals'],128)
            emitter=OwnedEmit([OwnedAnalyze(facts,i).run() for i in range(2)],'nano',facts['layouts'])
            cost=emitter.nano_local_cost(emitter.functions[0])
            self.assertGreater(cost,128)
            target=module.with_suffix('.nano');target.write_text('prior checked source')
            result=self.command([ROOT/'bin/nvm2hl','--language','nano',module,'-o',target],None)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('local',result.stderr)
            self.assertEqual(target.read_text(),'prior checked source')
            # I bypass only the diagnostic renderer to measure producer locals.
            emitter.nano_local_cost=lambda function:0
            diagnostic=self.module(compiler,emitter.run(0),'diagnostic-call-pretest-cap')
            assembly=self.command([ROOT/'bin/nanoisa','dump',diagnostic]).stdout
            self.assertRegex(assembly,rf'(?m)^\.function \S+ 0 {cost} ')
            self.assertNotEqual(self.command([ROOT/'bin/nanoisa_hl_facts',diagnostic],None).returncode,0)

    def test_return_body_and_zero_body_keep_complete_owner_consumption(self):
        cleanup="""let Tree { first, second, amount } = root
 let Counter { value } = first assert (== value 3)
 let Counter { value } = second assert (== value 2)
 let Flag { enabled } = amount assert enabled
 return 0"""
        for limit in (0,2):
            text="""resource struct Counter { value: int }
resource struct Flag { enabled: bool }
resource struct Tree { first: Counter, second: Counter, amount: Flag }
fn change(first: &mut Counter, second: &mut Counter, amount: &Flag) -> int {
 set first.value (+ first.value 1)
 set second.value (- second.value 1)
 assert amount.enabled
 return (- first.value second.value)
}
shadow change {
 let mut a: Counter = Counter { value: 1 }
 let mut b: Counter = Counter { value: 4 }
 let c: Flag = Flag { enabled: true }
 assert (== (change &mut a &mut b &c) -1)
 let Counter { value } = a assert (== value 2)
 let Counter { value } = b assert (== value 3)
 let Flag { enabled } = c assert enabled
}
fn main() -> int {
 let a: Counter = Counter { value: 1 }
 let b: Counter = Counter { value: 4 }
 let c: Flag = Flag { enabled: true }
 let mut root: Tree = Tree { first: a, second: b, amount: c }
 assert (== (change &mut root.first &mut root.second &root.amount) -1)
 while (< (change &mut root.first &mut root.second &root.amount) LIMIT) { CLEANUP }
 CLEANUP
}
shadow main { assert true }
""".replace('LIMIT',str(limit)).replace('CLEANUP',cleanup)
            for compiler in self.compilers:
                module=self.module(compiler,text,'call-pretest-return-small'+str(limit))
                native=module.with_suffix('.original.c')
                self.command([ROOT/'bin/nvm2c',module,'-o',native]);self.native(native,0)
                self.qualify(module)


if __name__ == '__main__':
    unittest.main()
