"""I use only the first eager pretest's actual before-call literal tree."""
import copy
import re
import unittest
from tests import test_owned_call_pretest_reconstruction as support
from tests import test_owned_double_call_pretest_reconstruction as double
from nanoisa_owned_reconstruction import OwnedAnalyze, OwnedEmit, reconstruct_owned
from nanoisa_reconstruction import Refusal
ROOT=support.ROOT
B='(change &mut root.branch.right &mut root.branch.left &root.other)'

class OwnedLoopFirstWitness(unittest.TestCase):
    setUpClass=support.OwnedCallPretestReconstruction.__dict__['setUpClass']
    command=support.OwnedCallPretestReconstruction.command
    facts=support.OwnedCallPretestReconstruction.facts
    native=support.OwnedCallPretestReconstruction.native
    module=support.OwnedCallPretestReconstruction.module
    qualify=support.OwnedCallPretestReconstruction.qualify
    native_assert_failure=double.OwnedDoubleCallPretest.native_assert_failure

    @staticmethod
    def source(iterations=0):
        text=support.OwnedCallPretestReconstruction.source(first_only=True)
        return text.replace(f'(< {B} 20)',f'(< {B} {26+6*iterations})').replace(
            'assert (== value 7)',f'assert (== value {7-3*iterations})').replace(
            'assert (== value 33)',f'assert (== value {33+3*iterations})')

    def test_unchanged_original_anchor(self):
        for compiler in self.compilers:
            self.qualify(self.module(compiler,support.OwnedCallPretestReconstruction.source(first_only=True),'loop-first-unchanged'))

    def test_actual_zero_entered_final_false(self):
        for count in (0,1,2):
            for compiler in self.compilers:self.qualify(self.module(compiler,self.source(count),'loop-first-'+str(count)))

    def test_immutable_before_call_and_selected_shadow(self):
        for compiler in self.compilers:
            facts=self.facts(self.module(compiler,self.source(),'loop-first-shadow'))
            functions=[OwnedAnalyze(facts,i).run() for i in range(2)];original=copy.deepcopy(functions)
            emit=OwnedEmit(functions,'nano',facts['layouts'])
            self.assertEqual([leaf[0].value for leaf in emit.witness[2]],[30,10,3])
            self.assertEqual(functions[0].body,original[0].body)
            self.assertFalse(emit.scratch_functions)
            self.assertEqual([emit.nano_local_cost(f) for f in emit.functions],[123,31])
            selected=support.nested.OwnedNestedPathReconstruction.selected_shadow(emit.run(0))
            for producer in self.compilers:
                rebuilt=self.module(producer,selected,'loop-first-selected')
                self.assertEqual([f['locals'] for f in self.facts(rebuilt)['functions']],[108,31])
                native=rebuilt.with_suffix('.native.c');self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native]);self.native(native,0)

    def test_no_body_or_after_loop_or_prefix_assertion_witness(self):
        original=self.source();head,main=original.split('fn main()',1)
        cases={
            'prefix-assert':main.replace(' while ',' assert true\n while ',1),
            'prefix-branch':main.replace(' while ',' if true { assert true }\n while ',1),
            'body-only':main.replace(f'while (< {B} 26) {{ assert true }}',f'while false {{ let result: int = {B} }}\n let result: int = {B}'),
            'nonliteral':main.replace('value: 3, enabled: true','value: (+ 1 2), enabled: true'),
        }
        for name,body in cases.items():
            for compiler in self.compilers:
                module=self.module(compiler,head+'fn main()'+body,name)
                for language in ('c','nano'):
                    output=module.with_suffix('.'+language);output.write_text('prior source')
                    result=self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output],None)
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn('literal exact child snapshots' if name=='nonliteral' else 'witness',result.stderr)
                    self.assertEqual(output.read_text(),'prior source')

    def test_false_initial_shared_bool_refuses_atomically(self):
        from tests.test_owned_void_reconstruction import OwnedVoidReconstruction as failure
        head,main=self.source().split('fn main()',1)
        main=main.replace('value: 3, enabled: true','value: 3, enabled: false')
        for compiler in self.compilers:
            source=self.work/('loop-first-false-'+compiler.name+'.nano');source.write_text(head+'fn main()'+main)
            module=source.with_suffix('.nvm');self.command([compiler,source,'--emit-nvm','-o',module]);failure.vm_failure(self,module)
            for language in ('c','nano'):
                output=module.with_suffix('.'+language);output.write_text('prior source')
                result=self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output],None)
                self.assertNotEqual(result.returncode,0);self.assertIn('true actual tree shared-BOOL',result.stderr)
                self.assertEqual(output.read_text(),'prior source')

    def test_no_call_in_first_pretest_cannot_reach_later_call(self):
        facts=self.facts(self.module(self.compilers[0],self.source(),'loop-first-defensive'))
        functions=[OwnedAnalyze(facts,i).run() for i in range(2)]
        index=next(i for i,s in enumerate(functions[0].body) if s[0]=='while_observe')
        loop=functions[0].body[index]
        pretest=list(loop[2]);call=next(s for s in pretest if s[0]=='path_borrow_call')
        # I inspect private proof refusal; these deliberately altered IRs are not
        # claimed verified executable modules.
        for prelude in ([],[('assert',loop[1])]):
            bad=copy.deepcopy(functions);bad[0].body[index]=('while_observe',loop[1],prelude,loop[3]);bad[0].body.insert(index+1,call)
            with self.assertRaises(Refusal):OwnedEmit(bad,'nano',facts['layouts'])
        bad=copy.deepcopy(functions);bad[0].loop_first_witness=False
        with self.assertRaises(Refusal):OwnedEmit(bad,'nano',facts['layouts'])

    def test_actual128_and_estimated129_atomic_refusal(self):
        for count in (5,6):
            head,main=self.source().split('fn main()',1)
            main=main.replace('return 0',''.join(f'let cap_{i}: int = {i}\n' for i in range(count))+' return 0',1)
            for compiler in self.compilers:
                module=self.module(compiler,head+'fn main()'+main,'loop-first-cap-'+str(count))
                facts=self.facts(module);emit=OwnedEmit([OwnedAnalyze(facts,i).run() for i in range(2)],'nano',facts['layouts'])
                self.assertEqual(emit.nano_local_cost(emit.functions[0]),123+count)
                output=module.with_suffix('.nano');output.write_text('prior source')
                result=self.command([ROOT/'bin/nvm2hl','--language','nano',module,'-o',output],None)
                if count==6:
                    self.assertNotEqual(result.returncode,0);self.assertIn('128 local slots',result.stderr);self.assertEqual(output.read_text(),'prior source')
                else:
                    self.assertEqual(result.returncode,0,result.stderr)
                    rebuilt=self.module(compiler,output.read_text(),'loop-first128')
                    self.assertEqual(self.facts(rebuilt)['functions'][0]['locals'],128)

    def test_later_body_helper_assertion_stays_runtime_failure(self):
        from tests.test_owned_void_reconstruction import OwnedVoidReconstruction as failure
        text=self.source().replace('resource struct Other { value: int, enabled: bool }\n','').replace('Other','Leaf')
        head,main=text.split('fn main()',1)
        bad='(change &mut root.other &mut root.branch.left &root.branch.right)'
        main=main.replace(f'while (< {B} 26) {{ assert true }}',f'while (< {B} 27) {{ let result: int = {bad} }}')
        main=re.sub(r'assert \(== value -?\d+\)', '',main)
        text=head+'fn main()'+main
        for compiler in self.compilers:
            self.module(compiler,text.replace('assert amount.enabled','assert true'),'loop-first-causality')
            source=self.work/('loop-first-failure-'+compiler.name+'.nano');source.write_text(text);module=source.with_suffix('.nvm')
            self.command([compiler,source,'--emit-nvm','-o',module]);failure.vm_failure(self,module)
            original_native=module.with_suffix('.original.c');self.command([ROOT/'bin/nvm2c',module,'-o',original_native]);self.native_assert_failure(original_native,False)
            failure.native_assert_status(self,original_native)
            for language in ('c','nano'):
                output=module.with_suffix('.'+language);self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output])
                if language=='c':self.native_assert_failure(output,True)
                else:
                    for producer in self.compilers:
                        rebuilt=output.with_suffix('.'+producer.name+'.nvm');self.command([producer,output,'--emit-nvm','-o',rebuilt]);failure.vm_failure(self,rebuilt)
                        native=rebuilt.with_suffix('.native.c');self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native]);self.native_assert_failure(native,False);failure.native_assert_status(self,native)

if __name__=='__main__':unittest.main()
