"""I retain eager ordered scalar snapshots across two complete borrow regions."""
import copy
import re
import os
import shlex
import signal
import sys
from tests.native_process import native_link_flags
import unittest
from tests import test_owned_call_pretest_reconstruction as support
from nanoisa_owned_reconstruction import OwnedAnalyze, OwnedEmit, reconstruct_owned
from nanoisa_reconstruction import Refusal
ROOT = support.ROOT
A = '(change &mut root.branch.left &mut root.branch.right &root.other)'
B = '(change &mut root.branch.right &mut root.branch.left &root.other)'


class OwnedDoubleCallPretest(unittest.TestCase):
    setUpClass = support.OwnedCallPretestReconstruction.__dict__['setUpClass']
    command = support.OwnedCallPretestReconstruction.command
    facts = support.OwnedCallPretestReconstruction.facts
    native = support.OwnedCallPretestReconstruction.native
    module = support.OwnedCallPretestReconstruction.module
    qualify = support.OwnedCallPretestReconstruction.qualify

    @staticmethod
    def source(iterations=0):
        text = support.OwnedCallPretestReconstruction.source(double=True)
        text = text.replace(f'(== (+ {B} {B}) 0)', f'(< (+ {B} {B}) {46+24*iterations})')
        return text.replace('assert (== value 7)', f'assert (== value {7-6*iterations})').replace(
            'assert (== value 33)', f'assert (== value {33+6*iterations})')

    def test_zero_entered_and_final_false_effects(self):
        for iterations in (0,1,2):
            for compiler in self.compilers:
                module = self.module(compiler,self.source(iterations),'double-call-'+str(iterations))
                self.qualify(module)

    def test_order_and_first_result_snapshot(self):
        text = support.OwnedCallPretestReconstruction.source(double=True)
        text = text.replace(f'(== (+ {B} {B}) 0)',f'(not (== (- {A} {B}) -22))')
        text = text.replace(') { assert true }',') { assert false }')
        text = text.replace('assert (== value 7)','assert (== value 13)').replace('assert (== value 33)','assert (== value 27)')
        for compiler in self.compilers:
            self.qualify(self.module(compiler,text,'double-call-order'))
        snapshot = support.OwnedCallPretestReconstruction.source(double=True).replace(
            f'(== (+ {B} {B}) 0)',f'(not (== (+ {B} {B}) 46))').replace(') { assert true }',') { assert false }')
        for compiler in self.compilers:
            self.qualify(self.module(compiler,snapshot,'double-call-snapshot'))

    def test_pretest_ir_and_unchanged_one_call_route(self):
        module = self.module(self.compilers[0],self.source(),'double-call-ir')
        facts = self.facts(module);functions = [OwnedAnalyze(facts,i).run() for i in range(2)]
        original = copy.deepcopy(functions)
        prior = copy.deepcopy(functions);prior[0].double_call_pretest = False
        old = OwnedEmit(prior,'nano',facts['layouts']);new = OwnedEmit(functions,'nano',facts['layouts'])
        def pretests(fn):return [s[2] for s in new.walk_statements(fn.body) if s[0]=='while_observe']
        self.assertEqual(pretests(old.functions[0]),pretests(new.functions[0]))
        self.assertEqual(sum(s[0]=='path_borrow_call' for s in pretests(new.functions[0])[0]),2)
        self.assertEqual(functions[0].body,original[0].body)
        self.assertEqual(new.functions[1].body,old.functions[1].body);self.assertFalse(new.scratch_functions)
        self.assertEqual([new.nano_local_cost(f) for f in new.functions],[85,31])
        self.assertEqual([old.nano_local_cost(f) for f in old.functions],[132,31])
        one = self.facts(self.module(self.compilers[0],support.OwnedCallPretestReconstruction.source(),'one-call'))
        self.assertFalse(OwnedAnalyze(one,0).run().double_call_pretest)

    def test_independently_selected_consistency_shadow(self):
        for compiler in self.compilers:
            module = self.module(compiler,self.source(),'double-shadow-source')
            facts = self.facts(module)
            functions = [OwnedAnalyze(facts,i).run() for i in range(2)]
            emitter = OwnedEmit(functions,'nano',facts['layouts'])
            text = emitter.run(0)
            selected = support.nested.OwnedNestedPathReconstruction.selected_shadow(text)
            expected = [emitter.nano_local_cost(emitter.path_shadow_function()),
                        emitter.nano_local_cost(emitter.functions[1])]
            for producer in self.compilers:
                rebuilt = self.module(producer,selected,'double-selected-shadow')
                actual = self.facts(rebuilt)
                self.assertEqual([fn['locals'] for fn in actual['functions']],expected)
                native = rebuilt.with_suffix('.native.c')
                self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native])
                self.native(native,0)

    def native_assert_failure(self,source,direct):
        for opt in ('-O0','-O2'):
            binary = source.with_suffix('.'+opt[1:]+'.failure.exe')
            self.command([*shlex.split(os.environ.get('NANO_NATIVE_TEST_CC','cc')),
                '-std=c11',opt,'-DNDEBUG','-Wall','-Wextra','-Werror',
                '-fsanitize=address,undefined','-fno-sanitize-recover=all',source,
                ROOT/'bin/nano_aot_runtime.o','-lm',*native_link_flags(),'-o',binary],build=True)
            result = self.command([sys.executable,'-c',
                'import os,resource,sys; resource.setrlimit(resource.RLIMIT_CORE,(0,0)); os.execv(sys.argv[1],sys.argv[1:])',binary],None)
            self.assertEqual(result.returncode,-signal.SIGABRT if direct else 1)
            if direct:self.assertIn('Assertion failed',result.stderr)
            else:self.assertEqual(result.stderr,'')
            self.assertNotIn('AddressSanitizer',result.stderr)
            self.assertNotIn('runtime error:',result.stderr)
            self.assertNotIn('UndefinedBehaviorSanitizer',result.stderr)

    def test_isolated_second_call_assertion_failure(self):
        from tests.test_owned_void_reconstruction import OwnedVoidReconstruction as failure
        text = support.OwnedCallPretestReconstruction.source(double=True)
        text = text.replace('resource struct Other { value: int, enabled: bool }\n','').replace('Other','Leaf')
        head, main = text.split('fn main()',1)
        bad = '(change &mut root.branch.left &mut root.other &root.branch.right)'
        main = main.replace(f'(+ {B} {B})',f'(+ {B} {bad})')
        main = re.sub(r'assert \(== value -?\d+\)', 'assert true',main)
        text = head+'fn main()'+main
        for compiler in self.compilers:
            control = text.replace('assert amount.enabled','assert true')
            self.module(compiler,control,'second-call-causality')
            source = self.work/('second-call-failure-'+compiler.name+'.nano');source.write_text(text)
            module = source.with_suffix('.nvm')
            self.command([compiler,source,'--emit-nvm','-o',module]);failure.vm_failure(self,module)
            original_native = module.with_suffix('.original.c')
            self.command([ROOT/'bin/nvm2c',module,'-o',original_native])
            self.native_assert_failure(original_native,False)
            failure.native_assert_status(self,original_native)
            for language in ('c','nano'):
                output=module.with_suffix('.'+language)
                self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output])
                if language=='c':self.native_assert_failure(output,True)
                else:
                    for producer in self.compilers:
                        rebuilt=output.with_suffix('.'+producer.name+'.nvm')
                        self.command([producer,output,'--emit-nvm','-o',rebuilt]);failure.vm_failure(self,rebuilt)
                        native = rebuilt.with_suffix('.native.c')
                        self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native])
                        self.native_assert_failure(native,False)
                        failure.native_assert_status(self,native)

    def test_actual_local_limit(self):
        for count in (43,44):
            text=self.source();head,main=text.split('fn main()',1)
            main=main.replace('return 0',''.join(f'let cap_{i}: int = {i}\n' for i in range(count))+' return 0',1)
            for compiler in self.compilers:
                module=self.module(compiler,head+'fn main()'+main,'double-cap-'+str(count))
                facts = self.facts(module)
                emitter = OwnedEmit([OwnedAnalyze(facts,i).run() for i in range(2)],'nano',facts['layouts'])
                self.assertEqual(emitter.nano_local_cost(emitter.functions[0]),85+count)
                output=module.with_suffix('.nano');output.write_text('prior source')
                result=self.command([ROOT/'bin/nvm2hl','--language','nano',module,'-o',output],None)
                if count==44:
                    self.assertNotEqual(result.returncode,0);self.assertIn('128 local slots',result.stderr)
                    self.assertEqual(output.read_text(),'prior source')
                else:
                    self.assertEqual(result.returncode,0,result.stderr)
                    rebuilt=self.module(compiler,output.read_text(),'double-cap-128')
                    self.assertEqual(self.facts(rebuilt)['functions'][0]['locals'],128)

    def test_third_call_and_split_region_refuse(self):
        text = support.OwnedCallPretestReconstruction.source(double=True).replace(
            f'(+ {B} {B})',f'(+ (+ {B} {B}) {B})').replace('assert (== value 7)','assert (== value 4)').replace('assert (== value 33)','assert (== value 36)')
        for compiler in self.compilers:
            module = self.module(compiler,text,'three-call')
            for language in ('c','nano'):
                output = module.with_suffix('.'+language);output.write_text('prior checked source')
                result = self.command([ROOT/'bin/nvm2hl','--language',language,module,'-o',output],None)
                self.assertNotEqual(result.returncode,0);self.assertIn('at most two',result.stderr)
                self.assertEqual(output.read_text(),'prior checked source')
        facts = self.facts(self.module(self.compilers[0],self.source(),'double-defensive'))
        code = facts['functions'][0]['code']
        calls = [i for i,ins in enumerate(code) if ins['op']=='CALL_REF'];self.assertEqual(len(calls),3)
        for change in ('region','descriptor','interior'):
            bad = copy.deepcopy(facts);body=bad['functions'][0]['code'];i=calls[-1]
            if change=='region':body[i+1]['op']='NOP'
            elif change=='descriptor':body[i]['args'][1]+=1
            else:
                back = next(ins for ins in reversed(body) if ins['op']=='JMP' and ins['target']<=ins['pc'])
                back['target']=body[i-3]['pc']
            for language in ('c','nano'):
                with self.assertRaises(Refusal):reconstruct_owned(bad,language)


if __name__ == '__main__':unittest.main()
