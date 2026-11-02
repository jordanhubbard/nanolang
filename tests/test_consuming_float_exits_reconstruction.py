"""I preserve consuming owner origins across FLOAT-local loop exits."""
import copy
import json
import os
import shlex
import sys
import unittest
from tests import test_owned_float_reconstruction as support
from nanoisa_owned_reconstruction import OwnedAnalyze, OwnedEmit, reconstruct_owned
from nanoisa_owned_exits import prove_exit_origins
from nanoisa_reconstruction import Expr, Function, INT, BOOL, Refusal
ROOT=support.ROOT
BASE=ROOT/'docs/evidence/consuming-float-exits-baseline-20261020'

class ConsumingFloatExits(unittest.TestCase):
    setUpClass=support.OwnedFloatReconstruction.__dict__['setUpClass']
    command=support.OwnedFloatReconstruction.command
    module=support.OwnedFloatReconstruction.module
    facts=support.OwnedFloatReconstruction.facts
    def native(self,source,failure=False,retained=False):
        for opt in ('-O0','-O2'):
            binary=source.with_suffix('.'+opt[1:]+'.exe')
            self.command([*shlex.split(os.environ.get('NANO_NATIVE_TEST_CC','cc')),'-std=c11',opt,
                '-DNDEBUG','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',
                source,self.tools/'nano_aot_runtime.o','-lm',*support.native_link_flags(),'-o',binary])
            if not failure:self.command([binary]);continue
            self.command([sys.executable,'-c',
                'import os,resource,sys; resource.setrlimit(resource.RLIMIT_CORE,(0,0)); os.execv(sys.argv[1],sys.argv[1:])',binary],1)
            status=json.loads((self.work/('command-'+str(self.serial)+'-status.json')).read_text())
            error=(self.work/('command-'+str(self.serial)+'.stderr')).read_text()
            self.assertNotIn('AddressSanitizer',error);self.assertNotIn('runtime error:',error)
            if retained:
                self.assertEqual(status['returncode'],1);self.assertEqual(error,'')
            else:
                self.assertEqual(status['returncode'],-6);self.assertIn('Assertion failed',error)

    def vm_assert_failure(self,module):
        self.command([self.tools/'nano_vm',module],1)
        status=json.loads((self.work/('command-'+str(self.serial)+'-status.json')).read_text())
        error=(self.work/('command-'+str(self.serial)+'.stderr')).read_text()
        self.assertEqual(status['returncode'],1)
        self.assertIn('I could not execute the module: Assertion failed',error)
        self.assertNotIn('AddressSanitizer',error);self.assertNotIn('runtime error:',error)

    def qualify(self,module):
        facts=self.facts(module)
        original=module.with_suffix('.original.c')
        self.command([self.tools/'nvm2c',module,'-o',original]);self.native(original)
        for language in ('c','nano'):
            output=module.with_suffix('.reconstructed.'+language);output.write_text(reconstruct_owned(facts,language))
            if language=='c':self.native(output)
            else:
                for compiler in self.compilers:
                    rebuilt=self.module(output.read_text(),compiler)
                    generated=self.facts(rebuilt)
                    self.assertEqual(generated['layouts'],facts['layouts'])
                    native=rebuilt.with_suffix('.native.c');self.command([self.tools/'nvm2c',rebuilt,'-o',native]);self.native(native)
        return facts

    def source(self):return (BASE/'original.nano').read_text()

    def test_unchanged_original_and_independent_shadow(self):
        for compiler in self.compilers:
            facts=self.qualify(self.module(self.source(),compiler))
            self.assertEqual([fn['locals'] for fn in facts['functions']],[2,3,8])
        assembly=self.work/'original-shadow.nasm';assembly.write_bytes((BASE/'original-shadows.nasm').read_bytes())
        module=assembly.with_suffix('.nvm');self.command([self.tools/'nanoisa','asm',assembly,'-o',module])
        self.command([self.tools/'nano_vm','--verify-only',module]);self.command([self.tools/'nano_vm',module])
        self.qualify(module)

    def test_later_assertion_is_not_elided(self):
        # I start with the independently verified baseline and alter only a
        # terminal assertion after its actual loops and owner-consuming call.
        text=(BASE/'nano_virt.nasm').read_text()
        needle='  OWN_MOVE_LOCAL 0\n  CALL 1\n  RET'
        self.assertEqual(text.count(needle),1)
        assembly=self.work/'later-failure.nasm'
        assembly.write_text(text.replace(needle,'  OWN_MOVE_LOCAL 0\n  CALL 1\n  PUSH_BOOL 0\n  ASSERT\n  RET'))
        module=assembly.with_suffix('.nvm');self.command([self.tools/'nanoisa','asm',assembly,'-o',module])
        self.command([self.tools/'nano_vm','--verify-only',module]);self.vm_assert_failure(module)
        facts=self.facts(module)
        original=module.with_suffix('.original.c');self.command([self.tools/'nvm2c',module,'-o',original]);self.native(original,True,True)
        for language in ('c','nano'):
            output=module.with_suffix('.'+language);output.write_text(reconstruct_owned(facts,language))
            if language=='c':self.native(output,True)
            else:
                for compiler in self.compilers:
                    rebuilt=output.with_suffix('.'+compiler.name+'.nvm')
                    self.command([compiler,output,'--emit-nvm','-o',rebuilt])
                    self.command([self.tools/'nano_vm','--verify-only',rebuilt]);self.vm_assert_failure(rebuilt)
                    native=rebuilt.with_suffix('.c');self.command([self.tools/'nvm2c',rebuilt,'-o',native]);self.native(native,True,True)
        # Removing only that injected assertion recovers the original success.
        good=self.work/'causality.nasm';good.write_text(text);good_module=good.with_suffix('.nvm')
        self.command([self.tools/'nanoisa','asm',good,'-o',good_module]);self.command([self.tools/'nano_vm',good_module])

    def test_actual128_estimated129_atomic_boundary(self):
        for count in (116,117):
            # I add scalar locals only to the entry, whose estimate is exact;
            # the existing noninline ADD helper makes worker estimates conservative.
            text=self.source();begin=text.index('fn main()');end=text.index('return 0',begin)
            text=text[:end]+''.join(f'let cap_{i}: int = {i}\n' for i in range(count))+text[end:]
            for compiler in self.compilers:
                module=self.module(text,compiler);facts=self.facts(module)
                functions=[OwnedAnalyze(facts,i).run() for i in range(len(facts['functions']))]
                emit=OwnedEmit(functions,'nano',facts['layouts'])
                self.assertEqual(emit.nano_local_cost(functions[0]),12+count)
                output=module.with_suffix('.cap.nano');output.write_text('prior source')
                self.command([sys.executable,ROOT/'scripts/nvm2hl.py','--language','nano',module,'-o',output],0 if count==116 else 1)
                if count==117:self.assertEqual(output.read_text(),'prior source')
                else:
                    rebuilt=self.module(output.read_text(),compiler)
                    self.assertEqual(self.facts(rebuilt)['functions'][0]['locals'],128)

    def test_immutable_origins_and_no_borrowed_witness(self):
        facts=json.loads((BASE/'nano_virt.facts.json').read_text())
        functions=[OwnedAnalyze(facts,i).run() for i in range(len(facts['functions']))]
        original=copy.deepcopy(functions)
        emit=OwnedEmit(functions,'nano',facts['layouts'])
        self.assertTrue(emit.consuming_exits);self.assertFalse(emit.loop_exits)
        self.assertIsNone(emit.witness);self.assertFalse(emit.scratch_functions)
        self.assertFalse(emit.reassignment_plans)
        self.assertEqual([fn.body for fn in functions],[fn.body for fn in original])
        self.assertNotRegex(emit.run(0),r'(?m)^shadow (?:main|nlr_f[0-9])')

    def test_cached_side_entry_and_zero_trip_initialization_refuse(self):
        original=json.loads((BASE/'nano_virt.facts.json').read_text())
        for case in ('side-entry','zero-trip-local','result','nested-owner'):
            facts=copy.deepcopy(original);code=facts['functions'][2]['code']
            if case=='side-entry':
                store=next(i for i in code if i['op']=='STORE_LOCAL' and i['arg']==3)
                branch=next(i for i in code if i['op']=='JMP_FALSE')
                following=code[code.index(branch)+1]
                store.update(op='JMP',target=following['pc'])
            elif case=='zero-trip-local':
                load=next(i for i in code if i['op']=='LOAD_LOCAL' and i['arg']==3)
                load.update(arg=5,args=[5])
            elif case=='result':facts['functions'][2]['result_type']=[BOOL,4294967295]
            else:facts['layouts'][0]['fields']=[8]
            for language in ('c','nano'):
                with self.subTest(case=case,language=language),self.assertRaises(Refusal):reconstruct_owned(facts,language)

    def test_parameter_origin_survives_every_break_and_backedge(self):
        facts=json.loads((BASE/'nano_virt.facts.json').read_text())
        functions=[OwnedAnalyze(facts,i).run() for i in range(len(facts['functions']))]
        owner=Expr(('owner',0),'temporary','moved');result=Expr(INT,'temporary','closed')
        consume=[('own_move',owner,0),('let',result,Expr(INT,'call',1,(owner,)))]
        for transfer in ('break','continue'):
            good=Function('worker',[('owner',0),BOOL],INT,{},[],[
                ('own_parameter',0,0),('while',Expr(BOOL,'constant',False),[(transfer,)]),
                *consume,('return',result)])
            prove_exit_origins(good,functions,facts['layouts'])
            for hidden in (consume,[('if',Expr(BOOL,'constant',True),consume,[])],
                           consume+[('own_pack',owner,0,(Expr(INT,'constant',42),)),('own_store',0,0,owner)]):
                bad=copy.deepcopy(good);bad.body[1]=('while',Expr(BOOL,'constant',False),hidden+[(transfer,)])
                with self.assertRaises(Refusal):prove_exit_origins(bad,functions,facts['layouts'])

    def test_consuming_call_origin_defensive_refusals(self):
        facts=json.loads((BASE/'nano_virt.facts.json').read_text())
        original=[OwnedAnalyze(facts,i).run() for i in range(len(facts['functions']))]
        worker=original[2]
        call_index=next(i for i,s in enumerate(worker.body) if s[0]=='let' and s[2].kind=='call')
        call=worker.body[call_index][2]
        for replacement in (Expr(INT,'call',0,call.args),Expr(INT,'call',call.value,()),
                            Expr(BOOL,'call',call.value,call.args)):
            functions=copy.deepcopy(original);statement=functions[2].body[call_index]
            functions[2].body[call_index]=('let',statement[1],replacement)
            with self.assertRaises(Refusal):OwnedEmit(functions,'nano',facts['layouts'])
        functions=copy.deepcopy(original)
        functions[2].body.insert(call_index,('assert',Expr(BOOL,'binary','==',(Expr(INT,'call',1,(Expr(INT,'constant',0),)),Expr(INT,'constant',0)))))
        with self.assertRaises(Refusal):OwnedEmit(functions,'nano',facts['layouts'])
        functions=copy.deepcopy(original);functions[2].body.pop(call_index)
        with self.assertRaises(Refusal):OwnedEmit(functions,'nano',facts['layouts'])

if __name__=='__main__':unittest.main()
