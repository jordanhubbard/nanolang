"""I compose exact borrowed formals with independent activation-local owners."""
import copy
import json
import os
from pathlib import Path
import re
import signal
import unittest
from tests import test_owned_nested_path_reconstruction as support
from tests import test_owned_void_reconstruction as failures
from nanoisa_owned_reconstruction import OwnedAnalyze, OwnedEmit, reconstruct_owned
from nanoisa_reconstruction import Refusal

ROOT = support.ROOT
ANCHOR = ROOT / 'tests/nanoisa/fixtures/source_borrow_helper_owners.nano'


class OwnedHelperLocalReconstruction(unittest.TestCase):
    setUpClass = support.OwnedNestedPathReconstruction.__dict__['setUpClass']
    command = support.OwnedNestedPathReconstruction.command
    facts = support.OwnedNestedPathReconstruction.facts
    native = support.OwnedNestedPathReconstruction.native
    module = support.OwnedNestedPathReconstruction.module
    qualify = support.OwnedNestedPathReconstruction.qualify
    selected_shadow = staticmethod(support.OwnedNestedPathReconstruction.selected_shadow)

    def native_failure(self, source, diagnostic):
        original_command = self.command
        def checked(args, expected=0, build=False):
            result = original_command(args, expected, build)
            if expected is None and not build:
                self.assertEqual(result.returncode, -signal.SIGABRT if diagnostic else 1)
                self.assertNotRegex(result.stderr, r'AddressSanitizer|UndefinedBehaviorSanitizer|runtime error:')
            return result
        self.command = checked
        try:
            failures.OwnedVoidReconstruction.native_failure(self, source, diagnostic)
        finally:
            self.command = original_command

    def helper_module(self, compiler, text=None, name='helper-owners'):
        return self.module(compiler, ANCHOR.read_text() if text is None else text, name)

    def test_unchanged_anchor_and_else_zero_paired_native(self):
        text = ANCHOR.read_text()
        variants = {'unchanged': text, 'else-zero': text.replace('if true {', 'if false {').replace('while (< index 2)', 'while (< index 0)')}
        for label, source in variants.items():
            for compiler in self.compilers:
                with self.subTest(case=label, producer=str(compiler)):
                    module = self.helper_module(compiler, source, label)
                    facts = self.facts(module)
                    self.assertEqual(facts['functions'][1]['param_modes'], [2, 1])
                    self.assertEqual([fn['locals'] for fn in facts['functions']], [12, 41])
                    native = module.with_suffix('.original.c')
                    self.command([ROOT/'bin/nvm2c', module, '-o', native]); self.native(native, 0)
                    self.qualify(module)

    def test_unchanged_origin_proof_and_exact_finite_plan(self):
        module = self.helper_module(self.compilers[0]); facts = self.facts(module)
        functions = [OwnedAnalyze(facts, i).run() for i in range(2)]
        original = copy.deepcopy(functions)
        c_before = OwnedEmit(functions, 'c', facts['layouts']).run(0)
        emitter = OwnedEmit(functions, 'nano', facts['layouts'])
        self.assertEqual(functions, original)
        self.assertEqual(emitter.functions[0], original[0])
        self.assertEqual(emitter.scratch_functions, set())
        self.assertEqual([emitter.nano_local_cost(fn) for fn in original], [49, 179])
        self.assertEqual([emitter.nano_local_cost(fn) for fn in emitter.functions], [49, 120])
        self.assertEqual(OwnedEmit(functions, 'c', facts['layouts']).run(0), c_before)
        self.assertEqual(emitter.helper_owner_plan['covered_statements'], 151)
        self.assertEqual(emitter.helper_owner_plan['original_statements'], 151)
        self.assertEqual(sum(s[0] == 'assert' for s in emitter.walk_statements(emitter.functions[1].body)), 19)
        self.assertEqual(emitter.control_proof['loops'][0]['iterations'], 2)
        self.assertLessEqual(emitter.control_proof['work'], 4096)
        self.assertLessEqual(emitter.control_proof['witness']['allocated_bindings'], 4096)

    def test_independent_original_shadow_input_and_generated_consistency_shadow(self):
        driver = os.environ.get('NANOLANG_OWNED_HELPER_SHADOW_EMITTER')
        assembly = self.work/'original-helper-shadow.nasm'
        if driver:
            assembly.write_text(self.command([driver, ANCHOR, '0', 'raw']).stdout)
        else:
            # This is retained original producer output, not recovered source shadows.
            assembly.write_bytes((ROOT/'docs/evidence/owned-helper-local-baseline-20261019/original-shadows.nasm').read_bytes())
        module = assembly.with_suffix('.nvm')
        self.command([ROOT/'bin/nanoisa', 'asm', assembly, '-o', module])
        self.command([ROOT/'bin/nano_vm', '--verify-only', module]); self.command([ROOT/'bin/nano_vm', module])
        facts = self.facts(module)
        self.assertEqual(len(facts['functions']), 2)
        self.assertEqual([fn['locals'] for fn in facts['functions']], [12, 41])
        self.qualify(module)
        text = reconstruct_owned(facts, 'nano')
        emitter = OwnedEmit([OwnedAnalyze(facts, i).run() for i in range(2)], 'nano', facts['layouts'])
        for compiler in self.compilers:
            selected = self.helper_module(compiler, self.selected_shadow(text), 'helper-consistency-shadow')
            actual = [fn['locals'] for fn in self.facts(selected)['functions']]
            self.assertLessEqual(actual[0], emitter.multi_shadow_local_cost())
            self.assertEqual(actual[1], 120)
            print('I measure independent helper consistency shadow '+json.dumps({'producer':str(compiler), 'actual':actual, 'estimated':emitter.multi_shadow_local_cost()}), flush=True)
            native = selected.with_suffix('.c')
            self.command([ROOT/'bin/nvm2c', selected, '-o', native]); self.native(native, 0)

    def test_later_different_shared_value_keeps_runtime_assertion(self):
        text = ANCHOR.read_text()
        head, main = text.split('fn main()', 1)
        # First actual invocation remains the literal source4 witness. A later
        # positional swap reads mutated target7 as shared source and must fail.
        main = main.replace('let source: Leaf', 'let mut source: Leaf')
        main = main.replace('(combine &mut target &source) 11', '(combine &mut source &target) 11')
        for compiler in self.compilers:
            source = self.work/'later-helper-false.nano'; source.write_text(head+'fn main()'+main)
            module = self.work/('later-helper-false-'+compiler.name+'.nvm')
            self.command([compiler, source, '--emit-nvm', '-o', module])
            failures.OwnedVoidReconstruction.vm_failure(self, module)
            for language in ('c', 'nano'):
                output = module.with_suffix('.'+language)
                self.command([ROOT/'bin/nvm2hl', '--language', language, module, '-o', output])
                if language == 'c':
                    self.native_failure(output, 'Assertion failed')
                else:
                    for producer in self.compilers:
                        rebuilt = output.with_suffix('.'+producer.name+'.nvm')
                        self.command([producer, output, '--emit-nvm', '-o', rebuilt])
                        failures.OwnedVoidReconstruction.vm_failure(self, rebuilt)
                        native = rebuilt.with_suffix('.c')
                        self.command([ROOT/'bin/nvm2c', rebuilt, '-o', native])
                        self.native_failure(native, None)

    def test_actual_generated_local_cap_preserves_original_input(self):
        text = ANCHOR.read_text()
        for count in (8, 9):
            added = ''.join(f' let boundary_{n}: int = {n}\n' for n in range(count))
            source = text.replace(' let first: Leaf =', added+' let first: Leaf =', 1)
            for compiler in self.compilers:
                module = self.helper_module(compiler, source, 'helper-cap-'+str(count))
                facts = self.facts(module)
                emitter = OwnedEmit([OwnedAnalyze(facts, i).run() for i in range(2)], 'nano', facts['layouts'])
                cost = emitter.nano_local_cost(emitter.functions[1])
                self.assertEqual(cost, 120+count)
                output = module.with_suffix('.nano'); output.write_text('prior accepted source')
                result = self.command([ROOT/'bin/nvm2hl', '--language', 'nano', module, '-o', output], None)
                if count == 9:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('128 local slots', result.stderr)
                    self.assertEqual(output.read_text(), 'prior accepted source')
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    rebuilt = output.with_suffix('.rebuilt.nvm')
                    self.command([compiler, output, '--emit-nvm', '-o', rebuilt])
                    self.command([ROOT/'bin/nano_vm', '--verify-only', rebuilt])
                    self.command([ROOT/'bin/nano_vm', rebuilt])
                    self.assertEqual(self.facts(rebuilt)['functions'][1]['locals'], 128)

    def test_defensive_descriptor_and_witness_refusals(self):
        facts = self.facts(self.helper_module(self.compilers[0]))
        variants = {}
        for label, operation in (
            ('formal-overwrite', lambda ins: ins['args'].__setitem__(0, 0)),
            ('float-descriptor', lambda ins: ins['args'].__setitem__(0, 2.0)),
            ('missing-operand', lambda ins: ins['args'].pop()),
            ('wrong-local-slot', lambda ins: ins['args'].__setitem__(1, 0)),
        ):
            bad = copy.deepcopy(facts)
            ins = next(i for i in bad['functions'][1]['code'] if i['op'] == 'BORROW_LOCAL_SHARED')
            operation(ins); variants[label] = bad
        bad = copy.deepcopy(facts)
        ins = next(i for i in bad['functions'][1]['code'] if i['op'] == 'REF_GET' and i['arg'] == 1)
        ins['arg'] = 0; variants['mismatched-formal-operand'] = bad
        bad = copy.deepcopy(facts)
        ins = next(i for i in bad['functions'][1]['code'] if i['op'] == 'BORROW_PATH_SHARED')
        ins['args'][0] = 1; variants['path-overwrite-formal'] = bad
        bad = copy.deepcopy(facts)
        ins = next(i for i in bad['functions'][0]['code'] if i['op'] == 'PUSH_I64' and i['arg'] == 4)
        ins['arg'] = 0; ins['args'] = [0]; variants['invalid-first-witness'] = bad
        for label, bad in variants.items():
            for language in ('c', 'nano'):
                with self.subTest(case=label, language=language), self.assertRaises(Refusal):
                    reconstruct_owned(bad, language)

    def test_original_verifier_rejects_formal_descriptor_overwrite(self):
        module = self.helper_module(self.compilers[0])
        text = self.command([ROOT/'bin/nanoisa', 'dump', module]).stdout
        self.assertIn('BORROW_LOCAL_SHARED 2 ', text)
        bad = text.replace('BORROW_LOCAL_SHARED 2 ', 'BORROW_LOCAL_SHARED 0 ', 1)
        assembly = self.work/'descriptor-overwrite.nasm'; assembly.write_text(bad)
        malformed = assembly.with_suffix('.nvm')
        assembled = self.command([ROOT/'bin/nanoisa', 'asm', assembly, '-o', malformed], None)
        if assembled.returncode == 0:
            rejected = self.command([ROOT/'bin/nano_vm', '--verify-only', malformed], None)
            self.assertNotEqual(rejected.returncode, 0)
            for language in ('c', 'nano'):
                output = malformed.with_suffix('.'+language); output.write_text('prior source')
                result = self.command([ROOT/'bin/nvm2hl', '--language', language, malformed, '-o', output], None)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(output.read_text(), 'prior source')
        else:
            self.assertRegex(assembled.stderr, r'(?i)verif|borrow|reference|region')


if __name__ == '__main__':
    unittest.main()
