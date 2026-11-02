"""I preserve one inherited exclusive view across independent local owners."""
import copy
import os
import unittest
from tests import test_owned_nested_path_reconstruction as support
from nanoisa_owned_reconstruction import OwnedAnalyze, OwnedEmit, reconstruct_owned
from nanoisa_reconstruction import Refusal

ROOT = support.ROOT


def original_source():
    text = (ROOT/'tests/nanoisa/fixtures/source_borrow_resource_paths.nano').read_text()
    old = 'set view.value (+ view.value 1)'
    assert text.count(old) == 1
    return text.replace(old, 'let owner: Leaf = Leaf { value: 1, active: true } '
                        'let Leaf { value, active } = owner '+old)


class SingleExclusiveHelperLocalReconstruction(unittest.TestCase):
    setUpClass = support.OwnedNestedPathReconstruction.__dict__['setUpClass']
    command = support.OwnedNestedPathReconstruction.command
    facts = support.OwnedNestedPathReconstruction.facts
    native = support.OwnedNestedPathReconstruction.native
    module = support.OwnedNestedPathReconstruction.module
    qualify = support.OwnedNestedPathReconstruction.qualify
    selected_shadow = staticmethod(support.OwnedNestedPathReconstruction.selected_shadow)

    def test_unchanged_source_and_native_roundtrips(self):
        for compiler in self.compilers:
            module = self.module(compiler, original_source(), 'single-helper')
            facts = self.facts(module)
            self.assertEqual([f['locals'] for f in facts['functions']], [52, 7])
            self.assertEqual(facts['functions'][1]['param_modes'], [2])
            native = module.with_suffix('.original.c')
            self.command([ROOT/'bin/nvm2c', module, '-o', native]); self.native(native, 0)
            self.qualify(module)

    def test_origin_proofs_and_caller_only_plan(self):
        facts = self.facts(self.module(self.compilers[0], original_source()))
        functions = [OwnedAnalyze(facts, i).run() for i in range(2)]
        frozen = copy.deepcopy(functions)
        emitter = OwnedEmit(functions, 'nano', facts['layouts'])
        self.assertEqual(functions, frozen)
        self.assertEqual(emitter.functions[1], frozen[1])
        self.assertEqual(emitter.scratch_functions, set())
        self.assertEqual([emitter.nano_local_cost(f) for f in emitter.functions], [127, 29])
        self.assertEqual(emitter.single_helper_caller_plan['covered_statements'], 156)
        self.assertEqual(emitter.single_helper_caller_plan['original_statements'], 156)
        self.assertEqual(sum(s[0] == 'assert' for s in emitter.walk_statements(emitter.functions[0].body)), 15)
        self.assertEqual(emitter.control_proof['work'], 288)

    def test_original_and_generated_shadows(self):
        driver = os.environ.get('NANOLANG_SINGLE_HELPER_SHADOW_EMITTER')
        assembly = self.work/'single-original-shadow.nasm'
        if driver:
            source = self.work/'single-original.nano'; source.write_text(original_source())
            assembly.write_text(self.command([driver, source, '0', 'raw']).stdout)
        else:
            assembly.write_bytes((ROOT/'docs/evidence/single-exclusive-helper-baseline-20261019/original-shadows.nasm').read_bytes())
        module = assembly.with_suffix('.nvm')
        self.command([ROOT/'bin/nanoisa', 'asm', assembly, '-o', module])
        self.command([ROOT/'bin/nano_vm', '--verify-only', module]); self.command([ROOT/'bin/nano_vm', module])
        self.qualify(module)
        facts = self.facts(module)
        text = reconstruct_owned(facts, 'nano')
        for compiler in self.compilers:
            rebuilt = self.module(compiler, self.selected_shadow(text), 'single-selected-shadow')
            self.assertEqual([f['locals'] for f in self.facts(rebuilt)['functions']], [60, 29])
            native = rebuilt.with_suffix('.c')
            self.command([ROOT/'bin/nvm2c', rebuilt, '-o', native]); self.native(native, 0)

    def test_actual_local_boundary_preserves_output(self):
        for count in (1, 2):
            extra = ''.join(f' let boundary_{n}: int = {n}\n' for n in range(count))
            text = original_source().replace(' # RETURN_POINT\n', extra+' # RETURN_POINT\n')
            for compiler in self.compilers:
                module = self.module(compiler, text, 'single-boundary-'+str(count))
                output = module.with_suffix('.nano'); output.write_text('prior source')
                result = self.command([ROOT/'bin/nvm2hl', '--language', 'nano', module, '-o', output], None)
                if count == 2:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('128 local slots', result.stderr)
                    self.assertEqual(output.read_text(), 'prior source')
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    rebuilt = self.module(compiler, output.read_text(), 'single-128')
                    self.assertEqual(self.facts(rebuilt)['functions'][0]['locals'], 128)

    def test_descriptor_and_unsupported_helper_controls(self):
        facts = self.facts(self.module(self.compilers[0], original_source()))
        for descriptor in (0, 1.0, 2):
            bad = copy.deepcopy(facts)
            ins = next(i for i in bad['functions'][1]['code'] if i['op'] == 'BORROW_LOCAL_SHARED')
            ins['arg'] = descriptor; ins['args'][0] = descriptor
            for language in ('c', 'nano'):
                with self.subTest(descriptor=descriptor, language=language), self.assertRaises(Refusal):
                    reconstruct_owned(bad, language)
        for opcode in ('ASSERT', 'JMP', 'CALL', 'SUB'):
            bad = copy.deepcopy(facts)
            ins = next(i for i in bad['functions'][1]['code'] if i['op'] == 'ADD')
            ins['op'] = opcode
            for language in ('c', 'nano'):
                with self.subTest(opcode=opcode, language=language), self.assertRaises(Refusal):
                    reconstruct_owned(bad, language)

    def test_valid_caller_reassignment_remains_separate(self):
        # The original producer/verifier/VM must accept the composition before
        # reconstruction deliberately refuses its unqualified planner overlap.
        text = original_source().replace(' # RETURN_POINT', '''
 let mut extra: Leaf = Leaf { value: 1, active: true }
 let Leaf { value, active } = extra
 let replacement: Leaf = Leaf { value: 2, active: false }
 set extra replacement
 let Leaf { value, active } = extra
 # RETURN_POINT''')
        for compiler in self.compilers:
            module = self.module(compiler, text, 'single-reassignment')
            for language in ('c', 'nano'):
                output = module.with_suffix('.'+language); output.write_text('prior source')
                result = self.command([ROOT/'bin/nvm2hl', '--language', language, module, '-o', output], None)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('refuse combining single-helper local ownership', result.stderr)
                self.assertEqual(output.read_text(), 'prior source')

    def test_original_verifier_formal_and_consumption_refusals(self):
        module = self.module(self.compilers[0], original_source())
        text = self.command([ROOT/'bin/nanoisa', 'dump', module]).stdout
        entry, helper = text.split('.function bump ', 1)
        cases = {
            'formal-overwrite': helper.replace('BORROW_LOCAL_SHARED 1 4', 'BORROW_LOCAL_SHARED 0 4', 1),
            'formal-move': helper.replace('OWN_MOVE_LOCAL 3', 'OWN_MOVE_LOCAL 0', 1),
            'local-leak': helper.replace('  OWN_UNPACK_LOCAL 4\n  POP\n  POP\n', ''),
        }
        for name, changed in cases.items():
            self.assertNotEqual(changed, helper)
            assembly = self.work/(name+'.nasm'); assembly.write_text(entry+'.function bump '+changed)
            malformed = assembly.with_suffix('.nvm')
            result = self.command([ROOT/'bin/nanoisa', 'asm', assembly, '-o', malformed], None)
            if result.returncode:
                self.assertRegex(result.stderr, r'(?i)verif|borrow|reference|owner|resource|live|region')
            else:
                self.assertNotEqual(self.command([ROOT/'bin/nano_vm', '--verify-only', malformed], None).returncode, 0)
                for language in ('c', 'nano'):
                    output = malformed.with_suffix('.'+language); output.write_text('prior source')
                    refused = self.command([ROOT/'bin/nvm2hl', '--language', language, malformed, '-o', output], None)
                    self.assertNotEqual(refused.returncode, 0)
                    self.assertEqual(output.read_text(), 'prior source')


if __name__ == '__main__':
    unittest.main()
