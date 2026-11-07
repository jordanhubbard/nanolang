"""I reconstruct each unchanged wrapper producer and independent shadow input."""
import json
import os
import shlex
import unittest
from tests import test_owned_branch_reconstruction as support
from tests import test_source_borrow_emission as source_support
from nanoisa_owned_reconstruction import OwnedAnalyze, OwnedEmit

ROOT = support.ROOT
BASE = ROOT/'docs/evidence/owned-wrapper-complete-inputs-20261020'


class OwnedWrapperReconstruction(unittest.TestCase):
    setUpClass = support.OwnedBranchReconstruction.__dict__['setUpClass']
    command = support.OwnedBranchReconstruction.command
    facts = support.OwnedBranchReconstruction.facts

    def native(self, source, expected):
        for opt in ('-O0', '-O2'):
            binary = source.with_name(source.name+opt+'.exe')
            self.command([*shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', 'cc')),
                          '-std=c11', opt, '-Wall', '-Wextra', '-Werror',
                          '-Wno-unused-function', '-Wno-unused-variable', '-Wno-unused-parameter',
                          '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                          source, ROOT/'bin/nano_aot_runtime.o', '-lm',
                          *support.native_link_flags(), '-o', binary], build=True)
            self.assertEqual(self.command([binary]).stdout, expected)

    def qualify(self, name, expected):
        text = getattr(source_support.SourceBorrowEmission(), name)()
        self.assertEqual(text, (BASE/(name+'.nano')).read_text())
        source = self.work/(name+'.nano'); source.write_text(text)
        modules = []
        for compiler in self.compilers:
            module = self.work/(name+'.'+compiler.name+'.nvm')
            self.command([compiler, source, '--emit-nvm', '-o', module])
            modules.append(module)
        shadow = self.work/(name+'.original-shadow.nvm')
        self.command([ROOT/'bin/nanoisa', 'asm', BASE/(name+'.shadow.nasm'), '-o', shadow])
        modules.append(shadow)
        for module in modules:
            with self.subTest(fixture=name, original=module.name):
                self.command([ROOT/'bin/nano_vm', '--verify-only', module])
                self.assertEqual(self.command([ROOT/'bin/nano_vm', module]).stdout, expected)
                original_c = module.with_suffix('.original.c')
                self.command([ROOT/'bin/nvm2c', module, '-o', original_c]); self.native(original_c, expected)
                facts = self.facts(module)
                emitter = OwnedEmit([OwnedAnalyze(facts, i).run() for i in range(len(facts['functions']))],
                                    'nano', facts['layouts'])
                costs = [emitter.nano_local_cost(fn) for fn in emitter.functions]
                reconstructed_c = module.with_suffix('.reconstructed.c')
                self.command([ROOT/'bin/nvm2hl', '--language', 'c', module, '-o', reconstructed_c])
                self.native(reconstructed_c, expected)
                nano = module.with_suffix('.reconstructed.nano')
                self.command([ROOT/'bin/nvm2hl', '--language', 'nano', module, '-o', nano])
                for compiler in self.compilers:
                    rebuilt = nano.with_suffix('.'+compiler.name+'.nvm')
                    self.command([compiler, nano, '--emit-nvm', '-o', rebuilt])
                    self.command([ROOT/'bin/nano_vm', '--verify-only', rebuilt])
                    self.assertEqual(self.command([ROOT/'bin/nano_vm', rebuilt]).stdout, expected)
                    regenerated = self.facts(rebuilt)
                    self.assertEqual(regenerated['layouts'], facts['layouts'])
                    # Existing scalar helpers may be inserted before reconstructed functions; estimates are conservative.
                    actual = [fn['locals'] for fn in regenerated['functions']]
                    self.assertTrue(all(count <= 128 for count in actual))
                    print('I retain wrapper counts '+json.dumps({'original': module.name,
                          'producer': compiler.name, 'original_locals': [f['locals'] for f in facts['functions']],
                          'estimated_original_locals': costs, 'actual_locals': actual, 'expected_stdout': expected}), flush=True)
                    native = rebuilt.with_suffix('.c')
                    self.command([ROOT/'bin/nvm2c', rebuilt, '-o', native]); self.native(native, expected)

    def test_temporary_arguments(self):
        self.qualify('temporary_owner_fixture', 'ABCABCABC')

    def test_transitive_wrappers(self):
        self.qualify('transitive_wrapper_fixture', 'ABAB')

    def test_nested_owner_results(self):
        self.qualify('nested_owner_result_source', 'ABAB')
