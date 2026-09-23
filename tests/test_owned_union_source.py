"""I retain the existing generic ownership corpus through raw VM/native lowering."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from tests import test_generic_selected_ownership as selected
from tests import test_selected_variant_ownership as nongeneric
from tests import test_affine_scalar_union_runtime as runtime

ROOT = Path(__file__).resolve().parents[1]


class OwnedUnionSource(unittest.TestCase):
    compiler = runtime.AffineScalarUnionRuntime.compiler

    def test_c_frontend_retains_both_selected_arm_names(self):
        with tempfile.TemporaryDirectory(prefix='nano-owned-union-names-') as directory:
            source, module = Path(directory) / 'source.nano', Path(directory) / 'source.nvm'
            source.write_text(selected.PREFIX + selected.CONSUME + selected.MAIN)
            result = subprocess.run([ROOT / 'bin/nano_virt', source, '--emit-nvm', '-o', module],
                                    cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([ROOT / 'obj/test_local_bindings', module],
                                    cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            bindings = [row.split() for row in result.stdout.splitlines()
                        if row.startswith('consume ') and row.endswith(' payload')]
            self.assertEqual(len(bindings), 2, result.stdout)
            self.assertEqual(bindings[0][1], bindings[1][1])
            self.assertLessEqual(int(bindings[0][3]), int(bindings[1][2]))

    def test_existing_source_corpus_and_previous_output(self):
        cases = []
        suite = selected.GenericSelectedOwnership()
        for name in unittest.defaultTestLoader.getTestCaseNames(type(suite)):
            # This case invokes the complete compiler diagnostic matrix directly;
            # its original suite retains that check after fresh bootstrap.
            if name == 'test_guarded_generic_match_remains_rejected':
                continue
            suite.check = lambda source, accepted: cases.append((name, source, accepted))
            getattr(suite, name)()
        self.assertEqual(sum(accepted for _, _, accepted in cases), 10)
        self.assertEqual(len(cases), 19)
        plain = nongeneric.SelectedVariantOwnership()
        for name in ('test_two_resources_and_ordinary_sibling_arms',
                     'test_nested_resource_record', 'test_compatible_outer_join'):
            plain.program = lambda source, accepted, diagnostics=None: cases.append((name, source, accepted))
            getattr(plain, name)()
        self.assertEqual(sum(accepted for _, _, accepted in cases), 13)
        with tempfile.TemporaryDirectory(prefix='nano-owned-union-source-') as directory:
            work = Path(directory)
            for producer in ('nanoisa_emit', 'nano_virt'):
                for name, source, accepted in cases:
                    with self.subTest(producer=producer, case=name):
                        program = work / 'source.nano'
                        module = work / 'source.nvm'
                        generated = work / 'source.c'
                        binary = work / 'source.native'
                        program.write_text(source)
                        module.write_bytes(b'prior module')
                        emitted = subprocess.run([ROOT / 'bin' / producer, program,
                                                  '--emit-nvm', '-o', module], cwd=ROOT,
                                                 capture_output=True, text=True, timeout=120)
                        if not accepted:
                            self.assertNotEqual(emitted.returncode, 0, emitted.stdout + emitted.stderr)
                            self.assertEqual(module.read_bytes(), b'prior module')
                            continue
                        self.assertEqual(emitted.returncode, 0, emitted.stdout + emitted.stderr)
                        commands = ([ROOT / 'bin/nano_vm', '--verify-only', module],
                                    [ROOT / 'bin/nano_vm', module],
                                    [ROOT / 'bin/nvm2c', module, '-o', generated],
                                    [self.compiler(), '-std=c11', '-Wall', '-Wextra', '-Werror',
                                     '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                                     generated, '-o', binary], [binary])
                        for command in commands:
                            result = subprocess.run(command, cwd=ROOT, capture_output=True,
                                                    text=True, timeout=120,
                                                    env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1',
                                                         'UBSAN_OPTIONS': 'halt_on_error=1'})
                            self.assertEqual(result.returncode, 0, str(command) + '\n' + result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
