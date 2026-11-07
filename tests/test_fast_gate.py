"""I test compile-plan safety and the two GCC regressions that escaped my gate."""
from pathlib import Path
import platform
import json
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import fast_gate as gate


class FastGatePlanning(unittest.TestCase):
    def test_object_valued_macro_preserves_the_compiler_refusal(self):
        compiler = shutil.which('cc')
        if not compiler:
            self.skipTest('I require a C compiler for the macro-refusal control.')
        with tempfile.TemporaryDirectory(prefix='nano-fast-macro-refusal-') as directory:
            source = Path(directory) / 'refusal.c'
            source.write_text('#ifdef OBJECT\n#error I refuse the requested OBJECT macro\n#endif\n'
                              'int native_profile_control;\n')
            plain = gate.run([compiler, '-S', str(source), '-o', '/dev/null'])
            self.assertEqual(plain[1], 0, plain[2])
            flags, sources = gate.compile_profile(['-DOBJECT=unused.o', str(source)])
            refused = gate.run([compiler, *flags, '-S', *sources, '-o', '/dev/null'])
            self.assertNotEqual(refused[1], 0)
            self.assertIn('I refuse the requested OBJECT macro', refused[2])

    def test_c_valued_options_and_forced_includes_keep_the_actual_source(self):
        compiler = shutil.which('cc')
        if not compiler:
            self.skipTest('I require a C compiler for the option control.')
        with tempfile.TemporaryDirectory(prefix='nano-fast-option-control-') as directory:
            forced = Path(directory) / 'forced.c'
            forced.write_text('#define NANO_VALID_FORCED_INCLUDE 1\n')
            definitions = ['-DINPUT=src/utf8.c', '-DOBJECT=unused.o', '-DARCHIVE=unused.a',
                           '-DSHARED=unused.so', '-DDYLIB=unused.dylib']
            words = ['-std=c99', '-Wall', '-Wextra', '-Werror', '-Isrc',
                     *definitions, '-include', str(forced), 'src/utf8.c']
            command = [compiler, *words, '-O2', '-S', '-o', '/dev/null']
            accepted = gate.run(command)
            self.assertEqual(accepted[1], 0, accepted[2])
            flags, sources = gate.compile_profile(words)
            for definition in definitions:
                self.assertIn(definition, flags)
            self.assertEqual(sources, ['src/utf8.c'])
            self.assertEqual(gate.source_of(command), 'src/utf8.c')

    def test_product_report_retains_uncovered_test_inventory(self):
        compiler = shutil.which('cc')
        if not compiler:
            self.skipTest('I require a C compiler for the report control.')
        with tempfile.TemporaryDirectory(prefix='nano-fast-report-control-') as directory:
            report = Path(directory) / 'report.json'
            command = [compiler, '-std=c99', '-Wall', '-Wextra', '-Werror', '-Isrc',
                       '-S', 'src/utf8.c', '-o', '/dev/null']
            with patch.object(gate, 'detect_compilers', return_value={'cc': compiler}), \
                    patch.object(gate, 'compile_commands', return_value=[command]), \
                    patch('sys.argv', ['fast_gate', '--jobs', '1', '--report', str(report)]):
                self.assertEqual(gate.main(), 0)
            data = json.loads(report.read_text())['compilers'][0]
            self.assertFalse(data['test_profiles_enabled'])
            self.assertFalse(data['python_profile_audit_complete'])
            self.assertIn('tests/test_module_builder_paths.c', data['uncovered_test_sources'])
            self.assertEqual(data['results'][0]['returncode'], 0)

    def test_preview_does_not_execute_recursive_or_plus_recipes(self):
        with tempfile.TemporaryDirectory(prefix='nano-fast-plan-control-') as directory:
            root = Path(directory)
            marker = root / 'must-not-exist'
            (root / 'Makefile.gnu').write_text(
                '.PHONY: probe\nprobe:\n'
                f'\t+touch {marker}\n'
                f'\t@++touch {marker}\n'
                f'\t$(MAKE) -f /dev/null; touch {marker}\n'
                f'\t${{MAKE}} -f /dev/null; touch {marker}\n'
                '\t$(CC) -DPRIVATE_MODE=1 -c source.c -o output.o\n')
            with patch.object(gate, 'ROOT', root):
                preview = gate.make_preview(['probe'])
            self.assertFalse(marker.exists())
            self.assertIn('-DPRIVATE_MODE=1', preview)

    def test_make_failure_is_not_an_empty_success(self):
        with tempfile.TemporaryDirectory(prefix='nano-fast-plan-control-') as directory:
            root = Path(directory)
            (root / 'Makefile.gnu').write_text('$(error I refuse this broken plan)\n')
            with patch.object(gate, 'ROOT', root), self.assertRaisesRegex(ValueError, 'broken plan'):
                gate.make_preview(['probe'])

    def test_real_shell_quoting_keeps_macro_operands_and_each_translation_unit(self):
        line = (gate.MARKER + ' -DNAME="two words" -I"path with space" '
                '-c src/utf8.c -o output.o && ' + gate.MARKER + ' -c src/lexer.c -o lexer.o')
        recipes = list(gate.recipe_commands(line))
        self.assertEqual(len(recipes), 2)
        flags, sources = gate.compile_profile(recipes[0])
        self.assertIn('-DNAME=two words', flags)
        self.assertIn('-Ipath with space', flags)
        self.assertEqual(sources, ['src/utf8.c'])
        with self.assertRaisesRegex(ValueError, 'shell operand'):
            gate.compile_profile(['-DVALUE=$unresolved', 'src/utf8.c'])


class FastGateGccControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gcc = gate.detect_compilers().get('gcc')
        if not cls.gcc:
            raise unittest.SkipTest('I require genuine GCC for these two diagnostic controls.')

    def test_actual_private_union_profile_rejects_old_source(self):
        self.check_old_source('src/nanovm/vm.c')

    def test_actual_glibc_disassembler_profiles_reject_old_source(self):
        if platform.libc_ver()[0] != 'glibc':
            self.skipTest('I require the original glibc sprintf wrapper; Darwin does not reproduce this diagnostic.')
        self.check_old_source('tests/nanoisa/test_disasm_roundtrip.c')

    def check_old_source(self, selected):
        with tempfile.TemporaryDirectory(prefix='nano-fast-old-source-') as directory:
            overlay = Path(directory)
            # I retain relative include resolution without editing the working tree.
            (overlay / 'src/nanovm').mkdir(parents=True)
            for source in (gate.ROOT / 'src').iterdir():
                if source.name != 'nanovm':
                    (overlay / 'src' / source.name).symlink_to(source, target_is_directory=source.is_dir())
            for source in (gate.ROOT / 'src/nanovm').iterdir():
                if source.name not in {'vm.c', 'union_field_vm_prepare.inc'}:
                    (overlay / 'src/nanovm' / source.name).symlink_to(source, target_is_directory=source.is_dir())
            vm = gate.ROOT / 'src/nanovm/vm.c'
            shutil.copyfile(vm, overlay / 'src/nanovm/vm.c')
            current = (gate.ROOT / 'src/nanovm/union_field_vm_prepare.inc').read_text()
            self.assertEqual(current.count('%.120s'), 1)
            (overlay / 'src/nanovm/union_field_vm_prepare.inc').write_text(current.replace('%.120s', '%.160s'))
            source = gate.ROOT / 'tests/nanoisa/test_disasm_roundtrip.c'
            text = source.read_text()
            start = text.index('    size_t used = 0;', text.index('static void large_symbol_and_branch_tables'))
            end = text.index('    AsmResult result;', start)
            old = '''    size_t used = 0;
    for (uint32_t i = 0; i < count; i++)
        used += (size_t)sprintf(text + used, ".string s%u \\"same\\"\\n", i);
    used += (size_t)sprintf(text + used, ".function main 0 0 0 int 1\\n");
    for (uint32_t i = 0; i < count; i++)
        used += (size_t)sprintf(text + used, " JMP label%u\\nlabel%u:\\n", i, i);
    sprintf(text + used, " PUSH_I64 37\\n RET\\n.end\\n");
'''
            # The old function uses my original relative NanoISA include paths.
            (overlay / 'tests/nanoisa').mkdir(parents=True)
            old_disasm = overlay / 'tests/nanoisa/test_disasm_roundtrip.c'
            old_disasm.write_text(text[:start] + old + text[end:])
            with tempfile.TemporaryDirectory(prefix='nano-fast-control-headers-') as headers:
                gate.stage_public_headers(headers)
                profiles, _ = gate.test_profiles(self.gcc, 'gcc', headers)
                for original, replacement, define, diagnostic in (
                    ('src/nanovm/vm.c', overlay / 'src/nanovm/vm.c',
                     '-DNANO_UNION_FIELD_PRIVATE_PREPARATION', 'format-truncation'),
                    ('tests/nanoisa/test_disasm_roundtrip.c', old_disasm, None, 'format-overflow')):
                    if original != selected:
                        continue
                    candidates = [command for command in profiles if gate.source_of(command) == original
                                  and (define is None or define in command)]
                    # I check both ordinary and instrumented glibc formatting. My union
                    # diagnostic is independent of instrumentation and needs one replay.
                    candidates = candidates[:2] if define is None else candidates[:1]
                    refusals = []
                    for command in candidates:
                        current = gate.run(command)
                        self.assertEqual(current[1], 0, current[2])
                        changed = [str(replacement) if word == original else word for word in command]
                        refused = gate.run(changed)
                        if refused[1] and diagnostic in refused[2]:
                            refusals.append(refused)
                    self.assertTrue(refusals, original + ': I did not reproduce the original GCC diagnostic.')


if __name__ == '__main__':
    unittest.main()
