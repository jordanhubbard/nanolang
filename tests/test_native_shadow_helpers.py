"""I bind explicit root-shadow providers without widening my C product ABI."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

from tests import test_native_shadows as native_shadows

ROOT = native_shadows.ROOT


class NativeShadowHelpers(unittest.TestCase):
    invoke = native_shadows.NativeShadows.invoke
    source = ('extern fn foreign_value() -> int\nfn f()->int{return 0}\n'
              'shadow f {unsafe {assert (== (foreign_value) 42)}}')

    def provider(self, directory, body=None, name='provider.c'):
        path = directory / name
        path.write_text(body or '#include <stdint.h>\nint64_t foreign_value(void){return 42;}\n')
        return shlex.quote(str(path))

    def compiler_control(self, directory, status):
        marker = directory / 'compiler.called'
        compiler = directory / 'compiler-control'
        compiler.write_text('#!/bin/sh\nprintf called >> ' + shlex.quote(str(marker)) +
                            '\nexit ' + str(status) + '\n')
        compiler.chmod(0o755)
        return str(compiler), marker

    def assert_clean(self, directory):
        self.assertFalse(list(directory.glob('nano_native_helpers_*')))

    def test_native_side_effect_and_independent_product(self):
        with tempfile.TemporaryDirectory(prefix='nano-helper-binding-') as tmp:
            directory = Path(tmp)
            marker = directory / 'native.called'
            flags = self.provider(directory, '''#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
int64_t foreign_value(void){
    FILE *f=fopen(getenv("NANO_NATIVE_HELPER_MARKER"),"ab");
    if(!f)abort();if(fputs("N",f)<0||fclose(f))abort();return 42;
}
''')
            result, output = self.invoke(directory, self.source, extra_env={
                'NANO_CFLAGS': flags, 'NANO_NATIVE_HELPER_MARKER': str(marker)})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(marker.read_bytes(), b'N')
            self.assert_clean(directory)
            self.assertNotIn(b'/nano_native_helpers_', output.read_bytes())
            executed = subprocess.run([str(output)], capture_output=True, timeout=5)
            self.assertEqual(executed.returncode, 0, executed.stderr)
            self.assertEqual(marker.read_bytes(), b'N')

    def test_relative_quoted_flags_and_response_input(self):
        for response in (False, True):
            with self.subTest(response=response), tempfile.TemporaryDirectory(prefix='nano-helper-flags-') as tmp:
                directory = Path(tmp)
                header = directory / 'provider ü include.h'
                header.write_text('#define HEADER_VALUE 1\n')
                provider = directory / 'provider λ with spaces.c'
                provider.write_text('#include <stdint.h>\nint64_t foreign_value(void){return HELPER_VALUE+HEADER_VALUE;}\n')
                flags = ('-DHELPER_VALUE=41 -include ' + shlex.quote(os.path.relpath(header, ROOT)) +
                         ' ' + shlex.quote(os.path.relpath(provider, ROOT)))
                if response:
                    options = directory / 'provider flags.rsp'
                    options.write_text(flags + '\n')
                    flags = '@' + shlex.quote(os.path.relpath(options, ROOT))
                result, output = self.invoke(directory, self.source, extra_env={'NANO_CFLAGS': flags})
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(subprocess.run([str(output)], timeout=5).returncode, 0)
                self.assert_clean(directory)

    def test_only_selected_helpers_are_bound(self):
        source = ('extern fn first_value()->int\nextern fn second_value()->int\n'
                  'extern fn missing_unselected()->void\nfn f()->int{return 0}\n'
                  'shadow f {unsafe {assert (== (+ (first_value) (second_value)) 42) '
                  'assert (== (first_value) 20)}}')
        with tempfile.TemporaryDirectory(prefix='nano-helper-selection-') as tmp:
            directory = Path(tmp)
            flags = self.provider(directory, '#include <stdint.h>\nint64_t first_value(void){return 20;}\nint64_t second_value(void){return 22;}\n')
            result, _ = self.invoke(directory, source, extra_env={'NANO_CFLAGS': flags})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assert_clean(directory)

    def test_primitive_parameters_and_results(self):
        source = ('extern fn helper_bool(value:bool)->bool\n'
                  'extern fn helper_byte(value:u8)->u8\n'
                  'extern fn helper_float(value:float)->float\n'
                  'fn f()->int{return 0}\nshadow f {unsafe {'
                  'assert (not (helper_bool true)) '
                  'let maximum:u8=255 assert (== (helper_byte maximum) 0) '
                  'assert (== (helper_float 8.0) 4.0)}}')
        with tempfile.TemporaryDirectory(prefix='nano-helper-primitives-') as tmp:
            directory = Path(tmp)
            flags = self.provider(directory, '#include <stdbool.h>\n#include <stdint.h>\n'
                                  'bool helper_bool(bool value){return !value;}\n'
                                  'uint8_t helper_byte(uint8_t value){return (uint8_t)(value+1); }\n'
                                  'double helper_float(double value){return value/2;}\n')
            result, _ = self.invoke(directory, source, extra_env={'NANO_CFLAGS': flags})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assert_clean(directory)

    def test_missing_symbol_preserves_output(self):
        with tempfile.TemporaryDirectory(prefix='nano-helper-symbol-') as tmp:
            directory = Path(tmp)
            flags = self.provider(directory, 'void unrelated_provider(void){}\n')
            (directory / 'program').write_bytes(b'prior')
            result, output = self.invoke(directory, self.source, extra_env={'NANO_CFLAGS': flags})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'foreign_value', result.stdout + result.stderr)
            self.assertEqual(output.read_bytes(), b'prior')
            self.assert_clean(directory)

    def test_no_native_context_still_refuses(self):
        with tempfile.TemporaryDirectory(prefix='nano-helper-absent-') as tmp:
            directory = Path(tmp)
            (directory / 'program').write_bytes(b'prior')
            result, output = self.invoke(directory, self.source,
                                         extra_env={'NANO_CFLAGS': '', 'NANO_LDFLAGS': ''})
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_bytes(), b'prior')
            self.assert_clean(directory)

    def test_known_abi_cannot_be_overridden(self):
        source = ('extern fn get_argv(index:int)->int\nfn f()->int{return 0}\n'
                  'shadow f {unsafe {(get_argv 0)}}')
        with tempfile.TemporaryDirectory(prefix='nano-helper-known-') as tmp:
            directory = Path(tmp)
            flags = self.provider(directory)
            compiler, marker = self.compiler_control(directory, 0)
            (directory / 'program').write_bytes(b'prior')
            result, output = self.invoke(directory, source, extra_env={'NANO_CFLAGS': flags, 'NANO_CC': compiler})
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(marker.exists())
            self.assertEqual(output.read_bytes(), b'prior')
            self.assert_clean(directory)

    def test_imported_declaration_keeps_its_provider(self):
        with tempfile.TemporaryDirectory(prefix='nano-helper-import-') as tmp:
            directory = Path(tmp)
            dependency = directory / 'dependency.nano'
            dependency.write_text('pub extern fn dependency_action()->void\n')
            source = (f'module "{dependency}" as dependency\nfn f()->int{{return 0}}\n'
                      'shadow f {unsafe {(dependency.dependency_action)}}')
            flags = self.provider(directory, 'void dependency_action(void){}\n')
            compiler, marker = self.compiler_control(directory, 0)
            (directory / 'program').write_bytes(b'prior')
            result, output = self.invoke(directory, source, extra_env={'NANO_CFLAGS': flags, 'NANO_CC': compiler})
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(marker.exists())
            self.assertEqual(output.read_bytes(), b'prior')
            self.assert_clean(directory)

    def test_partial_lowering_cleans_storage(self):
        source = ('extern fn first_action()->void\nextern fn unsupported_text()->string\n'
                  'fn f()->int{return 0}\nshadow f {unsafe {(first_action) (unsupported_text)}}')
        with tempfile.TemporaryDirectory(prefix='nano-helper-partial-') as tmp:
            directory = Path(tmp)
            flags = self.provider(directory, 'void first_action(void){}\n')
            compiler, marker = self.compiler_control(directory, 0)
            (directory / 'program').write_bytes(b'prior')
            result, output = self.invoke(directory, source, extra_env={'NANO_CFLAGS': flags, 'NANO_CC': compiler})
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(marker.exists())
            self.assertEqual(output.read_bytes(), b'prior')
            self.assert_clean(directory)

    def test_failed_or_false_success_compiler_preserves_output(self):
        for status in (0, 19):
            with self.subTest(status=status), tempfile.TemporaryDirectory(prefix='nano-helper-compiler-') as tmp:
                directory = Path(tmp)
                flags = self.provider(directory)
                compiler, marker = self.compiler_control(directory, status)
                (directory / 'program').write_bytes(b'prior')
                result, output = self.invoke(directory, self.source, extra_env={'NANO_CFLAGS': flags, 'NANO_CC': compiler})
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(marker.read_bytes(), b'called')
                self.assertEqual(output.read_bytes(), b'prior')
                self.assert_clean(directory)

    def test_c_only_never_builds_provider(self):
        with tempfile.TemporaryDirectory(prefix='nano-helper-source-') as tmp:
            directory = Path(tmp)
            flags = self.provider(directory)
            compiler, marker = self.compiler_control(directory, 19)
            result, output = self.invoke(directory, self.source, '--target', 'c',
                                         extra_env={'NANO_CFLAGS': flags, 'NANO_CC': compiler})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(b'without executing shadows', result.stdout)
            self.assertFalse(marker.exists())
            self.assertNotIn(b'/nano_native_helpers_', output.read_bytes())
            self.assert_clean(directory)

    def test_program_extern_does_not_gain_shadow_admission(self):
        source = 'extern fn foreign_value()->int\nfn main()->int{unsafe {return (foreign_value)}}\nshadow main {assert true}'
        with tempfile.TemporaryDirectory(prefix='nano-helper-product-') as tmp:
            directory = Path(tmp)
            flags = self.provider(directory)
            compiler, marker = self.compiler_control(directory, 0)
            (directory / 'program').write_bytes(b'prior')
            result, output = self.invoke(directory, source, extra_env={'NANO_CFLAGS': flags, 'NANO_CC': compiler})
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(marker.exists())
            self.assertEqual(output.read_bytes(), b'prior')
            self.assert_clean(directory)


if __name__ == '__main__':
    unittest.main()
