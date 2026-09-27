"""I execute explicit scalar artifact contracts through VM and native libffi."""
import os
import sys
from . import test_artifact_string_release as release_tests

ROOT = release_tests.ROOT


class DeclaredScalarArtifacts(release_tests.ArtifactStringRelease):
    def imports(self, entries):
        return super().imports(entries).replace(' artifact\n', ' declared_scalar_artifact\n')

    def native(self, module, transform=None):
        source, binary = self.work/'generated.c', self.work/'native'
        self.command([ROOT/'bin/nvm2c', module, '-o', source])
        if transform:
            source.write_text(transform(source.read_text()))
        self.command(['cc', '-std=c11', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                      '-fsanitize=address,undefined', '-fno-sanitize-recover=all', source,
                      '-o', binary, '-lm', '-lffi', *(['-ldl'] if sys.platform.startswith('linux') else [])])
        return binary

    def test_vm_heterogeneous_max_arity_provider_cleanup(self):
        super().test_vm_heterogeneous_max_arity_provider_cleanup()
        marker = self.work/'mixed-release-count'
        self.command([self.native(self.work/'input.nvm')],
                     env={**os.environ, 'ASAN_OPTIONS':'detect_leaks=1',
                          'NANO_ARTIFACT_RELEASE_MARKER':str(marker)})
        self.assertEqual(marker.read_text(), 'released\nreleased\n')

    def test_scalar_results_and_void_side_effect(self):
        lib = self.library('scalar_results', r'''
#include <stdint.h>
static int64_t count;
void tick(void) { count++; }
int64_t integer(void) { return -40 + count; }
double floating(double a, int64_t b) { return a + b; }
uint8_t byte(uint8_t a) { return a; }
uint8_t boolean(uint8_t a) { return !a; }
const char *alias(const char *text) { return text; }
int64_t enum_value(void) { return -4294967297LL; }
int64_t enum_echo(int64_t value) { return value; }
''')
        entries = [(lib,'tick','void'), (lib,'integer','int'),
                   (lib,'floating','float float int'), (lib,'byte','u8 u8'),
                   (lib,'boolean','bool bool'), (lib,'alias','string string'),
                   (lib,'enum_value','enum'), (lib,'enum_echo','int enum')]
        body = ('CALL_EXTERN 0\nCALL_EXTERN 1\nPUSH_I64 -39\nI64_EQ\nASSERT\n'
                'PUSH_F64 2.5\nPUSH_I64 -3\nCALL_EXTERN 2\nPUSH_F64 -0.5\nF64_EQ\nASSERT\n'
                'PUSH_U8 250\nCALL_EXTERN 3\nPUSH_U8 250\nEQ\nASSERT\n'
                'PUSH_BOOL 0\nCALL_EXTERN 4\nASSERT\n'
                'CALL_EXTERN 6\nDUP\nTYPE_CHECK 9\nASSERT\nCALL_EXTERN 7\nPUSH_I64 -4294967297\nI64_EQ\nASSERT\n'
                'PUSH_STR text\nCALL_EXTERN 5\nPUSH_STR text\nEQ\nASSERT\n')
        self.paired(self.imports(entries)+'.string text "borrowed"\n.entry main\n.function main 0 0 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n')

    def test_legacy_artifact_does_not_gain_arbitrary_native_admission(self):
        lib = self.library('legacy', '#include <stdint.h>\nint64_t arbitrary(void) { return 7; }\n')
        text = self.imports([(lib,'arbitrary','int')]).replace(' declared_scalar_artifact\n',' artifact\n')
        module = self.module(text+'.entry main\n.function main 0 0 0 int 1\nCALL_EXTERN 0\nRET\n.end\n')
        output = self.work/'preserved.c'
        output.write_text('prior output')
        self.command([ROOT/'bin/nvm2c',module,'-o',output],success=False)
        self.assertEqual(output.read_text(),'prior output')

    def test_dump_round_trip_retains_explicit_contract(self):
        lib = self.library('round_trip', '#include <stdint.h>\nint64_t answer(void) { return 23; }\n')
        text = self.imports([(lib,'answer','int')])
        module = self.module(text+'.entry main\n.function main 0 0 0 int 1\nCALL_EXTERN 0\nPUSH_I64 23\nI64_EQ\nASSERT\nPUSH_I64 0\nRET\n.end\n')
        dumped = self.command([ROOT/'bin/nanoisa','dump',module]).stdout
        self.assertIn('.import_kind 0 declared_scalar_artifact',dumped)
        self.paired(dumped)

    def test_missing_symbol_and_borrowed_null_fail_in_both_consumers(self):
        lib = self.library('failures', '#include <stddef.h>\nconst char *null_text(void) { return NULL; }\n')
        for symbol in ('missing_scalar_fixture_symbol','null_text'):
            with self.subTest(symbol=symbol):
                text = self.imports([(lib,symbol,'string')])
                module = self.module(text+'.entry main\n.function main 0 0 0 int 1\nCALL_EXTERN 0\nPOP\nPUSH_I64 0\nRET\n.end\n')
                self.command([ROOT/'bin/nano_vm',module],success=False)
                result = self.command([self.native(module)],success=False,
                    env={**os.environ,'ASAN_OPTIONS':'detect_leaks=1'})
                self.assertNotIn('ERROR: AddressSanitizer',result.stderr)
                self.assertNotIn('runtime error:',result.stderr)
