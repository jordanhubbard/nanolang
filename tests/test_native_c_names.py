"""I keep my C parameter spellings distinct from keywords and generated names."""
from pathlib import Path
import os
import shlex
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeCNames(unittest.TestCase):
    def checked(self, argv):
        from tests.native_process import run_captured
        result = run_captured(list(map(str, argv)), cwd=ROOT, timeout=30)
        self.assertEqual(result.returncode, 0,
                         result.stdout.decode(errors='replace') +
                         result.stderr.decode(errors='replace'))
        return result.stdout.decode()

    def test_keywords_and_prefixes_compile_as_distinct_parameters(self):
        # I exercise the shared encoder, then compile and execute its actual
        # output. Ordinary spellings and near-prefix names remain unchanged.
        names = ['signed', 'unsigned', 'register', 'restrict', 'volatile', 'auto',
                 'short', 'long', 'static', 'typedef', '_Atomic', '_Generic',
                 '_Thread_local', '__asm__', '__attribute__', '__typeof__',
                 '__auto_type', '__restrict__', '__volatile__', 'constexpr',
                 'typeof_unqual', 'nullptr', '_BitInt',
                 '__nl_keyword_p_7369676e6564', '__nl_keyword_l_0',
                 '__nl_keyword_', 'signed_values', 'Signed', '__nl_keyword']
        unchanged = {'signed_values', 'Signed', '__nl_keyword'}
        literals = ',\n'.join('"' + name + '"' for name in names)
        probe = r'''
#include "native_local_names.h"
#include <assert.h>
#include <stdio.h>
int main(void) {
    const char *names[] = { NAMES };
    assert(!native_c_name_needs_alias(NULL));
    assert(native_c_parameter_name(NULL) == NULL);
    assert(!native_c_name_needs_alias("signed_values"));
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        char *out = native_c_parameter_name(names[i]);
        assert(out);
        puts(out);
        free(out);
    }
    return 0;
}
'''.replace('NAMES', literals)
        compiler = shlex.split(os.environ.get('CC', 'cc'))
        flags = ['-std=c99', '-Wall', '-Wextra', '-Werror', '-pedantic']
        with tempfile.TemporaryDirectory(prefix='nano-c-names-') as temporary:
            directory = Path(temporary)
            source, program = directory/'probe.c', directory/'probe'
            source.write_text(probe)
            self.checked([*compiler, *flags, '-I', ROOT/'src', source,
                          '-o', program])
            encoded = self.checked([program]).splitlines()
            self.assertEqual(len(encoded), len(names))
            self.assertEqual(len(set(encoded)), len(names))
            self.assertEqual(encoded[0], '__nl_keyword_p_7369676e6564')
            for original, actual in zip(names, encoded):
                if original in unchanged:
                    self.assertEqual(actual, original)
                else:
                    self.assertNotEqual(actual, original)
                    self.assertRegex(actual, r'^__nl_keyword_p_[0-9a-f]+$')
            self.assertNotEqual(encoded[0], encoded[names.index(encoded[0])])
            functions = '\n'.join(
                f'static int case_{i}(int {name}) '
                f'{{ return {name} + {i}; }}'
                for i, name in enumerate(encoded))
            calls = '\n'.join(f'assert(case_{i}(41) == {41 + i});'
                              for i in range(len(encoded)))
            source.write_text('#include <assert.h>\n' + functions +
                              '\nint main(void) {\n' + calls +
                              '\nreturn 0;\n}\n')
            self.checked([*compiler, *flags, source, '-o', program])
            self.checked([program])


if __name__ == '__main__':
    unittest.main()
