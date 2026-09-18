"""I preserve equivalent present fields without widening static shape admission."""
from pathlib import Path
import os
import tempfile
import unittest
from tests import test_native_optional_array_reads as optional

ROOT = Path(__file__).resolve().parents[1]

class RecordArrayScalarTags(unittest.TestCase):
    checked = optional.OptionalArrayReads.checked
    paired = optional.OptionalArrayReads.paired

    def test_ordinary_replacement_and_alias_observation(self):
        for tag, first, second in [(1, 'PUSH_I64 17', 'PUSH_I64 42'),
                                   (4, 'PUSH_BOOL 0', 'PUSH_BOOL 1'),
                                   (5, 'PUSH_STR first', 'PUSH_STR second')]:
            for boxed in (False, True):
                with self.subTest(tag=tag, boxed=boxed):
                    suffix = f'\nARR_LITERAL {tag} 1\nPUSH_I64 0\nARR_GET' if boxed else ''
                    body = first + suffix + '\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\nSTORE_LOCAL 0\n'
                    body += 'LOAD_LOCAL 0\nSTORE_LOCAL 1\nLOAD_LOCAL 0\nPUSH_I64 0\n'
                    body += second + suffix + '\nAGG_PACK 0 0 0 1\nARR_SET\nPOP\n'
                    body += 'LOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\nAGG_GET 0\n'
                    body += f'DUP\nTYPE_CHECK {tag}\nASSERT\n{second}\nEQ\nASSERT\n'
                    self.paired('.types 1 0 0\n.string first "before"\n.string second "after"\n.entry main\n.function main 0 2 0 int 1\n' + body + 'PUSH_I64 0\nRET\n.end\n')

    def test_generated_present_field_compatibility_and_refusal(self):
        # I exercise the generated representation boundary directly. Ordinary
        # static shape admission remains independently checked by the compiler.
        text = '.entry main\n.function main 0 0 0 int 1\nPUSH_I64 1\nARR_LITERAL 1 1\nPUSH_I64 0\nPUSH_I64 2\nARR_SET\nPOP\nPUSH_I64 0\nRET\n.end\n'
        with tempfile.TemporaryDirectory(prefix='nano-record-field-tags-') as directory:
            p = Path(directory)
            (p/'input.nasm').write_text(text)
            self.checked([ROOT/'bin/nanoisa', 'asm', p/'input.nasm', '-o', p/'input.nvm'])
            self.checked([ROOT/'bin/nano_vm', p/'input.nvm'])
            self.checked([ROOT/'bin/nvm2c', p/'input.nvm', '-o', p/'generated.c'])
            (p/'check.c').write_text(r'''
#define main nano_generated_main
#include "generated.c"
#undef main
#include <assert.h>
int main(void) {
    assert(nano_generated_main() == 0);
    const unsigned plain[] = {0, 9, 1};
    const unsigned payload[] = {1, 4, 5};
    for (unsigned i = 0; i < 3; ++i) {
        nrec_t a = {0}, b = {0};
        a.n = b.n = 1;
        a.k[0] = plain[i]; b.k[0] = 8; b.vk[0] = payload[i];
        a.s[0] = "before"; b.s[0] = "after";
        assert(nrec_field_storage_matches(&a, &b, 0));
        assert(nrec_field_storage_matches(&b, &a, 0));
        b.vk[0] = 0;
        assert(!nrec_field_storage_matches(&a, &b, 0));
        assert(!nrec_field_storage_matches(&b, &a, 0));
        b.vk[0] = payload[(i + 1) % 3];
        assert(!nrec_field_storage_matches(&a, &b, 0));
        assert(!nrec_field_storage_matches(&b, &a, 0));
        b.vk[0] = payload[i];
        if (payload[i] == 5) {
            b.s[0] = NULL;
            assert(!nrec_field_storage_matches(&a, &b, 0));
            assert(!nrec_field_storage_matches(&b, &a, 0));
        }
        b.k[0] = 3;
        assert(!nrec_field_storage_matches(&a, &b, 0));
    }
    return 0;
}
''')
            self.checked([os.environ.get('CC', 'cc'), '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror',
                          '-fsanitize=address,undefined', '-fno-sanitize-recover=all', p/'check.c', '-o', p/'check'])
            self.checked([p/'check'])

if __name__ == '__main__':
    unittest.main()
