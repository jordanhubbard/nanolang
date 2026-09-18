"""I check tagged equality at strict O2 without passing missing text to strcmp."""
import os
import shlex
import sys
import tempfile
from pathlib import Path
import unittest
from tests import test_native_enum_scalars as enums
ROOT=enums.ROOT


class StringEqualityGuards(unittest.TestCase):
    run_command=enums.EnumScalars.run_command
    checked=enums.EnumScalars.checked
    assemble=enums.EnumScalars.assemble
    paired=enums.EnumScalars.paired

    def native(self,work,module,sanitize=False):
        source=work/'input.c';binary=work/'program'
        self.checked([ROOT/'bin/nvm2c',module,'-o',source])
        compiler=shlex.split(os.environ.get('NANO_NATIVE_TEST_CC','cc'))
        flags=['-std=c11','-O2','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all']
        self.checked([*compiler,*flags,source,'-o',binary])
        return binary

    test_optimized_enum_numeric_matrix=enums.EnumScalars.test_arithmetic_matrix_and_tags

    def test_ordinary_string_scalar_and_alias_comparisons(self):
        helpers='.string a "same"\n.string b "same"\n.string different "other"\n.string empty ""\n'
        body=''
        for left,right,equal in [('a','a',True),('a','b',True),('a','different',False),('empty','empty',True),('empty','a',False)]:
            body+=f'PUSH_STR {left}\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPUSH_STR {right}\nEQ\nPUSH_BOOL {int(equal)}\nEQ\nASSERT\n'
        body+='PUSH_STR a\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPUSH_I64 0\nNE\nASSERT\n'
        body+='PUSH_F64 nan\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nDUP\nNE\nASSERT\n'
        self.paired(body,helpers)

    def test_generated_helper_null_and_identity_match_vm(self):
        with tempfile.TemporaryDirectory(prefix='nano-string-guard-') as tmp:
            work=Path(tmp)
            module=self.assemble(work,'PUSH_VOID\nPOP\n')
            self.native(work,module)
            harness=work/'harness.c'
            harness.write_text('''#define main native_main
#include "input.c"
#undef main
#include "nanovm/heap.h"
#include "nanovm/value.h"
#include <assert.h>
static VmString *text(const char *s) {
    size_t n = strlen(s);
    VmString *v = calloc(1, sizeof *v + n + 1);
    assert(v); v->length = (uint32_t)n;
    memcpy(v->data, s, n + 1); return v;
}
int main(void) {
    VmString *a = text("same"), *b = text("same"), *c = text("other"), *e = text("");
    VmString *values[] = {NULL, a, a, b, c, e};
    assert(native_main() == 0);
    for (size_t i = 0; i < 6; ++i) for (size_t j = 0; j < 6; ++j) {
        nmap_value left = {5, 0, values[i] ? values[i]->data : NULL};
        nmap_value right = {5, 0, values[j] ? values[j]->data : NULL};
        assert(nvalue_equal(left, right) == val_equal(val_string(values[i]), val_string(values[j])));
    }
    free(a); free(b); free(c); free(e); return 0;
}
''')
            compiler=shlex.split(os.environ.get('NANO_NATIVE_TEST_CC','cc'))
            dead='-Wl,-dead_strip' if sys.platform=='darwin' else '-Wl,--gc-sections'
            binary=work/'harness'
            self.checked([*compiler,'-std=c11','-D_GNU_SOURCE','-O2','-Wall','-Wextra','-Werror',
                          '-ffunction-sections','-fdata-sections',dead,
                          '-fsanitize=address,undefined','-fno-sanitize-recover=all',
                          '-I'+str(ROOT/'src'),harness,ROOT/'src/nanovm/value.c',ROOT/'src/nanovm/heap.c','-o',binary])
            self.checked([binary])
