"""I retain exact integer-array handles in finite generic variant payloads."""
import unittest
import os
import subprocess
import tempfile
from pathlib import Path
from tests import test_native_variant_scalar_carriers as base
HEADER = base.HEADER


class VariantArrayCarriers(unittest.TestCase):
    checked = base.VariantScalarCarriers.checked
    paired = base.VariantScalarCarriers.paired

    def test_scalar_array_instances_both_orders(self):
        for reverse in (False, True):
            for variant in (0, 1):
                with self.subTest(reverse=reverse, variant=variant):
                    scalar = 'PUSH_I64 7\nAGG_PACK 1 0 0 1\nCALL relay\nAGG_GET 0\nPUSH_I64 7\nI64_EQ\nASSERT\n'
                    array = f'PUSH_I64 42\nARR_LITERAL 1 1\nAGG_PACK 1 0 {variant} 1\nCALL relay\nAGG_GET 0\nDUP\nTYPE_CHECK 7\nASSERT\nPUSH_I64 0\nARR_GET\nPUSH_I64 42\nI64_EQ\nASSERT\n'
                    body = array+scalar if reverse else scalar+array
                    self.paired(HEADER+'.function main 0 0 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n'+
                        '.function relay 1 2 0 union 1\nLOAD_LOCAL 0\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nRET\n.end\n')

    def test_record_join_padding_and_projected_call(self):
        for selected in (0, 1):
            with self.subTest(selected=selected):
                body = f'PUSH_BOOL {selected}\nJMP_FALSE scalar\n'
                body += 'ARR_NEW 1\nAGG_PACK 1 0 1 1\nJMP joined\nscalar:\nPUSH_I64 7\nAGG_PACK 1 0 0 1\n'
                body += f'joined:\nCALL relay\nAGG_GET 0\nCALL is_array\nPUSH_BOOL {selected}\nEQ\nASSERT\n'
                body += 'AGG_PACK 1 0 2 0\nCALL relay\nAGG_TAG\nPUSH_I64 2\nI64_EQ\nASSERT\n'
                self.paired(HEADER+'.function main 0 0 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n'+
                    '.function relay 1 1 0 union 1\nLOAD_LOCAL 0\nRET\n.end\n'+
                    '.function is_array 1 2 0 bool 1\nLOAD_LOCAL 0\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nTYPE_CHECK 7\nRET\n.end\n')

    def test_empty_array_and_checked_return(self):
        self.paired(HEADER+'''.function main 0 1 0 int 1
PUSH_I64 7
AGG_PACK 1 0 0 1
CALL relay
POP
ARR_NEW 1
AGG_PACK 1 0 1 1
CALL relay
CALL extract
STORE_LOCAL 0
LOAD_LOCAL 0
ARR_LEN
PUSH_I64 0
I64_EQ
ASSERT
LOAD_LOCAL 0
PUSH_I64 23
ARR_PUSH
POP
LOAD_LOCAL 0
PUSH_I64 0
ARR_GET
PUSH_I64 23
I64_EQ
ASSERT
PUSH_I64 0
RET
.end
.function relay 1 1 0 union 1
LOAD_LOCAL 0
RET
.end
.function extract 1 1 0 array 1
LOAD_LOCAL 0
AGG_GET 0
RET
.end
''')

    def test_aliases_after_churn_and_mutation(self):
        self.paired(HEADER+'''.function main 0 3 0 int 1
ARR_NEW 1
AGG_PACK 1 0 1 1
CALL relay
STORE_LOCAL 0
LOAD_LOCAL 0
AGG_GET 0
STORE_LOCAL 1
PUSH_STR kept
AGG_PACK 1 0 0 1
CALL relay
POP
PUSH_I64 0
STORE_LOCAL 2
loop:
LOAD_LOCAL 2
PUSH_I64 2048
I64_LT_S
JMP_FALSE done
CALL temporary
POP
LOAD_LOCAL 2
PUSH_I64 1
I64_ADD
STORE_LOCAL 2
JMP loop
done:
LOAD_LOCAL 1
PUSH_I64 42
ARR_PUSH
POP
LOAD_LOCAL 0
AGG_GET 0
PUSH_I64 0
ARR_GET
PUSH_I64 42
I64_EQ
ASSERT
LOAD_LOCAL 1
PUSH_I64 0
PUSH_I64 73
ARR_SET
POP
LOAD_LOCAL 0
AGG_GET 0
PUSH_I64 0
ARR_GET
PUSH_I64 73
I64_EQ
ASSERT
PUSH_I64 0
RET
.end
.function relay 1 1 0 union 1
LOAD_LOCAL 0
RET
.end
.function temporary 0 0 0 union 1
PUSH_I64 99
ARR_LITERAL 1 1
AGG_PACK 1 0 1 1
RET
.end
''')

    def test_slice_remove_pop_and_absent_reads_preserve_carrier(self):
        self.paired(HEADER + """.function main 0 2 0 int 1
PUSH_I64 7
AGG_PACK 1 0 0 1
CALL relay
POP
PUSH_I64 7
PUSH_I64 9
PUSH_I64 11
ARR_LITERAL 1 3
AGG_PACK 1 0 1 1
CALL relay
STORE_LOCAL 0
LOAD_LOCAL 0
AGG_GET 0
PUSH_I64 0
PUSH_I64 2
ARR_SLICE
PUSH_I64 0
ARR_GET
PUSH_I64 7
I64_EQ
ASSERT
LOAD_LOCAL 0
AGG_GET 0
PUSH_I64 0
ARR_REMOVE
POP
LOAD_LOCAL 0
AGG_GET 0
ARR_POP
PUSH_I64 11
I64_EQ
ASSERT
LOAD_LOCAL 0
AGG_GET 0
ARR_POP
PUSH_I64 9
I64_EQ
ASSERT
LOAD_LOCAL 0
AGG_GET 0
ARR_POP
TYPE_CHECK 0
ASSERT
LOAD_LOCAL 0
AGG_GET 0
PUSH_I64 0
ARR_GET
TYPE_CHECK 0
ASSERT
PUSH_I64 0
RET
.end
.function relay 1 1 0 union 1
LOAD_LOCAL 0
RET
.end
""")

    def test_scalar_member_still_fails_checked_array_extraction(self):
        operations = ['PUSH_I64 0\nARR_GET', 'ARR_POP',
                      'PUSH_I64 0\nARR_REMOVE', 'PUSH_I64 0\nPUSH_I64 1\nARR_SLICE']
        for operation in operations:
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                assembly, module, source, binary = [work / name for name in ('main.nasm', 'main.nvm', 'main.c', 'main')]
                assembly.write_text(HEADER + '.function main 0 0 0 int 1\n'
                    + 'ARR_NEW 1\nAGG_PACK 1 0 1 1\nCALL relay\nPOP\n'
                    + 'PUSH_I64 7\nAGG_PACK 1 0 0 1\nCALL relay\nAGG_GET 0\n'
                    + operation + '\nPOP\nPUSH_I64 0\nRET\n.end\n'
                    + '.function relay 1 1 0 union 1\nLOAD_LOCAL 0\nRET\n.end\n')
                self.checked([base.ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module])
                self.checked([base.ROOT / 'bin/nano_vm', '--verify-only', module])
                self.checked([base.ROOT / 'bin/nvm2c', module, '-o', source])
                self.checked([os.environ.get('CC', 'cc'), '-std=c11', '-Wall', '-Wextra', '-Werror',
                    '-fsanitize=address,undefined', '-fno-sanitize-recover=all', source, '-o', binary])
                for command in ([base.ROOT / 'bin/nano_vm', module], [binary]):
                    result = subprocess.run(list(map(str, command)), capture_output=True, text=True,
                        timeout=60, env={**os.environ, 'ASAN_OPTIONS':'detect_leaks=0'})
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    if command[0] == binary:
                        self.assertIn('native invariant', result.stderr)
                        self.assertNotIn('ERROR: AddressSanitizer', result.stderr)


if __name__ == '__main__': unittest.main()
