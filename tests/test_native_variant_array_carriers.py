"""I retain exact integer-array handles in finite generic variant payloads."""
import unittest
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


if __name__ == '__main__': unittest.main()
