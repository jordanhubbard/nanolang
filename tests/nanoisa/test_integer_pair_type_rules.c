/* I check the VM/ISA integer-pair contract without executing invalid inputs. */
#include <assert.h>
#include <stdio.h>
#include "../../src/nanoisa/verifier_types.c"

int main(void) {
    const uint8_t operations[] = {
        OP_I64_ADD_CARRY, OP_I64_SUB_BORROW,
        OP_I64_MUL_WIDE_S, OP_I64_MUL_WIDE_U
    };
    unsigned checks = 0;
    for (size_t i = 0; i < sizeof operations; ++i) {
        TypeRule rule;
        const InstructionInfo *info = isa_get_info(operations[i]);
        assert(info && type_rule_for(operations[i], &rule));
        assert(rule.arg_count == info->pop_count);
        assert(rule.result_count == info->push_count);
        assert(rule.arg_count == (i < 2 ? 3 : 2));
        assert(rule.result_count == 2 && rule.result == TAG_INT);
        checks += 5;
        for (uint8_t argument = 0; argument < rule.arg_count; ++argument) {
            assert(rule.args[argument] == TAG_INT);
            ++checks;
        }
    }
    /* I retain the existing unknown-join and truthy-branch boundaries. */
    TypeRule rule;
    assert(join(TAG_INT, TAG_INT) == TAG_INT);
    assert(join(TAG_INT, TAG_BOOL) == TYPE_UNKNOWN);
    assert(!type_rule_for(OP_JMP_TRUE, &rule));
    assert(!type_rule_for(OP_LOAD_LOCAL, &rule));
    printf("I passed %u integer-pair type-rule checks.\n", checks + 4);
    return 0;
}
