#ifndef NVM2C_EFFECT_FLOW_H
#define NVM2C_EFFECT_FLOW_H

/* I share arm reachability between native representation and callable analysis.
 * Nested registrations do not execute their arm as ordinary fallthrough. */
typedef struct {
    uint32_t function, operation, target;
    uint16_t start, count;
    uint8_t *flow;
} Nvm2cHandler;

static uint32_t effect_operation(const NvmModule *mod, uint32_t name) {
    for (uint32_t i = 0; i < name; ++i)
        if (!strcmp(mod->strings[i], mod->strings[name])) return i;
    return name;
}

static int effect_collect_entries(const NvmModule *mod, Nvm2cHandler **entries, size_t *entry_count) {
    for (uint32_t f = 0; f < mod->function_count; ++f) {
        VmDecodedFunction decoded = {0};
        char error[VM_DECODE_ERROR_SIZE];
        if (!vm_decode_function(mod, f, &decoded, error)) return 0;
        const NvmFunctionEntry *fn = &mod->functions[f];
        for (size_t i = 0; i < decoded.instruction_count; ++i) {
            const VmDecodedInstruction *instruction = &decoded.instructions[i];
            const DecodedInstruction *in = &instruction->instruction;
            if (in->opcode != OP_HANDLER_PUSH) continue;
            int64_t target = (int64_t)instruction->byte_offset + in->operands[1].i32;
            if (in->operands[0].u32 >= mod->string_count || target < 0 || (uint64_t)target >= fn->code_length ||
                (uint32_t)in->operands[2].u16 + in->operands[3].u16 > fn->local_count ||
                !vm_decoded_function_has_boundary(&decoded, (uint32_t)target)) {
                vm_decoded_function_free(&decoded); return 0;
            }
            Nvm2cHandler *next = realloc(*entries, (*entry_count + 1) * sizeof *next);
            if (!next) { vm_decoded_function_free(&decoded); return 0; }
            *entries = next;
            Nvm2cHandler *h = &next[(*entry_count)++];
            *h = (Nvm2cHandler){f, effect_operation(mod, in->operands[0].u32), (uint32_t)target,
                in->operands[2].u16, in->operands[3].u16, NULL};
            h->flow = calloc((size_t)fn->code_length + 1, 1);
            size_t *queue = malloc(((size_t)fn->code_length + 1) * sizeof *queue);
            if (!h->flow || !queue) { free(queue); vm_decoded_function_free(&decoded); return 0; }
            size_t head = 0, tail = 1;
            queue[0] = (size_t)target; h->flow[target] = 1;
            while (head < tail) {
                size_t at = queue[head++];
                if (at == fn->code_length) continue;
                const VmDecodedInstruction *item = vm_decoded_function_at(&decoded, (uint32_t)at);
                if (!item) { free(queue); vm_decoded_function_free(&decoded); return 0; }
                const DecodedInstruction *op = &item->instruction;
                if (op->opcode == OP_RET || op->opcode == OP_HALT || op->opcode == OP_EFFECT_RESUME || op->opcode == OP_TAIL_CALL) continue;
                size_t edges[2] = {item->next_byte_offset, 0}, count = 1;
                if (op->opcode == OP_JMP || op->opcode == OP_JMP_FALSE || op->opcode == OP_JMP_TRUE) {
                    int64_t to = (int64_t)at + op->operands[0].i32;
                    if (to < 0 || (uint64_t)to > fn->code_length || !vm_decoded_function_has_boundary(&decoded, (uint32_t)to)) {
                        free(queue); vm_decoded_function_free(&decoded); return 0;
                    }
                    if (op->opcode == OP_JMP) edges[0] = (size_t)to;
                    else edges[count++] = (size_t)to;
                }
                for (size_t e = 0; e < count; ++e) if (!h->flow[edges[e]]) {
                    h->flow[edges[e]] = 1; queue[tail++] = edges[e];
                }
            }
            free(queue);
        }
        vm_decoded_function_free(&decoded);
    }
    return 1;
}
#endif
