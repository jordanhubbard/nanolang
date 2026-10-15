#ifndef NVM2C_REACHABILITY_H
#define NVM2C_REACHABILITY_H

/* I use bounded constant flow only to establish that a refused call cannot
 * execute. Unknown facts, limits and allocation failures preserve refusal.
 * I forget all array lengths at mutations and call boundaries, including
 * aliases held in locals or the operand stack. I infer no callable targets. */
typedef struct { int kind; int64_t value; } Nvm2cConstant;
typedef struct { Nvm2cConstant *values; size_t stack; int seen; } Nvm2cConstantFrame;

static int nvm2c_constant_edge(Nvm2cConstantFrame *frame,
        const Nvm2cConstant *values, size_t locals, size_t stack,
        size_t *bytes, int *changed) {
    size_t size = locals + stack;
    if (!frame->seen) {
        if (size > (16u * 1024u * 1024u - *bytes) / sizeof *values) return 0;
        frame->values = calloc(size ? size : 1, sizeof *values);
        if (!frame->values) return 0;
        memcpy(frame->values, values, size * sizeof *values);
        *bytes += size * sizeof *values;
        frame->seen = 1;
        frame->stack = stack;
        *changed = 1;
    } else {
        if (frame->stack != stack) return 0;
        for (size_t i = 0; i < size; ++i) {
            Nvm2cConstant *old = &frame->values[i];
            if (old->kind && (old->kind != values[i].kind || old->value != values[i].value)) {
                old->kind = 0;
                *changed = 1;
            }
        }
    }
    return 1;
}

static int nvm2c_constant_unreachable(const NvmModule *module,
                                     uint32_t function, uint32_t offset) {
    VmDecodedFunction decoded = {0};
    Nvm2cConstantFrame *frames = NULL;
    Nvm2cConstant *values = NULL;
    char error[VM_DECODE_ERROR_SIZE];
    int unreachable = 0;
    if (function >= module->function_count) return 0;
    /* I decline unmodeled control transfers and access to frame/raw memory. */
    for (uint32_t f = 0; f < module->function_count; ++f) {
        const NvmFunctionEntry *entry = &module->functions[f];
        if (entry->code_offset > module->code_size ||
            entry->code_length > module->code_size - entry->code_offset) return 0;
        for (uint32_t pc = 0; pc < entry->code_length;) {
            DecodedInstruction instruction;
            uint32_t size = isa_decode(module->code + entry->code_offset + pc,
                                      entry->code_length - pc, &instruction);
            if (!size) return 0;
            if ((instruction.opcode >= OP_OWN_MOVE_LOCAL && instruction.opcode <= OP_CALL_REF) ||
                (instruction.opcode >= OP_REGION_BEGIN && instruction.opcode <= OP_REBORROW_EXCLUSIVE) ||
                (instruction.opcode >= OP_FILE_SERVICE && instruction.opcode <= OP_FILE_CALL_INDIRECT_REFS) ||
                (instruction.opcode >= OP_HANDLER_PUSH && instruction.opcode <= OP_EFFECT_RESUME) ||
                (instruction.opcode >= OP_MEM_STORE8 && instruction.opcode <= OP_MEM_STORE64) ||
                instruction.opcode == OP_MATCH_TAG) return 0;
            pc += size;
        }
    }
    if (!vm_decode_function(module, function, &decoded, error)) return 0;
    size_t count = decoded.instruction_count;
    size_t locals = module->functions[function].local_count;
    if (!count || count > 65536 || locals > 1024) goto done;
    size_t capacity = locals + count * 3 + 1;
    frames = calloc(count + 1, sizeof *frames);
    values = calloc(capacity, sizeof *values);
    if (!frames || !values) goto done;
    size_t bytes = 0, visits = 0;
    int changed = 0;
    if (!nvm2c_constant_edge(&frames[0], values, locals, 0, &bytes, &changed)) goto done;
    do {
        changed = 0;
        for (size_t at = 0; at < count; ++at) {
            if (!frames[at].seen) continue;
            if (++visits > 262144) goto done;
            const VmDecodedInstruction *current = &decoded.instructions[at];
            const DecodedInstruction *in = &current->instruction;
            const InstructionInfo *info = isa_get_info(in->opcode);
            if (!info) goto done;
            int pop = info->pop_count, push = info->push_count;
            uint32_t target = in->operands[0].u32;
            switch (in->opcode) {
            case OP_CALL: case OP_TAIL_CALL:
                if (target >= module->function_count) goto done;
                pop = module->functions[target].arity;
                push = module->functions[target].result_count; break;
            case OP_CALL_EXTERN:
                if (target >= module->import_count) goto done;
                pop = module->imports[target].param_count;
                push = module->imports[target].return_type != TAG_VOID; break;
            case OP_CALL_INDIRECT: pop = in->operands[0].u16 + 1; push = in->operands[1].u16; break;
            case OP_CLOSURE_NEW: pop = in->operands[1].u16; push = 1; break;
            case OP_AGG_PACK: pop = in->operands[3].u16; push = 1; break;
            case OP_ARR_LITERAL: pop = in->operands[1].u16; push = 1; break;
            case OP_RET: pop = module->functions[function].result_count; push = 0; break;
            default: break;
            }
            size_t before = frames[at].stack;
            if (pop < 0 || push < 0 || push > 3 || (size_t)pop > before) goto done;
            size_t base = before - (size_t)pop, after = base + (size_t)push;
            if (locals + after > capacity) goto done;
            memcpy(values, frames[at].values, (locals + before) * sizeof *values);
            Nvm2cConstant *stack = values + locals;
            switch (in->opcode) {
            case OP_ARR_PUSH: case OP_ARR_POP: case OP_ARR_REMOVE: case OP_ARR_SET:
            case OP_CALL: case OP_TAIL_CALL: case OP_CALL_INDIRECT:
            case OP_CALL_EXTERN: case OP_CALL_MODULE:
                for (size_t i = 0; i < locals + before; ++i)
                    if (values[i].kind == 3) values[i].kind = 0;
                break;
            default: break;
            }
            Nvm2cConstant output[3] = {{0}};
            Nvm2cConstant a = pop ? stack[base] : (Nvm2cConstant){0};
            Nvm2cConstant b = pop > 1 ? stack[base + 1] : (Nvm2cConstant){0};
            switch (in->opcode) {
            case OP_PUSH_I64: output[0] = (Nvm2cConstant){1, in->operands[0].i64}; break;
            case OP_PUSH_BOOL: output[0] = (Nvm2cConstant){2, !!in->operands[0].u8}; break;
            case OP_LOAD_LOCAL:
                if (in->operands[0].u16 >= locals) goto done;
                output[0] = values[in->operands[0].u16]; break;
            case OP_STORE_LOCAL:
                if (in->operands[0].u16 >= locals) goto done;
                values[in->operands[0].u16] = a; break;
            case OP_DUP: output[0] = output[1] = a; break;
            case OP_SWAP: output[0] = b; output[1] = a; break;
            case OP_ROT3: output[0] = stack[base + 2]; output[1] = a; output[2] = b; break;
            case OP_ARR_LITERAL:
                output[0] = (Nvm2cConstant){3, in->operands[1].u16};
                break;
            case OP_ARR_LEN:
                if (a.kind == 3) output[0] = (Nvm2cConstant){1, a.value};
                break;
            case OP_LT: case OP_I64_LT_S:
                if (a.kind == 1 && b.kind == 1) output[0] = (Nvm2cConstant){2, a.value < b.value};
                break;
            case OP_EQ: case OP_NE:
                if ((a.kind == 1 || a.kind == 2) && a.kind == b.kind)
                    output[0] = (Nvm2cConstant){2, in->opcode == OP_EQ ? a.value == b.value : a.value != b.value};
                break;
            case OP_NOT:
                if (a.kind == 2) output[0] = (Nvm2cConstant){2, !a.value};
                break;
            default: break;
            }
            for (int i = 0; i < push; ++i) stack[base + (size_t)i] = output[i];
            if (in->opcode == OP_RET || in->opcode == OP_HALT || in->opcode == OP_TAIL_CALL) continue;
            if (in->opcode == OP_JMP || in->opcode == OP_JMP_TRUE || in->opcode == OP_JMP_FALSE) {
                int64_t pc = (int64_t)current->byte_offset + in->operands[0].i32;
                if (pc < 0 || pc > decoded.code_size || !vm_decoded_function_has_boundary(&decoded, (uint32_t)pc)) goto done;
                size_t destination = pc == decoded.code_size ? count :
                    (size_t)(vm_decoded_function_at(&decoded, (uint32_t)pc) - decoded.instructions);
                int conditional = in->opcode != OP_JMP;
                int known = conditional && a.kind == 2;
                int taken = in->opcode == OP_JMP_TRUE ? !!a.value : !a.value;
                if ((!known || taken) && !nvm2c_constant_edge(&frames[destination], values, locals, after, &bytes, &changed)) goto done;
                if (!conditional || (known && taken)) continue;
            }
            if (!nvm2c_constant_edge(&frames[at + 1], values, locals, after, &bytes, &changed)) goto done;
        }
    } while (changed);
    for (size_t at = 0; at < count; ++at)
        if (decoded.instructions[at].byte_offset == offset) unreachable = !frames[at].seen;
done:
    if (frames) for (size_t at = 0; at <= decoded.instruction_count; ++at) free(frames[at].values);
    free(frames);
    free(values);
    vm_decoded_function_free(&decoded);
    return unreachable;
}
#endif
