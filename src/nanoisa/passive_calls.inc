/* I prove closed calls under scalar actual arguments, not declaration tags.
 * Definite local initialization is intersected at CFG joins. Every admitted
 * producer returns a scalar or traps, so a successfully returning path cannot
 * introduce an aggregate, foreign value, or mutable shared object. */
#define PASSIVE_CALL_DEPTH 64u
#define PASSIVE_CALL_INSNS 65536u
#define PASSIVE_CALL_WORDS 2097152u
#define PASSIVE_CALL_STEPS 1048576u

typedef struct {
    DecodedInstruction instruction;
    uint32_t pc, length;
    int32_t height;
    bool queued;
} ClosedInstruction;
typedef struct {
    const NvmModule *module;
    uint8_t *state; /* unseen, visiting, proved, refused */
    uint32_t depth;
} ClosedCalls;

static uint32_t closed_target(const ClosedInstruction *code, uint32_t count,
                              uint32_t target) {
    uint32_t low = 0, high = count;
    while (low < high) {
        uint32_t mid = low + (high - low) / 2;
        if (code[mid].pc < target) low = mid + 1;
        else high = mid;
    }
    return low < count && code[low].pc == target ? low : count;
}
static bool closed_function(ClosedCalls *calls, uint32_t index) {
    const NvmModule *m = calls->module;
    if (index >= m->function_count || calls->depth >= PASSIVE_CALL_DEPTH) return false;
    if (calls->state[index]) return calls->state[index] == 2;
    const NvmFunctionEntry *f = &m->functions[index];
    if (f->upvalue_count || f->arity > f->local_count || f->result_count > 1 ||
        !f->code_length || f->code_offset > m->code_size ||
        f->code_length > m->code_size - f->code_offset) return false;
    calls->state[index] = 1;
    calls->depth++;
    bool ok = true, returns = false;
    uint32_t count = 0, end = f->code_offset + f->code_length;
    for (uint32_t pc = f->code_offset; pc < end;) {
        DecodedInstruction d;
        uint32_t length = isa_decode(m->code + pc, end - pc, &d);
        if (!length || count == PASSIVE_CALL_INSNS) { ok = false; break; }
        ++count; pc += length;
    }
    uint32_t width = (f->local_count + 63u) / 64u;
    if (!width) width = 1;
    if (!ok || count > PASSIVE_CALL_WORDS / width) {
        calls->state[index] = 3; calls->depth--; return false;
    }
    ClosedInstruction *code = calloc(count, sizeof(*code));
    uint64_t *locals = calloc((size_t)count * width, sizeof(*locals));
    uint64_t *out = calloc(width, sizeof(*out));
    uint32_t *queue = calloc(count, sizeof(*queue));
    if (!code || !locals || !out || !queue) { ok = false; goto done; }
    uint32_t pc = f->code_offset;
    for (uint32_t i = 0; i < count; ++i) {
        code[i].pc = pc;
        code[i].length = isa_decode(m->code + pc, end - pc, &code[i].instruction);
        code[i].height = -1;
        pc += code[i].length;
    }
    for (uint32_t i = 0; i < f->arity; ++i) locals[i / 64] |= UINT64_C(1) << (i % 64);
    code[0].height = 0; code[0].queued = true;
    uint32_t head = 0, tail = 0, pending = 1, steps = 0;
    queue[tail++] = 0; if (tail == count) tail = 0;
    while (ok && pending) {
        if (++steps > PASSIVE_CALL_STEPS) { ok = false; break; }
        uint32_t i = queue[head++]; if (head == count) head = 0;
        --pending; code[i].queued = false;
        DecodedInstruction *d = &code[i].instruction;
        const InstructionInfo *info = isa_get_info(d->opcode);
        if (!info) { ok = false; break; }
        int32_t pop = info->pop_count, push = info->push_count;
        bool branch = d->opcode == OP_JMP || d->opcode == OP_JMP_TRUE || d->opcode == OP_JMP_FALSE;
        if ((d->opcode == OP_CALL || d->opcode == OP_TAIL_CALL)) {
            uint32_t callee = d->operands[0].u32;
            if (!closed_function(calls, callee)) { ok = false; break; }
            pop = m->functions[callee].arity; push = m->functions[callee].result_count;
        } else if (d->opcode == OP_RET) {
            pop = f->result_count; push = 0;
        } else if (!allowed(d->opcode, 2) && !branch &&
                   d->opcode != OP_TYPE_CHECK && d->opcode != OP_ASSERT &&
                   d->opcode != OP_CAST_INT && d->opcode != OP_CAST_FLOAT &&
                   d->opcode != OP_CAST_BOOL && d->opcode != OP_CAST_STRING) {
            ok = false; break;
        }
        if (pop < 0 || push < 0 || code[i].height < pop ||
            code[i].height - pop > INT32_MAX - push) { ok = false; break; }
        int32_t height = code[i].height - pop + push;
        for (uint32_t w = 0; w < width; ++w) out[w] = locals[(size_t)i * width + w];
        if (d->opcode == OP_LOAD_LOCAL || d->opcode == OP_STORE_LOCAL) {
            uint32_t local = d->operands[0].u16;
            if (local >= f->local_count) { ok = false; break; }
            if (d->opcode == OP_STORE_LOCAL) out[local / 64] |= UINT64_C(1) << (local % 64);
        }
        if (d->opcode == OP_RET || d->opcode == OP_TAIL_CALL) {
            if (d->opcode == OP_RET ? height != 0 :
                (height != f->result_count || m->functions[d->operands[0].u32].result_count != f->result_count)) ok = false;
            returns = true; continue;
        }
        uint32_t next[2], nnext = 0;
        if (branch) {
            int64_t target = (int64_t)code[i].pc + d->operands[0].i32;
            if (target < f->code_offset || target >= end) { ok = false; break; }
            next[nnext++] = closed_target(code, count, (uint32_t)target);
        }
        if (d->opcode != OP_JMP) next[nnext++] = i + 1;
        for (uint32_t edge = 0; ok && edge < nnext; ++edge) {
            uint32_t j = next[edge];
            if (j >= count) { ok = false; break; }
            bool fresh = code[j].height < 0, changed = fresh;
            if (!fresh && code[j].height != height) { ok = false; break; }
            code[j].height = height;
            for (uint32_t w = 0; w < width; ++w) {
                uint64_t *slot = &locals[(size_t)j * width + w];
                uint64_t value = fresh ? out[w] : *slot & out[w];
                if (*slot != value) changed = true;
                *slot = value;
            }
            if (changed && !code[j].queued) {
                code[j].queued = true; queue[tail++] = j;
                if (tail == count) tail = 0;
                ++pending;
            }
        }
    }
    /* I check reads after convergence, including every predecessor of loops. */
    for (uint32_t i = 0; ok && i < count; ++i) {
        if (code[i].height < 0 || code[i].instruction.opcode != OP_LOAD_LOCAL) continue;
        uint32_t local = code[i].instruction.operands[0].u16;
        if (!(locals[(size_t)i * width + local / 64] & (UINT64_C(1) << (local % 64)))) ok = false;
    }
done:
    free(code); free(locals); free(out); free(queue);
    calls->state[index] = ok && returns ? 2 : 3;
    calls->depth--;
    return calls->state[index] == 2;
}
