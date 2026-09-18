#include "passive.h"
#include "isa.h"
#include <stdlib.h>

/* I keep the encoded form so bridges cannot discard or renumber claims. */
typedef struct { const uint8_t *p; size_t left; bool ok; } Reader;
static uint32_t word(Reader *r) {
    if (r->left < 4) { r->ok = false; return 0; }
    const uint8_t *p = r->p;
    uint32_t v = (uint32_t)p[0] | (uint32_t)p[1] << 8 |
                 (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
    r->p += 4; r->left -= 4; return v;
}
typedef struct {
    uint32_t entry, exit, result, dependencies, reads;
    const uint8_t *deps, *inputs;
    bool done;
} Node;
static uint32_t at(const uint8_t *p, uint32_t index) {
    Reader r = {p + (size_t)index * 4, 4, true}; return word(&r);
}
static bool contains(const uint8_t *p, uint32_t count, uint32_t value) {
    for (uint32_t i = 0; i < count; ++i) if (at(p, i) == value) return true;
    return false;
}
static bool scalar(uint8_t tag) {
    return tag == TAG_INT || tag == TAG_BOOL || tag == TAG_FLOAT ||
           tag == TAG_U8 || tag == TAG_STRING;
}
/* A declaration is not a proof: the entry prefix checks the actual value. */
static bool guarded_input(const NvmModule *m, uint32_t function,
                          uint32_t input, uint32_t entry) {
    const NvmFunctionEntry *f = &m->functions[function];
    if (!m->function_param_types || !m->function_param_types[function]) return false;
    uint32_t pc = f->code_offset, previous = 0;
    bool first = true;
    while (pc < entry) {
        DecodedInstruction guard[3];
        for (unsigned i = 0; i < 3; ++i) {
            if (pc >= entry) return false;
            uint32_t length = isa_decode(m->code + pc, entry - pc, &guard[i]);
            if (!length) return false;
            pc += length;
        }
        if (guard[0].opcode != OP_LOAD_LOCAL || guard[1].opcode != OP_TYPE_CHECK ||
            guard[2].opcode != OP_ASSERT) return false;
        uint32_t parameter = guard[0].operands[0].u16;
        if (parameter >= f->arity || (!first && parameter <= previous)) return false;
        uint8_t tag = m->function_param_types[function][parameter];
        if (!scalar(tag) || tag == TAG_U8 || guard[1].operands[0].u8 != tag) return false;
        if (parameter == input) return true;
        if (parameter > input) return false;
        previous = parameter;
        first = false;
    }
    return false;
}
static bool allowed(uint8_t op, uint32_t version) {
    switch (op) {
    case OP_NOP: case OP_PUSH_I64: case OP_PUSH_F64: case OP_PUSH_BOOL:
    case OP_PUSH_U8: case OP_PUSH_STR: case OP_LOAD_LOCAL: case OP_STORE_LOCAL:
    case OP_DUP: case OP_POP: case OP_SWAP: case OP_ROT3:
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_MOD: case OP_NEG:
    case OP_EQ: case OP_NE: case OP_LT: case OP_LE: case OP_GT: case OP_GE:
    case OP_AND: case OP_OR: case OP_NOT:
        return true;
    case OP_I64_ADD: case OP_I64_SUB: case OP_I64_MUL: case OP_I64_DIV_S:
    case OP_I64_REM_S: case OP_I64_NEG: case OP_I64_EQ: case OP_I64_NE:
    case OP_I64_LT_S: case OP_I64_LE_S: case OP_I64_GT_S: case OP_I64_GE_S:
    case OP_F64_FROM_BITS: case OP_F64_TO_BITS:
    case OP_F64_ADD: case OP_F64_SUB: case OP_F64_MUL: case OP_F64_DIV:
    case OP_F64_NEG: case OP_F64_EQ: case OP_F64_NE: case OP_F64_LT:
    case OP_F64_LE: case OP_F64_GT: case OP_F64_GE:
    case OP_BOOL_AND: case OP_BOOL_OR: case OP_BOOL_NOT: case OP_STR_CONCAT:
        return version == 2;
    default: return false;
    }
}
#include "passive_calls.inc"

static bool node_code(const NvmModule *m, const NvmFunctionEntry *f,
                      Node *nodes, uint32_t count, uint32_t index, uint32_t version, ClosedCalls *calls) {
    Node *n = &nodes[index];
    uint32_t depth = 0;
    bool *seen_deps = calloc(count, sizeof(bool));
    bool *seen_reads = calloc(f->arity ? f->arity : 1, sizeof(bool));
    if (!seen_deps || !seen_reads) { free(seen_deps); free(seen_reads); return false; }
    bool ok = true;
    for (uint32_t pc = n->entry; ok && pc < n->exit;) {
        DecodedInstruction d;
        uint32_t length = isa_decode(m->code + pc, n->exit - pc, &d);
        if (!length || (!allowed(d.opcode, version) && !(version == 2 && d.opcode == OP_CALL))) { ok = false; break; }
        const InstructionInfo *info = isa_get_info(d.opcode);
        if (!info) { ok = false; break; }
        int32_t pop = info->pop_count, push = info->push_count;
        if (d.opcode == OP_CALL) {
            uint32_t callee = d.operands[0].u32;
            if (!closed_function(calls, callee)) { ok = false; break; }
            pop = m->functions[callee].arity; push = m->functions[callee].result_count;
        }
        if (pop < 0 || push < 0 || depth < (uint32_t)pop ||
            depth - (uint32_t)pop > UINT32_MAX - (uint32_t)push) { ok = false; break; }
        depth = depth - (uint32_t)pop + (uint32_t)push;
        if (d.opcode == OP_STORE_LOCAL) {
            if (pc + length != n->exit || d.operands[0].u16 != n->result || depth != 0) ok = false;
        } else if (pc + length == n->exit) ok = false;
        if (d.opcode == OP_LOAD_LOCAL) {
            uint32_t local = d.operands[0].u16;
            uint32_t producer = count;
            for (uint32_t j = 0; j < count; ++j) if (nodes[j].result == local) producer = j;
            if (producer < count) {
                if (!contains(n->deps, n->dependencies, producer)) ok = false;
                seen_deps[producer] = true;
            } else {
                if (local >= f->arity || !contains(n->inputs, n->reads, local)) ok = false;
                else seen_reads[local] = true;
            }
        }
        pc += length;
    }
    for (uint32_t i = 0; i < n->dependencies; ++i) if (!seen_deps[at(n->deps, i)]) ok = false;
    for (uint32_t i = 0; i < n->reads; ++i) if (!seen_reads[at(n->inputs, i)]) ok = false;
    free(seen_deps); free(seen_reads);
    return ok && depth == 0;
}
static bool block(Reader *r, const NvmModule *m, uint32_t *previous_function,
                  uint32_t *previous_exit, bool first, uint32_t version, ClosedCalls *calls) {
    uint32_t kind = word(r), function = word(r), entry = word(r), exit = word(r), count = word(r);
    if (!r->ok || (kind != 1 && kind != 2) || function >= m->function_count ||
        !count || count > r->left / 28 || entry >= exit || exit > m->code_size) return false;
    const NvmFunctionEntry *f = &m->functions[function];
    if (f->code_offset > m->code_size || f->code_length > m->code_size - f->code_offset ||
        entry < f->code_offset || exit > f->code_offset + f->code_length ||
        (!first && (function < *previous_function || (function == *previous_function && entry < *previous_exit)))) return false;
    Node *nodes = calloc(count, sizeof(Node));
    if (!nodes) return false;
    bool ok = true;
    for (uint32_t i = 0; ok && i < count; ++i) {
        Node *n = &nodes[i];
        n->entry = word(r); n->exit = word(r); n->result = word(r);
        n->dependencies = word(r); n->reads = word(r);
        uint32_t effects = word(r), resources = word(r);
        if (!r->ok || effects || resources || n->entry < entry || n->exit > exit ||
            n->entry >= n->exit || n->result < f->arity || n->result >= f->local_count ||
            n->dependencies > count || (version == 1 && n->reads != 0) ||
            (uint64_t)n->dependencies + n->reads > r->left / 4 || (kind == 1 && n->dependencies)) { ok = false; break; }
        n->deps = r->p;
        for (uint32_t j = 0; j < n->dependencies; ++j) {
            uint32_t d = word(r);
            if (d >= count || d == i || (j && d <= at(n->deps, j - 1))) ok = false;
        }
        n->inputs = r->p;
        for (uint32_t j = 0; j < n->reads; ++j) {
            uint32_t input = word(r);
            if (input >= f->arity || (j && input <= at(n->inputs, j - 1)) ||
                !m->function_param_types || !m->function_param_types[function] ||
                (input < f->arity && (!scalar(m->function_param_types[function][input]) ||
                 !guarded_input(m, function, input, entry)))) ok = false;
        }
        for (uint32_t j = 0; j < i; ++j) if (nodes[j].result == n->result) ok = false;
    }
    /* Physical ranges follow stable topological order, not source order. */
    uint32_t cursor = entry;
    for (uint32_t step = 0; ok && step < count; ++step) {
        uint32_t selected = count;
        for (uint32_t i = 0; i < count; ++i) {
            if (nodes[i].done) continue;
            bool ready = true;
            for (uint32_t j = 0; j < nodes[i].dependencies; ++j)
                if (!nodes[at(nodes[i].deps, j)].done) ready = false;
            if (ready) { selected = i; break; }
        }
        if (selected == count || nodes[selected].entry != cursor) { ok = false; break; }
        ok = node_code(m, f, nodes, count, selected, version, calls);
        cursor = nodes[selected].exit; nodes[selected].done = true;
    }
    if (cursor != exit) ok = false;
    /* I disallow mid-block entry and writes to immutable external inputs.
     * Decoding the entire function also proves block/node start boundaries. */
    bool saw_entry = false, saw_exit = exit == f->code_offset + f->code_length;
    for (uint32_t pc = f->code_offset; ok && pc < f->code_offset + f->code_length;) {
        DecodedInstruction d;
        uint32_t length = isa_decode(m->code + pc, f->code_offset + f->code_length - pc, &d);
        if (!length) { ok = false; break; }
        if (!allowed(d.opcode, version) && d.opcode != OP_JMP && d.opcode != OP_JMP_TRUE &&
            d.opcode != OP_JMP_FALSE && d.opcode != OP_RET && d.opcode != OP_HALT &&
            d.opcode != OP_PRINT && d.opcode != OP_PRINTLN && d.opcode != OP_ASSERT &&
            !(version == 2 && (d.opcode == OP_TYPE_CHECK ||
              d.opcode == OP_CALL))) {
            ok = false; break;
        }
        if (pc == entry) saw_entry = true;
        if (pc == exit) saw_exit = true;
        if (d.opcode == OP_JMP || d.opcode == OP_JMP_TRUE || d.opcode == OP_JMP_FALSE) {
            /* Relative branches use the instruction start, as in the ISA verifier. */
            int64_t target = (int64_t)pc + d.operands[0].i32;
            if (target > entry && target < exit) ok = false;
        }
        if (d.opcode == OP_STORE_LOCAL) {
            uint32_t local = d.operands[0].u16;
            for (uint32_t i = 0; i < count; ++i)
                if (contains(nodes[i].inputs, nodes[i].reads, local)) ok = false;
        }
        pc += length;
    }
    free(nodes);
    *previous_function = function; *previous_exit = exit;
    return ok && saw_entry && saw_exit;
}
bool nvm_passive_valid(const NvmModule *m) {
    if (!m) return false;
    if (!m->passive_size) return m->passive_data == NULL;
    if (!m->passive_data || !m->code || !m->functions) return false;
    Reader r = {m->passive_data, m->passive_size, true};
    uint32_t version = word(&r), count = word(&r);
    if (!r.ok || (version != 1 && version != 2) || !count || count > r.left / 48) return false;
    uint32_t function = 0, exit = 0;
    ClosedCalls calls = {m, NULL, 0};
    calls.state = calloc(m->function_count ? m->function_count : 1, 1);
    if (!calls.state) return false;
    bool ok = true;
    for (uint32_t i = 0; ok && i < count; ++i)
        ok = block(&r, m, &function, &exit, i == 0, version, &calls);
    /* I check purity only inside node ranges. Ordinary version-2 direct calls
     * elsewhere retain the module verifier's target and stack checks. */
    free(calls.state);
    return ok && r.ok && r.left == 0;
}
