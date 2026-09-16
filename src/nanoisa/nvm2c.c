/*
 * Structured C11 from a closed NanoISA subset.
 *
 * Temps are C arrays so backward goto is valid C. I64_ADD becomes
 * `t[i] = a + b`. Strings live in a parallel `s[]` of C string pointers.
 * The operand stack exists only while translating.
 */

#include "nvm2c.h"
#include "isa.h"
#include "utf8.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NVM2C_MAX_STACK  64
#define NVM2C_MAX_LOCALS 256
#define NVM2C_MAX_TEMPS  256
#define NVM2C_MAX_REC_FIELDS 64

#define NVM2C_VK_INT 0
#define NVM2C_VK_STR 1
#define NVM2C_VK_UNK 2
#define NVM2C_VK_ARR 3
#define NVM2C_VK_REC 4
#define NVM2C_VK_SARR 5
#define NVM2C_VK_RARR 6
#define NVM2C_VK_VALUE 7

typedef struct {
    char *data;
    size_t len;
    size_t cap;
    char *err;
    size_t err_len;
    int failed;
} Nvm2cBuf;

static void nvm2c_fail(Nvm2cBuf *b, const char *fmt, ...) {
    if (b->failed) return;
    b->failed = 1;
    if (!b->err || b->err_len == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b->err, b->err_len, fmt, ap);
    va_end(ap);
}

static int nvm2c_grow(Nvm2cBuf *b, size_t need) {
    if (b->failed) return 0;
    if (b->len + need + 1 <= b->cap) return 1;
    size_t cap = b->cap ? b->cap : 256;
    while (cap < b->len + need + 1) {
        if (cap > (size_t)-1 / 2) {
            nvm2c_fail(b, "output too large");
            return 0;
        }
        cap *= 2;
    }
    char *n = realloc(b->data, cap);
    if (!n) {
        nvm2c_fail(b, "out of memory");
        return 0;
    }
    b->data = n;
    b->cap = cap;
    return 1;
}

static void nvm2c_puts(Nvm2cBuf *b, const char *s) {
    size_t n = strlen(s);
    if (!nvm2c_grow(b, n)) return;
    memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = '\0';
}

static void nvm2c_printf(Nvm2cBuf *b, const char *fmt, ...) {
    if (b->failed) return;
    va_list ap;
    va_start(ap, fmt);
    va_list aq;
    va_copy(aq, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) {
        va_end(aq);
        nvm2c_fail(b, "format error");
        return;
    }
    if (!nvm2c_grow(b, (size_t)n)) {
        va_end(aq);
        return;
    }
    vsnprintf(b->data + b->len, (size_t)n + 1, fmt, aq);
    va_end(aq);
    b->len += (size_t)n;
}

static int ident_ok(const char *s) {
    if (!s || !*s) return 0;
    unsigned char c = (unsigned char)*s;
    if (!(nl_ascii_isalpha(c) || c == '_')) return 0;
    for (s++; *s; s++) {
        c = (unsigned char)*s;
        if (!(nl_ascii_isalnum(c) || c == '_')) return 0;
    }
    return 1;
}

static void fn_c_name(const NvmModule *mod, uint32_t idx, char *out, size_t n) {
    const char *name = NULL;
    if (idx < mod->function_count) {
        uint32_t ni = mod->functions[idx].name_idx;
        if (ni < mod->string_count) name = mod->strings[ni];
    }
    if (ident_ok(name)) {
        snprintf(out, n, "nl_%s", name);
    } else {
        snprintf(out, n, "nl_fn_%u", idx);
    }
}

static const char *c_result_type(const NvmFunctionEntry *fn, uint8_t result_kind) {
    if (fn->result_count == 0 || fn->result_tag == TAG_VOID) return "void";
    if (fn->result_count != 1) return NULL;
    if (fn->result_tag == TAG_INT || fn->result_tag == TAG_BOOL) return "int64_t";
    if (fn->result_tag == TAG_STRING) return "const char *";
    if (fn->result_tag == TAG_ARRAY) {
        if (result_kind == NVM2C_VK_ARR) return "narr_t";
        if (result_kind == NVM2C_VK_SARR) return "nsarr_t";
        if (result_kind == NVM2C_VK_RARR) return "nrarr_t";
        return NULL;
    }
    if (fn->result_tag == TAG_STRUCT || fn->result_tag == TAG_UNION) return "nrec_t";
    return NULL;
}

static int result_is_i64(const NvmFunctionEntry *fn) {
    return fn->result_count == 1 &&
           (fn->result_tag == TAG_INT || fn->result_tag == TAG_BOOL);
}

static int mark_reachable_functions(Nvm2cBuf *b, const NvmModule *mod,
                                    uint8_t *reachable) {
    uint32_t entry = mod->header.entry_point;
    uint32_t scan;
    if (entry >= mod->function_count) {
        nvm2c_fail(b, "entry_point %u is not a function", entry);
        return 0;
    }
    reachable[entry] = 1;
    for (scan = 0; scan < mod->function_count; scan++) {
        uint32_t i;
        int changed = 0;
        for (i = 0; i < mod->function_count; i++) {
            const NvmFunctionEntry *fn;
            const uint8_t *code;
            size_t remaining;
            size_t pc = 0;
            if (!reachable[i]) continue;
            fn = &mod->functions[i];
            if (fn->code_offset > mod->code_size ||
                fn->code_length > mod->code_size - fn->code_offset) {
                nvm2c_fail(b, "function %u: code range is outside the module", i);
                return 0;
            }
            code = mod->code + fn->code_offset;
            remaining = fn->code_length;
            while (pc < remaining) {
                DecodedInstruction ins;
                uint32_t n = isa_decode(code + pc, remaining - pc, &ins);
                if (n == 0) {
                    nvm2c_fail(b, "function %u: invalid instruction at offset %zu", i, pc);
                    return 0;
                }
                if (ins.opcode == OP_CALL || ins.opcode == OP_TAIL_CALL) {
                    uint32_t callee = ins.operands[0].u32;
                    if (callee >= mod->function_count) {
                        nvm2c_fail(b, "function %u: CALL target %u is out of range", i, callee);
                        return 0;
                    }
                    if (!reachable[callee]) {
                        reachable[callee] = 1;
                        changed = 1;
                    }
                }
                pc += n;
            }
        }
        if (!changed) return 1;
    }
    nvm2c_fail(b, "function reachability did not converge");
    return 0;
}

static const char *c_local_type(uint8_t kind) {
    if (kind == NVM2C_VK_STR) return "const char *";
    if (kind == NVM2C_VK_ARR) return "narr_t";
    if (kind == NVM2C_VK_SARR) return "nsarr_t";
    if (kind == NVM2C_VK_REC) return "nrec_t";
    if (kind == NVM2C_VK_RARR) return "nrarr_t";
    return "int64_t";
}

static uint8_t fn_local_kind(const uint8_t *kinds, uint32_t fn, uint16_t slot) {
    return kinds[(size_t)fn * NVM2C_MAX_LOCALS + slot];
}

static void emit_c_string_lit(Nvm2cBuf *b, const char *s, uint32_t len) {
    uint32_t i;
    for (i = 0; i < len; i++) {
        if (s[i] == '\0') {
            nvm2c_fail(b, "PUSH_STR: embedded NUL is not in the nvm2c subset");
            return;
        }
    }
    nvm2c_puts(b, "\"");
    for (i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '\\' || c == '"') {
            nvm2c_printf(b, "\\%c", (char)c);
        } else if (c == '\n') {
            nvm2c_puts(b, "\\n");
        } else if (c == '\t') {
            nvm2c_puts(b, "\\t");
        } else if (c == '\r') {
            nvm2c_puts(b, "\\r");
        } else if (c >= 32 && c < 127) {
            char tmp[2] = {(char)c, 0};
            nvm2c_puts(b, tmp);
        } else {
            nvm2c_fail(b, "PUSH_STR: non-ASCII or control bytes are not in the nvm2c subset");
            return;
        }
    }
    nvm2c_puts(b, "\"");
}

typedef struct {
    uint8_t kind;
    int origin;
    int origin_field;
    uint8_t rec_k[NVM2C_MAX_REC_FIELDS];
} Nvm2cSimSlot;

static int sim_push_slot(Nvm2cBuf *b, uint32_t idx, Nvm2cSimSlot *stk, int *sp,
                         Nvm2cSimSlot slot) {
    if (*sp >= NVM2C_MAX_STACK) {
        nvm2c_fail(b, "function %u: operand stack overflow", idx);
        return 0;
    }
    stk[*sp] = slot;
    (*sp)++;
    return 1;
}

static int sim_push(Nvm2cBuf *b, uint32_t idx, Nvm2cSimSlot *stk, int *sp,
                    uint8_t kind, int origin) {
    Nvm2cSimSlot slot;
    memset(&slot, 0, sizeof slot);
    slot.kind = kind;
    slot.origin = origin;
    slot.origin_field = -1;
    return sim_push_slot(b, idx, stk, sp, slot);
}

static int sim_pop(Nvm2cBuf *b, uint32_t idx, Nvm2cSimSlot *stk, int *sp,
                   Nvm2cSimSlot *out) {
    if (*sp <= 0) {
        nvm2c_fail(b, "function %u: operand stack underflow", idx);
        return 0;
    }
    *out = stk[--(*sp)];
    return 1;
}

static int merge_kind(Nvm2cBuf *b, uint8_t *dst, uint8_t kind, const char *what) {
    if (kind == NVM2C_VK_UNK) return 1;
    if ((*dst == NVM2C_VK_ARR && (kind == NVM2C_VK_SARR || kind == NVM2C_VK_RARR)) ||
        (kind == NVM2C_VK_ARR && (*dst == NVM2C_VK_SARR || *dst == NVM2C_VK_RARR))) {
        if (*dst == NVM2C_VK_ARR) *dst = kind;
        return 1;
    }
    if (*dst == NVM2C_VK_UNK || *dst == kind) {
        *dst = kind;
        return 1;
    }
    nvm2c_fail(b, "conflicting %s value kinds", what);
    return 0;
}

static int mark_origin(Nvm2cBuf *b, uint8_t *local_kind, uint16_t nloc,
                        int origin, uint8_t kind) {
    if (origin >= 0 && (uint16_t)origin < nloc) {
        return merge_kind(b, &local_kind[origin], kind, "local");
    }
    return 1;
}

static int mark_slot_origin(Nvm2cBuf *b, uint8_t *local_kind, uint8_t *rec_fields,
                            uint16_t nloc, Nvm2cSimSlot slot, uint8_t kind) {
    if (slot.origin >= 0 && (uint16_t)slot.origin < nloc && slot.origin_field >= 0) {
        return merge_kind(b, rec_fields +
                          (size_t)slot.origin * NVM2C_MAX_REC_FIELDS + slot.origin_field,
                          kind, "record field");
    }
    return mark_origin(b, local_kind, nloc, slot.origin, kind);
}

static int mark_slot_str_origin(Nvm2cBuf *b, uint8_t *local_kind,
                                uint8_t *rec_fields, uint16_t nloc,
                                Nvm2cSimSlot slot) {
    return mark_slot_origin(b, local_kind, rec_fields, nloc, slot, NVM2C_VK_STR);
}

static const uint8_t *fn_rec_k_const(const uint8_t *tab, uint32_t fn, uint16_t slot) {
    return tab + ((size_t)fn * NVM2C_MAX_LOCALS + slot) * NVM2C_MAX_REC_FIELDS;
}

static int classify_function(Nvm2cBuf *b, const NvmModule *mod, uint32_t idx,
                              uint8_t *kinds, uint8_t *all_rec_fields,
                              uint8_t *result_kinds, uint8_t *result_fields,
                              int *changed) {
    const NvmFunctionEntry *fn = &mod->functions[idx];
    uint8_t *local_kind = kinds + (size_t)idx * NVM2C_MAX_LOCALS;
    uint8_t *rec_fields = all_rec_fields +
        ((size_t)idx * NVM2C_MAX_LOCALS) * NVM2C_MAX_REC_FIELDS;
    uint16_t nloc = fn->local_count;
    uint16_t i;
    uint8_t old_kind[NVM2C_MAX_LOCALS];
    uint8_t old_rec[NVM2C_MAX_LOCALS * NVM2C_MAX_REC_FIELDS];
    uint8_t old_result_kind = result_kinds[idx];
    uint8_t old_result_fields[NVM2C_MAX_REC_FIELDS];
    memcpy(old_kind, local_kind, nloc);
    memcpy(old_rec, rec_fields, (size_t)nloc * NVM2C_MAX_REC_FIELDS);
    memcpy(old_result_fields, result_fields + (size_t)idx * NVM2C_MAX_REC_FIELDS,
           NVM2C_MAX_REC_FIELDS);

    if (fn->code_offset > mod->code_size ||
        fn->code_length > mod->code_size - fn->code_offset) {
        nvm2c_fail(b, "function %u: code range is outside the module", idx);
        return 0;
    }

    const uint8_t *code = mod->code + fn->code_offset;
    size_t remaining = fn->code_length;
    Nvm2cSimSlot stk[NVM2C_MAX_STACK];
    int sp = 0;
    size_t pc = 0;

    while (pc < remaining) {
        DecodedInstruction ins;
        uint32_t n = isa_decode(code + pc, remaining - pc, &ins);
        if (n == 0) {
            nvm2c_fail(b, "function %u: invalid instruction at offset %zu", idx, pc);
            return 0;
        }
        pc += n;

        switch (ins.opcode) {
        case OP_NOP:
        case OP_JMP:
            break;
        case OP_PUSH_I64:
        case OP_PUSH_BOOL:
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        case OP_PUSH_VOID:
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_VALUE, -1)) return 0;
            break;
        case OP_PUSH_STR:
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_STR, -1)) return 0;
            break;
        case OP_DUP: {
            if (sp <= 0) {
                nvm2c_fail(b, "function %u: DUP on empty stack", idx);
                return 0;
            }
            if (!sim_push_slot(b, idx, stk, &sp, stk[sp - 1])) return 0;
            break;
        }
        case OP_POP: {
            Nvm2cSimSlot dumped;
            if (!sim_pop(b, idx, stk, &sp, &dumped)) return 0;
            (void)dumped;
            break;
        }
        case OP_PRINT:
        case OP_PRINTLN: {
            Nvm2cSimSlot dumped;
            if (!sim_pop(b, idx, stk, &sp, &dumped)) return 0;
            (void)dumped;
            break;
        }
        case OP_ASSERT: {
            Nvm2cSimSlot dumped;
            if (!sim_pop(b, idx, stk, &sp, &dumped)) return 0;
            (void)dumped;
            break;
        }
        case OP_SWAP: {
            Nvm2cSimSlot x, y;
            if (!sim_pop(b, idx, stk, &sp, &x)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &y)) return 0;
            if (!sim_push_slot(b, idx, stk, &sp, x)) return 0;
            if (!sim_push_slot(b, idx, stk, &sp, y)) return 0;
            break;
        }
        case OP_LOAD_LOCAL: {
            uint16_t slot = ins.operands[0].u16;
            Nvm2cSimSlot loaded;
            if (slot >= nloc) {
                nvm2c_fail(b, "function %u: LOAD_LOCAL %u out of range", idx, slot);
                return 0;
            }
            memset(&loaded, 0, sizeof loaded);
            loaded.kind = local_kind[slot];
            loaded.origin = (int)slot;
            loaded.origin_field = -1;
            if (loaded.kind == NVM2C_VK_REC || loaded.kind == NVM2C_VK_RARR) {
                memcpy(loaded.rec_k, rec_fields + (size_t)slot * NVM2C_MAX_REC_FIELDS,
                       NVM2C_MAX_REC_FIELDS);
            }
            if (!sim_push_slot(b, idx, stk, &sp, loaded)) return 0;
            break;
        }
        case OP_STORE_LOCAL: {
            uint16_t slot = ins.operands[0].u16;
            Nvm2cSimSlot v;
            if (slot >= nloc) {
                nvm2c_fail(b, "function %u: STORE_LOCAL %u out of range", idx, slot);
                return 0;
            }
            if (!sim_pop(b, idx, stk, &sp, &v)) return 0;
            if (v.kind == NVM2C_VK_STR) {
                local_kind[slot] = NVM2C_VK_STR;
            } else if (v.kind == NVM2C_VK_ARR) {
                local_kind[slot] = NVM2C_VK_ARR;
            } else if (v.kind == NVM2C_VK_SARR) {
                local_kind[slot] = NVM2C_VK_SARR;
            } else if (v.kind == NVM2C_VK_REC) {
                local_kind[slot] = NVM2C_VK_REC;
                memcpy(rec_fields + (size_t)slot * NVM2C_MAX_REC_FIELDS,
                       v.rec_k, NVM2C_MAX_REC_FIELDS);
            } else if (v.kind == NVM2C_VK_RARR) {
                local_kind[slot] = NVM2C_VK_RARR;
                memcpy(rec_fields + (size_t)slot * NVM2C_MAX_REC_FIELDS,
                       v.rec_k, NVM2C_MAX_REC_FIELDS);
            } else if (v.kind == NVM2C_VK_INT && local_kind[slot] != NVM2C_VK_STR
                       && local_kind[slot] != NVM2C_VK_ARR
                        && local_kind[slot] != NVM2C_VK_SARR
                        && local_kind[slot] != NVM2C_VK_REC
                        && local_kind[slot] != NVM2C_VK_RARR) {
                local_kind[slot] = NVM2C_VK_INT;
            }
            break;
        }
        case OP_ADD:
        case OP_I64_ADD:
        case OP_SUB:
        case OP_I64_SUB:
        case OP_MUL:
        case OP_I64_MUL:
        case OP_DIV:
        case OP_I64_DIV_S:
        case OP_MOD:
        case OP_I64_REM_S:
        case OP_I64_EQ:
        case OP_I64_NE:
        case OP_I64_LT_S:
        case OP_I64_LE_S:
        case OP_I64_GT_S:
        case OP_I64_GE_S:
        case OP_BOOL_AND:
        case OP_BOOL_OR: {
            Nvm2cSimSlot rhs, lhs;
            if (!sim_pop(b, idx, stk, &sp, &rhs)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &lhs)) return 0;
            (void)rhs;
            (void)lhs;
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_NEG:
        case OP_I64_NEG:
        case OP_BOOL_NOT: {
            Nvm2cSimSlot x;
            if (!sim_pop(b, idx, stk, &sp, &x)) return 0;
            (void)x;
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_STR_LEN: {
            Nvm2cSimSlot v;
            if (!sim_pop(b, idx, stk, &sp, &v)) return 0;
            if (!mark_slot_str_origin(b, local_kind, rec_fields, nloc, v)) return 0;
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_STR_CONCAT: {
            Nvm2cSimSlot rhs, lhs;
            if (!sim_pop(b, idx, stk, &sp, &rhs)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &lhs)) return 0;
            if (!mark_slot_str_origin(b, local_kind, rec_fields, nloc, rhs) ||
                !mark_slot_str_origin(b, local_kind, rec_fields, nloc, lhs)) return 0;
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_STR, -1)) return 0;
            break;
        }
        case OP_STR_SUBSTR: {
            Nvm2cSimSlot len, start, s;
            if (!sim_pop(b, idx, stk, &sp, &len)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &start)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &s)) return 0;
            (void)len;
            (void)start;
            if (!mark_slot_str_origin(b, local_kind, rec_fields, nloc, s)) return 0;
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_STR, -1)) return 0;
            break;
        }
        case OP_STR_CONTAINS: {
            Nvm2cSimSlot needle, hay;
            if (!sim_pop(b, idx, stk, &sp, &needle)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &hay)) return 0;
            if (!mark_slot_str_origin(b, local_kind, rec_fields, nloc, needle) ||
                !mark_slot_str_origin(b, local_kind, rec_fields, nloc, hay)) return 0;
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_STR_CHAR_AT: {
            Nvm2cSimSlot ix, s;
            if (!sim_pop(b, idx, stk, &sp, &ix)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &s)) return 0;
            (void)ix;
            if (!mark_slot_str_origin(b, local_kind, rec_fields, nloc, s)) return 0;
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_CAST_STRING: {
            Nvm2cSimSlot v;
            if (!sim_pop(b, idx, stk, &sp, &v)) return 0;
            (void)v;
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_STR, -1)) return 0;
            break;
        }
        case OP_CAST_INT:
        case OP_CAST_BOOL:
        case OP_TYPE_CHECK: {
            Nvm2cSimSlot v;
            if (!sim_pop(b, idx, stk, &sp, &v)) return 0;
            (void)v;
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_EQ:
        case OP_NE: {
            Nvm2cSimSlot rhs, lhs;
            if (!sim_pop(b, idx, stk, &sp, &rhs)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &lhs)) return 0;
            if (lhs.kind != NVM2C_VK_INT || rhs.kind != NVM2C_VK_INT) {
                if (!mark_slot_str_origin(b, local_kind, rec_fields, nloc, lhs) ||
                    !mark_slot_str_origin(b, local_kind, rec_fields, nloc, rhs)) return 0;
            }
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_ARR_LITERAL: {
            uint8_t tag = ins.operands[0].u8;
            uint16_t count = ins.operands[1].u16;
            uint16_t ai;
            for (ai = 0; ai < count; ai++) {
                Nvm2cSimSlot v;
                if (!sim_pop(b, idx, stk, &sp, &v)) return 0;
                (void)v;
            }
            if (tag == TAG_STRING) {
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_SARR, -1)) return 0;
            } else if (tag == TAG_STRUCT) {
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_RARR, -1)) return 0;
            } else {
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_ARR, -1)) return 0;
            }
            break;
        }
        case OP_ARR_NEW: {
            uint8_t tag = ins.operands[0].u8;
            if (tag == TAG_STRING) {
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_SARR, -1)) return 0;
            } else if (tag == TAG_INT) {
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_ARR, -1)) return 0;
            } else if (tag == TAG_STRUCT) {
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_RARR, -1)) return 0;
            } else {
                nvm2c_fail(b, "function %u: ARR_NEW only supports int, string, or record elements", idx);
                return 0;
            }
            break;
        }
        case OP_ARR_LEN: {
            Nvm2cSimSlot v;
            if (!sim_pop(b, idx, stk, &sp, &v)) return 0;
            if (v.kind == NVM2C_VK_RARR) {
                if (!mark_slot_origin(b, local_kind, rec_fields, nloc, v,
                                      NVM2C_VK_RARR)) return 0;
            } else if (v.kind == NVM2C_VK_SARR) {
                if (!mark_slot_origin(b, local_kind, rec_fields, nloc, v,
                                      NVM2C_VK_SARR)) return 0;
            } else {
                if (!mark_slot_origin(b, local_kind, rec_fields, nloc, v,
                                      NVM2C_VK_ARR)) return 0;
            }
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_ARR_GET: {
            Nvm2cSimSlot ix, arr;
            if (!sim_pop(b, idx, stk, &sp, &ix)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &arr)) return 0;
            (void)ix;
            if (arr.kind == NVM2C_VK_RARR) {
                Nvm2cSimSlot rec;
                if (!mark_slot_origin(b, local_kind, rec_fields, nloc, arr,
                                      NVM2C_VK_RARR)) return 0;
                memset(&rec, 0, sizeof rec);
                rec.kind = NVM2C_VK_REC;
                rec.origin = -1;
                rec.origin_field = -1;
                memcpy(rec.rec_k, arr.rec_k, NVM2C_MAX_REC_FIELDS);
                if (!sim_push_slot(b, idx, stk, &sp, rec)) return 0;
            } else if (arr.kind == NVM2C_VK_SARR) {
                if (!mark_slot_origin(b, local_kind, rec_fields, nloc, arr,
                                      NVM2C_VK_SARR)) return 0;
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_VALUE, -1)) return 0;
            } else {
                if (!mark_slot_origin(b, local_kind, rec_fields, nloc, arr,
                                      NVM2C_VK_ARR)) return 0;
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_VALUE, -1)) return 0;
            }
            break;
        }
        case OP_ARR_PUSH: {
            Nvm2cSimSlot val, arr;
            if (!sim_pop(b, idx, stk, &sp, &val)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &arr)) return 0;
            if (val.kind == NVM2C_VK_REC) {
                Nvm2cSimSlot pushed = arr;
                if (!mark_origin(b, local_kind, nloc, arr.origin, NVM2C_VK_RARR)) return 0;
                if (arr.origin >= 0 && (uint16_t)arr.origin < nloc) {
                    memcpy(rec_fields + (size_t)arr.origin * NVM2C_MAX_REC_FIELDS,
                           val.rec_k, NVM2C_MAX_REC_FIELDS);
                }
                pushed.kind = NVM2C_VK_RARR;
                pushed.origin = -1;
                memcpy(pushed.rec_k, val.rec_k, NVM2C_MAX_REC_FIELDS);
                if (!sim_push_slot(b, idx, stk, &sp, pushed)) return 0;
            } else if (val.kind == NVM2C_VK_STR || arr.kind == NVM2C_VK_SARR) {
                if (!mark_origin(b, local_kind, nloc, arr.origin, NVM2C_VK_SARR)) return 0;
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_SARR, -1)) return 0;
            } else {
                if (!mark_origin(b, local_kind, nloc, arr.origin, NVM2C_VK_ARR)) return 0;
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_ARR, -1)) return 0;
            }
            break;
        }
        case OP_AGG_PACK: {
            uint16_t count = ins.operands[3].u16;
            Nvm2cSimSlot packed;
            uint16_t ai;
            memset(&packed, 0, sizeof packed);
            packed.kind = NVM2C_VK_REC;
            packed.origin = -1;
            packed.origin_field = -1;
            if (count > NVM2C_MAX_REC_FIELDS) {
                nvm2c_fail(b, "function %u: AGG_PACK has too many fields", idx);
                return 0;
            }
            for (ai = 0; ai < count; ai++) {
                Nvm2cSimSlot v;
                if (!sim_pop(b, idx, stk, &sp, &v)) return 0;
                if (v.kind != NVM2C_VK_INT && v.kind != NVM2C_VK_STR &&
                    v.kind != NVM2C_VK_ARR && v.kind != NVM2C_VK_SARR &&
                    v.kind != NVM2C_VK_RARR && v.kind != NVM2C_VK_UNK) {
                    nvm2c_fail(b, "function %u: AGG_PACK fields must be scalar or array values", idx);
                    return 0;
                }
                packed.rec_k[count - 1 - ai] = v.kind;
            }
            if (!sim_push_slot(b, idx, stk, &sp, packed)) return 0;
            break;
        }
        case OP_AGG_GET: {
            Nvm2cSimSlot rec;
            uint16_t fi = ins.operands[0].u16;
            uint8_t fk;
            if (!sim_pop(b, idx, stk, &sp, &rec)) return 0;
            if (!mark_origin(b, local_kind, nloc, rec.origin, NVM2C_VK_REC)) return 0;
            if (fi >= NVM2C_MAX_REC_FIELDS) {
                nvm2c_fail(b, "function %u: AGG_GET field is out of range", idx);
                return 0;
            }
            fk = rec.rec_k[fi];
            {
                Nvm2cSimSlot field;
                memset(&field, 0, sizeof field);
                field.kind = fk;
                field.origin = rec.origin;
                field.origin_field = (int)fi;
                if (!sim_push_slot(b, idx, stk, &sp, field)) return 0;
            }
            break;
        }
        case OP_CALL:
        case OP_TAIL_CALL: {
            uint32_t callee = ins.operands[0].u32;
            if (callee >= mod->function_count) {
                nvm2c_fail(b, "function %u: CALL target %u is out of range", idx, callee);
                return 0;
            }
            const NvmFunctionEntry *cf = &mod->functions[callee];
            for (i = 0; i < cf->arity; i++) {
                Nvm2cSimSlot arg;
                if (!sim_pop(b, idx, stk, &sp, &arg)) return 0;
                uint16_t param = (uint16_t)(cf->arity - 1 - i);
                uint8_t *callee_k = kinds + (size_t)callee * NVM2C_MAX_LOCALS;
                if (!merge_kind(b, &callee_k[param], arg.kind, "parameter")) return 0;
                if (arg.kind == NVM2C_VK_REC || arg.kind == NVM2C_VK_RARR) {
                    uint8_t *callee_r = all_rec_fields +
                        ((size_t)callee * NVM2C_MAX_LOCALS + param) *
                        NVM2C_MAX_REC_FIELDS;
                    uint16_t fi;
                    for (fi = 0; fi < NVM2C_MAX_REC_FIELDS; fi++) {
                        if (!merge_kind(b, &callee_r[fi], arg.rec_k[fi], "record field")) return 0;
                    }
                }
            }
            if (ins.opcode == OP_CALL) {
                if (cf->result_count == 1 && cf->result_tag == TAG_STRING) {
                    if (!sim_push(b, idx, stk, &sp, NVM2C_VK_STR, -1)) return 0;
                } else if (cf->result_count == 1 && cf->result_tag == TAG_ARRAY) {
                    Nvm2cSimSlot result;
                    memset(&result, 0, sizeof result);
                    result.kind = result_kinds[callee];
                result.origin = -1;
                result.origin_field = -1;
                    if (result.kind == NVM2C_VK_RARR) {
                        memcpy(result.rec_k, result_fields +
                               (size_t)callee * NVM2C_MAX_REC_FIELDS,
                               NVM2C_MAX_REC_FIELDS);
                    }
                    if (!sim_push_slot(b, idx, stk, &sp, result)) return 0;
                } else if (cf->result_count == 1 &&
                           (cf->result_tag == TAG_STRUCT || cf->result_tag == TAG_UNION)) {
                    Nvm2cSimSlot result;
                    memset(&result, 0, sizeof result);
                    result.kind = NVM2C_VK_REC;
                    result.origin = -1;
                    result.origin_field = -1;
                    memcpy(result.rec_k, result_fields +
                           (size_t)callee * NVM2C_MAX_REC_FIELDS,
                           NVM2C_MAX_REC_FIELDS);
                    if (!sim_push_slot(b, idx, stk, &sp, result)) return 0;
                } else if (result_is_i64(cf)) {
                    if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
                }
            } else if (fn->result_count == 1 && fn->result_tag == TAG_ARRAY) {
                if (!merge_kind(b, &result_kinds[idx], result_kinds[callee],
                                "array result") ||
                    !merge_kind(b, &result_kinds[callee], result_kinds[idx],
                                "array result")) return 0;
            }
            break;
        }
        case OP_JMP_FALSE: {
            Nvm2cSimSlot cond;
            if (!sim_pop(b, idx, stk, &sp, &cond)) return 0;
            (void)cond;
            break;
        }
        case OP_RET:
        case OP_HALT: {
            if (fn->result_count == 1 &&
                (fn->result_tag == TAG_INT || fn->result_tag == TAG_BOOL ||
                 fn->result_tag == TAG_STRING || fn->result_tag == TAG_ARRAY) &&
                sp > 0) {
                Nvm2cSimSlot v;
                if (!sim_pop(b, idx, stk, &sp, &v)) return 0;
                if (fn->result_tag == TAG_STRING) {
                    if (v.kind != NVM2C_VK_STR && v.kind != NVM2C_VK_VALUE) {
                        nvm2c_fail(b, "function %u: string return requires a string value", idx);
                        return 0;
                    }
                    if (v.kind == NVM2C_VK_STR &&
                        !mark_slot_str_origin(b, local_kind, rec_fields, nloc, v)) return 0;
                } else if (fn->result_tag == TAG_ARRAY) {
                    uint16_t fi;
                    if (v.kind != NVM2C_VK_ARR && v.kind != NVM2C_VK_SARR &&
                        v.kind != NVM2C_VK_RARR && v.kind != NVM2C_VK_UNK) {
                        nvm2c_fail(b, "function %u: array return requires an array value", idx);
                        return 0;
                    }
                    if (!merge_kind(b, &result_kinds[idx], v.kind, "array result") ||
                        !mark_origin(b, local_kind, nloc, v.origin,
                                     result_kinds[idx])) return 0;
                    if (v.kind == NVM2C_VK_RARR) {
                        for (fi = 0; fi < NVM2C_MAX_REC_FIELDS; fi++) {
                            if (!merge_kind(b, result_fields +
                                            (size_t)idx * NVM2C_MAX_REC_FIELDS + fi,
                                            v.rec_k[fi], "record field")) return 0;
                        }
                    }
                } else if (fn->result_tag == TAG_INT || fn->result_tag == TAG_BOOL) {
                    if (v.kind == NVM2C_VK_STR) {
                        nvm2c_fail(b, "function %u: integer return cannot use a string value", idx);
                        return 0;
                    }
                } else if (fn->result_tag == TAG_STRUCT || fn->result_tag == TAG_UNION) {
                    uint16_t fi;
                    if (v.kind != NVM2C_VK_REC) {
                        nvm2c_fail(b, "function %u: aggregate return requires a record value", idx);
                        return 0;
                    }
                    for (fi = 0; fi < NVM2C_MAX_REC_FIELDS; fi++) {
                        if (!merge_kind(b, result_fields +
                                        (size_t)idx * NVM2C_MAX_REC_FIELDS + fi,
                                        v.rec_k[fi], "record field")) return 0;
                    }
                }
            }
            break;
        }
        case OP_CALL_EXTERN:
            nvm2c_fail(b, "CALL_EXTERN is the VM FFI/co-process path; nvm2c does not emit it");
            return 0;
        default: {
            const InstructionInfo *info = isa_get_info(ins.opcode);
            nvm2c_fail(b, "unsupported opcode %s (0x%02X) in the nvm2c classifier",
                       info ? info->name : "UNKNOWN", ins.opcode);
            return 0;
        }
        }
        if (b->failed) return 0;
    }

    if (memcmp(old_kind, local_kind, nloc) != 0 ||
        memcmp(old_rec, rec_fields, (size_t)nloc * NVM2C_MAX_REC_FIELDS) != 0 ||
        old_result_kind != result_kinds[idx] ||
        memcmp(old_result_fields,
               result_fields + (size_t)idx * NVM2C_MAX_REC_FIELDS,
               NVM2C_MAX_REC_FIELDS) != 0) {
        *changed = 1;
    }
    return 1;
}

static void emit_prototype(Nvm2cBuf *b, const NvmModule *mod, uint32_t idx,
                            const uint8_t *kinds, const uint8_t *result_kinds) {
    const NvmFunctionEntry *fn = &mod->functions[idx];
    const char *rt = c_result_type(fn, result_kinds[idx]);
    if (!rt) {
        nvm2c_fail(b, "function %u: only void or a single supported value result is supported",
                   idx);
        return;
    }
    if (fn->upvalue_count != 0) {
        nvm2c_fail(b, "function %u: upvalues are not in the nvm2c subset", idx);
        return;
    }
    if (fn->local_count > NVM2C_MAX_LOCALS) {
        nvm2c_fail(b, "function %u: too many locals", idx);
        return;
    }
    if (fn->arity > fn->local_count) {
        nvm2c_fail(b, "function %u: arity exceeds local_count", idx);
        return;
    }
    char name[64];
    fn_c_name(mod, idx, name, sizeof name);
    nvm2c_printf(b, "static %s %s(", rt, name);
    if (fn->arity == 0) {
        nvm2c_puts(b, "void");
    } else {
        uint16_t i;
        for (i = 0; i < fn->arity; i++) {
            if (i) nvm2c_puts(b, ", ");
            nvm2c_printf(b, "%s a%u, uint8_t av%u",
                         c_local_type(fn_local_kind(kinds, idx, i)), (unsigned)i,
                         (unsigned)i);
        }
    }
    nvm2c_puts(b, ");\n");
}

typedef struct {
    int slots[NVM2C_MAX_STACK];
    int tags[NVM2C_MAX_STACK];
    uint8_t kinds[NVM2C_MAX_STACK];
    uint8_t rec_k[NVM2C_MAX_TEMPS][NVM2C_MAX_REC_FIELDS];
    int sp;
    int next_temp;
    int next_str;
    int next_arr;
    int next_sarr;
    int next_rec;
    int next_rarr;
    int next_tag;
    int next_value;
} Nvm2cStack;

static int stack_push_tag(Nvm2cBuf *b, Nvm2cStack *st, const char *rhs) {
    if (st->next_tag >= NVM2C_MAX_TEMPS) {
        nvm2c_fail(b, "too many value-tag temporaries");
        return -1;
    }
    int tag = st->next_tag++;
    nvm2c_printf(b, "    vt[%d] = %s;\n", tag, rhs);
    st->tags[st->sp] = tag;
    return tag;
}

static int stack_push_temp(Nvm2cBuf *b, Nvm2cStack *st, const char *rhs) {
    if (st->sp >= NVM2C_MAX_STACK) {
        nvm2c_fail(b, "operand stack overflow");
        return -1;
    }
    if (st->next_temp >= NVM2C_MAX_TEMPS) {
        nvm2c_fail(b, "too many temporaries");
        return -1;
    }
    int t = st->next_temp++;
    nvm2c_printf(b, "    t[%d] = %s;\n", t, rhs);
    if (stack_push_tag(b, st, "TAG_INT") < 0) return -1;
    st->slots[st->sp] = t;
    st->kinds[st->sp] = NVM2C_VK_INT;
    st->sp++;
    return t;
}

static int stack_push_str(Nvm2cBuf *b, Nvm2cStack *st, const char *rhs) {
    if (st->sp >= NVM2C_MAX_STACK) {
        nvm2c_fail(b, "operand stack overflow");
        return -1;
    }
    if (st->next_str >= NVM2C_MAX_TEMPS) {
        nvm2c_fail(b, "too many string temporaries");
        return -1;
    }
    int s = st->next_str++;
    nvm2c_printf(b, "    s[%d] = %s;\n", s, rhs);
    if (stack_push_tag(b, st, "TAG_STRING") < 0) return -1;
    st->slots[st->sp] = s;
    st->kinds[st->sp] = NVM2C_VK_STR;
    st->sp++;
    return s;
}

static int stack_push_arr(Nvm2cBuf *b, Nvm2cStack *st, const char *rhs) {
    if (st->sp >= NVM2C_MAX_STACK) {
        nvm2c_fail(b, "operand stack overflow");
        return -1;
    }
    if (st->next_arr >= NVM2C_MAX_TEMPS) {
        nvm2c_fail(b, "too many array temporaries");
        return -1;
    }
    int a = st->next_arr++;
    nvm2c_printf(b, "    a[%d] = %s;\n", a, rhs);
    if (stack_push_tag(b, st, "TAG_ARRAY") < 0) return -1;
    st->slots[st->sp] = a;
    st->kinds[st->sp] = NVM2C_VK_ARR;
    st->sp++;
    return a;
}

static int stack_push_sarr(Nvm2cBuf *b, Nvm2cStack *st, const char *rhs) {
    if (st->sp >= NVM2C_MAX_STACK) {
        nvm2c_fail(b, "operand stack overflow");
        return -1;
    }
    if (st->next_sarr >= NVM2C_MAX_TEMPS) {
        nvm2c_fail(b, "too many string-array temporaries");
        return -1;
    }
    int a = st->next_sarr++;
    nvm2c_printf(b, "    sa[%d] = %s;\n", a, rhs);
    if (stack_push_tag(b, st, "TAG_ARRAY") < 0) return -1;
    st->slots[st->sp] = a;
    st->kinds[st->sp] = NVM2C_VK_SARR;
    st->sp++;
    return a;
}

static int stack_push_rec(Nvm2cBuf *b, Nvm2cStack *st, const char *rhs) {
    if (st->sp >= NVM2C_MAX_STACK) {
        nvm2c_fail(b, "operand stack overflow");
        return -1;
    }
    if (st->next_rec >= NVM2C_MAX_TEMPS) {
        nvm2c_fail(b, "too many record temporaries");
        return -1;
    }
    int r = st->next_rec++;
    nvm2c_printf(b, "    r[%d] = %s;\n", r, rhs);
    if (stack_push_tag(b, st, "TAG_STRUCT") < 0) return -1;
    st->slots[st->sp] = r;
    st->kinds[st->sp] = NVM2C_VK_REC;
    st->sp++;
    return r;
}

static int stack_push_rarr(Nvm2cBuf *b, Nvm2cStack *st, const char *rhs) {
    if (st->sp >= NVM2C_MAX_STACK || st->next_rarr >= NVM2C_MAX_TEMPS) {
        nvm2c_fail(b, "too many record-array temporaries");
        return -1;
    }
    int a = st->next_rarr++;
    nvm2c_printf(b, "    ra[%d] = %s;\n", a, rhs);
    if (stack_push_tag(b, st, "TAG_ARRAY") < 0) return -1;
    st->slots[st->sp] = a;
    st->kinds[st->sp] = NVM2C_VK_RARR;
    st->sp++;
    return a;
}

static int stack_push_value(Nvm2cBuf *b, Nvm2cStack *st, const char *rhs) {
    if (st->sp >= NVM2C_MAX_STACK || st->next_value >= NVM2C_MAX_TEMPS) {
        nvm2c_fail(b, "too many tagged-value temporaries");
        return -1;
    }
    int v = st->next_value++;
    nvm2c_printf(b, "    v[%d] = %s;\n", v, rhs);
    if (stack_push_tag(b, st, "TAG_VOID") < 0) return -1;
    nvm2c_printf(b, "    vt[%d] = v[%d].tag;\n", st->tags[st->sp], v);
    st->slots[st->sp] = v;
    st->kinds[st->sp] = NVM2C_VK_VALUE;
    st->sp++;
    return v;
}

static int stack_pop_kind(Nvm2cBuf *b, Nvm2cStack *st, uint8_t *kind_out) {
    if (st->sp <= 0) {
        nvm2c_fail(b, "operand stack underflow");
        return -1;
    }
    st->sp--;
    if (kind_out) *kind_out = st->kinds[st->sp];
    return st->slots[st->sp];
}

static int stack_pop_value(Nvm2cBuf *b, Nvm2cStack *st, uint8_t *kind_out,
                           int *tag_out) {
    int slot = stack_pop_kind(b, st, kind_out);
    if (!b->failed && tag_out) *tag_out = st->tags[st->sp];
    return slot;
}

static int stack_pop(Nvm2cBuf *b, Nvm2cStack *st) {
    return stack_pop_kind(b, st, NULL);
}

static int stack_pop_expect(Nvm2cBuf *b, Nvm2cStack *st, uint8_t kind, const char *what) {
    uint8_t got = NVM2C_VK_INT;
    int slot = stack_pop_kind(b, st, &got);
    if (b->failed) return -1;
    if (got == NVM2C_VK_VALUE && kind == NVM2C_VK_INT) {
        if (st->next_temp >= NVM2C_MAX_TEMPS) {
            nvm2c_fail(b, "too many temporaries");
            return -1;
        }
        int t = st->next_temp++;
        nvm2c_printf(b, "    t[%d] = nvalue_as_int(v[%d]);\n", t, slot);
        return t;
    }
    if (got == NVM2C_VK_VALUE && kind == NVM2C_VK_STR) {
        if (st->next_str >= NVM2C_MAX_TEMPS) {
            nvm2c_fail(b, "too many string temporaries");
            return -1;
        }
        int s = st->next_str++;
        nvm2c_printf(b, "    s[%d] = nvalue_as_string(v[%d]);\n", s, slot);
        return s;
    }
    if (got != kind) {
        const char *want = "int";
        if (kind == NVM2C_VK_STR) want = "string";
        else if (kind == NVM2C_VK_ARR) want = "array";
        else if (kind == NVM2C_VK_SARR) want = "string array";
        else if (kind == NVM2C_VK_REC) want = "record";
        else if (kind == NVM2C_VK_RARR) want = "record array";
        nvm2c_fail(b, "%s: expected %s value", what, want);
        return -1;
    }
    return slot;
}

static void emit_binop(Nvm2cBuf *b, Nvm2cStack *st, const char *op) {
    int rhs = stack_pop_expect(b, st, NVM2C_VK_INT, "binary op rhs");
    int rhs_tag = st->tags[st->sp];
    int lhs = stack_pop_expect(b, st, NVM2C_VK_INT, "binary op lhs");
    int lhs_tag = st->tags[st->sp];
    if (b->failed) return;
    nvm2c_printf(b, "    if (vt[%d] != TAG_INT || vt[%d] != TAG_INT) abort();\n",
                 lhs_tag, rhs_tag);
    char expr[80];
    snprintf(expr, sizeof expr, "(t[%d] %s t[%d])", lhs, op, rhs);
    stack_push_temp(b, st, expr);
}

static void emit_unop(Nvm2cBuf *b, Nvm2cStack *st, const char *prefix) {
    int x = stack_pop_expect(b, st, NVM2C_VK_INT, "unary op");
    int tag = st->tags[st->sp];
    if (b->failed) return;
    nvm2c_printf(b, "    if (vt[%d] != TAG_INT) abort();\n", tag);
    char expr[64];
    snprintf(expr, sizeof expr, "(%s t[%d])", prefix, x);
    stack_push_temp(b, st, expr);
}

static void emit_bool_binop(Nvm2cBuf *b, Nvm2cStack *st, const char *op) {
    int rhs = stack_pop_expect(b, st, NVM2C_VK_INT, "boolean op rhs");
    int rhs_tag = st->tags[st->sp];
    int lhs = stack_pop_expect(b, st, NVM2C_VK_INT, "boolean op lhs");
    int lhs_tag = st->tags[st->sp];
    if (b->failed) return;
    nvm2c_printf(b, "    if (vt[%d] != TAG_BOOL || vt[%d] != TAG_BOOL) abort();\n",
                 lhs_tag, rhs_tag);
    char expr[80];
    snprintf(expr, sizeof expr, "(t[%d] %s t[%d])", lhs, op, rhs);
    int t = stack_push_temp(b, st, expr);
    if (t >= 0) nvm2c_printf(b, "    vt[%d] = TAG_BOOL;\n", st->tags[st->sp - 1]);
}

static void emit_bool_unop(Nvm2cBuf *b, Nvm2cStack *st) {
    int x = stack_pop_expect(b, st, NVM2C_VK_INT, "boolean op");
    int tag = st->tags[st->sp];
    if (b->failed) return;
    nvm2c_printf(b, "    if (vt[%d] != TAG_BOOL) abort();\n", tag);
    char expr[64];
    snprintf(expr, sizeof expr, "(!t[%d])", x);
    int t = stack_push_temp(b, st, expr);
    if (t >= 0) nvm2c_printf(b, "    vt[%d] = TAG_BOOL;\n", st->tags[st->sp - 1]);
}

static void stack_keep_high_water(Nvm2cStack *st, const Nvm2cStack *other) {
    if (other->next_temp > st->next_temp) st->next_temp = other->next_temp;
    if (other->next_str > st->next_str) st->next_str = other->next_str;
    if (other->next_arr > st->next_arr) st->next_arr = other->next_arr;
    if (other->next_sarr > st->next_sarr) st->next_sarr = other->next_sarr;
    if (other->next_rec > st->next_rec) st->next_rec = other->next_rec;
    if (other->next_rarr > st->next_rarr) st->next_rarr = other->next_rarr;
    if (other->next_tag > st->next_tag) st->next_tag = other->next_tag;
    if (other->next_value > st->next_value) st->next_value = other->next_value;
}

static int record_join(Nvm2cBuf *b, uint32_t idx, Nvm2cStack *joins, uint8_t *set,
                       size_t tgt, const Nvm2cStack *st) {
    int i;
    if (!set[tgt]) {
        joins[tgt] = *st;
        set[tgt] = 1;
        return 1;
    }
    if (joins[tgt].sp != st->sp) {
        nvm2c_fail(b, "function %u: join at %zu has stack height %d, incoming %d",
                   idx, tgt, joins[tgt].sp, st->sp);
        return 0;
    }
    for (i = 0; i < st->sp; i++) {
        if (joins[tgt].kinds[i] != st->kinds[i]) {
            nvm2c_fail(b, "function %u: join at %zu has a value-kind mismatch", idx, tgt);
            return 0;
        }
        if (joins[tgt].tags[i] != st->tags[i]) {
            nvm2c_printf(b, "    vt[%d] = vt[%d];\n",
                         joins[tgt].tags[i], st->tags[i]);
        }
        if (joins[tgt].slots[i] == st->slots[i]) continue;
        if (st->kinds[i] == NVM2C_VK_STR) {
            nvm2c_printf(b, "    s[%d] = s[%d];\n", joins[tgt].slots[i], st->slots[i]);
        } else if (st->kinds[i] == NVM2C_VK_ARR) {
            nvm2c_printf(b, "    a[%d] = a[%d];\n", joins[tgt].slots[i], st->slots[i]);
        } else if (st->kinds[i] == NVM2C_VK_SARR) {
            nvm2c_printf(b, "    sa[%d] = sa[%d];\n", joins[tgt].slots[i], st->slots[i]);
        } else if (st->kinds[i] == NVM2C_VK_REC) {
            nvm2c_printf(b, "    r[%d] = r[%d];\n", joins[tgt].slots[i], st->slots[i]);
            memcpy(joins[tgt].rec_k[joins[tgt].slots[i]],
                    st->rec_k[st->slots[i]], NVM2C_MAX_REC_FIELDS);
        } else if (st->kinds[i] == NVM2C_VK_RARR) {
            nvm2c_printf(b, "    ra[%d] = ra[%d];\n", joins[tgt].slots[i], st->slots[i]);
            memcpy(joins[tgt].rec_k[joins[tgt].slots[i]],
                   st->rec_k[st->slots[i]], NVM2C_MAX_REC_FIELDS);
        } else if (st->kinds[i] == NVM2C_VK_VALUE) {
            nvm2c_printf(b, "    v[%d] = v[%d];\n", joins[tgt].slots[i], st->slots[i]);
        } else {
            nvm2c_printf(b, "    t[%d] = t[%d];\n", joins[tgt].slots[i], st->slots[i]);
        }
    }
    stack_keep_high_water(&joins[tgt], st);
    return 1;
}

static void stack_restore_join(Nvm2cStack *st, const Nvm2cStack *join) {
    Nvm2cStack cur = *st;
    *st = *join;
    stack_keep_high_water(st, &cur);
}

static int jump_target(Nvm2cBuf *b, uint32_t idx, size_t start, int32_t rel,
                       size_t remaining, size_t *out) {
    int64_t tgt = (int64_t)start + (int64_t)rel;
    if (tgt < 0 || (uint64_t)tgt > (uint64_t)remaining) {
        nvm2c_fail(b, "function %u: jump at offset %zu is out of range", idx, start);
        return 0;
    }
    *out = (size_t)tgt;
    return 1;
}

static int build_direct_call(Nvm2cBuf *b, Nvm2cStack *st, const NvmModule *mod,
                              uint32_t idx, uint32_t callee, const uint8_t *kinds,
                              const uint8_t *result_kinds,
                              char *call, size_t call_sz) {
    if (callee >= mod->function_count) {
        nvm2c_fail(b, "function %u: CALL target %u is out of range", idx, callee);
        return 0;
    }
    const NvmFunctionEntry *cf = &mod->functions[callee];
    if (c_result_type(cf, result_kinds[callee]) == NULL) {
        nvm2c_fail(b, "function %u: CALL target %u has an unsupported result", idx, callee);
        return 0;
    }
    int args[NVM2C_MAX_LOCALS];
    int argv[NVM2C_MAX_LOCALS];
    uint8_t argk[NVM2C_MAX_LOCALS];
    int i;
    for (i = (int)cf->arity - 1; i >= 0; i--) {
        uint8_t pk = fn_local_kind(kinds, callee, (uint16_t)i);
        uint8_t got = NVM2C_VK_INT;
        argk[i] = pk;
        args[i] = stack_pop_value(b, st, &got, &argv[i]);
        if (!b->failed && got != pk) {
            nvm2c_fail(b, "CALL argument: value-kind mismatch");
        }
        if (b->failed) return 0;
    }
    char cname[64];
    fn_c_name(mod, callee, cname, sizeof cname);
    size_t pos = 0;
    pos += (size_t)snprintf(call + pos, call_sz - pos, "%s(", cname);
    for (uint16_t a = 0; a < cf->arity; a++) {
        if (a) pos += (size_t)snprintf(call + pos, call_sz - pos, ", ");
        if (argk[a] == NVM2C_VK_STR) {
            pos += (size_t)snprintf(call + pos, call_sz - pos, "s[%d]", args[a]);
        } else if (argk[a] == NVM2C_VK_ARR) {
            pos += (size_t)snprintf(call + pos, call_sz - pos, "a[%d]", args[a]);
        } else if (argk[a] == NVM2C_VK_SARR) {
            pos += (size_t)snprintf(call + pos, call_sz - pos, "sa[%d]", args[a]);
        } else if (argk[a] == NVM2C_VK_REC) {
            pos += (size_t)snprintf(call + pos, call_sz - pos, "r[%d]", args[a]);
        } else if (argk[a] == NVM2C_VK_RARR) {
            pos += (size_t)snprintf(call + pos, call_sz - pos, "ra[%d]", args[a]);
        } else {
            pos += (size_t)snprintf(call + pos, call_sz - pos, "t[%d]", args[a]);
        }
        pos += (size_t)snprintf(call + pos, call_sz - pos, ", vt[%d]", argv[a]);
        if (pos >= call_sz) {
            nvm2c_fail(b, "function %u: CALL argument list overflow", idx);
            return 0;
        }
    }
    snprintf(call + pos, call_sz - pos, ")");
    return 1;
}

static void emit_function_body(Nvm2cBuf *b, const NvmModule *mod, uint32_t idx,
                               const uint8_t *kinds, const uint8_t *rec_fields,
                                const uint8_t *result_kinds,
                                const uint8_t *result_fields) {
    const NvmFunctionEntry *fn = &mod->functions[idx];
    const char *rt = c_result_type(fn, result_kinds[idx]);
    if (!rt || b->failed) return;

    char name[64];
    uint16_t i;
    fn_c_name(mod, idx, name, sizeof name);
    nvm2c_printf(b, "static %s %s(", rt, name);
    if (fn->arity == 0) {
        nvm2c_puts(b, "void");
    } else {
        for (i = 0; i < fn->arity; i++) {
            if (i) nvm2c_puts(b, ", ");
            nvm2c_printf(b, "%s a%u, uint8_t av%u",
                         c_local_type(fn_local_kind(kinds, idx, i)), (unsigned)i,
                         (unsigned)i);
        }
    }
    nvm2c_puts(b, ") {\n");

    for (i = 0; i < fn->local_count; i++) {
        uint8_t lk = fn_local_kind(kinds, idx, i);
        if (i < fn->arity) {
            nvm2c_printf(b, "    %s l%u = a%u;\n", c_local_type(lk), (unsigned)i, (unsigned)i);
        } else if (lk == NVM2C_VK_STR) {
            nvm2c_printf(b, "    const char *l%u = \"\";\n", (unsigned)i);
        } else if (lk == NVM2C_VK_ARR) {
            nvm2c_printf(b, "    narr_t l%u = {0};\n", (unsigned)i);
        } else if (lk == NVM2C_VK_SARR) {
            nvm2c_printf(b, "    nsarr_t l%u = {0};\n", (unsigned)i);
        } else if (lk == NVM2C_VK_REC) {
            nvm2c_printf(b, "    nrec_t l%u = {0};\n", (unsigned)i);
        } else if (lk == NVM2C_VK_RARR) {
            nvm2c_printf(b, "    nrarr_t l%u = {0};\n", (unsigned)i);
        } else {
            nvm2c_printf(b, "    int64_t l%u = 0;\n", (unsigned)i);
        }
        if (i < fn->arity) {
            nvm2c_printf(b, "    uint8_t lt%u = av%u;\n", (unsigned)i, (unsigned)i);
        } else {
            nvm2c_printf(b, "    uint8_t lt%u = TAG_VOID;\n", (unsigned)i);
        }
        nvm2c_printf(b, "    (void)l%u;\n", (unsigned)i);
    }
    nvm2c_printf(b, "    int64_t t[%d] = {0};\n", NVM2C_MAX_TEMPS);
    nvm2c_puts(b, "    (void)t;\n");
    nvm2c_printf(b, "    const char *s[%d] = {0};\n", NVM2C_MAX_TEMPS);
    nvm2c_puts(b, "    (void)s;\n");
    nvm2c_printf(b, "    narr_t a[%d] = {0};\n", NVM2C_MAX_TEMPS);
    nvm2c_puts(b, "    (void)a;\n");
    nvm2c_printf(b, "    nsarr_t sa[%d] = {0};\n", NVM2C_MAX_TEMPS);
    nvm2c_puts(b, "    (void)sa;\n");
    /* Reserve a fixed-width count, then patch it with this function's high
     * water mark after translation. This keeps record storage automatic and
     * recursion-safe without charging every call for the global limit. */
    size_t rec_count_offset = b->len + strlen("    nrec_t r[");
    nvm2c_printf(b, "    nrec_t r[%03d] = {0};\n", NVM2C_MAX_TEMPS);
    nvm2c_puts(b, "    (void)r;\n");
    nvm2c_printf(b, "    nrarr_t ra[%d] = {0};\n", NVM2C_MAX_TEMPS);
    nvm2c_puts(b, "    (void)ra;\n");
    nvm2c_printf(b, "    uint8_t vt[%d] = {0};\n", NVM2C_MAX_TEMPS);
    nvm2c_puts(b, "    (void)vt;\n");
    nvm2c_printf(b, "    nvalue_t v[%d] = {0};\n", NVM2C_MAX_TEMPS);
    nvm2c_puts(b, "    (void)v;\n");

    if (fn->code_offset > mod->code_size ||
        fn->code_length > mod->code_size - fn->code_offset) {
        nvm2c_fail(b, "function %u: code range is outside the module", idx);
        return;
    }

    const uint8_t *code = mod->code + fn->code_offset;
    size_t remaining = fn->code_length;
    uint8_t *is_start = calloc(remaining + 1, 1);
    uint8_t *is_target = calloc(remaining + 1, 1);
    Nvm2cStack *joins = NULL;
    uint8_t *join_set = NULL;
    if (!is_start || !is_target) {
        nvm2c_fail(b, "out of memory");
        goto done;
    }

    size_t scan = 0;
    while (scan < remaining) {
        is_start[scan] = 1;
        DecodedInstruction look;
        uint32_t n = isa_decode(code + scan, remaining - scan, &look);
        if (n == 0) {
            nvm2c_fail(b, "function %u: invalid instruction at offset %zu", idx, scan);
            goto done;
        }
        if (look.opcode == OP_JMP || look.opcode == OP_JMP_FALSE) {
            size_t tgt = 0;
            if (!jump_target(b, idx, scan, look.operands[0].i32, remaining, &tgt)) {
                goto done;
            }
            is_target[tgt] = 1;
        }
        scan += n;
    }
    is_start[remaining] = 1;
    {
        size_t off;
        for (off = 0; off <= remaining; off++) {
            if (is_target[off] && !is_start[off]) {
                nvm2c_fail(b, "function %u: jump targets a non-instruction boundary at %zu",
                           idx, off);
                goto done;
            }
        }
    }

    joins = calloc(remaining + 1, sizeof(Nvm2cStack));
    join_set = calloc(remaining + 1, 1);
    if (!joins || !join_set) {
        nvm2c_fail(b, "out of memory");
        goto done;
    }

    size_t pc = 0;
    Nvm2cStack st;
    memset(&st, 0, sizeof st);
    int terminated = 0;

    while (pc < remaining) {
        size_t start = pc;
        DecodedInstruction ins;
        uint32_t n = isa_decode(code + pc, remaining - pc, &ins);
        if (n == 0) {
            nvm2c_fail(b, "function %u: invalid instruction at offset %zu", idx, pc);
            goto done;
        }
        if (terminated && !is_target[start]) {
            pc += n;
            continue;
        }
        if (is_target[start]) {
            if (terminated) {
                if (!join_set[start]) {
                    nvm2c_fail(b, "function %u: label at %zu has no incoming stack",
                               idx, start);
                    goto done;
                }
                stack_restore_join(&st, &joins[start]);
                terminated = 0;
                nvm2c_printf(b, "L_%zu: ;\n", start);
            } else {
                if (!record_join(b, idx, joins, join_set, start, &st)) goto done;
                stack_restore_join(&st, &joins[start]);
                nvm2c_printf(b, "L_%zu: ;\n", start);
            }
        }
        pc += n;

        switch (ins.opcode) {
        case OP_NOP:
            break;
        case OP_PUSH_I64: {
            char rhs[32];
            snprintf(rhs, sizeof rhs, "%lldLL", (long long)ins.operands[0].i64);
            stack_push_temp(b, &st, rhs);
            break;
        }
        case OP_PUSH_BOOL: {
            char rhs[8];
            snprintf(rhs, sizeof rhs, "%dLL", ins.operands[0].u8 ? 1 : 0);
            stack_push_temp(b, &st, rhs);
            if (!b->failed) nvm2c_printf(b, "    vt[%d] = TAG_BOOL;\n", st.tags[st.sp - 1]);
            break;
        }
        case OP_PUSH_VOID:
            stack_push_value(b, &st, "nvalue_void()");
            break;
        case OP_PUSH_STR: {
            uint32_t sidx = ins.operands[0].u32;
            const char *lit = nvm_get_string(mod, sidx);
            uint32_t slen = nvm_get_string_len(mod, sidx);
            if (!lit) {
                nvm2c_fail(b, "function %u: PUSH_STR string index %u is out of range", idx, sidx);
                goto done;
            }
            if (st.sp >= NVM2C_MAX_STACK) {
                nvm2c_fail(b, "operand stack overflow");
                goto done;
            }
            if (st.next_str >= NVM2C_MAX_TEMPS) {
                nvm2c_fail(b, "too many string temporaries");
                goto done;
            }
            {
                int slot = st.next_str++;
                nvm2c_printf(b, "    s[%d] = ", slot);
                emit_c_string_lit(b, lit, slen);
                if (b->failed) goto done;
                nvm2c_puts(b, ";\n");
                if (stack_push_tag(b, &st, "TAG_STRING") < 0) goto done;
                st.slots[st.sp] = slot;
                st.kinds[st.sp] = NVM2C_VK_STR;
                st.sp++;
            }
            break;
        }
        case OP_DUP: {
            if (st.sp <= 0) {
                nvm2c_fail(b, "function %u: DUP on empty stack", idx);
                goto done;
            }
            {
                int src = st.slots[st.sp - 1];
                uint8_t k = st.kinds[st.sp - 1];
                char rhs[32];
                if (k == NVM2C_VK_STR) {
                    snprintf(rhs, sizeof rhs, "s[%d]", src);
                    stack_push_str(b, &st, rhs);
                } else if (k == NVM2C_VK_ARR) {
                    snprintf(rhs, sizeof rhs, "a[%d]", src);
                    stack_push_arr(b, &st, rhs);
                } else if (k == NVM2C_VK_SARR) {
                    snprintf(rhs, sizeof rhs, "sa[%d]", src);
                    stack_push_sarr(b, &st, rhs);
                } else if (k == NVM2C_VK_REC) {
                    snprintf(rhs, sizeof rhs, "r[%d]", src);
                    {
                        int nr = stack_push_rec(b, &st, rhs);
                        if (nr >= 0) {
                            memcpy(st.rec_k[nr], st.rec_k[src], NVM2C_MAX_REC_FIELDS);
                        }
                    }
                } else if (k == NVM2C_VK_RARR) {
                    snprintf(rhs, sizeof rhs, "ra[%d]", src);
                    {
                        int nr = stack_push_rarr(b, &st, rhs);
                        if (nr >= 0) memcpy(st.rec_k[nr], st.rec_k[src], NVM2C_MAX_REC_FIELDS);
                    }
                } else if (k == NVM2C_VK_VALUE) {
                    snprintf(rhs, sizeof rhs, "v[%d]", src);
                    stack_push_value(b, &st, rhs);
                } else {
                    snprintf(rhs, sizeof rhs, "t[%d]", src);
                    stack_push_temp(b, &st, rhs);
                }
                if (!b->failed) {
                    nvm2c_printf(b, "    vt[%d] = vt[%d];\n",
                                 st.tags[st.sp - 1], st.tags[st.sp - 2]);
                }
            }
            break;
        }
        case OP_POP:
            (void)stack_pop(b, &st);
            break;
        case OP_PRINT:
        case OP_PRINTLN: {
            uint8_t k = NVM2C_VK_INT;
            int is_void = -1;
            int slot = stack_pop_value(b, &st, &k, &is_void);
            int nl = (ins.opcode == OP_PRINTLN);
            if (b->failed) goto done;
            nvm2c_printf(b, "    if (vt[%d] == TAG_VOID) fputs(\"void\", stdout); else ", is_void);
            if (k == NVM2C_VK_INT) {
                if (nl) {
                    nvm2c_printf(b, "printf(\"%%lld\", (long long)t[%d]);\n", slot);
                } else {
                    nvm2c_printf(b, "printf(\"%%lld\", (long long)t[%d]);\n", slot);
                }
            } else if (k == NVM2C_VK_STR) {
                nvm2c_printf(b, "fputs(s[%d] ? s[%d] : \"\", stdout);\n", slot, slot);
            } else {
                nvm2c_fail(b, "function %u: PRINT of arrays and records is refused", idx);
                goto done;
            }
            if (nl) nvm2c_puts(b, "    fputc('\\n', stdout);\n");
            nvm2c_puts(b, "    fflush(stdout);\n");
            break;
        }
        case OP_ASSERT: {
            int cond = stack_pop_expect(b, &st, NVM2C_VK_INT, "ASSERT");
            if (b->failed) goto done;
            nvm2c_printf(b, "    if (!t[%d]) abort();\n", cond);
            break;
        }
        case OP_SWAP: {
            uint8_t kx = 0, ky = 0;
            int vx = -1, vy = -1;
            int x = stack_pop_value(b, &st, &kx, &vx);
            int y = stack_pop_value(b, &st, &ky, &vy);
            if (b->failed) goto done;
            st.slots[st.sp] = x;
            st.kinds[st.sp] = kx;
            st.tags[st.sp] = vx;
            st.sp++;
            st.slots[st.sp] = y;
            st.kinds[st.sp] = ky;
            st.tags[st.sp] = vy;
            st.sp++;
            break;
        }
        case OP_LOAD_LOCAL: {
            uint16_t slot = ins.operands[0].u16;
            if (slot >= fn->local_count) {
                nvm2c_fail(b, "function %u: LOAD_LOCAL %u out of range", idx, slot);
                goto done;
            }
            char rhs[32];
            snprintf(rhs, sizeof rhs, "l%u", (unsigned)slot);
            if (fn_local_kind(kinds, idx, slot) == NVM2C_VK_STR) {
                stack_push_str(b, &st, rhs);
            } else if (fn_local_kind(kinds, idx, slot) == NVM2C_VK_ARR) {
                stack_push_arr(b, &st, rhs);
            } else if (fn_local_kind(kinds, idx, slot) == NVM2C_VK_SARR) {
                stack_push_sarr(b, &st, rhs);
            } else if (fn_local_kind(kinds, idx, slot) == NVM2C_VK_REC) {
                int r = stack_push_rec(b, &st, rhs);
                if (r >= 0) {
                    memcpy(st.rec_k[r], fn_rec_k_const(rec_fields, idx, slot),
                           NVM2C_MAX_REC_FIELDS);
                }
            } else if (fn_local_kind(kinds, idx, slot) == NVM2C_VK_RARR) {
                int a = stack_push_rarr(b, &st, rhs);
                if (a >= 0) {
                    memcpy(st.rec_k[a], fn_rec_k_const(rec_fields, idx, slot),
                           NVM2C_MAX_REC_FIELDS);
                }
            } else {
                stack_push_temp(b, &st, rhs);
            }
            if (!b->failed) {
                nvm2c_printf(b, "    vt[%d] = lt%u;\n",
                             st.tags[st.sp - 1], (unsigned)slot);
            }
            break;
        }
        case OP_STORE_LOCAL: {
            uint16_t slot = ins.operands[0].u16;
            if (slot >= fn->local_count) {
                nvm2c_fail(b, "function %u: STORE_LOCAL %u out of range", idx, slot);
                goto done;
            }
            {
                uint8_t expect = fn_local_kind(kinds, idx, slot);
                uint8_t got = 0;
                int is_void = -1;
                int t = stack_pop_value(b, &st, &got, &is_void);
                if (!b->failed && got != expect) {
                    nvm2c_fail(b, "STORE_LOCAL: value-kind mismatch");
                }
                if (b->failed) goto done;
                nvm2c_printf(b, "    lt%u = vt[%d];\n", (unsigned)slot, is_void);
                if (expect == NVM2C_VK_STR) {
                    nvm2c_printf(b, "    if (lt%u != TAG_VOID) l%u = s[%d];\n", (unsigned)slot, (unsigned)slot, t);
                } else if (expect == NVM2C_VK_ARR) {
                    nvm2c_printf(b, "    if (lt%u != TAG_VOID) l%u = a[%d];\n", (unsigned)slot, (unsigned)slot, t);
                } else if (expect == NVM2C_VK_SARR) {
                    nvm2c_printf(b, "    if (lt%u != TAG_VOID) l%u = sa[%d];\n", (unsigned)slot, (unsigned)slot, t);
                } else if (expect == NVM2C_VK_REC) {
                    nvm2c_printf(b, "    if (lt%u != TAG_VOID) l%u = r[%d];\n", (unsigned)slot, (unsigned)slot, t);
                } else if (expect == NVM2C_VK_RARR) {
                    nvm2c_printf(b, "    if (lt%u != TAG_VOID) l%u = ra[%d];\n", (unsigned)slot, (unsigned)slot, t);
                } else {
                    nvm2c_printf(b, "    if (lt%u != TAG_VOID) l%u = t[%d];\n", (unsigned)slot, (unsigned)slot, t);
                }
            }
            break;
        }
        case OP_ADD:
        case OP_I64_ADD:
            emit_binop(b, &st, "+");
            break;
        case OP_SUB:
        case OP_I64_SUB:
            emit_binop(b, &st, "-");
            break;
        case OP_MUL:
        case OP_I64_MUL:
            emit_binop(b, &st, "*");
            break;
        case OP_DIV:
        case OP_I64_DIV_S: {
            int rhs = stack_pop_expect(b, &st, NVM2C_VK_INT, "div rhs");
            int lhs = stack_pop_expect(b, &st, NVM2C_VK_INT, "div lhs");
            if (b->failed) goto done;
            char expr[96];
            snprintf(expr, sizeof expr, "(t[%d] == 0 ? (int64_t)0 : t[%d] / t[%d])",
                     rhs, lhs, rhs);
            stack_push_temp(b, &st, expr);
            break;
        }
        case OP_MOD:
        case OP_I64_REM_S: {
            int rhs = stack_pop_expect(b, &st, NVM2C_VK_INT, "mod rhs");
            int lhs = stack_pop_expect(b, &st, NVM2C_VK_INT, "mod lhs");
            if (b->failed) goto done;
            char expr[96];
            snprintf(expr, sizeof expr, "(t[%d] == 0 ? (int64_t)0 : t[%d] %% t[%d])",
                     rhs, lhs, rhs);
            stack_push_temp(b, &st, expr);
            break;
        }
        case OP_NEG:
        case OP_I64_NEG:
            emit_unop(b, &st, "-");
            break;
        case OP_BOOL_NOT:
            emit_bool_unop(b, &st);
            break;
        case OP_I64_EQ:
            emit_binop(b, &st, "==");
            break;
        case OP_EQ:
        case OP_NE: {
            uint8_t rk = NVM2C_VK_INT;
            uint8_t lk = NVM2C_VK_INT;
            int rhs = stack_pop_kind(b, &st, &rk);
            int lhs = stack_pop_kind(b, &st, &lk);
            if (b->failed) goto done;
            if (lk == NVM2C_VK_INT && rk == NVM2C_VK_INT) {
                char expr[64];
                snprintf(expr, sizeof expr, "t[%d] %s t[%d]",
                         lhs, ins.opcode == OP_EQ ? "==" : "!=", rhs);
                stack_push_temp(b, &st, expr);
            } else if (lk == NVM2C_VK_STR && rk == NVM2C_VK_STR) {
                char expr[192];
                snprintf(expr, sizeof expr,
                         "(int64_t)(strcmp(s[%d] ? s[%d] : \"\", s[%d] ? s[%d] : \"\") %s 0)",
                         lhs, lhs, rhs, rhs, ins.opcode == OP_EQ ? "==" : "!=");
                stack_push_temp(b, &st, expr);
            } else {
                nvm2c_fail(b, "function %u: EQ/NE of mixed or non-string values is refused", idx);
                goto done;
            }
            break;
        }
        case OP_I64_NE:
            emit_binop(b, &st, "!=");
            break;
        case OP_I64_LT_S:
            emit_binop(b, &st, "<");
            break;
        case OP_I64_LE_S:
            emit_binop(b, &st, "<=");
            break;
        case OP_I64_GT_S:
            emit_binop(b, &st, ">");
            break;
        case OP_I64_GE_S:
            emit_binop(b, &st, ">=");
            break;
        case OP_BOOL_AND:
            emit_bool_binop(b, &st, "&&");
            break;
        case OP_BOOL_OR:
            emit_bool_binop(b, &st, "||");
            break;
        case OP_STR_LEN: {
            int s = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_LEN");
            if (b->failed) goto done;
            char expr[80];
            snprintf(expr, sizeof expr, "(int64_t)strlen(s[%d] ? s[%d] : \"\")", s, s);
            stack_push_temp(b, &st, expr);
            break;
        }
        case OP_STR_CONCAT: {
            int rhs = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_CONCAT rhs");
            int lhs = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_CONCAT lhs");
            if (b->failed) goto done;
            char expr[80];
            snprintf(expr, sizeof expr, "nstr_concat(s[%d], s[%d])", lhs, rhs);
            stack_push_str(b, &st, expr);
            break;
        }
        case OP_STR_SUBSTR: {
            int len = stack_pop_expect(b, &st, NVM2C_VK_INT, "STR_SUBSTR length");
            int start = stack_pop_expect(b, &st, NVM2C_VK_INT, "STR_SUBSTR start");
            int s = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_SUBSTR");
            if (b->failed) goto done;
            char expr[96];
            snprintf(expr, sizeof expr, "nstr_substr(s[%d], t[%d], t[%d])", s, start, len);
            stack_push_str(b, &st, expr);
            break;
        }
        case OP_STR_CONTAINS: {
            int needle = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_CONTAINS needle");
            int hay = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_CONTAINS haystack");
            if (b->failed) goto done;
            char expr[160];
            snprintf(expr, sizeof expr,
                     "(int64_t)(strstr(s[%d] ? s[%d] : \"\", s[%d] ? s[%d] : \"\") != NULL)",
                     hay, hay, needle, needle);
            stack_push_temp(b, &st, expr);
            break;
        }
        case OP_STR_CHAR_AT: {
            int ix = stack_pop_expect(b, &st, NVM2C_VK_INT, "STR_CHAR_AT index");
            int s = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_CHAR_AT");
            if (b->failed) goto done;
            char expr[80];
            snprintf(expr, sizeof expr, "nstr_char_at(s[%d], t[%d])", s, ix);
            stack_push_temp(b, &st, expr);
            break;
        }
        case OP_CAST_STRING: {
            uint8_t kind = NVM2C_VK_INT;
            int value = stack_pop_kind(b, &st, &kind);
            if (b->failed) goto done;
            char expr[48];
            if (kind == NVM2C_VK_VALUE) {
                snprintf(expr, sizeof expr, "nvalue_as_string(v[%d])", value);
            } else if (kind == NVM2C_VK_INT) {
                snprintf(expr, sizeof expr, "nstr_from_i64(t[%d])", value);
            } else {
                nvm2c_fail(b, "CAST_STRING: unsupported value kind");
                goto done;
            }
            stack_push_str(b, &st, expr);
            break;
        }
        case OP_CAST_INT: {
            uint8_t kind = NVM2C_VK_INT;
            int value = stack_pop_kind(b, &st, &kind);
            if (b->failed) goto done;
            if (kind == NVM2C_VK_VALUE) {
                char expr[48];
                snprintf(expr, sizeof expr, "nvalue_as_int(v[%d])", value);
                stack_push_temp(b, &st, expr);
            } else if (kind == NVM2C_VK_INT) {
                char expr[32];
                snprintf(expr, sizeof expr, "t[%d]", value);
                stack_push_temp(b, &st, expr);
            } else if (kind == NVM2C_VK_STR) {
                char expr[64];
                snprintf(expr, sizeof expr, "(int64_t)strtoll(s[%d] ? s[%d] : \"\", NULL, 10)",
                         value, value);
                stack_push_temp(b, &st, expr);
            } else {
                nvm2c_fail(b, "CAST_INT: unsupported value kind");
                goto done;
            }
            break;
        }
        case OP_CAST_BOOL: {
            uint8_t kind = NVM2C_VK_INT;
            int value = stack_pop_kind(b, &st, &kind);
            if (b->failed) goto done;
            char expr[48];
            if (kind == NVM2C_VK_VALUE) {
                snprintf(expr, sizeof expr, "nvalue_truthy(v[%d])", value);
            } else if (kind == NVM2C_VK_INT) {
                snprintf(expr, sizeof expr, "(t[%d] != 0)", value);
            } else if (kind == NVM2C_VK_STR) {
                snprintf(expr, sizeof expr, "(s[%d] != NULL)", value);
            } else {
                nvm2c_fail(b, "CAST_BOOL: unsupported value kind");
                goto done;
            }
            stack_push_temp(b, &st, expr);
            break;
        }
        case OP_TYPE_CHECK: {
            uint8_t kind = NVM2C_VK_INT;
            int value = stack_pop_kind(b, &st, &kind);
            uint8_t expected = ins.operands[0].u8;
            if (b->failed) goto done;
            char expr[64];
            if (kind == NVM2C_VK_VALUE) {
                snprintf(expr, sizeof expr, "(v[%d].tag == %u)", value, (unsigned)expected);
            } else {
                uint8_t actual = kind == NVM2C_VK_STR ? TAG_STRING :
                    kind == NVM2C_VK_INT ? TAG_INT : TAG_ARRAY;
                snprintf(expr, sizeof expr, "%dLL", actual == expected);
            }
            stack_push_temp(b, &st, expr);
            break;
        }
        case OP_ARR_NEW: {
            uint8_t tag = ins.operands[0].u8;
            int as_sarr = (tag == TAG_STRING);
            if (tag != TAG_INT && tag != TAG_STRING && tag != TAG_STRUCT) {
                nvm2c_fail(b, "function %u: ARR_NEW only supports int, string, or record elements", idx);
                goto done;
            }
            if (tag == TAG_STRUCT) {
                int a = stack_push_rarr(b, &st, "nrarr_new()");
                if (a >= 0) {
                    memcpy(st.rec_k[a], result_fields +
                           (size_t)idx * NVM2C_MAX_REC_FIELDS,
                           NVM2C_MAX_REC_FIELDS);
                }
                break;
            }
            if (tag == TAG_INT) {
                DecodedInstruction nxt;
                uint32_t nn = isa_decode(code + pc, remaining - pc, &nxt);
                if (nn != 0 && nxt.opcode == OP_STORE_LOCAL) {
                    uint16_t slot = nxt.operands[0].u16;
                    if (fn_local_kind(kinds, idx, slot) == NVM2C_VK_SARR) {
                        as_sarr = 1;
                    } else if (fn_local_kind(kinds, idx, slot) == NVM2C_VK_RARR) {
                        int a = stack_push_rarr(b, &st, "nrarr_new()");
                        if (a >= 0) {
                            memcpy(st.rec_k[a], fn_rec_k_const(rec_fields, idx, slot),
                                   NVM2C_MAX_REC_FIELDS);
                        }
                        break;
                    }
                }
            }
            if (as_sarr) {
                stack_push_sarr(b, &st, "nsarr_new()");
            } else {
                stack_push_arr(b, &st, "narr_new()");
            }
            break;
        }
        case OP_ARR_LITERAL: {
            uint8_t tag = ins.operands[0].u8;
            uint16_t count = ins.operands[1].u16;
            int elems[NVM2C_MAX_STACK];
            int ei;
            uint8_t ekind = NVM2C_VK_INT;
            if (tag == TAG_INT) {
                ekind = NVM2C_VK_INT;
            } else if (tag == TAG_STRING) {
                ekind = NVM2C_VK_STR;
            } else {
                nvm2c_fail(b, "function %u: ARR_LITERAL only supports int or string elements", idx);
                goto done;
            }
            if (count > NVM2C_MAX_STACK) {
                nvm2c_fail(b, "function %u: ARR_LITERAL is too large", idx);
                goto done;
            }
            for (ei = (int)count - 1; ei >= 0; ei--) {
                elems[ei] = stack_pop_expect(b, &st, ekind, "ARR_LITERAL");
                if (b->failed) goto done;
            }
            if (tag == TAG_STRING) {
                if (count == 0) {
                    stack_push_sarr(b, &st, "nsarr_lit(0, 0)");
                } else {
                    char rhs[768];
                    size_t pos = 0;
                    pos += (size_t)snprintf(rhs + pos, sizeof rhs - pos,
                                            "nsarr_lit((const char *[]){");
                    for (ei = 0; ei < (int)count; ei++) {
                        if (ei) pos += (size_t)snprintf(rhs + pos, sizeof rhs - pos, ", ");
                        pos += (size_t)snprintf(rhs + pos, sizeof rhs - pos, "s[%d]", elems[ei]);
                        if (pos >= sizeof rhs) {
                            nvm2c_fail(b, "function %u: ARR_LITERAL overflow", idx);
                            goto done;
                        }
                    }
                    snprintf(rhs + pos, sizeof rhs - pos, "}, %u)", (unsigned)count);
                    stack_push_sarr(b, &st, rhs);
                }
            } else if (count == 0) {
                stack_push_arr(b, &st, "narr_lit(0, 0)");
            } else {
                char rhs[768];
                size_t pos = 0;
                pos += (size_t)snprintf(rhs + pos, sizeof rhs - pos, "narr_lit((int64_t[]){");
                for (ei = 0; ei < (int)count; ei++) {
                    if (ei) pos += (size_t)snprintf(rhs + pos, sizeof rhs - pos, ", ");
                    pos += (size_t)snprintf(rhs + pos, sizeof rhs - pos, "t[%d]", elems[ei]);
                    if (pos >= sizeof rhs) {
                        nvm2c_fail(b, "function %u: ARR_LITERAL overflow", idx);
                        goto done;
                    }
                }
                snprintf(rhs + pos, sizeof rhs - pos, "}, %u)", (unsigned)count);
                stack_push_arr(b, &st, rhs);
            }
            break;
        }
        case OP_ARR_LEN: {
            uint8_t ak = NVM2C_VK_INT;
            int arr = stack_pop_kind(b, &st, &ak);
            if (b->failed) goto done;
            if (ak == NVM2C_VK_ARR) {
                char expr[64];
                snprintf(expr, sizeof expr, "(int64_t)(a[%d] ? a[%d]->len : 0)", arr, arr);
                stack_push_temp(b, &st, expr);
            } else if (ak == NVM2C_VK_SARR) {
                char expr[64];
                snprintf(expr, sizeof expr, "(int64_t)(sa[%d] ? sa[%d]->len : 0)", arr, arr);
                stack_push_temp(b, &st, expr);
            } else if (ak == NVM2C_VK_RARR) {
                char expr[64];
                snprintf(expr, sizeof expr, "(int64_t)(ra[%d] ? ra[%d]->len : 0)", arr, arr);
                stack_push_temp(b, &st, expr);
            } else {
                nvm2c_fail(b, "function %u: ARR_LEN expected an array", idx);
                goto done;
            }
            break;
        }
        case OP_ARR_GET: {
            uint8_t ak = NVM2C_VK_INT;
            int ix = stack_pop_expect(b, &st, NVM2C_VK_INT, "ARR_GET index");
            int arr = stack_pop_kind(b, &st, &ak);
            if (b->failed) goto done;
            if (ak == NVM2C_VK_ARR) {
                char expr[80];
                snprintf(expr, sizeof expr, "narr_get(a[%d], t[%d])", arr, ix);
                stack_push_value(b, &st, expr);
            } else if (ak == NVM2C_VK_SARR) {
                char expr[80];
                snprintf(expr, sizeof expr, "nsarr_get(sa[%d], t[%d])", arr, ix);
                stack_push_value(b, &st, expr);
            } else if (ak == NVM2C_VK_RARR) {
                char expr[80];
                snprintf(expr, sizeof expr, "nrarr_get(ra[%d], t[%d])", arr, ix);
                int r = stack_push_rec(b, &st, expr);
                if (r >= 0) memcpy(st.rec_k[r], st.rec_k[arr], NVM2C_MAX_REC_FIELDS);
            } else {
                nvm2c_fail(b, "function %u: ARR_GET expected an array", idx);
                goto done;
            }
            break;
        }
        case OP_ARR_PUSH: {
            uint8_t vk = NVM2C_VK_INT;
            uint8_t ak = NVM2C_VK_INT;
            int val = stack_pop_kind(b, &st, &vk);
            int arr = stack_pop_kind(b, &st, &ak);
            if (b->failed) goto done;
            if (ak == NVM2C_VK_ARR && vk == NVM2C_VK_INT) {
                char expr[80];
                snprintf(expr, sizeof expr, "narr_push(a[%d], t[%d])", arr, val);
                stack_push_arr(b, &st, expr);
            } else if (ak == NVM2C_VK_SARR && vk == NVM2C_VK_STR) {
                char expr[80];
                snprintf(expr, sizeof expr, "nsarr_push(sa[%d], s[%d])", arr, val);
                stack_push_sarr(b, &st, expr);
            } else if (ak == NVM2C_VK_RARR && vk == NVM2C_VK_REC) {
                char expr[80];
                snprintf(expr, sizeof expr, "nrarr_push(ra[%d], r[%d])", arr, val);
                int a = stack_push_rarr(b, &st, expr);
                if (a >= 0) memcpy(st.rec_k[a], st.rec_k[val], NVM2C_MAX_REC_FIELDS);
            } else {
                nvm2c_fail(b, "function %u: ARR_PUSH type mismatch", idx);
                goto done;
            }
            break;
        }
        case OP_AGG_PACK: {
            uint8_t kind = ins.operands[0].u8;
            uint16_t count = ins.operands[3].u16;
            int elems[NVM2C_MAX_REC_FIELDS];
            uint8_t fkind[NVM2C_MAX_REC_FIELDS];
            int ei;
            if (kind != AGG_RECORD) {
                nvm2c_fail(b, "function %u: AGG_PACK only supports int or string fields", idx);
                goto done;
            }
            if (count > NVM2C_MAX_REC_FIELDS) {
                nvm2c_fail(b, "function %u: AGG_PACK has too many fields", idx);
                goto done;
            }
            for (ei = (int)count - 1; ei >= 0; ei--) {
                uint8_t vk = NVM2C_VK_INT;
                elems[ei] = stack_pop_kind(b, &st, &vk);
                if (b->failed) goto done;
                if (vk != NVM2C_VK_INT && vk != NVM2C_VK_STR &&
                    vk != NVM2C_VK_ARR && vk != NVM2C_VK_SARR &&
                    vk != NVM2C_VK_RARR) {
                    nvm2c_fail(b, "function %u: AGG_PACK fields must be scalar or array values", idx);
                    goto done;
                }
                fkind[ei] = vk;
            }
            if (st.next_rec >= NVM2C_MAX_TEMPS) {
                nvm2c_fail(b, "too many record temporaries");
                goto done;
            }
            if (st.sp >= NVM2C_MAX_STACK) {
                nvm2c_fail(b, "operand stack overflow");
                goto done;
            }
            {
                int r = st.next_rec++;
                nvm2c_printf(b, "    r[%d].n = %u;\n", r, (unsigned)count);
                for (ei = 0; ei < (int)count; ei++) {
                    st.rec_k[r][ei] = fkind[ei];
                    if (fkind[ei] == NVM2C_VK_STR) {
                        nvm2c_printf(b, "    r[%d].s[%d] = s[%d];\n", r, ei, elems[ei]);
                    } else if (fkind[ei] == NVM2C_VK_ARR) {
                        nvm2c_printf(b, "    r[%d].a[%d] = a[%d];\n", r, ei, elems[ei]);
                    } else if (fkind[ei] == NVM2C_VK_SARR) {
                        nvm2c_printf(b, "    r[%d].sa[%d] = sa[%d];\n", r, ei, elems[ei]);
                    } else if (fkind[ei] == NVM2C_VK_RARR) {
                        nvm2c_printf(b, "    r[%d].ra[%d] = ra[%d];\n", r, ei, elems[ei]);
                    } else {
                        nvm2c_printf(b, "    r[%d].f[%d] = t[%d];\n", r, ei, elems[ei]);
                    }
                    nvm2c_printf(b, "    r[%d].k[%d] = %u;\n", r, ei,
                                  (unsigned)fkind[ei]);
                }
                st.slots[st.sp] = r;
                st.kinds[st.sp] = NVM2C_VK_REC;
                if (stack_push_tag(b, &st, "TAG_STRUCT") < 0) goto done;
                st.sp++;
            }
            break;
        }
        case OP_AGG_GET: {
            uint16_t fi = ins.operands[0].u16;
            int rec = stack_pop_expect(b, &st, NVM2C_VK_REC, "AGG_GET");
            if (b->failed) goto done;
            if (fi >= NVM2C_MAX_REC_FIELDS) {
                nvm2c_fail(b, "function %u: AGG_GET field is out of range", idx);
                goto done;
            }
            nvm2c_printf(b, "    if (%u >= r[%d].n) abort();\n", (unsigned)fi, rec);
            {
                char expr[64];
                if (st.rec_k[rec][fi] == NVM2C_VK_STR) {
                    snprintf(expr, sizeof expr, "r[%d].s[%u]", rec, (unsigned)fi);
                    stack_push_str(b, &st, expr);
                } else if (st.rec_k[rec][fi] == NVM2C_VK_ARR) {
                    snprintf(expr, sizeof expr, "r[%d].a[%u]", rec, (unsigned)fi);
                    stack_push_arr(b, &st, expr);
                } else if (st.rec_k[rec][fi] == NVM2C_VK_SARR) {
                    snprintf(expr, sizeof expr, "r[%d].sa[%u]", rec, (unsigned)fi);
                    stack_push_sarr(b, &st, expr);
                } else if (st.rec_k[rec][fi] == NVM2C_VK_RARR) {
                    snprintf(expr, sizeof expr, "r[%d].ra[%u]", rec, (unsigned)fi);
                    stack_push_rarr(b, &st, expr);
                } else {
                    snprintf(expr, sizeof expr, "r[%d].f[%u]", rec, (unsigned)fi);
                    stack_push_temp(b, &st, expr);
                }
            }
            break;
        }
        case OP_CALL: {
            uint32_t callee = ins.operands[0].u32;
            char call[768];
            if (!build_direct_call(b, &st, mod, idx, callee, kinds, result_kinds,
                                   call, sizeof call)) {
                goto done;
            }
            const NvmFunctionEntry *cf = &mod->functions[callee];
            if (result_is_i64(cf)) {
                stack_push_temp(b, &st, call);
            } else if (cf->result_count == 1 && cf->result_tag == TAG_STRING) {
                stack_push_str(b, &st, call);
            } else if (cf->result_count == 1 && cf->result_tag == TAG_ARRAY) {
                if (result_kinds[callee] == NVM2C_VK_ARR) {
                    stack_push_arr(b, &st, call);
                } else if (result_kinds[callee] == NVM2C_VK_SARR) {
                    stack_push_sarr(b, &st, call);
                } else {
                    int a = stack_push_rarr(b, &st, call);
                    if (a >= 0) {
                        memcpy(st.rec_k[a], result_fields +
                               (size_t)callee * NVM2C_MAX_REC_FIELDS,
                               NVM2C_MAX_REC_FIELDS);
                    }
                }
            } else if (cf->result_count == 1 &&
                       (cf->result_tag == TAG_STRUCT || cf->result_tag == TAG_UNION)) {
                int r = stack_push_rec(b, &st, call);
                if (r >= 0) {
                    memcpy(st.rec_k[r], result_fields +
                           (size_t)callee * NVM2C_MAX_REC_FIELDS,
                           NVM2C_MAX_REC_FIELDS);
                }
            } else {
                nvm2c_printf(b, "    %s;\n", call);
            }
            break;
        }
        case OP_TAIL_CALL: {
            uint32_t callee = ins.operands[0].u32;
            if (callee >= mod->function_count) {
                nvm2c_fail(b, "function %u: TAIL_CALL target %u is out of range", idx, callee);
                goto done;
            }
            const NvmFunctionEntry *cf = &mod->functions[callee];
            if (cf->result_count != fn->result_count || cf->result_tag != fn->result_tag) {
                nvm2c_fail(b, "function %u: TAIL_CALL result signature mismatch", idx);
                goto done;
            }
            char call[768];
            if (!build_direct_call(b, &st, mod, idx, callee, kinds, result_kinds,
                                   call, sizeof call)) {
                goto done;
            }
            if (st.sp != 0) {
                nvm2c_fail(b, "function %u: TAIL_CALL leaves extra stack values", idx);
                goto done;
            }
            if (fn->result_count == 1 &&
                 (result_is_i64(fn) || fn->result_tag == TAG_STRING ||
                  fn->result_tag == TAG_ARRAY || fn->result_tag == TAG_STRUCT ||
                  fn->result_tag == TAG_UNION)) {
                nvm2c_printf(b, "    return %s;\n", call);
            } else {
                nvm2c_printf(b, "    %s;\n    return;\n", call);
            }
            terminated = 1;
            break;
        }
        case OP_JMP: {
            size_t tgt = 0;
            if (!jump_target(b, idx, start, ins.operands[0].i32, remaining, &tgt)) {
                goto done;
            }
            nvm2c_printf(b, "    goto L_%zu;\n", tgt);
            if (!record_join(b, idx, joins, join_set, tgt, &st)) goto done;
            terminated = 1;
            break;
        }
        case OP_JMP_FALSE: {
            int cond = stack_pop_expect(b, &st, NVM2C_VK_INT, "JMP_FALSE");
            if (b->failed) goto done;
            size_t tgt = 0;
            if (!jump_target(b, idx, start, ins.operands[0].i32, remaining, &tgt)) {
                goto done;
            }
            nvm2c_printf(b, "    if (!t[%d]) goto L_%zu;\n", cond, tgt);
            if (!record_join(b, idx, joins, join_set, tgt, &st)) goto done;
            break;
        }
        case OP_RET:
            if (result_is_i64(fn)) {
                int t = stack_pop_expect(b, &st, NVM2C_VK_INT, "RET");
                if (b->failed) goto done;
                if (st.sp != 0) {
                    nvm2c_fail(b, "function %u: RET leaves extra stack values", idx);
                    goto done;
                }
                nvm2c_printf(b, "    return t[%d];\n", t);
            } else if (fn->result_count == 1 && fn->result_tag == TAG_STRING) {
                int s = stack_pop_expect(b, &st, NVM2C_VK_STR, "RET");
                if (b->failed) goto done;
                if (st.sp != 0) {
                    nvm2c_fail(b, "function %u: RET leaves extra stack values", idx);
                    goto done;
                }
                nvm2c_printf(b, "    return s[%d];\n", s);
            } else if (fn->result_count == 1 && fn->result_tag == TAG_ARRAY) {
                uint8_t result_kind = result_kinds[idx];
                int a = stack_pop_expect(b, &st, result_kind, "RET");
                if (b->failed) goto done;
                if (st.sp != 0) {
                    nvm2c_fail(b, "function %u: RET leaves extra stack values", idx);
                    goto done;
                }
                if (result_kind == NVM2C_VK_ARR) {
                    nvm2c_printf(b, "    return a[%d];\n", a);
                } else if (result_kind == NVM2C_VK_SARR) {
                    nvm2c_printf(b, "    return sa[%d];\n", a);
                } else {
                    nvm2c_printf(b, "    return ra[%d];\n", a);
                }
            } else if (fn->result_count == 1 &&
                       (fn->result_tag == TAG_STRUCT || fn->result_tag == TAG_UNION)) {
                int r = stack_pop_expect(b, &st, NVM2C_VK_REC, "RET");
                if (b->failed) goto done;
                if (st.sp != 0) {
                    nvm2c_fail(b, "function %u: RET leaves extra stack values", idx);
                    goto done;
                }
                nvm2c_printf(b, "    return r[%d];\n", r);
            } else {
                if (st.sp != 0) {
                    nvm2c_fail(b, "function %u: void RET leaves extra stack values", idx);
                    goto done;
                }
                nvm2c_puts(b, "    return;\n");
            }
            terminated = 1;
            break;
        case OP_HALT:
            if (result_is_i64(fn) && st.sp == 1) {
                nvm2c_printf(b, "    return t[%d];\n",
                             stack_pop_expect(b, &st, NVM2C_VK_INT, "HALT"));
            } else if (fn->result_count == 1 && fn->result_tag == TAG_ARRAY && st.sp == 1) {
                uint8_t result_kind = result_kinds[idx];
                int a = stack_pop_expect(b, &st, result_kind, "HALT");
                if (result_kind == NVM2C_VK_ARR) {
                    nvm2c_printf(b, "    return a[%d];\n", a);
                } else if (result_kind == NVM2C_VK_SARR) {
                    nvm2c_printf(b, "    return sa[%d];\n", a);
                } else {
                    nvm2c_printf(b, "    return ra[%d];\n", a);
                }
            } else if (st.sp == 0 && (fn->result_count == 0 || fn->result_tag == TAG_VOID)) {
                nvm2c_puts(b, "    return;\n");
            } else if (st.sp == 0 && result_is_i64(fn)) {
                nvm2c_puts(b, "    return 0;\n");
            } else {
                nvm2c_fail(b, "function %u: HALT with unexpected stack height %d", idx, st.sp);
                goto done;
            }
            terminated = 1;
            break;
        case OP_CALL_EXTERN:
            nvm2c_fail(b, "CALL_EXTERN is the VM FFI/co-process path; nvm2c does not emit it");
            goto done;
        default: {
            const InstructionInfo *info = isa_get_info(ins.opcode);
            nvm2c_fail(b, "unsupported opcode %s (0x%02X) in the nvm2c subset",
                       info ? info->name : "UNKNOWN", ins.opcode);
            goto done;
        }
        }
        if (b->failed) goto done;
    }

    if (is_target[remaining]) {
        nvm2c_printf(b, "L_%zu: ;\n", remaining);
    }
    if (!terminated) {
        nvm2c_fail(b, "function %u: falls off the end without RET or HALT", idx);
        goto done;
    }
    {
        char count[4];
        unsigned rec_count = (unsigned)(st.next_rec > 0 ? st.next_rec : 1);
        count[0] = (char)('0' + rec_count / 100);
        count[1] = (char)('0' + rec_count / 10 % 10);
        count[2] = (char)('0' + rec_count % 10);
        count[3] = '\0';
        memcpy(b->data + rec_count_offset, count, 3);
    }
    nvm2c_puts(b, "}\n\n");

done:
    free(is_start);
    free(is_target);
    free(joins);
    free(join_set);
}

static int module_has_opcode(const NvmModule *mod, const uint8_t *reachable, uint8_t op) {
    uint32_t i;
    for (i = 0; i < mod->function_count; i++) {
        if (!reachable[i]) continue;
        const NvmFunctionEntry *fn = &mod->functions[i];
        if (fn->code_offset > mod->code_size ||
            fn->code_length > mod->code_size - fn->code_offset) {
            continue;
        }
        const uint8_t *code = mod->code + fn->code_offset;
        size_t remaining = fn->code_length;
        size_t pc = 0;
        while (pc < remaining) {
            DecodedInstruction ins;
            uint32_t n = isa_decode(code + pc, remaining - pc, &ins);
            if (n == 0) break;
            if (ins.opcode == op) return 1;
            pc += n;
        }
    }
    return 0;
}

static int module_has_arr_op_tag(const NvmModule *mod, const uint8_t *reachable,
                                 uint8_t op, uint8_t tag) {
    uint32_t i;
    for (i = 0; i < mod->function_count; i++) {
        if (!reachable[i]) continue;
        const NvmFunctionEntry *fn = &mod->functions[i];
        if (fn->code_offset > mod->code_size ||
            fn->code_length > mod->code_size - fn->code_offset) {
            continue;
        }
        const uint8_t *code = mod->code + fn->code_offset;
        size_t remaining = fn->code_length;
        size_t pc = 0;
        while (pc < remaining) {
            DecodedInstruction ins;
            uint32_t n = isa_decode(code + pc, remaining - pc, &ins);
            if (n == 0) break;
            if (ins.opcode == op && ins.operands[0].u8 == tag) return 1;
            pc += n;
        }
    }
    return 0;
}

static int module_has_local_kind(const uint8_t *kinds, uint32_t fn_count, uint8_t kind) {
    uint32_t i;
    uint16_t li;
    for (i = 0; i < fn_count; i++) {
        for (li = 0; li < NVM2C_MAX_LOCALS; li++) {
            if (fn_local_kind(kinds, i, li) == kind) return 1;
        }
    }
    return 0;
}

static int module_has_result_kind(const uint8_t *result_kinds, uint32_t fn_count,
                                  uint8_t kind) {
    uint32_t i;
    for (i = 0; i < fn_count; i++) {
        if (result_kinds[i] == kind) return 1;
    }
    return 0;
}

static void emit_nstr_storage(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "typedef struct nstr_owned_s {\n"
        "    struct nstr_owned_s *next;\n"
        "    char data[];\n"
        "} nstr_owned_t;\n"
        "static nstr_owned_t *nstr_owned;\n"
        "static char *nstr_alloc(size_t len) {\n"
        "    if (len == SIZE_MAX) abort();\n"
        "    size_t bytes = sizeof(nstr_owned_t) + len + 1;\n"
        "    if (bytes < len) abort();\n"
        "    nstr_owned_t *owned = (nstr_owned_t *)malloc(bytes);\n"
        "    if (!owned) abort();\n"
        "    owned->next = nstr_owned;\n"
        "    nstr_owned = owned;\n"
        "    return owned->data;\n"
        "}\n"
        "static void nstr_free_all(void) {\n"
        "    while (nstr_owned) {\n"
        "        nstr_owned_t *next = nstr_owned->next;\n"
        "        free(nstr_owned);\n"
        "        nstr_owned = next;\n"
        "    }\n"
        "}\n");
}

static void emit_nstr_concat(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nstr_concat(const char *a, const char *b) {\n"
        "    size_t na = strlen(a ? a : \"\");\n"
        "    size_t nb = strlen(b ? b : \"\");\n"
        "    if (nb > SIZE_MAX - na) abort();\n"
        "    char *p = nstr_alloc(na + nb);\n"
        "    memcpy(p, a ? a : \"\", na);\n"
        "    memcpy(p + na, b ? b : \"\", nb);\n"
        "    p[na + nb] = 0;\n"
        "    return p;\n"
        "}\n\n");
}

static void emit_nstr_substr(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nstr_substr(const char *s, int64_t start, int64_t len) {\n"
        "    const char *src = s ? s : \"\";\n"
        "    int64_t slen = (int64_t)strlen(src);\n"
        "    if (start < 0) start = 0;\n"
        "    if (start >= slen || len <= 0) return \"\";\n"
        "    if (len > slen - start) len = slen - start;\n"
        "    char *p = nstr_alloc((size_t)len);\n"
        "    memcpy(p, src + start, (size_t)len);\n"
        "    p[len] = 0;\n"
        "    return p;\n"
        "}\n\n");
}

static void emit_nstr_char_at(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static int64_t nstr_char_at(const char *s, int64_t idx) {\n"
        "    const char *src = s ? s : \"\";\n"
        "    size_t n = strlen(src);\n"
        "    if (idx < 0 || (size_t)idx >= n) return -1;\n"
        "    return (int64_t)(unsigned char)src[idx];\n"
        "}\n\n");
}

static void emit_nstr_from_i64(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static inline const char *nstr_from_i64(int64_t v) {\n"
        "    char tmp[32];\n"
        "    int n = snprintf(tmp, sizeof tmp, \"%lld\", (long long)v);\n"
        "    if (n < 0 || (size_t)n >= sizeof tmp) abort();\n"
        "    char *p = nstr_alloc((size_t)n);\n"
        "    memcpy(p, tmp, (size_t)n + 1);\n"
        "    return p;\n"
        "}\n\n");
}

static void emit_narr_new(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static narr_t narr_new(void) {\n"
        "    narr_t a = (narr_t)calloc(1, sizeof(narr_s));\n"
        "    if (!a) abort();\n"
        "    return a;\n"
        "}\n\n");
}

static void emit_narr_storage(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "typedef struct narr_owned_s {\n"
        "    struct narr_owned_s *next;\n"
        "    int64_t data[];\n"
        "} narr_owned_t;\n"
        "static narr_owned_t *narr_owned;\n"
        "static void narr_free_all(void) {\n"
        "    while (narr_owned) {\n"
        "        narr_owned_t *next = narr_owned->next;\n"
        "        free(narr_owned);\n"
        "        narr_owned = next;\n"
        "    }\n"
        "}\n\n");
}

static void emit_narr_alloc(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static int64_t *narr_alloc(size_t cap) {\n"
        "    if (cap > (SIZE_MAX - sizeof(narr_owned_t)) / sizeof(int64_t)) abort();\n"
        "    size_t bytes = sizeof(narr_owned_t) + cap * sizeof(int64_t);\n"
        "    narr_owned_t *owned = (narr_owned_t *)malloc(bytes);\n"
        "    if (!owned) abort();\n"
        "    owned->next = narr_owned;\n"
        "    narr_owned = owned;\n"
        "    return owned->data;\n"
        "}\n\n");
}

static void emit_narr_reserve(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static void narr_reserve(narr_t a, size_t n) {\n"
        "    if (!a || a->len > a->cap || (a->len && !a->data)) abort();\n"
        "    if (n <= a->cap) return;\n"
        "    size_t cap = a->cap ? a->cap : 8;\n"
        "    while (cap < n) {\n"
        "        if (cap > SIZE_MAX / 2) abort();\n"
        "        cap *= 2;\n"
        "    }\n"
        "    int64_t *data = narr_alloc(cap);\n"
        "    if (a->len) memcpy(data, a->data, a->len * sizeof(int64_t));\n"
        "    a->data = data;\n"
        "    a->cap = cap;\n"
        "}\n\n");
}

static void emit_narr_lit(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static narr_t narr_lit(const int64_t *elems, size_t n) {\n"
        "    if (n > 0 && !elems) abort();\n"
        "    narr_t a = narr_new();\n"
        "    if (n) {\n"
        "        narr_reserve(a, n);\n"
        "        memcpy(a->data, elems, n * sizeof(int64_t));\n"
        "    }\n"
        "    a->len = n;\n"
        "    return a;\n"
        "}\n\n");
}

static void emit_narr_get(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static nvalue_t narr_get(narr_t a, int64_t idx) {\n"
        "    uint32_t narrowed = (uint32_t)idx;\n"
        "    if (!a || !a->data || (size_t)narrowed >= a->len) return nvalue_void();\n"
        "    return nvalue_int(a->data[narrowed]);\n"
        "}\n\n");
}

static void emit_narr_push(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static narr_t narr_push(narr_t a, int64_t v) {\n"
        "    if (!a) abort();\n"
        "    if (a->len == SIZE_MAX) abort();\n"
        "    narr_reserve(a, a->len + 1);\n"
        "    a->data[a->len++] = v;\n"
        "    return a;\n"
        "}\n\n");
}

static void emit_nsarr_new(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static nsarr_t nsarr_new(void) {\n"
        "    nsarr_t a = (nsarr_t)calloc(1, sizeof(nsarr_s));\n"
        "    if (!a) abort();\n"
        "    return a;\n"
        "}\n\n");
}

static void emit_nsarr_lit(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static nsarr_t nsarr_lit(const char *const *elems, size_t n) {\n"
        "    if (n > 0 && !elems) abort();\n"
        "    nsarr_t a = nsarr_new();\n"
        "    if (n) {\n"
        "        a->data = (const char **)malloc(n * sizeof(const char *));\n"
        "        if (!a->data) abort();\n"
        "        memcpy(a->data, elems, n * sizeof(const char *));\n"
        "    }\n"
        "    a->len = n;\n"
        "    a->cap = n;\n"
        "    return a;\n"
        "}\n\n");
}

static void emit_nsarr_get(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static nvalue_t nsarr_get(nsarr_t a, int64_t idx) {\n"
        "    uint32_t narrowed = (uint32_t)idx;\n"
        "    if (!a || !a->data || (size_t)narrowed >= a->len) return nvalue_void();\n"
        "    return nvalue_string(a->data[narrowed]);\n"
        "}\n\n");
}

static void emit_nsarr_push(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static nsarr_t nsarr_push(nsarr_t a, const char *v) {\n"
        "    if (!a) abort();\n"
        "    if (a->len && !a->data) abort();\n"
        "    if (a->len == a->cap) {\n"
        "        size_t cap = a->cap ? a->cap * 2 : 8;\n"
        "        if (cap < a->cap || cap > SIZE_MAX / sizeof(const char *)) abort();\n"
        "        const char **data = (const char **)realloc(a->data, cap * sizeof(const char *));\n"
        "        if (!data) abort();\n"
        "        a->data = data;\n"
        "        a->cap = cap;\n"
        "    }\n"
        "    a->data[a->len++] = v ? v : \"\";\n"
        "    return a;\n"
        "}\n\n");
}

static void emit_nrarr_helpers(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static nrarr_t nrarr_new(void) {\n"
        "    nrarr_t a = (nrarr_t)calloc(1, sizeof(nrarr_s));\n"
        "    if (!a) abort();\n"
        "    return a;\n"
        "}\n\n"
        "static nrarr_t nrarr_push(nrarr_t a, nrec_t v) {\n"
        "    if (!a || a->len >= NVM2C_RECORD_ARRAY_CAP) abort();\n"
        "    a->data[a->len++] = v;\n"
        "    return a;\n"
        "}\n\n"
        "static nrec_t nrarr_get(nrarr_t a, int64_t idx) {\n"
        "    if (!a || idx < 0 || (size_t)idx >= a->len) abort();\n"
        "    return a->data[idx];\n"
        "}\n\n");
}

char *nvm2c_emit(const NvmModule *mod, char *err, size_t err_len) {
    if (err && err_len) err[0] = '\0';
    if (!mod) {
        if (err && err_len) snprintf(err, err_len, "module is null");
        return NULL;
    }
    if (mod->function_count == 0) {
        if (err && err_len) snprintf(err, err_len, "module has no functions");
        return NULL;
    }
    if (mod->import_count != 0) {
        if (err && err_len) {
            snprintf(err, err_len, "imports require a host ABI; nvm2c refuses CALL_EXTERN");
        }
        return NULL;
    }

    Nvm2cBuf b;
    memset(&b, 0, sizeof b);
    b.err = err;
    b.err_len = err_len;

    uint8_t *kinds = calloc((size_t)mod->function_count * NVM2C_MAX_LOCALS, 1);
    uint8_t *rec_fields = calloc((size_t)mod->function_count * NVM2C_MAX_LOCALS
                                   * NVM2C_MAX_REC_FIELDS, 1);
    uint8_t *result_kinds = calloc(mod->function_count, 1);
    uint8_t *result_fields = calloc((size_t)mod->function_count * NVM2C_MAX_REC_FIELDS, 1);
    uint8_t *reachable = calloc(mod->function_count, 1);
    int need_owned_strings;
    int need_owned_iarrays = 0;
    if (!kinds || !rec_fields || !result_kinds || !result_fields || !reachable) {
        free(kinds);
        free(rec_fields);
        free(result_kinds);
        free(result_fields);
        free(reachable);
        if (err && err_len) snprintf(err, err_len, "out of memory");
        return NULL;
    }
    memset(kinds, NVM2C_VK_UNK,
           (size_t)mod->function_count * NVM2C_MAX_LOCALS);
    memset(rec_fields, NVM2C_VK_UNK,
           (size_t)mod->function_count * NVM2C_MAX_LOCALS * NVM2C_MAX_REC_FIELDS);
    memset(result_kinds, NVM2C_VK_UNK, mod->function_count);
    memset(result_fields, NVM2C_VK_UNK,
           (size_t)mod->function_count * NVM2C_MAX_REC_FIELDS);
    if (!mark_reachable_functions(&b, mod, reachable)) goto fail;

    {
        uint32_t pass;
        for (pass = 0; pass <= mod->function_count * NVM2C_MAX_LOCALS; pass++) {
            uint32_t i;
            int changed = 0;
            for (i = 0; i < mod->function_count; i++) {
                if (!reachable[i]) continue;
                if (!classify_function(&b, mod, i, kinds, rec_fields, result_kinds,
                                       result_fields,
                                       &changed)) {
                    goto fail;
                }
            }
            if (!changed) break;
        }
        if (pass > mod->function_count * NVM2C_MAX_LOCALS) {
            nvm2c_fail(&b, "parameter and record-field classification did not converge");
            goto fail;
        }
        for (pass = 0; pass < mod->function_count * NVM2C_MAX_LOCALS; pass++) {
            if (kinds[pass] == NVM2C_VK_UNK) kinds[pass] = NVM2C_VK_INT;
        }
        for (pass = 0; pass < mod->function_count; pass++) {
            if (!reachable[pass]) continue;
            if (mod->functions[pass].result_count == 1 &&
                mod->functions[pass].result_tag == TAG_ARRAY &&
                result_kinds[pass] == NVM2C_VK_UNK) {
                nvm2c_fail(&b, "function %u: array result kind could not be inferred", pass);
                goto fail;
            }
        }
    }

    need_owned_strings = module_has_opcode(mod, reachable, OP_STR_CONCAT) ||
        module_has_opcode(mod, reachable, OP_CAST_STRING) ||
        module_has_opcode(mod, reachable, OP_STR_SUBSTR);

    {
        int need_concat = module_has_opcode(mod, reachable, OP_STR_CONCAT);
        int need_cast_string = module_has_opcode(mod, reachable, OP_CAST_STRING);
        int need_cast_int = module_has_opcode(mod, reachable, OP_CAST_INT);
        int need_contains = module_has_opcode(mod, reachable, OP_STR_CONTAINS);
        int need_substr = module_has_opcode(mod, reachable, OP_STR_SUBSTR);
        int need_char_at = module_has_opcode(mod, reachable, OP_STR_CHAR_AT);
        int need_string = need_concat || need_cast_string || need_contains || need_substr ||
            need_char_at ||
            module_has_opcode(mod, reachable, OP_PUSH_STR) ||
            module_has_opcode(mod, reachable, OP_STR_LEN);
        int need_arr_lit = module_has_opcode(mod, reachable, OP_ARR_LITERAL);
        int need_arr_get = module_has_opcode(mod, reachable, OP_ARR_GET);
        int need_arr_push = module_has_opcode(mod, reachable, OP_ARR_PUSH);
        int need_iarr_new = module_has_arr_op_tag(mod, reachable, OP_ARR_NEW, TAG_INT);
        int need_iarr_lit = module_has_arr_op_tag(mod, reachable, OP_ARR_LITERAL, TAG_INT);
        int need_sarr_lit = module_has_arr_op_tag(mod, reachable, OP_ARR_LITERAL, TAG_STRING);
        int need_iarr = need_iarr_lit ||
            module_has_local_kind(kinds, mod->function_count, NVM2C_VK_ARR) ||
            module_has_result_kind(result_kinds, mod->function_count, NVM2C_VK_ARR);
        int need_sarr = need_sarr_lit ||
            module_has_local_kind(kinds, mod->function_count, NVM2C_VK_SARR) ||
            module_has_result_kind(result_kinds, mod->function_count, NVM2C_VK_SARR);
        int need_rarr = module_has_local_kind(kinds, mod->function_count, NVM2C_VK_RARR) ||
            module_has_result_kind(result_kinds, mod->function_count, NVM2C_VK_RARR);
        int need_iarr_get = need_arr_get && need_iarr;
        int need_sarr_get = need_arr_get && need_sarr;
        int need_iarr_push = need_arr_push && need_iarr;
        int need_sarr_push = need_arr_push && need_sarr;
        need_owned_iarrays = need_iarr_lit || (need_iarr && need_iarr_new);
        int need_sarr_new = need_sarr && module_has_opcode(mod, reachable, OP_ARR_NEW);
        int need_agg_get = module_has_opcode(mod, reachable, OP_AGG_GET);
        int need_print = module_has_opcode(mod, reachable, OP_PRINT) ||
            module_has_opcode(mod, reachable, OP_PRINTLN);
        int need_assert = module_has_opcode(mod, reachable, OP_ASSERT);
        uint32_t i;
        for (i = 0; i < mod->function_count && !need_string; i++) {
            if (!reachable[i]) continue;
            const NvmFunctionEntry *fn = &mod->functions[i];
            uint16_t li;
            if (fn->result_tag == TAG_STRING) need_string = 1;
            for (li = 0; li < fn->local_count; li++) {
                if (fn_local_kind(kinds, i, li) == NVM2C_VK_STR) need_string = 1;
            }
        }

        nvm2c_puts(&b,
            "/* Generated by nvm2c from NanoISA. Not a VM wrapper. */\n"
            "#include <stddef.h>\n"
            "#include <stdint.h>\n"
            "#include <stdlib.h>\n"
            "enum { TAG_VOID = 0, TAG_INT = 1, TAG_BOOL = 3, TAG_STRING = 5, "
            "TAG_ARRAY = 6, TAG_STRUCT = 7 };\n");
        if (need_print || need_cast_string) {
            nvm2c_puts(&b, "#include <stdio.h>\n");
        }
        if (need_concat || need_cast_string || need_cast_int || need_substr || need_arr_lit || need_arr_get ||
            need_arr_push || need_iarr_new || need_sarr_new || need_agg_get ||
            need_assert) {
            nvm2c_puts(&b, "#include <string.h>\n");
        } else if (need_string) {
            nvm2c_puts(&b, "#include <string.h>\n");
        }
        nvm2c_puts(&b,
            "\n"
            "typedef struct { int64_t *data; size_t len; size_t cap; } narr_s;\n"
            "typedef narr_s *narr_t;\n"
            "typedef struct { const char **data; size_t len; size_t cap; } nsarr_s;\n"
            "typedef nsarr_s *nsarr_t;\n"
            "typedef struct nrarr_s nrarr_s;\n"
            "typedef nrarr_s *nrarr_t;\n"
            "typedef struct { uint8_t tag; int64_t i64; const char *string; } nvalue_t;\n"
            "static inline nvalue_t nvalue_void(void) {\n"
            "    nvalue_t v = {0};\n"
            "    return v;\n"
            "}\n"
            "static inline nvalue_t nvalue_int(int64_t n) {\n"
            "    nvalue_t v = {1, n, NULL};\n"
            "    return v;\n"
            "}\n"
            "static inline nvalue_t nvalue_string(const char *s) {\n"
            "    nvalue_t v = {5, 0, s};\n"
            "    return v;\n"
            "}\n"
            "static inline int64_t nvalue_as_int(nvalue_t v) {\n"
            "    if (v.tag != 1) abort();\n"
            "    return v.i64;\n"
            "}\n"
            "static inline const char *nvalue_as_string(nvalue_t v) {\n"
            "    if (v.tag != 5) abort();\n"
            "    return v.string;\n"
            "}\n"
            "static inline int64_t nvalue_truthy(nvalue_t v) {\n"
            "    if (v.tag == 0) return 0;\n"
            "    if (v.tag == 1) return v.i64 != 0;\n"
            "    if (v.tag == 5) return v.string != NULL;\n"
            "    return 1;\n"
            "}\n");
        nvm2c_printf(&b,
            "typedef struct { int64_t f[%d]; const char *s[%d]; narr_t a[%d]; "
            "nsarr_t sa[%d]; nrarr_t ra[%d]; uint8_t k[%d]; uint16_t n; } nrec_t;\n",
            NVM2C_MAX_REC_FIELDS, NVM2C_MAX_REC_FIELDS, NVM2C_MAX_REC_FIELDS,
            NVM2C_MAX_REC_FIELDS, NVM2C_MAX_REC_FIELDS, NVM2C_MAX_REC_FIELDS);
        nvm2c_puts(&b,
            "enum { NVM2C_RECORD_ARRAY_CAP = 256 };\n"
            "struct nrarr_s { nrec_t data[NVM2C_RECORD_ARRAY_CAP]; size_t len; };\n\n");
        if (need_concat || need_cast_string || need_substr) emit_nstr_storage(&b);
        if (need_concat) emit_nstr_concat(&b);
        if (need_substr) emit_nstr_substr(&b);
        if (need_char_at) emit_nstr_char_at(&b);
        if (need_cast_string) emit_nstr_from_i64(&b);
        if (need_owned_iarrays) {
            emit_narr_storage(&b);
            emit_narr_new(&b);
            if (need_iarr_lit || need_iarr_push) {
                emit_narr_alloc(&b);
                emit_narr_reserve(&b);
            }
        }
        if (need_iarr_lit) emit_narr_lit(&b);
        if (need_iarr_get) emit_narr_get(&b);
        if (need_iarr_push) emit_narr_push(&b);
        if (need_sarr_lit || need_sarr_new) {
            emit_nsarr_new(&b);
        }
        if (need_sarr_lit) emit_nsarr_lit(&b);
        if (need_sarr_get) emit_nsarr_get(&b);
        if (need_sarr_push) emit_nsarr_push(&b);
        if (need_rarr) emit_nrarr_helpers(&b);
    }

    {
        uint32_t i;
        for (i = 0; i < mod->function_count; i++) {
            if (!reachable[i]) continue;
            emit_prototype(&b, mod, i, kinds, result_kinds);
            if (b.failed) goto fail;
        }
    }
    nvm2c_puts(&b, "\n");

    {
        uint32_t i;
        for (i = 0; i < mod->function_count; i++) {
            if (!reachable[i]) continue;
            emit_function_body(&b, mod, i, kinds, rec_fields, result_kinds,
                               result_fields);
            if (b.failed) goto fail;
        }
    }

    {
        uint32_t entry = mod->header.entry_point;
        if (entry >= mod->function_count) {
            nvm2c_fail(&b, "entry_point %u is not a function", entry);
            goto fail;
        }
        const NvmFunctionEntry *ef = &mod->functions[entry];
        if (ef->arity != 0) {
            nvm2c_fail(&b, "entry function must have arity 0 to become C main");
            goto fail;
        }
        if (!(ef->result_count == 1 && ef->result_tag == TAG_INT)) {
            nvm2c_fail(&b, "entry function must return a single int");
            goto fail;
        }
        char ename[64];
        fn_c_name(mod, entry, ename, sizeof ename);
        if (need_owned_strings || need_owned_iarrays) {
            nvm2c_printf(&b,
                "int main(void) {\n"
                "    int result = (int)%s();\n"
                "%s%s"
                "    return result;\n"
                "}\n",
                ename,
                need_owned_strings ? "    nstr_free_all();\n" : "",
                need_owned_iarrays ? "    narr_free_all();\n" : "");
        } else {
            nvm2c_printf(&b,
                "int main(void) {\n"
                "    return (int)%s();\n"
                "}\n",
                ename);
        }
    }

    if (b.failed) goto fail;
    if (strstr(b.data, "nano_vm") != NULL || strstr(b.data, "nvm_blob") != NULL) {
        nvm2c_fail(&b, "internal error: emitted a VM wrapper rather than structured C");
        goto fail;
    }
    free(kinds);
    free(rec_fields);
    free(result_kinds);
    free(result_fields);
    free(reachable);
    return b.data;

fail:
    free(kinds);
    free(rec_fields);
    free(result_kinds);
    free(result_fields);
    free(reachable);
    free(b.data);
    return NULL;
}
