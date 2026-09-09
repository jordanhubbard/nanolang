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
#define NVM2C_MAX_LOCALS 1024
#define NVM2C_MAX_GLOBALS 256
#define NVM2C_MAX_TEMPS  256
#define NVM2C_MAX_REC_FIELDS 8

#define NVM2C_VK_INT 0
#define NVM2C_VK_STR 1
#define NVM2C_VK_UNK 2
#define NVM2C_VK_ARR 3
#define NVM2C_VK_REC 4
#define NVM2C_VK_SARR 5
#define NVM2C_VK_RARR 6
#define NVM2C_VK_HM 7
#define NVM2C_HM_CAP 64

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

static const char *c_result_type(const NvmFunctionEntry *fn, uint8_t arr_k) {
    if (fn->result_count == 0 || fn->result_tag == TAG_VOID) return "void";
    if (fn->result_count != 1) return NULL;
    if (fn->result_tag == TAG_INT || fn->result_tag == TAG_BOOL) return "int64_t";
    if (fn->result_tag == TAG_STRING) return "const char *";
    if (fn->result_tag == TAG_STRUCT || fn->result_tag == TAG_TUPLE) return "nrec_t";
    if (fn->result_tag == TAG_HASHMAP) return "nhm_t";
    if (fn->result_tag == TAG_ARRAY) {
        if (arr_k == NVM2C_VK_ARR) return "narr_t";
        if (arr_k == NVM2C_VK_SARR) return "nsarr_t";
        if (arr_k == NVM2C_VK_RARR) return "nrarr_t";
        return NULL;
    }
    return NULL;
}

static int result_is_i64(const NvmFunctionEntry *fn) {
    return fn->result_count == 1 &&
           (fn->result_tag == TAG_INT || fn->result_tag == TAG_BOOL);
}

static int result_is_rec(const NvmFunctionEntry *fn) {
    return fn->result_count == 1 &&
           (fn->result_tag == TAG_STRUCT || fn->result_tag == TAG_TUPLE);
}

static int result_is_hm(const NvmFunctionEntry *fn) {
    return fn->result_count == 1 && fn->result_tag == TAG_HASHMAP;
}

static int result_is_arr(const NvmFunctionEntry *fn) {
    return fn->result_count == 1 && fn->result_tag == TAG_ARRAY;
}

static const char *c_local_type(uint8_t kind) {
    if (kind == NVM2C_VK_STR) return "const char *";
    if (kind == NVM2C_VK_ARR) return "narr_t";
    if (kind == NVM2C_VK_SARR) return "nsarr_t";
    if (kind == NVM2C_VK_REC) return "nrec_t";
    if (kind == NVM2C_VK_RARR) return "nrarr_t";
    if (kind == NVM2C_VK_HM) return "nhm_t";
    return "int64_t";
}

typedef enum {
    NVM2C_HOST_UNKNOWN = 0,
    NVM2C_HOST_GETCWD,
    NVM2C_HOST_GETENV,
    NVM2C_HOST_TMP_DIR,
    NVM2C_HOST_SYSTEM,
    NVM2C_HOST_FROM_CHAR,
    NVM2C_HOST_GET_ARGC,
    NVM2C_HOST_GET_ARGV
} Nvm2cHostKind;

static Nvm2cHostKind nvm2c_host_kind(const char *name) {
    if (!name) return NVM2C_HOST_UNKNOWN;
    if (strcmp(name, "vm_getcwd") == 0) return NVM2C_HOST_GETCWD;
    if (strcmp(name, "vm_getenv") == 0) return NVM2C_HOST_GETENV;
    if (strcmp(name, "vm_tmp_dir") == 0) return NVM2C_HOST_TMP_DIR;
    if (strcmp(name, "vm_system") == 0) return NVM2C_HOST_SYSTEM;
    if (strcmp(name, "vm_string_from_char") == 0) return NVM2C_HOST_FROM_CHAR;
    if (strcmp(name, "get_argc") == 0) return NVM2C_HOST_GET_ARGC;
    if (strcmp(name, "get_argv") == 0) return NVM2C_HOST_GET_ARGV;
    return NVM2C_HOST_UNKNOWN;
}

static int nvm2c_host_sig_ok(Nvm2cHostKind k, const NvmImportEntry *imp, const uint8_t *pt) {
    switch (k) {
    case NVM2C_HOST_GETCWD:
    case NVM2C_HOST_TMP_DIR:
        return imp->param_count == 0 && imp->return_type == TAG_STRING;
    case NVM2C_HOST_GETENV:
        return imp->param_count == 1 && imp->return_type == TAG_STRING && pt && pt[0] == TAG_STRING;
    case NVM2C_HOST_SYSTEM:
        return imp->param_count == 1 && (imp->return_type == TAG_INT || imp->return_type == TAG_BOOL)
            && pt && pt[0] == TAG_STRING;
    case NVM2C_HOST_FROM_CHAR:
        return imp->param_count == 1 && imp->return_type == TAG_STRING && pt && pt[0] == TAG_INT;
    case NVM2C_HOST_GET_ARGC:
        return imp->param_count == 0 && (imp->return_type == TAG_INT || imp->return_type == TAG_BOOL);
    case NVM2C_HOST_GET_ARGV:
        return imp->param_count == 1 && imp->return_type == TAG_STRING && pt && pt[0] == TAG_INT;
    default:
        return 0;
    }
}

static Nvm2cHostKind nvm2c_import_host(const NvmModule *mod, uint32_t idx) {
    const NvmImportEntry *imp;
    const char *name;
    const uint8_t *pt;
    Nvm2cHostKind k;
    if (idx >= mod->import_count) return NVM2C_HOST_UNKNOWN;
    imp = &mod->imports[idx];
    name = nvm_get_string(mod, imp->function_name_idx);
    k = nvm2c_host_kind(name);
    pt = mod->import_param_types ? mod->import_param_types[idx] : NULL;
    if (!nvm2c_host_sig_ok(k, imp, pt)) return NVM2C_HOST_UNKNOWN;
    return k;
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
    int from_fn;
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
    slot.from_fn = -1;
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

static void mark_origin(uint8_t *local_kind, uint16_t nloc, int origin, uint8_t kind) {
    if (origin >= 0 && (uint16_t)origin < nloc) {
        local_kind[origin] = kind;
    }
}

static void mark_str_origin(uint8_t *local_kind, uint16_t nloc, int origin) {
    mark_origin(local_kind, nloc, origin, NVM2C_VK_STR);
}

static const uint8_t *fn_rec_k_const(const uint8_t *tab, uint32_t fn, uint16_t slot) {
    return tab + ((size_t)fn * NVM2C_MAX_LOCALS + slot) * NVM2C_MAX_REC_FIELDS;
}

static const uint8_t *fn_result_rec_k(const uint8_t *tab, uint32_t fn) {
    return tab + (size_t)fn * NVM2C_MAX_REC_FIELDS;
}

static int classify_function(Nvm2cBuf *b, const NvmModule *mod, uint32_t idx,
                             uint8_t *local_kind, uint8_t *rec_fields,
                             uint8_t *result_rec_k, uint8_t *result_arr_k,
                             uint8_t *global_kind, uint8_t *global_rec_k) {
    const NvmFunctionEntry *fn = &mod->functions[idx];
    uint16_t nloc = fn->local_count;
    uint16_t i;
    int local_from_fn[NVM2C_MAX_LOCALS];
    memset(local_kind, NVM2C_VK_UNK, nloc);
    for (i = 0; i < nloc; i++) {
        local_from_fn[i] = -1;
    }

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
        case OP_ENUM_VAL:
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
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
            loaded.from_fn = local_from_fn[slot];
            if (loaded.kind == NVM2C_VK_REC || loaded.kind == NVM2C_VK_RARR) {
                memcpy(loaded.rec_k, rec_fields + (size_t)slot * NVM2C_MAX_REC_FIELDS,
                       NVM2C_MAX_REC_FIELDS);
            }
            if (!sim_push_slot(b, idx, stk, &sp, loaded)) return 0;
            break;
        }
        case OP_LOAD_GLOBAL: {
            uint32_t gi = ins.operands[0].u32;
            Nvm2cSimSlot gloaded;
            if (gi >= NVM2C_MAX_GLOBALS) {
                nvm2c_fail(b, "function %u: LOAD_GLOBAL %u out of range", idx, gi);
                return 0;
            }
            memset(&gloaded, 0, sizeof gloaded);
            gloaded.kind = global_kind[gi];
            gloaded.origin = -1;
            gloaded.from_fn = -1;
            if (gloaded.kind == NVM2C_VK_REC || gloaded.kind == NVM2C_VK_RARR) {
                memcpy(gloaded.rec_k, global_rec_k + (size_t)gi * NVM2C_MAX_REC_FIELDS,
                       NVM2C_MAX_REC_FIELDS);
            }
            if (!sim_push_slot(b, idx, stk, &sp, gloaded)) return 0;
            break;
        }
        case OP_STORE_GLOBAL: {
            uint32_t gi = ins.operands[0].u32;
            Nvm2cSimSlot gv;
            if (gi >= NVM2C_MAX_GLOBALS) {
                nvm2c_fail(b, "function %u: STORE_GLOBAL %u out of range", idx, gi);
                return 0;
            }
            if (!sim_pop(b, idx, stk, &sp, &gv)) return 0;
            if (gv.kind == NVM2C_VK_STR) {
                global_kind[gi] = NVM2C_VK_STR;
            } else if (gv.kind == NVM2C_VK_ARR) {
                global_kind[gi] = NVM2C_VK_ARR;
            } else if (gv.kind == NVM2C_VK_SARR) {
                global_kind[gi] = NVM2C_VK_SARR;
            } else if (gv.kind == NVM2C_VK_REC) {
                global_kind[gi] = NVM2C_VK_REC;
                memcpy(global_rec_k + (size_t)gi * NVM2C_MAX_REC_FIELDS,
                       gv.rec_k, NVM2C_MAX_REC_FIELDS);
            } else if (gv.kind == NVM2C_VK_RARR) {
                global_kind[gi] = NVM2C_VK_RARR;
                memcpy(global_rec_k + (size_t)gi * NVM2C_MAX_REC_FIELDS,
                       gv.rec_k, NVM2C_MAX_REC_FIELDS);
            } else if (gv.kind == NVM2C_VK_HM) {
                global_kind[gi] = NVM2C_VK_HM;
            } else if (gv.kind == NVM2C_VK_INT && global_kind[gi] != NVM2C_VK_STR
                       && global_kind[gi] != NVM2C_VK_ARR
                       && global_kind[gi] != NVM2C_VK_SARR
                       && global_kind[gi] != NVM2C_VK_REC
                       && global_kind[gi] != NVM2C_VK_RARR
                       && global_kind[gi] != NVM2C_VK_HM) {
                global_kind[gi] = NVM2C_VK_INT;
            }
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
            } else if (v.kind == NVM2C_VK_HM) {
                local_kind[slot] = NVM2C_VK_HM;
            } else if (v.kind == NVM2C_VK_INT && local_kind[slot] != NVM2C_VK_STR
                       && local_kind[slot] != NVM2C_VK_ARR
                       && local_kind[slot] != NVM2C_VK_SARR
                       && local_kind[slot] != NVM2C_VK_REC
                       && local_kind[slot] != NVM2C_VK_RARR
                       && local_kind[slot] != NVM2C_VK_HM) {
                local_kind[slot] = NVM2C_VK_INT;
            }
            local_from_fn[slot] = v.from_fn;
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
        case OP_LT:
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
            mark_str_origin(local_kind, nloc, v.origin);
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_STR_CONCAT: {
            Nvm2cSimSlot rhs, lhs;
            if (!sim_pop(b, idx, stk, &sp, &rhs)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &lhs)) return 0;
            mark_str_origin(local_kind, nloc, rhs.origin);
            mark_str_origin(local_kind, nloc, lhs.origin);
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
            mark_str_origin(local_kind, nloc, s.origin);
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_STR, -1)) return 0;
            break;
        }
        case OP_STR_TRIM: {
            Nvm2cSimSlot s;
            if (!sim_pop(b, idx, stk, &sp, &s)) return 0;
            mark_str_origin(local_kind, nloc, s.origin);
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_STR, -1)) return 0;
            break;
        }
        case OP_STR_TO_LOWER:
        case OP_STR_TO_UPPER: {
            Nvm2cSimSlot s;
            if (!sim_pop(b, idx, stk, &sp, &s)) return 0;
            mark_str_origin(local_kind, nloc, s.origin);
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_STR, -1)) return 0;
            break;
        }
        case OP_STR_REPLACE: {
            Nvm2cSimSlot neu, old, s;
            if (!sim_pop(b, idx, stk, &sp, &neu)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &old)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &s)) return 0;
            mark_str_origin(local_kind, nloc, neu.origin);
            mark_str_origin(local_kind, nloc, old.origin);
            mark_str_origin(local_kind, nloc, s.origin);
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_STR, -1)) return 0;
            break;
        }
        case OP_STR_SPLIT: {
            Nvm2cSimSlot delim, s;
            if (!sim_pop(b, idx, stk, &sp, &delim)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &s)) return 0;
            mark_str_origin(local_kind, nloc, delim.origin);
            mark_str_origin(local_kind, nloc, s.origin);
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_SARR, -1)) return 0;
            break;
        }
        case OP_STR_CONTAINS:
        case OP_STR_STARTS_WITH:
        case OP_STR_ENDS_WITH: {
            Nvm2cSimSlot needle, hay;
            if (!sim_pop(b, idx, stk, &sp, &needle)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &hay)) return 0;
            mark_str_origin(local_kind, nloc, needle.origin);
            mark_str_origin(local_kind, nloc, hay.origin);
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_STR_CHAR_AT: {
            Nvm2cSimSlot ix, s;
            if (!sim_pop(b, idx, stk, &sp, &ix)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &s)) return 0;
            (void)ix;
            mark_str_origin(local_kind, nloc, s.origin);
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
        case OP_CAST_INT: {
            Nvm2cSimSlot v;
            if (!sim_pop(b, idx, stk, &sp, &v)) return 0;
            if (v.kind != NVM2C_VK_STR && v.kind != NVM2C_VK_INT) {
                nvm2c_fail(b, "CAST_INT only supports int or string");
                return 0;
            }
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_EQ:
        case OP_NE: {
            Nvm2cSimSlot rhs, lhs;
            if (!sim_pop(b, idx, stk, &sp, &rhs)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &lhs)) return 0;
            if (lhs.kind != NVM2C_VK_INT || rhs.kind != NVM2C_VK_INT) {
                mark_str_origin(local_kind, nloc, lhs.origin);
                mark_str_origin(local_kind, nloc, rhs.origin);
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
            } else {
                nvm2c_fail(b, "function %u: ARR_NEW only supports int or string elements", idx);
                return 0;
            }
            break;
        }
        case OP_ARR_LEN: {
            Nvm2cSimSlot v;
            if (!sim_pop(b, idx, stk, &sp, &v)) return 0;
            if (v.kind == NVM2C_VK_SARR) {
                mark_origin(local_kind, nloc, v.origin, NVM2C_VK_SARR);
            } else if (v.kind == NVM2C_VK_RARR) {
                mark_origin(local_kind, nloc, v.origin, NVM2C_VK_RARR);
            } else {
                mark_origin(local_kind, nloc, v.origin, NVM2C_VK_ARR);
            }
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_ARR_GET: {
            Nvm2cSimSlot ix, arr;
            if (!sim_pop(b, idx, stk, &sp, &ix)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &arr)) return 0;
            (void)ix;
            if (arr.kind == NVM2C_VK_SARR) {
                mark_origin(local_kind, nloc, arr.origin, NVM2C_VK_SARR);
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_STR, -1)) return 0;
            } else if (arr.kind == NVM2C_VK_RARR) {
                Nvm2cSimSlot rec;
                memset(&rec, 0, sizeof rec);
                rec.kind = NVM2C_VK_REC;
                rec.origin = -1;
                rec.from_fn = -1;
                memcpy(rec.rec_k, arr.rec_k, NVM2C_MAX_REC_FIELDS);
                mark_origin(local_kind, nloc, arr.origin, NVM2C_VK_RARR);
                if (!sim_push_slot(b, idx, stk, &sp, rec)) return 0;
            } else {
                mark_origin(local_kind, nloc, arr.origin, NVM2C_VK_ARR);
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            }
            break;
        }
        case OP_ARR_PUSH: {
            Nvm2cSimSlot val, arr;
            if (!sim_pop(b, idx, stk, &sp, &val)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &arr)) return 0;
            if (val.kind == NVM2C_VK_REC || arr.kind == NVM2C_VK_RARR) {
                Nvm2cSimSlot out;
                memset(&out, 0, sizeof out);
                out.kind = NVM2C_VK_RARR;
                out.origin = arr.origin;
                out.from_fn = arr.from_fn;
                if (val.kind == NVM2C_VK_REC) {
                    memcpy(out.rec_k, val.rec_k, NVM2C_MAX_REC_FIELDS);
                } else {
                    memcpy(out.rec_k, arr.rec_k, NVM2C_MAX_REC_FIELDS);
                }
                mark_origin(local_kind, nloc, arr.origin, NVM2C_VK_RARR);
                if (arr.origin >= 0 && (uint16_t)arr.origin < nloc) {
                    memcpy(rec_fields + (size_t)arr.origin * NVM2C_MAX_REC_FIELDS,
                           out.rec_k, NVM2C_MAX_REC_FIELDS);
                }
                if (arr.from_fn >= 0 && (uint32_t)arr.from_fn < mod->function_count) {
                    result_arr_k[arr.from_fn] = NVM2C_VK_RARR;
                    memcpy(result_rec_k + (size_t)arr.from_fn * NVM2C_MAX_REC_FIELDS,
                           out.rec_k, NVM2C_MAX_REC_FIELDS);
                }
                if (!sim_push_slot(b, idx, stk, &sp, out)) return 0;
            } else if (val.kind == NVM2C_VK_STR || arr.kind == NVM2C_VK_SARR) {
                mark_origin(local_kind, nloc, arr.origin, NVM2C_VK_SARR);
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_SARR, -1)) return 0;
            } else {
                mark_origin(local_kind, nloc, arr.origin, NVM2C_VK_ARR);
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_ARR, -1)) return 0;
            }
            break;
        }
        case OP_ARR_SET: {
            Nvm2cSimSlot val, ix, arr;
            if (!sim_pop(b, idx, stk, &sp, &val)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &ix)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &arr)) return 0;
            if (ix.kind != NVM2C_VK_INT) {
                nvm2c_fail(b, "function %u: ARR_SET index must be int", idx);
                return 0;
            }
            if (arr.kind == NVM2C_VK_ARR && val.kind == NVM2C_VK_INT) {
                mark_origin(local_kind, nloc, arr.origin, NVM2C_VK_ARR);
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_ARR, arr.origin)) return 0;
            } else if (arr.kind == NVM2C_VK_SARR && val.kind == NVM2C_VK_STR) {
                mark_origin(local_kind, nloc, arr.origin, NVM2C_VK_SARR);
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_SARR, arr.origin)) return 0;
            } else if (arr.kind == NVM2C_VK_RARR && val.kind == NVM2C_VK_REC) {
                Nvm2cSimSlot out;
                memset(&out, 0, sizeof out);
                out.kind = NVM2C_VK_RARR;
                out.origin = arr.origin;
                memcpy(out.rec_k, val.rec_k, NVM2C_MAX_REC_FIELDS);
                mark_origin(local_kind, nloc, arr.origin, NVM2C_VK_RARR);
                if (arr.origin >= 0 && (uint16_t)arr.origin < nloc) {
                    memcpy(rec_fields + (size_t)arr.origin * NVM2C_MAX_REC_FIELDS,
                           out.rec_k, NVM2C_MAX_REC_FIELDS);
                }
                if (!sim_push_slot(b, idx, stk, &sp, out)) return 0;
            } else {
                nvm2c_fail(b, "function %u: ARR_SET only supports int, string, or record arrays", idx);
                return 0;
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
            if (count > NVM2C_MAX_REC_FIELDS) {
                nvm2c_fail(b, "function %u: AGG_PACK has too many fields", idx);
                return 0;
            }
            for (ai = 0; ai < count; ai++) {
                Nvm2cSimSlot v;
                if (!sim_pop(b, idx, stk, &sp, &v)) return 0;
                if (v.kind != NVM2C_VK_INT && v.kind != NVM2C_VK_STR
                    && v.kind != NVM2C_VK_REC && v.kind != NVM2C_VK_HM) {
                    nvm2c_fail(b, "function %u: AGG_PACK fields must be int, string, record, or hashmap", idx);
                    return 0;
                }
                packed.rec_k[count - 1 - ai] = v.kind;
            }
            if (!sim_push_slot(b, idx, stk, &sp, packed)) return 0;
            break;
        }
        case OP_HM_NEW: {
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_HM, -1)) return 0;
            break;
        }
        case OP_HM_HAS: {
            Nvm2cSimSlot key, map;
            if (!sim_pop(b, idx, stk, &sp, &key)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &map)) return 0;
            mark_origin(local_kind, nloc, map.origin, NVM2C_VK_HM);
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
            break;
        }
        case OP_HM_SET: {
            Nvm2cSimSlot val, key, map;
            if (!sim_pop(b, idx, stk, &sp, &val)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &key)) return 0;
            if (!sim_pop(b, idx, stk, &sp, &map)) return 0;
            mark_origin(local_kind, nloc, map.origin, NVM2C_VK_HM);
            (void)val;
            (void)key;
            if (!sim_push(b, idx, stk, &sp, NVM2C_VK_HM, map.origin)) return 0;
            break;
        }
        case OP_AGG_GET: {
            Nvm2cSimSlot rec;
            uint16_t fi = ins.operands[0].u16;
            uint8_t fk;
            if (!sim_pop(b, idx, stk, &sp, &rec)) return 0;
            mark_origin(local_kind, nloc, rec.origin, NVM2C_VK_REC);
            if (fi >= NVM2C_MAX_REC_FIELDS) {
                nvm2c_fail(b, "function %u: AGG_GET field is out of range", idx);
                return 0;
            }
            fk = rec.rec_k[fi];
            if (fk == NVM2C_VK_STR) {
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_STR, -1)) return 0;
            } else if (fk == NVM2C_VK_REC) {
                Nvm2cSimSlot inner;
                memset(&inner, 0, sizeof inner);
                inner.kind = NVM2C_VK_REC;
                inner.origin = -1;
                if (!sim_push_slot(b, idx, stk, &sp, inner)) return 0;
            } else if (fk == NVM2C_VK_HM) {
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_HM, -1)) return 0;
            } else {
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
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
                (void)arg;
            }
            if (ins.opcode == OP_CALL) {
                if (cf->result_count == 1 && cf->result_tag == TAG_STRING) {
                    if (!sim_push(b, idx, stk, &sp, NVM2C_VK_STR, -1)) return 0;
                } else if (result_is_i64(cf)) {
                    if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
                } else if (result_is_rec(cf)) {
                    Nvm2cSimSlot rec;
                    memset(&rec, 0, sizeof rec);
                    rec.kind = NVM2C_VK_REC;
                    rec.origin = -1;
                    rec.from_fn = -1;
                    memcpy(rec.rec_k, fn_result_rec_k(result_rec_k, callee),
                           NVM2C_MAX_REC_FIELDS);
                    if (!sim_push_slot(b, idx, stk, &sp, rec)) return 0;
                } else if (result_is_arr(cf)) {
                    uint8_t ak = result_arr_k[callee];
                    if (ak == NVM2C_VK_SARR) {
                        if (!sim_push(b, idx, stk, &sp, NVM2C_VK_SARR, -1)) return 0;
                    } else if (ak == NVM2C_VK_RARR) {
                        Nvm2cSimSlot arr;
                        memset(&arr, 0, sizeof arr);
                        arr.kind = NVM2C_VK_RARR;
                        arr.origin = -1;
                        arr.from_fn = (int)callee;
                        memcpy(arr.rec_k, fn_result_rec_k(result_rec_k, callee),
                               NVM2C_MAX_REC_FIELDS);
                        if (!sim_push_slot(b, idx, stk, &sp, arr)) return 0;
                    } else if (ak == NVM2C_VK_ARR) {
                        Nvm2cSimSlot arr;
                        memset(&arr, 0, sizeof arr);
                        arr.kind = NVM2C_VK_ARR;
                        arr.origin = -1;
                        arr.from_fn = (int)callee;
                        if (!sim_push_slot(b, idx, stk, &sp, arr)) return 0;
                    }
                } else if (result_is_hm(cf)) {
                    if (!sim_push(b, idx, stk, &sp, NVM2C_VK_HM, -1)) return 0;
                }
            }
            break;
        }
        case OP_CALL_EXTERN: {
            uint32_t imp_idx = ins.operands[0].u32;
            Nvm2cHostKind hk = nvm2c_import_host(mod, imp_idx);
            uint16_t pi;
            if (hk == NVM2C_HOST_UNKNOWN) {
                nvm2c_fail(b, "function %u: CALL_EXTERN import %u has no host ABI", idx, imp_idx);
                return 0;
            }
            for (pi = 0; pi < mod->imports[imp_idx].param_count; pi++) {
                Nvm2cSimSlot arg;
                if (!sim_pop(b, idx, stk, &sp, &arg)) return 0;
                (void)arg;
            }
            if (mod->imports[imp_idx].return_type == TAG_STRING) {
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_STR, -1)) return 0;
            } else if (mod->imports[imp_idx].return_type == TAG_INT
                       || mod->imports[imp_idx].return_type == TAG_BOOL) {
                if (!sim_push(b, idx, stk, &sp, NVM2C_VK_INT, -1)) return 0;
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
                 fn->result_tag == TAG_STRING || fn->result_tag == TAG_STRUCT ||
                 fn->result_tag == TAG_TUPLE ||
                 fn->result_tag == TAG_ARRAY || fn->result_tag == TAG_HASHMAP) &&
                sp > 0) {
                Nvm2cSimSlot v;
                if (!sim_pop(b, idx, stk, &sp, &v)) return 0;
                if (fn->result_tag == TAG_STRING) {
                    mark_str_origin(local_kind, nloc, v.origin);
                } else if (fn->result_tag == TAG_STRUCT || fn->result_tag == TAG_TUPLE) {
                    mark_origin(local_kind, nloc, v.origin, NVM2C_VK_REC);
                    memcpy(result_rec_k + (size_t)idx * NVM2C_MAX_REC_FIELDS,
                           v.rec_k, NVM2C_MAX_REC_FIELDS);
                } else if (fn->result_tag == TAG_HASHMAP) {
                    mark_origin(local_kind, nloc, v.origin, NVM2C_VK_HM);
                } else if (fn->result_tag == TAG_ARRAY) {
                    mark_origin(local_kind, nloc, v.origin, v.kind);
                    if (result_arr_k[idx] != NVM2C_VK_RARR) {
                        result_arr_k[idx] = v.kind;
                    }
                    if (v.kind == NVM2C_VK_RARR) {
                        memcpy(result_rec_k + (size_t)idx * NVM2C_MAX_REC_FIELDS,
                               v.rec_k, NVM2C_MAX_REC_FIELDS);
                    }
                }
            }
            break;
        }
        default:
            break;
        }
        if (b->failed) return 0;
    }

    for (i = 0; i < nloc; i++) {
        if (local_kind[i] == NVM2C_VK_UNK) local_kind[i] = NVM2C_VK_INT;
    }
    return 1;
}

static void emit_prototype(Nvm2cBuf *b, const NvmModule *mod, uint32_t idx,
                           const uint8_t *kinds, const uint8_t *result_arr_k) {
    const NvmFunctionEntry *fn = &mod->functions[idx];
    const char *rt = c_result_type(fn, result_arr_k[idx]);
    if (!rt) {
        nvm2c_fail(b, "function %u: only void, a single int, string, record, array, or hashmap result is supported",
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
            nvm2c_printf(b, "%s a%u", c_local_type(fn_local_kind(kinds, idx, i)), (unsigned)i);
        }
    }
    nvm2c_puts(b, ");\n");
}

typedef struct {
    int slots[NVM2C_MAX_STACK];
    uint8_t kinds[NVM2C_MAX_STACK];
    uint8_t rec_k[NVM2C_MAX_TEMPS][NVM2C_MAX_REC_FIELDS];
    uint8_t rec_k_arr[NVM2C_MAX_TEMPS][NVM2C_MAX_REC_FIELDS];
    int sp;
    int next_temp;
    int next_str;
    int next_arr;
    int next_sarr;
    int next_rec;
    int next_rarr;
    int next_hm;
} Nvm2cStack;

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
    st->slots[st->sp] = r;
    st->kinds[st->sp] = NVM2C_VK_REC;
    st->sp++;
    return r;
}

static int stack_push_rarr(Nvm2cBuf *b, Nvm2cStack *st, const char *rhs) {
    if (st->sp >= NVM2C_MAX_STACK) {
        nvm2c_fail(b, "operand stack overflow");
        return -1;
    }
    if (st->next_rarr >= NVM2C_MAX_TEMPS) {
        nvm2c_fail(b, "too many record-array temporaries");
        return -1;
    }
    int a = st->next_rarr++;
    nvm2c_printf(b, "    ra[%d] = %s;\n", a, rhs);
    st->slots[st->sp] = a;
    st->kinds[st->sp] = NVM2C_VK_RARR;
    st->sp++;
    return a;
}

static int stack_push_hm(Nvm2cBuf *b, Nvm2cStack *st, const char *rhs) {
    if (st->sp >= NVM2C_MAX_STACK) {
        nvm2c_fail(b, "operand stack overflow");
        return -1;
    }
    if (st->next_hm >= NVM2C_MAX_TEMPS) {
        nvm2c_fail(b, "too many hashmap temporaries");
        return -1;
    }
    int h = st->next_hm++;
    nvm2c_printf(b, "    h[%d] = %s;\n", h, rhs);
    st->slots[st->sp] = h;
    st->kinds[st->sp] = NVM2C_VK_HM;
    st->sp++;
    return h;
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

static int stack_pop(Nvm2cBuf *b, Nvm2cStack *st) {
    return stack_pop_kind(b, st, NULL);
}

static int stack_pop_expect(Nvm2cBuf *b, Nvm2cStack *st, uint8_t kind, const char *what) {
    uint8_t got = NVM2C_VK_INT;
    int slot = stack_pop_kind(b, st, &got);
    if (b->failed) return -1;
    if (got != kind) {
        const char *want = "int";
        if (kind == NVM2C_VK_STR) want = "string";
        else if (kind == NVM2C_VK_ARR) want = "array";
        else if (kind == NVM2C_VK_SARR) want = "string array";
        else if (kind == NVM2C_VK_REC) want = "record";
        else if (kind == NVM2C_VK_RARR) want = "record array";
        else if (kind == NVM2C_VK_HM) want = "hashmap";
        nvm2c_fail(b, "%s: expected %s value", what, want);
        return -1;
    }
    return slot;
}

static void emit_binop(Nvm2cBuf *b, Nvm2cStack *st, const char *op) {
    int rhs = stack_pop_expect(b, st, NVM2C_VK_INT, "binary op rhs");
    int lhs = stack_pop_expect(b, st, NVM2C_VK_INT, "binary op lhs");
    if (b->failed) return;
    char expr[80];
    snprintf(expr, sizeof expr, "(t[%d] %s t[%d])", lhs, op, rhs);
    stack_push_temp(b, st, expr);
}

static void emit_unop(Nvm2cBuf *b, Nvm2cStack *st, const char *prefix) {
    int x = stack_pop_expect(b, st, NVM2C_VK_INT, "unary op");
    if (b->failed) return;
    char expr[64];
    snprintf(expr, sizeof expr, "(%s t[%d])", prefix, x);
    stack_push_temp(b, st, expr);
}

static void stack_keep_high_water(Nvm2cStack *st, const Nvm2cStack *other) {
    if (other->next_temp > st->next_temp) st->next_temp = other->next_temp;
    if (other->next_str > st->next_str) st->next_str = other->next_str;
    if (other->next_arr > st->next_arr) st->next_arr = other->next_arr;
    if (other->next_sarr > st->next_sarr) st->next_sarr = other->next_sarr;
    if (other->next_rec > st->next_rec) st->next_rec = other->next_rec;
    if (other->next_rarr > st->next_rarr) st->next_rarr = other->next_rarr;
    if (other->next_hm > st->next_hm) st->next_hm = other->next_hm;
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
            memcpy(joins[tgt].rec_k_arr[joins[tgt].slots[i]],
                   st->rec_k_arr[st->slots[i]], NVM2C_MAX_REC_FIELDS);
        } else if (st->kinds[i] == NVM2C_VK_HM) {
            nvm2c_printf(b, "    h[%d] = h[%d];\n", joins[tgt].slots[i], st->slots[i]);
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
                             const uint8_t *result_arr_k, char *call, size_t call_sz) {
    if (callee >= mod->function_count) {
        nvm2c_fail(b, "function %u: CALL target %u is out of range", idx, callee);
        return 0;
    }
    const NvmFunctionEntry *cf = &mod->functions[callee];
    if (c_result_type(cf, result_arr_k[callee]) == NULL) {
        nvm2c_fail(b, "function %u: CALL target %u has an unsupported result", idx, callee);
        return 0;
    }
    int args[NVM2C_MAX_LOCALS];
    uint8_t argk[NVM2C_MAX_LOCALS];
    int i;
    for (i = (int)cf->arity - 1; i >= 0; i--) {
        uint8_t pk = fn_local_kind(kinds, callee, (uint16_t)i);
        argk[i] = pk;
        args[i] = stack_pop_expect(b, st, pk, "CALL argument");
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
        } else if (argk[a] == NVM2C_VK_HM) {
            pos += (size_t)snprintf(call + pos, call_sz - pos, "h[%d]", args[a]);
        } else {
            pos += (size_t)snprintf(call + pos, call_sz - pos, "t[%d]", args[a]);
        }
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
                               const uint8_t *result_rec_k, const uint8_t *result_arr_k,
                               const uint8_t *global_kind, const uint8_t *global_rec_k) {
    const NvmFunctionEntry *fn = &mod->functions[idx];
    const char *rt = c_result_type(fn, result_arr_k[idx]);
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
            nvm2c_printf(b, "%s a%u", c_local_type(fn_local_kind(kinds, idx, i)), (unsigned)i);
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
        } else if (lk == NVM2C_VK_HM) {
            nvm2c_printf(b, "    nhm_t l%u = {0};\n", (unsigned)i);
        } else {
            nvm2c_printf(b, "    int64_t l%u = 0;\n", (unsigned)i);
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
    nvm2c_printf(b, "    nrec_t r[%d] = {0};\n", NVM2C_MAX_TEMPS);
    nvm2c_puts(b, "    (void)r;\n");
    nvm2c_printf(b, "    nrarr_t ra[%d] = {0};\n", NVM2C_MAX_TEMPS);
    nvm2c_puts(b, "    (void)ra;\n");
    nvm2c_printf(b, "    nhm_t h[%d] = {0};\n", NVM2C_MAX_TEMPS);
    nvm2c_puts(b, "    (void)h;\n");

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
        case OP_ENUM_VAL: {
            char rhs[32];
            snprintf(rhs, sizeof rhs, "%lldLL", (long long)ins.operands[1].u16);
            stack_push_temp(b, &st, rhs);
            break;
        }
        case OP_PUSH_BOOL: {
            char rhs[8];
            snprintf(rhs, sizeof rhs, "%dLL", ins.operands[0].u8 ? 1 : 0);
            stack_push_temp(b, &st, rhs);
            break;
        }
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
                        int na = stack_push_rarr(b, &st, rhs);
                        if (na >= 0) {
                            memcpy(st.rec_k_arr[na], st.rec_k_arr[src], NVM2C_MAX_REC_FIELDS);
                        }
                    }
                } else if (k == NVM2C_VK_HM) {
                    snprintf(rhs, sizeof rhs, "h[%d]", src);
                    stack_push_hm(b, &st, rhs);
                } else {
                    snprintf(rhs, sizeof rhs, "t[%d]", src);
                    stack_push_temp(b, &st, rhs);
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
            int slot = stack_pop_kind(b, &st, &k);
            int nl = (ins.opcode == OP_PRINTLN);
            if (b->failed) goto done;
            if (k == NVM2C_VK_INT) {
                if (nl) {
                    nvm2c_printf(b, "    printf(\"%%lld\\n\", (long long)t[%d]);\n", slot);
                } else {
                    nvm2c_printf(b, "    printf(\"%%lld\", (long long)t[%d]);\n", slot);
                }
            } else if (k == NVM2C_VK_STR) {
                nvm2c_printf(b, "    fputs(s[%d] ? s[%d] : \"\", stdout);\n", slot, slot);
                if (nl) nvm2c_puts(b, "    fputc('\\n', stdout);\n");
            } else {
                nvm2c_fail(b, "function %u: PRINT of arrays and records is refused", idx);
                goto done;
            }
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
            int x = stack_pop_kind(b, &st, &kx);
            int y = stack_pop_kind(b, &st, &ky);
            if (b->failed) goto done;
            st.slots[st.sp] = x;
            st.kinds[st.sp] = kx;
            st.sp++;
            st.slots[st.sp] = y;
            st.kinds[st.sp] = ky;
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
                    memcpy(st.rec_k_arr[a], fn_rec_k_const(rec_fields, idx, slot),
                           NVM2C_MAX_REC_FIELDS);
                }
            } else if (fn_local_kind(kinds, idx, slot) == NVM2C_VK_HM) {
                stack_push_hm(b, &st, rhs);
            } else {
                stack_push_temp(b, &st, rhs);
            }
            break;
        }
        case OP_LOAD_GLOBAL: {
            uint32_t gi = ins.operands[0].u32;
            uint8_t gk;
            char grhs[32];
            if (gi >= NVM2C_MAX_GLOBALS) {
                nvm2c_fail(b, "function %u: LOAD_GLOBAL %u out of range", idx, gi);
                goto done;
            }
            gk = global_kind[gi];
            if (gk == NVM2C_VK_UNK) {
                nvm2c_fail(b, "function %u: LOAD_GLOBAL %u is unclassified", idx, gi);
                goto done;
            }
            snprintf(grhs, sizeof grhs, "g%u", gi);
            if (gk == NVM2C_VK_STR) {
                stack_push_str(b, &st, grhs);
            } else if (gk == NVM2C_VK_ARR) {
                stack_push_arr(b, &st, grhs);
            } else if (gk == NVM2C_VK_SARR) {
                stack_push_sarr(b, &st, grhs);
            } else if (gk == NVM2C_VK_REC) {
                int r = stack_push_rec(b, &st, grhs);
                if (r >= 0) {
                    memcpy(st.rec_k[r], global_rec_k + (size_t)gi * NVM2C_MAX_REC_FIELDS,
                           NVM2C_MAX_REC_FIELDS);
                }
            } else if (gk == NVM2C_VK_RARR) {
                int a = stack_push_rarr(b, &st, grhs);
                if (a >= 0) {
                    memcpy(st.rec_k_arr[a], global_rec_k + (size_t)gi * NVM2C_MAX_REC_FIELDS,
                           NVM2C_MAX_REC_FIELDS);
                }
            } else if (gk == NVM2C_VK_HM) {
                stack_push_hm(b, &st, grhs);
            } else {
                stack_push_temp(b, &st, grhs);
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
                int t = stack_pop_expect(b, &st, expect, "STORE_LOCAL");
                if (b->failed) goto done;
                if (expect == NVM2C_VK_STR) {
                    nvm2c_printf(b, "    l%u = s[%d];\n", (unsigned)slot, t);
                } else if (expect == NVM2C_VK_ARR) {
                    nvm2c_printf(b, "    l%u = a[%d];\n", (unsigned)slot, t);
                } else if (expect == NVM2C_VK_SARR) {
                    nvm2c_printf(b, "    l%u = sa[%d];\n", (unsigned)slot, t);
                } else if (expect == NVM2C_VK_REC) {
                    nvm2c_printf(b, "    l%u = r[%d];\n", (unsigned)slot, t);
                } else if (expect == NVM2C_VK_RARR) {
                    nvm2c_printf(b, "    l%u = ra[%d];\n", (unsigned)slot, t);
                } else if (expect == NVM2C_VK_HM) {
                    nvm2c_printf(b, "    l%u = h[%d];\n", (unsigned)slot, t);
                } else {
                    nvm2c_printf(b, "    l%u = t[%d];\n", (unsigned)slot, t);
                }
            }
            break;
        }
        case OP_STORE_GLOBAL: {
            uint32_t gi = ins.operands[0].u32;
            uint8_t expect;
            int t;
            if (gi >= NVM2C_MAX_GLOBALS) {
                nvm2c_fail(b, "function %u: STORE_GLOBAL %u out of range", idx, gi);
                goto done;
            }
            expect = global_kind[gi];
            if (expect == NVM2C_VK_UNK) {
                nvm2c_fail(b, "function %u: STORE_GLOBAL %u is unclassified", idx, gi);
                goto done;
            }
            t = stack_pop_expect(b, &st, expect, "STORE_GLOBAL");
            if (b->failed) goto done;
            if (expect == NVM2C_VK_STR) {
                nvm2c_printf(b, "    g%u = s[%d];\n", gi, t);
            } else if (expect == NVM2C_VK_ARR) {
                nvm2c_printf(b, "    g%u = a[%d];\n", gi, t);
            } else if (expect == NVM2C_VK_SARR) {
                nvm2c_printf(b, "    g%u = sa[%d];\n", gi, t);
            } else if (expect == NVM2C_VK_REC) {
                nvm2c_printf(b, "    g%u = r[%d];\n", gi, t);
            } else if (expect == NVM2C_VK_RARR) {
                nvm2c_printf(b, "    g%u = ra[%d];\n", gi, t);
            } else if (expect == NVM2C_VK_HM) {
                nvm2c_printf(b, "    g%u = h[%d];\n", gi, t);
            } else {
                nvm2c_printf(b, "    g%u = t[%d];\n", gi, t);
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
            emit_unop(b, &st, "!");
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
        case OP_LT:
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
            emit_binop(b, &st, "&&");
            break;
        case OP_BOOL_OR:
            emit_binop(b, &st, "||");
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
        case OP_STR_TRIM: {
            int s = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_TRIM");
            if (b->failed) goto done;
            char expr[48];
            snprintf(expr, sizeof expr, "nstr_trim(s[%d])", s);
            stack_push_str(b, &st, expr);
            break;
        }
        case OP_STR_TO_LOWER: {
            int s = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_TO_LOWER");
            if (b->failed) goto done;
            char expr[48];
            snprintf(expr, sizeof expr, "nstr_to_lower(s[%d])", s);
            stack_push_str(b, &st, expr);
            break;
        }
        case OP_STR_TO_UPPER: {
            int s = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_TO_UPPER");
            if (b->failed) goto done;
            char expr[48];
            snprintf(expr, sizeof expr, "nstr_to_upper(s[%d])", s);
            stack_push_str(b, &st, expr);
            break;
        }
        case OP_STR_REPLACE: {
            int neu = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_REPLACE new");
            int old = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_REPLACE old");
            int s = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_REPLACE");
            if (b->failed) goto done;
            char expr[80];
            snprintf(expr, sizeof expr, "nstr_replace(s[%d], s[%d], s[%d])", s, old, neu);
            stack_push_str(b, &st, expr);
            break;
        }
        case OP_STR_SPLIT: {
            int delim = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_SPLIT delim");
            int s = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_SPLIT");
            if (b->failed) goto done;
            char expr[64];
            snprintf(expr, sizeof expr, "nstr_split(s[%d], s[%d])", s, delim);
            stack_push_sarr(b, &st, expr);
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
        case OP_STR_STARTS_WITH: {
            int pre = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_STARTS_WITH prefix");
            int hay = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_STARTS_WITH");
            if (b->failed) goto done;
            char expr[80];
            snprintf(expr, sizeof expr, "nstr_starts_with(s[%d], s[%d])", hay, pre);
            stack_push_temp(b, &st, expr);
            break;
        }
        case OP_STR_ENDS_WITH: {
            int suf = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_ENDS_WITH suffix");
            int hay = stack_pop_expect(b, &st, NVM2C_VK_STR, "STR_ENDS_WITH");
            if (b->failed) goto done;
            char expr[80];
            snprintf(expr, sizeof expr, "nstr_ends_with(s[%d], s[%d])", hay, suf);
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
            int v = stack_pop_expect(b, &st, NVM2C_VK_INT, "CAST_STRING");
            if (b->failed) goto done;
            char expr[48];
            snprintf(expr, sizeof expr, "nstr_from_i64(t[%d])", v);
            stack_push_str(b, &st, expr);
            break;
        }
        case OP_CAST_INT: {
            uint8_t vk = NVM2C_VK_INT;
            int v = stack_pop_kind(b, &st, &vk);
            if (b->failed) goto done;
            if (vk == NVM2C_VK_STR) {
                char expr[48];
                snprintf(expr, sizeof expr, "nstr_to_i64(s[%d])", v);
                stack_push_temp(b, &st, expr);
            } else if (vk == NVM2C_VK_INT) {
                char expr[32];
                snprintf(expr, sizeof expr, "t[%d]", v);
                stack_push_temp(b, &st, expr);
            } else {
                nvm2c_fail(b, "CAST_INT only supports int or string");
                goto done;
            }
            break;
        }
        case OP_ARR_NEW: {
            uint8_t tag = ins.operands[0].u8;
            int as_sarr = (tag == TAG_STRING);
            if (tag != TAG_INT && tag != TAG_STRING) {
                nvm2c_fail(b, "function %u: ARR_NEW only supports int or string elements", idx);
                goto done;
            }
            if (tag == TAG_INT) {
                DecodedInstruction nxt;
                uint32_t nn = isa_decode(code + pc, remaining - pc, &nxt);
                if (nn != 0 && nxt.opcode == OP_STORE_LOCAL) {
                    uint16_t slot = nxt.operands[0].u16;
                    if (fn_local_kind(kinds, idx, slot) == NVM2C_VK_SARR) {
                        as_sarr = 1;
                    } else if (fn_local_kind(kinds, idx, slot) == NVM2C_VK_RARR) {
                        as_sarr = 2;
                    }
                }
                if (as_sarr == 0 && result_arr_k[idx] == NVM2C_VK_RARR) {
                    as_sarr = 2;
                }
            }
            if (as_sarr == 1) {
                stack_push_sarr(b, &st, "nsarr_new()");
            } else if (as_sarr == 2) {
                stack_push_rarr(b, &st, "nrarr_new()");
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
                stack_push_temp(b, &st, expr);
            } else if (ak == NVM2C_VK_SARR) {
                char expr[80];
                snprintf(expr, sizeof expr, "nsarr_get(sa[%d], t[%d])", arr, ix);
                stack_push_str(b, &st, expr);
            } else if (ak == NVM2C_VK_RARR) {
                char expr[80];
                int r;
                snprintf(expr, sizeof expr, "nrarr_get(ra[%d], t[%d])", arr, ix);
                r = stack_push_rec(b, &st, expr);
                if (r >= 0) {
                    memcpy(st.rec_k[r], st.rec_k_arr[arr], NVM2C_MAX_REC_FIELDS);
                }
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
                int na;
                snprintf(expr, sizeof expr, "nrarr_push(ra[%d], r[%d])", arr, val);
                na = stack_push_rarr(b, &st, expr);
                if (na >= 0) {
                    memcpy(st.rec_k_arr[na], st.rec_k[val], NVM2C_MAX_REC_FIELDS);
                }
            } else {
                nvm2c_fail(b, "function %u: ARR_PUSH type mismatch", idx);
                goto done;
            }
            break;
        }
        case OP_ARR_SET: {
            uint8_t vk = NVM2C_VK_INT;
            uint8_t ik = NVM2C_VK_INT;
            uint8_t ak = NVM2C_VK_INT;
            int val = stack_pop_kind(b, &st, &vk);
            int ix = stack_pop_kind(b, &st, &ik);
            int arr = stack_pop_kind(b, &st, &ak);
            if (b->failed) goto done;
            if (ik != NVM2C_VK_INT) {
                nvm2c_fail(b, "function %u: ARR_SET index must be int", idx);
                goto done;
            }
            if (ak == NVM2C_VK_ARR && vk == NVM2C_VK_INT) {
                char expr[96];
                snprintf(expr, sizeof expr, "narr_set(a[%d], t[%d], t[%d])", arr, ix, val);
                stack_push_arr(b, &st, expr);
            } else if (ak == NVM2C_VK_SARR && vk == NVM2C_VK_STR) {
                char expr[96];
                snprintf(expr, sizeof expr, "nsarr_set(sa[%d], t[%d], s[%d])", arr, ix, val);
                stack_push_sarr(b, &st, expr);
            } else if (ak == NVM2C_VK_RARR && vk == NVM2C_VK_REC) {
                char expr[96];
                int na;
                snprintf(expr, sizeof expr, "nrarr_set(ra[%d], t[%d], r[%d])", arr, ix, val);
                na = stack_push_rarr(b, &st, expr);
                if (na >= 0) {
                    memcpy(st.rec_k_arr[na], st.rec_k[val], NVM2C_MAX_REC_FIELDS);
                }
            } else {
                nvm2c_fail(b, "function %u: ARR_SET only supports int, string, or record arrays", idx);
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
            if (kind != AGG_RECORD && kind != AGG_TUPLE) {
                nvm2c_fail(b, "function %u: AGG_PACK only supports record or tuple aggregates", idx);
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
                if (vk != NVM2C_VK_INT && vk != NVM2C_VK_STR && vk != NVM2C_VK_REC
                    && vk != NVM2C_VK_HM) {
                    nvm2c_fail(b, "function %u: AGG_PACK fields must be int, string, record, or hashmap", idx);
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
                    } else if (fkind[ei] == NVM2C_VK_REC) {
                        nvm2c_printf(b, "    r[%d].nest[%d] = nrec_store(r[%d]);\n",
                                     r, ei, elems[ei]);
                    } else if (fkind[ei] == NVM2C_VK_HM) {
                        nvm2c_printf(b, "    r[%d].h[%d] = h[%d];\n", r, ei, elems[ei]);
                    } else {
                        nvm2c_printf(b, "    r[%d].f[%d] = t[%d];\n", r, ei, elems[ei]);
                    }
                }
                st.slots[st.sp] = r;
                st.kinds[st.sp] = NVM2C_VK_REC;
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
                } else if (st.rec_k[rec][fi] == NVM2C_VK_REC) {
                    nvm2c_printf(b, "    if (!r[%d].nest[%u]) abort();\n", rec, (unsigned)fi);
                    snprintf(expr, sizeof expr, "*r[%d].nest[%u]", rec, (unsigned)fi);
                    stack_push_rec(b, &st, expr);
                } else if (st.rec_k[rec][fi] == NVM2C_VK_HM) {
                    nvm2c_printf(b, "    if (!r[%d].h[%u]) abort();\n", rec, (unsigned)fi);
                    snprintf(expr, sizeof expr, "r[%d].h[%u]", rec, (unsigned)fi);
                    stack_push_hm(b, &st, expr);
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
            if (!build_direct_call(b, &st, mod, idx, callee, kinds, result_arr_k, call, sizeof call)) {
                goto done;
            }
            const NvmFunctionEntry *cf = &mod->functions[callee];
            if (result_is_i64(cf)) {
                stack_push_temp(b, &st, call);
            } else if (cf->result_count == 1 && cf->result_tag == TAG_STRING) {
                stack_push_str(b, &st, call);
            } else if (result_is_rec(cf)) {
                int nr = stack_push_rec(b, &st, call);
                if (nr >= 0) {
                    memcpy(st.rec_k[nr], fn_result_rec_k(result_rec_k, callee),
                           NVM2C_MAX_REC_FIELDS);
                }
            } else if (result_is_arr(cf)) {
                uint8_t ak = result_arr_k[callee];
                if (ak == NVM2C_VK_SARR) {
                    stack_push_sarr(b, &st, call);
                } else if (ak == NVM2C_VK_RARR) {
                    int na = stack_push_rarr(b, &st, call);
                    if (na >= 0) {
                        memcpy(st.rec_k_arr[na], fn_result_rec_k(result_rec_k, callee),
                               NVM2C_MAX_REC_FIELDS);
                    }
                } else {
                    stack_push_arr(b, &st, call);
                }
            } else if (result_is_hm(cf)) {
                stack_push_hm(b, &st, call);
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
            if (!build_direct_call(b, &st, mod, idx, callee, kinds, result_arr_k, call, sizeof call)) {
                goto done;
            }
            if (st.sp != 0) {
                nvm2c_fail(b, "function %u: TAIL_CALL leaves extra stack values", idx);
                goto done;
            }
            if (fn->result_count == 1 &&
                (result_is_i64(fn) || fn->result_tag == TAG_STRING ||
                 result_is_rec(fn) || result_is_arr(fn) || result_is_hm(fn))) {
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
            } else if (result_is_rec(fn)) {
                int r = stack_pop_expect(b, &st, NVM2C_VK_REC, "RET");
                if (b->failed) goto done;
                if (st.sp != 0) {
                    nvm2c_fail(b, "function %u: RET leaves extra stack values", idx);
                    goto done;
                }
                nvm2c_printf(b, "    return r[%d];\n", r);
            } else if (result_is_arr(fn)) {
                uint8_t ak = result_arr_k[idx];
                int a;
                if (ak == NVM2C_VK_SARR) {
                    a = stack_pop_expect(b, &st, NVM2C_VK_SARR, "RET");
                    if (b->failed) goto done;
                    if (st.sp != 0) {
                        nvm2c_fail(b, "function %u: RET leaves extra stack values", idx);
                        goto done;
                    }
                    nvm2c_printf(b, "    return sa[%d];\n", a);
                } else if (ak == NVM2C_VK_RARR) {
                    a = stack_pop_expect(b, &st, NVM2C_VK_RARR, "RET");
                    if (b->failed) goto done;
                    if (st.sp != 0) {
                        nvm2c_fail(b, "function %u: RET leaves extra stack values", idx);
                        goto done;
                    }
                    nvm2c_printf(b, "    return ra[%d];\n", a);
                } else {
                    a = stack_pop_expect(b, &st, NVM2C_VK_ARR, "RET");
                    if (b->failed) goto done;
                    if (st.sp != 0) {
                        nvm2c_fail(b, "function %u: RET leaves extra stack values", idx);
                        goto done;
                    }
                    nvm2c_printf(b, "    return a[%d];\n", a);
                }
            } else if (result_is_hm(fn)) {
                int h = stack_pop_expect(b, &st, NVM2C_VK_HM, "RET");
                if (b->failed) goto done;
                if (st.sp != 0) {
                    nvm2c_fail(b, "function %u: RET leaves extra stack values", idx);
                    goto done;
                }
                nvm2c_printf(b, "    return h[%d];\n", h);
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
        case OP_HM_NEW: {
            uint8_t ktag = ins.operands[0].u8;
            if (ktag != TAG_STRING) {
                nvm2c_fail(b, "function %u: HM_NEW keys must be string", idx);
                goto done;
            }
            stack_push_hm(b, &st, "nhm_new()");
            break;
        }
        case OP_HM_HAS: {
            int key = stack_pop_expect(b, &st, NVM2C_VK_STR, "HM_HAS key");
            int map = stack_pop_expect(b, &st, NVM2C_VK_HM, "HM_HAS map");
            if (b->failed) goto done;
            {
                char expr[80];
                snprintf(expr, sizeof expr, "nhm_has(h[%d], s[%d])", map, key);
                stack_push_temp(b, &st, expr);
            }
            break;
        }
        case OP_HM_SET: {
            uint8_t vk = NVM2C_VK_INT;
            int val = stack_pop_kind(b, &st, &vk);
            int key = stack_pop_expect(b, &st, NVM2C_VK_STR, "HM_SET key");
            int map = stack_pop_expect(b, &st, NVM2C_VK_HM, "HM_SET map");
            if (b->failed) goto done;
            if (vk == NVM2C_VK_STR) {
                char expr[112];
                snprintf(expr, sizeof expr, "nhm_set(h[%d], s[%d], s[%d], (int64_t)0, 1)",
                         map, key, val);
                stack_push_hm(b, &st, expr);
            } else if (vk == NVM2C_VK_INT) {
                char expr[96];
                snprintf(expr, sizeof expr, "nhm_set(h[%d], s[%d], \"\", t[%d], 0)",
                         map, key, val);
                stack_push_hm(b, &st, expr);
            } else {
                nvm2c_fail(b, "function %u: HM_SET values must be int or string", idx);
                goto done;
            }
            break;
        }
        case OP_HM_GET:
        case OP_HM_DELETE:
        case OP_HM_KEYS:
        case OP_HM_VALUES:
        case OP_HM_LEN: {
            const InstructionInfo *hm_info = isa_get_info(ins.opcode);
            nvm2c_fail(b, "function %u: %s is not in the nvm2c Cut A subset",
                       idx, hm_info ? hm_info->name : "HM_*");
            goto done;
        }
        case OP_CALL_EXTERN: {
            uint32_t imp_idx = ins.operands[0].u32;
            Nvm2cHostKind hk = nvm2c_import_host(mod, imp_idx);
            if (hk == NVM2C_HOST_UNKNOWN) {
                const char *nm = (imp_idx < mod->import_count)
                    ? nvm_get_string(mod, mod->imports[imp_idx].function_name_idx)
                    : NULL;
                nvm2c_fail(b, "no host ABI for import %s; nvm2c refuses CALL_EXTERN",
                           nm ? nm : "?");
                goto done;
            }
            switch (hk) {
            case NVM2C_HOST_GETCWD:
                stack_push_str(b, &st, "nhost_getcwd()");
                break;
            case NVM2C_HOST_TMP_DIR:
                stack_push_str(b, &st, "nhost_tmp_dir()");
                break;
            case NVM2C_HOST_GET_ARGC:
                stack_push_temp(b, &st, "nhost_argc()");
                break;
            case NVM2C_HOST_GETENV: {
                int name = stack_pop_expect(b, &st, NVM2C_VK_STR, "vm_getenv");
                char expr[80];
                if (b->failed) goto done;
                snprintf(expr, sizeof expr, "nhost_getenv(s[%d])", name);
                stack_push_str(b, &st, expr);
                break;
            }
            case NVM2C_HOST_SYSTEM: {
                int cmd = stack_pop_expect(b, &st, NVM2C_VK_STR, "vm_system");
                char expr[80];
                if (b->failed) goto done;
                snprintf(expr, sizeof expr, "nhost_system(s[%d])", cmd);
                stack_push_temp(b, &st, expr);
                break;
            }
            case NVM2C_HOST_FROM_CHAR: {
                int code = stack_pop_expect(b, &st, NVM2C_VK_INT, "vm_string_from_char");
                char expr[80];
                if (b->failed) goto done;
                snprintf(expr, sizeof expr, "nhost_from_char(t[%d])", code);
                stack_push_str(b, &st, expr);
                break;
            }
            case NVM2C_HOST_GET_ARGV: {
                int ix = stack_pop_expect(b, &st, NVM2C_VK_INT, "get_argv");
                char expr[80];
                if (b->failed) goto done;
                snprintf(expr, sizeof expr, "nhost_argv(t[%d])", ix);
                stack_push_str(b, &st, expr);
                break;
            }
            default:
                nvm2c_fail(b, "CALL_EXTERN host kind is missing an emitter");
                goto done;
            }
            break;
        }
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
    nvm2c_puts(b, "}\n\n");

done:
    free(is_start);
    free(is_target);
    free(joins);
    free(join_set);
}

static int module_has_opcode(const NvmModule *mod, uint8_t op) {
    uint32_t i;
    for (i = 0; i < mod->function_count; i++) {
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

static int module_has_arr_op_tag(const NvmModule *mod, uint8_t op, uint8_t tag) {
    uint32_t i;
    for (i = 0; i < mod->function_count; i++) {
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

static int module_has_global_kind(const uint8_t *global_kind, uint32_t gcount, uint8_t kind) {
    uint32_t i;
    for (i = 0; i < gcount; i++) {
        if (global_kind[i] == kind) return 1;
    }
    return 0;
}

static uint32_t module_global_count(const NvmModule *mod) {
    uint32_t i;
    uint32_t max = 0;
    int seen = 0;
    for (i = 0; i < mod->function_count; i++) {
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
            if (ins.opcode == OP_LOAD_GLOBAL || ins.opcode == OP_STORE_GLOBAL) {
                uint32_t gi = ins.operands[0].u32;
                seen = 1;
                if (gi + 1 > max) max = gi + 1;
            }
            pc += n;
        }
    }
    return seen ? max : 0;
}

static int find_fn_named(const NvmModule *mod, const char *want) {
    uint32_t i;
    for (i = 0; i < mod->function_count; i++) {
        uint32_t ni = mod->functions[i].name_idx;
        if (ni < mod->string_count && mod->strings[ni]
            && strcmp(mod->strings[ni], want) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static void emit_globals(Nvm2cBuf *b, const uint8_t *global_kind, uint32_t gcount) {
    uint32_t i;
    int any = 0;
    for (i = 0; i < gcount; i++) {
        uint8_t k = global_kind[i];
        if (k == NVM2C_VK_UNK) continue;
        nvm2c_printf(b, "static %s g%u = {0};\n", c_local_type(k), i);
        any = 1;
    }
    if (any) nvm2c_puts(b, "\n");
}

static void emit_nstr_arena(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static char nstr_arena[65536];\n"
        "static size_t nstr_used;\n");
}

static void emit_nstr_copy(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nstr_copy(const char *s) {\n"
        "    size_t n = strlen(s ? s : \"\");\n"
        "    if (nstr_used + n + 1 > sizeof nstr_arena) abort();\n"
        "    {\n"
        "        char *p = nstr_arena + nstr_used;\n"
        "        memcpy(p, s ? s : \"\", n + 1);\n"
        "        nstr_used += n + 1;\n"
        "        return p;\n"
        "    }\n"
        "}\n\n");
}

static void emit_nhost_getcwd(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nhost_getcwd(void) {\n"
        "    char buf[1024];\n"
        "    if (!getcwd(buf, sizeof buf)) return \"\";\n"
        "    return nstr_copy(buf);\n"
        "}\n\n");
}

static void emit_nhost_getenv(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nhost_getenv(const char *name) {\n"
        "    const char *v = getenv(name ? name : \"\");\n"
        "    return nstr_copy(v ? v : \"\");\n"
        "}\n\n");
}

static void emit_nhost_tmp_dir(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nhost_tmp_dir(void) {\n"
        "    const char *tmp = getenv(\"TMPDIR\");\n"
        "    if (!tmp || !tmp[0]) tmp = \"/tmp\";\n"
        "    return nstr_copy(tmp);\n"
        "}\n\n");
}

static void emit_nhost_system(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static int64_t nhost_system(const char *cmd) {\n"
        "    if (!cmd) return (int64_t)-1;\n"
        "    return (int64_t)system(cmd);\n"
        "}\n\n");
}

static void emit_nhost_from_char(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nhost_from_char(int64_t code) {\n"
        "    char buf[2];\n"
        "    buf[0] = (char)code;\n"
        "    buf[1] = 0;\n"
        "    return nstr_copy(buf);\n"
        "}\n\n");
}

static void emit_nhost_cli_state(Nvm2cBuf *b, int need_argv) {
    nvm2c_puts(b, "static int nvm2c_argc;\n");
    if (need_argv) nvm2c_puts(b, "static char **nvm2c_argv;\n");
}

static void emit_nhost_argc(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static int64_t nhost_argc(void) {\n"
        "    return (int64_t)nvm2c_argc;\n"
        "}\n\n");
}

static void emit_nhost_argv(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nhost_argv(int64_t index) {\n"
        "    if (index < 0 || index >= nvm2c_argc || !nvm2c_argv) return \"\";\n"
        "    return nvm2c_argv[index] ? nvm2c_argv[index] : \"\";\n"
        "}\n\n");
}

static void emit_nstr_concat(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nstr_concat(const char *a, const char *b) {\n"
        "    size_t na = strlen(a ? a : \"\");\n"
        "    size_t nb = strlen(b ? b : \"\");\n"
        "    if (nstr_used + na + nb + 1 > sizeof nstr_arena) abort();\n"
        "    char *p = nstr_arena + nstr_used;\n"
        "    memcpy(p, a ? a : \"\", na);\n"
        "    memcpy(p + na, b ? b : \"\", nb);\n"
        "    p[na + nb] = 0;\n"
        "    nstr_used += na + nb + 1;\n"
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
        "    if (nstr_used + (size_t)len + 1 > sizeof nstr_arena) abort();\n"
        "    char *p = nstr_arena + nstr_used;\n"
        "    memcpy(p, src + start, (size_t)len);\n"
        "    p[len] = 0;\n"
        "    nstr_used += (size_t)len + 1;\n"
        "    return p;\n"
        "}\n\n");
}

static void emit_nstr_trim(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nstr_trim(const char *s) {\n"
        "    const char *src = s ? s : \"\";\n"
        "    int64_t slen = (int64_t)strlen(src);\n"
        "    int64_t start = 0;\n"
        "    int64_t end = slen;\n"
        "    while (start < slen && (src[start] == ' ' || src[start] == '\\t' ||\n"
        "           src[start] == '\\n' || src[start] == '\\r')) start++;\n"
        "    while (end > start && (src[end - 1] == ' ' || src[end - 1] == '\\t' ||\n"
        "           src[end - 1] == '\\n' || src[end - 1] == '\\r')) end--;\n"
        "    return nstr_substr(src, start, end - start);\n"
        "}\n\n");
}

static void emit_nstr_to_lower(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nstr_to_lower(const char *s) {\n"
        "    const char *src = s ? s : \"\";\n"
        "    size_t n = strlen(src);\n"
        "    size_t i;\n"
        "    char *p;\n"
        "    if (nstr_used + n + 1 > sizeof nstr_arena) abort();\n"
        "    p = nstr_arena + nstr_used;\n"
        "    for (i = 0; i < n; i++) {\n"
        "        unsigned char c = (unsigned char)src[i];\n"
        "        p[i] = (c >= 'A' && c <= 'Z') ? (char)(c + 32) : (char)c;\n"
        "    }\n"
        "    p[n] = 0;\n"
        "    nstr_used += n + 1;\n"
        "    return p;\n"
        "}\n\n");
}

static void emit_nstr_to_upper(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nstr_to_upper(const char *s) {\n"
        "    const char *src = s ? s : \"\";\n"
        "    size_t n = strlen(src);\n"
        "    size_t i;\n"
        "    char *p;\n"
        "    if (nstr_used + n + 1 > sizeof nstr_arena) abort();\n"
        "    p = nstr_arena + nstr_used;\n"
        "    for (i = 0; i < n; i++) {\n"
        "        unsigned char c = (unsigned char)src[i];\n"
        "        p[i] = (c >= 'a' && c <= 'z') ? (char)(c - 32) : (char)c;\n"
        "    }\n"
        "    p[n] = 0;\n"
        "    nstr_used += n + 1;\n"
        "    return p;\n"
        "}\n\n");
}

static void emit_nstr_replace(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nstr_replace(const char *s, const char *old_s, const char *new_s) {\n"
        "    const char *src = s ? s : \"\";\n"
        "    const char *oldv = old_s ? old_s : \"\";\n"
        "    const char *newv = new_s ? new_s : \"\";\n"
        "    size_t slen = strlen(src);\n"
        "    size_t olen = strlen(oldv);\n"
        "    size_t nlen = strlen(newv);\n"
        "    size_t count = 0;\n"
        "    const char *p;\n"
        "    const char *found;\n"
        "    char *out;\n"
        "    char *dst;\n"
        "    if (olen == 0) return src;\n"
        "    p = src;\n"
        "    while ((found = strstr(p, oldv)) != NULL) {\n"
        "        count++;\n"
        "        p = found + olen;\n"
        "    }\n"
        "    if (count == 0) return src;\n"
        "    {\n"
        "        long long out_len = (long long)slen +\n"
        "            (long long)count * ((long long)nlen - (long long)olen);\n"
        "        size_t n = out_len > 0 ? (size_t)out_len : 0;\n"
        "        if (nstr_used + n + 1 > sizeof nstr_arena) abort();\n"
        "        out = nstr_arena + nstr_used;\n"
        "        dst = out;\n"
        "        p = src;\n"
        "        while ((found = strstr(p, oldv)) != NULL) {\n"
        "            size_t seg = (size_t)(found - p);\n"
        "            memcpy(dst, p, seg); dst += seg;\n"
        "            memcpy(dst, newv, nlen); dst += nlen;\n"
        "            p = found + olen;\n"
        "        }\n"
        "        {\n"
        "            size_t rest = strlen(p);\n"
        "            memcpy(dst, p, rest);\n"
        "            dst[rest] = 0;\n"
        "            nstr_used += n + 1;\n"
        "            return out;\n"
        "        }\n"
        "    }\n"
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

static void emit_nstr_starts_with(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static int64_t nstr_starts_with(const char *s, const char *pre) {\n"
        "    const char *a = s ? s : \"\";\n"
        "    const char *b = pre ? pre : \"\";\n"
        "    size_t nb = strlen(b);\n"
        "    return (int64_t)(strncmp(a, b, nb) == 0);\n"
        "}\n\n");
}

static void emit_nstr_ends_with(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static int64_t nstr_ends_with(const char *s, const char *suf) {\n"
        "    const char *a = s ? s : \"\";\n"
        "    const char *b = suf ? suf : \"\";\n"
        "    size_t na = strlen(a);\n"
        "    size_t nb = strlen(b);\n"
        "    if (nb > na) return 0;\n"
        "    return (int64_t)(memcmp(a + (na - nb), b, nb) == 0);\n"
        "}\n\n");
}

static void emit_nstr_from_i64(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nstr_from_i64(int64_t v) {\n"
        "    char tmp[32];\n"
        "    int n = snprintf(tmp, sizeof tmp, \"%lld\", (long long)v);\n"
        "    if (n < 0 || (size_t)n + 1 > sizeof nstr_arena - nstr_used) abort();\n"
        "    char *p = nstr_arena + nstr_used;\n"
        "    memcpy(p, tmp, (size_t)n + 1);\n"
        "    nstr_used += (size_t)n + 1;\n"
        "    return p;\n"
        "}\n\n");
}

static void emit_nstr_to_i64(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static int64_t nstr_to_i64(const char *s) {\n"
        "    return s ? (int64_t)strtoll(s, NULL, 10) : 0;\n"
        "}\n\n");
}

static void emit_narr_arena(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static int64_t narr_arena[65536];\n"
        "static size_t narr_used;\n");
}

static void emit_narr_new(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static narr_t narr_new(void) {\n"
        "    narr_t a = (narr_t)calloc(1, sizeof(narr_s));\n"
        "    if (!a) abort();\n"
        "    return a;\n"
        "}\n\n");
}

static void emit_narr_lit(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static narr_t narr_lit(const int64_t *elems, size_t n) {\n"
        "    if (n > 0 && !elems) abort();\n"
        "    if (narr_used + n > (sizeof narr_arena / sizeof narr_arena[0])) abort();\n"
        "    int64_t *p = narr_arena + narr_used;\n"
        "    if (n) memcpy(p, elems, n * sizeof(int64_t));\n"
        "    narr_used += n;\n"
        "    narr_t a = narr_new();\n"
        "    a->data = p;\n"
        "    a->len = n;\n"
        "    return a;\n"
        "}\n\n");
}

static void emit_narr_get(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static int64_t narr_get(narr_t a, int64_t idx) {\n"
        "    if (!a || !a->data || idx < 0 || (size_t)idx >= a->len) abort();\n"
        "    return a->data[idx];\n"
        "}\n\n");
}

static void emit_narr_push(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static narr_t narr_push(narr_t a, int64_t v) {\n"
        "    if (!a) abort();\n"
        "    size_t n = a->len + 1;\n"
        "    if (a->len && !a->data) abort();\n"
        "    if (narr_used + n > (sizeof narr_arena / sizeof narr_arena[0])) abort();\n"
        "    int64_t *p = narr_arena + narr_used;\n"
        "    if (a->len) memcpy(p, a->data, a->len * sizeof(int64_t));\n"
        "    p[a->len] = v;\n"
        "    narr_used += n;\n"
        "    a->data = p;\n"
        "    a->len = n;\n"
        "    return a;\n"
        "}\n\n");
}

static void emit_narr_set(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static narr_t narr_set(narr_t a, int64_t idx, int64_t v) {\n"
        "    if (!a || !a->data || idx < 0 || (size_t)idx >= a->len) abort();\n"
        "    a->data[idx] = v;\n"
        "    return a;\n"
        "}\n\n");
}

static void emit_nsarr_arena(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nsarr_arena[65536];\n"
        "static size_t nsarr_used;\n");
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
        "    if (nsarr_used + n > (sizeof nsarr_arena / sizeof nsarr_arena[0])) abort();\n"
        "    const char **p = nsarr_arena + nsarr_used;\n"
        "    if (n) memcpy(p, elems, n * sizeof(const char *));\n"
        "    nsarr_used += n;\n"
        "    nsarr_t a = nsarr_new();\n"
        "    a->data = p;\n"
        "    a->len = n;\n"
        "    return a;\n"
        "}\n\n");
}

static void emit_nsarr_get(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static const char *nsarr_get(nsarr_t a, int64_t idx) {\n"
        "    if (!a || !a->data || idx < 0 || (size_t)idx >= a->len) abort();\n"
        "    return a->data[idx] ? a->data[idx] : \"\";\n"
        "}\n\n");
}

static void emit_nsarr_push(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static nsarr_t nsarr_push(nsarr_t a, const char *v) {\n"
        "    if (!a) abort();\n"
        "    size_t n = a->len + 1;\n"
        "    if (a->len && !a->data) abort();\n"
        "    if (nsarr_used + n > (sizeof nsarr_arena / sizeof nsarr_arena[0])) abort();\n"
        "    const char **p = nsarr_arena + nsarr_used;\n"
        "    if (a->len) memcpy(p, a->data, a->len * sizeof(const char *));\n"
        "    p[a->len] = v ? v : \"\";\n"
        "    nsarr_used += n;\n"
        "    a->data = p;\n"
        "    a->len = n;\n"
        "    return a;\n"
        "}\n\n");
}

static void emit_nstr_split(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static nsarr_t nstr_split(const char *s, const char *delim) {\n"
        "    const char *src = s ? s : \"\";\n"
        "    const char *d = delim ? delim : \"\";\n"
        "    nsarr_t a = nsarr_new();\n"
        "    size_t slen = strlen(src);\n"
        "    size_t dlen = strlen(d);\n"
        "    if (dlen == 0) {\n"
        "        size_t i;\n"
        "        for (i = 0; i < slen; i++) {\n"
        "            nsarr_push(a, nstr_substr(src, (int64_t)i, 1));\n"
        "        }\n"
        "        return a;\n"
        "    }\n"
        "    {\n"
        "        const char *start = src;\n"
        "        const char *found;\n"
        "        while ((found = strstr(start, d)) != NULL) {\n"
        "            nsarr_push(a, nstr_substr(src, (int64_t)(start - src),\n"
        "                                     (int64_t)(found - start)));\n"
        "            start = found + dlen;\n"
        "        }\n"
        "        nsarr_push(a, nstr_substr(src, (int64_t)(start - src),\n"
        "                                 (int64_t)strlen(start)));\n"
        "        return a;\n"
        "    }\n"
        "}\n\n");
}

static void emit_nsarr_set(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static nsarr_t nsarr_set(nsarr_t a, int64_t idx, const char *v) {\n"
        "    if (!a || !a->data || idx < 0 || (size_t)idx >= a->len) abort();\n"
        "    a->data[idx] = v ? v : \"\";\n"
        "    return a;\n"
        "}\n\n");
}

static void emit_nrarr_arena(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static nrec_t nrarr_arena[65536];\n"
        "static size_t nrarr_used;\n");
}

static void emit_nrarr_new(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static nrarr_t nrarr_new(void) {\n"
        "    nrarr_t a = (nrarr_t)calloc(1, sizeof(nrarr_s));\n"
        "    if (!a) abort();\n"
        "    return a;\n"
        "}\n\n");
}

static void emit_nrarr_get(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static nrec_t nrarr_get(nrarr_t a, int64_t idx) {\n"
        "    if (!a || !a->data || idx < 0 || (size_t)idx >= a->len) abort();\n"
        "    return a->data[idx];\n"
        "}\n\n");
}

static void emit_nrarr_push(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static nrarr_t nrarr_push(nrarr_t a, nrec_t v) {\n"
        "    if (!a) abort();\n"
        "    size_t n = a->len + 1;\n"
        "    if (a->len && !a->data) abort();\n"
        "    if (nrarr_used + n > (sizeof nrarr_arena / sizeof nrarr_arena[0])) abort();\n"
        "    nrec_t *p = nrarr_arena + nrarr_used;\n"
        "    if (a->len) memcpy(p, a->data, a->len * sizeof(nrec_t));\n"
        "    p[a->len] = v;\n"
        "    nrarr_used += n;\n"
        "    a->data = p;\n"
        "    a->len = n;\n"
        "    return a;\n"
        "}\n\n");
}

static void emit_nrarr_set(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static nrarr_t nrarr_set(nrarr_t a, int64_t idx, nrec_t v) {\n"
        "    if (!a || !a->data || idx < 0 || (size_t)idx >= a->len) abort();\n"
        "    a->data[idx] = v;\n"
        "    return a;\n"
        "}\n\n");
}

static void emit_nhm_new(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static nhm_t nhm_new(void) {\n"
        "    nhm_t m = (nhm_t)calloc(1, sizeof(nhm_s));\n"
        "    if (!m) abort();\n"
        "    return m;\n"
        "}\n\n");
}

static void emit_nhm_has(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static int64_t nhm_has(nhm_t m, const char *key) {\n"
        "    size_t i;\n"
        "    if (!m) abort();\n"
        "    if (!key) key = \"\";\n"
        "    for (i = 0; i < m->n; i++) {\n"
        "        if (strcmp(m->k[i] ? m->k[i] : \"\", key) == 0) return 1;\n"
        "    }\n"
        "    return 0;\n"
        "}\n\n");
}

static void emit_nhm_set(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static nhm_t nhm_set(nhm_t m, const char *key, const char *sv, int64_t iv, uint8_t vk) {\n"
        "    size_t i;\n"
        "    if (!m) abort();\n"
        "    if (!key) key = \"\";\n"
        "    for (i = 0; i < m->n; i++) {\n"
        "        if (strcmp(m->k[i] ? m->k[i] : \"\", key) == 0) {\n"
        "            m->sv[i] = sv ? sv : \"\";\n"
        "            m->iv[i] = iv;\n"
        "            m->vk[i] = vk;\n"
        "            return m;\n"
        "        }\n"
        "    }\n"
        "    if (m->n >= NVM2C_HM_CAP) abort();\n"
        "    m->k[m->n] = key;\n"
        "    m->sv[m->n] = sv ? sv : \"\";\n"
        "    m->iv[m->n] = iv;\n"
        "    m->vk[m->n] = vk;\n"
        "    m->n++;\n"
        "    return m;\n"
        "}\n\n");
}

static void emit_nrec_store(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "static nrec_t nrec_nest_arena[65536];\n"
        "static size_t nrec_nest_used;\n"
        "static nrec_t *nrec_store(nrec_t v) {\n"
        "    if (nrec_nest_used >= (sizeof nrec_nest_arena / sizeof nrec_nest_arena[0])) abort();\n"
        "    nrec_nest_arena[nrec_nest_used] = v;\n"
        "    return &nrec_nest_arena[nrec_nest_used++];\n"
        "}\n\n");
}

static int rec_k_has_nested(const uint8_t *k) {
    uint16_t i;
    for (i = 0; i < NVM2C_MAX_REC_FIELDS; i++) {
        if (k[i] == NVM2C_VK_REC) return 1;
    }
    return 0;
}

static int module_has_nested_rec(const uint8_t *result_rec_k, const uint8_t *rec_fields,
                                 uint32_t fn_count) {
    uint32_t i;
    uint16_t slot;
    for (i = 0; i < fn_count; i++) {
        if (rec_k_has_nested(result_rec_k + (size_t)i * NVM2C_MAX_REC_FIELDS)) return 1;
        for (slot = 0; slot < NVM2C_MAX_LOCALS; slot++) {
            if (rec_k_has_nested(rec_fields
                                 + ((size_t)i * NVM2C_MAX_LOCALS + slot)
                                       * NVM2C_MAX_REC_FIELDS)) {
                return 1;
            }
        }
    }
    return 0;
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
    {
        uint32_t i;
        for (i = 0; i < mod->import_count; i++) {
            const char *nm = nvm_get_string(mod, mod->imports[i].function_name_idx);
            if (nvm2c_import_host(mod, i) == NVM2C_HOST_UNKNOWN) {
                if (err && err_len) {
                    snprintf(err, err_len,
                             "no host ABI for import %s; nvm2c refuses CALL_EXTERN",
                             nm ? nm : "?");
                }
                return NULL;
            }
        }
    }

    Nvm2cBuf b;
    memset(&b, 0, sizeof b);
    b.err = err;
    b.err_len = err_len;

    uint8_t *kinds = calloc((size_t)mod->function_count * NVM2C_MAX_LOCALS, 1);
    uint8_t *rec_fields = calloc((size_t)mod->function_count * NVM2C_MAX_LOCALS
                                 * NVM2C_MAX_REC_FIELDS, 1);
    uint8_t *result_rec_k = calloc((size_t)mod->function_count * NVM2C_MAX_REC_FIELDS, 1);
    uint8_t *result_arr_k = calloc((size_t)mod->function_count, 1);
    uint8_t *global_kind = calloc(NVM2C_MAX_GLOBALS, 1);
    uint8_t *global_rec_k = calloc((size_t)NVM2C_MAX_GLOBALS * NVM2C_MAX_REC_FIELDS, 1);
    uint32_t gcount = 0;
    if (!kinds || !rec_fields || !result_rec_k || !result_arr_k
        || !global_kind || !global_rec_k) {
        free(kinds);
        free(rec_fields);
        free(result_rec_k);
        free(result_arr_k);
        free(global_kind);
        free(global_rec_k);
        if (err && err_len) snprintf(err, err_len, "out of memory");
        return NULL;
    }
    memset(global_kind, NVM2C_VK_UNK, NVM2C_MAX_GLOBALS);

    {
        int pass;
        uint32_t i;
        for (pass = 0; pass < 2; pass++) {
            for (i = 0; i < mod->function_count; i++) {
                if (!classify_function(&b, mod, i,
                                       kinds + (size_t)i * NVM2C_MAX_LOCALS,
                                       rec_fields + ((size_t)i * NVM2C_MAX_LOCALS)
                                           * NVM2C_MAX_REC_FIELDS,
                                       result_rec_k, result_arr_k,
                                       global_kind, global_rec_k)) {
                    goto fail;
                }
            }
        }
    }
    gcount = module_global_count(mod);
    if (gcount > NVM2C_MAX_GLOBALS) {
        nvm2c_fail(&b, "too many globals for the nvm2c subset");
        goto fail;
    }

    int need_cli = 0;
    int need_argv_main = 0;
    {
        int need_concat = module_has_opcode(mod, OP_STR_CONCAT);
        int need_cast = module_has_opcode(mod, OP_CAST_STRING);
        int need_cast_int = module_has_opcode(mod, OP_CAST_INT);
        int need_contains = module_has_opcode(mod, OP_STR_CONTAINS);
        int need_starts = module_has_opcode(mod, OP_STR_STARTS_WITH);
        int need_ends = module_has_opcode(mod, OP_STR_ENDS_WITH);
        int need_substr_op = module_has_opcode(mod, OP_STR_SUBSTR);
        int need_trim = module_has_opcode(mod, OP_STR_TRIM);
        int need_lower = module_has_opcode(mod, OP_STR_TO_LOWER);
        int need_upper = module_has_opcode(mod, OP_STR_TO_UPPER);
        int need_replace = module_has_opcode(mod, OP_STR_REPLACE);
        int need_split = module_has_opcode(mod, OP_STR_SPLIT);
        int need_substr = need_substr_op || need_trim || need_split;
        int need_char_at = module_has_opcode(mod, OP_STR_CHAR_AT);
        int need_string = need_concat || need_cast || need_cast_int || need_contains || need_starts ||
            need_ends || need_substr || need_trim || need_lower || need_upper || need_replace || need_split ||
            need_char_at ||
            module_has_opcode(mod, OP_PUSH_STR) ||
            module_has_opcode(mod, OP_STR_LEN);
        int need_arr_lit = module_has_opcode(mod, OP_ARR_LITERAL);
        int need_arr_get = module_has_opcode(mod, OP_ARR_GET);
        int need_arr_push = module_has_opcode(mod, OP_ARR_PUSH);
        int need_arr_set = module_has_opcode(mod, OP_ARR_SET);
        int need_iarr_new = module_has_arr_op_tag(mod, OP_ARR_NEW, TAG_INT);
        int need_iarr_lit = module_has_arr_op_tag(mod, OP_ARR_LITERAL, TAG_INT);
        int need_sarr_lit = module_has_arr_op_tag(mod, OP_ARR_LITERAL, TAG_STRING);
        int need_iarr = need_iarr_lit ||
            module_has_local_kind(kinds, mod->function_count, NVM2C_VK_ARR) ||
            module_has_global_kind(global_kind, gcount, NVM2C_VK_ARR);
        int need_sarr = need_sarr_lit || need_split ||
            module_has_local_kind(kinds, mod->function_count, NVM2C_VK_SARR) ||
            module_has_global_kind(global_kind, gcount, NVM2C_VK_SARR);
        int need_iarr_get = need_arr_get && need_iarr;
        int need_sarr_get = need_arr_get && need_sarr;
        int need_iarr_push = need_arr_push && need_iarr;
        int need_iarr_set = need_arr_set && need_iarr;
        int need_sarr_push = (need_arr_push && need_sarr) || need_split;
        int need_sarr_set = need_arr_set && need_sarr;
        int need_sarr_new = (need_sarr && module_has_opcode(mod, OP_ARR_NEW)) || need_split;
        int need_rarr = module_has_local_kind(kinds, mod->function_count, NVM2C_VK_RARR) ||
            module_has_global_kind(global_kind, gcount, NVM2C_VK_RARR);
        int need_rarr_get = need_arr_get && need_rarr;
        int need_rarr_push = need_arr_push && need_rarr;
        int need_rarr_set = need_arr_set && need_rarr;
        int need_rarr_new = need_rarr && module_has_opcode(mod, OP_ARR_NEW);
        int need_hm_new = module_has_opcode(mod, OP_HM_NEW);
        int need_hm_set = module_has_opcode(mod, OP_HM_SET);
        int need_hm_has = module_has_opcode(mod, OP_HM_HAS);
        int need_agg_get = module_has_opcode(mod, OP_AGG_GET);
        int need_nested = module_has_nested_rec(result_rec_k, rec_fields,
                                               mod->function_count);
        int need_print = module_has_opcode(mod, OP_PRINT) ||
            module_has_opcode(mod, OP_PRINTLN);
        int need_assert = module_has_opcode(mod, OP_ASSERT);
        int need_host_getcwd = 0;
        int need_host_getenv = 0;
        int need_host_tmp = 0;
        int need_host_system = 0;
        int need_host_from_char = 0;
        int need_host_argc = 0;
        int need_host_argv = 0;
        int need_host_copy;
        {
            uint32_t hi;
            for (hi = 0; hi < mod->import_count; hi++) {
                switch (nvm2c_import_host(mod, hi)) {
                case NVM2C_HOST_GETCWD: need_host_getcwd = 1; break;
                case NVM2C_HOST_GETENV: need_host_getenv = 1; break;
                case NVM2C_HOST_TMP_DIR: need_host_tmp = 1; break;
                case NVM2C_HOST_SYSTEM: need_host_system = 1; break;
                case NVM2C_HOST_FROM_CHAR: need_host_from_char = 1; break;
                case NVM2C_HOST_GET_ARGC: need_host_argc = 1; break;
                case NVM2C_HOST_GET_ARGV: need_host_argv = 1; break;
                default: break;
                }
            }
        }
        need_host_copy = need_host_getcwd || need_host_getenv || need_host_tmp
            || need_host_from_char;
        need_cli = need_host_argc || need_host_argv;
        need_argv_main = need_host_argv;
        uint32_t i;
        for (i = 0; i < mod->function_count && !need_string; i++) {
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
            "#include <stdint.h>\n");
        if (need_print || need_cast) {
            nvm2c_puts(&b, "#include <stdio.h>\n");
        }
        if (need_concat || need_cast || need_cast_int || need_substr || need_trim || need_lower ||
            need_upper || need_replace ||
            need_split || need_arr_lit || need_arr_get ||
            need_arr_push || need_arr_set || need_iarr_new || need_sarr_new || need_rarr_new ||
            need_agg_get || need_nested || need_assert || need_hm_new || need_hm_set || need_hm_has ||
            need_host_copy || need_host_system) {
            nvm2c_puts(&b, "#include <stdlib.h>\n#include <string.h>\n");
        } else if (need_string) {
            nvm2c_puts(&b, "#include <string.h>\n");
        }
        if (need_host_getcwd) {
            nvm2c_puts(&b, "#include <unistd.h>\n");
        }
        nvm2c_puts(&b,
            "\n"
            "typedef struct { int64_t *data; size_t len; } narr_s;\n"
            "typedef narr_s *narr_t;\n"
            "typedef struct { const char **data; size_t len; } nsarr_s;\n"
            "typedef nsarr_s *nsarr_t;\n");
        nvm2c_printf(&b,
            "#define NVM2C_HM_CAP %d\n"
            "typedef struct nhm_s {\n"
            "    const char *k[NVM2C_HM_CAP];\n"
            "    const char *sv[NVM2C_HM_CAP];\n"
            "    int64_t iv[NVM2C_HM_CAP];\n"
            "    uint8_t vk[NVM2C_HM_CAP];\n"
            "    uint16_t n;\n"
            "} nhm_s;\n"
            "typedef nhm_s *nhm_t;\n"
            "typedef struct nrec_s {\n"
            "    int64_t f[%d];\n"
            "    const char *s[%d];\n"
            "    struct nrec_s *nest[%d];\n"
            "    nhm_t h[%d];\n"
            "    uint16_t n;\n"
            "} nrec_t;\n"
            "typedef struct { nrec_t *data; size_t len; } nrarr_s;\n"
            "typedef nrarr_s *nrarr_t;\n\n",
            NVM2C_HM_CAP, NVM2C_MAX_REC_FIELDS, NVM2C_MAX_REC_FIELDS,
            NVM2C_MAX_REC_FIELDS, NVM2C_MAX_REC_FIELDS);
        if (need_nested) emit_nrec_store(&b);
        if (need_concat || need_cast || need_substr || need_replace || need_lower || need_upper ||
            need_host_copy) emit_nstr_arena(&b);
        if (need_host_copy) emit_nstr_copy(&b);
        if (need_concat) emit_nstr_concat(&b);
        if (need_substr) emit_nstr_substr(&b);
        if (need_trim) emit_nstr_trim(&b);
        if (need_lower) emit_nstr_to_lower(&b);
        if (need_upper) emit_nstr_to_upper(&b);
        if (need_replace) emit_nstr_replace(&b);
        if (need_char_at) emit_nstr_char_at(&b);
        if (need_starts) emit_nstr_starts_with(&b);
        if (need_ends) emit_nstr_ends_with(&b);
        if (need_cast) emit_nstr_from_i64(&b);
        if (need_cast_int) emit_nstr_to_i64(&b);
        if (need_iarr_lit || (need_iarr && need_iarr_new)) {
            emit_narr_new(&b);
        }
        if (need_iarr_lit || need_iarr_push) {
            emit_narr_arena(&b);
        }
        if (need_iarr_lit) emit_narr_lit(&b);
        if (need_iarr_get) emit_narr_get(&b);
        if (need_iarr_push) emit_narr_push(&b);
        if (need_iarr_set) emit_narr_set(&b);
        if (need_sarr_lit || need_sarr_new) {
            emit_nsarr_new(&b);
        }
        if (need_sarr_lit || need_sarr_push) {
            emit_nsarr_arena(&b);
        }
        if (need_sarr_lit) emit_nsarr_lit(&b);
        if (need_sarr_get) emit_nsarr_get(&b);
        if (need_sarr_push) emit_nsarr_push(&b);
        if (need_sarr_set) emit_nsarr_set(&b);
        if (need_split) emit_nstr_split(&b);
        if (need_rarr_new) emit_nrarr_new(&b);
        if (need_rarr_push) emit_nrarr_arena(&b);
        if (need_rarr_get) emit_nrarr_get(&b);
        if (need_rarr_push) emit_nrarr_push(&b);
        if (need_rarr_set) emit_nrarr_set(&b);
        if (need_hm_new) emit_nhm_new(&b);
        if (need_hm_has) emit_nhm_has(&b);
        if (need_hm_set) emit_nhm_set(&b);
        if (need_cli) emit_nhost_cli_state(&b, need_host_argv);
        if (need_host_argc) emit_nhost_argc(&b);
        if (need_host_argv) emit_nhost_argv(&b);
        if (need_host_getcwd) emit_nhost_getcwd(&b);
        if (need_host_getenv) emit_nhost_getenv(&b);
        if (need_host_tmp) emit_nhost_tmp_dir(&b);
        if (need_host_system) emit_nhost_system(&b);
        if (need_host_from_char) emit_nhost_from_char(&b);
    }

    emit_globals(&b, global_kind, gcount);

    {
        uint32_t i;
        for (i = 0; i < mod->function_count; i++) {
            emit_prototype(&b, mod, i, kinds, result_arr_k);
            if (b.failed) goto fail;
        }
    }
    nvm2c_puts(&b, "\n");

    {
        uint32_t i;
        for (i = 0; i < mod->function_count; i++) {
            emit_function_body(&b, mod, i, kinds, rec_fields, result_rec_k, result_arr_k,
                               global_kind, global_rec_k);
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
        int init_idx = find_fn_named(mod, "__init__");
        fn_c_name(mod, entry, ename, sizeof ename);
        if (need_cli) {
            nvm2c_puts(&b, "int main(int argc, char **argv) {\n");
            nvm2c_puts(&b, "    nvm2c_argc = argc;\n");
            if (need_argv_main) {
                nvm2c_puts(&b, "    nvm2c_argv = argv;\n");
            } else {
                nvm2c_puts(&b, "    (void)argv;\n");
            }
        } else {
            nvm2c_puts(&b, "int main(void) {\n");
        }
        if (init_idx >= 0 && (uint32_t)init_idx != entry) {
            char iname[64];
            fn_c_name(mod, (uint32_t)init_idx, iname, sizeof iname);
            nvm2c_printf(&b, "    %s();\n", iname);
        }
        nvm2c_printf(&b, "    return (int)%s();\n}\n", ename);
    }

    if (b.failed) goto fail;
    if (strstr(b.data, "nano_vm") != NULL || strstr(b.data, "nvm_blob") != NULL) {
        nvm2c_fail(&b, "internal error: emitted a VM wrapper rather than structured C");
        goto fail;
    }
    free(kinds);
    free(rec_fields);
    free(result_rec_k);
    free(result_arr_k);
    free(global_kind);
    free(global_rec_k);
    return b.data;

fail:
    free(kinds);
    free(rec_fields);
    free(result_rec_k);
    free(result_arr_k);
    free(global_kind);
    free(global_rec_k);
    free(b.data);
    return NULL;
}
