/*
 * Structured C11 from a closed NanoISA subset.
 *
 * Temps are one C array so backward goto is valid C. I64_ADD becomes
 * `t[i] = a + b`. The operand stack exists only while translating.
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

static const char *c_result_type(const NvmFunctionEntry *fn) {
    if (fn->result_count == 0 || fn->result_tag == TAG_VOID) return "void";
    if (fn->result_count == 1 && fn->result_tag == TAG_INT) return "int64_t";
    return NULL;
}

static void emit_prototype(Nvm2cBuf *b, const NvmModule *mod, uint32_t idx) {
    const NvmFunctionEntry *fn = &mod->functions[idx];
    const char *rt = c_result_type(fn);
    if (!rt) {
        nvm2c_fail(b, "function %u: only void or a single int result is supported", idx);
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
        for (uint16_t i = 0; i < fn->arity; i++) {
            if (i) nvm2c_puts(b, ", ");
            nvm2c_printf(b, "int64_t a%u", (unsigned)i);
        }
    }
    nvm2c_puts(b, ");\n");
}

typedef struct {
    int slots[NVM2C_MAX_STACK];
    int sp;
    int next_temp;
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
    st->slots[st->sp++] = t;
    return t;
}

static int stack_pop(Nvm2cBuf *b, Nvm2cStack *st) {
    if (st->sp <= 0) {
        nvm2c_fail(b, "operand stack underflow");
        return -1;
    }
    return st->slots[--st->sp];
}

static void emit_binop(Nvm2cBuf *b, Nvm2cStack *st, const char *op) {
    int rhs = stack_pop(b, st);
    int lhs = stack_pop(b, st);
    if (b->failed) return;
    char expr[80];
    snprintf(expr, sizeof expr, "(t[%d] %s t[%d])", lhs, op, rhs);
    stack_push_temp(b, st, expr);
}

static void emit_unop(Nvm2cBuf *b, Nvm2cStack *st, const char *prefix) {
    int x = stack_pop(b, st);
    if (b->failed) return;
    char expr[64];
    snprintf(expr, sizeof expr, "(%s t[%d])", prefix, x);
    stack_push_temp(b, st, expr);
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
                             uint32_t idx, uint32_t callee, char *call, size_t call_sz) {
    if (callee >= mod->function_count) {
        nvm2c_fail(b, "function %u: CALL target %u is out of range", idx, callee);
        return 0;
    }
    const NvmFunctionEntry *cf = &mod->functions[callee];
    if (c_result_type(cf) == NULL) {
        nvm2c_fail(b, "function %u: CALL target %u has an unsupported result", idx, callee);
        return 0;
    }
    int args[NVM2C_MAX_LOCALS];
    for (int i = (int)cf->arity - 1; i >= 0; i--) {
        args[i] = stack_pop(b, st);
        if (b->failed) return 0;
    }
    char cname[64];
    fn_c_name(mod, callee, cname, sizeof cname);
    size_t pos = 0;
    pos += (size_t)snprintf(call + pos, call_sz - pos, "%s(", cname);
    for (uint16_t i = 0; i < cf->arity; i++) {
        if (i) pos += (size_t)snprintf(call + pos, call_sz - pos, ", ");
        pos += (size_t)snprintf(call + pos, call_sz - pos, "t[%d]", args[i]);
        if (pos >= call_sz) {
            nvm2c_fail(b, "function %u: CALL argument list overflow", idx);
            return 0;
        }
    }
    snprintf(call + pos, call_sz - pos, ")");
    return 1;
}

static void emit_function_body(Nvm2cBuf *b, const NvmModule *mod, uint32_t idx) {
    const NvmFunctionEntry *fn = &mod->functions[idx];
    const char *rt = c_result_type(fn);
    if (!rt || b->failed) return;

    char name[64];
    fn_c_name(mod, idx, name, sizeof name);
    nvm2c_printf(b, "static %s %s(", rt, name);
    if (fn->arity == 0) {
        nvm2c_puts(b, "void");
    } else {
        for (uint16_t i = 0; i < fn->arity; i++) {
            if (i) nvm2c_puts(b, ", ");
            nvm2c_printf(b, "int64_t a%u", (unsigned)i);
        }
    }
    nvm2c_puts(b, ") {\n");

    for (uint16_t i = 0; i < fn->local_count; i++) {
        if (i < fn->arity) {
            nvm2c_printf(b, "    int64_t l%u = a%u;\n", (unsigned)i, (unsigned)i);
        } else {
            nvm2c_printf(b, "    int64_t l%u = 0;\n", (unsigned)i);
        }
        nvm2c_printf(b, "    (void)l%u;\n", (unsigned)i);
    }
    nvm2c_printf(b, "    int64_t t[%d] = {0};\n", NVM2C_MAX_TEMPS);
    nvm2c_puts(b, "    (void)t;\n");

    if (fn->code_offset > mod->code_size ||
        fn->code_length > mod->code_size - fn->code_offset) {
        nvm2c_fail(b, "function %u: code range is outside the module", idx);
        return;
    }

    const uint8_t *code = mod->code + fn->code_offset;
    size_t remaining = fn->code_length;
    uint8_t *is_start = calloc(remaining + 1, 1);
    uint8_t *is_target = calloc(remaining + 1, 1);
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

    size_t pc = 0;
    Nvm2cStack st;
    memset(&st, 0, sizeof st);
    int terminated = 0;

    while (pc < remaining) {
        size_t start = pc;
        if (is_target[start]) {
            nvm2c_printf(b, "L_%zu: ;\n", start);
        }
        DecodedInstruction ins;
        uint32_t n = isa_decode(code + pc, remaining - pc, &ins);
        if (n == 0) {
            nvm2c_fail(b, "function %u: invalid instruction at offset %zu", idx, pc);
            goto done;
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
        case OP_DUP: {
            if (st.sp <= 0) {
                nvm2c_fail(b, "function %u: DUP on empty stack", idx);
                goto done;
            }
            char rhs[32];
            snprintf(rhs, sizeof rhs, "t[%d]", st.slots[st.sp - 1]);
            stack_push_temp(b, &st, rhs);
            break;
        }
        case OP_POP:
            (void)stack_pop(b, &st);
            break;
        case OP_SWAP: {
            int x = stack_pop(b, &st);
            int y = stack_pop(b, &st);
            if (b->failed) goto done;
            st.slots[st.sp++] = x;
            st.slots[st.sp++] = y;
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
            stack_push_temp(b, &st, rhs);
            break;
        }
        case OP_STORE_LOCAL: {
            uint16_t slot = ins.operands[0].u16;
            if (slot >= fn->local_count) {
                nvm2c_fail(b, "function %u: STORE_LOCAL %u out of range", idx, slot);
                goto done;
            }
            int t = stack_pop(b, &st);
            if (b->failed) goto done;
            nvm2c_printf(b, "    l%u = t[%d];\n", (unsigned)slot, t);
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
            int rhs = stack_pop(b, &st);
            int lhs = stack_pop(b, &st);
            if (b->failed) goto done;
            char expr[96];
            snprintf(expr, sizeof expr, "(t[%d] == 0 ? (int64_t)0 : t[%d] / t[%d])",
                     rhs, lhs, rhs);
            stack_push_temp(b, &st, expr);
            break;
        }
        case OP_MOD:
        case OP_I64_REM_S: {
            int rhs = stack_pop(b, &st);
            int lhs = stack_pop(b, &st);
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
        case OP_I64_EQ:
            emit_binop(b, &st, "==");
            break;
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
        case OP_CALL: {
            uint32_t callee = ins.operands[0].u32;
            char call[768];
            if (!build_direct_call(b, &st, mod, idx, callee, call, sizeof call)) {
                goto done;
            }
            const NvmFunctionEntry *cf = &mod->functions[callee];
            if (cf->result_count == 1 && cf->result_tag == TAG_INT) {
                stack_push_temp(b, &st, call);
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
            if (!build_direct_call(b, &st, mod, idx, callee, call, sizeof call)) {
                goto done;
            }
            if (st.sp != 0) {
                nvm2c_fail(b, "function %u: TAIL_CALL leaves extra stack values", idx);
                goto done;
            }
            if (fn->result_count == 1 && fn->result_tag == TAG_INT) {
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
            terminated = 1;
            break;
        }
        case OP_JMP_FALSE: {
            int cond = stack_pop(b, &st);
            if (b->failed) goto done;
            size_t tgt = 0;
            if (!jump_target(b, idx, start, ins.operands[0].i32, remaining, &tgt)) {
                goto done;
            }
            nvm2c_printf(b, "    if (!t[%d]) goto L_%zu;\n", cond, tgt);
            break;
        }
        case OP_RET:
            if (fn->result_count == 1 && fn->result_tag == TAG_INT) {
                int t = stack_pop(b, &st);
                if (b->failed) goto done;
                if (st.sp != 0) {
                    nvm2c_fail(b, "function %u: RET leaves extra stack values", idx);
                    goto done;
                }
                nvm2c_printf(b, "    return t[%d];\n", t);
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
            if (fn->result_count == 1 && fn->result_tag == TAG_INT && st.sp == 1) {
                nvm2c_printf(b, "    return t[%d];\n", stack_pop(b, &st));
            } else if (st.sp == 0 && (fn->result_count == 0 || fn->result_tag == TAG_VOID)) {
                nvm2c_puts(b, "    return;\n");
            } else if (st.sp == 0 && fn->result_count == 1 && fn->result_tag == TAG_INT) {
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
    nvm2c_puts(b, "}\n\n");

done:
    free(is_start);
    free(is_target);
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

    nvm2c_puts(&b,
        "/* Generated by nvm2c from NanoISA. Not a VM wrapper. */\n"
        "#include <stdint.h>\n\n");

    for (uint32_t i = 0; i < mod->function_count; i++) {
        emit_prototype(&b, mod, i);
        if (b.failed) goto fail;
    }
    nvm2c_puts(&b, "\n");

    for (uint32_t i = 0; i < mod->function_count; i++) {
        emit_function_body(&b, mod, i);
        if (b.failed) goto fail;
    }

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
    nvm2c_printf(&b,
        "int main(void) {\n"
        "    return (int)%s();\n"
        "}\n",
        ename);

    if (b.failed) goto fail;
    if (strstr(b.data, "nano_vm") != NULL || strstr(b.data, "nvm_blob") != NULL) {
        nvm2c_fail(&b, "internal error: emitted a VM wrapper rather than structured C");
        goto fail;
    }
    return b.data;

fail:
    free(b.data);
    return NULL;
}
