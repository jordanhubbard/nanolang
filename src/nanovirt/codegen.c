/*
 * codegen.c - NanoISA bytecode generation from nanolang AST
 *
 * Two-pass compilation:
 *   Pass 1: Register all functions (name → index) for forward references
 *   Pass 2: Compile each function body to bytecode
 *
 * Each function is compiled independently with its own code buffer,
 * local variable table, and jump patch list.
 */

#include "nanovirt/codegen.h"
#include "nanolang.h"
#include "nanoisa/isa.h"
#include "nanoisa/nvm_format.h"
#include "generated/compiler_schema.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

/* ── Limits ─────────────────────────────────────────────────────── */

#define MAX_LOCALS      256
#define MAX_FUNCTIONS   512
#define MAX_PATCHES     1024
#define MAX_LOOP_DEPTH  32
#define MAX_BREAKS      64
#define CODE_INITIAL    4096

/* ── Internal structures ────────────────────────────────────────── */

typedef struct {
    char *name;
    uint16_t slot;
} Local;

typedef struct {
    uint32_t patch_offset;  /* byte offset of the i32 operand to patch */
    uint32_t instr_offset;  /* byte offset of the instruction start (for relative calc) */
} Patch;

typedef struct {
    uint32_t top_offset;            /* bytecode offset of loop top */
    Patch breaks[MAX_BREAKS];       /* break jump patches */
    int break_count;
} LoopCtx;

typedef struct {
    char *name;
    uint32_t fn_idx;
} FnEntry;

typedef struct {
    /* Module being built */
    NvmModule *module;
    Environment *env;

    /* Current function's code buffer */
    uint8_t *code;
    uint32_t code_size;
    uint32_t code_cap;

    /* Local variables for current function */
    Local locals[MAX_LOCALS];
    uint16_t local_count;
    uint16_t param_count;

    /* Function table (populated in pass 1) */
    FnEntry functions[MAX_FUNCTIONS];
    uint16_t fn_count;

    /* Loop context stack */
    LoopCtx loops[MAX_LOOP_DEPTH];
    int loop_depth;

    /* Error state */
    bool had_error;
    int error_line;
    char error_msg[256];
} CG;

/* ── Error reporting ────────────────────────────────────────────── */

static void cg_error(CG *cg, int line, const char *fmt, ...) {
    if (cg->had_error) return;
    cg->had_error = true;
    cg->error_line = line;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(cg->error_msg, sizeof(cg->error_msg), fmt, ap);
    va_end(ap);
}

/* ── Code buffer helpers ────────────────────────────────────────── */

static void code_ensure(CG *cg, uint32_t extra) {
    while (cg->code_size + extra > cg->code_cap) {
        cg->code_cap *= 2;
        cg->code = realloc(cg->code, cg->code_cap);
        if (!cg->code) {
            cg_error(cg, 0, "out of memory");
            return;
        }
    }
}

/* Emit a single instruction, return its byte offset in the code buffer */
static uint32_t emit_op(CG *cg, NanoOpcode op, ...) {
    if (cg->had_error) return cg->code_size;

    DecodedInstruction instr = {0};
    instr.opcode = op;

    const InstructionInfo *info = isa_get_info(op);
    if (!info) {
        cg_error(cg, 0, "unknown opcode 0x%02x", op);
        return cg->code_size;
    }

    va_list args;
    va_start(args, op);
    for (int i = 0; i < info->operand_count; i++) {
        switch (info->operands[i]) {
            case OPERAND_U8:  instr.operands[i].u8  = (uint8_t)va_arg(args, int); break;
            case OPERAND_U16: instr.operands[i].u16 = (uint16_t)va_arg(args, int); break;
            case OPERAND_U32: instr.operands[i].u32 = va_arg(args, uint32_t); break;
            case OPERAND_I32: instr.operands[i].i32 = va_arg(args, int32_t); break;
            case OPERAND_I64: instr.operands[i].i64 = va_arg(args, int64_t); break;
            case OPERAND_F64: instr.operands[i].f64 = va_arg(args, double); break;
            default: break;
        }
    }
    va_end(args);

    code_ensure(cg, 32);
    uint32_t off = cg->code_size;
    uint32_t n = isa_encode(&instr, cg->code + off, cg->code_cap - off);
    if (n == 0) {
        cg_error(cg, 0, "failed to encode opcode %s", info->name);
        return off;
    }
    cg->code_size += n;
    return off;
}

/* Patch an i32 operand at a specific code offset */
static void patch_jump(CG *cg, uint32_t patch_off, uint32_t instr_off, uint32_t target_off) {
    int32_t rel = (int32_t)(target_off - instr_off);
    /* i32 is stored little-endian after the opcode byte */
    cg->code[patch_off]     = (uint8_t)(rel & 0xFF);
    cg->code[patch_off + 1] = (uint8_t)((rel >> 8) & 0xFF);
    cg->code[patch_off + 2] = (uint8_t)((rel >> 16) & 0xFF);
    cg->code[patch_off + 3] = (uint8_t)((rel >> 24) & 0xFF);
}

/* ── Local variable management ──────────────────────────────────── */

static int16_t local_find(CG *cg, const char *name) {
    for (int i = cg->local_count - 1; i >= 0; i--) {
        if (strcmp(cg->locals[i].name, name) == 0)
            return (int16_t)cg->locals[i].slot;
    }
    return -1;
}

static uint16_t local_add(CG *cg, const char *name, int line) {
    if (cg->local_count >= MAX_LOCALS) {
        cg_error(cg, line, "too many local variables");
        return 0;
    }
    uint16_t slot = cg->local_count;
    cg->locals[slot].name = (char *)name;
    cg->locals[slot].slot = slot;
    cg->local_count++;
    return slot;
}

/* ── Function lookup ────────────────────────────────────────────── */

static int32_t fn_find(CG *cg, const char *name) {
    for (int i = 0; i < cg->fn_count; i++) {
        if (strcmp(cg->functions[i].name, name) == 0)
            return (int32_t)cg->functions[i].fn_idx;
    }
    return -1;
}

/* ── Expression compilation ─────────────────────────────────────── */

static void compile_expr(CG *cg, ASTNode *node);
static void compile_stmt(CG *cg, ASTNode *node);

/* Handle built-in function calls. Returns true if handled, false if not a builtin. */
static bool compile_builtin_call(CG *cg, ASTNode *node) {
    const char *name = node->as.call.name;
    int argc = node->as.call.arg_count;
    ASTNode **args = node->as.call.args;

    /* println / print (as function calls, not AST_PRINT) */
    if (strcmp(name, "println") == 0 || strcmp(name, "print") == 0) {
        if (argc >= 1) compile_expr(cg, args[0]);
        else emit_op(cg, OP_PUSH_VOID);
        emit_op(cg, OP_PRINT);
        /* println/print return void - push void so caller has a value */
        emit_op(cg, OP_PUSH_VOID);
        return true;
    }

    /* String operations */
    if (strcmp(name, "str_length") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_STR_LEN);
        return true;
    }
    if (strcmp(name, "str_concat") == 0 && argc == 2) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        emit_op(cg, OP_STR_CONCAT);
        return true;
    }
    if (strcmp(name, "str_contains") == 0 && argc == 2) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        emit_op(cg, OP_STR_CONTAINS);
        return true;
    }
    if (strcmp(name, "str_equals") == 0 && argc == 2) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        emit_op(cg, OP_STR_EQ);
        return true;
    }
    if (strcmp(name, "str_substring") == 0 && argc == 3) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        compile_expr(cg, args[2]);
        emit_op(cg, OP_STR_SUBSTR);
        return true;
    }

    /* Type casts */
    if (strcmp(name, "cast_int") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_CAST_INT);
        return true;
    }
    if (strcmp(name, "cast_float") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_CAST_FLOAT);
        return true;
    }
    if (strcmp(name, "cast_bool") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_CAST_BOOL);
        return true;
    }
    if ((strcmp(name, "cast_string") == 0 || strcmp(name, "to_string") == 0 ||
         strcmp(name, "int_to_string") == 0 || strcmp(name, "float_to_string") == 0 ||
         strcmp(name, "bool_to_string") == 0) && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_CAST_STRING);
        return true;
    }

    /* Math */
    if (strcmp(name, "abs") == 0 && argc == 1) {
        /* abs(x) = if x < 0 then -x else x */
        compile_expr(cg, args[0]);
        emit_op(cg, OP_DUP);
        emit_op(cg, OP_PUSH_I64, (int64_t)0);
        emit_op(cg, OP_LT);
        uint32_t jf_instr = cg->code_size;
        uint32_t jf_off = emit_op(cg, OP_JMP_FALSE, (int32_t)0);
        emit_op(cg, OP_NEG);
        patch_jump(cg, jf_off + 1, jf_instr, cg->code_size);
        return true;
    }
    if (strcmp(name, "min") == 0 && argc == 2) {
        /* min(a,b) = if a < b then a else b */
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        /* Stack: a b */
        emit_op(cg, OP_DUP);     /* a b b */
        emit_op(cg, OP_ROT3);    /* b b a */
        emit_op(cg, OP_DUP);     /* b b a a */
        emit_op(cg, OP_ROT3);    /* b a a b */
        emit_op(cg, OP_LT);      /* b a (a<b) */
        uint32_t jf_instr = cg->code_size;
        uint32_t jf_off = emit_op(cg, OP_JMP_FALSE, (int32_t)0);
        /* a < b: keep a, drop b */
        emit_op(cg, OP_SWAP);
        emit_op(cg, OP_POP);
        uint32_t je_instr = cg->code_size;
        uint32_t je_off = emit_op(cg, OP_JMP, (int32_t)0);
        /* a >= b: keep b, drop a */
        patch_jump(cg, jf_off + 1, jf_instr, cg->code_size);
        emit_op(cg, OP_POP);
        patch_jump(cg, je_off + 1, je_instr, cg->code_size);
        return true;
    }
    if (strcmp(name, "max") == 0 && argc == 2) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        emit_op(cg, OP_DUP);
        emit_op(cg, OP_ROT3);
        emit_op(cg, OP_DUP);
        emit_op(cg, OP_ROT3);
        emit_op(cg, OP_GT);
        uint32_t jf_instr = cg->code_size;
        uint32_t jf_off = emit_op(cg, OP_JMP_FALSE, (int32_t)0);
        emit_op(cg, OP_SWAP);
        emit_op(cg, OP_POP);
        uint32_t je_instr = cg->code_size;
        uint32_t je_off = emit_op(cg, OP_JMP, (int32_t)0);
        patch_jump(cg, jf_off + 1, jf_instr, cg->code_size);
        emit_op(cg, OP_POP);
        patch_jump(cg, je_off + 1, je_instr, cg->code_size);
        return true;
    }

    /* Array operations */
    if (strcmp(name, "array_length") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_ARR_LEN);
        return true;
    }
    if (strcmp(name, "at") == 0 && argc == 2) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        emit_op(cg, OP_ARR_GET);
        return true;
    }

    /* Array operations */
    if (strcmp(name, "array_new") == 0 && argc >= 1) {
        /* array_new creates an empty array. For now just create int array. */
        compile_expr(cg, args[0]); /* capacity - ignored, use ARR_NEW */
        emit_op(cg, OP_POP);
        emit_op(cg, OP_ARR_NEW, (int)TAG_INT);
        return true;
    }
    if (strcmp(name, "array_push") == 0 && argc == 2) {
        compile_expr(cg, args[0]); /* array */
        compile_expr(cg, args[1]); /* value */
        emit_op(cg, OP_ARR_PUSH);
        return true;
    }
    if (strcmp(name, "array_pop") == 0 && argc == 1) {
        compile_expr(cg, args[0]); /* array */
        emit_op(cg, OP_ARR_POP);
        /* ARR_POP pushes value then array. We want just the value. */
        emit_op(cg, OP_SWAP);
        emit_op(cg, OP_POP);
        return true;
    }
    if (strcmp(name, "array_set") == 0 && argc == 3) {
        compile_expr(cg, args[0]); /* array */
        compile_expr(cg, args[1]); /* index */
        compile_expr(cg, args[2]); /* value */
        emit_op(cg, OP_ARR_SET);
        return true;
    }
    if (strcmp(name, "array_remove_at") == 0 && argc == 2) {
        compile_expr(cg, args[0]); /* array */
        compile_expr(cg, args[1]); /* index */
        emit_op(cg, OP_ARR_REMOVE);
        return true;
    }

    /* String char_at */
    if (strcmp(name, "str_char_at") == 0 && argc == 2) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        emit_op(cg, OP_STR_CHAR_AT);
        return true;
    }

    /* range(n) or range(start, end) - create array of integers */
    if (strcmp(name, "range") == 0 && (argc == 1 || argc == 2)) {
        /*
         * range(n):       [0, 1, ..., n-1]
         * range(start,n): [start, start+1, ..., n-1]
         */
        if (argc == 2) {
            compile_expr(cg, args[0]);  /* start */
            compile_expr(cg, args[1]);  /* end */
        } else {
            emit_op(cg, OP_PUSH_I64, (int64_t)0);  /* start = 0 */
            compile_expr(cg, args[0]);               /* end */
        }
        uint16_t end_slot = local_add(cg, "__range_end__", 0);
        emit_op(cg, OP_STORE_LOCAL, (int)end_slot);
        uint16_t i_slot = local_add(cg, "__range_i__", 0);
        emit_op(cg, OP_STORE_LOCAL, (int)i_slot);

        emit_op(cg, OP_ARR_NEW, (int)TAG_INT);
        uint16_t arr_slot = local_add(cg, "__range_arr__", 0);
        emit_op(cg, OP_STORE_LOCAL, (int)arr_slot);

        /* Loop top: i < end ? */
        uint32_t loop_top = cg->code_size;
        emit_op(cg, OP_LOAD_LOCAL, (int)i_slot);
        emit_op(cg, OP_LOAD_LOCAL, (int)end_slot);
        emit_op(cg, OP_LT);
        uint32_t jf_instr = cg->code_size;
        uint32_t jf_off = emit_op(cg, OP_JMP_FALSE, (int32_t)0);

        /* Body: arr = push(arr, i) */
        emit_op(cg, OP_LOAD_LOCAL, (int)arr_slot);
        emit_op(cg, OP_LOAD_LOCAL, (int)i_slot);
        emit_op(cg, OP_ARR_PUSH);
        emit_op(cg, OP_STORE_LOCAL, (int)arr_slot);

        /* i = i + 1 */
        emit_op(cg, OP_LOAD_LOCAL, (int)i_slot);
        emit_op(cg, OP_PUSH_I64, (int64_t)1);
        emit_op(cg, OP_ADD);
        emit_op(cg, OP_STORE_LOCAL, (int)i_slot);

        /* Jump back to top */
        uint32_t jmp_instr = cg->code_size;
        emit_op(cg, OP_JMP, (int32_t)0);
        patch_jump(cg, jmp_instr + 1, jmp_instr, loop_top);

        /* After loop: push arr as result */
        patch_jump(cg, jf_off + 1, jf_instr, cg->code_size);
        emit_op(cg, OP_LOAD_LOCAL, (int)arr_slot);

        return true;
    }

    return false;
}

static void compile_expr(CG *cg, ASTNode *node) {
    if (!node || cg->had_error) return;

    switch (node->type) {
    case AST_NUMBER:
        emit_op(cg, OP_PUSH_I64, (int64_t)node->as.number);
        break;

    case AST_FLOAT:
        emit_op(cg, OP_PUSH_F64, node->as.float_val);
        break;

    case AST_BOOL:
        emit_op(cg, OP_PUSH_BOOL, node->as.bool_val ? 1 : 0);
        break;

    case AST_STRING: {
        uint32_t idx = nvm_add_string(cg->module, node->as.string_val,
                                       (uint32_t)strlen(node->as.string_val));
        emit_op(cg, OP_PUSH_STR, idx);
        break;
    }

    case AST_IDENTIFIER: {
        int16_t slot = local_find(cg, node->as.identifier);
        if (slot >= 0) {
            emit_op(cg, OP_LOAD_LOCAL, (int)slot);
        } else {
            cg_error(cg, node->line, "undefined variable '%s'", node->as.identifier);
        }
        break;
    }

    case AST_ARRAY_LITERAL: {
        int count = node->as.array_literal.element_count;
        /* Push all elements left-to-right */
        for (int i = 0; i < count; i++) {
            compile_expr(cg, node->as.array_literal.elements[i]);
        }
        /* Determine element type tag */
        uint8_t elem_tag = TAG_INT; /* default */
        switch (node->as.array_literal.element_type) {
            case TYPE_FLOAT:  elem_tag = TAG_FLOAT;  break;
            case TYPE_BOOL:   elem_tag = TAG_BOOL;   break;
            case TYPE_STRING: elem_tag = TAG_STRING;  break;
            default:          elem_tag = TAG_INT;     break;
        }
        emit_op(cg, OP_ARR_LITERAL, (int)elem_tag, count);
        break;
    }

    case AST_PREFIX_OP: {
        TokenType op = node->as.prefix_op.op;
        int argc = node->as.prefix_op.arg_count;
        ASTNode **args = node->as.prefix_op.args;

        if (argc == 1) {
            /* Unary operators */
            compile_expr(cg, args[0]);
            switch (op) {
                case TOKEN_MINUS: emit_op(cg, OP_NEG); break;
                case TOKEN_NOT:   emit_op(cg, OP_NOT); break;
                default:
                    cg_error(cg, node->line, "unsupported unary operator %d", op);
            }
        } else if (argc == 2) {
            /* Binary operators */
            compile_expr(cg, args[0]);
            compile_expr(cg, args[1]);
            switch (op) {
                case TOKEN_PLUS:    emit_op(cg, OP_ADD); break;
                case TOKEN_MINUS:   emit_op(cg, OP_SUB); break;
                case TOKEN_STAR:    emit_op(cg, OP_MUL); break;
                case TOKEN_SLASH:   emit_op(cg, OP_DIV); break;
                case TOKEN_PERCENT: emit_op(cg, OP_MOD); break;
                case TOKEN_EQ:      emit_op(cg, OP_EQ);  break;
                case TOKEN_NE:      emit_op(cg, OP_NE);  break;
                case TOKEN_LT:      emit_op(cg, OP_LT);  break;
                case TOKEN_LE:      emit_op(cg, OP_LE);  break;
                case TOKEN_GT:      emit_op(cg, OP_GT);  break;
                case TOKEN_GE:      emit_op(cg, OP_GE);  break;
                case TOKEN_AND:     emit_op(cg, OP_AND); break;
                case TOKEN_OR:      emit_op(cg, OP_OR);  break;
                default:
                    cg_error(cg, node->line, "unsupported binary operator %d", op);
            }
        } else {
            cg_error(cg, node->line, "unexpected arg count %d for prefix op", argc);
        }
        break;
    }

    case AST_CALL: {
        const char *name = node->as.call.name;
        int argc = node->as.call.arg_count;

        /* Handle built-in functions */
        if (name && compile_builtin_call(cg, node)) {
            break;
        }

        /* Emit arguments left-to-right */
        for (int i = 0; i < argc; i++) {
            compile_expr(cg, node->as.call.args[i]);
        }

        /* Look up function index */
        int32_t fn_idx = fn_find(cg, name);
        if (fn_idx < 0) {
            cg_error(cg, node->line, "undefined function '%s'", name);
            break;
        }
        emit_op(cg, OP_CALL, (uint32_t)fn_idx);
        break;
    }

    case AST_COND: {
        /* cond is an expression: evaluates to a value
         * (cond (c1 v1) (c2 v2) ... (else ve)) */
        int clause_count = node->as.cond_expr.clause_count;
        /* We need end-patches for each clause's JMP to end */
        uint32_t end_patches[64];
        uint32_t end_instrs[64];
        int end_count = 0;

        for (int i = 0; i < clause_count; i++) {
            compile_expr(cg, node->as.cond_expr.conditions[i]);
            uint32_t jf_instr = cg->code_size;
            uint32_t jf_off = emit_op(cg, OP_JMP_FALSE, (int32_t)0);
            uint32_t jf_patch = jf_off + 1; /* skip opcode byte */

            compile_expr(cg, node->as.cond_expr.values[i]);

            /* Jump to end */
            uint32_t je_instr = cg->code_size;
            uint32_t je_off = emit_op(cg, OP_JMP, (int32_t)0);
            if (end_count < 64) {
                end_patches[end_count] = je_off + 1;
                end_instrs[end_count] = je_instr;
                end_count++;
            }

            /* Patch JMP_FALSE to here (next clause) */
            patch_jump(cg, jf_patch, jf_instr, cg->code_size);
        }

        /* Else clause */
        if (node->as.cond_expr.else_value) {
            compile_expr(cg, node->as.cond_expr.else_value);
        } else {
            emit_op(cg, OP_PUSH_VOID);
        }

        /* Patch all end jumps */
        for (int i = 0; i < end_count; i++) {
            patch_jump(cg, end_patches[i], end_instrs[i], cg->code_size);
        }
        break;
    }

    case AST_IF: {
        /* If used as expression (returns value from branches) */
        compile_expr(cg, node->as.if_stmt.condition);
        uint32_t jf_instr = cg->code_size;
        uint32_t jf_off = emit_op(cg, OP_JMP_FALSE, (int32_t)0);
        uint32_t jf_patch = jf_off + 1;

        compile_expr(cg, node->as.if_stmt.then_branch);

        if (node->as.if_stmt.else_branch) {
            uint32_t je_instr = cg->code_size;
            uint32_t je_off = emit_op(cg, OP_JMP, (int32_t)0);
            uint32_t je_patch = je_off + 1;

            patch_jump(cg, jf_patch, jf_instr, cg->code_size);
            compile_expr(cg, node->as.if_stmt.else_branch);
            patch_jump(cg, je_patch, je_instr, cg->code_size);
        } else {
            patch_jump(cg, jf_patch, jf_instr, cg->code_size);
        }
        break;
    }

    default:
        /* Try compiling as a statement (blocks, etc.) */
        compile_stmt(cg, node);
        break;
    }
}

/* ── Statement compilation ──────────────────────────────────────── */

static void compile_stmt(CG *cg, ASTNode *node) {
    if (!node || cg->had_error) return;

    switch (node->type) {
    case AST_LET: {
        compile_expr(cg, node->as.let.value);
        uint16_t slot = local_add(cg, node->as.let.name, node->line);
        emit_op(cg, OP_STORE_LOCAL, (int)slot);
        break;
    }

    case AST_SET: {
        int16_t slot = local_find(cg, node->as.set.name);
        if (slot < 0) {
            cg_error(cg, node->line, "undefined variable '%s'", node->as.set.name);
            break;
        }
        compile_expr(cg, node->as.set.value);
        emit_op(cg, OP_STORE_LOCAL, (int)slot);
        break;
    }

    case AST_IF: {
        compile_expr(cg, node->as.if_stmt.condition);
        uint32_t jf_instr = cg->code_size;
        uint32_t jf_off = emit_op(cg, OP_JMP_FALSE, (int32_t)0);
        uint32_t jf_patch = jf_off + 1;

        compile_stmt(cg, node->as.if_stmt.then_branch);

        if (node->as.if_stmt.else_branch) {
            uint32_t je_instr = cg->code_size;
            uint32_t je_off = emit_op(cg, OP_JMP, (int32_t)0);
            uint32_t je_patch = je_off + 1;

            patch_jump(cg, jf_patch, jf_instr, cg->code_size);
            compile_stmt(cg, node->as.if_stmt.else_branch);
            patch_jump(cg, je_patch, je_instr, cg->code_size);
        } else {
            patch_jump(cg, jf_patch, jf_instr, cg->code_size);
        }
        break;
    }

    case AST_WHILE: {
        if (cg->loop_depth >= MAX_LOOP_DEPTH) {
            cg_error(cg, node->line, "loop nesting too deep");
            break;
        }

        LoopCtx *loop = &cg->loops[cg->loop_depth++];
        loop->break_count = 0;
        loop->top_offset = cg->code_size;

        compile_expr(cg, node->as.while_stmt.condition);
        uint32_t jf_instr = cg->code_size;
        uint32_t jf_off = emit_op(cg, OP_JMP_FALSE, (int32_t)0);
        uint32_t jf_patch = jf_off + 1;

        compile_stmt(cg, node->as.while_stmt.body);

        /* Jump back to top */
        uint32_t jmp_instr = cg->code_size;
        emit_op(cg, OP_JMP, (int32_t)(loop->top_offset - cg->code_size));
        /* Actually: the offset is computed from the instruction start, but
         * we already advanced code_size past the emit. Let me fix this. */
        /* Re-patch: the jump was emitted at jmp_instr, targeting loop->top_offset */
        patch_jump(cg, jmp_instr + 1, jmp_instr, loop->top_offset);

        uint32_t loop_end = cg->code_size;
        /* Patch the conditional jump to after loop */
        patch_jump(cg, jf_patch, jf_instr, loop_end);

        /* Patch all break statements */
        for (int i = 0; i < loop->break_count; i++) {
            patch_jump(cg, loop->breaks[i].patch_offset,
                       loop->breaks[i].instr_offset, loop_end);
        }

        cg->loop_depth--;
        break;
    }

    case AST_FOR: {
        /* for var in range_expr { body }
         * range_expr should be an array. We iterate with an index. */
        if (cg->loop_depth >= MAX_LOOP_DEPTH) {
            cg_error(cg, node->line, "loop nesting too deep");
            break;
        }

        /* Compile the range expression (should produce an array) */
        compile_expr(cg, node->as.for_stmt.range_expr);
        /* Store array in a temp local */
        uint16_t arr_slot = local_add(cg, "__for_arr__", node->line);
        emit_op(cg, OP_STORE_LOCAL, (int)arr_slot);

        /* Initialize counter to 0 */
        emit_op(cg, OP_PUSH_I64, (int64_t)0);
        uint16_t idx_slot = local_add(cg, "__for_idx__", node->line);
        emit_op(cg, OP_STORE_LOCAL, (int)idx_slot);

        /* Get array length */
        emit_op(cg, OP_LOAD_LOCAL, (int)arr_slot);
        emit_op(cg, OP_ARR_LEN);
        uint16_t len_slot = local_add(cg, "__for_len__", node->line);
        emit_op(cg, OP_STORE_LOCAL, (int)len_slot);

        LoopCtx *loop = &cg->loops[cg->loop_depth++];
        loop->break_count = 0;
        loop->top_offset = cg->code_size;

        /* Check: idx < len */
        emit_op(cg, OP_LOAD_LOCAL, (int)idx_slot);
        emit_op(cg, OP_LOAD_LOCAL, (int)len_slot);
        emit_op(cg, OP_LT);
        uint32_t jf_instr = cg->code_size;
        uint32_t jf_off = emit_op(cg, OP_JMP_FALSE, (int32_t)0);
        uint32_t jf_patch = jf_off + 1;

        /* Load current element: arr[idx] */
        emit_op(cg, OP_LOAD_LOCAL, (int)arr_slot);
        emit_op(cg, OP_LOAD_LOCAL, (int)idx_slot);
        emit_op(cg, OP_ARR_GET);

        /* Store in loop variable */
        uint16_t var_slot = local_add(cg, node->as.for_stmt.var_name, node->line);
        emit_op(cg, OP_STORE_LOCAL, (int)var_slot);

        /* Compile body */
        compile_stmt(cg, node->as.for_stmt.body);

        /* Increment counter */
        emit_op(cg, OP_LOAD_LOCAL, (int)idx_slot);
        emit_op(cg, OP_PUSH_I64, (int64_t)1);
        emit_op(cg, OP_ADD);
        emit_op(cg, OP_STORE_LOCAL, (int)idx_slot);

        /* Jump back to top */
        uint32_t jmp_instr = cg->code_size;
        emit_op(cg, OP_JMP, (int32_t)0);
        patch_jump(cg, jmp_instr + 1, jmp_instr, loop->top_offset);

        uint32_t loop_end = cg->code_size;
        patch_jump(cg, jf_patch, jf_instr, loop_end);

        /* Patch breaks */
        for (int i = 0; i < loop->break_count; i++) {
            patch_jump(cg, loop->breaks[i].patch_offset,
                       loop->breaks[i].instr_offset, loop_end);
        }

        cg->loop_depth--;
        break;
    }

    case AST_BLOCK: {
        for (int i = 0; i < node->as.block.count; i++) {
            compile_stmt(cg, node->as.block.statements[i]);
        }
        break;
    }

    case AST_RETURN: {
        if (node->as.return_stmt.value) {
            compile_expr(cg, node->as.return_stmt.value);
        } else {
            emit_op(cg, OP_PUSH_VOID);
        }
        emit_op(cg, OP_RET);
        break;
    }

    case AST_BREAK: {
        if (cg->loop_depth == 0) {
            cg_error(cg, node->line, "break outside loop");
            break;
        }
        LoopCtx *loop = &cg->loops[cg->loop_depth - 1];
        if (loop->break_count >= MAX_BREAKS) {
            cg_error(cg, node->line, "too many breaks in loop");
            break;
        }
        uint32_t jmp_instr = cg->code_size;
        uint32_t jmp_off = emit_op(cg, OP_JMP, (int32_t)0);
        loop->breaks[loop->break_count].patch_offset = jmp_off + 1;
        loop->breaks[loop->break_count].instr_offset = jmp_instr;
        loop->break_count++;
        break;
    }

    case AST_CONTINUE: {
        if (cg->loop_depth == 0) {
            cg_error(cg, node->line, "continue outside loop");
            break;
        }
        LoopCtx *loop = &cg->loops[cg->loop_depth - 1];
        uint32_t jmp_instr = cg->code_size;
        emit_op(cg, OP_JMP, (int32_t)0);
        patch_jump(cg, jmp_instr + 1, jmp_instr, loop->top_offset);
        break;
    }

    case AST_PRINT: {
        compile_expr(cg, node->as.print.expr);
        emit_op(cg, OP_PRINT);
        break;
    }

    case AST_ASSERT: {
        compile_expr(cg, node->as.assert.condition);
        emit_op(cg, OP_ASSERT);
        break;
    }

    /* Expressions used as statements: compile and discard result */
    case AST_NUMBER:
    case AST_FLOAT:
    case AST_BOOL:
    case AST_STRING:
    case AST_IDENTIFIER:
    case AST_PREFIX_OP:
    case AST_COND:
    case AST_ARRAY_LITERAL:
        compile_expr(cg, node);
        emit_op(cg, OP_POP);
        break;

    case AST_CALL: {
        /* Function call as statement: compile and discard result */
        compile_expr(cg, node);
        /* Check if function returns void - if so, no POP needed
         * For now, always POP since we don't track return types in codegen */
        emit_op(cg, OP_POP);
        break;
    }

    /* Skip these in Phase 3 */
    case AST_SHADOW:
    case AST_IMPORT:
    case AST_MODULE_DECL:
    case AST_OPAQUE_TYPE:
    case AST_STRUCT_DEF:
    case AST_ENUM_DEF:
    case AST_UNION_DEF:
        break;

    case AST_FUNCTION:
        /* Functions are compiled in a separate pass, not inline */
        break;

    case AST_UNSAFE_BLOCK: {
        for (int i = 0; i < node->as.block.count; i++) {
            compile_stmt(cg, node->as.block.statements[i]);
        }
        break;
    }

    default:
        cg_error(cg, node->line, "unsupported AST node type %d in statement position", node->type);
        break;
    }
}

/* ── Function compilation ───────────────────────────────────────── */

static void compile_function(CG *cg, ASTNode *fn_node) {
    if (cg->had_error) return;
    if (fn_node->type != AST_FUNCTION) return;
    if (fn_node->as.function.is_extern) return;  /* skip extern declarations */

    const char *name = fn_node->as.function.name;
    int32_t fn_idx = fn_find(cg, name);
    if (fn_idx < 0) {
        cg_error(cg, fn_node->line, "function '%s' not registered", name);
        return;
    }

    /* Reset per-function state */
    cg->code_size = 0;
    cg->local_count = 0;
    cg->param_count = (uint16_t)fn_node->as.function.param_count;
    cg->loop_depth = 0;

    /* Parameters become the first locals */
    for (int i = 0; i < fn_node->as.function.param_count; i++) {
        local_add(cg, fn_node->as.function.params[i].name, fn_node->line);
    }

    /* Compile function body */
    ASTNode *body = fn_node->as.function.body;
    if (body) {
        if (body->type == AST_BLOCK) {
            for (int i = 0; i < body->as.block.count; i++) {
                compile_stmt(cg, body->as.block.statements[i]);
            }
        } else {
            /* Single expression body - treat as return expr */
            compile_expr(cg, body);
            emit_op(cg, OP_RET);
        }
    }

    /* Ensure function always returns (implicit return void) */
    if (cg->code_size == 0 || cg->code[cg->code_size - 1] != OP_RET) {
        /* Check last instruction - a rough check on the opcode byte.
         * If the last emitted instruction wasn't RET, add implicit return. */
        emit_op(cg, OP_PUSH_VOID);
        emit_op(cg, OP_RET);
    }

    if (cg->had_error) return;

    /* Append code to module */
    uint32_t code_off = nvm_append_code(cg->module, cg->code, cg->code_size);

    /* Update function entry */
    NvmFunctionEntry *entry = &cg->module->functions[fn_idx];
    entry->code_offset = code_off;
    entry->code_length = cg->code_size;
    entry->local_count = cg->local_count;
}

/* ── Main compilation entry point ───────────────────────────────── */

CodegenResult codegen_compile(ASTNode *program, Environment *env) {
    CodegenResult result = {0};

    if (!program || program->type != AST_PROGRAM) {
        result.ok = false;
        snprintf(result.error_msg, sizeof(result.error_msg), "expected AST_PROGRAM root node");
        return result;
    }

    CG cg = {0};
    cg.module = nvm_module_new();
    cg.env = env;
    cg.code = malloc(CODE_INITIAL);
    cg.code_cap = CODE_INITIAL;

    if (!cg.code || !cg.module) {
        result.ok = false;
        snprintf(result.error_msg, sizeof(result.error_msg), "out of memory");
        if (cg.module) nvm_module_free(cg.module);
        free(cg.code);
        return result;
    }

    /* ── Pass 1: Register all functions ─────────────────────────── */
    int main_fn_idx = -1;

    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_FUNCTION && !item->as.function.is_extern) {
            const char *name = item->as.function.name;
            uint32_t name_idx = nvm_add_string(cg.module, name, (uint32_t)strlen(name));

            NvmFunctionEntry fn = {0};
            fn.name_idx = name_idx;
            fn.arity = (uint16_t)item->as.function.param_count;
            /* code_offset and code_length will be filled in pass 2 */

            uint32_t idx = nvm_add_function(cg.module, &fn);

            if (cg.fn_count < MAX_FUNCTIONS) {
                cg.functions[cg.fn_count].name = (char *)name;
                cg.functions[cg.fn_count].fn_idx = idx;
                cg.fn_count++;
            }

            if (strcmp(name, "main") == 0) {
                main_fn_idx = (int)idx;
            }
        }
    }

    /* ── Pass 2: Compile function bodies ────────────────────────── */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_FUNCTION && !item->as.function.is_extern) {
            compile_function(&cg, item);
            if (cg.had_error) break;
        }
    }

    /* Set entry point */
    if (main_fn_idx >= 0) {
        cg.module->header.flags = NVM_FLAG_HAS_MAIN;
        cg.module->header.entry_point = (uint32_t)main_fn_idx;
    }

    free(cg.code);

    if (cg.had_error) {
        result.ok = false;
        result.error_line = cg.error_line;
        memcpy(result.error_msg, cg.error_msg, sizeof(result.error_msg));
        nvm_module_free(cg.module);
        return result;
    }

    result.ok = true;
    result.module = cg.module;
    return result;
}
