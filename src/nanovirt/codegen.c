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
#define MAX_STRUCT_DEFS 128
#define MAX_ENUM_DEFS   64
#define MAX_UNION_DEFS  64
#define MAX_GLOBALS     128
#define MAX_EXTERNS     256
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
    char *name;
    char **field_names;
    int field_count;
    uint32_t def_idx;
} CgStructDef;

typedef struct {
    char *name;
    char **variant_names;
    int *variant_values;
    int variant_count;
    uint32_t def_idx;
} CgEnumDef;

typedef struct {
    char *name;
    int variant_count;
    char **variant_names;
    int *variant_field_counts;
    char ***variant_field_names;
    uint32_t def_idx;
} CgUnionDef;

typedef struct {
    char *name;
    uint16_t slot;
} GlobalVar;

typedef struct {
    char *name;              /* C function name (e.g., "nl_regex_compile" or "path_normalize") */
    char *module_name;       /* Module name (e.g., "regex" or "") */
    uint32_t import_idx;     /* Index into NVM import table */
    uint16_t param_count;
    uint8_t return_tag;      /* NanoValueTag for return type */
} ExternFn;

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

    /* Type definitions (populated in pass 1) */
    CgStructDef structs[MAX_STRUCT_DEFS];
    uint16_t struct_count;
    CgEnumDef enums[MAX_ENUM_DEFS];
    uint16_t enum_count;
    CgUnionDef unions[MAX_UNION_DEFS];
    uint16_t union_count;

    /* Global variables (top-level let bindings) */
    GlobalVar globals[MAX_GLOBALS];
    uint16_t global_count;

    /* Extern functions (populated in pass 1 from extern fn + imports) */
    ExternFn externs[MAX_EXTERNS];
    uint16_t extern_count;

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

/* ── Type definition lookup ─────────────────────────────────────── */

static CgStructDef *struct_find(CG *cg, const char *name) {
    for (int i = 0; i < cg->struct_count; i++) {
        if (strcmp(cg->structs[i].name, name) == 0)
            return &cg->structs[i];
    }
    return NULL;
}

static int16_t struct_field_index(CgStructDef *sd, const char *field) {
    for (int i = 0; i < sd->field_count; i++) {
        if (strcmp(sd->field_names[i], field) == 0)
            return (int16_t)i;
    }
    return -1;
}

static CgEnumDef *enum_find(CG *cg, const char *name) {
    for (int i = 0; i < cg->enum_count; i++) {
        if (strcmp(cg->enums[i].name, name) == 0)
            return &cg->enums[i];
    }
    return NULL;
}

static int16_t enum_variant_index(CgEnumDef *ed, const char *variant) {
    for (int i = 0; i < ed->variant_count; i++) {
        if (strcmp(ed->variant_names[i], variant) == 0)
            return (int16_t)i;
    }
    return -1;
}

static CgUnionDef *union_find(CG *cg, const char *name) {
    for (int i = 0; i < cg->union_count; i++) {
        if (strcmp(cg->unions[i].name, name) == 0)
            return &cg->unions[i];
    }
    return NULL;
}

static int16_t union_variant_index(CgUnionDef *ud, const char *variant) {
    for (int i = 0; i < ud->variant_count; i++) {
        if (strcmp(ud->variant_names[i], variant) == 0)
            return (int16_t)i;
    }
    return -1;
}

static int16_t global_find(CG *cg, const char *name) {
    for (int i = 0; i < cg->global_count; i++) {
        if (strcmp(cg->globals[i].name, name) == 0)
            return (int16_t)cg->globals[i].slot;
    }
    return -1;
}

/* ── Extern function lookup ────────────────────────────────────── */

static int32_t extern_find(CG *cg, const char *name) {
    for (int i = 0; i < cg->extern_count; i++) {
        if (strcmp(cg->externs[i].name, name) == 0)
            return (int32_t)cg->externs[i].import_idx;
    }
    return -1;
}

/* Also search by qualified name "Module.function" */
static int32_t extern_find_qualified(CG *cg, const char *module_alias, const char *func_name) {
    char qualified[512];
    snprintf(qualified, sizeof(qualified), "%s.%s", module_alias, func_name);
    return extern_find(cg, qualified);
}

/* Convert nanolang Type enum to NanoValueTag */
static uint8_t type_to_tag(Type t) {
    switch (t) {
        case TYPE_INT:     return TAG_INT;
        case TYPE_FLOAT:   return TAG_FLOAT;
        case TYPE_BOOL:    return TAG_BOOL;
        case TYPE_STRING:  return TAG_STRING;
        case TYPE_VOID:    return TAG_VOID;
        case TYPE_ARRAY:   return TAG_ARRAY;
        case TYPE_STRUCT:  return TAG_STRUCT;
        case TYPE_ENUM:    return TAG_ENUM;
        case TYPE_TUPLE:   return TAG_TUPLE;
        case TYPE_HASHMAP: return TAG_HASHMAP;
        case TYPE_OPAQUE:  return TAG_OPAQUE;
        default:           return TAG_VOID;
    }
}

/* Register an extern function in the codegen extern table and NVM import table */
static void register_extern(CG *cg, const char *name, const char *module_name,
                           uint16_t param_count, uint8_t return_tag,
                           const uint8_t *param_tags) {
    if (cg->extern_count >= MAX_EXTERNS) return;

    /* Add to NVM import table */
    uint32_t mod_str = nvm_add_string(cg->module, module_name, (uint32_t)strlen(module_name));
    uint32_t fn_str = nvm_add_string(cg->module, name, (uint32_t)strlen(name));
    uint32_t imp_idx = nvm_add_import(cg->module, mod_str, fn_str,
                                       param_count, return_tag, param_tags);

    /* Add to codegen extern table */
    ExternFn *ef = &cg->externs[cg->extern_count];
    ef->name = (char *)name;
    ef->module_name = (char *)module_name;
    ef->import_idx = imp_idx;
    ef->param_count = param_count;
    ef->return_tag = return_tag;
    cg->extern_count++;
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
    if (strcmp(name, "array_new") == 0 && (argc == 1 || argc == 2)) {
        /* array_new(size) or array_new(size, fill_value) */
        if (argc == 2) {
            /* Create array and fill with value */
            compile_expr(cg, args[0]);  /* size */
            uint16_t sz_slot = local_add(cg, "__anew_sz__", 0);
            emit_op(cg, OP_STORE_LOCAL, (int)sz_slot);
            compile_expr(cg, args[1]);  /* fill value */
            uint16_t fill_slot = local_add(cg, "__anew_fill__", 0);
            emit_op(cg, OP_STORE_LOCAL, (int)fill_slot);
            emit_op(cg, OP_ARR_NEW, (int)TAG_INT);
            uint16_t arr_slot = local_add(cg, "__anew_arr__", 0);
            emit_op(cg, OP_STORE_LOCAL, (int)arr_slot);
            emit_op(cg, OP_PUSH_I64, (int64_t)0);
            uint16_t i_slot = local_add(cg, "__anew_i__", 0);
            emit_op(cg, OP_STORE_LOCAL, (int)i_slot);
            uint32_t loop_top = cg->code_size;
            emit_op(cg, OP_LOAD_LOCAL, (int)i_slot);
            emit_op(cg, OP_LOAD_LOCAL, (int)sz_slot);
            emit_op(cg, OP_LT);
            uint32_t jf_instr = cg->code_size;
            uint32_t jf_off = emit_op(cg, OP_JMP_FALSE, (int32_t)0);
            emit_op(cg, OP_LOAD_LOCAL, (int)arr_slot);
            emit_op(cg, OP_LOAD_LOCAL, (int)fill_slot);
            emit_op(cg, OP_ARR_PUSH);
            emit_op(cg, OP_STORE_LOCAL, (int)arr_slot);
            emit_op(cg, OP_LOAD_LOCAL, (int)i_slot);
            emit_op(cg, OP_PUSH_I64, (int64_t)1);
            emit_op(cg, OP_ADD);
            emit_op(cg, OP_STORE_LOCAL, (int)i_slot);
            uint32_t jmp_instr = cg->code_size;
            emit_op(cg, OP_JMP, (int32_t)0);
            patch_jump(cg, jmp_instr + 1, jmp_instr, loop_top);
            patch_jump(cg, jf_off + 1, jf_instr, cg->code_size);
            emit_op(cg, OP_LOAD_LOCAL, (int)arr_slot);
        } else {
            /* array_new(size) - just creates empty array for now */
            compile_expr(cg, args[0]);
            emit_op(cg, OP_POP);
            emit_op(cg, OP_ARR_NEW, (int)TAG_INT);
        }
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
        /* ARR_POP stack effect: [... v, arr] (array on top, value below).
         * We want just the value, so pop the array off the top. */
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

    /* filter(arr, fn) - returns new array with elements where fn returns true */
    if (strcmp(name, "filter") == 0 && argc == 2) {
        compile_expr(cg, args[0]);  /* source array */
        uint16_t src_slot = local_add(cg, "__filter_src__", 0);
        emit_op(cg, OP_STORE_LOCAL, (int)src_slot);

        compile_expr(cg, args[1]);  /* predicate function */
        uint16_t fn_slot = local_add(cg, "__filter_fn__", 0);
        emit_op(cg, OP_STORE_LOCAL, (int)fn_slot);

        /* Get source length */
        emit_op(cg, OP_LOAD_LOCAL, (int)src_slot);
        emit_op(cg, OP_ARR_LEN);
        uint16_t len_slot = local_add(cg, "__filter_len__", 0);
        emit_op(cg, OP_STORE_LOCAL, (int)len_slot);

        /* Create result array */
        emit_op(cg, OP_ARR_NEW, (int)TAG_INT);
        uint16_t res_slot = local_add(cg, "__filter_res__", 0);
        emit_op(cg, OP_STORE_LOCAL, (int)res_slot);

        /* Index counter */
        emit_op(cg, OP_PUSH_I64, (int64_t)0);
        uint16_t idx_slot = local_add(cg, "__filter_i__", 0);
        emit_op(cg, OP_STORE_LOCAL, (int)idx_slot);

        /* Loop: while i < len */
        uint32_t loop_top = cg->code_size;
        emit_op(cg, OP_LOAD_LOCAL, (int)idx_slot);
        emit_op(cg, OP_LOAD_LOCAL, (int)len_slot);
        emit_op(cg, OP_LT);
        uint32_t jf_instr = cg->code_size;
        uint32_t jf_off = emit_op(cg, OP_JMP_FALSE, (int32_t)0);

        /* Get element: src[i] */
        emit_op(cg, OP_LOAD_LOCAL, (int)src_slot);
        emit_op(cg, OP_LOAD_LOCAL, (int)idx_slot);
        emit_op(cg, OP_ARR_GET);
        uint16_t elem_slot = local_add(cg, "__filter_elem__", 0);
        emit_op(cg, OP_STORE_LOCAL, (int)elem_slot);

        /* Call predicate: fn(elem) */
        emit_op(cg, OP_LOAD_LOCAL, (int)elem_slot);
        emit_op(cg, OP_LOAD_LOCAL, (int)fn_slot);
        emit_op(cg, OP_CALL_INDIRECT);

        /* If true, push element to result */
        uint32_t skip_instr = cg->code_size;
        uint32_t skip_off = emit_op(cg, OP_JMP_FALSE, (int32_t)0);
        emit_op(cg, OP_LOAD_LOCAL, (int)res_slot);
        emit_op(cg, OP_LOAD_LOCAL, (int)elem_slot);
        emit_op(cg, OP_ARR_PUSH);
        emit_op(cg, OP_STORE_LOCAL, (int)res_slot);
        patch_jump(cg, skip_off + 1, skip_instr, cg->code_size);

        /* i++ */
        emit_op(cg, OP_LOAD_LOCAL, (int)idx_slot);
        emit_op(cg, OP_PUSH_I64, (int64_t)1);
        emit_op(cg, OP_ADD);
        emit_op(cg, OP_STORE_LOCAL, (int)idx_slot);

        uint32_t jmp_instr = cg->code_size;
        emit_op(cg, OP_JMP, (int32_t)0);
        patch_jump(cg, jmp_instr + 1, jmp_instr, loop_top);
        patch_jump(cg, jf_off + 1, jf_instr, cg->code_size);

        emit_op(cg, OP_LOAD_LOCAL, (int)res_slot);
        return true;
    }

    /* map(arr, fn) - returns new array with fn applied to each element */
    if (strcmp(name, "map") == 0 && argc == 2) {
        compile_expr(cg, args[0]);  /* source array */
        uint16_t src_slot = local_add(cg, "__map_src__", 0);
        emit_op(cg, OP_STORE_LOCAL, (int)src_slot);

        compile_expr(cg, args[1]);  /* transform function */
        uint16_t fn_slot = local_add(cg, "__map_fn__", 0);
        emit_op(cg, OP_STORE_LOCAL, (int)fn_slot);

        emit_op(cg, OP_LOAD_LOCAL, (int)src_slot);
        emit_op(cg, OP_ARR_LEN);
        uint16_t len_slot = local_add(cg, "__map_len__", 0);
        emit_op(cg, OP_STORE_LOCAL, (int)len_slot);

        emit_op(cg, OP_ARR_NEW, (int)TAG_INT);
        uint16_t res_slot = local_add(cg, "__map_res__", 0);
        emit_op(cg, OP_STORE_LOCAL, (int)res_slot);

        emit_op(cg, OP_PUSH_I64, (int64_t)0);
        uint16_t idx_slot = local_add(cg, "__map_i__", 0);
        emit_op(cg, OP_STORE_LOCAL, (int)idx_slot);

        uint32_t loop_top = cg->code_size;
        emit_op(cg, OP_LOAD_LOCAL, (int)idx_slot);
        emit_op(cg, OP_LOAD_LOCAL, (int)len_slot);
        emit_op(cg, OP_LT);
        uint32_t jf_instr = cg->code_size;
        uint32_t jf_off = emit_op(cg, OP_JMP_FALSE, (int32_t)0);

        /* Call fn(src[i]) */
        emit_op(cg, OP_LOAD_LOCAL, (int)src_slot);
        emit_op(cg, OP_LOAD_LOCAL, (int)idx_slot);
        emit_op(cg, OP_ARR_GET);
        emit_op(cg, OP_LOAD_LOCAL, (int)fn_slot);
        emit_op(cg, OP_CALL_INDIRECT);

        /* Push result to output array */
        emit_op(cg, OP_LOAD_LOCAL, (int)res_slot);
        emit_op(cg, OP_SWAP);
        emit_op(cg, OP_ARR_PUSH);
        emit_op(cg, OP_STORE_LOCAL, (int)res_slot);

        emit_op(cg, OP_LOAD_LOCAL, (int)idx_slot);
        emit_op(cg, OP_PUSH_I64, (int64_t)1);
        emit_op(cg, OP_ADD);
        emit_op(cg, OP_STORE_LOCAL, (int)idx_slot);

        uint32_t jmp_instr = cg->code_size;
        emit_op(cg, OP_JMP, (int32_t)0);
        patch_jump(cg, jmp_instr + 1, jmp_instr, loop_top);
        patch_jump(cg, jf_off + 1, jf_instr, cg->code_size);

        emit_op(cg, OP_LOAD_LOCAL, (int)res_slot);
        return true;
    }

    /* reduce(arr, init, fn) - fold array with fn(acc, elem) */
    if (strcmp(name, "reduce") == 0 && argc == 3) {
        compile_expr(cg, args[0]);  /* source array */
        uint16_t src_slot = local_add(cg, "__reduce_src__", 0);
        emit_op(cg, OP_STORE_LOCAL, (int)src_slot);

        compile_expr(cg, args[1]);  /* initial value */
        uint16_t acc_slot = local_add(cg, "__reduce_acc__", 0);
        emit_op(cg, OP_STORE_LOCAL, (int)acc_slot);

        compile_expr(cg, args[2]);  /* reducer function */
        uint16_t fn_slot = local_add(cg, "__reduce_fn__", 0);
        emit_op(cg, OP_STORE_LOCAL, (int)fn_slot);

        emit_op(cg, OP_LOAD_LOCAL, (int)src_slot);
        emit_op(cg, OP_ARR_LEN);
        uint16_t len_slot = local_add(cg, "__reduce_len__", 0);
        emit_op(cg, OP_STORE_LOCAL, (int)len_slot);

        emit_op(cg, OP_PUSH_I64, (int64_t)0);
        uint16_t idx_slot = local_add(cg, "__reduce_i__", 0);
        emit_op(cg, OP_STORE_LOCAL, (int)idx_slot);

        uint32_t loop_top = cg->code_size;
        emit_op(cg, OP_LOAD_LOCAL, (int)idx_slot);
        emit_op(cg, OP_LOAD_LOCAL, (int)len_slot);
        emit_op(cg, OP_LT);
        uint32_t jf_instr = cg->code_size;
        uint32_t jf_off = emit_op(cg, OP_JMP_FALSE, (int32_t)0);

        /* Call fn(acc, src[i]) */
        emit_op(cg, OP_LOAD_LOCAL, (int)acc_slot);
        emit_op(cg, OP_LOAD_LOCAL, (int)src_slot);
        emit_op(cg, OP_LOAD_LOCAL, (int)idx_slot);
        emit_op(cg, OP_ARR_GET);
        emit_op(cg, OP_LOAD_LOCAL, (int)fn_slot);
        emit_op(cg, OP_CALL_INDIRECT);
        emit_op(cg, OP_STORE_LOCAL, (int)acc_slot);

        emit_op(cg, OP_LOAD_LOCAL, (int)idx_slot);
        emit_op(cg, OP_PUSH_I64, (int64_t)1);
        emit_op(cg, OP_ADD);
        emit_op(cg, OP_STORE_LOCAL, (int)idx_slot);

        uint32_t jmp_instr = cg->code_size;
        emit_op(cg, OP_JMP, (int32_t)0);
        patch_jump(cg, jmp_instr + 1, jmp_instr, loop_top);
        patch_jump(cg, jf_off + 1, jf_instr, cg->code_size);

        emit_op(cg, OP_LOAD_LOCAL, (int)acc_slot);
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
        const char *id = node->as.identifier;
        int16_t slot = local_find(cg, id);
        if (slot >= 0) {
            emit_op(cg, OP_LOAD_LOCAL, (int)slot);
        } else {
            int16_t gslot = global_find(cg, id);
            if (gslot >= 0) {
                emit_op(cg, OP_LOAD_GLOBAL, (uint32_t)gslot);
            } else {
                /* Check if it's a function name (function-as-value) */
                int32_t fn_idx = fn_find(cg, id);
                if (fn_idx >= 0) {
                    /* Push function reference as a TAG_FUNCTION value */
                    emit_op(cg, OP_CLOSURE_NEW, (uint32_t)fn_idx, 0);
                } else {
                    cg_error(cg, node->line, "undefined variable '%s'", id);
                }
            }
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
        int32_t fn_idx = name ? fn_find(cg, name) : -1;
        if (fn_idx >= 0) {
            emit_op(cg, OP_CALL, (uint32_t)fn_idx);
        } else {
            /* Check if it's an extern function */
            int32_t ext_idx = name ? extern_find(cg, name) : -1;
            if (ext_idx >= 0) {
                emit_op(cg, OP_CALL_EXTERN, (uint32_t)ext_idx);
            } else {
                /* Check if callee is a variable holding a function reference */
                int16_t slot = name ? local_find(cg, name) : -1;
                if (slot >= 0) {
                    emit_op(cg, OP_LOAD_LOCAL, (int)slot);
                    emit_op(cg, OP_CALL_INDIRECT);
                } else if (node->as.call.func_expr) {
                    /* Computed function expression: ((get_fn) args) */
                    compile_expr(cg, node->as.call.func_expr);
                    emit_op(cg, OP_CALL_INDIRECT);
                } else {
                    cg_error(cg, node->line, "undefined function '%s'",
                             name ? name : "(null)");
                }
            }
        }
        break;
    }

    case AST_MODULE_QUALIFIED_CALL: {
        const char *mod_alias = node->as.module_qualified_call.module_alias;
        const char *func_name = node->as.module_qualified_call.function_name;
        int argc = node->as.module_qualified_call.arg_count;

        /* Emit arguments left-to-right */
        for (int i = 0; i < argc; i++) {
            compile_expr(cg, node->as.module_qualified_call.args[i]);
        }

        /* Look up as qualified extern: "Module.function" */
        int32_t ext_idx = extern_find_qualified(cg, mod_alias, func_name);
        if (ext_idx >= 0) {
            emit_op(cg, OP_CALL_EXTERN, (uint32_t)ext_idx);
        } else {
            /* Try unqualified name (maybe imported with from...import) */
            ext_idx = extern_find(cg, func_name);
            if (ext_idx >= 0) {
                emit_op(cg, OP_CALL_EXTERN, (uint32_t)ext_idx);
            } else {
                cg_error(cg, node->line, "undefined module function '%s.%s'",
                         mod_alias, func_name);
            }
        }
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

    case AST_STRUCT_LITERAL: {
        const char *sname = node->as.struct_literal.struct_name;

        /* Check for Union.Variant pattern (e.g., Result.Ok { value: 42 }) */
        if (sname) {
            const char *dot = strchr(sname, '.');
            if (dot) {
                /* Split into union_name and variant_name */
                size_t ulen = (size_t)(dot - sname);
                char uname[256];
                if (ulen >= sizeof(uname)) ulen = sizeof(uname) - 1;
                memcpy(uname, sname, ulen);
                uname[ulen] = '\0';
                const char *vname = dot + 1;

                CgUnionDef *ud = union_find(cg, uname);
                if (ud) {
                    int16_t vi = union_variant_index(ud, vname);
                    if (vi < 0) {
                        cg_error(cg, node->line, "unknown variant '%s.%s'", uname, vname);
                        break;
                    }
                    int fc = node->as.struct_literal.field_count;
                    for (int i = 0; i < fc; i++) {
                        compile_expr(cg, node->as.struct_literal.field_values[i]);
                    }
                    emit_op(cg, OP_UNION_CONSTRUCT, ud->def_idx, (int)vi, fc);
                    break;
                }
            }
        }

        CgStructDef *sd = struct_find(cg, sname);
        if (!sd) {
            cg_error(cg, node->line, "undefined struct '%s'", sname ? sname : "(null)");
            break;
        }
        /* Push field values in definition order */
        for (int i = 0; i < sd->field_count; i++) {
            /* Find matching field in the literal */
            bool found = false;
            for (int j = 0; j < node->as.struct_literal.field_count; j++) {
                if (strcmp(node->as.struct_literal.field_names[j],
                           sd->field_names[i]) == 0) {
                    compile_expr(cg, node->as.struct_literal.field_values[j]);
                    found = true;
                    break;
                }
            }
            if (!found) {
                /* Default to void for missing fields */
                emit_op(cg, OP_PUSH_VOID);
            }
        }
        emit_op(cg, OP_STRUCT_LITERAL, sd->def_idx, sd->field_count);
        break;
    }

    case AST_FIELD_ACCESS: {
        ASTNode *obj = node->as.field_access.object;
        const char *field = node->as.field_access.field_name;

        /* Check if this is EnumType.Variant (obj is an identifier naming an enum) */
        if (obj->type == AST_IDENTIFIER) {
            CgEnumDef *ed = enum_find(cg, obj->as.identifier);
            if (ed) {
                int16_t vi = enum_variant_index(ed, field);
                if (vi < 0) {
                    cg_error(cg, node->line, "unknown enum variant '%s.%s'",
                             ed->name, field);
                    break;
                }
                int val = (ed->variant_values) ? ed->variant_values[vi] : vi;
                emit_op(cg, OP_ENUM_VAL, ed->def_idx, val);
                break;
            }
        }

        /* Regular struct field access */
        compile_expr(cg, obj);

        /* Resolve field index from the struct type */
        const char *type_name = NULL;

        /* Try to determine struct type from the object expression */
        if (obj->type == AST_IDENTIFIER) {
            Symbol *sym = env_get_var(cg->env, obj->as.identifier);
            if (sym && sym->struct_type_name) {
                type_name = sym->struct_type_name;
            }
        } else if (obj->type == AST_STRUCT_LITERAL) {
            type_name = obj->as.struct_literal.struct_name;
        } else if (obj->type == AST_CALL && obj->as.call.return_struct_type_name) {
            type_name = obj->as.call.return_struct_type_name;
        }

        if (type_name) {
            CgStructDef *sd = struct_find(cg, type_name);
            if (sd) {
                int16_t fi = struct_field_index(sd, field);
                if (fi >= 0) {
                    emit_op(cg, OP_STRUCT_GET, (int)fi);
                    break;
                }
            }
        }
        /* Fallback: search all known structs for a unique field name match */
        for (int i = 0; i < cg->struct_count; i++) {
            int16_t fi = struct_field_index(&cg->structs[i], field);
            if (fi >= 0) {
                emit_op(cg, OP_STRUCT_GET, (int)fi);
                goto field_done;
            }
        }
        /* Fallback: search union variant fields (for match bindings like v.value) */
        for (int i = 0; i < cg->union_count; i++) {
            CgUnionDef *ud = &cg->unions[i];
            for (int vi = 0; vi < ud->variant_count; vi++) {
                for (int fi = 0; fi < ud->variant_field_counts[vi]; fi++) {
                    if (strcmp(ud->variant_field_names[vi][fi], field) == 0) {
                        emit_op(cg, OP_UNION_FIELD, fi);
                        goto field_done;
                    }
                }
            }
        }
        cg_error(cg, node->line, "cannot resolve field '%s'", field);
        field_done:
        break;
    }

    case AST_TUPLE_LITERAL: {
        int count = node->as.tuple_literal.element_count;
        for (int i = 0; i < count; i++) {
            compile_expr(cg, node->as.tuple_literal.elements[i]);
        }
        emit_op(cg, OP_TUPLE_NEW, count);
        break;
    }

    case AST_TUPLE_INDEX: {
        compile_expr(cg, node->as.tuple_index.tuple);
        emit_op(cg, OP_TUPLE_GET, node->as.tuple_index.index);
        break;
    }

    case AST_UNION_CONSTRUCT: {
        const char *uname = node->as.union_construct.union_name;
        const char *vname = node->as.union_construct.variant_name;
        CgUnionDef *ud = union_find(cg, uname);
        if (!ud) {
            cg_error(cg, node->line, "undefined union '%s'", uname);
            break;
        }
        int16_t vi = union_variant_index(ud, vname);
        if (vi < 0) {
            cg_error(cg, node->line, "unknown variant '%s.%s'", uname, vname);
            break;
        }
        int fc = node->as.union_construct.field_count;
        for (int i = 0; i < fc; i++) {
            compile_expr(cg, node->as.union_construct.field_values[i]);
        }
        emit_op(cg, OP_UNION_CONSTRUCT, ud->def_idx, (int)vi, fc);
        break;
    }

    case AST_MATCH: {
        /* Compile: match expr { Variant(binding) => body, ... } */
        compile_expr(cg, node->as.match_expr.expr);

        int arm_count = node->as.match_expr.arm_count;
        uint32_t end_patches[64];
        uint32_t end_instrs[64];
        int end_count = 0;

        /* Find union definition for variant name → index mapping.
         * The typechecker may set a monomorphized name (e.g., "Result_int_string")
         * but we register unions under their base name (e.g., "Result").
         * Try exact match first, then try base name (before first '_'). */
        const char *utype = node->as.match_expr.union_type_name;
        CgUnionDef *ud = utype ? union_find(cg, utype) : NULL;
        if (!ud && utype) {
            /* Try base name: look for first '_' and try prefix */
            const char *underscore = strchr(utype, '_');
            if (underscore) {
                char base[256];
                size_t blen = (size_t)(underscore - utype);
                if (blen >= sizeof(base)) blen = sizeof(base) - 1;
                memcpy(base, utype, blen);
                base[blen] = '\0';
                ud = union_find(cg, base);
            }
        }

        for (int i = 0; i < arm_count; i++) {
            const char *variant = node->as.match_expr.pattern_variants[i];
            const char *binding = node->as.match_expr.pattern_bindings[i];

            /* DUP the union value for tag check */
            emit_op(cg, OP_DUP);
            emit_op(cg, OP_UNION_TAG);

            /* Push variant index */
            int16_t vi = 0;
            if (ud) {
                vi = union_variant_index(ud, variant);
            }
            emit_op(cg, OP_PUSH_I64, (int64_t)vi);
            emit_op(cg, OP_EQ);

            /* Jump to next arm if not matching */
            uint32_t jf_instr = cg->code_size;
            uint32_t jf_off = emit_op(cg, OP_JMP_FALSE, (int32_t)0);

            /* Match succeeded: bind the entire union to the pattern variable
             * so v.value / v.error etc. can access variant fields via UNION_FIELD */
            if (binding && binding[0] != '\0') {
                emit_op(cg, OP_DUP);  /* keep union on stack */
                uint16_t bslot = local_add(cg, binding, node->line);
                emit_op(cg, OP_STORE_LOCAL, (int)bslot);
            }

            /* Pop the union value before executing body */
            emit_op(cg, OP_POP);

            /* Compile arm body */
            compile_expr(cg, node->as.match_expr.arm_bodies[i]);

            /* Jump to end */
            if (end_count < 64) {
                end_instrs[end_count] = cg->code_size;
                uint32_t je_off = emit_op(cg, OP_JMP, (int32_t)0);
                end_patches[end_count] = je_off + 1;
                end_count++;
            }

            /* Patch JMP_FALSE to here */
            patch_jump(cg, jf_off + 1, jf_instr, cg->code_size);
        }

        /* Default: pop the union, push void */
        emit_op(cg, OP_POP);
        emit_op(cg, OP_PUSH_VOID);

        /* Patch all end jumps */
        for (int i = 0; i < end_count; i++) {
            patch_jump(cg, end_patches[i], end_instrs[i], cg->code_size);
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
    case AST_STRUCT_LITERAL:
    case AST_FIELD_ACCESS:
    case AST_TUPLE_LITERAL:
    case AST_TUPLE_INDEX:
    case AST_UNION_CONSTRUCT:
    case AST_MATCH:
        compile_expr(cg, node);
        emit_op(cg, OP_POP);
        break;

    case AST_CALL:
    case AST_MODULE_QUALIFIED_CALL: {
        /* Function call as statement: compile and discard result */
        compile_expr(cg, node);
        /* Check if function returns void - if so, no POP needed
         * For now, always POP since we don't track return types in codegen */
        emit_op(cg, OP_POP);
        break;
    }

    /* Skip these - handled in Pass 1 */
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

CodegenResult codegen_compile(ASTNode *program, Environment *env,
                              ModuleList *modules) {
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

    /* ── Pass 1: Register all functions, types, and globals ──────── */
    int main_fn_idx = -1;

    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];

        if (item->type == AST_FUNCTION && !item->as.function.is_extern) {
            const char *name = item->as.function.name;
            uint32_t name_idx = nvm_add_string(cg.module, name, (uint32_t)strlen(name));

            NvmFunctionEntry fn = {0};
            fn.name_idx = name_idx;
            fn.arity = (uint16_t)item->as.function.param_count;

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

        /* Register struct definitions */
        if (item->type == AST_STRUCT_DEF && cg.struct_count < MAX_STRUCT_DEFS) {
            CgStructDef *sd = &cg.structs[cg.struct_count];
            sd->name = item->as.struct_def.name;
            sd->field_names = item->as.struct_def.field_names;
            sd->field_count = item->as.struct_def.field_count;
            sd->def_idx = cg.struct_count;
            cg.struct_count++;
        }

        /* Register enum definitions */
        if (item->type == AST_ENUM_DEF && cg.enum_count < MAX_ENUM_DEFS) {
            CgEnumDef *ed = &cg.enums[cg.enum_count];
            ed->name = item->as.enum_def.name;
            ed->variant_names = item->as.enum_def.variant_names;
            ed->variant_values = item->as.enum_def.variant_values;
            ed->variant_count = item->as.enum_def.variant_count;
            ed->def_idx = cg.enum_count;
            cg.enum_count++;
        }

        /* Register union definitions */
        if (item->type == AST_UNION_DEF && cg.union_count < MAX_UNION_DEFS) {
            CgUnionDef *ud = &cg.unions[cg.union_count];
            ud->name = item->as.union_def.name;
            ud->variant_count = item->as.union_def.variant_count;
            ud->variant_names = item->as.union_def.variant_names;
            ud->variant_field_counts = item->as.union_def.variant_field_counts;
            ud->variant_field_names = item->as.union_def.variant_field_names;
            ud->def_idx = cg.union_count;
            cg.union_count++;
        }

        /* Register extern function declarations */
        if (item->type == AST_FUNCTION && item->as.function.is_extern) {
            const char *name = item->as.function.name;
            uint16_t pc = (uint16_t)item->as.function.param_count;
            uint8_t ret_tag = type_to_tag(item->as.function.return_type);

            /* Build param type tags */
            uint8_t param_tags[16] = {0};
            for (int p = 0; p < pc && p < 16; p++) {
                param_tags[p] = type_to_tag(item->as.function.params[p].type);
            }

            register_extern(&cg, name, "", pc, ret_tag, param_tags);
        }

        /* Process import statements - load module ASTs and register their contents */
        if (item->type == AST_IMPORT) {
            const char *mod_path = item->as.import_stmt.module_path;
            const char *mod_alias = item->as.import_stmt.module_alias;

            /* Try to get the module AST from the cache (loaded by process_imports) */
            ASTNode *mod_ast = mod_path ? get_cached_module_ast(mod_path) : NULL;

            if (mod_ast && mod_ast->type == AST_PROGRAM) {
                /* Register ALL module functions (public + private) so internal
                 * calls within module functions resolve correctly. */
                for (int m = 0; m < mod_ast->as.program.count; m++) {
                    ASTNode *mitem = mod_ast->as.program.items[m];

                    /* Register all non-extern functions as bytecode functions */
                    if (mitem->type == AST_FUNCTION && !mitem->as.function.is_extern) {
                        const char *fname = mitem->as.function.name;

                        /* Check for alias: selective import may rename */
                        const char *use_name = fname;
                        if (item->as.import_stmt.is_selective) {
                            for (int s = 0; s < item->as.import_stmt.import_symbol_count; s++) {
                                if (strcmp(item->as.import_stmt.import_symbols[s], fname) == 0 &&
                                    item->as.import_stmt.import_aliases[s]) {
                                    use_name = item->as.import_stmt.import_aliases[s];
                                    break;
                                }
                            }
                        } else if (mod_alias) {
                            /* module import with alias: register as Alias.func */
                            /* Only for public functions */
                            if (mitem->as.function.is_pub) {
                                char qname[512];
                                snprintf(qname, sizeof(qname), "%s.%s", mod_alias, fname);
                                /* Also register as qualified name */
                                bool already = false;
                                for (int f = 0; f < cg.fn_count; f++) {
                                    if (strcmp(cg.functions[f].name, qname) == 0) {
                                        already = true;
                                        break;
                                    }
                                }
                                if (!already && cg.fn_count < MAX_FUNCTIONS) {
                                    uint32_t qi = nvm_add_string(cg.module, qname,
                                                                 (uint32_t)strlen(qname));
                                    NvmFunctionEntry qfn = {0};
                                    qfn.name_idx = qi;
                                    qfn.arity = (uint16_t)mitem->as.function.param_count;
                                    uint32_t qidx = nvm_add_function(cg.module, &qfn);
                                    cg.functions[cg.fn_count].name = strdup(qname);
                                    cg.functions[cg.fn_count].fn_idx = qidx;
                                    cg.fn_count++;
                                }
                            }
                        }

                        /* Register under original name (always needed for internal calls) */
                        bool already = false;
                        for (int f = 0; f < cg.fn_count; f++) {
                            if (strcmp(cg.functions[f].name, use_name) == 0) {
                                already = true;
                                break;
                            }
                        }
                        if (already) continue;

                        uint32_t name_idx = nvm_add_string(cg.module, use_name,
                                                           (uint32_t)strlen(use_name));
                        NvmFunctionEntry fn = {0};
                        fn.name_idx = name_idx;
                        fn.arity = (uint16_t)mitem->as.function.param_count;
                        uint32_t idx = nvm_add_function(cg.module, &fn);
                        if (cg.fn_count < MAX_FUNCTIONS) {
                            cg.functions[cg.fn_count].name = (char *)use_name;
                            cg.functions[cg.fn_count].fn_idx = idx;
                            cg.fn_count++;
                        }
                    }

                    /* Register module's extern function declarations */
                    if (mitem->type == AST_FUNCTION && mitem->as.function.is_extern) {
                        const char *ename = mitem->as.function.name;
                        if (extern_find(&cg, ename) < 0) {
                            uint16_t pc = (uint16_t)mitem->as.function.param_count;
                            uint8_t ret_tag = type_to_tag(mitem->as.function.return_type);
                            uint8_t param_tags[16] = {0};
                            for (int p = 0; p < pc && p < 16; p++) {
                                param_tags[p] = type_to_tag(mitem->as.function.params[p].type);
                            }
                            register_extern(&cg, ename, mod_path ? mod_path : "",
                                           pc, ret_tag, param_tags);
                        }
                    }

                    /* Register module struct definitions */
                    if (mitem->type == AST_STRUCT_DEF && cg.struct_count < MAX_STRUCT_DEFS) {
                        bool dup = false;
                        for (int s = 0; s < cg.struct_count; s++) {
                            if (strcmp(cg.structs[s].name, mitem->as.struct_def.name) == 0) {
                                dup = true;
                                break;
                            }
                        }
                        if (!dup) {
                            CgStructDef *sd = &cg.structs[cg.struct_count];
                            sd->name = mitem->as.struct_def.name;
                            sd->field_names = mitem->as.struct_def.field_names;
                            sd->field_count = mitem->as.struct_def.field_count;
                            sd->def_idx = cg.struct_count;
                            cg.struct_count++;
                        }
                    }

                    /* Register module enum definitions */
                    if (mitem->type == AST_ENUM_DEF && cg.enum_count < MAX_ENUM_DEFS) {
                        bool dup = false;
                        for (int e = 0; e < cg.enum_count; e++) {
                            if (strcmp(cg.enums[e].name, mitem->as.enum_def.name) == 0) {
                                dup = true;
                                break;
                            }
                        }
                        if (!dup) {
                            CgEnumDef *ed = &cg.enums[cg.enum_count];
                            ed->name = mitem->as.enum_def.name;
                            ed->variant_names = mitem->as.enum_def.variant_names;
                            ed->variant_values = mitem->as.enum_def.variant_values;
                            ed->variant_count = mitem->as.enum_def.variant_count;
                            ed->def_idx = cg.enum_count;
                            cg.enum_count++;
                        }
                    }
                }
            } else {
                /* Module AST not available - fall back to registering as externs */
                const char *mod_name = mod_alias ? mod_alias : mod_path;
                if (item->as.import_stmt.is_selective) {
                    for (int s = 0; s < item->as.import_stmt.import_symbol_count; s++) {
                        const char *sym = item->as.import_stmt.import_symbols[s];
                        const char *alias = item->as.import_stmt.import_aliases[s];
                        const char *local_name = alias ? alias : sym;
                        Function *fn = env_get_function(cg.env, sym);
                        if (!fn) {
                            char qname[512];
                            snprintf(qname, sizeof(qname), "%s.%s", mod_name, sym);
                            fn = env_get_function(cg.env, qname);
                        }
                        if (fn) {
                            uint16_t pc = (uint16_t)fn->param_count;
                            uint8_t ret_tag = type_to_tag(fn->return_type);
                            uint8_t param_tags[16] = {0};
                            for (int p = 0; p < pc && p < 16; p++) {
                                param_tags[p] = type_to_tag(fn->params[p].type);
                            }
                            register_extern(&cg, local_name, mod_name ? mod_name : "",
                                           pc, ret_tag, param_tags);
                        }
                    }
                } else {
                    if (mod_name && cg.env) {
                        for (int f = 0; f < cg.env->function_count; f++) {
                            Function *fn = &cg.env->functions[f];
                            size_t prefix_len = strlen(mod_name);
                            if (fn->name && strncmp(fn->name, mod_name, prefix_len) == 0 &&
                                fn->name[prefix_len] == '.') {
                                uint16_t pc = (uint16_t)fn->param_count;
                                uint8_t ret_tag = type_to_tag(fn->return_type);
                                uint8_t param_tags[16] = {0};
                                for (int p = 0; p < pc && p < 16; p++) {
                                    param_tags[p] = type_to_tag(fn->params[p].type);
                                }
                                register_extern(&cg, fn->name, mod_name,
                                               pc, ret_tag, param_tags);
                            }
                        }
                    }
                }
            }
        }

        /* Register top-level let bindings as globals */
        if (item->type == AST_LET && cg.global_count < MAX_GLOBALS) {
            cg.globals[cg.global_count].name = item->as.let.name;
            cg.globals[cg.global_count].slot = cg.global_count;
            cg.global_count++;
        }
    }

    /* ── Pass 1.5: Compile top-level let bindings as globals ─────── */
    if (cg.global_count > 0) {
        /* Create an __init__ function to initialize globals */
        NvmFunctionEntry init_fn = {0};
        uint32_t init_name = nvm_add_string(cg.module, "__init__", 8);
        init_fn.name_idx = init_name;
        init_fn.arity = 0;
        uint32_t init_idx = nvm_add_function(cg.module, &init_fn);

        cg.code_size = 0;
        cg.local_count = 0;
        cg.loop_depth = 0;

        for (int i = 0; i < program->as.program.count; i++) {
            ASTNode *item = program->as.program.items[i];
            if (item->type == AST_LET) {
                compile_expr(&cg, item->as.let.value);
                int16_t gslot = global_find(&cg, item->as.let.name);
                if (gslot >= 0) {
                    emit_op(&cg, OP_STORE_GLOBAL, (uint32_t)gslot);
                }
            }
        }
        emit_op(&cg, OP_PUSH_VOID);
        emit_op(&cg, OP_RET);

        if (!cg.had_error) {
            uint32_t code_off = nvm_append_code(cg.module, cg.code, cg.code_size);
            NvmFunctionEntry *entry = &cg.module->functions[init_idx];
            entry->code_offset = code_off;
            entry->code_length = cg.code_size;
            entry->local_count = cg.local_count;
        }
        (void)init_idx;
    }

    /* ── Pass 2: Compile function bodies ────────────────────────── */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_FUNCTION && !item->as.function.is_extern) {
            compile_function(&cg, item);
            if (cg.had_error) break;
        }
    }

    /* ── Pass 2b: Compile imported module function bodies ──────── */
    if (modules && !cg.had_error) {
        for (int mi = 0; mi < modules->count; mi++) {
            ASTNode *mod_ast = get_cached_module_ast(modules->module_paths[mi]);
            if (!mod_ast || mod_ast->type != AST_PROGRAM) continue;

            for (int m = 0; m < mod_ast->as.program.count; m++) {
                ASTNode *mitem = mod_ast->as.program.items[m];
                if (mitem->type == AST_FUNCTION && !mitem->as.function.is_extern) {
                    compile_function(&cg, mitem);
                    if (cg.had_error) break;
                }
            }
            if (cg.had_error) break;
        }
    }

    /* Set entry point and flags */
    if (main_fn_idx >= 0) {
        cg.module->header.flags = NVM_FLAG_HAS_MAIN;
        cg.module->header.entry_point = (uint32_t)main_fn_idx;
    }
    if (cg.extern_count > 0) {
        cg.module->header.flags |= NVM_FLAG_NEEDS_EXTERN;
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
