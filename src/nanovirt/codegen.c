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
#define MAX_UPVALUES    64
#define CODE_INITIAL    4096

/* ── Internal structures ────────────────────────────────────────── */

typedef struct {
    char *name;
    uint16_t slot;
    char *struct_type;  /* Struct type name for field resolution (NULL if not a struct) */
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
    char **field_type_names;  /* Struct type names for struct-typed fields (NULL entries for non-struct) */
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

/* Upvalue descriptor: a captured variable from a parent scope */
typedef struct {
    char *name;              /* Variable name */
    uint16_t parent_slot;    /* Slot in parent's locals (or parent's upvalues) */
    bool is_local;           /* true = parent local, false = parent upvalue */
} Upvalue;

typedef struct CG CG;
struct CG {
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

    /* Upvalue tracking for closure captures */
    Upvalue upvalues[MAX_UPVALUES];
    uint16_t upvalue_count;
    CG *parent;              /* Parent scope for nested function compilation */

    /* Error state */
    bool had_error;
    int error_line;
    char error_msg[256];
};

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
    cg->locals[slot].struct_type = NULL;
    cg->local_count++;
    return slot;
}

/* Find the struct type name for a local variable (for field access resolution) */
static const char *local_struct_type(CG *cg, const char *name) {
    for (int i = cg->local_count - 1; i >= 0; i--) {
        if (strcmp(cg->locals[i].name, name) == 0)
            return cg->locals[i].struct_type;
    }
    return NULL;
}

/* ── Function lookup ────────────────────────────────────────────── */

static int32_t fn_find(CG *cg, const char *name) {
    for (int i = 0; i < cg->fn_count; i++) {
        if (strcmp(cg->functions[i].name, name) == 0)
            return (int32_t)cg->functions[i].fn_idx;
    }
    return -1;
}

/* ── Upvalue resolution ─────────────────────────────────────────── */

/* Check if this CG already has an upvalue for 'name' */
static int16_t upvalue_find(CG *cg, const char *name) {
    for (int i = 0; i < cg->upvalue_count; i++) {
        if (strcmp(cg->upvalues[i].name, name) == 0)
            return (int16_t)i;
    }
    return -1;
}

/* Add an upvalue entry. Returns index or -1 on overflow. */
static int16_t upvalue_add(CG *cg, const char *name, uint16_t parent_slot, bool is_local) {
    if (cg->upvalue_count >= MAX_UPVALUES) return -1;
    int16_t idx = (int16_t)cg->upvalue_count;
    cg->upvalues[idx].name = (char *)name;
    cg->upvalues[idx].parent_slot = parent_slot;
    cg->upvalues[idx].is_local = is_local;
    cg->upvalue_count++;
    return idx;
}

/*
 * Resolve a variable as an upvalue by walking the parent CG chain.
 * Returns the upvalue index in cg->upvalues[], or -1 if not found.
 *
 * If found in parent's locals: is_local=true, parent_slot = local slot
 * If found in parent's upvalues: is_local=false, parent_slot = upvalue index
 */
static int16_t upvalue_resolve(CG *cg, const char *name) {
    if (!cg->parent) return -1;

    /* Already resolved? */
    int16_t existing = upvalue_find(cg, name);
    if (existing >= 0) return existing;

    /* Check parent's locals */
    int16_t parent_local = local_find(cg->parent, name);
    if (parent_local >= 0) {
        return upvalue_add(cg, name, (uint16_t)parent_local, true);
    }

    /* Check parent's upvalues (recursive capture through grandparent) */
    int16_t parent_upval = upvalue_resolve(cg->parent, name);
    if (parent_upval >= 0) {
        return upvalue_add(cg, name, (uint16_t)parent_upval, false);
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

/*
 * Infer the struct type name of an expression node.
 * Used for chained field access like `a.b.c` where we need to know
 * the type of intermediate expressions.
 */
static const char *infer_expr_struct_type(CG *cg, ASTNode *node) {
    if (!node) return NULL;

    if (node->type == AST_IDENTIFIER) {
        const char *lt = local_struct_type(cg, node->as.identifier);
        if (lt) return lt;
        Symbol *sym = env_get_var(cg->env, node->as.identifier);
        if (sym && sym->struct_type_name) return sym->struct_type_name;
        return NULL;
    }

    if (node->type == AST_STRUCT_LITERAL) {
        return node->as.struct_literal.struct_name;
    }

    if (node->type == AST_CALL && node->as.call.return_struct_type_name) {
        return node->as.call.return_struct_type_name;
    }

    if (node->type == AST_FIELD_ACCESS) {
        /* Recursively determine: what struct type does the object have? */
        const char *obj_type = infer_expr_struct_type(cg, node->as.field_access.object);
        if (obj_type) {
            CgStructDef *sd = struct_find(cg, obj_type);
            if (sd) {
                int16_t fi = struct_field_index(sd, node->as.field_access.field_name);
                if (fi >= 0 && sd->field_type_names && sd->field_type_names[fi]) {
                    return sd->field_type_names[fi];
                }
            }
        }
        return NULL;
    }

    return NULL;
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

/* ── Module introspection inline compilation ───────────────────── */

/* Handle ___module_* calls inline instead of FFI so wrapper binaries work.
 * Returns true if the call was handled, false if not a ___module_ pattern. */
static bool compile_module_introspection(CG *cg, const char *name) {
    if (!name || strncmp(name, "___module_", 10) != 0 || !cg->env)
        return false;

    const char *rest = name + 10;

    /* Extract the pattern and module name from function name */
    const char *mname = NULL;
    ModuleInfo *mi = NULL;

    if (strncmp(rest, "is_unsafe_", 10) == 0) {
        mname = rest + 10;
        mi = env_get_module(cg->env, mname);
        emit_op(cg, OP_PUSH_BOOL, mi ? (uint32_t)mi->is_unsafe : 0);
        return true;
    }
    if (strncmp(rest, "has_ffi_", 8) == 0) {
        mname = rest + 8;
        mi = env_get_module(cg->env, mname);
        emit_op(cg, OP_PUSH_BOOL, mi ? (uint32_t)mi->has_ffi : 0);
        return true;
    }
    if (strncmp(rest, "function_count_", 15) == 0) {
        mname = rest + 15;
        mi = env_get_module(cg->env, mname);
        emit_op(cg, OP_PUSH_I64, (uint32_t)(mi ? mi->function_count : 0));
        return true;
    }
    if (strncmp(rest, "struct_count_", 13) == 0) {
        mname = rest + 13;
        mi = env_get_module(cg->env, mname);
        emit_op(cg, OP_PUSH_I64, (uint32_t)(mi ? mi->struct_count : 0));
        return true;
    }
    if (strncmp(rest, "name_", 5) == 0) {
        mname = rest + 5;
        uint32_t sidx = nvm_add_string(cg->module, mname, (uint32_t)strlen(mname));
        emit_op(cg, OP_PUSH_STR, sidx);
        return true;
    }
    if (strncmp(rest, "path_", 5) == 0) {
        mname = rest + 5;
        mi = env_get_module(cg->env, mname);
        const char *path = (mi && mi->path) ? mi->path : "";
        uint32_t sidx = nvm_add_string(cg->module, path, (uint32_t)strlen(path));
        emit_op(cg, OP_PUSH_STR, sidx);
        return true;
    }
    /* function_name_ and struct_name_ take an index arg - skip for now,
     * those are rare and still work via FFI in --run mode */
    return false;
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

    /* Math library functions - handled as extern FFI calls */
    {
        static const char *math_fns_1arg[] = {
            "sqrt", "sin", "cos", "tan", "asin", "acos", "atan",
            "floor", "ceil", "round", "log", "log2", "log10", "exp",
            NULL
        };
        static const char *math_fns_2arg[] = {
            "pow", "atan2", "fmod",
            NULL
        };
        for (int mi = 0; math_fns_1arg[mi]; mi++) {
            if (strcmp(name, math_fns_1arg[mi]) == 0 && argc == 1) {
                compile_expr(cg, args[0]);
                /* Look up or auto-register as extern */
                int32_t ext_idx = extern_find(cg, name);
                if (ext_idx < 0) {
                    uint8_t ptags[1] = {TAG_FLOAT};
                    register_extern(cg, name, "", 1, TAG_FLOAT, ptags);
                    ext_idx = extern_find(cg, name);
                }
                if (ext_idx >= 0) emit_op(cg, OP_CALL_EXTERN, (uint32_t)ext_idx);
                return true;
            }
        }
        for (int mi = 0; math_fns_2arg[mi]; mi++) {
            if (strcmp(name, math_fns_2arg[mi]) == 0 && argc == 2) {
                compile_expr(cg, args[0]);
                compile_expr(cg, args[1]);
                int32_t ext_idx = extern_find(cg, name);
                if (ext_idx < 0) {
                    uint8_t ptags[2] = {TAG_FLOAT, TAG_FLOAT};
                    register_extern(cg, name, "", 2, TAG_FLOAT, ptags);
                    ext_idx = extern_find(cg, name);
                }
                if (ext_idx >= 0) emit_op(cg, OP_CALL_EXTERN, (uint32_t)ext_idx);
                return true;
            }
        }
    }

    /* Math (inline implementations) */
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
    if ((strcmp(name, "at") == 0 || strcmp(name, "array_get") == 0) && argc == 2) {
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
    if (strcmp(name, "array_slice") == 0 && argc == 3) {
        compile_expr(cg, args[0]); /* array */
        compile_expr(cg, args[1]); /* start */
        compile_expr(cg, args[2]); /* end */
        emit_op(cg, OP_ARR_SLICE);
        return true;
    }

    /* String char_at */
    if ((strcmp(name, "str_char_at") == 0 || strcmp(name, "char_at") == 0) && argc == 2) {
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

    /* Hashmap operations */
    if (strcmp(name, "hashmap_new") == 0 && argc == 0) {
        emit_op(cg, OP_HM_NEW, TAG_STRING, TAG_INT);
        return true;
    }
    if (strcmp(name, "hashmap_get") == 0 && argc == 2) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        emit_op(cg, OP_HM_GET);
        return true;
    }
    if (strcmp(name, "hashmap_set") == 0 && argc == 3) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        compile_expr(cg, args[2]);
        emit_op(cg, OP_HM_SET);
        return true;
    }
    if (strcmp(name, "hashmap_has") == 0 && argc == 2) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        emit_op(cg, OP_HM_HAS);
        return true;
    }
    if (strcmp(name, "hashmap_delete") == 0 && argc == 2) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        emit_op(cg, OP_HM_DELETE);
        return true;
    }
    if (strcmp(name, "hashmap_keys") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_HM_KEYS);
        return true;
    }
    if (strcmp(name, "hashmap_values") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_HM_VALUES);
        return true;
    }
    if (strcmp(name, "hashmap_length") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_HM_LEN);
        return true;
    }

    /* Array concat */
    if (strcmp(name, "array_concat") == 0 && argc == 2) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        /* Concatenate arrays: iterate second array and push each to first */
        uint16_t arr1 = local_add(cg, "__concat_a__", 0);
        emit_op(cg, OP_STORE_LOCAL, (int)arr1);
        uint16_t arr2 = local_add(cg, "__concat_b__", 0);
        emit_op(cg, OP_STORE_LOCAL, (int)arr2);
        /* Copy arr1 into result (in-place for now) */
        emit_op(cg, OP_LOAD_LOCAL, (int)arr2);
        emit_op(cg, OP_ARR_LEN);
        uint16_t len2 = local_add(cg, "__concat_len__", 0);
        emit_op(cg, OP_STORE_LOCAL, (int)len2);
        emit_op(cg, OP_PUSH_I64, (int64_t)0);
        uint16_t idx = local_add(cg, "__concat_i__", 0);
        emit_op(cg, OP_STORE_LOCAL, (int)idx);
        uint32_t loop_top = cg->code_size;
        emit_op(cg, OP_LOAD_LOCAL, (int)idx);
        emit_op(cg, OP_LOAD_LOCAL, (int)len2);
        emit_op(cg, OP_LT);
        uint32_t jf_instr = cg->code_size;
        uint32_t jf_off = emit_op(cg, OP_JMP_FALSE, (int32_t)0);
        emit_op(cg, OP_LOAD_LOCAL, (int)arr1);
        emit_op(cg, OP_LOAD_LOCAL, (int)arr2);
        emit_op(cg, OP_LOAD_LOCAL, (int)idx);
        emit_op(cg, OP_ARR_GET);
        emit_op(cg, OP_ARR_PUSH);
        emit_op(cg, OP_STORE_LOCAL, (int)arr1);
        emit_op(cg, OP_LOAD_LOCAL, (int)idx);
        emit_op(cg, OP_PUSH_I64, (int64_t)1);
        emit_op(cg, OP_ADD);
        emit_op(cg, OP_STORE_LOCAL, (int)idx);
        uint32_t jmp_instr = cg->code_size;
        emit_op(cg, OP_JMP, (int32_t)0);
        patch_jump(cg, jmp_instr + 1, jmp_instr, loop_top);
        patch_jump(cg, jf_off + 1, jf_instr, cg->code_size);
        emit_op(cg, OP_LOAD_LOCAL, (int)arr1);
        return true;
    }

    /* null_opaque() - returns a null opaque pointer (0) */
    if (strcmp(name, "null_opaque") == 0 && argc == 0) {
        emit_op(cg, OP_PUSH_I64, (int64_t)0);
        return true;
    }

    /* map_* aliases for hashmap_* operations */
    if (strcmp(name, "map_new") == 0 && argc == 0) {
        emit_op(cg, OP_HM_NEW, TAG_STRING, TAG_INT);
        return true;
    }
    if ((strcmp(name, "map_get") == 0 || strcmp(name, "hashmap_get") == 0) && argc == 2) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        emit_op(cg, OP_HM_GET);
        return true;
    }
    if ((strcmp(name, "map_set") == 0 || strcmp(name, "map_put") == 0) && argc == 3) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        compile_expr(cg, args[2]);
        emit_op(cg, OP_HM_SET);
        return true;
    }
    if (strcmp(name, "map_has") == 0 && argc == 2) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        emit_op(cg, OP_HM_HAS);
        return true;
    }
    if (strcmp(name, "map_delete") == 0 && argc == 2) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        emit_op(cg, OP_HM_DELETE);
        return true;
    }
    if (strcmp(name, "map_keys") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_HM_KEYS);
        return true;
    }
    if (strcmp(name, "map_values") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_HM_VALUES);
        return true;
    }
    if ((strcmp(name, "map_length") == 0 || strcmp(name, "map_size") == 0) && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_HM_LEN);
        return true;
    }

    /* string_to_int / string_to_float - parse string to number */
    if (strcmp(name, "string_to_int") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_CAST_INT);
        return true;
    }
    if (strcmp(name, "string_to_float") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_CAST_FLOAT);
        return true;
    }

    /* string_from_char(code) - convert char code to 1-char string */
    if (strcmp(name, "string_from_char") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        int32_t ext_idx = extern_find(cg, "vm_string_from_char");
        if (ext_idx < 0) {
            uint8_t ptags[1] = {TAG_INT};
            register_extern(cg, "vm_string_from_char", "", 1, TAG_STRING, ptags);
            ext_idx = extern_find(cg, "vm_string_from_char");
        }
        if (ext_idx >= 0) emit_op(cg, OP_CALL_EXTERN, (uint32_t)ext_idx);
        return true;
    }

    /* string_from_bytes(arr) - convert byte array to string */
    if (strcmp(name, "string_from_bytes") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        int32_t ext_idx = extern_find(cg, "vm_string_from_bytes");
        if (ext_idx < 0) {
            uint8_t ptags[1] = {TAG_ARRAY};
            register_extern(cg, "vm_string_from_bytes", "", 1, TAG_STRING, ptags);
            ext_idx = extern_find(cg, "vm_string_from_bytes");
        }
        if (ext_idx >= 0) emit_op(cg, OP_CALL_EXTERN, (uint32_t)ext_idx);
        return true;
    }

    /* digit_value(c) - convert char to digit 0-9 or -1 */
    if (strcmp(name, "digit_value") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        int32_t ext_idx = extern_find(cg, "vm_digit_value");
        if (ext_idx < 0) {
            uint8_t ptags[1] = {TAG_INT};
            register_extern(cg, "vm_digit_value", "", 1, TAG_INT, ptags);
            ext_idx = extern_find(cg, "vm_digit_value");
        }
        if (ext_idx >= 0) emit_op(cg, OP_CALL_EXTERN, (uint32_t)ext_idx);
        return true;
    }

    /* bstr_validate_utf8(str) - check if string is valid UTF-8 */
    if (strcmp(name, "bstr_validate_utf8") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        int32_t ext_idx = extern_find(cg, "vm_bstr_validate_utf8");
        if (ext_idx < 0) {
            uint8_t ptags[1] = {TAG_STRING};
            register_extern(cg, "vm_bstr_validate_utf8", "", 1, TAG_BOOL, ptags);
            ext_idx = extern_find(cg, "vm_bstr_validate_utf8");
        }
        if (ext_idx >= 0) emit_op(cg, OP_CALL_EXTERN, (uint32_t)ext_idx);
        return true;
    }

    /* Character classification builtins */
    if ((strcmp(name, "is_digit") == 0 || strcmp(name, "is_alpha") == 0 ||
         strcmp(name, "is_alnum") == 0 || strcmp(name, "is_space") == 0 ||
         strcmp(name, "is_upper") == 0 || strcmp(name, "is_lower") == 0 ||
         strcmp(name, "is_whitespace") == 0) && argc == 1) {
        compile_expr(cg, args[0]);
        char c_name[64];
        snprintf(c_name, sizeof(c_name), "vm_%s", name);
        int32_t ext_idx = extern_find(cg, c_name);
        if (ext_idx < 0) {
            uint8_t ptags[1] = {TAG_INT};
            register_extern(cg, c_name, "", 1, TAG_BOOL, ptags);
            ext_idx = extern_find(cg, c_name);
        }
        if (ext_idx >= 0) emit_op(cg, OP_CALL_EXTERN, (uint32_t)ext_idx);
        return true;
    }

    /* Generic list operations: list_T_new, list_T_push, list_T_get, list_T_length, list_T_set */
    if (strncmp(name, "list_", 5) == 0 || strncmp(name, "List_", 5) == 0) {
        /* Find the operation suffix */
        const char *suffix = strrchr(name, '_');
        if (suffix) {
            if (strcmp(suffix, "_new") == 0 && argc == 0) {
                /* list_T_new() -> create empty array */
                emit_op(cg, OP_ARR_NEW, (int)TAG_INT);
                return true;
            }
            if (strcmp(suffix, "_push") == 0 && argc == 2) {
                /* list_T_push(list, value) -> array_push */
                compile_expr(cg, args[0]);
                compile_expr(cg, args[1]);
                emit_op(cg, OP_ARR_PUSH);
                return true;
            }
            if (strcmp(suffix, "_get") == 0 && argc == 2) {
                /* list_T_get(list, index) -> array_get */
                compile_expr(cg, args[0]);
                compile_expr(cg, args[1]);
                emit_op(cg, OP_ARR_GET);
                return true;
            }
            if (strcmp(suffix, "_length") == 0 && argc == 1) {
                /* list_T_length(list) -> array_length */
                compile_expr(cg, args[0]);
                emit_op(cg, OP_ARR_LEN);
                return true;
            }
            if (strcmp(suffix, "_set") == 0 && argc == 3) {
                /* list_T_set(list, index, value) -> array_set */
                compile_expr(cg, args[0]);
                compile_expr(cg, args[1]);
                compile_expr(cg, args[2]);
                emit_op(cg, OP_ARR_SET);
                return true;
            }
        }
    }

    /* bstring operations */
    if (strcmp(name, "bstr_new") == 0 && argc == 1) {
        /* bstr_new(str) - for now, just pass through as string */
        compile_expr(cg, args[0]);
        return true;
    }
    if (strcmp(name, "bstr_length") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_STR_LEN);
        return true;
    }
    if (strcmp(name, "bstr_concat") == 0 && argc == 2) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        emit_op(cg, OP_STR_CONCAT);
        return true;
    }
    if (strcmp(name, "bstr_substring") == 0 && argc == 3) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        compile_expr(cg, args[2]);
        emit_op(cg, OP_STR_SUBSTR);
        return true;
    }
    if (strcmp(name, "bstr_to_string") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        return true;
    }
    if ((strcmp(name, "bstr_get_byte") == 0 || strcmp(name, "bstr_byte_at") == 0) && argc == 2) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        emit_op(cg, OP_STR_CHAR_AT);
        /* STR_CHAR_AT returns a string, we need the byte value */
        emit_op(cg, OP_CAST_INT);
        return true;
    }
    if (strcmp(name, "bstr_new_binary") == 0 && argc == 1) {
        /* bstr_new_binary(bytes_array) - create bstring from byte array */
        compile_expr(cg, args[0]);
        return true;
    }
    if ((strcmp(name, "bstr_to_cstr") == 0 || strcmp(name, "bstr_to_str") == 0) && argc == 1) {
        /* bstr_to_cstr/bstr_to_str - identity operation in VM (strings are strings) */
        compile_expr(cg, args[0]);
        return true;
    }
    if (strcmp(name, "bstr_equals") == 0 && argc == 2) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        emit_op(cg, OP_STR_EQ);
        return true;
    }
    if (strcmp(name, "bstr_free") == 0 && argc == 1) {
        /* No-op in VM - GC handles memory.
         * Evaluate arg, discard it, push void so caller can POP. */
        compile_expr(cg, args[0]);
        emit_op(cg, OP_POP);
        emit_op(cg, OP_PUSH_VOID);
        return true;
    }
    if (strcmp(name, "bstr_utf8_length") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        int32_t ext_idx = extern_find(cg, "vm_bstr_utf8_length");
        if (ext_idx < 0) {
            uint8_t ptags[1] = {TAG_STRING};
            register_extern(cg, "vm_bstr_utf8_length", "", 1, TAG_INT, ptags);
            ext_idx = extern_find(cg, "vm_bstr_utf8_length");
        }
        if (ext_idx >= 0) emit_op(cg, OP_CALL_EXTERN, (uint32_t)ext_idx);
        return true;
    }
    if (strcmp(name, "bstr_utf8_char_at") == 0 && argc == 2) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        int32_t ext_idx = extern_find(cg, "vm_bstr_utf8_char_at");
        if (ext_idx < 0) {
            uint8_t ptags[2] = {TAG_STRING, TAG_INT};
            register_extern(cg, "vm_bstr_utf8_char_at", "", 2, TAG_INT, ptags);
            ext_idx = extern_find(cg, "vm_bstr_utf8_char_at");
        }
        if (ext_idx >= 0) emit_op(cg, OP_CALL_EXTERN, (uint32_t)ext_idx);
        return true;
    }
    if (strcmp(name, "char_to_lower") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        int32_t ext_idx = extern_find(cg, "vm_char_to_lower");
        if (ext_idx < 0) {
            uint8_t ptags[1] = {TAG_INT};
            register_extern(cg, "vm_char_to_lower", "", 1, TAG_INT, ptags);
            ext_idx = extern_find(cg, "vm_char_to_lower");
        }
        if (ext_idx >= 0) emit_op(cg, OP_CALL_EXTERN, (uint32_t)ext_idx);
        return true;
    }
    if (strcmp(name, "char_to_upper") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        int32_t ext_idx = extern_find(cg, "vm_char_to_upper");
        if (ext_idx < 0) {
            uint8_t ptags[1] = {TAG_INT};
            register_extern(cg, "vm_char_to_upper", "", 1, TAG_INT, ptags);
            ext_idx = extern_find(cg, "vm_char_to_upper");
        }
        if (ext_idx >= 0) emit_op(cg, OP_CALL_EXTERN, (uint32_t)ext_idx);
        return true;
    }
    if (strcmp(name, "bytes_from_string") == 0 && argc == 1) {
        /* bytes_from_string(str) - convert string to byte array */
        compile_expr(cg, args[0]);
        int32_t ext_idx = extern_find(cg, "vm_bytes_from_string");
        if (ext_idx < 0) {
            uint8_t ptags[1] = {TAG_STRING};
            register_extern(cg, "vm_bytes_from_string", "", 1, TAG_ARRAY, ptags);
            ext_idx = extern_find(cg, "vm_bytes_from_string");
        }
        if (ext_idx >= 0) emit_op(cg, OP_CALL_EXTERN, (uint32_t)ext_idx);
        return true;
    }

    /* Result type helpers */
    if (strcmp(name, "result_is_ok") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_UNION_TAG);
        emit_op(cg, OP_PUSH_I64, (int64_t)0);
        emit_op(cg, OP_EQ);
        return true;
    }
    if (strcmp(name, "result_is_err") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_UNION_TAG);
        emit_op(cg, OP_PUSH_I64, (int64_t)1);
        emit_op(cg, OP_EQ);
        return true;
    }
    if (strcmp(name, "result_unwrap") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_UNION_FIELD, 0);
        return true;
    }
    if (strcmp(name, "result_unwrap_err") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_UNION_FIELD, 0);
        return true;
    }

    /* OS/IO builtins - map nanolang names to vm_* C functions */
    {
        static const struct { const char *nano_name; const char *c_name; int arity; uint8_t ret_tag; } os_builtins[] = {
            {"getcwd",     "vm_getcwd",      0, TAG_STRING},
            {"chdir",      "vm_chdir",       1, TAG_INT},
            {"file_read",  "vm_file_read",   1, TAG_STRING},
            {"file_write", "vm_file_write",  2, TAG_INT},
            {"file_exists","vm_file_exists",  1, TAG_BOOL},
            {"dir_exists", "vm_dir_exists",   1, TAG_BOOL},
            {"dir_create", "vm_dir_create",   1, TAG_INT},
            {"dir_list",   "vm_dir_list",     1, TAG_ARRAY},
            {"mktemp_dir", "vm_mktemp_dir",   1, TAG_STRING},
            {"getenv",     "vm_getenv",       1, TAG_STRING},
            {"setenv",     "vm_setenv",       2, TAG_INT},
            {"str_index_of","vm_str_index_of",2, TAG_INT},
            {"process_run","vm_process_run",  1, TAG_ARRAY},
            {NULL, NULL, 0, 0}
        };
        for (int bi = 0; os_builtins[bi].nano_name; bi++) {
            if (strcmp(name, os_builtins[bi].nano_name) == 0 &&
                argc == os_builtins[bi].arity) {
                for (int a = 0; a < argc; a++) compile_expr(cg, args[a]);
                /* Register the C function name as extern */
                int32_t ext_idx = extern_find(cg, os_builtins[bi].c_name);
                if (ext_idx < 0) {
                    uint8_t ptags[4] = {TAG_STRING, TAG_STRING, TAG_STRING, TAG_STRING};
                    register_extern(cg, os_builtins[bi].c_name, "",
                                   (uint16_t)os_builtins[bi].arity,
                                   os_builtins[bi].ret_tag, ptags);
                    ext_idx = extern_find(cg, os_builtins[bi].c_name);
                }
                if (ext_idx >= 0) emit_op(cg, OP_CALL_EXTERN, (uint32_t)ext_idx);
                return true;
            }
        }
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
                    /* Check if it's a captured variable from parent scope */
                    int16_t uv = upvalue_resolve(cg, id);
                    if (uv >= 0) {
                        emit_op(cg, OP_LOAD_UPVALUE, 0, (int)uv);
                    } else {
                        cg_error(cg, node->line, "undefined variable '%s'", id);
                    }
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

        /* Handle module introspection inline (no FFI needed) */
        if (name && compile_module_introspection(cg, name)) {
            break;
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
                } else {
                    /* Check if callee is a captured upvalue */
                    int16_t uv = name ? upvalue_resolve(cg, name) : -1;
                    if (uv >= 0) {
                        emit_op(cg, OP_LOAD_UPVALUE, 0, (int)uv);
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

        /* Try qualified name "Module.function" in bytecode functions first */
        char qname[512];
        snprintf(qname, sizeof(qname), "%s.%s", mod_alias, func_name);
        int32_t fn_idx = fn_find(cg, qname);
        if (fn_idx >= 0) {
            emit_op(cg, OP_CALL, (uint32_t)fn_idx);
        } else {
            /* Try unqualified name in bytecode functions */
            fn_idx = fn_find(cg, func_name);
            if (fn_idx >= 0) {
                emit_op(cg, OP_CALL, (uint32_t)fn_idx);
            } else {
                /* Try as extern (qualified then unqualified) */
                int32_t ext_idx = extern_find_qualified(cg, mod_alias, func_name);
                if (ext_idx >= 0) {
                    emit_op(cg, OP_CALL_EXTERN, (uint32_t)ext_idx);
                } else {
                    ext_idx = extern_find(cg, func_name);
                    if (ext_idx >= 0) {
                        emit_op(cg, OP_CALL_EXTERN, (uint32_t)ext_idx);
                    } else {
                        cg_error(cg, node->line, "undefined module function '%s.%s'",
                                 mod_alias, func_name);
                    }
                }
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
        const char *type_name = infer_expr_struct_type(cg, obj);

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

            /* Statement-block arm bodies don't leave a value on the stack.
             * Push void so match consistently produces exactly one value,
             * preventing the statement-level POP from eating local slots. */
            if (node->as.match_expr.arm_bodies[i]->type == AST_BLOCK) {
                emit_op(cg, OP_PUSH_VOID);
            }

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
        /* Track struct type for field access resolution */
        if (node->as.let.type_name) {
            cg->locals[slot].struct_type = node->as.let.type_name;
        }
        emit_op(cg, OP_STORE_LOCAL, (int)slot);
        break;
    }

    case AST_SET: {
        int16_t slot = local_find(cg, node->as.set.name);
        if (slot >= 0) {
            compile_expr(cg, node->as.set.value);
            emit_op(cg, OP_STORE_LOCAL, (int)slot);
        } else {
            int16_t gslot = global_find(cg, node->as.set.name);
            if (gslot >= 0) {
                compile_expr(cg, node->as.set.value);
                emit_op(cg, OP_STORE_GLOBAL, (uint32_t)gslot);
            } else {
                /* Check upvalues for captured mutable variables */
                int16_t uv = upvalue_resolve(cg, node->as.set.name);
                if (uv >= 0) {
                    compile_expr(cg, node->as.set.value);
                    emit_op(cg, OP_STORE_UPVALUE, 0, (int)uv);
                } else {
                    cg_error(cg, node->line, "undefined variable '%s'", node->as.set.name);
                }
            }
        }
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

    case AST_FUNCTION: {
        /* Nested function definition: compile with parent context for captures */
        if (node->as.function.is_extern) break;

        const char *name = node->as.function.name;

        /* Register nested function in module function table if not already there */
        int32_t fn_idx = fn_find(cg, name);
        if (fn_idx < 0) {
            uint32_t name_idx = nvm_add_string(cg->module, name, (uint32_t)strlen(name));
            NvmFunctionEntry fn = {0};
            fn.name_idx = name_idx;
            fn.arity = (uint16_t)node->as.function.param_count;
            fn_idx = (int32_t)nvm_add_function(cg->module, &fn);
            if (cg->fn_count < MAX_FUNCTIONS) {
                cg->functions[cg->fn_count].name = (char *)name;
                cg->functions[cg->fn_count].fn_idx = (uint32_t)fn_idx;
                cg->fn_count++;
            }
        }

        /* Save parent's compilation state */
        uint8_t *saved_code = cg->code;
        uint32_t saved_code_size = cg->code_size;
        uint32_t saved_code_cap = cg->code_cap;
        Local saved_locals[MAX_LOCALS];
        memcpy(saved_locals, cg->locals, sizeof(cg->locals));
        uint16_t saved_local_count = cg->local_count;
        uint16_t saved_param_count = cg->param_count;
        LoopCtx saved_loops[MAX_LOOP_DEPTH];
        memcpy(saved_loops, cg->loops, sizeof(cg->loops));
        int saved_loop_depth = cg->loop_depth;
        Upvalue saved_upvalues[MAX_UPVALUES];
        memcpy(saved_upvalues, cg->upvalues, sizeof(cg->upvalues));
        uint16_t saved_upvalue_count = cg->upvalue_count;
        CG *saved_parent = cg->parent;

        /* Set up child compilation context using same CG struct */
        CG parent_snapshot;
        memcpy(&parent_snapshot, cg, sizeof(CG));
        /* Restore parent's locals for upvalue resolution */
        memcpy(parent_snapshot.locals, saved_locals, sizeof(saved_locals));
        parent_snapshot.local_count = saved_local_count;
        parent_snapshot.upvalues[0].name = NULL; /* sentinel */
        parent_snapshot.upvalue_count = saved_upvalue_count;
        parent_snapshot.parent = saved_parent;

        cg->parent = &parent_snapshot;
        cg->code = malloc(CODE_INITIAL);
        cg->code_size = 0;
        cg->code_cap = CODE_INITIAL;
        cg->local_count = 0;
        cg->param_count = (uint16_t)node->as.function.param_count;
        cg->loop_depth = 0;
        cg->upvalue_count = 0;

        /* Parameters become the first locals of nested function */
        for (int i = 0; i < node->as.function.param_count; i++) {
            uint16_t slot = local_add(cg, node->as.function.params[i].name, node->line);
            if (node->as.function.params[i].struct_type_name) {
                cg->locals[slot].struct_type = node->as.function.params[i].struct_type_name;
            }
        }

        /* Compile nested function body */
        ASTNode *body = node->as.function.body;
        if (body) {
            if (body->type == AST_BLOCK) {
                for (int i = 0; i < body->as.block.count; i++) {
                    compile_stmt(cg, body->as.block.statements[i]);
                }
            } else {
                compile_expr(cg, body);
                emit_op(cg, OP_RET);
            }
        }
        if (cg->code_size == 0 || cg->code[cg->code_size - 1] != OP_RET) {
            emit_op(cg, OP_PUSH_VOID);
            emit_op(cg, OP_RET);
        }

        /* Save nested function's upvalue info before restoring parent state */
        uint16_t child_upvalue_count = cg->upvalue_count;
        Upvalue child_upvalues[MAX_UPVALUES];
        memcpy(child_upvalues, cg->upvalues, sizeof(Upvalue) * child_upvalue_count);

        /* Finalize nested function in module */
        if (!cg->had_error) {
            uint32_t code_off = nvm_append_code(cg->module, cg->code, cg->code_size);
            NvmFunctionEntry *entry = &cg->module->functions[fn_idx];
            entry->code_offset = code_off;
            entry->code_length = cg->code_size;
            entry->local_count = cg->local_count;
            entry->upvalue_count = child_upvalue_count;
        }

        /* Free child code buffer and restore parent state */
        free(cg->code);
        cg->code = saved_code;
        cg->code_size = saved_code_size;
        cg->code_cap = saved_code_cap;
        memcpy(cg->locals, saved_locals, sizeof(cg->locals));
        cg->local_count = saved_local_count;
        cg->param_count = saved_param_count;
        memcpy(cg->loops, saved_loops, sizeof(cg->loops));
        cg->loop_depth = saved_loop_depth;
        memcpy(cg->upvalues, saved_upvalues, sizeof(cg->upvalues));
        cg->upvalue_count = saved_upvalue_count;
        cg->parent = saved_parent;

        /* At the definition site: push captured values, then emit CLOSURE_NEW */
        for (int i = 0; i < child_upvalue_count; i++) {
            if (child_upvalues[i].is_local) {
                emit_op(cg, OP_LOAD_LOCAL, (int)child_upvalues[i].parent_slot);
            } else {
                emit_op(cg, OP_LOAD_UPVALUE, 0, (int)child_upvalues[i].parent_slot);
            }
        }
        emit_op(cg, OP_CLOSURE_NEW, (uint32_t)fn_idx, (int)child_upvalue_count);

        /* Store closure in a local variable named after the function */
        uint16_t closure_slot = local_add(cg, name, node->line);
        emit_op(cg, OP_STORE_LOCAL, (int)closure_slot);
        break;
    }

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
    cg->upvalue_count = 0;

    /* Parameters become the first locals */
    for (int i = 0; i < fn_node->as.function.param_count; i++) {
        uint16_t slot = local_add(cg, fn_node->as.function.params[i].name, fn_node->line);
        /* Track struct type for field access resolution */
        if (fn_node->as.function.params[i].struct_type_name) {
            cg->locals[slot].struct_type = fn_node->as.function.params[i].struct_type_name;
        }
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
    entry->upvalue_count = cg->upvalue_count;
}

/* ── Main compilation entry point ───────────────────────────────── */

CodegenResult codegen_compile(ASTNode *program, Environment *env,
                              ModuleList *modules, const char *input_file) {
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
            sd->field_type_names = item->as.struct_def.field_type_names;
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

            /* Resolve the module path (process_imports caches with resolved paths,
             * but item->as.import_stmt.module_path is the original unresolved path) */
            const char *resolved_path = mod_path ? resolve_module_path(mod_path, input_file) : NULL;
            const char *lookup_path = resolved_path ? resolved_path : mod_path;

            /* Try to get the module AST from the cache (loaded by process_imports) */
            ASTNode *mod_ast = lookup_path ? get_cached_module_ast(lookup_path) : NULL;
            if (!mod_ast && mod_path) {
                /* Fallback: try the original unresolved path */
                mod_ast = get_cached_module_ast(mod_path);
            }
            if (!mod_ast && modules) {
                /* Fallback: try matching by suffix against cached module paths */
                for (int mi = 0; mi < modules->count; mi++) {
                    const char *mp = modules->module_paths[mi];
                    if (mod_path) {
                        size_t mp_len = strlen(mp);
                        size_t path_len = strlen(mod_path);
                        if (mp_len >= path_len &&
                            strcmp(mp + mp_len - path_len, mod_path) == 0) {
                            mod_ast = get_cached_module_ast(mp);
                            if (mod_ast) break;
                        }
                    }
                }
            }

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
                        }

                        /* Register under original/aliased name first (needed for internal calls
                         * and for compile_function to find the entry by AST name) */
                        bool already = false;
                        uint32_t idx = 0;
                        for (int f = 0; f < cg.fn_count; f++) {
                            if (strcmp(cg.functions[f].name, use_name) == 0) {
                                already = true;
                                idx = cg.functions[f].fn_idx;
                                break;
                            }
                        }
                        if (!already) {
                            uint32_t name_idx = nvm_add_string(cg.module, use_name,
                                                               (uint32_t)strlen(use_name));
                            NvmFunctionEntry fn = {0};
                            fn.name_idx = name_idx;
                            fn.arity = (uint16_t)mitem->as.function.param_count;
                            idx = nvm_add_function(cg.module, &fn);
                            if (cg.fn_count < MAX_FUNCTIONS) {
                                cg.functions[cg.fn_count].name = (char *)use_name;
                                cg.functions[cg.fn_count].fn_idx = idx;
                                cg.fn_count++;
                            }
                        }

                        /* For module imports with alias, register qualified name
                         * (e.g. "Math.add") as an alias sharing the SAME fn_idx.
                         * This ensures compile_function fills in one entry and both
                         * qualified and unqualified calls resolve to it. */
                        if (!item->as.import_stmt.is_selective && mod_alias &&
                            mitem->as.function.is_pub) {
                            char qname[512];
                            snprintf(qname, sizeof(qname), "%s.%s", mod_alias, fname);
                            bool qalready = false;
                            for (int f = 0; f < cg.fn_count; f++) {
                                if (strcmp(cg.functions[f].name, qname) == 0) {
                                    qalready = true;
                                    break;
                                }
                            }
                            if (!qalready && cg.fn_count < MAX_FUNCTIONS) {
                                cg.functions[cg.fn_count].name = strdup(qname);
                                cg.functions[cg.fn_count].fn_idx = idx; /* same fn_idx! */
                                cg.fn_count++;
                            }
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
                            sd->field_type_names = mitem->as.struct_def.field_type_names;
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

                    /* Register module union definitions */
                    if (mitem->type == AST_UNION_DEF && cg.union_count < MAX_UNION_DEFS) {
                        bool dup = false;
                        for (int u = 0; u < cg.union_count; u++) {
                            if (strcmp(cg.unions[u].name, mitem->as.union_def.name) == 0) {
                                dup = true;
                                break;
                            }
                        }
                        if (!dup) {
                            CgUnionDef *ud = &cg.unions[cg.union_count];
                            ud->name = mitem->as.union_def.name;
                            ud->variant_count = mitem->as.union_def.variant_count;
                            ud->variant_names = mitem->as.union_def.variant_names;
                            ud->variant_field_counts = mitem->as.union_def.variant_field_counts;
                            ud->variant_field_names = mitem->as.union_def.variant_field_names;
                            ud->def_idx = cg.union_count;
                            cg.union_count++;
                        }
                    }

                    /* Register module-level let bindings as globals */
                    if (mitem->type == AST_LET && cg.global_count < MAX_GLOBALS) {
                        bool dup = false;
                        for (int g = 0; g < cg.global_count; g++) {
                            if (strcmp(cg.globals[g].name, mitem->as.let.name) == 0) {
                                dup = true;
                                break;
                            }
                        }
                        if (!dup) {
                            cg.globals[cg.global_count].name = mitem->as.let.name;
                            cg.globals[cg.global_count].slot = cg.global_count;
                            cg.global_count++;
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
            if (resolved_path) free((char *)resolved_path);
        }

        /* Register top-level let bindings as globals */
        if (item->type == AST_LET && cg.global_count < MAX_GLOBALS) {
            cg.globals[cg.global_count].name = item->as.let.name;
            cg.globals[cg.global_count].slot = cg.global_count;
            cg.global_count++;
        }
    }

    /* ── Pass 1b: Register transitive module dependencies ──────── */
    /* process_imports collects all modules (including transitive deps) in the
     * modules list. Pass 1 above only processes direct imports from the program.
     * Here we register any remaining modules not yet handled. */
    if (modules) {
        for (int mi = 0; mi < modules->count; mi++) {
            ASTNode *mod_ast = get_cached_module_ast(modules->module_paths[mi]);
            if (!mod_ast || mod_ast->type != AST_PROGRAM) continue;
            for (int m = 0; m < mod_ast->as.program.count; m++) {
                ASTNode *mitem = mod_ast->as.program.items[m];

                if (mitem->type == AST_FUNCTION && !mitem->as.function.is_extern) {
                    const char *fname = mitem->as.function.name;
                    if (fn_find(&cg, fname) < 0 && cg.fn_count < MAX_FUNCTIONS) {
                        uint32_t ni = nvm_add_string(cg.module, fname, (uint32_t)strlen(fname));
                        NvmFunctionEntry fn = {0};
                        fn.name_idx = ni;
                        fn.arity = (uint16_t)mitem->as.function.param_count;
                        uint32_t idx = nvm_add_function(cg.module, &fn);
                        cg.functions[cg.fn_count].name = (char *)fname;
                        cg.functions[cg.fn_count].fn_idx = idx;
                        cg.fn_count++;
                    }
                }

                if (mitem->type == AST_FUNCTION && mitem->as.function.is_extern) {
                    const char *ename = mitem->as.function.name;
                    if (extern_find(&cg, ename) < 0) {
                        uint16_t pc = (uint16_t)mitem->as.function.param_count;
                        uint8_t ret_tag = type_to_tag(mitem->as.function.return_type);
                        uint8_t param_tags[16] = {0};
                        for (int p = 0; p < pc && p < 16; p++) {
                            param_tags[p] = type_to_tag(mitem->as.function.params[p].type);
                        }
                        register_extern(&cg, ename, modules->module_paths[mi],
                                       pc, ret_tag, param_tags);
                    }
                }

                if (mitem->type == AST_STRUCT_DEF && cg.struct_count < MAX_STRUCT_DEFS) {
                    bool dup = false;
                    for (int s = 0; s < cg.struct_count; s++) {
                        if (strcmp(cg.structs[s].name, mitem->as.struct_def.name) == 0) {
                            dup = true; break;
                        }
                    }
                    if (!dup) {
                        CgStructDef *sd = &cg.structs[cg.struct_count];
                        sd->name = mitem->as.struct_def.name;
                        sd->field_names = mitem->as.struct_def.field_names;
                        sd->field_type_names = mitem->as.struct_def.field_type_names;
                        sd->field_count = mitem->as.struct_def.field_count;
                        sd->def_idx = cg.struct_count;
                        cg.struct_count++;
                    }
                }

                if (mitem->type == AST_ENUM_DEF && cg.enum_count < MAX_ENUM_DEFS) {
                    bool dup = false;
                    for (int e = 0; e < cg.enum_count; e++) {
                        if (strcmp(cg.enums[e].name, mitem->as.enum_def.name) == 0) {
                            dup = true; break;
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

                if (mitem->type == AST_UNION_DEF && cg.union_count < MAX_UNION_DEFS) {
                    bool dup = false;
                    for (int u = 0; u < cg.union_count; u++) {
                        if (strcmp(cg.unions[u].name, mitem->as.union_def.name) == 0) {
                            dup = true; break;
                        }
                    }
                    if (!dup) {
                        CgUnionDef *ud = &cg.unions[cg.union_count];
                        ud->name = mitem->as.union_def.name;
                        ud->variant_count = mitem->as.union_def.variant_count;
                        ud->variant_names = mitem->as.union_def.variant_names;
                        ud->variant_field_counts = mitem->as.union_def.variant_field_counts;
                        ud->variant_field_names = mitem->as.union_def.variant_field_names;
                        ud->def_idx = cg.union_count;
                        cg.union_count++;
                    }
                }

                if (mitem->type == AST_LET && cg.global_count < MAX_GLOBALS) {
                    bool dup = false;
                    for (int g = 0; g < cg.global_count; g++) {
                        if (strcmp(cg.globals[g].name, mitem->as.let.name) == 0) {
                            dup = true; break;
                        }
                    }
                    if (!dup) {
                        cg.globals[cg.global_count].name = mitem->as.let.name;
                        cg.globals[cg.global_count].slot = cg.global_count;
                        cg.global_count++;
                    }
                }
            }
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

        /* Initialize module globals first (they may be referenced by module functions) */
        if (modules) {
            for (int mi = 0; mi < modules->count; mi++) {
                ASTNode *mod_ast = get_cached_module_ast(modules->module_paths[mi]);
                if (!mod_ast || mod_ast->type != AST_PROGRAM) continue;
                for (int m = 0; m < mod_ast->as.program.count; m++) {
                    ASTNode *mitem = mod_ast->as.program.items[m];
                    if (mitem->type == AST_LET) {
                        compile_expr(&cg, mitem->as.let.value);
                        int16_t gslot = global_find(&cg, mitem->as.let.name);
                        if (gslot >= 0) {
                            emit_op(&cg, OP_STORE_GLOBAL, (uint32_t)gslot);
                        }
                    }
                }
            }
        }
        /* Then initialize program globals */
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

    /* For shadow-only programs (no main), generate a synthetic main that returns 0 */
    if (main_fn_idx < 0 && !cg.had_error) {
        NvmFunctionEntry syn_fn = {0};
        uint32_t syn_name = nvm_add_string(cg.module, "main", 4);
        syn_fn.name_idx = syn_name;
        syn_fn.arity = 0;
        uint32_t syn_idx = nvm_add_function(cg.module, &syn_fn);

        cg.code_size = 0;
        cg.local_count = 0;
        cg.loop_depth = 0;
        emit_op(&cg, OP_PUSH_I64, (int64_t)0);
        emit_op(&cg, OP_RET);

        uint32_t code_off = nvm_append_code(cg.module, cg.code, cg.code_size);
        NvmFunctionEntry *entry = &cg.module->functions[syn_idx];
        entry->code_offset = code_off;
        entry->code_length = cg.code_size;
        entry->local_count = 0;
        main_fn_idx = (int)syn_idx;
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
