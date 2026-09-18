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

#include "../checked_loop_binding.h"
#include "nanovirt/codegen.h"
#include "../nanoisa/local_bindings.h"
#include "nanolang.h"
#include "nanoisa/isa.h"
#include "nanoisa/nvm_format.h"
#include "nanovm/vm.h"
#include "generated/compiler_schema.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

/* ── Limits ─────────────────────────────────────────────────────── */

/* My compiler bodies exceed 256 lets; local operands are u16. */
#define MAX_LOCALS      1024
/* I include dependency functions, qualified aliases, and selected shadows. */
#define MAX_FUNCTIONS   4096
#define MAX_PATCHES     1024
#define MAX_LOOP_DEPTH  32
#define MAX_BREAKS      64
#define MAX_STRUCT_DEFS 128
#define MAX_ENUM_DEFS   64
#define MAX_UNION_DEFS  64
#define MAX_GLOBALS     VM_MAX_GLOBALS
#define MAX_EXTERNS     256
#define MAX_UPVALUES    64
#define CODE_INITIAL    4096

/* ── Internal structures ────────────────────────────────────────── */

typedef struct CgLocalName {
    NvmLocalBinding binding;
    struct CgLocalName *next;
} CgLocalName;

typedef struct {
    char *name;
    uint16_t slot;
    CgLocalName *advisory;
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
    int continue_index_slot;        /* for index, or -1 for while */
} LoopCtx;

typedef struct {
    char *name;
    uint32_t fn_idx;
    ASTNode *body; /* I identify a source function independently of its short name. */
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

typedef struct CgPassive {
    uint32_t *words, word_count;
    struct CgPassive *next;
} CgPassive;

typedef struct CG CG;
struct CG {
    /* Module being built */
    NvmModule *module;
    Environment *env;
    CgPassive *passive;
    CgLocalName **local_names;
    bool names_enabled;

    /* Current function's code buffer */
    uint8_t *code;
    uint32_t code_size;
    uint32_t code_cap;

    /* Local variables for current function */
    Local locals[MAX_LOCALS];
    uint16_t local_count;
    uint16_t local_binding_count;
    uint16_t param_count;
    uint32_t current_fn_idx;
    Type current_return_element_type;

    /* Function table (populated in pass 1) */
    FnEntry functions[MAX_FUNCTIONS];
    uint16_t fn_count;

    /* Loop context stack */
    LoopCtx loops[MAX_LOOP_DEPTH];
    int loop_depth;
    int effect_depth;
    int handler_loop_floor;

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
    for (int i = cg->local_binding_count - 1; i >= 0; i--) {
        if (strcmp(cg->locals[i].name, name) == 0)
            return (int16_t)cg->locals[i].slot;
    }
    return -1;
}

static uint16_t local_add(CG *cg, const char *name, int line) {
    if (cg->local_count >= MAX_LOCALS || cg->local_binding_count >= MAX_LOCALS) {
        cg_error(cg, line, "too many local variables");
        return 0;
    }
    uint16_t slot = cg->local_count;
    Local *binding = &cg->locals[cg->local_binding_count++];
    binding->name = (char *)name;
    binding->slot = slot;
    binding->struct_type = NULL;
    binding->advisory = NULL;
    cg->local_count++;
    return slot;
}

static bool local_name_scalar(Type type) {
    return type==TYPE_INT || type==TYPE_FLOAT || type==TYPE_BOOL || type==TYPE_U8 || type==TYPE_STRING;
}
static void local_name_begin(CG *cg,uint16_t slot,const char *name,Type type,int line) {
    if(!cg->names_enabled || !local_name_scalar(type) || !name || !*name || cg->had_error)return;
    CgLocalName *entry=calloc(1,sizeof *entry);
    if(!entry){cg_error(cg,line,"I cannot retain a lexical local name");return;}
    entry->binding=(NvmLocalBinding){.function=cg->current_fn_idx,.slot=slot,
        .begin=cg->code_size,.end=cg->code_size,.name=(const uint8_t *)name,.name_size=(uint32_t)strlen(name)};
    entry->next=*cg->local_names;*cg->local_names=entry;
    cg->locals[cg->local_binding_count-1].advisory=entry;
}
static void local_names_end(CG *cg,uint16_t first) {
    for(uint16_t i=first;i<cg->local_binding_count;i++) {
        if(cg->locals[i].advisory) {
            cg->locals[i].advisory->binding.end=cg->code_size;
            cg->locals[i].advisory=NULL;
        }
    }
}
static void publish_local_names(CG *cg) {
    while(*cg->local_names) {
        CgLocalName *entry=*cg->local_names;
        if(!cg->had_error && !nvm_add_local_binding(cg->module,&entry->binding))
            cg_error(cg,0,"I cannot publish this lexical local name");
        *cg->local_names=entry->next;free(entry);
    }
}

/* Find the struct type name for a local variable (for field access resolution) */
static const char *local_struct_type(CG *cg, const char *name) {
    for (int i = cg->local_binding_count - 1; i >= 0; i--) {
        if (strcmp(cg->locals[i].name, name) == 0)
            return cg->locals[i].struct_type;
    }
    return NULL;
}

/* ── Function lookup ────────────────────────────────────────────── */

static int32_t fn_find_body(CG *cg, ASTNode *body) {
    if (!body) return -1;
    for (int i = 0; i < cg->fn_count; i++) {
        if (cg->functions[i].body == body)
            return (int32_t)cg->functions[i].fn_idx;
    }
    return -1;
}

static int32_t fn_find(CG *cg, const char *name) {
    Function *function = env_get_function(cg->env, name);
    if (function && !function->is_extern && function->body)
        return fn_find_body(cg, function->body);
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

    if (node->type == AST_CALL) {
        if (node->as.call.return_struct_type_name)
            return node->as.call.return_struct_type_name;
        Function *fn = env_get_function(cg->env, node->as.call.name);
        if (fn) return fn->return_struct_type_name;
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
static uint8_t type_to_tag(Type t, const char *name, Environment *env) {
    /* I retain the runtime kind of named opaque signatures. The parser uses
     * TYPE_STRUCT for named types, so the enum alone is not a representation. */
    if (t == TYPE_STRUCT && name && env_get_opaque_type(env, name)) return TAG_OPAQUE;
    if (t == TYPE_STRUCT && name && env_get_union(env, name)) return TAG_UNION;
    if (t == TYPE_STRUCT && name && env_get_enum(env, name)) return TAG_INT;
    switch (t) {
        case TYPE_INT:     return TAG_INT;
        case TYPE_U8:      return TAG_U8;
        case TYPE_FLOAT:   return TAG_FLOAT;
        case TYPE_BOOL:    return TAG_BOOL;
        case TYPE_STRING:  return TAG_STRING;
        case TYPE_BSTRING: return TAG_BSTRING;
        case TYPE_VOID:    return TAG_VOID;
        case TYPE_ARRAY:   return TAG_ARRAY;
        case TYPE_LIST_INT:
        case TYPE_LIST_STRING:
        case TYPE_LIST_TOKEN:
        case TYPE_LIST_GENERIC: return TAG_ARRAY;
        case TYPE_STRUCT:  return TAG_STRUCT;
        case TYPE_OPEN_RECORD: return TAG_STRUCT;
        case TYPE_ENUM:    return TAG_INT;
        case TYPE_UNION:   return TAG_UNION;
        case TYPE_FUNCTION: return TAG_FUNCTION;
        case TYPE_TUPLE:   return TAG_TUPLE;
        case TYPE_HASHMAP: return TAG_HASHMAP;
        case TYPE_OPAQUE:  return TAG_OPAQUE;
        default:           return TAG_VOID;
    }
}

static uint8_t list_element_tag(CG *cg, const char *name, const char *suffix) {
    const char *type_name = name + 5;
    size_t type_name_len = (size_t)(suffix - type_name);

    if ((type_name_len == 3 && strncmp(type_name, "int", 3) == 0) ||
        (type_name_len == 5 && strncmp(type_name, "Token", 5) == 0)) {
        return TAG_INT;
    }
    if (type_name_len == 2 && strncmp(type_name, "u8", 2) == 0) return TAG_U8;
    if (type_name_len == 5 && strncmp(type_name, "float", 5) == 0) return TAG_FLOAT;
    if (type_name_len == 4 && strncmp(type_name, "bool", 4) == 0) return TAG_BOOL;
    if (type_name_len == 6 && strncmp(type_name, "string", 6) == 0) return TAG_STRING;

    for (int i = 0; i < cg->struct_count; i++) {
        if (strlen(cg->structs[i].name) == type_name_len &&
            strncmp(cg->structs[i].name, type_name, type_name_len) == 0) {
            return TAG_STRUCT;
        }
    }
    for (int i = 0; i < cg->enum_count; i++) {
        if (strlen(cg->enums[i].name) == type_name_len &&
            strncmp(cg->enums[i].name, type_name, type_name_len) == 0) {
            return TAG_INT;
        }
    }
    return TAG_INT;
}

/* I preserve the declaration's callback shape at its selected import boundary. */
bool codegen_bind_callback_contract(NvmModule *module, uint32_t import_index,
                                   const ASTNode *declaration, Environment *env,
                                   const char *adapter_symbol, bool worker_thread) {
    if (!module || import_index >= module->import_count || !declaration ||
        declaration->type != AST_FUNCTION || !declaration->as.function.is_extern ||
        !adapter_symbol || !adapter_symbol[0]) return false;
    const NvmImportEntry *import = &module->imports[import_index];
    const char *name = nvm_get_string(module, import->function_name_idx);
    if (!name || !declaration->as.function.name || strcmp(name, declaration->as.function.name) ||
        declaration->as.function.param_count != import->param_count ||
        import->param_count > NANO_MAX_FFI_ARGS ||
        type_to_tag(declaration->as.function.return_type,
                    declaration->as.function.return_struct_type_name, env) != import->return_type)
        return false;
    NvmCallbackContract contracts[NANO_MAX_FFI_ARGS] = {{0}};
    uint16_t count = 0;
    for (uint16_t p = 0; p < import->param_count; p++) {
        const Parameter *parameter = &declaration->as.function.params[p];
        if (!module->import_param_types || !module->import_param_types[import_index] ||
            type_to_tag(parameter->type, parameter->struct_type_name, env) !=
                module->import_param_types[import_index][p]) return false;
        if (parameter->type != TYPE_FUNCTION) continue;
        const FunctionSignature *signature = parameter->fn_sig;
        if (!signature || signature->param_count < 0 || signature->param_count > NANO_MAX_FFI_ARGS ||
            (signature->param_count && !signature->param_types)) return false;
        NvmCallbackContract *contract = &contracts[count++];
        contract->parameter_idx = p;
        contract->param_count = (uint16_t)signature->param_count;
        contract->return_tag = type_to_tag(signature->return_type, signature->return_struct_name, env);
        if (contract->return_tag == TAG_VOID && signature->return_type != TYPE_VOID) return false;
        for (uint16_t a = 0; a < contract->param_count; a++)
            contract->param_tags[a] = type_to_tag(signature->param_types[a],
                signature->param_struct_names ? signature->param_struct_names[a] : NULL, env);
        if (!nvm_callback_shape_valid(contract->param_tags, contract->param_count, contract->return_tag))
            return false;
    }
    if (!count) {
        count = 1;
        contracts[0].parameter_idx = NVM_CALLBACK_NO_PARAMETER;
    }
    uint32_t adapter = nvm_add_string(module, adapter_symbol, (uint32_t)strlen(adapter_symbol));
    if (adapter == UINT32_MAX) return false;
    for (uint16_t i = 0; i < count; i++) {
        contracts[i].import_idx = import_index;
        contracts[i].adapter_name_idx = adapter;
        contracts[i].abi_version = NVM_CALLBACK_ABI_RETAINED_V1;
        contracts[i].execution = worker_thread ? NVM_FOREIGN_WORKER_THREAD : NVM_FOREIGN_OWNER_THREAD;
        if (!nvm_add_callback_contract(module, &contracts[i])) return false;
    }
    return true;
}

static uint8_t module_introspection_result(const char *name, uint16_t *arity) {
    if (!name || strncmp(name, "___module_", 10)) return TAG_VOID;
    const char *rest = name + 10;
    *arity = 0;
    if (!strncmp(rest, "is_unsafe_", 10) || !strncmp(rest, "has_ffi_", 8)) return TAG_BOOL;
    if (!strncmp(rest, "function_count_", 15) || !strncmp(rest, "struct_count_", 13)) return TAG_INT;
    if (!strncmp(rest, "name_", 5) || !strncmp(rest, "path_", 5)) return TAG_STRING;
    if (!strncmp(rest, "function_name_", 14) || !strncmp(rest, "struct_name_", 12)) {
        *arity = 1;
        return TAG_STRING;
    }
    return TAG_VOID;
}

/* Register an extern function in the codegen extern table and NVM import table */
static void register_extern(CG *cg, const char *name, const char *module_name,
                           uint16_t param_count, uint8_t return_tag,
                           const uint8_t *param_tags) {
    uint16_t intrinsic_arity = 0;
    uint8_t intrinsic_result = module_introspection_result(name, &intrinsic_arity);
    if (intrinsic_result != TAG_VOID) {
        if (param_count != intrinsic_arity || return_tag != intrinsic_result ||
            (intrinsic_arity && (!param_tags || param_tags[0] != TAG_INT)))
            cg_error(cg, 0, "I require the declared module introspection signature");
        /* These declarations lower to module facts, not external host calls. */
        return;
    }
    if (param_count > NANO_MAX_FFI_ARGS) {
        cg_error(cg, 0, "I cannot import a function with more than 16 foreign arguments");
        return;
    }
    if (cg->extern_count >= MAX_EXTERNS) return;

    /* These declarations name my host runtime, not a foreign module's
     * artifact. Binding them to libstd would invent exports it does not own. */
    const char *runtime_names[] = {"get_argc", "get_argv", "nl_os_system",
        "nl_os_getenv", "nl_os_setenv", "nl_os_unsetenv", "nl_exec_capture", "nl_exec_shell",
        "nl_timing_get_microseconds", "nl_timing_get_nanoseconds", "nl_get_time_ms"};
    for (size_t i = 0; i < sizeof(runtime_names) / sizeof(runtime_names[0]); i++)
        if (strcmp(name, runtime_names[i]) == 0) { module_name = ""; break; }

    /* Add to NVM import table */
    uint32_t mod_str = nvm_add_string(cg->module, module_name, (uint32_t)strlen(module_name));
    uint32_t fn_str = nvm_add_string(cg->module, name, (uint32_t)strlen(name));
    if (mod_str == UINT32_MAX || fn_str == UINT32_MAX) {
        cg_error(cg, 0, "I could not allocate foreign import names");
        return;
    }
    uint32_t imp_idx = nvm_add_import(cg->module, mod_str, fn_str,
                                       param_count, return_tag, param_tags);
    if (imp_idx == UINT32_MAX) {
        cg_error(cg, 0, "I could not allocate a foreign import");
        return;
    }

    /* Add to codegen extern table */
    ExternFn *ef = &cg->externs[cg->extern_count];
    ef->name = strdup(name);
    ef->module_name = strdup(module_name);
    ef->import_idx = imp_idx;
    ef->param_count = param_count;
    ef->return_tag = return_tag;
    cg->extern_count++;
}

/* ── Module introspection inline compilation ───────────────────── */

/* I consume the already evaluated index once and preserve the empty-string
 * result outside the exported-name range. */
static void compile_module_names(CG *cg, char **names, int count) {
    emit_op(cg, OP_DUP);
    emit_op(cg, OP_PUSH_I64, (int64_t)0);
    emit_op(cg, OP_I64_LT_S);
    uint32_t lower = emit_op(cg, OP_JMP_TRUE, (int32_t)0);
    emit_op(cg, OP_DUP);
    emit_op(cg, OP_PUSH_I64, (int64_t)count);
    emit_op(cg, OP_I64_GE_S);
    uint32_t upper = emit_op(cg, OP_JMP_TRUE, (int32_t)0);
    uint16_t index = local_add(cg, "", 0);
    emit_op(cg, OP_STORE_LOCAL, (int)index);
    for (int i = 0; i < count; i++) {
        uint32_t text = nvm_add_string(cg->module, names[i], (uint32_t)strlen(names[i]));
        emit_op(cg, OP_PUSH_STR, text);
    }
    emit_op(cg, OP_ARR_LITERAL, (int)TAG_STRING, (uint32_t)count);
    emit_op(cg, OP_LOAD_LOCAL, (int)index);
    emit_op(cg, OP_ARR_GET);
    uint32_t done = emit_op(cg, OP_JMP, (int32_t)0);
    patch_jump(cg, lower + 1, lower, cg->code_size);
    patch_jump(cg, upper + 1, upper, cg->code_size);
    emit_op(cg, OP_POP);
    emit_op(cg, OP_PUSH_STR, nvm_add_string(cg->module, "", 0));
    patch_jump(cg, done + 1, done, cg->code_size);
}

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
        emit_op(cg, OP_PUSH_I64, (int64_t)(mi ? mi->function_count : 0));
        return true;
    }
    if (strncmp(rest, "struct_count_", 13) == 0) {
        mname = rest + 13;
        mi = env_get_module(cg->env, mname);
        emit_op(cg, OP_PUSH_I64, (int64_t)(mi ? mi->struct_count : 0));
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
    if (strncmp(rest, "function_name_", 14) == 0) {
        mi = env_get_module(cg->env, rest + 14);
        compile_module_names(cg, mi ? mi->exported_functions : NULL,
                             mi ? mi->function_count : 0);
        return true;
    }
    if (strncmp(rest, "struct_name_", 12) == 0) {
        mi = env_get_module(cg->env, rest + 12);
        compile_module_names(cg, mi ? mi->exported_structs : NULL,
                             mi ? mi->struct_count : 0);
        return true;
    }
    return false;
}

/* ── Expression compilation ─────────────────────────────────────── */

static void compile_expr(CG *cg, ASTNode *node);
static void compile_stmt(CG *cg, ASTNode *node);
static void compile_nested_function(CG *cg, ASTNode *node);
static bool stmt_falls_through(ASTNode *node);
static void compile_par_guards(CG *cg, ASTNode *body);
static bool expr_leaves_value(CG *cg, ASTNode *node);
static void bind_parameter_type(CG *cg, const Parameter *param, int line);

static void compile_numeric_expr(CG *cg, ASTNode *node, Type type,
                                 bool want_float) {
    compile_expr(cg, node);
    if (type == TYPE_U8) emit_op(cg, OP_CAST_INT);
    if (want_float && type != TYPE_FLOAT) emit_op(cg, OP_CAST_FLOAT);
}

/* Handle built-in function calls. Returns true if handled, false if not a builtin. */
static bool compile_builtin_call(CG *cg, ASTNode *node) {
    const char *name = node->as.call.name;
    int argc = node->as.call.arg_count;
    ASTNode **args = node->as.call.args;

    /* println / print (as function calls, not AST_PRINT) */
    if (strcmp(name, "println") == 0 || strcmp(name, "print") == 0) {
        if (argc >= 1) compile_expr(cg, args[0]);
        else emit_op(cg, OP_PUSH_VOID);
        /* println adds a newline; print does not */
        emit_op(cg, strcmp(name, "println") == 0 ? OP_PRINTLN : OP_PRINT);
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
    if (strcmp(name, "str_trim") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_STR_TRIM);
        return true;
    }
    if (strcmp(name, "str_to_lower") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_STR_TO_LOWER);
        return true;
    }
    if (strcmp(name, "str_to_upper") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_STR_TO_UPPER);
        return true;
    }
    if (strcmp(name, "str_starts_with") == 0 && argc == 2) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        emit_op(cg, OP_STR_STARTS_WITH);
        return true;
    }
    if (strcmp(name, "str_ends_with") == 0 && argc == 2) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        emit_op(cg, OP_STR_ENDS_WITH);
        return true;
    }
    if (strcmp(name, "str_split") == 0 && argc == 2) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        emit_op(cg, OP_STR_SPLIT);
        return true;
    }
    if (strcmp(name, "str_replace") == 0 && argc == 3) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        compile_expr(cg, args[2]);
        emit_op(cg, OP_STR_REPLACE);
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
    if (strcmp(name, "float_to_string") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_CAST_STRING);
        const char *markers[] = {".", "e", "n", "i"};
        uint32_t done[4];
        for (int i = 0; i < 4; i++) {
            emit_op(cg, OP_DUP);
            emit_op(cg, OP_PUSH_STR, nvm_add_string(cg->module, markers[i], 1));
            emit_op(cg, OP_STR_CONTAINS);
            done[i] = emit_op(cg, OP_JMP_TRUE, (int32_t)0);
        }
        emit_op(cg, OP_PUSH_STR, nvm_add_string(cg->module, ".0", 2));
        emit_op(cg, OP_STR_CONCAT);
        for (int i = 0; i < 4; i++)
            patch_jump(cg, done[i] + 1, done[i], cg->code_size);
        return true;
    }
    if ((strcmp(name, "cast_string") == 0 || strcmp(name, "to_string") == 0 ||
         strcmp(name, "int_to_string") == 0 ||
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
        bool is_float = check_expression(args[0], cg->env) == TYPE_FLOAT;
        compile_expr(cg, args[0]);
        emit_op(cg, OP_DUP);
        if (is_float) emit_op(cg, OP_PUSH_F64, 0.0);
        else emit_op(cg, OP_PUSH_I64, (int64_t)0);
        emit_op(cg, is_float ? OP_F64_LT : OP_I64_LT_S);
        uint32_t jf_instr = cg->code_size;
        uint32_t jf_off = emit_op(cg, OP_JMP_FALSE, (int32_t)0);
        emit_op(cg, is_float ? OP_F64_NEG : OP_I64_NEG);
        patch_jump(cg, jf_off + 1, jf_instr, cg->code_size);
        return true;
    }
    if ((strcmp(name, "min") == 0 || strcmp(name, "max") == 0) && argc == 2) {
        /* I evaluate once in source order, compare copies, and keep an original. */
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        emit_op(cg, OP_PICK, 1); /* a b a */
        emit_op(cg, OP_PICK, 1); /* a b a b */
        emit_op(cg, strcmp(name, "min") == 0 ? OP_LT : OP_GT);
        uint32_t jf_instr = cg->code_size;
        uint32_t jf_off = emit_op(cg, OP_JMP_FALSE, (int32_t)0);
        /* The comparison selected a: discard b. */
        emit_op(cg, OP_POP);
        uint32_t je_instr = cg->code_size;
        uint32_t je_off = emit_op(cg, OP_JMP, (int32_t)0);
        /* Otherwise keep b, including the equal case. */
        patch_jump(cg, jf_off + 1, jf_instr, cg->code_size);
        emit_op(cg, OP_SWAP);
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

    if (strcmp(name, "array_sort") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        int32_t ext_idx = extern_find(cg, "vm_array_sort");
        if (ext_idx < 0) {
            uint8_t ptags[1] = {TAG_ARRAY};
            register_extern(cg, "vm_array_sort", "", 1, TAG_ARRAY, ptags);
            ext_idx = extern_find(cg, "vm_array_sort");
        }
        if (ext_idx >= 0) emit_op(cg, OP_CALL_EXTERN, (uint32_t)ext_idx);
        return true;
    }

    if (strcmp(name, "array_reverse") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        uint16_t src = local_add(cg, "__reverse_src__", node->line);
        emit_op(cg, OP_STORE_LOCAL, (int)src);
        emit_op(cg, OP_LOAD_LOCAL, (int)src);
        emit_op(cg, OP_ARR_LEN);
        uint16_t index = local_add(cg, "__reverse_index__", node->line);
        emit_op(cg, OP_STORE_LOCAL, (int)index);
        emit_op(cg, OP_LOAD_LOCAL, (int)src);
        emit_op(cg, OP_PUSH_I64, (int64_t)0);
        emit_op(cg, OP_PUSH_I64, (int64_t)0);
        emit_op(cg, OP_ARR_SLICE);
        uint16_t result = local_add(cg, "__reverse_result__", node->line);
        emit_op(cg, OP_STORE_LOCAL, (int)result);
        uint32_t top = cg->code_size;
        emit_op(cg, OP_LOAD_LOCAL, (int)index);
        emit_op(cg, OP_PUSH_I64, (int64_t)0);
        emit_op(cg, OP_I64_GT_S);
        uint32_t end = emit_op(cg, OP_JMP_FALSE, (int32_t)0);
        emit_op(cg, OP_LOAD_LOCAL, (int)index);
        emit_op(cg, OP_PUSH_I64, (int64_t)1);
        emit_op(cg, OP_I64_SUB);
        emit_op(cg, OP_STORE_LOCAL, (int)index);
        emit_op(cg, OP_LOAD_LOCAL, (int)result);
        emit_op(cg, OP_LOAD_LOCAL, (int)src);
        emit_op(cg, OP_LOAD_LOCAL, (int)index);
        emit_op(cg, OP_ARR_GET);
        emit_op(cg, OP_ARR_PUSH);
        emit_op(cg, OP_STORE_LOCAL, (int)result);
        uint32_t again = emit_op(cg, OP_JMP, (int32_t)0);
        patch_jump(cg, again + 1, again, top);
        patch_jump(cg, end + 1, end, cg->code_size);
        emit_op(cg, OP_LOAD_LOCAL, (int)result);
        return true;
    }

    if ((strcmp(name, "array_contains") == 0 || strcmp(name, "array_index_of") == 0) && argc == 2) {
        compile_expr(cg, args[0]);
        uint16_t src = local_add(cg, "__search_src__", node->line);
        emit_op(cg, OP_STORE_LOCAL, (int)src);
        compile_expr(cg, args[1]);
        uint16_t needle = local_add(cg, "__search_needle__", node->line);
        emit_op(cg, OP_STORE_LOCAL, (int)needle);
        emit_op(cg, OP_LOAD_LOCAL, (int)src);
        emit_op(cg, OP_ARR_LEN);
        uint16_t length = local_add(cg, "__search_length__", node->line);
        emit_op(cg, OP_STORE_LOCAL, (int)length);
        emit_op(cg, OP_PUSH_I64, (int64_t)0);
        uint16_t index = local_add(cg, "__search_index__", node->line);
        emit_op(cg, OP_STORE_LOCAL, (int)index);
        emit_op(cg, OP_PUSH_I64, (int64_t)-1);
        uint16_t result = local_add(cg, "__search_result__", node->line);
        emit_op(cg, OP_STORE_LOCAL, (int)result);
        uint32_t top = cg->code_size;
        emit_op(cg, OP_LOAD_LOCAL, (int)index);
        emit_op(cg, OP_LOAD_LOCAL, (int)length);
        emit_op(cg, OP_I64_LT_S);
        uint32_t end = emit_op(cg, OP_JMP_FALSE, (int32_t)0);
        emit_op(cg, OP_LOAD_LOCAL, (int)src);
        emit_op(cg, OP_LOAD_LOCAL, (int)index);
        emit_op(cg, OP_ARR_GET);
        emit_op(cg, OP_LOAD_LOCAL, (int)needle);
        emit_op(cg, OP_EQ);
        uint32_t next = emit_op(cg, OP_JMP_FALSE, (int32_t)0);
        emit_op(cg, OP_LOAD_LOCAL, (int)index);
        emit_op(cg, OP_STORE_LOCAL, (int)result);
        uint32_t found = emit_op(cg, OP_JMP, (int32_t)0);
        patch_jump(cg, next + 1, next, cg->code_size);
        emit_op(cg, OP_LOAD_LOCAL, (int)index);
        emit_op(cg, OP_PUSH_I64, (int64_t)1);
        emit_op(cg, OP_I64_ADD);
        emit_op(cg, OP_STORE_LOCAL, (int)index);
        uint32_t again = emit_op(cg, OP_JMP, (int32_t)0);
        patch_jump(cg, again + 1, again, top);
        patch_jump(cg, end + 1, end, cg->code_size);
        patch_jump(cg, found + 1, found, cg->code_size);
        emit_op(cg, OP_LOAD_LOCAL, (int)result);
        if (strcmp(name, "array_contains") == 0) {
            emit_op(cg, OP_PUSH_I64, (int64_t)0);
            emit_op(cg, OP_I64_GE_S);
        }
        return true;
    }

    /* Array operations */
    if (strcmp(name, "array_new") == 0 && (argc == 1 || argc == 2)) {
        /* array_new(size) or array_new(size, fill_value) */
        if (argc == 2) {
            /* Create array and fill with value */
            compile_expr(cg, args[0]);  /* size */
            uint16_t sz_slot = local_add(cg, "", 0);
            emit_op(cg, OP_STORE_LOCAL, (int)sz_slot);
            compile_expr(cg, args[1]);  /* fill value */
            uint16_t fill_slot = local_add(cg, "", 0);
            emit_op(cg, OP_STORE_LOCAL, (int)fill_slot);
            /* I evaluate both operands once before rejecting a negative size. */
            emit_op(cg, OP_LOAD_LOCAL, (int)sz_slot);
            emit_op(cg, OP_PUSH_I64, (int64_t)0);
            emit_op(cg, OP_I64_GE_S);
            emit_op(cg, OP_ASSERT);
            Type fill_type = check_expression(args[1], cg->env);
            emit_op(cg, OP_ARR_NEW, (int)type_to_tag(fill_type,
                    infer_expr_struct_type(cg, args[1]), cg->env));
            uint16_t arr_slot = local_add(cg, "", 0);
            emit_op(cg, OP_STORE_LOCAL, (int)arr_slot);
            emit_op(cg, OP_PUSH_I64, (int64_t)0);
            uint16_t i_slot = local_add(cg, "", 0);
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
            emit_op(cg, OP_I64_ADD);
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
        /* ARR_POP leaves just the removed element, which is what
         * array_pop evaluates to. */
        emit_op(cg, OP_ARR_POP);
        return true;
    }
    if (strcmp(name, "array_set") == 0 && argc == 3) {
        compile_expr(cg, args[0]); /* array */
        compile_expr(cg, args[1]); /* index */
        compile_expr(cg, args[2]); /* value */
        emit_op(cg, OP_ARR_SET);
        emit_op(cg, OP_POP); /* array_set is declared void */
        return true;
    }
    if (strcmp(name, "array_remove_at") == 0 && argc == 2) {
        compile_expr(cg, args[0]); /* array */
        compile_expr(cg, args[1]); /* index */
        emit_op(cg, OP_ARR_REMOVE);
        return true;
    }
    if (strcmp(name, "array_slice") == 0 && argc == 3) {
        uint16_t slots[3];
        for (int i = 0; i < 3; i++) {
            compile_expr(cg, args[i]);
            slots[i] = local_add(cg, "__slice_arg__", 0);
            emit_op(cg, OP_STORE_LOCAL, (int)slots[i]);
        }
        emit_op(cg, OP_LOAD_LOCAL, (int)slots[0]);
        emit_op(cg, OP_ARR_LEN);
        uint16_t bound = local_add(cg, "__slice_bound__", 0);
        emit_op(cg, OP_STORE_LOCAL, (int)bound);
        /* I clamp before adding, so start + length cannot overflow. */
        for (int i = 1; i < 3; i++) {
            emit_op(cg, OP_LOAD_LOCAL, (int)slots[i]);
            emit_op(cg, OP_PUSH_I64, (int64_t)0);
            emit_op(cg, OP_I64_LT_S);
            uint32_t lower = emit_op(cg, OP_JMP_FALSE, (int32_t)0);
            emit_op(cg, OP_PUSH_I64, (int64_t)0);
            emit_op(cg, OP_STORE_LOCAL, (int)slots[i]);
            patch_jump(cg, lower + 1, lower, cg->code_size);
            emit_op(cg, OP_LOAD_LOCAL, (int)slots[i]);
            emit_op(cg, OP_LOAD_LOCAL, (int)bound);
            emit_op(cg, OP_I64_GT_S);
            uint32_t upper = emit_op(cg, OP_JMP_FALSE, (int32_t)0);
            emit_op(cg, OP_LOAD_LOCAL, (int)bound);
            emit_op(cg, OP_STORE_LOCAL, (int)slots[i]);
            patch_jump(cg, upper + 1, upper, cg->code_size);
            if (i == 1) {
                emit_op(cg, OP_LOAD_LOCAL, (int)bound);
                emit_op(cg, OP_LOAD_LOCAL, (int)slots[1]);
                emit_op(cg, OP_I64_SUB);
                emit_op(cg, OP_STORE_LOCAL, (int)bound);
            }
        }
        emit_op(cg, OP_LOAD_LOCAL, (int)slots[0]);
        emit_op(cg, OP_LOAD_LOCAL, (int)slots[1]);
        emit_op(cg, OP_LOAD_LOCAL, (int)slots[1]);
        emit_op(cg, OP_LOAD_LOCAL, (int)slots[2]);
        emit_op(cg, OP_I64_ADD);
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
        emit_op(cg, OP_I64_ADD);
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
        Type input = filter_predicate_element_type(args[1], cg->env);
        if (args[0]->type == AST_ARRAY_LITERAL &&
            args[0]->as.array_literal.element_count == 0 &&
            (input == TYPE_INT || input == TYPE_FLOAT ||
             input == TYPE_BOOL || input == TYPE_STRING)) {
            emit_op(cg, OP_ARR_NEW, (int)type_to_tag(input, NULL, cg->env));
        } else {
            compile_expr(cg, args[0]);
        }
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

        /* I preserve the source representation even when no element survives. */
        emit_op(cg, OP_LOAD_LOCAL, (int)src_slot);
        emit_op(cg, OP_PUSH_I64, (int64_t)0);
        emit_op(cg, OP_PUSH_I64, (int64_t)0);
        emit_op(cg, OP_ARR_SLICE);
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

        /* Call predicate: fn(elem) -> bool */
        emit_op(cg, OP_LOAD_LOCAL, (int)elem_slot);
        emit_op(cg, OP_LOAD_LOCAL, (int)fn_slot);
        emit_op(cg, OP_CALL_INDIRECT, 1, 1);

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
        emit_op(cg, OP_I64_ADD);
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

        Type mapped_type = map_transform_result_type(args[1], cg->env);
        emit_op(cg, OP_ARR_NEW, (int)type_to_tag(mapped_type, NULL, cg->env));
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
        emit_op(cg, OP_CALL_INDIRECT, 1, 1);   /* fn(elem) -> value */

        /* Push result to output array */
        emit_op(cg, OP_LOAD_LOCAL, (int)res_slot);
        emit_op(cg, OP_SWAP);
        emit_op(cg, OP_ARR_PUSH);
        emit_op(cg, OP_STORE_LOCAL, (int)res_slot);

        emit_op(cg, OP_LOAD_LOCAL, (int)idx_slot);
        emit_op(cg, OP_PUSH_I64, (int64_t)1);
        emit_op(cg, OP_I64_ADD);
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
        emit_op(cg, OP_CALL_INDIRECT, 2, 1);   /* fn(acc, elem) -> acc */
        emit_op(cg, OP_STORE_LOCAL, (int)acc_slot);

        emit_op(cg, OP_LOAD_LOCAL, (int)idx_slot);
        emit_op(cg, OP_PUSH_I64, (int64_t)1);
        emit_op(cg, OP_I64_ADD);
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
        emit_op(cg, OP_POP); /* hashmap_set is declared void */
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
        emit_op(cg, OP_I64_ADD);
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
        if (!node->as.call.map_context_checked) {
            cg_error(cg, node->line, "I require checked key/value types for map_new");
            return true;
        }
        emit_op(cg, OP_HM_NEW,
            (int)type_to_tag(node->as.call.map_key_type, NULL, cg->env),
            (int)type_to_tag(node->as.call.map_value_type, NULL, cg->env));
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
        emit_op(cg, OP_POP); /* map_put/map_set are declared void */
        return true;
    }
    if (strcmp(name, "map_has") == 0 && argc == 2) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        emit_op(cg, OP_HM_HAS);
        return true;
    }
    if ((strcmp(name, "map_delete") == 0 || strcmp(name, "map_remove") == 0) && argc == 2) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        emit_op(cg, OP_HM_DELETE);
        if (strcmp(name, "map_remove") == 0) emit_op(cg, OP_POP);
        return true;
    }
    if (strcmp(name, "map_clear") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        uint16_t map = local_add(cg, "__clear_map__", node->line);
        emit_op(cg, OP_STORE_LOCAL, (int)map);
        emit_op(cg, OP_LOAD_LOCAL, (int)map);
        emit_op(cg, OP_HM_KEYS);
        uint16_t keys = local_add(cg, "__clear_keys__", node->line);
        emit_op(cg, OP_STORE_LOCAL, (int)keys);
        emit_op(cg, OP_LOAD_LOCAL, (int)keys);
        emit_op(cg, OP_ARR_LEN);
        uint16_t index = local_add(cg, "__clear_index__", node->line);
        emit_op(cg, OP_STORE_LOCAL, (int)index);
        uint32_t top = cg->code_size;
        emit_op(cg, OP_LOAD_LOCAL, (int)index);
        emit_op(cg, OP_PUSH_I64, (int64_t)0);
        emit_op(cg, OP_I64_GT_S);
        uint32_t end = emit_op(cg, OP_JMP_FALSE, (int32_t)0);
        emit_op(cg, OP_LOAD_LOCAL, (int)index);
        emit_op(cg, OP_PUSH_I64, (int64_t)1);
        emit_op(cg, OP_I64_SUB);
        emit_op(cg, OP_STORE_LOCAL, (int)index);
        emit_op(cg, OP_LOAD_LOCAL, (int)map);
        emit_op(cg, OP_LOAD_LOCAL, (int)keys);
        emit_op(cg, OP_LOAD_LOCAL, (int)index);
        emit_op(cg, OP_ARR_GET);
        emit_op(cg, OP_HM_DELETE);
        emit_op(cg, OP_POP);
        uint32_t again = emit_op(cg, OP_JMP, (int32_t)0);
        patch_jump(cg, again + 1, again, top);
        patch_jump(cg, end + 1, end, cg->code_size);
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

    if (strcmp(name, "str_join") == 0 && argc == 2) {
        compile_expr(cg, args[0]);
        compile_expr(cg, args[1]);
        int32_t ext_idx = extern_find(cg, "vm_str_join");
        if (ext_idx < 0) {
            uint8_t ptags[2] = {TAG_ARRAY, TAG_STRING};
            register_extern(cg, "vm_str_join", "", 2, TAG_STRING, ptags);
            ext_idx = extern_find(cg, "vm_str_join");
        }
        if (ext_idx >= 0) emit_op(cg, OP_CALL_EXTERN, (uint32_t)ext_idx);
        return true;
    }

    if (strcmp(name, "format") == 0 && argc >= 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_ARR_NEW, TAG_STRING);
        for (int i = 1; i < argc; i++) {
            compile_expr(cg, args[i]);
            emit_op(cg, OP_CAST_STRING);
            emit_op(cg, OP_ARR_PUSH);
        }
        int32_t ext_idx = extern_find(cg, "vm_format");
        if (ext_idx < 0) {
            uint8_t ptags[2] = {TAG_STRING, TAG_ARRAY};
            register_extern(cg, "vm_format", "", 2, TAG_STRING, ptags);
            ext_idx = extern_find(cg, "vm_format");
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
                emit_op(cg, OP_ARR_NEW, (int)list_element_tag(cg, name, suffix));
                return true;
            }
            if (strcmp(suffix, "_push") == 0 && argc == 2) {
                /* list_T_push(list, value) -> array_push */
                compile_expr(cg, args[0]);
                compile_expr(cg, args[1]);
                emit_op(cg, OP_ARR_PUSH);
                emit_op(cg, OP_POP); /* list_T_push is declared void */
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
                emit_op(cg, OP_POP); /* list_T_set is declared void */
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
        /* No-op in VM - GC handles memory. Evaluate and discard the argument. */
        compile_expr(cg, args[0]);
        emit_op(cg, OP_POP);
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
        emit_op(cg, OP_AGG_TAG);
        emit_op(cg, OP_PUSH_I64, (int64_t)0);
        emit_op(cg, OP_EQ);
        return true;
    }
    if (strcmp(name, "result_is_err") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_AGG_TAG);
        emit_op(cg, OP_PUSH_I64, (int64_t)1);
        emit_op(cg, OP_EQ);
        return true;
    }
    if (strcmp(name, "result_unwrap") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_AGG_GET, 0);
        return true;
    }
    if (strcmp(name, "result_unwrap_err") == 0 && argc == 1) {
        compile_expr(cg, args[0]);
        emit_op(cg, OP_AGG_GET, 0);
        return true;
    }

    /* GPU launch-geometry intrinsics.
     *
     * These are only meaningful inside a `gpu fn` body that a GPU backend
     * lowers to PTX or OpenCL. When the same source is compiled for a host
     * target the intrinsics collapse to the constants for "thread 0 of a
     * single block" — that is what the interpreter (eval.c) and the host-side
     * runtime stubs under modules/gpu already return. The VM is a host
     * target, so I fold them to the same constants at compile time rather than
     * routing them through FFI: it keeps GPU examples inside the sandbox and
     * keeps the three host paths agreeing on one answer. */
    {
        static const struct { const char *nano_name; int64_t value; } gpu_geometry[] = {
            {"thread_id_x", 0}, {"thread_id_y", 0}, {"thread_id_z", 0},
            {"block_id_x",  0}, {"block_id_y",  0}, {"block_id_z",  0},
            {"block_dim_x", 256}, {"block_dim_y", 256}, {"block_dim_z", 1},
            {"grid_dim_x",  1}, {"grid_dim_y",  1}, {"grid_dim_z",  1},
            {"global_id_x", 0}, {"global_id_y", 0},
            {NULL, 0}
        };
        if (argc == 0) {
            for (int gi = 0; gpu_geometry[gi].nano_name; gi++) {
                if (strcmp(name, gpu_geometry[gi].nano_name) == 0) {
                    emit_op(cg, OP_PUSH_I64, gpu_geometry[gi].value);
                    return true;
                }
            }
            /* A barrier synchronises threads within a block; with one thread
             * there is nothing to wait for. */
            if (strcmp(name, "gpu_barrier") == 0) {
                return true;
            }
        }
    }

    /* OS/IO builtins - map nanolang names to vm_* C functions */
    {
        static const struct { const char *nano_name; const char *c_name; int arity; uint8_t ret_tag; } os_builtins[] = {
            {"getcwd",     "vm_getcwd",      0, TAG_STRING},
            {"chdir",      "vm_chdir",       1, TAG_INT},
            {"file_read",  "vm_file_read",   1, TAG_STRING},
            {"file_read_bytes", "vm_file_read_bytes", 1, TAG_ARRAY},
            {"file_write", "vm_file_write",  2, TAG_INT},
            {"file_exists","vm_file_exists",  1, TAG_BOOL},
            {"dir_exists", "vm_dir_exists",   1, TAG_BOOL},
            {"dir_create", "vm_dir_create",   1, TAG_INT},
            {"dir_list",   "vm_dir_list",     1, TAG_ARRAY},
            {"tmp_dir",    "vm_tmp_dir",      0, TAG_STRING},
            {"mktemp",     "vm_mktemp",       1, TAG_STRING},
            {"mktemp_dir", "vm_mktemp_dir",   1, TAG_STRING},
            {"getenv",     "vm_getenv",       1, TAG_STRING},
            {"setenv",     "vm_setenv",       2, TAG_INT},
            {"str_index_of","vm_str_index_of",2, TAG_INT},
            {"str_trim_left","vm_str_trim_left",1, TAG_STRING},
            {"str_trim_right","vm_str_trim_right",1, TAG_STRING},
            {"str_last_index_of","vm_str_last_index_of",2, TAG_INT},
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

/* Effect bodies and arms have expression-block semantics: a final value
 * resumes the perform, whereas an explicit return exits the lexical function. */
static void compile_effect_block(CG *cg, ASTNode *body) {
    bool value = false;
    if (body->type == AST_BLOCK) {
        uint16_t bindings = cg->local_binding_count;
        for (int i = 0; i < body->as.block.count; i++) {
            ASTNode *statement = body->as.block.statements[i];
            value = i == body->as.block.count - 1 && ast_is_value_expression(statement->type);
            if (value) {
                compile_expr(cg, statement);
                value = expr_leaves_value(cg, statement);
            } else compile_stmt(cg, statement);
            if (!stmt_falls_through(statement)) break;
        }
        local_names_end(cg, bindings);
        cg->local_binding_count = bindings;
    } else {
        compile_expr(cg, body);
        value = expr_leaves_value(cg, body);
    }
    if (!value) emit_op(cg, OP_PUSH_VOID);
}

static uint32_t effect_operation(CG *cg, const char *effect, const char *operation) {
    if (!effect || !operation) { cg_error(cg, 0, "I require a resolved effect operation."); return 0; }
    size_t length = strlen(effect) + strlen(operation) + 2;
    char *name = malloc(length);
    if (!name) { cg_error(cg, 0, "I cannot allocate an effect name."); return 0; }
    snprintf(name, length, "%s.%s", effect, operation);
    uint32_t index = nvm_add_string(cg->module, name, (uint32_t)strlen(name));
    free(name);
    return index;
}

static ASTNode *bytecode_declaration(ASTNode *node) {
    return node && node->type == AST_ASYNC_FN ? node->as.async_fn.function : node;
}

static void compile_expr(CG *cg, ASTNode *node) {
    if (!node || cg->had_error) return;

    switch (node->type) {
    case AST_EFFECT_HANDLER: {
        ASTNode normalized = *node;
        int count = node->as.effect_handler.handler_count;
        if (count <= 0 || count > 1024) { cg_error(cg, node->line, "I require bounded, nonempty handlers."); break; }
        int *counts = calloc((size_t)count, sizeof(*counts));
        char ***parameters = calloc((size_t)count, sizeof(*parameters));
        if (!counts || !parameters) { free(counts); free(parameters); cg_error(cg, node->line, "I cannot allocate handler parameters."); break; }
        for (int i = 0; i < count; i++) {
            counts[i] = node->as.effect_handler.handler_param_names[i] ? 1 : 0;
            parameters[i] = &node->as.effect_handler.handler_param_names[i];
        }
        normalized.type = AST_HANDLE_EXPR;
        normalized.as.handle_expr.body = node->as.effect_handler.body;
        normalized.as.handle_expr.effect_name = node->as.effect_handler.effect_name;
        normalized.as.handle_expr.handler_count = count;
        normalized.as.handle_expr.handler_op_names = node->as.effect_handler.handler_op_names;
        normalized.as.handle_expr.handler_param_names = parameters;
        normalized.as.handle_expr.handler_param_counts = counts;
        normalized.as.handle_expr.handler_bodies = node->as.effect_handler.handler_bodies;
        compile_expr(cg, &normalized);
        free(counts); free(parameters);
        break;
    }
    case AST_HANDLE_EXPR: {
        int count = node->as.handle_expr.handler_count;
        if (count <= 0 || count > 1024) { cg_error(cg, node->line, "I require bounded, nonempty handlers."); break; }
        uint32_t *patches = calloc((size_t)count, sizeof(*patches));
        uint16_t *starts = calloc((size_t)count, sizeof(*starts));
        if (!patches || !starts) { free(patches); free(starts); cg_error(cg, node->line, "I cannot allocate handlers."); break; }
        uint16_t outer_bindings = cg->local_binding_count;
        /* Reserve each arm's parameter slots before lowering its body. */
        for (int i = 0; i < count; i++) {
            starts[i] = cg->local_count;
            int argc = node->as.handle_expr.handler_param_counts[i];
            for (int j = 0; j < argc; j++) local_add(cg, "", node->line);
            local_names_end(cg, outer_bindings);
        cg->local_binding_count = outer_bindings;
            patches[i] = emit_op(cg, OP_HANDLER_PUSH,
                effect_operation(cg, node->as.handle_expr.effect_name, node->as.handle_expr.handler_op_names[i]),
                (int32_t)0, (int)starts[i], argc);
        }
        if (cg->had_error) { free(patches); free(starts); break; }
        cg->effect_depth++;
        compile_effect_block(cg, node->as.handle_expr.body);
        emit_op(cg, OP_HANDLER_POP, count);
        uint32_t skip = emit_op(cg, OP_JMP, (int32_t)0);
        for (int i = 0; i < count && !cg->had_error; i++) {
            patch_jump(cg, patches[i] + 5, patches[i], cg->code_size);
            int argc = node->as.handle_expr.handler_param_counts[i];
            int symbol_start = cg->env->symbol_count;
            EffectDef *effect = env_get_effect(cg->env, node->as.handle_expr.effect_name);
            EffectOp *operation = effect ? effect_get_op(effect, node->as.handle_expr.handler_op_names[i]) : NULL;
            if (!operation || operation->param_count != argc) {
                cg_error(cg, node->line, "I require a resolved handler signature."); break;
            }
            for (int j = 0; j < argc; j++) {
                Local *local = &cg->locals[cg->local_binding_count++];
                local->advisory = NULL;
                local->name = node->as.handle_expr.handler_param_names[i][j];
                local->slot = starts[i] + j;
                local->struct_type = operation->params[j].struct_type_name;
                Parameter parameter = operation->params[j];
                parameter.name = local->name;
                bind_parameter_type(cg, &parameter, node->as.handle_expr.handler_bodies[i]->line);
            }
            /* An arm may loop locally; it cannot jump into a suspended
             * caller's lexical loop without unwinding the effect activation. */
            int saved_loop_floor = cg->handler_loop_floor;
            cg->handler_loop_floor = cg->loop_depth;
            compile_effect_block(cg, node->as.handle_expr.handler_bodies[i]);
            cg->handler_loop_floor = saved_loop_floor;
            emit_op(cg, OP_EFFECT_RESUME);
            cg->env->symbol_count = symbol_start;
            local_names_end(cg, outer_bindings);
        cg->local_binding_count = outer_bindings;
        }
        cg->effect_depth--;
        if (!cg->had_error) patch_jump(cg, skip + 1, skip, cg->code_size);
        free(patches); free(starts);
        break;
    }
    case AST_EFFECT_OP:
        for (int i = 0; i < node->as.effect_op.arg_count; i++) compile_expr(cg, node->as.effect_op.args[i]);
        emit_op(cg, OP_PERFORM, effect_operation(cg, node->as.effect_op.effect_name, node->as.effect_op.op_name), node->as.effect_op.arg_count);
        break;

    case AST_AWAIT:
        /* My scalar async contract is synchronous; I do not create promises. */
        compile_expr(cg, node->as.await_expr.expr);
        break;

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
        char *decoded = nl_unescape_string(node->as.string_val);
        if (!decoded) { cg_error(cg, node->line, "I could not decode the string literal"); break; }
        uint32_t idx = nvm_add_string(cg->module, decoded, (uint32_t)strlen(decoded));
        free(decoded);
        emit_op(cg, OP_PUSH_STR, idx);
        break;
    }

    case AST_IDENTIFIER: {
        const char *id = node->as.identifier;
        if (node->lambda_definition) {
            compile_nested_function(cg, node->lambda_definition);
            int16_t slot = local_find(cg, id);
            if (slot >= 0) emit_op(cg, OP_LOAD_LOCAL, (int)slot);
            else cg_error(cg, node->line, "I could not instantiate this anonymous closure");
            break;
        }
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
                    emit_op(cg, OP_FUNCREF, (uint32_t)fn_idx);
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
            case TYPE_ARRAY:  elem_tag = TAG_ARRAY;  break;
            case TYPE_FLOAT:  elem_tag = TAG_FLOAT;  break;
            case TYPE_BOOL:   elem_tag = TAG_BOOL;   break;
            case TYPE_STRING: elem_tag = TAG_STRING;  break;
            case TYPE_STRUCT: elem_tag = TAG_STRUCT; break;
            case TYPE_UNION:  elem_tag = TAG_UNION; break;
            case TYPE_ENUM:   elem_tag = TAG_INT; break;
            case TYPE_TUPLE:  elem_tag = TAG_TUPLE; break;
            case TYPE_FUNCTION: elem_tag = TAG_FUNCTION; break;
            case TYPE_U8: elem_tag = TAG_U8; break;
            case TYPE_BSTRING: elem_tag = TAG_BSTRING; break;
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
            Type arg_type = check_expression(args[0], cg->env);
            compile_expr(cg, args[0]);
            switch (op) {
                case TOKEN_MINUS:
                    emit_op(cg, arg_type == TYPE_FLOAT ? OP_F64_NEG : OP_I64_NEG);
                    break;
                case TOKEN_NOT: emit_op(cg, OP_BOOL_NOT); break;
                default:
                    cg_error(cg, node->line, "unsupported unary operator %d", op);
            }
        } else if (argc == 2) {
            /* Binary operators */
            Type left = check_expression(args[0], cg->env);
            Type right = check_expression(args[1], cg->env);
            bool array_op = left == TYPE_ARRAY || right == TYPE_ARRAY;
            bool float_op = left == TYPE_FLOAT || right == TYPE_FLOAT;
            bool string_concat = op == TOKEN_PLUS
                && left == TYPE_STRING && right == TYPE_STRING;
            compile_numeric_expr(cg, args[0], left, float_op && !array_op);
            compile_numeric_expr(cg, args[1], right, float_op && !array_op);
            switch (op) {
                case TOKEN_PLUS:
                    emit_op(cg, string_concat ? OP_STR_CONCAT
                        : array_op ? OP_ARRAY_ADD
                        : float_op ? OP_F64_ADD : OP_I64_ADD); break;
                case TOKEN_MINUS:
                    emit_op(cg, array_op ? OP_ARRAY_SUB
                        : float_op ? OP_F64_SUB : OP_I64_SUB); break;
                case TOKEN_STAR:
                    emit_op(cg, array_op ? OP_ARRAY_MUL
                        : float_op ? OP_F64_MUL : OP_I64_MUL); break;
                case TOKEN_SLASH:
                    emit_op(cg, array_op ? OP_ARRAY_DIV
                        : float_op ? OP_F64_DIV : OP_I64_DIV_S); break;
                case TOKEN_PERCENT: emit_op(cg, OP_I64_REM_S); break;
                case TOKEN_EQ:
                    emit_op(cg, array_op || (left != TYPE_INT && left != TYPE_ENUM && !float_op)
                        ? OP_EQ : (float_op ? OP_F64_EQ : OP_I64_EQ)); break;
                case TOKEN_NE:
                    emit_op(cg, array_op || (left != TYPE_INT && left != TYPE_ENUM && !float_op)
                        ? OP_NE : (float_op ? OP_F64_NE : OP_I64_NE)); break;
                case TOKEN_LT: emit_op(cg, float_op ? OP_F64_LT : OP_I64_LT_S); break;
                case TOKEN_LE: emit_op(cg, float_op ? OP_F64_LE : OP_I64_LE_S); break;
                case TOKEN_GT: emit_op(cg, float_op ? OP_F64_GT : OP_I64_GT_S); break;
                case TOKEN_GE: emit_op(cg, float_op ? OP_F64_GE : OP_I64_GE_S); break;
                case TOKEN_AND: emit_op(cg, OP_BOOL_AND); break;
                case TOKEN_OR: emit_op(cg, OP_BOOL_OR); break;
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

        /* I invoke the lexical callable, including its captures. A same-name
         * entry in the function table is not a substitute for that value. */
        int16_t callable_slot = name ? local_find(cg, name) : -1;
        int16_t callable_upvalue = name && callable_slot < 0 ? upvalue_resolve(cg, name) : -1;
        if (callable_slot >= 0 || callable_upvalue >= 0 || node->as.call.func_expr) {
            /* I snapshot the callee before arguments can mutate its binding. */
            if (node->as.call.func_expr) compile_expr(cg, node->as.call.func_expr);
            else if (callable_slot >= 0) emit_op(cg, OP_LOAD_LOCAL, (int)callable_slot);
            else emit_op(cg, OP_LOAD_UPVALUE, 0, (int)callable_upvalue);
            uint16_t saved_callee = local_add(cg, "", node->line);
            emit_op(cg, OP_STORE_LOCAL, (int)saved_callee);
            for (int i = 0; i < argc; i++) compile_expr(cg, node->as.call.args[i]);
            emit_op(cg, OP_LOAD_LOCAL, (int)saved_callee);
            emit_op(cg, OP_CALL_INDIRECT, argc,
                    check_expression(node, cg->env) == TYPE_VOID ? 0 : 1);
            break;
        }

        /* Handle built-in functions */
        if (name && fn_find(cg, name) < 0 && local_find(cg, name) < 0
                && compile_builtin_call(cg, node)) {
            break;
        }

        /* Emit arguments left-to-right */
        for (int i = 0; i < argc; i++) {
            compile_expr(cg, node->as.call.args[i]);
        }

        /* Handle module introspection inline (no FFI needed) */
        if (name && fn_find(cg, name) < 0 && compile_module_introspection(cg, name)) {
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
                    emit_op(cg, OP_CALL_INDIRECT, argc, 1);
                } else {
                    /* Check if callee is a captured upvalue */
                    int16_t uv = name ? upvalue_resolve(cg, name) : -1;
                    if (uv >= 0) {
                        emit_op(cg, OP_LOAD_UPVALUE, 0, (int)uv);
                        emit_op(cg, OP_CALL_INDIRECT, argc, 1);
                    } else if (node->as.call.func_expr) {
                        /* Computed function expression: ((get_fn) args) */
                        compile_expr(cg, node->as.call.func_expr);
                        emit_op(cg, OP_CALL_INDIRECT, argc, 1);
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
                    emit_op(cg, OP_AGG_PACK, AGG_VARIANT, ud->def_idx,
                            (int)vi, fc);
                    break;
                }
            }
        }

        CgStructDef *sd = struct_find(cg, sname);
        if (!sd) {
            cg_error(cg, node->line, "undefined struct '%s'", sname ? sname : "(null)");
            break;
        }
        ASTNode *spread = node->as.struct_literal.spread_source;
        CgStructDef *spread_def = NULL;
        uint16_t spread_slot = 0;
        if (spread) {
            const char *spread_name = infer_expr_struct_type(cg, spread);
            spread_def = spread_name ? struct_find(cg, spread_name) : NULL;
            if (!spread_def) {
                cg_error(cg, node->line, "I cannot resolve this spread source's record layout");
                break;
            }
            compile_expr(cg, spread);
            spread_slot = local_add(cg, "__spread_source__", node->line);
            emit_op(cg, OP_STORE_LOCAL, (int)spread_slot);
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
                if (spread_def) {
                    int16_t field = struct_field_index(spread_def, sd->field_names[i]);
                    if (field < 0) {
                        cg_error(cg, node->line, "I cannot inherit field '%s' from this spread source", sd->field_names[i]);
                        break;
                    }
                    emit_op(cg, OP_LOAD_LOCAL, (int)spread_slot);
                    emit_op(cg, OP_AGG_GET, (int)field);
                } else {
                    emit_op(cg, OP_PUSH_VOID);
                }
            }
        }
        emit_op(cg, OP_AGG_PACK, AGG_RECORD, sd->def_idx, 0,
                sd->field_count);
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
                /* My source enums interoperate with integers, including
                 * negative and non-contiguous explicitly assigned values. */
                emit_op(cg, OP_PUSH_I64, (int64_t)val);
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
                    emit_op(cg, OP_AGG_GET, (int)fi);
                    break;
                }
            }
        }
        /* Fallback: search all known structs for a unique field name match */
        for (int i = 0; i < cg->struct_count; i++) {
            int16_t fi = struct_field_index(&cg->structs[i], field);
            if (fi >= 0) {
                emit_op(cg, OP_AGG_GET, (int)fi);
                goto field_done;
            }
        }
        /* Fallback: search union variant fields (for match bindings like v.value) */
        for (int i = 0; i < cg->union_count; i++) {
            CgUnionDef *ud = &cg->unions[i];
            for (int vi = 0; vi < ud->variant_count; vi++) {
                for (int fi = 0; fi < ud->variant_field_counts[vi]; fi++) {
                    if (strcmp(ud->variant_field_names[vi][fi], field) == 0) {
                        emit_op(cg, OP_AGG_GET, fi);
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
        emit_op(cg, OP_AGG_PACK, AGG_TUPLE, 0, 0, count);
        break;
    }

    case AST_TUPLE_INDEX: {
        compile_expr(cg, node->as.tuple_index.tuple);
        emit_op(cg, OP_AGG_GET, node->as.tuple_index.index);
        break;
    }

    case AST_TRY_OP:
        cg_error(cg, node->line, "'?' operator not supported in VM bytecode compiler");
        break;

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
        emit_op(cg, OP_AGG_PACK, AGG_VARIANT, ud->def_idx, (int)vi, fc);
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

            uint32_t jf_instr, jf_off;

            if (strcmp(variant, "_") == 0) {
                emit_op(cg, OP_PUSH_BOOL, 1);
                jf_instr = cg->code_size;
                jf_off = emit_op(cg, OP_JMP_FALSE, (int32_t)0);
            } else if (strncmp(variant, "INT:", 4) == 0) {
                emit_op(cg, OP_DUP);
                emit_op(cg, OP_PUSH_I64, (int64_t)strtoll(variant + 4, NULL, 10));
                emit_op(cg, OP_EQ);
                jf_instr = cg->code_size;
                jf_off = emit_op(cg, OP_JMP_FALSE, (int32_t)0);
            } else if (strncmp(variant, "OR:", 3) == 0) {
                /* Or-pattern: split alternatives, emit chain of checks */
                char or_buf[512]; strncpy(or_buf, variant + 3, sizeof(or_buf)-1); or_buf[sizeof(or_buf)-1]='\0';
                char *alts[64]; int n_alts = 0;
                char *tok = strtok(or_buf, ":"); while (tok && n_alts < 64) { alts[n_alts++] = tok; tok = strtok(NULL, ":"); }

                /* For each alt except last: DUP UNION_TAG PUSH vi EQ JMP_TRUE body, then continue to next alt */
                uint32_t body_patch_offs[64];
                uint32_t body_patch_instrs[64];
                int n_body_patches = 0;
                for (int ai = 0; ai < n_alts - 1; ai++) {
                    int16_t vi_alt = ud ? union_variant_index(ud, alts[ai]) : (int16_t)ai;
                    emit_op(cg, OP_DUP); emit_op(cg, OP_AGG_TAG);
                    emit_op(cg, OP_PUSH_I64, (int64_t)vi_alt); emit_op(cg, OP_EQ);
                    uint32_t jt_instr = cg->code_size;
                    uint32_t jt_off = emit_op(cg, OP_JMP_TRUE, (int32_t)0);
                    if (n_body_patches < 64) {
                        body_patch_offs[n_body_patches] = jt_off + 1;
                        body_patch_instrs[n_body_patches] = jt_instr;
                        n_body_patches++;
                    }
                }
                /* Last alt: DUP UNION_TAG PUSH vi EQ JMP_FALSE to next */
                int16_t vi_last = ud ? union_variant_index(ud, alts[n_alts-1]) : (int16_t)(n_alts-1);
                emit_op(cg, OP_DUP); emit_op(cg, OP_AGG_TAG);
                emit_op(cg, OP_PUSH_I64, (int64_t)vi_last); emit_op(cg, OP_EQ);
                jf_instr = cg->code_size;
                jf_off = emit_op(cg, OP_JMP_FALSE, (int32_t)0);

                /* Patch all JMP_TRUE targets to here (body start, after JMP_FALSE) */
                for (int bp = 0; bp < n_body_patches; bp++) {
                    patch_jump(cg, body_patch_offs[bp], body_patch_instrs[bp], cg->code_size);
                }
            } else {
                /* DUP the union value for tag check */
                emit_op(cg, OP_DUP);
                emit_op(cg, OP_AGG_TAG);

                /* Push variant index */
                int16_t vi = 0;
                if (ud) {
                    vi = union_variant_index(ud, variant);
                }
                emit_op(cg, OP_PUSH_I64, (int64_t)vi);
                emit_op(cg, OP_EQ);

                /* Jump to next arm if not matching */
                jf_instr = cg->code_size;
                jf_off = emit_op(cg, OP_JMP_FALSE, (int32_t)0);
            }

            /* Match succeeded: bind the entire union to the pattern variable
             * so v.value / v.error etc. can access variant fields via UNION_FIELD */
            uint16_t arm_scope = cg->local_count;
            if (binding && binding[0] != '\0' && strcmp(binding, "_") != 0) {
                emit_op(cg, OP_DUP);  /* keep union on stack */
                uint16_t bslot = local_add(cg, binding, node->line);
                emit_op(cg, OP_STORE_LOCAL, (int)bslot);
            }

            uint32_t guard_instr = 0, guard_off = 0;
            ASTNode *guard = node->as.match_expr.guard_exprs ? node->as.match_expr.guard_exprs[i] : NULL;
            if (guard) {
                compile_expr(cg, guard);
                guard_instr = cg->code_size;
                guard_off = emit_op(cg, OP_JMP_FALSE, (int32_t)0);
            }

            /* Pop the union value before executing body */
            emit_op(cg, OP_POP);

            /* Compile arm body */
            ASTNode *body = node->as.match_expr.arm_bodies[i];
            if (body->type == AST_BLOCK) {
                bool value = false;
                for (int j = 0; j < body->as.block.count; j++) {
                    ASTNode *stmt = body->as.block.statements[j];
                    value = j == body->as.block.count - 1 && ast_is_value_expression(stmt->type)
                        && check_expression(stmt, cg->env) != TYPE_VOID;
                    if (value) compile_expr(cg, stmt);
                    else compile_stmt(cg, stmt);
                }
                if (!value) emit_op(cg, OP_PUSH_VOID);
            } else {
                compile_expr(cg, body);
            }
            /* Slots remain allocated, but arm-local names cannot escape. */
            for (uint16_t j = arm_scope; j < cg->local_count; j++) {
                cg->locals[j].name = "";
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
            if (guard) patch_jump(cg, guard_off + 1, guard_instr, cg->code_size);
        }

        /* No arm matched. This used to pop the union and push void, which
         * made the path look like it produced a value of the match's type
         * when it had not: for `match` in a value-returning function whose
         * arms all return, the void then reached the function's implicit RET
         * and the module would trap there with "returned 0 values, expected
         * 1" -- if it ever got there. Reaching this point is a runtime error,
         * so say so and stop. The trailing HALT is unreachable after the
         * assert traps; it is what tells the verifier this path does not
         * continue, which in turn makes an all-arms-return match correctly
         * have no fall-through at all. */
        emit_op(cg, OP_POP);
        emit_op(cg, OP_PUSH_BOOL, 0);
        emit_op(cg, OP_ASSERT);
        emit_op(cg, OP_HALT);

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

/* ── Nested function (closure) compilation ──────────────────────── */

/* Compiling a nested function has to snapshot the whole parent CG plus its
 * locals, loops and upvalue tables — well over 100 KB. That state used to live
 * in the AST_FUNCTION arm of compile_stmt's switch, and because a compiler
 * reserves the union of every arm's locals in the function's single frame,
 * *every* compile_stmt frame paid for it. Statement nesting is recursive
 * (block → if → block → ...), so roughly sixty levels of ordinary nesting
 * exhausted an 8 MB stack and nano_virt died with SIGSEGV instead of emitting
 * bytecode.
 *
 * Keeping this arm in its own function shrinks compile_stmt's frame to the
 * handful of scalars the other arms need, and heap-allocating the snapshot
 * keeps closure nesting cheap too. */
typedef struct {
    Local   locals[MAX_LOCALS];
    LoopCtx loops[MAX_LOOP_DEPTH];
    Upvalue child_upvalues[MAX_UPVALUES];
    CG      parent_snapshot;
} NestedFnState;

static void bind_parameter_type(CG *cg, const Parameter *param, int line) {
    env_define_var_with_type_info(cg->env, param->name, param->type,
                                  param->element_type, param->type_info,
                                  false, create_void());
    Symbol *symbol = env_get_var(cg->env, param->name);
    if (symbol) {
        symbol->def_line = line;
        symbol->def_column = 0;
        free(symbol->struct_type_name);
        symbol->struct_type_name = param->struct_type_name
            ? strdup(param->struct_type_name) : NULL;
        symbol->is_used = true;
    }
}

static bool record_function_parameters(CG *cg, ASTNode *node, uint32_t index) {
    int count = node->as.function.param_count;
    if (count < 0 || count > UINT16_MAX || index >= cg->module->function_count) {
        cg_error(cg, node->line, "I cannot represent this function's parameter signature");
        return false;
    }
    uint8_t *tags = count ? malloc((size_t)count) : NULL;
    if (count && !tags) {
        cg_error(cg, node->line, "I could not allocate function parameter tags");
        return false;
    }
    for (int i = 0; i < count; i++) {
        Parameter *p = &node->as.function.params[i];
        tags[i] = type_to_tag(p->type, p->struct_type_name, cg->env);
    }
    bool ok = nvm_set_function_param_types(cg->module, index, tags, (uint16_t)count);
    free(tags);
    if (!ok) cg_error(cg, node->line, "I could not preserve function parameter tags");
    return ok;
}

static void compile_nested_function(CG *cg, ASTNode *node) {
    if (node->as.function.is_extern) return;

    const char *name = node->as.function.name;

    /* Register nested function in module function table if not already there */
    int32_t fn_idx = fn_find_body(cg, node->as.function.body);
    if (fn_idx < 0) {
        uint32_t name_idx = nvm_add_string(cg->module, name, (uint32_t)strlen(name));
        NvmFunctionEntry fn = {0};
        fn.name_idx = name_idx;
        fn.arity = (uint16_t)node->as.function.param_count;
        fn.result_tag = type_to_tag(node->as.function.return_type, node->as.function.return_struct_type_name, cg->env);
        fn.result_count = fn.result_tag == TAG_VOID ? 0 : 1;
        fn_idx = (int32_t)nvm_add_function(cg->module, &fn);
        if (cg->fn_count < MAX_FUNCTIONS) {
            cg->functions[cg->fn_count].name = (char *)name;
            cg->functions[cg->fn_count].fn_idx = (uint32_t)fn_idx;
            cg->functions[cg->fn_count].body = node->as.function.body;
            cg->fn_count++;
        }
    }

    if (!record_function_parameters(cg, node, (uint32_t)fn_idx)) return;
    NestedFnState *st = malloc(sizeof(NestedFnState));
    if (!st) {
        cg_error(cg, node->line, "out of memory compiling nested function '%s'", name);
        return;
    }

    /* Save parent's compilation state */
    uint8_t *saved_code = cg->code;
    uint32_t saved_code_size = cg->code_size;
    uint32_t saved_code_cap = cg->code_cap;
    memcpy(st->locals, cg->locals, sizeof(cg->locals));
    uint16_t saved_local_count = cg->local_count;
    uint16_t saved_local_binding_count = cg->local_binding_count;
    bool saved_names_enabled = cg->names_enabled;
    cg->names_enabled = false;
    uint16_t saved_param_count = cg->param_count;
    uint32_t saved_current_fn_idx = cg->current_fn_idx;
    Type saved_return_element_type = cg->current_return_element_type;
    memcpy(st->loops, cg->loops, sizeof(cg->loops));
    int saved_loop_depth = cg->loop_depth;
    int saved_effect_depth = cg->effect_depth;
    int saved_handler_loop_floor = cg->handler_loop_floor;
    uint16_t saved_upvalue_count = cg->upvalue_count;
    CG *saved_parent = cg->parent;

    /* Set up child compilation context using same CG struct */
    memcpy(&st->parent_snapshot, cg, sizeof(CG));
    /* Restore parent's locals for upvalue resolution */
    memcpy(st->parent_snapshot.locals, st->locals, sizeof(st->locals));
    st->parent_snapshot.local_count = saved_local_count;
    st->parent_snapshot.local_binding_count = saved_local_binding_count;
    st->parent_snapshot.upvalue_count = saved_upvalue_count;
    st->parent_snapshot.parent = saved_parent;

    cg->parent = &st->parent_snapshot;
    cg->code = malloc(CODE_INITIAL);
    cg->code_size = 0;
    cg->code_cap = CODE_INITIAL;
    cg->local_count = 0;
    cg->local_binding_count = 0;
    cg->param_count = (uint16_t)node->as.function.param_count;
    cg->loop_depth = 0;
    cg->effect_depth = 0;
    cg->handler_loop_floor = 0;
    cg->upvalue_count = 0;
    cg->current_fn_idx = (uint32_t)fn_idx;
    cg->current_return_element_type = node->as.function.return_element_type;

    /* Parameters become the first locals of nested function */
    for (int i = 0; i < node->as.function.param_count; i++) {
        bind_parameter_type(cg, &node->as.function.params[i], node->line);
        local_add(cg, node->as.function.params[i].name, node->line);
        if (node->as.function.params[i].struct_type_name) {
            cg->locals[cg->local_binding_count - 1].struct_type =
                node->as.function.params[i].struct_type_name;
        }
    }

    /* Compile nested function body */
    ASTNode *body = node->as.function.body;
    compile_par_guards(cg, body);
    if (body) {
        if (body->type == AST_BLOCK) {
            for (int i = 0; i < body->as.block.count; i++) {
                compile_stmt(cg, body->as.block.statements[i]);
                if (!stmt_falls_through(body->as.block.statements[i])) break;
            }
        } else {
            compile_expr(cg, body);
            emit_op(cg, OP_RET);
        }
    }
    if (!body || stmt_falls_through(body)) {
        emit_op(cg, OP_RET);
    }

    /* Save nested function's upvalue info before restoring parent state */
    uint16_t child_upvalue_count = cg->upvalue_count;
    memcpy(st->child_upvalues, cg->upvalues, sizeof(Upvalue) * child_upvalue_count);

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
    memcpy(cg->locals, st->locals, sizeof(cg->locals));
    cg->local_count = saved_local_count;
    cg->local_binding_count = saved_local_binding_count;
    cg->names_enabled = saved_names_enabled;
    cg->param_count = saved_param_count;
    cg->current_fn_idx = saved_current_fn_idx;
    cg->current_return_element_type = saved_return_element_type;
    memcpy(cg->loops, st->loops, sizeof(cg->loops));
    cg->loop_depth = saved_loop_depth;
    cg->effect_depth = saved_effect_depth;
    cg->handler_loop_floor = saved_handler_loop_floor;
    /* Resolving a grandchild's free variable can add a capture to this
     * suspended parent. I keep those additions when resuming its compilation. */
    memcpy(cg->upvalues, st->parent_snapshot.upvalues, sizeof(cg->upvalues));
    cg->upvalue_count = st->parent_snapshot.upvalue_count;
    cg->parent = saved_parent;

    /* At the definition site: push captured values, then emit CLOSURE_NEW */
    for (int i = 0; i < child_upvalue_count; i++) {
        if (st->child_upvalues[i].is_local) {
            emit_op(cg, OP_LOAD_LOCAL, (int)st->child_upvalues[i].parent_slot);
        } else {
            emit_op(cg, OP_LOAD_UPVALUE, 0, (int)st->child_upvalues[i].parent_slot);
        }
    }
    emit_op(cg, OP_CLOSURE_NEW, (uint32_t)fn_idx, (int)child_upvalue_count);

    /* Store closure in a local variable named after the function */
    uint16_t closure_slot = local_add(cg, name, node->line);
    emit_op(cg, OP_STORE_LOCAL, (int)closure_slot);

    free(st);
}

/* ── Statement compilation ──────────────────────────────────────── */

/* Does compiling `node` as an expression leave a value on the operand stack?
 *
 * Almost always the declared type answers this: a void call like `(println x)`
 * consumes its argument and leaves nothing, so a statement that discards the
 * result must not POP. Popping anyway reaches below the operand stack into the
 * frame's locals and corrupts the frame -- silently, because it often happened
 * to cancel out, so affected programs still printed the right answers.
 *
 * `match` is an exception: its lowering pushes an explicit PUSH_VOID at the
 * end of every arm, so it leaves a value even when that value is void. Asking
 * the type for a match would skip a POP that IS needed, which is the same bug
 * mirrored. An identifier also loads a value, including a stored void value. */
static bool expr_leaves_value(CG *cg, ASTNode *node) {
    if (!node) return false;
    if (node->type == AST_MATCH || node->type == AST_IDENTIFIER || node->type == AST_HANDLE_EXPR || node->type == AST_EFFECT_HANDLER || node->type == AST_EFFECT_OP) return true;
    return check_expression(node, cg->env) != TYPE_VOID;
}

/* I materialize the sole void value when a no-result expression is stored. */
static void compile_stored_expr(CG *cg, ASTNode *node) {
    compile_expr(cg, node);
    if (!expr_leaves_value(cg, node)) emit_op(cg, OP_PUSH_VOID);
}

static bool stmt_falls_through(ASTNode *node) {
    if (!node) return true;
    switch (node->type) {
        case AST_RETURN:
        case AST_BREAK:
        case AST_CONTINUE:
            return false;
        case AST_BLOCK:
            for (int i = 0; i < node->as.block.count; i++) {
                if (!stmt_falls_through(node->as.block.statements[i])) return false;
            }
            return true;
        case AST_UNSAFE_BLOCK:
            for (int i = 0; i < node->as.unsafe_block.count; i++) {
                if (!stmt_falls_through(node->as.unsafe_block.statements[i])) return false;
            }
            return true;
        case AST_IF:
            return !node->as.if_stmt.else_branch
                || stmt_falls_through(node->as.if_stmt.then_branch)
                || stmt_falls_through(node->as.if_stmt.else_branch);
        case AST_MATCH: {
            /* Sound only because the synthesized no-arm-matched path now
             * terminates rather than producing void: with every arm body
             * terminating too, nothing reaches the code after the match. */
            for (int i = 0; i < node->as.match_expr.arm_count; i++) {
                if (stmt_falls_through(node->as.match_expr.arm_bodies[i]))
                    return true;
            }
            return node->as.match_expr.arm_count == 0;
        }
        default:
            return true;
    }
}

static int32_t direct_call_target(CG *cg, ASTNode *node) {
    if (!node) return -1;
    if (node->type == AST_CALL) {
        const char *name = node->as.call.name;
        if (!name || local_find(cg, name) >= 0 || upvalue_resolve(cg, name) >= 0)
            return -1;
        return fn_find(cg, name);
    }
    if (node->type == AST_MODULE_QUALIFIED_CALL) {
        char qualified[512];
        snprintf(qualified, sizeof(qualified), "%s.%s",
                 node->as.module_qualified_call.module_alias,
                 node->as.module_qualified_call.function_name);
        int32_t target = fn_find(cg, qualified);
        return target >= 0 ? target
            : fn_find(cg, node->as.module_qualified_call.function_name);
    }
    return -1;
}

static bool compile_tail_call(CG *cg, ASTNode *node) {
    int32_t target = direct_call_target(cg, node);
    if (target < 0) return false;
    const NvmFunctionEntry *caller = &cg->module->functions[cg->current_fn_idx];
    const NvmFunctionEntry *callee = &cg->module->functions[target];
    if (caller->result_count != callee->result_count
            || caller->result_tag != callee->result_tag)
        return false;

    ASTNode **args = node->type == AST_CALL ? node->as.call.args
        : node->as.module_qualified_call.args;
    int argc = node->type == AST_CALL ? node->as.call.arg_count
        : node->as.module_qualified_call.arg_count;
    if ((uint16_t)argc != callee->arity) return false;
    for (int i = 0; i < argc; i++) compile_expr(cg, args[i]);
    emit_op(cg, OP_TAIL_CALL, (uint32_t)target);
    return true;
}

static bool contains_par(ASTNode *node) {
    if (!node) return false;
    switch (node->type) {
        case AST_PAR_BLOCK: return true;
        case AST_BLOCK:
            for (int i = 0; i < node->as.block.count; ++i)
                if (contains_par(node->as.block.statements[i])) return true;
            return false;
        case AST_IF: return contains_par(node->as.if_stmt.then_branch) || contains_par(node->as.if_stmt.else_branch);
        case AST_WHILE: return contains_par(node->as.while_stmt.body);
        case AST_FOR: return contains_par(node->as.for_stmt.body);
        case AST_UNSAFE_BLOCK:
            for (int i = 0; i < node->as.unsafe_block.count; ++i)
                if (contains_par(node->as.unsafe_block.statements[i])) return true;
            return false;
        default: return false;
    }
}

static void compile_par_guards(CG *cg, ASTNode *body) {
    if (!contains_par(body)) return;
    uint8_t *tags = cg->module->function_param_types[cg->current_fn_idx];
    for (uint16_t i = 0; i < cg->param_count; ++i) {
        uint8_t tag = tags[i];
        if (tag != TAG_INT && tag != TAG_BOOL && tag != TAG_STRING && tag != TAG_FLOAT) continue;
        emit_op(cg, OP_LOAD_LOCAL, (int)i);
        emit_op(cg, OP_TYPE_CHECK, (int)tag);
        emit_op(cg, OP_ASSERT);
    }
}

static bool par_inputs(CG *cg, ASTNode *expr, ASTNode *block, bool *reads) {
    if (!expr) return false;
    switch (expr->type) {
        case AST_NUMBER: case AST_FLOAT: case AST_BOOL: case AST_STRING: return true;
        case AST_PREFIX_OP:
            for (int i = 0; i < expr->as.prefix_op.arg_count; ++i)
                if (!par_inputs(cg, expr->as.prefix_op.args[i], block, reads)) return false;
            return true;
        case AST_IDENTIFIER: {
            for (int i = 0; i < block->as.par_block.count; ++i)
                if (!strcmp(expr->as.identifier, block->as.par_block.bindings[i]->as.let.name)) return block->as.par_block.is_flow;
            int slot = local_find(cg, expr->as.identifier);
            if (slot < 0 || slot >= cg->param_count) return false;
            uint8_t tag = cg->module->function_param_types[cg->current_fn_idx][slot];
            if (tag != TAG_INT && tag != TAG_BOOL && tag != TAG_STRING && tag != TAG_FLOAT) return false;
            reads[slot] = true;
            return true;
        }
        case AST_CALL:
            if (expr->as.call.func_expr) return false;
            for (int i = 0; i < expr->as.call.arg_count; ++i)
                if (!par_inputs(cg, expr->as.call.args[i], block, reads)) return false;
            return true;
        case AST_MODULE_QUALIFIED_CALL:
            for (int i = 0; i < expr->as.module_qualified_call.arg_count; ++i)
                if (!par_inputs(cg, expr->as.module_qualified_call.args[i], block, reads)) return false;
            return true;
        default: return false;
    }
}

static void compile_par(CG *cg, ASTNode *block) {
    int count = block->as.par_block.count;
    int *order = passive_binding_order(block);
    if (!order) {
        cg_error(cg, block->line, "I require distinct immutable passive bindings and a valid dependency order");
        return;
    }
    uint64_t capacity = 5 + (uint64_t)count * (7ull + cg->param_count + (block->as.par_block.is_flow ? count : 0));
    if (capacity > UINT32_MAX / 4 || capacity > SIZE_MAX / sizeof(uint32_t)) {
        free(order);
        cg_error(cg, block->line, "I cannot represent this passive record"); return;
    }
    CgPassive *record = calloc(1, sizeof(*record));
    uint32_t *offsets = calloc((size_t)count, sizeof(*offsets));
    bool *reads = calloc(cg->param_count ? cg->param_count : 1, sizeof(bool));
    if (record) record->words = calloc((size_t)capacity, sizeof(uint32_t));
    if (!record || !reads || !offsets || !record->words) {
        if (record) free(record->words);
        free(record); free(reads); free(offsets); free(order);
        cg_error(cg, block->line, "I cannot allocate this passive record"); return;
    }
    uint32_t *words = record->words;
    words[0] = block->as.par_block.is_flow ? 2 : 1;
    words[1] = cg->current_fn_idx; words[2] = cg->code_size; words[4] = (uint32_t)count;
    CgPassive **place = &cg->passive;
    while (*place && ((*place)->words[1] < words[1] ||
           ((*place)->words[1] == words[1] && (*place)->words[2] < words[2]))) place = &(*place)->next;
    record->next = *place; *place = record;
    uint32_t at = 5;
    /* Metadata remains in source-ID order even when instructions do not. */
    for (int i = 0; i < count && !cg->had_error; ++i) {
        ASTNode *binding = block->as.par_block.bindings[i];
        offsets[i] = at;
        memset(reads, 0, cg->param_count * sizeof(bool));
        if (!par_inputs(cg, binding->as.let.value, block, reads)) {
            cg_error(cg, binding->line, "I require scalar passive expressions over dependencies and guarded parameters"); break;
        }
        uint32_t input = at + 7;
        if (block->as.par_block.is_flow) {
            for (int j = 0; j < count; ++j)
                if (passive_expression_reads_name(binding->as.let.value, block->as.par_block.bindings[j]->as.let.name))
                    words[input++] = (uint32_t)j;
        }
        words[at + 3] = input - at - 7;
        uint32_t first_read = input;
        for (uint16_t j = 0; j < cg->param_count; ++j) if (reads[j]) words[input++] = j;
        words[at + 4] = input - first_read;
        at = input;
    }
    for (int step = 0; step < count && !cg->had_error; ++step) {
        int i = order[step];
        ASTNode *binding = block->as.par_block.bindings[i];
        uint32_t node = offsets[i];
        words[node] = cg->code_size;
        compile_stmt(cg, binding);
        words[node + 1] = cg->code_size;
        words[node + 2] = (uint32_t)local_find(cg, binding->as.let.name);
    }
    words[3] = cg->code_size;
    record->word_count = at;
    free(reads); free(offsets); free(order);
}

static void publish_passive(CG *cg) {
    uint64_t count = 0, words = 2;
    for (CgPassive *record = cg->passive; record; record = record->next) {
        ++count; words += record->word_count;
    }
    if (count && !cg->had_error) {
        if (words > UINT32_MAX / 4) cg_error(cg, 0, "I cannot represent my passive records");
        else {
            cg->module->passive_data = malloc((size_t)words * 4);
            if (!cg->module->passive_data) cg_error(cg, 0, "I cannot allocate my passive records");
            else {
                cg->module->passive_size = (uint32_t)words * 4;
                uint32_t cursor = 0;
                for (unsigned i = 0; i < 4; ++i) cg->module->passive_data[cursor++] = (uint8_t)(2u >> (8*i));
                for (unsigned i = 0; i < 4; ++i) cg->module->passive_data[cursor++] = (uint8_t)(count >> (8*i));
                for (CgPassive *record = cg->passive; record; record = record->next) {
                    uint32_t base = cg->module->functions[record->words[1]].code_offset;
                    record->words[2] += base; record->words[3] += base;
                    uint32_t at = 5;
                    for (uint32_t i = 0; i < record->words[4]; ++i) {
                        record->words[at] += base; record->words[at + 1] += base;
                        at += 7 + record->words[at + 3] + record->words[at + 4];
                    }
                    for (uint32_t i = 0; i < record->word_count; ++i)
                        for (unsigned j = 0; j < 4; ++j)
                            cg->module->passive_data[cursor++] = (uint8_t)(record->words[i] >> (8*j));
                }
            }
        }
    }
    while (cg->passive) {
        CgPassive *next = cg->passive->next;
        free(cg->passive->words); free(cg->passive); cg->passive = next;
    }
}

static void compile_stmt(CG *cg, ASTNode *node) {
    if (!node || cg->had_error) return;

    /* Source locations live in the side table, never in executable code. */
    if (node->line > 0) {
        uint32_t off = cg->module->code_size + cg->code_size;
        nvm_add_debug_entry(cg->module, off, (uint32_t)node->line,
                            (uint32_t)(node->column > 0 ? node->column : 0));
    }

    switch (node->type) {
    case AST_LET: {
        if (node->as.let.value->type == AST_ARRAY_LITERAL &&
            node->as.let.value->as.array_literal.element_count == 0 &&
            node->as.let.element_type != TYPE_UNKNOWN) {
            node->as.let.value->as.array_literal.element_type = node->as.let.element_type;
        }
        if (node->as.let.var_type == TYPE_INT || node->as.let.var_type == TYPE_FLOAT)
            compile_numeric_expr(cg, node->as.let.value,
                check_expression(node->as.let.value, cg->env), node->as.let.var_type == TYPE_FLOAT);
        else compile_stored_expr(cg, node->as.let.value);
        uint16_t slot = local_add(cg, node->as.let.name, node->line);
        /* Track struct type for field access resolution */
        if (node->as.let.type_name) {
            cg->locals[cg->local_binding_count - 1].struct_type = node->as.let.type_name;
        }
        /* I re-establish this declaration's checked type. The shared checker
         * retains symbols across functions, and emitting a previous function's
         * parameter can otherwise override a same-named local's metadata. */
        env_define_var_with_type_info(cg->env, node->as.let.name, node->as.let.var_type,
                                      node->as.let.element_type, node->as.let.type_info,
                                      node->as.let.is_mut, create_void());
        Symbol *local_type = env_get_var(cg->env, node->as.let.name);
        if (local_type) {
            local_type->def_line = node->line;
            local_type->def_column = node->column;
            if (node->as.let.type_name)
                local_type->struct_type_name = strdup(node->as.let.type_name);
        }
        emit_op(cg, OP_STORE_LOCAL, (int)slot);
        local_name_begin(cg,slot,node->as.let.name,node->as.let.var_type,node->line);
        break;
    }

    case AST_SET: {
        if (node->as.set.field_name) {
            cg_error(cg, node->line, "I require reference IR before lowering field-place assignment");
            break;
        }
        int16_t slot = local_find(cg, node->as.set.name);
        if (slot >= 0) {
            compile_stored_expr(cg, node->as.set.value);
            emit_op(cg, OP_STORE_LOCAL, (int)slot);
        } else {
            int16_t gslot = global_find(cg, node->as.set.name);
            if (gslot >= 0) {
                compile_stored_expr(cg, node->as.set.value);
                emit_op(cg, OP_STORE_GLOBAL, (uint32_t)gslot);
            } else {
                /* Check upvalues for captured mutable variables */
                int16_t uv = upvalue_resolve(cg, node->as.set.name);
                if (uv >= 0) {
                    compile_stored_expr(cg, node->as.set.value);
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

        bool then_falls_through = stmt_falls_through(node->as.if_stmt.then_branch);
        compile_stmt(cg, node->as.if_stmt.then_branch);

        if (node->as.if_stmt.else_branch) {
            uint32_t je_instr = 0;
            uint32_t je_patch = 0;
            if (then_falls_through) {
                je_instr = cg->code_size;
                uint32_t je_off = emit_op(cg, OP_JMP, (int32_t)0);
                je_patch = je_off + 1;
            }

            patch_jump(cg, jf_patch, jf_instr, cg->code_size);
            compile_stmt(cg, node->as.if_stmt.else_branch);
            if (then_falls_through)
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
        loop->continue_index_slot = -1;
        loop->top_offset = cg->code_size;

        /* `while true` has no normal exit, so emitting a test-and-branch
         * invents one: control could apparently leave the loop and fall into
         * whatever follows, which for a value-returning function is its
         * implicit RET with nothing on the stack. That path would trap if it
         * were ever taken, and it made read_varint in std/binary -- an
         * infinite loop whose every path returns -- fail verification. Only
         * `break` leaves such a loop, and break patches jump to the end
         * independently. */
        bool unconditional = node->as.while_stmt.condition
            && node->as.while_stmt.condition->type == AST_BOOL
            && node->as.while_stmt.condition->as.bool_val;
        uint32_t jf_instr = 0, jf_patch = 0;
        if (!unconditional) {
            compile_expr(cg, node->as.while_stmt.condition);
            jf_instr = cg->code_size;
            jf_patch = emit_op(cg, OP_JMP_FALSE, (int32_t)0) + 1;
        }

        compile_stmt(cg, node->as.while_stmt.body);

        if (stmt_falls_through(node->as.while_stmt.body)) {
            uint32_t jmp_instr = cg->code_size;
            emit_op(cg, OP_JMP, (int32_t)0);
            patch_jump(cg, jmp_instr + 1, jmp_instr, loop->top_offset);
        }

        uint32_t loop_end = cg->code_size;
        /* Patch the conditional jump to after loop */
        if (!unconditional)
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

        uint16_t saved_binding_count = cg->local_binding_count;

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
        loop->continue_index_slot = idx_slot;
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

        /* Emitted outer lets must not hide the exact checked loop declaration. */
        if (!reestablish_checked_loop_binding(cg->env, node)) {
            cg_error(cg, node->line, "I require checked loop binding metadata");
            cg->loop_depth--;
            local_names_end(cg, saved_binding_count);
        cg->local_binding_count = saved_binding_count;
            break;
        }
        /* Compile body */
        compile_stmt(cg, node->as.for_stmt.body);

        if (stmt_falls_through(node->as.for_stmt.body)) {
            emit_op(cg, OP_LOAD_LOCAL, (int)idx_slot);
            emit_op(cg, OP_PUSH_I64, (int64_t)1);
            emit_op(cg, OP_I64_ADD);
            emit_op(cg, OP_STORE_LOCAL, (int)idx_slot);
            uint32_t jmp_instr = cg->code_size;
            emit_op(cg, OP_JMP, (int32_t)0);
            patch_jump(cg, jmp_instr + 1, jmp_instr, loop->top_offset);
        }

        uint32_t loop_end = cg->code_size;
        patch_jump(cg, jf_patch, jf_instr, loop_end);

        /* Patch breaks */
        for (int i = 0; i < loop->break_count; i++) {
            patch_jump(cg, loop->breaks[i].patch_offset,
                       loop->breaks[i].instr_offset, loop_end);
        }

        cg->loop_depth--;
        local_names_end(cg, saved_binding_count);
        cg->local_binding_count = saved_binding_count;
        break;
    }

    case AST_BLOCK: {
        uint16_t saved_binding_count = cg->local_binding_count;
        int symbol_start = cg->env->symbol_count;
        for (int i = 0; i < node->as.block.count; i++) {
            compile_stmt(cg, node->as.block.statements[i]);
            if (!stmt_falls_through(node->as.block.statements[i])) break;
        }
        local_names_end(cg, saved_binding_count);
        cg->local_binding_count = saved_binding_count;
        for (int i = symbol_start; i < cg->env->symbol_count; i++) {
            Symbol *symbol = &cg->env->symbols[i];
            if (!symbol->scope_end_line) {
                symbol->scope_end_line = node->scope_end_line;
                symbol->scope_end_column = node->scope_end_column;
            }
        }
        break;
    }

    case AST_RETURN: {
        ASTNode *value = node->as.return_stmt.value;
        if (value && value->type == AST_ARRAY_LITERAL &&
            value->as.array_literal.element_count == 0 &&
            cg->module->functions[cg->current_fn_idx].result_tag == TAG_ARRAY &&
            cg->current_return_element_type != TYPE_UNKNOWN) {
            value->as.array_literal.element_type = cg->current_return_element_type;
        }
        if (node->as.return_stmt.value
                && cg->effect_depth == 0 && compile_tail_call(cg, node->as.return_stmt.value)) {
            break;
        }
        if (node->as.return_stmt.value) {
            compile_expr(cg, node->as.return_stmt.value);
            if (cg->module->functions[cg->current_fn_idx].result_count == 0 &&
                expr_leaves_value(cg, node->as.return_stmt.value))
                emit_op(cg, OP_POP);
        }
        emit_op(cg, OP_RET);
        break;
    }

    case AST_BREAK: {
        if (cg->loop_depth <= cg->handler_loop_floor) {
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
        if (cg->loop_depth <= cg->handler_loop_floor) {
            cg_error(cg, node->line, "continue outside loop");
            break;
        }
        LoopCtx *loop = &cg->loops[cg->loop_depth - 1];
        /* I advance the innermost for index even when its body cannot fall
         * through to the normal increment. While loops have no index. */
        if (loop->continue_index_slot >= 0) {
            emit_op(cg, OP_LOAD_LOCAL, loop->continue_index_slot);
            emit_op(cg, OP_PUSH_I64, (int64_t)1);
            emit_op(cg, OP_I64_ADD);
            emit_op(cg, OP_STORE_LOCAL, loop->continue_index_slot);
        }
        uint32_t jmp_instr = cg->code_size;
        emit_op(cg, OP_JMP, (int32_t)0);
        patch_jump(cg, jmp_instr + 1, jmp_instr, loop->top_offset);
        break;
    }

    case AST_PRINT: {
        compile_expr(cg, node->as.print.expr);
        emit_op(cg, node->as.print.is_println ? OP_PRINTLN : OP_PRINT);
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
    case AST_EFFECT_HANDLER:
    case AST_HANDLE_EXPR:
    case AST_EFFECT_OP:
    case AST_MATCH:
    case AST_TRY_OP:
    case AST_AWAIT:
        compile_expr(cg, node);
        if (expr_leaves_value(cg, node))
            emit_op(cg, OP_POP);
        break;

    case AST_CALL:
    case AST_MODULE_QUALIFIED_CALL: {
        /* Function call as statement: discard only a declared result. */
        compile_expr(cg, node);
        int32_t called = -1;
        if (node->type == AST_CALL && node->as.call.name)
            called = fn_find(cg, node->as.call.name);
        if (node->type == AST_MODULE_QUALIFIED_CALL) {
            char qualified[512];
            snprintf(qualified, sizeof(qualified), "%s.%s",
                     node->as.module_qualified_call.module_alias,
                     node->as.module_qualified_call.function_name);
            called = fn_find(cg, qualified);
            if (called < 0) called = fn_find(cg, node->as.module_qualified_call.function_name);
        }
        Type result_type = check_expression(node, cg->env);
        if ((called >= 0 && cg->module->functions[called].result_count != 0)
                || (called < 0 && result_type != TYPE_VOID))
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
        /* Nested function definition. Kept in its own function so that its
         * large saved-state buffers stay out of compile_stmt's frame. */
        compile_nested_function(cg, node);
        break;

    case AST_UNSAFE_BLOCK: {
        uint16_t saved_binding_count = cg->local_binding_count;
        int symbol_start = cg->env->symbol_count;
        for (int i = 0; i < node->as.unsafe_block.count; i++) {
            compile_stmt(cg, node->as.unsafe_block.statements[i]);
            if (!stmt_falls_through(node->as.unsafe_block.statements[i])) break;
        }
        local_names_end(cg, saved_binding_count);
        cg->local_binding_count = saved_binding_count;
        for (int i = symbol_start; i < cg->env->symbol_count; i++) {
            Symbol *symbol = &cg->env->symbols[i];
            if (!symbol->scope_end_line) {
                symbol->scope_end_line = node->scope_end_line;
                symbol->scope_end_column = node->scope_end_column;
            }
        }
        break;
    }

    case AST_PAR_BLOCK:
        compile_par(cg, node);
        break;

    case AST_PAR_LET: {
        /* par-let: bindings evaluated sequentially, then body (result discarded as stmt) */
        for (int i = 0; i < node->as.par_let.count; i++) {
            compile_expr(cg, node->as.par_let.values[i]);
            uint16_t slot = local_add(cg, node->as.par_let.names[i], node->line);
            emit_op(cg, OP_STORE_LOCAL, (int)slot);
        }
        compile_expr(cg, node->as.par_let.body);
        if (expr_leaves_value(cg, node->as.par_let.body))
            emit_op(cg, OP_POP);
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
    int32_t fn_idx = fn_find_body(cg, fn_node->as.function.body);
    if (fn_idx < 0) {
        cg_error(cg, fn_node->line, "function '%s' not registered", name);
        return;
    }

    char *saved_module = cg->env->current_module;
    for (int i = 0; i < cg->env->function_count; i++) {
        if (cg->env->functions[i].body == fn_node->as.function.body) {
            cg->env->current_module = cg->env->functions[i].module_name;
            break;
        }
    }

    if (!record_function_parameters(cg, fn_node, (uint32_t)fn_idx)) {
        cg->env->current_module = saved_module;
        return;
    }

    cg->names_enabled = true;
    /* Reset per-function state */
    cg->code_size = 0;
    cg->local_count = 0;
    cg->local_binding_count = 0;
    cg->param_count = (uint16_t)fn_node->as.function.param_count;
    cg->loop_depth = 0;
    cg->effect_depth = 0;
    cg->handler_loop_floor = 0;
    cg->upvalue_count = 0;
    cg->current_fn_idx = (uint32_t)fn_idx;
    cg->current_return_element_type = fn_node->as.function.return_element_type;

    /* Parameters become the first locals */
    for (int i = 0; i < fn_node->as.function.param_count; i++) {
        uint16_t slot=local_add(cg, fn_node->as.function.params[i].name, fn_node->line);
        local_name_begin(cg,slot,fn_node->as.function.params[i].name,
                         fn_node->as.function.params[i].type,fn_node->line);
        /* Track struct type for field access resolution */
        if (fn_node->as.function.params[i].struct_type_name) {
            cg->locals[cg->local_binding_count - 1].struct_type =
                fn_node->as.function.params[i].struct_type_name;
        }

        /* Put the parameter's declared type where check_expression can find
         * it. Operator lowering chooses the integer or the float opcode from
         * that type, so a parameter the environment cannot resolve silently
         * becomes an integer -- which is how a transitively imported
         * `lerp(a: float, b: float, t: float)` compiled to I64_SUB and
         * I64_ADD and would have trapped with "requires two integers" if it
         * were ever called.
         *
         * Stamped with the file and line of this function, so it is visible
         * exactly where it should be: within this file, at or after this
         * point. Two modules may both have a parameter named `a` without
         * either seeing the other's. */
        bind_parameter_type(cg, &fn_node->as.function.params[i], fn_node->line);
    }

    /* Compile function body */
    ASTNode *body = fn_node->as.function.body;
    compile_par_guards(cg, body);
    if (body) {
        if (body->type == AST_BLOCK) {
            for (int i = 0; i < body->as.block.count; i++) {
                compile_stmt(cg, body->as.block.statements[i]);
                if (!stmt_falls_through(body->as.block.statements[i])) break;
            }
        } else {
            /* Single expression body - treat as return expr */
            compile_expr(cg, body);
            emit_op(cg, OP_RET);
        }
    }

    /* Ensure function always returns (implicit return void) */
    if (!body || stmt_falls_through(body)) {
        emit_op(cg, OP_RET);
    }

    local_names_end(cg,0);
    cg->names_enabled = false;
    cg->env->current_module = saved_module;
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

/* Copy a struct definition out of an imported module's AST into the
 * environment. The environment owns and frees these strings, and the AST is
 * freed independently, so every field is duplicated -- handing over an AST
 * pointer here is a double free at teardown, which is how I first wrote it. */
static void register_imported_struct(Environment *env, ASTNode *item) {
    if (!env || !item || item->type != AST_STRUCT_DEF) return;
    StructDef sdef = {0};
    sdef.original_name = NULL;
    memset(&sdef, 0, sizeof(sdef));
    sdef.name = strdup(item->as.struct_def.name);
    sdef.field_count = item->as.struct_def.field_count;
    sdef.field_names = malloc(sizeof(char *) * (size_t)sdef.field_count);
    sdef.field_types = malloc(sizeof(Type) * (size_t)sdef.field_count);
    sdef.field_type_names = malloc(sizeof(char *) * (size_t)sdef.field_count);
    sdef.field_element_types = malloc(sizeof(Type) * (size_t)sdef.field_count);
    if (!sdef.field_names || !sdef.field_types || !sdef.field_type_names
            || !sdef.field_element_types) {
        free(sdef.name); free(sdef.field_names); free(sdef.field_types);
        free(sdef.field_type_names); free(sdef.field_element_types);
        return;
    }
    for (int j = 0; j < sdef.field_count; j++) {
        sdef.field_names[j] = strdup(item->as.struct_def.field_names[j]);
        sdef.field_types[j] = item->as.struct_def.field_types[j];
        sdef.field_type_names[j] =
            item->as.struct_def.field_type_names && item->as.struct_def.field_type_names[j]
              ? strdup(item->as.struct_def.field_type_names[j]) : NULL;
        sdef.field_element_types[j] =
            item->as.struct_def.field_element_types
              ? item->as.struct_def.field_element_types[j] : TYPE_UNKNOWN;
    }
    sdef.is_resource = item->as.struct_def.is_resource;
    sdef.is_extern = item->as.struct_def.is_extern;
    sdef.is_pub = item->as.struct_def.is_pub;
    sdef.module_name = NULL;
    env_define_struct(env, sdef);
}

static CodegenResult codegen_compile_internal(ASTNode *program, Environment *env,
                                              ModuleList *modules, const char *input_file,
                                              bool shadows, bool include_imports) {
    CodegenResult result = {0};

    if (!program || program->type != AST_PROGRAM) {
        result.ok = false;
        snprintf(result.error_msg, sizeof(result.error_msg), "expected AST_PROGRAM root node");
        return result;
    }

    /* I do not substitute by-value aggregates for an unimplemented borrow ABI. */
    for (int i = 0; i < env->function_count; ++i) {
        Function *function = &env->functions[i];
        for (int p = 0; p < function->param_count; ++p) {
            if (function->params && (function->params[p].type == TYPE_BORROW_SHARED ||
                                     function->params[p].type == TYPE_BORROW_MUT)) {
                snprintf(result.error_msg, sizeof(result.error_msg),
                         "I cannot lower borrowed parameters to NanoISA yet");
                return result;
            }
        }
    }

    CgLocalName *local_names=NULL;
    CG cg = {0};
    cg.local_names=&local_names;
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
        ASTNode *item = bytecode_declaration(program->as.program.items[i]);

        if (item->type == AST_FUNCTION && !item->as.function.is_extern && !item->as.function.is_anonymous) {
            const char *name = item->as.function.name;
            uint32_t name_idx = nvm_add_string(cg.module, name, (uint32_t)strlen(name));

            NvmFunctionEntry fn = {0};
            fn.name_idx = name_idx;
            fn.arity = (uint16_t)item->as.function.param_count;
            fn.result_tag = type_to_tag(item->as.function.return_type, item->as.function.return_struct_type_name, cg.env);
            fn.result_count = fn.result_tag == TAG_VOID ? 0 : 1;

            uint32_t idx = nvm_add_function(cg.module, &fn);

            if (cg.fn_count < MAX_FUNCTIONS) {
                cg.functions[cg.fn_count].name = (char *)name;
                cg.functions[cg.fn_count].fn_idx = idx;
                cg.functions[cg.fn_count].body = item->as.function.body;
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
            uint8_t ret_tag = type_to_tag(item->as.function.return_type, item->as.function.return_struct_type_name, cg.env);

            /* Build param type tags */
            uint8_t param_tags[16] = {0};
            for (int p = 0; p < pc && p < 16; p++) {
                param_tags[p] = type_to_tag(item->as.function.params[p].type, item->as.function.params[p].struct_type_name, cg.env);
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
                    ASTNode *mitem = bytecode_declaration(mod_ast->as.program.items[m]);

                    /* Register all non-extern functions as bytecode functions */
                    if (mitem->type == AST_FUNCTION && !mitem->as.function.is_extern && !mitem->as.function.is_anonymous) {
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

                        /* I share an entry only for the same source body, not
                         * for an unrelated module's same-named function. */
                        bool already = false;
                        uint32_t idx = 0;
                        for (int f = 0; f < cg.fn_count; f++) {
                            if (cg.functions[f].body == mitem->as.function.body) {
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
                            fn.result_tag = type_to_tag(mitem->as.function.return_type, mitem->as.function.return_struct_type_name, cg.env);
                            fn.result_count = fn.result_tag == TAG_VOID ? 0 : 1;
                            idx = nvm_add_function(cg.module, &fn);
                            if (cg.fn_count < MAX_FUNCTIONS) {
                                cg.functions[cg.fn_count].name = (char *)use_name;
                                cg.functions[cg.fn_count].fn_idx = idx;
                                cg.functions[cg.fn_count].body = mitem->as.function.body;
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
                                cg.functions[cg.fn_count].body = mitem->as.function.body;
                                cg.fn_count++;
                            }
                        }
                    }

                    /* Register module's extern function declarations */
                    if (mitem->type == AST_FUNCTION && mitem->as.function.is_extern) {
                        const char *ename = mitem->as.function.name;
                        if (extern_find(&cg, ename) < 0) {
                            uint16_t pc = (uint16_t)mitem->as.function.param_count;
                            uint8_t ret_tag = type_to_tag(mitem->as.function.return_type, mitem->as.function.return_struct_type_name, cg.env);
                            uint8_t param_tags[16] = {0};
                            for (int p = 0; p < pc && p < 16; p++) {
                                param_tags[p] = type_to_tag(mitem->as.function.params[p].type, mitem->as.function.params[p].struct_type_name, cg.env);
                            }
                            register_extern(&cg, ename, lookup_path ? lookup_path : "",
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
                            uint8_t ret_tag = type_to_tag(fn->return_type, fn->return_struct_type_name, cg.env);
                            uint8_t param_tags[16] = {0};
                            for (int p = 0; p < pc && p < 16; p++) {
                                param_tags[p] = type_to_tag(fn->params[p].type, fn->params[p].struct_type_name, cg.env);
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
                                uint8_t ret_tag = type_to_tag(fn->return_type, fn->return_struct_type_name, cg.env);
                                uint8_t param_tags[16] = {0};
                                for (int p = 0; p < pc && p < 16; p++) {
                                    param_tags[p] = type_to_tag(fn->params[p].type, fn->params[p].struct_type_name, cg.env);
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
                ASTNode *mitem = bytecode_declaration(mod_ast->as.program.items[m]);

                if (mitem->type == AST_FUNCTION && !mitem->as.function.is_extern && !mitem->as.function.is_anonymous) {
                    const char *fname = mitem->as.function.name;
                    if (fn_find_body(&cg, mitem->as.function.body) < 0 && cg.fn_count < MAX_FUNCTIONS) {
                        uint32_t ni = nvm_add_string(cg.module, fname, (uint32_t)strlen(fname));
                        NvmFunctionEntry fn = {0};
                        fn.name_idx = ni;
                        fn.arity = (uint16_t)mitem->as.function.param_count;
                        fn.result_tag = type_to_tag(mitem->as.function.return_type, mitem->as.function.return_struct_type_name, cg.env);
                        fn.result_count = fn.result_tag == TAG_VOID ? 0 : 1;
                        uint32_t idx = nvm_add_function(cg.module, &fn);
                        cg.functions[cg.fn_count].name = (char *)fname;
                        cg.functions[cg.fn_count].fn_idx = idx;
                        cg.functions[cg.fn_count].body = mitem->as.function.body;
                        cg.fn_count++;
                    }
                }

                if (mitem->type == AST_FUNCTION && mitem->as.function.is_extern) {
                    const char *ename = mitem->as.function.name;
                    if (extern_find(&cg, ename) < 0) {
                        uint16_t pc = (uint16_t)mitem->as.function.param_count;
                        uint8_t ret_tag = type_to_tag(mitem->as.function.return_type, mitem->as.function.return_struct_type_name, cg.env);
                        uint8_t param_tags[16] = {0};
                        for (int p = 0; p < pc && p < 16; p++) {
                            param_tags[p] = type_to_tag(mitem->as.function.params[p].type, mitem->as.function.params[p].struct_type_name, cg.env);
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
                    /* Also give the environment the definition, so a field
                     * access inside this module's own function bodies can be
                     * typed. Codegen's table above answers "which struct index"
                     * for AGG_GET; check_expression needs "what type is this
                     * field" to pick the integer or float operator, and it
                     * reads the environment. Without it, `a.x` on a float
                     * field lowered to I64_ADD -- the same failure as an
                     * untyped parameter, one level further in. */
                    if (!env_get_struct(env, mitem->as.struct_def.name)) {
                        register_imported_struct(env, mitem);
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

        /* Pass 1b-alias: a transitively-imported module may itself pull in
         * functions under a selective-import alias (e.g. process_manager does
         * `from std/env import get as env_get`). Pass 1 only wires aliases for
         * the *main program's* direct imports, so without this a module's call
         * to its own alias (env_get) fails codegen with "undefined function".
         * Register each such alias as a second name sharing the target's
         * fn_idx. */
        for (int mi = 0; mi < modules->count; mi++) {
            ASTNode *mod_ast = get_cached_module_ast(modules->module_paths[mi]);
            if (!mod_ast || mod_ast->type != AST_PROGRAM) continue;
            for (int m = 0; m < mod_ast->as.program.count; m++) {
                ASTNode *imp = mod_ast->as.program.items[m];
                if (imp->type != AST_IMPORT || !imp->as.import_stmt.is_selective) continue;
                for (int s = 0; s < imp->as.import_stmt.import_symbol_count; s++) {
                    const char *orig = imp->as.import_stmt.import_symbols[s];
                    const char *alias = imp->as.import_stmt.import_aliases[s];
                    if (!alias || strcmp(alias, orig) == 0) continue;
                    if (fn_find(&cg, alias) >= 0) continue;   /* already known */
                    int32_t target = fn_find(&cg, orig);
                    if (target < 0) continue;                 /* not a bytecode fn */
                    if (cg.fn_count >= MAX_FUNCTIONS) break;
                    cg.functions[cg.fn_count].name = (char *)alias;
                    cg.functions[cg.fn_count].fn_idx = (uint32_t)target;
                    cg.fn_count++;
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
        init_fn.result_tag = TAG_VOID;
        init_fn.result_count = 0;
        uint32_t init_idx = nvm_add_function(cg.module, &init_fn);

        cg.code_size = 0;
        cg.local_count = 0;
        cg.local_binding_count = 0;
        cg.loop_depth = 0;

        /* Initialize module globals first (they may be referenced by module functions) */
        if (modules) {
            for (int mi = 0; mi < modules->count; mi++) {
                ASTNode *mod_ast = get_cached_module_ast(modules->module_paths[mi]);
                if (!mod_ast || mod_ast->type != AST_PROGRAM) continue;
                for (int m = 0; m < mod_ast->as.program.count; m++) {
                    ASTNode *mitem = bytecode_declaration(mod_ast->as.program.items[m]);
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
            ASTNode *item = bytecode_declaration(program->as.program.items[i]);
            if (item->type == AST_LET) {
                compile_expr(&cg, item->as.let.value);
                int16_t gslot = global_find(&cg, item->as.let.name);
                if (gslot >= 0) {
                    emit_op(&cg, OP_STORE_GLOBAL, (uint32_t)gslot);
                }
            }
        }
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
    /* Symbol visibility is decided by source position, and a position only
     * means something inside the file it came from. Each pass below tells the
     * environment which file it is walking, so a lookup for an identifier in
     * an imported module cannot resolve to a symbol in the main program that
     * happens to sit at a lower line number. */
    const char *outer_file = env_current_file(env);
    env_set_current_file(env, input_file);
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = bytecode_declaration(program->as.program.items[i]);
        if (item->type == AST_FUNCTION && !item->as.function.is_extern && !item->as.function.is_anonymous) {
            compile_function(&cg, item);
            if (cg.had_error) break;
        }
    }

    /* ── Pass 2b: Compile imported module function bodies ──────── */
    if (modules && !cg.had_error) {
        for (int mi = 0; mi < modules->count; mi++) {
            ASTNode *mod_ast = get_cached_module_ast(modules->module_paths[mi]);
            if (!mod_ast || mod_ast->type != AST_PROGRAM) continue;

            env_set_current_file(env, modules->module_paths[mi]);
            for (int m = 0; m < mod_ast->as.program.count; m++) {
                ASTNode *mitem = bytecode_declaration(mod_ast->as.program.items[m]);
                if (mitem->type == AST_FUNCTION && !mitem->as.function.is_extern && !mitem->as.function.is_anonymous) {
                    compile_function(&cg, mitem);
                    if (cg.had_error) break;
                }
            }
            if (cg.had_error) break;
        }
    }
    env_set_current_file(env, outer_file);   /* leave the environment as found */

    if (shadows && !cg.had_error) {
        uint32_t shadow_functions[MAX_FUNCTIONS];
        int shadow_count = 0;
        char *outer_module = env->current_module;
        int imported_count = include_imports && modules ? modules->count : 0;
        for (int source = 0; source <= imported_count && !cg.had_error; source++) {
          bool imported = source < imported_count;
          const char *file = imported ? modules->module_paths[source] : input_file;
          ASTNode *selected = imported ? get_cached_module_ast(file) : program;
          char *owner = imported ? module_program_name(selected, file) : NULL;
          if (!selected || (imported && !owner)) {
              free(owner);
              cg_error(&cg, 0, "I cannot load a selected shadow module: %s", file);
              break;
          }
          env_set_current_file(env, file);
          env->current_module = imported ? owner : outer_module;
          for (int i = 0; i < selected->as.program.count; i++) {
            ASTNode *shadow = selected->as.program.items[i];
            if (shadow->type != AST_SHADOW) continue;
            if (cg.fn_count >= MAX_FUNCTIONS) {
                cg_error(&cg, shadow->line, "I cannot register another shadow function");
                break;
            }
            char name[64];
            snprintf(name, sizeof name, "$shadow_%d_%.32s", shadow_count, shadow->as.shadow.function_name);
            uint32_t name_idx = nvm_add_string(cg.module, name, (uint32_t)strlen(name));
            NvmFunctionEntry entry = {0};
            entry.name_idx = name_idx;
            entry.result_tag = TAG_VOID;
            uint32_t index = nvm_add_function(cg.module, &entry);
            cg.functions[cg.fn_count].name = cg.module->strings[name_idx];
            cg.functions[cg.fn_count].body = shadow->as.shadow.body;
            cg.functions[cg.fn_count++].fn_idx = index;
            ASTNode function = {0};
            function.type = AST_FUNCTION;
            function.line = shadow->line;
            function.column = shadow->column;
            function.as.function.name = cg.module->strings[name_idx];
            function.as.function.return_type = TYPE_VOID;
            function.as.function.body = shadow->as.shadow.body;
            compile_function(&cg, &function);
            if (cg.had_error) break;
            shadow_functions[shadow_count++] = index;
          }
          env->current_module = outer_module;
          free(owner);
        }
        if (!cg.had_error) {
            NvmFunctionEntry entry = {0};
            entry.name_idx = nvm_add_string(cg.module, "$shadow_entry", 13);
            entry.result_tag = TAG_INT;
            entry.result_count = 1;
            uint32_t index = nvm_add_function(cg.module, &entry);
            cg.code_size = 0;
            cg.local_count = 0;
            cg.loop_depth = 0;
            cg.upvalue_count = 0;
            cg.current_fn_idx = index;
            for (int i = 0; i < shadow_count; i++) emit_op(&cg, OP_CALL, shadow_functions[i]);
            emit_op(&cg, OP_PUSH_I64, (int64_t)0);
            emit_op(&cg, OP_RET);
            uint32_t offset = nvm_append_code(cg.module, cg.code, cg.code_size);
            cg.module->functions[index].code_offset = offset;
            cg.module->functions[index].code_length = cg.code_size;
            main_fn_idx = (int)index;
        }
        env_set_current_file(env, outer_file);
    }

    /* For shadow-only programs (no main), generate a synthetic main that returns 0 */
    if (main_fn_idx < 0 && !cg.had_error) {
        NvmFunctionEntry syn_fn = {0};
        uint32_t syn_name = nvm_add_string(cg.module, "main", 4);
        syn_fn.name_idx = syn_name;
        syn_fn.arity = 0;
        syn_fn.result_tag = TAG_INT;
        syn_fn.result_count = 1;
        uint32_t syn_idx = nvm_add_function(cg.module, &syn_fn);

        cg.code_size = 0;
        cg.local_count = 0;
        cg.local_binding_count = 0;
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

    /* Publish type definition counts so the verifier can validate def_idx operands */
    cg.module->struct_count = (uint32_t)cg.struct_count;
    cg.module->enum_count   = (uint32_t)cg.enum_count;
    cg.module->union_count  = (uint32_t)cg.union_count;

    /* Set entry point and flags */
    if (main_fn_idx >= 0) {
        cg.module->header.flags = NVM_FLAG_HAS_MAIN;
        cg.module->header.entry_point = (uint32_t)main_fn_idx;
    }
    if (cg.extern_count > 0) {
        cg.module->header.flags |= NVM_FLAG_NEEDS_EXTERN;
    }
    /* Always emit debug info so the VM can produce source-mapped traces */
    if (cg.module->debug_count > 0) {
        cg.module->header.flags |= NVM_FLAG_DEBUG_INFO;
    }
    /* Store source file path in string pool for stack traces */
    if (input_file && input_file[0]) {
        cg.module->source_file_idx = nvm_add_string(cg.module, input_file,
                                                     (uint32_t)strlen(input_file));
    }

    publish_local_names(&cg);
    publish_passive(&cg);
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

CodegenResult codegen_compile(ASTNode *program, Environment *env,
                              ModuleList *modules, const char *input_file) {
    return codegen_compile_internal(program, env, modules, input_file, false, false);
}

CodegenResult codegen_compile_shadows(ASTNode *program, Environment *env,
                                      ModuleList *modules, const char *input_file) {
    return codegen_compile_shadow_scope(program, env, modules, input_file, false);
}

CodegenResult codegen_compile_shadow_scope(ASTNode *program, Environment *env,
                                          ModuleList *modules, const char *input_file,
                                          bool include_imports) {
    return codegen_compile_internal(program, env, modules, input_file, true, include_imports);
}
