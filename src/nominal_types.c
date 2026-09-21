/* I retain source names while binding module-owned record identities. */
#include "nanolang.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static const char *nominal_name(ASTNode *program, Environment *env, const char *name) {
    if (!name) return NULL;
    for (int i = 0; i < program->as.program.count; ++i) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_STRUCT_DEF &&
            (!strcmp(item->as.struct_def.name, name) ||
             (item->as.struct_def.original_name && !strcmp(item->as.struct_def.original_name, name))))
            return item->as.struct_def.name;
    }
    OpaqueTypeDef *opaque = env_get_opaque_type(env, name);
    if (opaque) return opaque->identity;
    StructDef *record = env_get_struct(env, name);
    return record ? record->name : name;
}

static bool nominal_slot(ASTNode *program, Environment *env, char **slot) {
    if (!slot || !*slot) return true;
    if (!env_reserve_opaque_symbol_prefix(env, *slot)) return false;
    const char *name = nominal_name(program, env, *slot);
    if (env->opaque_resolution_failed) return false;
    if (name == *slot && strchr(name, '.') && !env_get_enum(env, name) && !env_get_union(env, name)) {
        fprintf(stderr, "I cannot resolve a qualified type in this source: %s\n", name);
        return false;
    }
    if (!strcmp(name, *slot)) return true;
    char *bound = strdup(name);
    if (!bound) return false;
    free(*slot);
    *slot = bound;
    return true;
}

static bool nominal_scoped_slot(ASTNode *program, Environment *env, char **slot, char **formals, int count) {
    for (int i = 0; slot && *slot && i < count; ++i)
        if (!strcmp(*slot, formals[i])) return true;
    return nominal_slot(program, env, slot);
}
/* I resolve parser placeholders from declarations, without changing identity. */
static Type nominal_union_kind(ASTNode *program, Environment *env, Type type,
                               const char *name, char **formals, int count) {
    if (type != TYPE_STRUCT || !name) return type;
    for (int i = 0; i < count; ++i)
        if (!strcmp(name, formals[i])) return type;
    for (int i = 0; i < program->as.program.count; ++i) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_UNION_DEF && !strcmp(item->as.union_def.name, name))
            return TYPE_UNION;
    }
    return env_get_union(env, name) ? TYPE_UNION : type;
}
static bool nominal_scoped_signature(ASTNode *, Environment *, FunctionSignature *, char **, int);
static bool nominal_scoped_info(ASTNode *program, Environment *env, TypeInfo *info, char **formals, int count) {
    if (!info) return true;
    info->base_type = nominal_union_kind(program, env, info->base_type, info->generic_name, formals, count);
    if (!nominal_scoped_slot(program, env, &info->generic_name, formals, count) ||
        !nominal_scoped_slot(program, env, &info->opaque_type_name, formals, count) ||
        !nominal_scoped_info(program, env, info->element_type, formals, count) ||
        !nominal_scoped_signature(program, env, info->fn_sig, formals, count)) return false;
    for (int i = 0; i < info->type_param_count; ++i)
        if (info->type_params && !nominal_scoped_info(program, env, info->type_params[i], formals, count)) return false;
    for (int i = 0; i < info->tuple_element_count; ++i)
        if (info->tuple_type_names && !nominal_scoped_slot(program, env, &info->tuple_type_names[i], formals, count)) return false;
    for (int i = 0; i < info->row_field_count; ++i)
        if (info->row_field_type_names && !nominal_scoped_slot(program, env, &info->row_field_type_names[i], formals, count)) return false;
    return true;
}
static bool nominal_scoped_signature(ASTNode *program, Environment *env, FunctionSignature *signature, char **formals, int count) {
    if (!signature) return true;
    for (int i = 0; i < signature->param_count; ++i) {
        if (signature->param_struct_names && !nominal_scoped_slot(program, env, &signature->param_struct_names[i], formals, count)) return false;
        if (signature->param_type_info && !nominal_scoped_info(program, env, signature->param_type_info[i], formals, count)) return false;
        signature->param_types[i] = nominal_union_kind(program, env, signature->param_types[i],
            signature->param_struct_names ? signature->param_struct_names[i] : NULL, formals, count);
    }
    if (!nominal_scoped_slot(program, env, &signature->return_struct_name, formals, count) ||
        !nominal_scoped_info(program, env, signature->return_type_info, formals, count) ||
        !nominal_scoped_signature(program, env, signature->return_fn_sig, formals, count)) return false;
    signature->return_type = nominal_union_kind(program, env, signature->return_type,
                                               signature->return_struct_name, formals, count);
    return true;
}
static bool nominal_info(ASTNode *program, Environment *env, TypeInfo *info) {
    return nominal_scoped_info(program, env, info, NULL, 0);
}
static bool nominal_signature(ASTNode *program, Environment *env, FunctionSignature *signature) {
    return nominal_scoped_signature(program, env, signature, NULL, 0);
}
static bool nominal_parameter(ASTNode *program, Environment *env, Parameter *parameter) {
    if (!env_reserve_opaque_symbol_prefix(env, parameter->name) ||
        !nominal_slot(program, env, &parameter->struct_type_name) ||
        !nominal_signature(program, env, parameter->fn_sig) ||
        !nominal_info(program, env, parameter->type_info)) return false;
    parameter->type = nominal_union_kind(program, env, parameter->type,
                                         parameter->struct_type_name, NULL, 0);
    if (parameter->type != TYPE_BORROW_SHARED && parameter->type != TYPE_BORROW_MUT) return true;
    TypeInfo *inner = parameter->type_info ? parameter->type_info->element_type : NULL;
    if (inner && inner->base_type == TYPE_STRUCT && !inner->type_param_count && inner->generic_name) {
        for (int i = 0; i < program->as.program.count; ++i) {
            ASTNode *record = program->as.program.items[i];
            if (record->type != AST_STRUCT_DEF || !record->as.struct_def.is_resource ||
                strcmp(record->as.struct_def.name, inner->generic_name)) continue;
            bool scalar = true;
            for (int f = 0; f < record->as.struct_def.field_count; ++f) {
                Type t = record->as.struct_def.field_types[f];
                scalar &= t == TYPE_INT || t == TYPE_FLOAT || t == TYPE_BOOL;
            }
            if (scalar) return true;
        }
    }
    fprintf(stderr, "I currently borrow only fixed resource records with numeric or boolean fields\n");
    return false;
}

static bool nominal_node(ASTNode *program, Environment *env, ASTNode *node) {
    if (!node) return true;
#define NAME(value) do { if (!env_reserve_opaque_symbol_prefix(env, (value))) return false; } while (0)
#define NAMES(values, count) do { for (int k = 0; (values) && k < (count); ++k) NAME((values)[k]); } while (0)
#define CHILD(value) do { if (!nominal_node(program, env, (value))) return false; } while (0)
#define SLOT(value) do { if (!nominal_slot(program, env, &(value))) return false; } while (0)
#define CHILDREN(values, count) do { for (int n = 0; n < (count); ++n) CHILD((values)[n]); } while (0)
    switch (node->type) {
        case AST_PROGRAM: CHILDREN(node->as.program.items, node->as.program.count); break;
        case AST_IDENTIFIER: NAME(node->as.identifier); break;
        case AST_OPAQUE_TYPE: NAME(node->as.opaque_type.name); break;
        case AST_ENUM_DEF:
            NAME(node->as.enum_def.name); NAMES(node->as.enum_def.variant_names, node->as.enum_def.variant_count); break;
        case AST_FUNCTION:
            NAME(node->as.function.name);
            for (int i = 0; i < node->as.function.param_count; ++i)
                if (node->as.function.is_extern && (node->as.function.params[i].type == TYPE_BORROW_SHARED || node->as.function.params[i].type == TYPE_BORROW_MUT)) {
                    fprintf(stderr, "I require a checked body for a borrowed parameter\n"); return false;
                }
            for (int i = 0; i < node->as.function.param_count; ++i)
                if (!nominal_parameter(program, env, &node->as.function.params[i])) return false;
            SLOT(node->as.function.return_struct_type_name);
            node->as.function.return_type = nominal_union_kind(program, env,
                node->as.function.return_type, node->as.function.return_struct_type_name, NULL, 0);
            if (!nominal_signature(program, env, node->as.function.return_fn_sig) ||
                !nominal_info(program, env, node->as.function.return_type_info)) return false;
            CHILD(node->as.function.body); break;
        case AST_STRUCT_DEF:
            NAME(node->as.struct_def.name); NAME(node->as.struct_def.original_name);
            NAMES(node->as.struct_def.field_names, node->as.struct_def.field_count);
            for (int i = 0; i < node->as.struct_def.field_count; ++i) {
                if (node->as.struct_def.field_type_names) SLOT(node->as.struct_def.field_type_names[i]);
                if (node->as.struct_def.field_type_info && !nominal_info(program, env, node->as.struct_def.field_type_info[i])) return false;
            }
            break;
        case AST_UNION_DEF:
            NAME(node->as.union_def.name);
            NAMES(node->as.union_def.generic_params, node->as.union_def.generic_param_count);
            NAMES(node->as.union_def.variant_names, node->as.union_def.variant_count);
            for (int i = 0; i < node->as.union_def.variant_count; ++i)
                NAMES(node->as.union_def.variant_field_names[i], node->as.union_def.variant_field_counts[i]);
            for (int i = 0; i < node->as.union_def.variant_count; ++i)
                for (int j = 0; j < node->as.union_def.variant_field_counts[i]; ++j) {
                    if (node->as.union_def.variant_field_type_info && node->as.union_def.variant_field_type_info[i] &&
                        !nominal_scoped_info(program, env, node->as.union_def.variant_field_type_info[i][j],
                                             node->as.union_def.generic_params, node->as.union_def.generic_param_count)) return false;
                    if (node->as.union_def.variant_field_type_names && node->as.union_def.variant_field_type_names[i]) {
                        const char *name = node->as.union_def.variant_field_type_names[i][j];
                        bool formal = false;
                        for (int param = 0; name && param < node->as.union_def.generic_param_count; ++param)
                            if (!strcmp(name, node->as.union_def.generic_params[param])) formal = true;
                        if (!formal) SLOT(node->as.union_def.variant_field_type_names[i][j]);
                    }
                }
            break;
        case AST_STRUCT_LITERAL:
            NAMES(node->as.struct_literal.field_names, node->as.struct_literal.field_count);
            SLOT(node->as.struct_literal.struct_name);
            CHILDREN(node->as.struct_literal.field_values, node->as.struct_literal.field_count);
            CHILD(node->as.struct_literal.spread_source); break;
        case AST_UNION_CONSTRUCT:
            NAME(node->as.union_construct.union_name); NAME(node->as.union_construct.variant_name);
            NAMES(node->as.union_construct.field_names, node->as.union_construct.field_count);
            if (!nominal_info(program, env, node->as.union_construct.type_info)) return false;
            CHILDREN(node->as.union_construct.field_values, node->as.union_construct.field_count); break;
        case AST_LET:
            NAME(node->as.let.name); NAMES(node->as.let.destructure_names, node->as.let.destructure_count);
            SLOT(node->as.let.type_name);
            if (!nominal_signature(program, env, node->as.let.fn_sig) || !nominal_info(program, env, node->as.let.type_info)) return false;
            CHILD(node->as.let.value); break;
        case AST_SET: NAME(node->as.set.name); NAME(node->as.set.field_name); CHILD(node->as.set.value); break;
        case AST_BLOCK: CHILDREN(node->as.block.statements, node->as.block.count); break;
        case AST_UNSAFE_BLOCK: CHILDREN(node->as.unsafe_block.statements, node->as.unsafe_block.count); break;
        case AST_SHADOW: NAME(node->as.shadow.function_name); CHILD(node->as.shadow.body); break;
        case AST_RETURN: CHILD(node->as.return_stmt.value); break;
        case AST_IF:
            CHILD(node->as.if_stmt.condition); CHILD(node->as.if_stmt.then_branch); CHILD(node->as.if_stmt.else_branch); break;
        case AST_COND:
            CHILDREN(node->as.cond_expr.conditions, node->as.cond_expr.clause_count);
            CHILDREN(node->as.cond_expr.values, node->as.cond_expr.clause_count); CHILD(node->as.cond_expr.else_value); break;
        case AST_WHILE: CHILD(node->as.while_stmt.condition); CHILD(node->as.while_stmt.body); break;
        case AST_FOR: NAME(node->as.for_stmt.var_name); CHILD(node->as.for_stmt.range_expr); CHILD(node->as.for_stmt.body); break;
        case AST_CALL:
            NAME(node->as.call.name);
            SLOT(node->as.call.return_struct_type_name); CHILD(node->as.call.func_expr);
            CHILDREN(node->as.call.args, node->as.call.arg_count); break;
        case AST_MODULE_QUALIFIED_CALL:
            NAME(node->as.module_qualified_call.module_alias); NAME(node->as.module_qualified_call.function_name);
            SLOT(node->as.module_qualified_call.return_struct_type_name);
            CHILDREN(node->as.module_qualified_call.args, node->as.module_qualified_call.arg_count); break;
        case AST_PREFIX_OP: CHILDREN(node->as.prefix_op.args, node->as.prefix_op.arg_count); break;
        case AST_ARRAY_LITERAL: CHILDREN(node->as.array_literal.elements, node->as.array_literal.element_count); break;
        case AST_FIELD_ACCESS: NAME(node->as.field_access.field_name); CHILD(node->as.field_access.object); break;
        case AST_TUPLE_LITERAL: CHILDREN(node->as.tuple_literal.elements, node->as.tuple_literal.element_count); break;
        case AST_TUPLE_INDEX: CHILD(node->as.tuple_index.tuple); break;
        case AST_ASSERT: CHILD(node->as.assert.condition); break;
        case AST_PRINT: CHILD(node->as.print.expr); break;
        case AST_MATCH:
            NAMES(node->as.match_expr.pattern_variants, node->as.match_expr.arm_count);
            NAMES(node->as.match_expr.pattern_bindings, node->as.match_expr.arm_count);
            CHILD(node->as.match_expr.expr); CHILDREN(node->as.match_expr.arm_bodies, node->as.match_expr.arm_count);
            if (node->as.match_expr.guard_exprs) CHILDREN(node->as.match_expr.guard_exprs, node->as.match_expr.arm_count);
            break;
        case AST_TRY_OP: CHILD(node->as.try_op.operand); break;
        case AST_PAR_BLOCK: CHILDREN(node->as.par_block.bindings, node->as.par_block.count); break;
        case AST_PAR_LET: NAMES(node->as.par_let.names, node->as.par_let.count); CHILDREN(node->as.par_let.values, node->as.par_let.count); CHILD(node->as.par_let.body); break;
        case AST_EFFECT_DECL:
            NAME(node->as.effect_decl.effect_name); NAMES(node->as.effect_decl.op_names, node->as.effect_decl.op_count);
            for (int i = 0; i < node->as.effect_decl.op_count; ++i) {
                if (node->as.effect_decl.op_return_type_names) SLOT(node->as.effect_decl.op_return_type_names[i]);
                if (node->as.effect_decl.op_param_type_names) SLOT(node->as.effect_decl.op_param_type_names[i]);
                if (node->as.effect_decl.op_params && node->as.effect_decl.op_params[i])
                    for (int j = 0; j < node->as.effect_decl.op_param_counts[i]; ++j)
                        if (!nominal_parameter(program, env, &node->as.effect_decl.op_params[i][j])) return false;
            }
            break;
        case AST_EFFECT_OP: NAME(node->as.effect_op.effect_name); NAME(node->as.effect_op.op_name); CHILDREN(node->as.effect_op.args, node->as.effect_op.arg_count); break;
        case AST_HANDLE_EXPR:
            NAME(node->as.handle_expr.effect_name);
            NAMES(node->as.handle_expr.handler_op_names, node->as.handle_expr.handler_count);
            for (int i = 0; node->as.handle_expr.handler_param_names && i < node->as.handle_expr.handler_count; ++i)
                NAMES(node->as.handle_expr.handler_param_names[i], node->as.handle_expr.handler_param_counts[i]);
            CHILD(node->as.handle_expr.body); CHILDREN(node->as.handle_expr.handler_bodies, node->as.handle_expr.handler_count); break;
        case AST_EFFECT_HANDLER:
            NAME(node->as.effect_handler.effect_name);
            NAMES(node->as.effect_handler.handler_op_names, node->as.effect_handler.handler_count);
            NAMES(node->as.effect_handler.handler_param_names, node->as.effect_handler.handler_count);
            CHILD(node->as.effect_handler.body); CHILDREN(node->as.effect_handler.handler_bodies, node->as.effect_handler.handler_count); break;
        case AST_ASYNC_FN: CHILD(node->as.async_fn.function); break;
        case AST_AWAIT: CHILD(node->as.await_expr.expr); break;
        default: break;
    }
#undef NAME
#undef NAMES
#undef CHILD
#undef SLOT
#undef CHILDREN
    return true;
}

bool bind_nominal_records(ASTNode *program, Environment *env) {
    if (!env_prepare_opaque_types(env, program)) return false;
    const char *owner = env->current_module;
    for (int i = 0; i < program->as.program.count; ++i)
        if (program->as.program.items[i]->type == AST_MODULE_DECL) owner = program->as.program.items[i]->as.module_decl.name;
    {
        const char hex[] = "0123456789abcdef";
        size_t length = owner ? strlen(owner) : 0;
        for (int i = 0; i < program->as.program.count; ++i) {
            ASTNode *item = program->as.program.items[i];
            if (item->type != AST_STRUCT_DEF || item->as.struct_def.original_name) continue;
            /* I give declared one-letter records an identity distinct from
             * free generic variables before the native emitter sees them. */
            const char *source_name = item->as.struct_def.name;
            bool resembles_formal = source_name[0] >= 'A' && source_name[0] <= 'Z' &&
                source_name[1] == '\0' && !item->as.struct_def.is_extern;
            StructDef *existing = env_get_struct(env, source_name);
            if (!existing && !resembles_formal) continue;
            bool same_owner = existing && ((!owner && !existing->module_name) ||
                (owner && existing->module_name && strcmp(owner, existing->module_name) == 0));
            if (same_owner) continue;
            if (item->as.struct_def.is_extern || (existing && existing->is_extern)) {
                fprintf(stderr, "I cannot bind colliding foreign record declarations without an ABI identity: %s\n", item->as.struct_def.name);
                return false;
            }
            size_t name_length = strlen(item->as.struct_def.name);
            if (length > (SIZE_MAX - name_length - 32) / 2) return false;
            char *bound = malloc(2 * length + name_length + 32);
            if (!bound) return false;
            strcpy(bound, owner ? "__nano_record_" : "__nano_root_record_");
            size_t offset = strlen(bound);
            for (size_t j = 0; j < length; ++j) {
                unsigned char c = (unsigned char)owner[j];
                bound[offset++] = hex[c >> 4]; bound[offset++] = hex[c & 15];
            }
            bound[offset++] = '_'; strcpy(bound + offset, item->as.struct_def.name);
            item->as.struct_def.original_name = item->as.struct_def.name;
            item->as.struct_def.name = bound;
        }
    }
    if (!nominal_node(program, env, program)) {
        fprintf(stderr, "I cannot allocate module-owned record identities\n");
        return false;
    }
    return true;
}
