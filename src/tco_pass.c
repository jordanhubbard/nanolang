/* I lower admitted checked self-tail returns to simultaneous updates and a
 * loop. Ordinary returns keep their meaning and their original result type. */
#include "tco_pass.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    ASTNode *program;
    ASTNode *function;
    char prefix[64];
    int *shadowed;
    bool collision;
    bool rename;
    int calls;
    int loop_calls;
} TCO;

static ASTNode *node_new(ASTNodeType type) {
    ASTNode *node = calloc(1, sizeof(*node));
    if (!node) { fprintf(stderr, "I ran out of memory in TCO.\n"); exit(1); }
    node->type = type;
    return node;
}

static char *copy_name(const char *name) {
    char *copy = strdup(name);
    if (!copy) { fprintf(stderr, "I ran out of memory in TCO.\n"); exit(1); }
    return copy;
}

static char *slot_name(TCO *ctx, int index, bool temporary) {
    char name[96];
    snprintf(name, sizeof(name), "%s%c%d", ctx->prefix, temporary ? 'a' : 'p', index);
    return copy_name(name);
}

static char *flag_name(TCO *ctx) {
    char name[96];
    snprintf(name, sizeof(name), "%sr", ctx->prefix);
    return copy_name(name);
}

static int parameter(TCO *ctx, const char *name) {
    if (!name) return -1;
    for (int i = 0; i < ctx->function->as.function.param_count; i++)
        if (!strcmp(name, ctx->function->as.function.params[i].name)) return i;
    return -1;
}

static void inspect_name(TCO *ctx, char **name, bool reference) {
    if (!*name) return;
    if (!strncmp(*name, ctx->prefix, strlen(ctx->prefix))) ctx->collision = true;
    int index = parameter(ctx, *name);
    if (ctx->rename && reference && index >= 0 && !ctx->shadowed[index]) {
        free(*name);
        *name = slot_name(ctx, index, false);
    }
}

static bool bind_name(TCO *ctx, char **name) {
    inspect_name(ctx, name, false);
    if (!*name) return true;
    if (!strcmp(*name, ctx->function->as.function.name)) return false;
    int index = parameter(ctx, *name);
    if (index >= 0) ctx->shadowed[index]++;
    return true;
}

static int *save_bindings(TCO *ctx) {
    int count = ctx->function->as.function.param_count;
    int *saved = calloc((size_t)(count ? count : 1), sizeof(*saved));
    if (!saved) { fprintf(stderr, "I ran out of memory in TCO.\n"); exit(1); }
    if (count) memcpy(saved, ctx->shadowed, (size_t)count * sizeof(*saved));
    return saved;
}

static void restore_bindings(TCO *ctx, int *saved) {
    int count = ctx->function->as.function.param_count;
    if (count) memcpy(ctx->shadowed, saved, (size_t)count * sizeof(*saved));
    free(saved);
}

static ASTNode *ident(char *name);
static bool visit(TCO *ctx, ASTNode *node);

static bool visit_scoped_nodes(TCO *ctx, ASTNode **nodes, int count) {
    int *saved = save_bindings(ctx);
    bool ok = true;
    for (int i = 0; i < count && ok; i++) ok = visit(ctx, nodes[i]);
    restore_bindings(ctx, saved);
    return ok;
}

/* I leave unsupported binding/ownership constructs unchanged. This preflight
 * completes before mutation; skipping optimization must preserve the program. */
static bool visit(TCO *ctx, ASTNode *node) {
    if (!node) return true;
    switch (node->type) {
    case AST_NUMBER: case AST_FLOAT: case AST_BOOL: case AST_STRING:
    case AST_BREAK: case AST_CONTINUE:
        return true;
    case AST_IDENTIFIER:
        inspect_name(ctx, &node->as.identifier, true); return true;
    case AST_CALL:
        if (node->as.call.func_expr) {
            if (!visit(ctx, node->as.call.func_expr)) return false;
        } else {
            int index = parameter(ctx, node->as.call.name);
            bool parameter_callee = index >= 0 && !ctx->shadowed[index];
            if (ctx->rename && parameter_callee) {
                /* A checked call through a function-valued parameter is a
                 * callable expression. Leaving my hidden local in `name`
                 * would send it through declared-function lookup instead. */
                free(node->as.call.name);
                node->as.call.name = NULL;
                node->as.call.func_expr = ident(slot_name(ctx, index, false));
            } else {
                inspect_name(ctx, &node->as.call.name, parameter_callee);
            }
        }
        for (int i = 0; i < node->as.call.arg_count; i++)
            if (!visit(ctx, node->as.call.args[i])) return false;
        return true;
    case AST_MODULE_QUALIFIED_CALL:
        for (int i = 0; i < node->as.module_qualified_call.arg_count; i++)
            if (!visit(ctx, node->as.module_qualified_call.args[i])) return false;
        return true;
    case AST_PREFIX_OP:
        for (int i = 0; i < node->as.prefix_op.arg_count; i++)
            if (!visit(ctx, node->as.prefix_op.args[i])) return false;
        return true;
    case AST_LET:
        if (node->as.let.is_destructure) return false;
        if (!visit(ctx, node->as.let.value)) return false;
        return bind_name(ctx, &node->as.let.name);
    case AST_SET:
        inspect_name(ctx, &node->as.set.name, true);
        return visit(ctx, node->as.set.value);
    case AST_RETURN: return visit(ctx, node->as.return_stmt.value);
    case AST_ASSERT: return visit(ctx, node->as.assert.condition);
    case AST_PRINT: return visit(ctx, node->as.print.expr);
    case AST_IF: {
        if (!visit(ctx, node->as.if_stmt.condition)) return false;
        int *saved = save_bindings(ctx);
        bool ok = visit(ctx, node->as.if_stmt.then_branch);
        restore_bindings(ctx, saved);
        if (!ok) return false;
        saved = save_bindings(ctx);
        ok = visit(ctx, node->as.if_stmt.else_branch);
        restore_bindings(ctx, saved);
        return ok;
    }
    case AST_COND:
        for (int i = 0; i < node->as.cond_expr.clause_count; i++) {
            if (!visit(ctx, node->as.cond_expr.conditions[i])) return false;
            int *saved = save_bindings(ctx);
            bool ok = visit(ctx, node->as.cond_expr.values[i]);
            restore_bindings(ctx, saved);
            if (!ok) return false;
        }
        return visit(ctx, node->as.cond_expr.else_value);
    case AST_WHILE:
        if (!visit(ctx, node->as.while_stmt.condition)) return false;
        return visit(ctx, node->as.while_stmt.body);
    case AST_FOR: {
        if (!visit(ctx, node->as.for_stmt.range_expr)) return false;
        int *saved = save_bindings(ctx);
        bool ok = bind_name(ctx, &node->as.for_stmt.var_name) &&
            visit(ctx, node->as.for_stmt.body);
        restore_bindings(ctx, saved);
        return ok;
    }
    case AST_BLOCK:
        return visit_scoped_nodes(ctx, node->as.block.statements,
                                  node->as.block.count);
    case AST_FUNCTION: {
        inspect_name(ctx, &node->as.function.name, false);
        if (!strcmp(node->as.function.name, ctx->function->as.function.name)) return false;
        int *saved = save_bindings(ctx);
        bool ok = bind_name(ctx, &node->as.function.name);
        for (int i = 0; i < node->as.function.param_count && ok; i++)
            ok = bind_name(ctx, &node->as.function.params[i].name);
        if (ok) ok = visit(ctx, node->as.function.body);
        restore_bindings(ctx, saved);
        if (!ok) return false;
        return bind_name(ctx, &node->as.function.name);
    }
    case AST_ARRAY_LITERAL:
        for (int i = 0; i < node->as.array_literal.element_count; i++)
            if (!visit(ctx, node->as.array_literal.elements[i])) return false;
        return true;
    case AST_STRUCT_LITERAL:
        if (!visit(ctx, node->as.struct_literal.spread_source)) return false;
        for (int i = 0; i < node->as.struct_literal.field_count; i++)
            if (!visit(ctx, node->as.struct_literal.field_values[i])) return false;
        return true;
    case AST_FIELD_ACCESS: return visit(ctx, node->as.field_access.object);
    case AST_UNION_CONSTRUCT:
        for (int i = 0; i < node->as.union_construct.field_count; i++)
            if (!visit(ctx, node->as.union_construct.field_values[i])) return false;
        return true;
    case AST_MATCH:
        if (!visit(ctx, node->as.match_expr.expr)) return false;
        for (int i = 0; i < node->as.match_expr.arm_count; i++) {
            int *saved = save_bindings(ctx);
            bool ok = true;
            if (node->as.match_expr.pattern_bindings &&
                node->as.match_expr.pattern_bindings[i])
                ok = bind_name(ctx, &node->as.match_expr.pattern_bindings[i]);
            if (ok && node->as.match_expr.guard_exprs)
                ok = visit(ctx, node->as.match_expr.guard_exprs[i]);
            if (ok) ok = visit(ctx, node->as.match_expr.arm_bodies[i]);
            restore_bindings(ctx, saved);
            if (!ok) return false;
        }
        return true;
    case AST_TUPLE_LITERAL:
        for (int i = 0; i < node->as.tuple_literal.element_count; i++)
            if (!visit(ctx, node->as.tuple_literal.elements[i])) return false;
        return true;
    case AST_TUPLE_INDEX: return visit(ctx, node->as.tuple_index.tuple);
    case AST_QUALIFIED_NAME: return true;
    case AST_UNSAFE_BLOCK:
        return visit_scoped_nodes(ctx, node->as.unsafe_block.statements,
                                  node->as.unsafe_block.count);
    case AST_TRY_OP: return visit(ctx, node->as.try_op.operand);
    case AST_PAR_LET:
        for (int i = 0; i < node->as.par_let.count; i++)
            if (!visit(ctx, node->as.par_let.values[i])) return false;
        {
            int *saved = save_bindings(ctx);
            bool ok = true;
            for (int i = 0; i < node->as.par_let.count && ok; i++)
                ok = bind_name(ctx, &node->as.par_let.names[i]);
            if (ok) ok = visit(ctx, node->as.par_let.body);
            restore_bindings(ctx, saved);
            return ok;
        }
    case AST_PAR_BLOCK:
        /* Parallel bindings do not have sequential lexical publication. */
        for (int i = 0; i < node->as.par_block.count; i++) {
            ASTNode *binding = node->as.par_block.bindings[i];
            if (!binding || binding->type != AST_LET ||
                !visit(ctx, binding->as.let.value)) return false;
        }
        return true;
    default: return false;
    }
}

static ASTNode *ident(char *name) {
    ASTNode *node = node_new(AST_IDENTIFIER);
    node->as.identifier = name;
    return node;
}

static ASTNode *binding(char *name, Type type, ASTNode *value) {
    ASTNode *node = node_new(AST_LET);
    node->as.let.name = name;
    node->as.let.var_type = type;
    node->as.let.is_mut = true;
    node->as.let.value = value;
    return node;
}

static ASTNode *parameter_binding(TCO *ctx, int index, bool temporary,
                                  ASTNode *value) {
    Parameter *parameter_info = &ctx->function->as.function.params[index];
    ASTNode *node = binding(slot_name(ctx, index, temporary),
                            parameter_info->type, value);
    node->as.let.type_name = parameter_info->struct_type_name
        ? copy_name(parameter_info->struct_type_name) : NULL;
    node->as.let.element_type = parameter_info->element_type;
    node->as.let.fn_sig = copy_function_signature(parameter_info->fn_sig);
    node->as.let.type_info = copy_payload_type_info(parameter_info->type_info);
    return node;
}

static ASTNode *block(int count) {
    ASTNode *node = node_new(AST_BLOCK);
    node->as.block.count = count;
    node->as.block.statements = calloc(count ? count : 1, sizeof(ASTNode *));
    if (!node->as.block.statements) { fprintf(stderr, "I ran out of memory in TCO.\n"); exit(1); }
    return node;
}

static ASTNode *control_if(TCO *ctx, ASTNodeType control) {
    ASTNode *conditional = node_new(AST_IF);
    conditional->as.if_stmt.condition = ident(flag_name(ctx));
    conditional->as.if_stmt.then_branch = block(1);
    conditional->as.if_stmt.then_branch->as.block.statements[0] = node_new(control);
    return conditional;
}

static bool rewrite(TCO *ctx, ASTNode *node, bool mutate, int loop_depth) {
    if (!node) return false;
    if (node->type == AST_BLOCK) {
        bool found = false;
        for (int i = 0; i < node->as.block.count; i++)
            found |= rewrite(ctx, node->as.block.statements[i], mutate, loop_depth);
        return found;
    } else if (node->type == AST_IF) {
        bool found = rewrite(ctx, node->as.if_stmt.then_branch, mutate, loop_depth);
        found |= rewrite(ctx, node->as.if_stmt.else_branch, mutate, loop_depth);
        return found;
    } else if (node->type == AST_COND) {
        bool found = false;
        for (int i = 0; i < node->as.cond_expr.clause_count; i++)
            found |= rewrite(ctx, node->as.cond_expr.values[i], mutate, loop_depth);
        found |= rewrite(ctx, node->as.cond_expr.else_value, mutate, loop_depth);
        return found;
    } else if (node->type == AST_WHILE || node->type == AST_FOR) {
        ASTNode *body = node->type == AST_WHILE ? node->as.while_stmt.body
                                                : node->as.for_stmt.body;
        bool found = rewrite(ctx, body, mutate, loop_depth + 1);
        if (found && mutate) {
            ASTNode *original = node_new(node->type);
            *original = *node;
            node->type = AST_BLOCK;
            node->as.block.count = 2;
            node->as.block.statements = calloc(2, sizeof(ASTNode *));
            if (!node->as.block.statements) {
                fprintf(stderr, "I ran out of memory in TCO.\n"); exit(1);
            }
            node->as.block.statements[0] = original;
            node->as.block.statements[1] = control_if(ctx,
                loop_depth > 0 ? AST_BREAK : AST_CONTINUE);
        }
        return found;
    } else if (node->type == AST_RETURN) {
        ASTNode *call = node->as.return_stmt.value;
        if (!call || call->type != AST_CALL || call->as.call.func_expr ||
            !call->as.call.name || strcmp(call->as.call.name, ctx->function->as.function.name) ||
            call->as.call.arg_count != ctx->function->as.function.param_count) return false;
        ctx->calls++;
        if (loop_depth > 0) ctx->loop_calls++;
        if (!mutate) return true;
        int count = call->as.call.arg_count;
        ASTNode *replacement = block(count * 2 + (loop_depth > 0 ? 2 : 1));
        for (int i = 0; i < count; i++) {
            replacement->as.block.statements[i] = parameter_binding(ctx, i, true,
                call->as.call.args[i]);
            call->as.call.args[i] = NULL;
            ASTNode *set = node_new(AST_SET);
            set->as.set.name = slot_name(ctx, i, false);
            set->as.set.value = ident(slot_name(ctx, i, true));
            replacement->as.block.statements[count + i] = set;
        }
        int final = count * 2;
        if (loop_depth > 0) {
            ASTNode *pending = node_new(AST_SET);
            pending->as.set.name = flag_name(ctx);
            pending->as.set.value = node_new(AST_BOOL);
            pending->as.set.value->as.bool_val = true;
            replacement->as.block.statements[final++] = pending;
            replacement->as.block.statements[final] = node_new(AST_BREAK);
        } else {
            replacement->as.block.statements[final] = node_new(AST_CONTINUE);
        }
        free_ast(call);
        node->type = AST_BLOCK;
        node->as.block = replacement->as.block;
        free(replacement);
        return true;
    }
    return false;
}

static bool resource_struct(TCO *ctx, const char *name) {
    if (!ctx->program || ctx->program->type != AST_PROGRAM || !name) return false;
    for (int i = 0; i < ctx->program->as.program.count; ++i) {
        ASTNode *item = ctx->program->as.program.items[i];
        if (!item || item->type != AST_STRUCT_DEF || !item->as.struct_def.is_resource)
            continue;
        if ((item->as.struct_def.name && !strcmp(item->as.struct_def.name, name)) ||
            (item->as.struct_def.original_name &&
             !strcmp(item->as.struct_def.original_name, name))) return true;
    }
    return false;
}

static bool supported_parameter(TCO *ctx, const Parameter *parameter_info) {
    switch (parameter_info->type) {
    case TYPE_INT: case TYPE_U8: case TYPE_FLOAT: case TYPE_BOOL:
    case TYPE_STRING: case TYPE_BSTRING: case TYPE_ARRAY: case TYPE_STRUCT:
    case TYPE_ENUM: case TYPE_FUNCTION: case TYPE_TUPLE:
        break;
    default:
        return false;
    }
    if (parameter_info->type == TYPE_STRUCT &&
        resource_struct(ctx, parameter_info->struct_type_name)) return false;
    /* I have no ownership environment here. Until complete nested aggregate
     * ownership facts are attached to the parameter, I refuse containers whose
     * immediate element can carry an affine/opaque value. */
    if (parameter_info->type == TYPE_ARRAY &&
        (parameter_info->element_type == TYPE_STRUCT ||
         parameter_info->element_type == TYPE_UNION ||
         parameter_info->element_type == TYPE_OPAQUE ||
         parameter_info->element_type == TYPE_BORROW_SHARED ||
         parameter_info->element_type == TYPE_BORROW_MUT)) return false;
    return true;
}

static bool transform(ASTNode *program, ASTNode *function, bool verbose) {
    if (function->as.function.is_extern || !function->as.function.body) return false;
    TCO ctx = {.program = program, .function = function};
    for (int i = 0; i < function->as.function.param_count; i++)
        if (!supported_parameter(&ctx, &function->as.function.params[i])) return false;
    int count = function->as.function.param_count;
    ctx.shadowed = calloc((size_t)(count ? count : 1), sizeof(*ctx.shadowed));
    if (!ctx.shadowed) { fprintf(stderr, "I ran out of memory in TCO.\n"); exit(1); }
    ASTNode *body = function->as.function.body;
    for (unsigned serial = 0; ; serial++) {
        snprintf(ctx.prefix, sizeof(ctx.prefix), "__tco_%u_", serial);
        ctx.collision = false;
        memset(ctx.shadowed, 0, (size_t)count * sizeof(*ctx.shadowed));
        inspect_name(&ctx, &function->as.function.name, false);
        for (int i = 0; i < function->as.function.param_count; i++)
            inspect_name(&ctx, &function->as.function.params[i].name, false);
        if (!visit(&ctx, body)) { free(ctx.shadowed); return false; }
        if (!ctx.collision) break;
    }
    ctx.calls = 0;
    ctx.loop_calls = 0;
    rewrite(&ctx, body, false, 0);
    if (!ctx.calls) { free(ctx.shadowed); return false; }
    bool needs_flag = ctx.loop_calls > 0;
    ctx.rename = true;
    memset(ctx.shadowed, 0, (size_t)count * sizeof(*ctx.shadowed));
    if (!visit(&ctx, body)) { free(ctx.shadowed); return false; }
    ctx.calls = 0;
    ctx.loop_calls = 0;
    rewrite(&ctx, body, true, 0);

    ASTNode *outer = block(count + (needs_flag ? 1 : 0) + 1);
    for (int i = 0; i < count; i++)
        outer->as.block.statements[i] = parameter_binding(&ctx, i, false,
            ident(copy_name(function->as.function.params[i].name)));
    int loop_index = count;
    if (needs_flag) {
        ASTNode *initial = node_new(AST_BOOL);
        initial->as.bool_val = false;
        outer->as.block.statements[loop_index++] = binding(flag_name(&ctx),
                                                           TYPE_BOOL, initial);
    }
    ASTNode *loop = node_new(AST_WHILE);
    loop->as.while_stmt.condition = node_new(AST_BOOL);
    loop->as.while_stmt.condition->as.bool_val = true;
    ASTNode *iteration = block(needs_flag ? 3 : 2);
    int body_index = 0;
    if (needs_flag) {
        ASTNode *reset = node_new(AST_SET);
        reset->as.set.name = flag_name(&ctx);
        reset->as.set.value = node_new(AST_BOOL);
        reset->as.set.value->as.bool_val = false;
        iteration->as.block.statements[body_index++] = reset;
    }
    iteration->as.block.statements[body_index] = node_new(body->type);
    *iteration->as.block.statements[body_index] = *body;
    /* I do not repeat a void function's ordinary fallthrough. */
    iteration->as.block.statements[body_index + 1] = node_new(AST_BREAK);
    loop->as.while_stmt.body = iteration;
    outer->as.block.statements[loop_index] = loop;
    /* I preserve the body address already registered in the interpreter. */
    *body = *outer;
    free(outer);
    free(ctx.shadowed);
    if (verbose) fprintf(stderr, "I lowered self-tail returns in %s.\n", function->as.function.name);
    return true;
}

static int run(ASTNode *program, bool pure_only, bool verbose) {
    if (!program || program->type != AST_PROGRAM) return 0;
    int count = 0;
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *function = program->as.program.items[i];
        if (!function || function->type != AST_FUNCTION ||
            (pure_only && !function->as.function.is_pure)) continue;
        count += transform(program, function, verbose);
    }
    return count;
}

int tco_pass_run(ASTNode *program, bool verbose) { return run(program, false, verbose); }
void tco_pass(ASTNode *program) { (void)tco_pass_run(program, false); }
int tco_pass_pure(ASTNode *program) { return run(program, true, false); }
