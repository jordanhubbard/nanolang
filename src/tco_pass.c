/* I lower checked scalar self-tail returns to simultaneous updates and a loop.
 * Ordinary returns keep their meaning and their original result type. */
#include "tco_pass.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    ASTNode *function;
    char prefix[64];
    bool collision;
    bool rename;
    int calls;
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

static int parameter(TCO *ctx, const char *name) {
    for (int i = 0; i < ctx->function->as.function.param_count; i++)
        if (!strcmp(name, ctx->function->as.function.params[i].name)) return i;
    return -1;
}

static void inspect_name(TCO *ctx, char **name, bool reference) {
    if (!*name) return;
    if (!strncmp(*name, ctx->prefix, strlen(ctx->prefix))) ctx->collision = true;
    int index = parameter(ctx, *name);
    if (ctx->rename && reference && index >= 0) {
        free(*name);
        *name = slot_name(ctx, index, false);
    }
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
        if (node->as.call.func_expr) return false;
        inspect_name(ctx, &node->as.call.name, false);
        for (int i = 0; i < node->as.call.arg_count; i++)
            if (!visit(ctx, node->as.call.args[i])) return false;
        return true;
    case AST_PREFIX_OP:
        for (int i = 0; i < node->as.prefix_op.arg_count; i++)
            if (!visit(ctx, node->as.prefix_op.args[i])) return false;
        return true;
    case AST_LET:
        if (parameter(ctx, node->as.let.name) >= 0 ||
            !strcmp(node->as.let.name, ctx->function->as.function.name)) return false;
        inspect_name(ctx, &node->as.let.name, false);
        return visit(ctx, node->as.let.value);
    case AST_SET:
        inspect_name(ctx, &node->as.set.name, true);
        return visit(ctx, node->as.set.value);
    case AST_RETURN: return visit(ctx, node->as.return_stmt.value);
    case AST_ASSERT: return visit(ctx, node->as.assert.condition);
    case AST_PRINT: return visit(ctx, node->as.print.expr);
    case AST_IF:
        return visit(ctx, node->as.if_stmt.condition) &&
            visit(ctx, node->as.if_stmt.then_branch) && visit(ctx, node->as.if_stmt.else_branch);
    case AST_BLOCK:
        for (int i = 0; i < node->as.block.count; i++)
            if (!visit(ctx, node->as.block.statements[i])) return false;
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

static ASTNode *block(int count) {
    ASTNode *node = node_new(AST_BLOCK);
    node->as.block.count = count;
    node->as.block.statements = calloc(count ? count : 1, sizeof(ASTNode *));
    if (!node->as.block.statements) { fprintf(stderr, "I ran out of memory in TCO.\n"); exit(1); }
    return node;
}

static void rewrite(TCO *ctx, ASTNode *node, bool mutate) {
    if (!node) return;
    if (node->type == AST_BLOCK) {
        for (int i = 0; i < node->as.block.count; i++)
            rewrite(ctx, node->as.block.statements[i], mutate);
    } else if (node->type == AST_IF) {
        rewrite(ctx, node->as.if_stmt.then_branch, mutate);
        rewrite(ctx, node->as.if_stmt.else_branch, mutate);
    } else if (node->type == AST_RETURN) {
        ASTNode *call = node->as.return_stmt.value;
        if (!call || call->type != AST_CALL || call->as.call.func_expr ||
            !call->as.call.name || strcmp(call->as.call.name, ctx->function->as.function.name) ||
            call->as.call.arg_count != ctx->function->as.function.param_count) return;
        ctx->calls++;
        if (!mutate) return;
        int count = call->as.call.arg_count;
        ASTNode *replacement = block(count * 2 + 1);
        for (int i = 0; i < count; i++) {
            replacement->as.block.statements[i] = binding(slot_name(ctx, i, true),
                ctx->function->as.function.params[i].type, call->as.call.args[i]);
            call->as.call.args[i] = NULL;
            ASTNode *set = node_new(AST_SET);
            set->as.set.name = slot_name(ctx, i, false);
            set->as.set.value = ident(slot_name(ctx, i, true));
            replacement->as.block.statements[count + i] = set;
        }
        replacement->as.block.statements[count * 2] = node_new(AST_CONTINUE);
        free_ast(call);
        node->type = AST_BLOCK;
        node->as.block = replacement->as.block;
        free(replacement);
    }
}

static bool scalar(Type type) {
    return type == TYPE_INT || type == TYPE_FLOAT || type == TYPE_BOOL || type == TYPE_STRING;
}

static bool transform(ASTNode *function, bool verbose) {
    if (function->as.function.is_extern || !function->as.function.body) return false;
    for (int i = 0; i < function->as.function.param_count; i++)
        if (!scalar(function->as.function.params[i].type)) return false;
    TCO ctx = {.function = function};
    ASTNode *body = function->as.function.body;
    for (unsigned serial = 0; ; serial++) {
        snprintf(ctx.prefix, sizeof(ctx.prefix), "__tco_%u_", serial);
        ctx.collision = false;
        inspect_name(&ctx, &function->as.function.name, false);
        for (int i = 0; i < function->as.function.param_count; i++)
            inspect_name(&ctx, &function->as.function.params[i].name, false);
        if (!visit(&ctx, body)) return false;
        if (!ctx.collision) break;
    }
    rewrite(&ctx, body, false);
    if (!ctx.calls) return false;
    ctx.rename = true;
    visit(&ctx, body);
    rewrite(&ctx, body, true);

    int count = function->as.function.param_count;
    ASTNode *outer = block(count + 1);
    for (int i = 0; i < count; i++)
        outer->as.block.statements[i] = binding(slot_name(&ctx, i, false),
            function->as.function.params[i].type,
            ident(copy_name(function->as.function.params[i].name)));
    ASTNode *loop = node_new(AST_WHILE);
    loop->as.while_stmt.condition = node_new(AST_BOOL);
    loop->as.while_stmt.condition->as.bool_val = true;
    ASTNode *iteration = block(2);
    iteration->as.block.statements[0] = node_new(body->type);
    *iteration->as.block.statements[0] = *body;
    /* I do not repeat a void function's ordinary fallthrough. */
    iteration->as.block.statements[1] = node_new(AST_BREAK);
    loop->as.while_stmt.body = iteration;
    outer->as.block.statements[count] = loop;
    /* I preserve the body address already registered in the interpreter. */
    *body = *outer;
    free(outer);
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
        count += transform(function, verbose);
    }
    return count;
}

int tco_pass_run(ASTNode *program, bool verbose) { return run(program, false, verbose); }
void tco_pass(ASTNode *program) { (void)tco_pass_run(program, false); }
int tco_pass_pure(ASTNode *program) { return run(program, true, false); }
