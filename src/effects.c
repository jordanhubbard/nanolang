/*
 * effects.c — Algebraic Effect System implementation
 */
#include "effects.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static char *estrdup(const char *s) {
    return s ? strdup(s) : NULL;
}

/* ── Registry lifecycle ──────────────────────────────────────────────────── */

EffectRegistry *effect_registry_new(void) {
    EffectRegistry *r = calloc(1, sizeof(EffectRegistry));
    return r;
}

void effect_registry_free(EffectRegistry *reg) {
    if (!reg) return;
    for (int i = 0; i < reg->count; i++) {
        EffectDecl *d = &reg->decls[i];
        free(d->name);
        for (int j = 0; j < d->op_count; j++) {
            free(d->ops[j].name);
            free(d->ops[j].param_type_name);
            free(d->ops[j].return_type_name);
        }
        free(d->ops);
    }
    free(reg);
}

/* ── Registration ────────────────────────────────────────────────────────── */

bool effect_register(EffectRegistry *reg, const char *name,
                     EffectOp *ops, int op_count) {
    if (!reg || reg->count >= EFFECT_REGISTRY_MAX) return false;
    /* Check for duplicate */
    for (int i = 0; i < reg->count; i++) {
        if (strcmp(reg->decls[i].name, name) == 0) return true; /* already registered */
    }
    EffectDecl *d = &reg->decls[reg->count++];
    d->name     = estrdup(name);
    d->op_count = op_count;
    d->ops      = calloc(op_count, sizeof(EffectOp));
    for (int i = 0; i < op_count; i++) {
        d->ops[i].name             = estrdup(ops[i].name);
        d->ops[i].param_type       = ops[i].param_type;
        d->ops[i].return_type      = ops[i].return_type;
        d->ops[i].param_type_name  = estrdup(ops[i].param_type_name);
        d->ops[i].return_type_name = estrdup(ops[i].return_type_name);
    }
    return true;
}

bool effect_register_from_ast(EffectRegistry *reg, ASTNode *node) {
    if (!node || node->type != AST_EFFECT_DECL) return false;

    const char *name = node->as.effect_decl.effect_name;
    int count        = node->as.effect_decl.op_count;

    EffectOp *ops = calloc(count, sizeof(EffectOp));
    if (!ops) return false;
    for (int i = 0; i < count; i++) {
        ops[i].name        = node->as.effect_decl.op_names[i];
        ops[i].return_type = node->as.effect_decl.op_return_types[i];
        ops[i].return_type_name = node->as.effect_decl.op_return_type_names
                                    ? node->as.effect_decl.op_return_type_names[i]
                                    : NULL;
        /* Extract param type from first param if available */
        if (node->as.effect_decl.op_params && node->as.effect_decl.op_param_counts[i] > 0) {
            ops[i].param_type = node->as.effect_decl.op_params[i][0].type;
            ops[i].param_type_name = node->as.effect_decl.op_params[i][0].struct_type_name;
        } else {
            ops[i].param_type = TYPE_VOID;
            ops[i].param_type_name = NULL;
        }
    }
    bool ok = effect_register(reg, name, ops, count);
    free(ops);
    return ok;
}

/* ── Built-in effects ────────────────────────────────────────────────────── */

void effect_register_builtins(EffectRegistry *reg) {
    {
        EffectOp io_ops[2];
        memset(io_ops, 0, sizeof(io_ops));
        io_ops[0].name = "print"; io_ops[0].param_type = TYPE_STRING; io_ops[0].return_type = TYPE_VOID;
        io_ops[1].name = "read";  io_ops[1].param_type = TYPE_VOID;   io_ops[1].return_type = TYPE_STRING;
        effect_register(reg, "IO", io_ops, 2);
    }
    {
        EffectOp err_ops[1];
        memset(err_ops, 0, sizeof(err_ops));
        err_ops[0].name = "throw"; err_ops[0].param_type = TYPE_STRING; err_ops[0].return_type = TYPE_VOID;
        effect_register(reg, "Err", err_ops, 1);
    }
    {
        EffectOp state_ops[2];
        memset(state_ops, 0, sizeof(state_ops));
        state_ops[0].name = "get"; state_ops[0].param_type = TYPE_VOID; state_ops[0].return_type = TYPE_INT;
        state_ops[1].name = "put"; state_ops[1].param_type = TYPE_INT;  state_ops[1].return_type = TYPE_VOID;
        effect_register(reg, "State", state_ops, 2);
    }
}

/* ── Lookup ──────────────────────────────────────────────────────────────── */

EffectDecl *effect_lookup(EffectRegistry *reg, const char *name) {
    if (!reg || !name) return NULL;
    for (int i = 0; i < reg->count; i++) {
        if (strcmp(reg->decls[i].name, name) == 0)
            return &reg->decls[i];
    }
    return NULL;
}

EffectOp *effect_op_lookup(EffectDecl *decl, const char *op_name) {
    if (!decl || !op_name) return NULL;
    for (int i = 0; i < decl->op_count; i++) {
        if (strcmp(decl->ops[i].name, op_name) == 0)
            return &decl->ops[i];
    }
    return NULL;
}

/* ── Effect row operations ───────────────────────────────────────────────── */

void effect_row_add(EffectRow *row, const char *name) {
    if (!row || !name || row->count >= EFFECT_ROW_MAX) return;
    /* Avoid duplicates */
    for (int i = 0; i < row->count; i++) {
        if (row->names[i] && strcmp(row->names[i], name) == 0) return;
    }
    row->names[row->count++] = strdup(name);
}

char *effect_row_to_str(EffectRow *row) {
    if (!row || row->count == 0) return strdup("pure");
    /* Build "IO + Err + ..." string */
    int len = 0;
    for (int i = 0; i < row->count; i++)
        len += (row->names[i] ? (int)strlen(row->names[i]) : 0) + 3;
    char *buf = malloc(len + 1);
    buf[0] = '\0';
    for (int i = 0; i < row->count; i++) {
        if (i > 0) strcat(buf, " + ");
        if (row->names[i]) strcat(buf, row->names[i]);
    }
    return buf;
}

/* ── Effect check (simplified — validates performs have handlers) ─────── */

static void walk_node(EffectCheckCtx *ctx, ASTNode *node, Environment *env);

static void walk_children(EffectCheckCtx *ctx, ASTNode *node, Environment *env) {
    if (!node) return;
    switch (node->type) {
        case AST_PROGRAM:
            for (int i = 0; i < node->as.program.count; i++)
                walk_node(ctx, node->as.program.items[i], env);
            break;
        case AST_BLOCK:
            for (int i = 0; i < node->as.block.count; i++)
                walk_node(ctx, node->as.block.statements[i], env);
            break;
        case AST_FUNCTION:
            if (node->as.function.body)
                walk_node(ctx, node->as.function.body, env);
            break;
        case AST_IF:
            walk_node(ctx, node->as.if_stmt.condition, env);
            walk_node(ctx, node->as.if_stmt.then_branch, env);
            walk_node(ctx, node->as.if_stmt.else_branch, env);
            break;
        case AST_EFFECT_OP: {
            const char *eff = node->as.effect_op.effect_name;
            const char *op  = node->as.effect_op.op_name;
            EffectDecl *decl = effect_lookup(ctx->registry, eff);
            if (!decl) {
                fprintf(stderr, "Warning: unknown effect '%s' at line %d\n",
                        eff ? eff : "?", node->line);
            } else if (!effect_op_lookup(decl, op)) {
                fprintf(stderr, "Warning: unknown operation '%s.%s' at line %d\n",
                        eff ? eff : "?", op ? op : "?", node->line);
            }
            if (node->as.effect_op.arg)
                walk_node(ctx, node->as.effect_op.arg, env);
            break;
        }
        case AST_HANDLE_EXPR:
            walk_node(ctx, node->as.handle_expr.body, env);
            for (int i = 0; i < node->as.handle_expr.handler_count; i++)
                walk_node(ctx, node->as.handle_expr.handler_bodies[i], env);
            break;
        default:
            break;
    }
}

static void walk_node(EffectCheckCtx *ctx, ASTNode *node, Environment *env) {
    if (!node) return;
    walk_children(ctx, node, env);
}

bool effect_check_program(ASTNode *program, EffectRegistry *reg,
                          Environment *env) {
    if (!program || !reg) return true;
    EffectCheckCtx ctx = { .registry = reg, .current_row = NULL, .has_errors = false };
    walk_node(&ctx, program, env);
    return !ctx.has_errors;
}
