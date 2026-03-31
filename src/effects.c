/*
 * effects.c — Algebraic Effect System for nanolang
 *
 * Implements:
 *   - Effect declarations: effect IO { print : String -> Unit }
 *   - Effect handlers:     handle { ... } with { print s -> ... }
 *   - Effect polymorphism: function types carry effect rows
 *   - Effect row integration with the HM type inferencer
 *
 * Three built-in effects are pre-registered:
 *   IO    – print : string -> void,  read : void -> string
 *   Err   – throw : string -> void
 *   State – get : void -> int,       put : int -> void
 */

#include "effects.h"
#include "colors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Global interpreter handler stack ───────────────────────────────────── */

static EffectHandlerFrame *g_effect_stack = NULL;

void nl_effect_frame_push(EffectHandlerFrame *frame) {
    frame->outer   = g_effect_stack;
    g_effect_stack = frame;
}

void nl_effect_frame_pop(void) {
    if (g_effect_stack)
        g_effect_stack = g_effect_stack->outer;
}

EffectHandlerFrame *nl_effect_find_handler(const char *effect_name,
                                            const char *op_name,
                                            int        *arm_idx_out) {
    for (EffectHandlerFrame *f = g_effect_stack; f; f = f->outer) {
        if (!f->effect_name || strcmp(f->effect_name, effect_name) != 0)
            continue;
        for (int i = 0; i < f->handler_count; i++) {
            if (f->handler_op_names[i] &&
                strcmp(f->handler_op_names[i], op_name) == 0) {
                if (arm_idx_out) *arm_idx_out = i;
                return f;
            }
        }
    }
    return NULL;
}

/* ── Per-environment effect registry helpers ─────────────────────────────── */

void env_effect_register(Environment *env, ASTNode *decl_node) {
    if (!env || !decl_node) return;
    if (!env->effect_registry) {
        env->effect_registry = effect_registry_new();
        effect_register_builtins((EffectRegistry *)env->effect_registry);
    }
    effect_register_from_ast((EffectRegistry *)env->effect_registry, decl_node);
}

EffectDecl *env_effect_lookup(Environment *env, const char *name) {
    if (!env || !name) return NULL;
    if (!env->effect_registry) return NULL;
    return effect_lookup((EffectRegistry *)env->effect_registry, name);
}

/* ── Helpers ──────────────────────────────────────────────────────────────── */

static char *estrdup(const char *s) {
    if (!s) return NULL;
    char *p = malloc(strlen(s) + 1);
    if (!p) { fprintf(stderr, "[effects] out of memory\n"); exit(1); }
    strcpy(p, s);
    return p;
}

/* ── Registry lifecycle ──────────────────────────────────────────────────── */

EffectRegistry *effect_registry_new(void) {
    EffectRegistry *reg = calloc(1, sizeof(EffectRegistry));
    if (!reg) { fprintf(stderr, "[effects] out of memory\n"); exit(1); }
    return reg;
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

/* ── Effect context lifecycle ────────────────────────────────────────────── */

EffectCtx *effect_ctx_new(EffectRegistry *reg) {
    EffectCtx *ctx = calloc(1, sizeof(EffectCtx));
    if (!ctx) { fprintf(stderr, "[effects] out of memory\n"); exit(1); }
    ctx->registry = reg;
    return ctx;
}

void effect_ctx_free(EffectCtx *ctx) {
    if (!ctx) return;
    /* Handler frames are freed externally (they may be stack-allocated) */
    free(ctx);
}

/* ── Effect registration ─────────────────────────────────────────────────── */

bool effect_register(EffectRegistry *reg, const char *name,
                      EffectOp *ops, int op_count) {
    if (!reg || !name) return false;
    if (reg->count >= EFFECT_REGISTRY_MAX) {
        fprintf(stderr, "[effects] registry full — cannot register '%s'\n", name);
        return false;
    }
    /* Duplicate-check */
    for (int i = 0; i < reg->count; i++) {
        if (strcmp(reg->decls[i].name, name) == 0) {
            fprintf(stderr, "[effects] duplicate effect declaration '%s'\n", name);
            return false;
        }
    }
    EffectDecl *d = &reg->decls[reg->count++];
    d->name     = estrdup(name);
    d->op_count = op_count;
    d->ops      = calloc(op_count, sizeof(EffectOp));
    if (!d->ops) { fprintf(stderr, "[effects] out of memory\n"); exit(1); }
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
    if (!ops) { fprintf(stderr, "[effects] out of memory\n"); exit(1); }
    for (int i = 0; i < count; i++) {
        ops[i].name             = node->as.effect_decl.op_names[i];
        ops[i].param_type       = node->as.effect_decl.op_param_types[i];
        ops[i].return_type      = node->as.effect_decl.op_return_types[i];
        ops[i].param_type_name  = node->as.effect_decl.op_param_type_names
                                    ? node->as.effect_decl.op_param_type_names[i]
                                    : NULL;
        ops[i].return_type_name = node->as.effect_decl.op_return_type_names
                                    ? node->as.effect_decl.op_return_type_names[i]
                                    : NULL;
    }
    bool ok = effect_register(reg, name, ops, count);
    free(ops);
    return ok;
}

/* ── Effect lookup ───────────────────────────────────────────────────────── */

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

/* ── Effect row utilities ────────────────────────────────────────────────── */

EffectRow *effect_row_new(void) {
    EffectRow *r = calloc(1, sizeof(EffectRow));
    if (!r) { fprintf(stderr, "[effects] out of memory\n"); exit(1); }
    r->is_open = false;
    return r;
}

void effect_row_free(EffectRow *row) {
    if (!row) return;
    for (int i = 0; i < row->count; i++)
        free(row->effects[i]);
    free(row);
}

bool effect_row_add(EffectRow *row, const char *effect_name) {
    if (!row || !effect_name) return false;
    /* Dedup */
    for (int i = 0; i < row->count; i++) {
        if (strcmp(row->effects[i], effect_name) == 0)
            return true;
    }
    if (row->count >= EFFECT_ROW_MAX) {
        fprintf(stderr, "[effects] effect row overflow (max %d)\n", EFFECT_ROW_MAX);
        return false;
    }
    row->effects[row->count++] = estrdup(effect_name);
    return true;
}

bool effect_row_contains(const EffectRow *row, const char *effect_name) {
    if (!row || !effect_name) return false;
    for (int i = 0; i < row->count; i++) {
        if (strcmp(row->effects[i], effect_name) == 0)
            return true;
    }
    return false;
}

bool effect_row_merge(EffectRow *dst, const EffectRow *src) {
    if (!dst || !src) return false;
    for (int i = 0; i < src->count; i++) {
        if (!effect_row_add(dst, src->effects[i])) return false;
    }
    return true;
}

bool effect_row_subset(const EffectRow *sub, const EffectRow *sup) {
    if (!sub) return true;
    if (!sup) return sub->count == 0;
    for (int i = 0; i < sub->count; i++) {
        if (!effect_row_contains(sup, sub->effects[i]))
            return false;
    }
    return true;
}

char *effect_row_to_str(const EffectRow *row) {
    if (!row || row->count == 0) return estrdup("·");
    /* Estimate size */
    size_t sz = 2;
    for (int i = 0; i < row->count; i++)
        sz += strlen(row->effects[i]) + 2;
    char *buf = malloc(sz);
    if (!buf) { fprintf(stderr, "[effects] out of memory\n"); exit(1); }
    buf[0] = '\0';
    for (int i = 0; i < row->count; i++) {
        if (i) strcat(buf, ", ");
        strcat(buf, row->effects[i]);
    }
    return buf;
}

char *effect_arrow_type_str(const char *param_type, const char *ret_type,
                              const EffectRow *row) {
    char *row_str = effect_row_to_str(row);
    size_t sz = strlen(param_type) + strlen(ret_type) + strlen(row_str) + 16;
    char *buf = malloc(sz);
    if (!buf) { free(row_str); return NULL; }
    if (row && row->count > 0)
        snprintf(buf, sz, "(%s) -[%s]-> %s", param_type, row_str, ret_type);
    else
        snprintf(buf, sz, "(%s) -> %s", param_type, ret_type);
    free(row_str);
    return buf;
}

/* ── Built-in effects ────────────────────────────────────────────────────── */

void effect_register_builtins(EffectRegistry *reg) {
    /* IO effect: print and read */
    {
        EffectOp io_ops[] = {
            { "print", TYPE_STRING, TYPE_VOID, NULL, NULL },
            { "read",  TYPE_VOID,   TYPE_STRING, NULL, NULL },
        };
        effect_register(reg, "IO", io_ops, 2);
    }
    /* Err effect: throw */
    {
        EffectOp err_ops[] = {
            { "throw", TYPE_STRING, TYPE_VOID, NULL, NULL },
        };
        effect_register(reg, "Err", err_ops, 1);
    }
    /* State effect: get and put (int state for simplicity) */
    {
        EffectOp state_ops[] = {
            { "get", TYPE_VOID, TYPE_INT, NULL, NULL },
            { "put", TYPE_INT,  TYPE_VOID, NULL, NULL },
        };
        effect_register(reg, "State", state_ops, 2);
    }
}

/* ── Type-check helpers ──────────────────────────────────────────────────── */

static void emit_effect_error(const char *file, int line, int col,
                               const char *msg) {
    fprintf(stderr,
            "\n%s[E010] EFFECT ERROR%s  %s%s:%d:%d%s\n"
            "  %s\n",
            CSTART_ERROR, CEND,
            CSTART_DIM, file ? file : "<unknown>", line, col, CEND,
            msg);
}

/* ── Check an effect declaration ─────────────────────────────────────────── */

bool effect_check_decl(EffectCtx *ctx, ASTNode *node) {
    if (!node || node->type != AST_EFFECT_DECL) return false;

    const char *name = node->as.effect_decl.effect_name;
    if (!name || name[0] == '\0') {
        emit_effect_error("<unknown>", node->line, node->column,
                          "Effect declaration missing name.");
        ctx->has_error = true;
        return false;
    }

    /* Re-registering from AST (already done once at discovery) */
    if (effect_lookup(ctx->registry, name)) {
        /* already registered — this is a re-check; just validate ops */
    } else {
        if (!effect_register_from_ast(ctx->registry, node)) {
            ctx->has_error = true;
            return false;
        }
    }
    return true;
}

/* ── Check a handle expression ───────────────────────────────────────────── */

bool effect_check_handler(EffectCtx *ctx, ASTNode *node, Environment *env) {
    (void)env;
    if (!node || node->type != AST_EFFECT_HANDLER) return false;

    const char *eff_name = node->as.effect_handler.effect_name;
    if (!eff_name) {
        emit_effect_error("<unknown>", node->line, node->column,
                          "handle expression missing effect name.");
        ctx->has_error = true;
        return false;
    }

    EffectDecl *decl = effect_lookup(ctx->registry, eff_name);
    if (!decl) {
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "Unknown effect '%s' in handle expression.", eff_name);
        emit_effect_error("<unknown>", node->line, node->column, msg);
        ctx->has_error = true;
        return false;
    }

    /* Validate each handler clause names a real operation */
    int hcount = node->as.effect_handler.handler_count;
    for (int i = 0; i < hcount; i++) {
        const char *op = node->as.effect_handler.handler_op_names[i];
        if (!effect_op_lookup(decl, op)) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "Effect '%s' has no operation named '%s'.", eff_name, op);
            emit_effect_error("<unknown>", node->line, node->column, msg);
            ctx->has_error = true;
            return false;
        }
    }

    /* Push handler frame */
    EffectHandlerFrame frame;
    memset(&frame, 0, sizeof(frame));
    frame.effect_name        = (char *)eff_name;
    frame.handler_count      = hcount;
    frame.handler_op_names   = node->as.effect_handler.handler_op_names;
    frame.handler_param_names= node->as.effect_handler.handler_param_names;
    frame.handler_bodies     = node->as.effect_handler.handler_bodies;
    frame.outer              = ctx->handler_stack;
    ctx->handler_stack       = &frame;

    /* Pop handler frame */
    ctx->handler_stack = frame.outer;
    return !ctx->has_error;
}

/* ── Check a perform expression ──────────────────────────────────────────── */

bool effect_check_perform(EffectCtx *ctx, ASTNode *node) {
    if (!node || node->type != AST_EFFECT_OP) return false;

    const char *eff_name = node->as.effect_op.effect_name;
    const char *op_name  = node->as.effect_op.op_name;

    EffectDecl *decl = effect_lookup(ctx->registry, eff_name);
    if (!decl) {
        char msg[256];
        snprintf(msg, sizeof(msg), "Unknown effect '%s'.", eff_name);
        emit_effect_error("<unknown>", node->line, node->column, msg);
        ctx->has_error = true;
        return false;
    }

    EffectOp *op = effect_op_lookup(decl, op_name);
    if (!op) {
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "Effect '%s' has no operation '%s'.", eff_name, op_name);
        emit_effect_error("<unknown>", node->line, node->column, msg);
        ctx->has_error = true;
        return false;
    }

    /* Check that this perform is inside a handler for the effect */
    bool handled = false;
    for (EffectHandlerFrame *f = ctx->handler_stack; f; f = f->outer) {
        if (strcmp(f->effect_name, eff_name) == 0) {
            handled = true;
            break;
        }
    }
    if (!handled) {
        /* Unhandled perform is allowed in effect-polymorphic contexts;
         * emit a warning but do not error (the caller may handle it). */
        fprintf(stderr,
                "\n%s[W010] UNHANDLED EFFECT%s  perform %s.%s at line %d "
                "(no enclosing handler — effect is propagated)\n",
                CSTART_WARNING, CEND, eff_name, op_name, node->line);
    }
    return true;
}

/* ── Walk the program ────────────────────────────────────────────────────── */

static void walk_node(EffectCtx *ctx, ASTNode *node, Environment *env) {
    if (!node) return;
    switch (node->type) {
        case AST_EFFECT_DECL:
            effect_check_decl(ctx, node);
            return;
        case AST_EFFECT_HANDLER:
            effect_check_handler(ctx, node, env);
            /* Also walk the body and handler bodies */
            walk_node(ctx, node->as.effect_handler.body, env);
            for (int i = 0; i < node->as.effect_handler.handler_count; i++)
                walk_node(ctx, node->as.effect_handler.handler_bodies[i], env);
            return;
        case AST_EFFECT_OP:
            effect_check_perform(ctx, node);
            if (node->as.effect_op.arg)
                walk_node(ctx, node->as.effect_op.arg, env);
            return;
        case AST_PROGRAM:
            for (int i = 0; i < node->as.program.count; i++)
                walk_node(ctx, node->as.program.items[i], env);
            return;
        case AST_BLOCK:
            for (int i = 0; i < node->as.block.count; i++)
                walk_node(ctx, node->as.block.statements[i], env);
            return;
        case AST_FUNCTION:
            if (node->as.function.body)
                walk_node(ctx, node->as.function.body, env);
            return;
        case AST_IF:
            walk_node(ctx, node->as.if_stmt.condition,   env);
            walk_node(ctx, node->as.if_stmt.then_branch, env);
            walk_node(ctx, node->as.if_stmt.else_branch, env);
            return;
        case AST_WHILE:
            walk_node(ctx, node->as.while_stmt.condition, env);
            walk_node(ctx, node->as.while_stmt.body,      env);
            return;
        case AST_FOR:
            walk_node(ctx, node->as.for_stmt.range_expr, env);
            walk_node(ctx, node->as.for_stmt.body,       env);
            return;
        case AST_LET:
            walk_node(ctx, node->as.let.value, env);
            return;
        case AST_RETURN:
            walk_node(ctx, node->as.return_stmt.value, env);
            return;
        case AST_CALL:
            for (int i = 0; i < node->as.call.arg_count; i++)
                walk_node(ctx, node->as.call.args[i], env);
            return;
        case AST_PREFIX_OP:
            for (int i = 0; i < node->as.prefix_op.arg_count; i++)
                walk_node(ctx, node->as.prefix_op.args[i], env);
            return;
        case AST_UNSAFE_BLOCK:
            for (int i = 0; i < node->as.unsafe_block.count; i++)
                walk_node(ctx, node->as.unsafe_block.statements[i], env);
            return;
        case AST_PAR_BLOCK:
            for (int i = 0; i < node->as.par_block.count; i++)
                walk_node(ctx, node->as.par_block.bindings[i], env);
            return;
        case AST_PAR_LET:
            for (int i = 0; i < node->as.par_let.count; i++)
                walk_node(ctx, node->as.par_let.values[i], env);
            walk_node(ctx, node->as.par_let.body, env);
            return;
        case AST_MATCH:
            walk_node(ctx, node->as.match_expr.expr, env);
            for (int i = 0; i < node->as.match_expr.arm_count; i++)
                walk_node(ctx, node->as.match_expr.arm_bodies[i], env);
            return;
        default:
            return;
    }
}

bool effect_check_program(ASTNode *program, EffectRegistry *reg,
                           Environment *env) {
    if (!program || !reg) return true;  /* nothing to check */

    EffectCtx *ctx = effect_ctx_new(reg);

    /* First pass: register all effect declarations */
    if (program->type == AST_PROGRAM) {
        for (int i = 0; i < program->as.program.count; i++) {
            ASTNode *n = program->as.program.items[i];
            if (n && n->type == AST_EFFECT_DECL)
                effect_check_decl(ctx, n);
        }
    }

    /* Second pass: check handlers and performs */
    walk_node(ctx, program, env);

    bool ok = !ctx->has_error;
    effect_ctx_free(ctx);
    return ok;
}
