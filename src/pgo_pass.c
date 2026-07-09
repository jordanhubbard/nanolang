/*
 * pgo_pass.c — Profile-Guided Optimization pass for nanolang
 *
 * Implementation of pgo_pass.h.
 *
 * Inlining strategy
 * ─────────────────
 * For each AST_CALL to a hot function F(a, b, ...) with known body:
 *
 *   1. Clone the function body (deep copy of AST subtree).
 *   2. Substitute parameter references with argument expressions
 *      (simple rename; no alpha-renaming needed because nanolang has
 *       no closures that capture the inlined scope variables by name).
 *   3. Wrap the result in an AST_BLOCK so side-effects are sequenced.
 *   4. Replace the AST_CALL node's type = AST_BLOCK in-place.
 *
 * Limitations (v1.0):
 *   - Recursive functions are not inlined (cycle detection).
 *   - Variadic functions are not inlined.
 *   - Functions with side-effects on outer scope (set statements) are
 *     inlined conservatively — the transpiler handles them correctly
 *     because we preserve the block structure.
 *   - No escape analysis; all argument expressions are re-evaluated
 *     (safe because nanolang expressions are pure except for print/IO).
 */

#include "pgo_pass.h"
#include "colors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

/* ── .nano.prof parser ──────────────────────────────────────────────────── */

PGOProfile *pgo_load_profile_threshold(const char *path, uint64_t threshold) {
    if (!path) return NULL;
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[pgo] cannot open profile: %s\n", path);
        return NULL;
    }

    PGOProfile *prof = calloc(1, sizeof(PGOProfile));
    if (!prof) { fclose(f); return NULL; }
    prof->hot_threshold = threshold;

    char line[512];
    while (fgets(line, sizeof(line), f) && prof->count < PGO_MAX_ENTRIES) {
        /* Format: "fn_name count\n"  (collapsed-stack) */
        char  name[256];
        uint64_t cnt = 0;
        if (sscanf(line, "%255s %llu", name, (unsigned long long *)&cnt) != 2)
            continue;
        /* Skip flamegraph separator lines (contain ';') */
        if (strchr(name, ';')) continue;

        PGOEntry *e = &prof->entries[prof->count++];
        e->name   = strdup(name);
        e->calls  = cnt;
        e->is_hot = false;  /* determined after full load */
    }
    fclose(f);

    /* Determine hotness: auto-threshold = p90 call count if threshold == 0 */
    if (threshold == 0 && prof->count > 0) {
        /* Sort by calls descending to find p90 */
        for (int i = 0; i < prof->count - 1; i++) {
            for (int j = i + 1; j < prof->count; j++) {
                if (prof->entries[j].calls > prof->entries[i].calls) {
                    PGOEntry tmp    = prof->entries[i];
                    prof->entries[i] = prof->entries[j];
                    prof->entries[j] = tmp;
                }
            }
        }
        /* Top 10% are hot */
        int top_n = (prof->count / 10) + 1;
        prof->hot_threshold = prof->entries[top_n - 1].calls;
        if (prof->hot_threshold == 0) prof->hot_threshold = 1;
    }

    int hot = 0;
    for (int i = 0; i < prof->count; i++) {
        if (prof->entries[i].calls >= prof->hot_threshold) {
            prof->entries[i].is_hot = true;
            hot++;
        }
    }
    prof->functions_hot = hot;
    return prof;
}

PGOProfile *pgo_load_profile(const char *path) {
    return pgo_load_profile_threshold(path, 0);
}

void pgo_profile_free(PGOProfile *prof) {
    if (!prof) return;
    for (int i = 0; i < prof->count; i++) free(prof->entries[i].name);
    free(prof);
}

bool pgo_is_hot(const PGOProfile *prof, const char *name) {
    if (!prof || !name) return false;
    for (int i = 0; i < prof->count; i++) {
        if (prof->entries[i].is_hot &&
            prof->entries[i].name &&
            strcmp(prof->entries[i].name, name) == 0)
            return true;
    }
    return false;
}

void pgo_print_report(const PGOProfile *prof) {
    if (!prof) return;
    fprintf(stderr, "\n[pgo] Profile report — threshold: %llu calls\n",
            (unsigned long long)prof->hot_threshold);
    fprintf(stderr, "  %-40s  %12s  %s\n", "Function", "Calls", "Hot");
    for (int i = 0; i < prof->count && i < 20; i++) {
        fprintf(stderr, "  %-40s  %12llu  %s\n",
                prof->entries[i].name ? prof->entries[i].name : "?",
                (unsigned long long)prof->entries[i].calls,
                prof->entries[i].is_hot ? "✓" : "");
    }
    if (prof->count > 20)
        fprintf(stderr, "  ... (%d more)\n", prof->count - 20);
    fprintf(stderr, "\n  Hot: %d/%d functions, %d call sites inlined\n\n",
            prof->functions_hot, prof->count, prof->sites_inlined);
}

/* ── AST utilities ──────────────────────────────────────────────────────── */

/* Find a top-level function by name in the program AST */
static ASTNode *find_function(ASTNode *program, const char *name) {
    if (!program || program->type != AST_PROGRAM || !name) return NULL;
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *n = program->as.program.items[i];
        if (!n) continue;
        if (n->type == AST_FUNCTION &&
            n->as.function.name &&
            strcmp(n->as.function.name, name) == 0)
            return n;
    }
    return NULL;
}

/* Count statements in a function body */
static int count_body_stmts(ASTNode *fn) {
    if (!fn || fn->type != AST_FUNCTION || !fn->as.function.body) return 0;
    ASTNode *body = fn->as.function.body;
    if (body->type == AST_BLOCK) return body->as.block.count;
    return 1;
}

/* Check if function is recursive (simple name scan, no full call-graph) */
static bool expr_calls(ASTNode *node, const char *name);
static bool expr_calls(ASTNode *node, const char *name) {
    if (!node || !name) return false;
    if (node->type == AST_CALL) {
        if (node->as.call.name != NULL &&
            strcmp(node->as.call.name, name) == 0)
            return true;
        for (int i = 0; i < node->as.call.arg_count; i++)
            if (expr_calls(node->as.call.args[i], name)) return true;
    }
    if (node->type == AST_BLOCK)
        for (int i = 0; i < node->as.block.count; i++)
            if (expr_calls(node->as.block.statements[i], name)) return true;
    if (node->type == AST_IF)
        return expr_calls(node->as.if_stmt.condition, name) ||
               expr_calls(node->as.if_stmt.then_branch, name) ||
               expr_calls(node->as.if_stmt.else_branch, name);
    if (node->type == AST_RETURN)
        return expr_calls(node->as.return_stmt.value, name);
    if (node->type == AST_LET)
        return expr_calls(node->as.let.value, name);
    if (node->type == AST_PREFIX_OP)
        for (int i = 0; i < node->as.prefix_op.arg_count; i++)
            if (expr_calls(node->as.prefix_op.args[i], name)) return true;
    return false;
}

static bool is_recursive(ASTNode *fn) {
    if (!fn || fn->type != AST_FUNCTION) return false;
    return expr_calls(fn->as.function.body, fn->as.function.name);
}

/* ── AST node deep-copy ─────────────────────────────────────────────────── */
/*
 * Shallow clone of a node — enough for inlining.
 * We share string pointers (they are const in the AST) and only
 * deep-copy the structural arrays (args, statements, etc.).
 */
static ASTNode *clone_node(ASTNode *src);

static ASTNode **clone_nodes(ASTNode **arr, int n) {
    if (!arr || n <= 0) return NULL;
    ASTNode **out = calloc((size_t)n, sizeof(ASTNode *));
    if (!out) return NULL;
    for (int i = 0; i < n; i++) out[i] = clone_node(arr[i]);
    return out;
}

static ASTNode *clone_node(ASTNode *src) {
    if (!src) return NULL;
    ASTNode *dst = calloc(1, sizeof(ASTNode));
    if (!dst) return NULL;
    *dst = *src;  /* bitwise copy — overwrite array fields below */

    switch (src->type) {
        case AST_BLOCK:
            dst->as.block.statements = clone_nodes(src->as.block.statements,
                                                    src->as.block.count);
            break;
        case AST_CALL:
            /* name is a char* shared constant; no need to deep-copy */
            dst->as.call.args   = clone_nodes(src->as.call.args,
                                               src->as.call.arg_count);
            dst->as.call.func_expr = clone_node(src->as.call.func_expr);
            break;
        case AST_PREFIX_OP:
            dst->as.prefix_op.args = clone_nodes(src->as.prefix_op.args,
                                                   src->as.prefix_op.arg_count);
            break;
        case AST_IF:
            dst->as.if_stmt.condition   = clone_node(src->as.if_stmt.condition);
            dst->as.if_stmt.then_branch = clone_node(src->as.if_stmt.then_branch);
            dst->as.if_stmt.else_branch = clone_node(src->as.if_stmt.else_branch);
            break;
        case AST_WHILE:
            dst->as.while_stmt.condition = clone_node(src->as.while_stmt.condition);
            dst->as.while_stmt.body      = clone_node(src->as.while_stmt.body);
            break;
        case AST_FOR:
            dst->as.for_stmt.range_expr = clone_node(src->as.for_stmt.range_expr);
            dst->as.for_stmt.body       = clone_node(src->as.for_stmt.body);
            break;
        case AST_LET:
            dst->as.let.value = clone_node(src->as.let.value);
            break;
        case AST_RETURN:
            dst->as.return_stmt.value = clone_node(src->as.return_stmt.value);
            break;
        case AST_MATCH:
            dst->as.match_expr.expr     = clone_node(src->as.match_expr.expr);
            if (src->as.match_expr.arm_count > 0) {
                dst->as.match_expr.arm_bodies = clone_nodes(
                    src->as.match_expr.arm_bodies, src->as.match_expr.arm_count);
            }
            break;
        case AST_HANDLE_EXPR:
            dst->as.handle_expr.body = clone_node(src->as.handle_expr.body);
            if (src->as.handle_expr.handler_count > 0) {
                dst->as.handle_expr.handler_bodies = clone_nodes(
                    src->as.handle_expr.handler_bodies,
                    src->as.handle_expr.handler_count);
            }
            break;
        default:
            /* Leaf nodes or nodes we don't recurse into — bitwise copy is fine */
            break;
    }
    return dst;
}

/* ── Argument substitution ──────────────────────────────────────────────── */
/*
 * substitute_params(node, param_names, param_count, args)
 *
 * Walk the cloned body and replace any AST_IDENTIFIER that matches a
 * parameter name with a clone of the corresponding argument expression.
 *
 * Mutates node in-place.
 */
static void substitute_params(ASTNode *node, char **params, int pcount,
                               ASTNode **args) {
    if (!node) return;

    /* If this is a parameter identifier, replace its content with the arg */
    if (node->type == AST_IDENTIFIER) {
        for (int i = 0; i < pcount; i++) {
            if (params[i] && strcmp(node->as.identifier, params[i]) == 0) {
                /* Replace in-place: copy arg node content into this slot */
                ASTNode *replacement = clone_node(args[i]);
                if (replacement) *node = *replacement;
                /* Note: replacement is now an orphan; minor leak in inliner */
                return;
            }
        }
        return;
    }

    /* Recurse */
    switch (node->type) {
        case AST_BLOCK:
            for (int i = 0; i < node->as.block.count; i++)
                substitute_params(node->as.block.statements[i],
                                   params, pcount, args);
            break;
        case AST_CALL:
            substitute_params(node->as.call.func_expr, params, pcount, args);
            for (int i = 0; i < node->as.call.arg_count; i++)
                substitute_params(node->as.call.args[i], params, pcount, args);
            break;
        case AST_PREFIX_OP:
            for (int i = 0; i < node->as.prefix_op.arg_count; i++)
                substitute_params(node->as.prefix_op.args[i],
                                   params, pcount, args);
            break;
        case AST_IF:
            substitute_params(node->as.if_stmt.condition, params, pcount, args);
            substitute_params(node->as.if_stmt.then_branch, params, pcount, args);
            substitute_params(node->as.if_stmt.else_branch, params, pcount, args);
            break;
        case AST_LET:
            substitute_params(node->as.let.value, params, pcount, args);
            break;
        case AST_RETURN:
            substitute_params(node->as.return_stmt.value, params, pcount, args);
            break;
        case AST_WHILE:
            substitute_params(node->as.while_stmt.condition, params, pcount, args);
            substitute_params(node->as.while_stmt.body, params, pcount, args);
            break;
        case AST_FOR:
            substitute_params(node->as.for_stmt.range_expr, params, pcount, args);
            substitute_params(node->as.for_stmt.body, params, pcount, args);
            break;
        case AST_MATCH:
            substitute_params(node->as.match_expr.expr, params, pcount, args);
            for (int i = 0; i < node->as.match_expr.arm_count; i++)
                substitute_params(node->as.match_expr.arm_bodies[i],
                                   params, pcount, args);
            break;
        default:
            break;
    }
}

/* ── Inline one call site ───────────────────────────────────────────────── */
/*
 * Try to inline the call node `call_node` given the definition `fn_node`.
 * On success, mutates call_node in-place to become an AST_BLOCK containing
 * the substituted function body.
 * Returns true on success.
 */
static bool inline_call(ASTNode *call_node, ASTNode *fn_node, int depth) {
    if (depth >= PGO_MAX_INLINE_DEPTH) return false;
    if (!call_node || !fn_node) return false;
    if (fn_node->type != AST_FUNCTION) return false;

    int param_count = fn_node->as.function.param_count;
    int arg_count   = call_node->as.call.arg_count;

    /* Arity must match (no variadic support in inliner) */
    if (param_count != arg_count) return false;

    /* Body size check */
    if (count_body_stmts(fn_node) > PGO_MAX_INLINE_STMTS) return false;

    /* Recursion guard */
    if (is_recursive(fn_node)) return false;

    /* Clone the function body */
    ASTNode *body_clone = clone_node(fn_node->as.function.body);
    if (!body_clone) return false;

    /* Build param name array */
    char **param_names = calloc((size_t)param_count, sizeof(char *));
    if (!param_names) { return false; }
    for (int i = 0; i < param_count; i++)
        param_names[i] = fn_node->as.function.params[i].name;

    /* Substitute */
    substitute_params(body_clone, param_names, param_count,
                      call_node->as.call.args);
    free(param_names);

    /* Rewrite call_node in-place as the inlined body */
    /* Preserve line/col for debug */
    int saved_line = call_node->line;
    int saved_col  = call_node->column;

    if (body_clone->type == AST_BLOCK) {
        *call_node = *body_clone;
    } else {
        /* Wrap single expression in a block */
        call_node->type = AST_BLOCK;
        call_node->as.block.count = 1;
        call_node->as.block.statements = calloc(1, sizeof(ASTNode *));
        if (call_node->as.block.statements)
            call_node->as.block.statements[0] = body_clone;
    }
    call_node->line   = saved_line;
    call_node->column = saved_col;

    return true;
}

/* ── AST walker — apply inlining ────────────────────────────────────────── */

typedef struct {
    ASTNode    *program;
    PGOProfile *prof;
    int         inlined;
    int         depth;
} InlineCtx;

static void walk_inline(InlineCtx *ctx, ASTNode *node);

static void inline_if_hot(InlineCtx *ctx, ASTNode *call_node) {
    if (!call_node || call_node->type != AST_CALL) return;
    if (!call_node->as.call.name) return;
    /* func_expr != NULL means indirect call — skip (can't determine callee at compile time) */
    if (call_node->as.call.func_expr != NULL) return;

    const char *callee_name = call_node->as.call.name;
    if (!pgo_is_hot(ctx->prof, callee_name)) return;

    ASTNode *fn = find_function(ctx->program, callee_name);
    if (!fn) return;

    if (inline_call(call_node, fn, ctx->depth)) {
        ctx->inlined++;
        ctx->prof->sites_inlined++;
        /* Walk the inlined body to opportunistically inline nested hot calls */
        ctx->depth++;
        walk_inline(ctx, call_node);
        ctx->depth--;
    }
}

static void walk_inline(InlineCtx *ctx, ASTNode *node) {
    if (!node) return;

    /* Try to inline this node itself */
    if (node->type == AST_CALL) {
        inline_if_hot(ctx, node);
        /* After possible inlining, re-walk the (possibly new) block */
        if (node->type == AST_BLOCK)
            walk_inline(ctx, node);
        return;
    }

    /* Recurse into compound nodes */
    switch (node->type) {
        case AST_PROGRAM:
            for (int i = 0; i < node->as.program.count; i++)
                walk_inline(ctx, node->as.program.items[i]);
            break;
        case AST_FUNCTION:
            walk_inline(ctx, node->as.function.body);
            break;
        case AST_BLOCK:
            for (int i = 0; i < node->as.block.count; i++)
                walk_inline(ctx, node->as.block.statements[i]);
            break;
        case AST_IF:
            walk_inline(ctx, node->as.if_stmt.condition);
            walk_inline(ctx, node->as.if_stmt.then_branch);
            walk_inline(ctx, node->as.if_stmt.else_branch);
            break;
        case AST_WHILE:
            walk_inline(ctx, node->as.while_stmt.condition);
            walk_inline(ctx, node->as.while_stmt.body);
            break;
        case AST_FOR:
            walk_inline(ctx, node->as.for_stmt.range_expr);
            walk_inline(ctx, node->as.for_stmt.body);
            break;
        case AST_LET:
            walk_inline(ctx, node->as.let.value);
            break;
        case AST_RETURN:
            walk_inline(ctx, node->as.return_stmt.value);
            break;
        case AST_PREFIX_OP:
            for (int i = 0; i < node->as.prefix_op.arg_count; i++)
                walk_inline(ctx, node->as.prefix_op.args[i]);
            break;
        case AST_MATCH:
            walk_inline(ctx, node->as.match_expr.expr);
            for (int i = 0; i < node->as.match_expr.arm_count; i++)
                walk_inline(ctx, node->as.match_expr.arm_bodies[i]);
            break;
        case AST_HANDLE_EXPR:
            walk_inline(ctx, node->as.handle_expr.body);
            for (int i = 0; i < node->as.handle_expr.handler_count; i++)
                if (node->as.handle_expr.handler_bodies)
                    walk_inline(ctx, node->as.handle_expr.handler_bodies[i]);
            break;
        case AST_UNSAFE_BLOCK:
            for (int i = 0; i < node->as.unsafe_block.count; i++)
                walk_inline(ctx, node->as.unsafe_block.statements[i]);
            break;
        case AST_PAR_BLOCK:
            for (int i = 0; i < node->as.par_block.count; i++)
                walk_inline(ctx, node->as.par_block.bindings[i]);
            break;
        default:
            break;
    }
}

/* ── Public pgo_apply ───────────────────────────────────────────────────── */

int pgo_apply(ASTNode *program, PGOProfile *prof) {
    if (!program || !prof) return 0;

    InlineCtx ctx = {
        .program = program,
        .prof    = prof,
        .inlined = 0,
        .depth   = 0,
    };

    walk_inline(&ctx, program);

    if (ctx.inlined > 0) {
        fprintf(stderr, "[pgo] inlined %d call site(s) for %d hot function(s) "
                        "(threshold: %llu calls)\n",
                ctx.inlined, prof->functions_hot,
                (unsigned long long)prof->hot_threshold);
    }
    return ctx.inlined;
}
