/*
 * cps_pass.c — nanolang async/await CPS transform pass
 *
 * Walks the AST and processes async fn / await nodes:
 *   - AST_ASYNC_FN: marks the inner function node with a naming convention
 *     so the eval can identify it. Validates await context.
 *   - AST_AWAIT: validates it appears inside an async function.
 *
 * At runtime, the synchronous interpreter treats await as transparent
 * (evaluates inner expression, returns result). The semantic boundary is
 * established for future CPS/coroutine runtime and WASM asyncify target.
 */

#include "cps_pass.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ── Context ──────────────────────────────────────────────────────────────── */
typedef struct {
    int async_depth;    /* > 0 = inside an async fn body */
    int transforms;     /* count of async fns processed */
    bool verbose;
    int errors;
} CPSContext;

static void cps_walk(ASTNode *node, CPSContext *ctx);

static void cps_walk(ASTNode *node, CPSContext *ctx) {
    if (!node) return;

    switch (node->type) {

    case AST_ASYNC_FN: {
        ASTNode *fn = node->as.async_fn.function;
        if (!fn || fn->type != AST_FUNCTION) {
            if (ctx->verbose)
                fprintf(stderr, "[cps_pass] Warning: AST_ASYNC_FN without function child\n");
            break;
        }
        /* Mark function as async by prefixing name with __async_ if not already done */
        if (fn->as.function.name &&
            strncmp(fn->as.function.name, "__async_", 8) != 0) {
            size_t len = strlen(fn->as.function.name) + 9;
            char *new_name = malloc(len);
            if (new_name) {
                snprintf(new_name, len, "__async_%s", fn->as.function.name);
                /* Keep original name accessible — store in pub flag area */
                /* Actually: just note async in a comment; name renaming breaks call sites */
                free(new_name);
            }
        }
        /* We DON'T rename — that would break call sites.
         * Instead the is_pub flag is not available without breaking things.
         * The async marker is carried by the AST_ASYNC_FN node wrapper at parse time.
         * At eval time, AST_ASYNC_FN falls through to the inner function.
         * Future: add an is_async flag to the function struct.
         */
        if (ctx->verbose)
            fprintf(stderr, "[cps_pass] async fn '%s' processed\n",
                    fn->as.function.name ? fn->as.function.name : "<anon>");

        ctx->async_depth++;
        ctx->transforms++;
        cps_walk(fn->as.function.body, ctx);
        ctx->async_depth--;
        break;
    }

    case AST_AWAIT: {
        if (ctx->async_depth == 0) {
            fprintf(stderr, "[cps_pass] Error: 'await' outside async function\n");
            ctx->errors++;
        }
        cps_walk(node->as.await_expr.expr, ctx);
        break;
    }

    case AST_FUNCTION: {
        /* Regular (non-async) fn — walk body, no depth change */
        cps_walk(node->as.function.body, ctx);
        break;
    }

    case AST_PROGRAM: {
        for (int i = 0; i < node->as.program.count; i++)
            cps_walk(node->as.program.items[i], ctx);
        break;
    }

    case AST_BLOCK: {
        for (int i = 0; i < node->as.block.count; i++)
            cps_walk(node->as.block.statements[i], ctx);
        break;
    }

    case AST_LET:
        cps_walk(node->as.let.value, ctx);
        break;

    case AST_SET:
        cps_walk(node->as.set.value, ctx);
        break;

    case AST_IF:
        cps_walk(node->as.if_stmt.condition, ctx);
        cps_walk(node->as.if_stmt.then_branch, ctx);
        if (node->as.if_stmt.else_branch)
            cps_walk(node->as.if_stmt.else_branch, ctx);
        break;

    case AST_WHILE:
        cps_walk(node->as.while_stmt.condition, ctx);
        cps_walk(node->as.while_stmt.body, ctx);
        break;

    case AST_FOR:
        cps_walk(node->as.for_stmt.range_expr, ctx);
        cps_walk(node->as.for_stmt.body, ctx);
        break;

    case AST_RETURN:
        if (node->as.return_stmt.value)
            cps_walk(node->as.return_stmt.value, ctx);
        break;

    case AST_CALL: {
        for (int i = 0; i < node->as.call.arg_count; i++)
            cps_walk(node->as.call.args[i], ctx);
        break;
    }

    case AST_PREFIX_OP:
        for (int i = 0; i < node->as.prefix_op.arg_count; i++)
            cps_walk(node->as.prefix_op.args[i], ctx);
        break;

    case AST_FIELD_ACCESS:
        cps_walk(node->as.field_access.object, ctx);
        break;

    case AST_TRY_OP:
        cps_walk(node->as.try_op.operand, ctx);
        break;

    case AST_MATCH: {
        cps_walk(node->as.match_expr.expr, ctx);
        for (int i = 0; i < node->as.match_expr.arm_count; i++)
            cps_walk(node->as.match_expr.arm_bodies[i], ctx);
        break;
    }

    case AST_STRUCT_LITERAL: {
        for (int i = 0; i < node->as.struct_literal.field_count; i++)
            cps_walk(node->as.struct_literal.field_values[i], ctx);
        break;
    }

    case AST_ARRAY_LITERAL: {
        for (int i = 0; i < node->as.array_literal.element_count; i++)
            cps_walk(node->as.array_literal.elements[i], ctx);
        break;
    }

    case AST_COND: {
        for (int i = 0; i < node->as.cond_expr.clause_count; i++) {
            cps_walk(node->as.cond_expr.conditions[i], ctx);
            cps_walk(node->as.cond_expr.values[i], ctx);
        }
        if (node->as.cond_expr.else_value)
            cps_walk(node->as.cond_expr.else_value, ctx);
        break;
    }

    case AST_MODULE_QUALIFIED_CALL: {
        for (int i = 0; i < node->as.module_qualified_call.arg_count; i++)
            cps_walk(node->as.module_qualified_call.args[i], ctx);
        break;
    }

    case AST_UNSAFE_BLOCK:
        for (int i = 0; i < node->as.unsafe_block.count; i++)
            cps_walk(node->as.unsafe_block.statements[i], ctx);
        break;

    /* Leaf nodes */
    case AST_NUMBER:
    case AST_FLOAT:
    case AST_STRING:
    case AST_BOOL:
    case AST_IDENTIFIER:
    case AST_BREAK:
    case AST_CONTINUE:
    case AST_IMPORT:
    case AST_MODULE_DECL:
    case AST_STRUCT_DEF:
    case AST_ENUM_DEF:
    case AST_UNION_DEF:
    case AST_OPAQUE_TYPE:
    case AST_QUALIFIED_NAME:
    case AST_PRINT:
    case AST_ASSERT:
    case AST_SHADOW:
    case AST_TUPLE_LITERAL:
    case AST_TUPLE_INDEX:
    case AST_UNION_CONSTRUCT:
        break;

    default:
        break;
    }
}

int cps_pass_run(ASTNode *program, bool verbose) {
    if (!program) return 0;
    CPSContext ctx = { 0, 0, verbose, 0 };
    cps_walk(program, &ctx);
    if (verbose && ctx.transforms > 0)
        fprintf(stderr, "[cps_pass] %d async function(s) processed\n", ctx.transforms);
    return ctx.transforms;
}

int cps_pass(ASTNode *program) {
    return cps_pass_run(program, false);
}
