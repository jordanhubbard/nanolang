/*
 * fold_constants.c — Compile-time constant folding pass.
 *
 * See fold_constants.h for the public API and design notes.
 */

#include "fold_constants.h"
#include <stdio.h>
#include <string.h>

/* Module-level state reset on each top-level call */
static int  g_fold_count;
static bool g_verbose;

static void note(const char *msg) {
    if (g_verbose) fprintf(stderr, "[fold_constants] %s\n", msg);
}

/* ── Literal predicates ──────────────────────────────────────────────────── */

static bool is_int_lit(const ASTNode *n)   { return n && n->type == AST_NUMBER; }
static bool is_float_lit(const ASTNode *n) { return n && n->type == AST_FLOAT;  }
static bool is_bool_lit(const ASTNode *n)  { return n && n->type == AST_BOOL;   }
static bool is_numeric(const ASTNode *n)   { return is_int_lit(n) || is_float_lit(n); }

static double as_double(const ASTNode *n) {
    return (n->type == AST_NUMBER) ? (double)n->as.number : n->as.float_val;
}

/* ── In-place replacement helpers ───────────────────────────────────────── */
/*
 * Each helper:
 *   1. Saves args/argc from prefix_op (stack copies, before union overwrite).
 *   2. Recursively frees all child AST nodes.
 *   3. Frees the args pointer array.
 *   4. Overwrites node->type and the relevant union field.
 *
 * The node pointer itself never changes, so the parent does not need
 * to be updated.
 */

static void replace_int(ASTNode *node, long long val,
                         ASTNode **args, int argc) {
    for (int i = 0; i < argc; i++) free_ast(args[i]);
    free(args);
    node->type       = AST_NUMBER;
    node->as.number  = val;
    g_fold_count++;
    note("folded → int literal");
}

static void replace_float(ASTNode *node, double val,
                           ASTNode **args, int argc) {
    for (int i = 0; i < argc; i++) free_ast(args[i]);
    free(args);
    node->type          = AST_FLOAT;
    node->as.float_val  = val;
    g_fold_count++;
    note("folded → float literal");
}

static void replace_bool(ASTNode *node, bool val,
                          ASTNode **args, int argc) {
    for (int i = 0; i < argc; i++) free_ast(args[i]);
    free(args);
    node->type         = AST_BOOL;
    node->as.bool_val  = val;
    g_fold_count++;
    note("folded → bool literal");
}

/* ── Forward declaration ─────────────────────────────────────────────────── */

static void fold_walk(ASTNode *node);

/* ── Core folding logic ──────────────────────────────────────────────────── */

static void try_fold_prefix_op(ASTNode *node) {
    /* Snapshot before any union writes */
    TokenType  op   = node->as.prefix_op.op;
    ASTNode  **args = node->as.prefix_op.args;
    int        argc = node->as.prefix_op.arg_count;

    /* ── Unary minus ── */
    if (op == TOKEN_MINUS && argc == 1) {
        if (is_int_lit(args[0])) {
            replace_int(node, -args[0]->as.number, args, argc);
            return;
        }
        if (is_float_lit(args[0])) {
            replace_float(node, -args[0]->as.float_val, args, argc);
            return;
        }
        return;
    }

    /* ── Unary not ── */
    if (op == TOKEN_NOT && argc == 1 && is_bool_lit(args[0])) {
        replace_bool(node, !args[0]->as.bool_val, args, argc);
        return;
    }

    if (argc != 2) return;

    /* ── Boolean && / || ── */
    if (op == TOKEN_AND && is_bool_lit(args[0]) && is_bool_lit(args[1])) {
        replace_bool(node, args[0]->as.bool_val && args[1]->as.bool_val, args, argc);
        return;
    }
    if (op == TOKEN_OR && is_bool_lit(args[0]) && is_bool_lit(args[1])) {
        replace_bool(node, args[0]->as.bool_val || args[1]->as.bool_val, args, argc);
        return;
    }

    /* ── Equality on booleans ── */
    if (op == TOKEN_EQ && is_bool_lit(args[0]) && is_bool_lit(args[1])) {
        replace_bool(node, args[0]->as.bool_val == args[1]->as.bool_val, args, argc);
        return;
    }
    if (op == TOKEN_NE && is_bool_lit(args[0]) && is_bool_lit(args[1])) {
        replace_bool(node, args[0]->as.bool_val != args[1]->as.bool_val, args, argc);
        return;
    }

    /* ── Numeric ops: both must be numeric ── */
    if (!is_numeric(args[0]) || !is_numeric(args[1])) return;

    /* Both integers: keep integer precision */
    if (is_int_lit(args[0]) && is_int_lit(args[1])) {
        long long a = args[0]->as.number;
        long long b = args[1]->as.number;
        switch (op) {
            case TOKEN_PLUS:    replace_int(node, a + b, args, argc); return;
            case TOKEN_MINUS:   replace_int(node, a - b, args, argc); return;
            case TOKEN_STAR:    replace_int(node, a * b, args, argc); return;
            case TOKEN_SLASH:   if (b != 0) replace_int(node, a / b, args, argc); return;
            case TOKEN_PERCENT: if (b != 0) replace_int(node, a % b, args, argc); return;
            case TOKEN_EQ:  replace_bool(node, a == b, args, argc); return;
            case TOKEN_NE:  replace_bool(node, a != b, args, argc); return;
            case TOKEN_LT:  replace_bool(node, a <  b, args, argc); return;
            case TOKEN_LE:  replace_bool(node, a <= b, args, argc); return;
            case TOKEN_GT:  replace_bool(node, a >  b, args, argc); return;
            case TOKEN_GE:  replace_bool(node, a >= b, args, argc); return;
            default: return;
        }
    }

    /* At least one float: promote to float */
    {
        double a = as_double(args[0]);
        double b = as_double(args[1]);
        switch (op) {
            case TOKEN_PLUS:    replace_float(node, a + b, args, argc); return;
            case TOKEN_MINUS:   replace_float(node, a - b, args, argc); return;
            case TOKEN_STAR:    replace_float(node, a * b, args, argc); return;
            case TOKEN_SLASH:   if (b != 0.0) replace_float(node, a / b, args, argc); return;
            case TOKEN_EQ:  replace_bool(node, a == b, args, argc); return;
            case TOKEN_NE:  replace_bool(node, a != b, args, argc); return;
            case TOKEN_LT:  replace_bool(node, a <  b, args, argc); return;
            case TOKEN_LE:  replace_bool(node, a <= b, args, argc); return;
            case TOKEN_GT:  replace_bool(node, a >  b, args, argc); return;
            case TOKEN_GE:  replace_bool(node, a >= b, args, argc); return;
            default: return;
        }
    }
}

/* ── Recursive walker ────────────────────────────────────────────────────── */

static void fold_walk_arr(ASTNode **arr, int count) {
    for (int i = 0; i < count; i++) {
        if (arr[i]) fold_walk(arr[i]);
    }
}

static void fold_walk(ASTNode *node) {
    if (!node) return;

    switch (node->type) {
        /* Pure literals / no-children declarations: nothing to do */
        case AST_NUMBER:
        case AST_FLOAT:
        case AST_STRING:
        case AST_BOOL:
        case AST_IDENTIFIER:
        case AST_BREAK:
        case AST_CONTINUE:
        case AST_IMPORT:
        case AST_MODULE_DECL:
        case AST_OPAQUE_TYPE:
        case AST_ENUM_DEF:
        case AST_QUALIFIED_NAME:
            return;

        case AST_PREFIX_OP:
            fold_walk_arr(node->as.prefix_op.args, node->as.prefix_op.arg_count);
            try_fold_prefix_op(node);
            return;

        case AST_LET:
            fold_walk(node->as.let.value);
            return;

        case AST_SET:
            fold_walk(node->as.set.value);
            return;

        case AST_RETURN:
            fold_walk(node->as.return_stmt.value);
            return;

        case AST_IF:
            fold_walk(node->as.if_stmt.condition);
            fold_walk(node->as.if_stmt.then_branch);
            fold_walk(node->as.if_stmt.else_branch);
            return;

        case AST_COND:
            fold_walk_arr(node->as.cond_expr.conditions, node->as.cond_expr.clause_count);
            fold_walk_arr(node->as.cond_expr.values,     node->as.cond_expr.clause_count);
            fold_walk(node->as.cond_expr.else_value);
            return;

        case AST_WHILE:
            fold_walk(node->as.while_stmt.condition);
            fold_walk(node->as.while_stmt.body);
            return;

        case AST_FOR:
            fold_walk(node->as.for_stmt.range_expr);
            fold_walk(node->as.for_stmt.body);
            return;

        case AST_BLOCK:
            fold_walk_arr(node->as.block.statements, node->as.block.count);
            return;

        case AST_PROGRAM:
            fold_walk_arr(node->as.program.items, node->as.program.count);
            return;

        case AST_FUNCTION:
            fold_walk(node->as.function.body);
            return;

        case AST_SHADOW:
            fold_walk(node->as.shadow.body);
            return;

        case AST_CALL:
            if (node->as.call.func_expr) fold_walk(node->as.call.func_expr);
            fold_walk_arr(node->as.call.args, node->as.call.arg_count);
            return;

        case AST_MODULE_QUALIFIED_CALL:
            fold_walk_arr(node->as.module_qualified_call.args,
                          node->as.module_qualified_call.arg_count);
            return;

        case AST_PRINT:
            fold_walk(node->as.print.expr);
            return;

        case AST_ASSERT:
            fold_walk(node->as.assert.condition);
            return;

        case AST_ARRAY_LITERAL:
            fold_walk_arr(node->as.array_literal.elements,
                          node->as.array_literal.element_count);
            return;

        case AST_TUPLE_LITERAL:
            fold_walk_arr(node->as.tuple_literal.elements,
                          node->as.tuple_literal.element_count);
            return;

        case AST_STRUCT_LITERAL:
            fold_walk_arr(node->as.struct_literal.field_values,
                          node->as.struct_literal.field_count);
            return;

        case AST_UNION_CONSTRUCT:
            fold_walk_arr(node->as.union_construct.field_values,
                          node->as.union_construct.field_count);
            return;

        case AST_MATCH:
            fold_walk(node->as.match_expr.expr);
            fold_walk_arr(node->as.match_expr.arm_bodies, node->as.match_expr.arm_count);
            fold_walk_arr(node->as.match_expr.guard_exprs, node->as.match_expr.arm_count);
            return;

        case AST_FIELD_ACCESS:
            fold_walk(node->as.field_access.object);
            return;

        case AST_TUPLE_INDEX:
            fold_walk(node->as.tuple_index.tuple);
            return;

        case AST_TRY_OP:
            fold_walk(node->as.try_op.operand);
            return;

        case AST_UNSAFE_BLOCK:
            fold_walk_arr(node->as.unsafe_block.statements,
                          node->as.unsafe_block.count);
            return;

        case AST_PAR_BLOCK:
            fold_walk_arr(node->as.par_block.bindings,
                          node->as.par_block.count);
            return;

        case AST_STRUCT_DEF:
        case AST_UNION_DEF:
        case AST_EFFECT_DECL:
        case AST_EFFECT_HANDLER:
        case AST_EFFECT_OP:
            return;

        default:
            return;
    }
}

/* ── Public API ──────────────────────────────────────────────────────────── */

int fold_constants(ASTNode *program, bool verbose) {
    g_fold_count = 0;
    g_verbose    = verbose;
    fold_walk(program);
    return g_fold_count;
}
