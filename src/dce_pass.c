/*
 * dce_pass.c — Dead Code Elimination pass.
 *
 * See dce_pass.h for the public API and design notes.
 */

#include "dce_pass.h"
#include <stdio.h>
#include <string.h>

/* Module-level state reset on each top-level call */
static int  g_elim_count;
static bool g_verbose;

static void note(const char *msg) {
    if (g_verbose) fprintf(stderr, "[dce_pass] %s\n", msg);
}

/* ── Forward declarations ────────────────────────────────────────────────── */

static void dce_walk(ASTNode *node);

/* ── Side-effect detection ───────────────────────────────────────────────── */
/*
 * Returns true if the expression subtree *definitely* has a side effect:
 * function calls, print, or assert.  Conservative (returns true when unsure).
 */
static bool has_side_effect(const ASTNode *node) {
    if (!node) return false;
    switch (node->type) {
        case AST_CALL:
        case AST_MODULE_QUALIFIED_CALL:
        case AST_PRINT:
        case AST_ASSERT:
        case AST_EFFECT_OP:
            return true;

        case AST_PREFIX_OP:
            for (int i = 0; i < node->as.prefix_op.arg_count; i++)
                if (has_side_effect(node->as.prefix_op.args[i])) return true;
            return false;

        case AST_IF:
            return has_side_effect(node->as.if_stmt.condition)
                || has_side_effect(node->as.if_stmt.then_branch)
                || has_side_effect(node->as.if_stmt.else_branch);

        case AST_BLOCK:
            for (int i = 0; i < node->as.block.count; i++)
                if (has_side_effect(node->as.block.statements[i])) return true;
            return false;

        /* Literals and identifiers are pure */
        case AST_NUMBER:
        case AST_FLOAT:
        case AST_STRING:
        case AST_BOOL:
        case AST_IDENTIFIER:
            return false;

        /* Struct/tuple/array construction: check fields */
        case AST_STRUCT_LITERAL:
            for (int i = 0; i < node->as.struct_literal.field_count; i++)
                if (has_side_effect(node->as.struct_literal.field_values[i])) return true;
            return false;

        case AST_ARRAY_LITERAL:
            for (int i = 0; i < node->as.array_literal.element_count; i++)
                if (has_side_effect(node->as.array_literal.elements[i])) return true;
            return false;

        case AST_TUPLE_LITERAL:
            for (int i = 0; i < node->as.tuple_literal.element_count; i++)
                if (has_side_effect(node->as.tuple_literal.elements[i])) return true;
            return false;

        case AST_FIELD_ACCESS:
            return has_side_effect(node->as.field_access.object);

        case AST_TUPLE_INDEX:
            return has_side_effect(node->as.tuple_index.tuple);

        /* Conservative: assume side effects for anything else */
        default:
            return true;
    }
}

/* ── Reference counting ──────────────────────────────────────────────────── */
/*
 * Count how many times the identifier *name* appears as an AST_IDENTIFIER
 * in the subtree rooted at *node*.
 */
static int count_refs(const ASTNode *node, const char *name) {
    if (!node) return 0;
    switch (node->type) {
        case AST_IDENTIFIER:
            return (strcmp(node->as.identifier, name) == 0) ? 1 : 0;

        case AST_NUMBER: case AST_FLOAT: case AST_STRING: case AST_BOOL:
        case AST_BREAK:  case AST_CONTINUE: case AST_IMPORT:
        case AST_MODULE_DECL: case AST_OPAQUE_TYPE: case AST_ENUM_DEF:
        case AST_QUALIFIED_NAME:
            return 0;

        case AST_PREFIX_OP: {
            int s = 0;
            for (int i = 0; i < node->as.prefix_op.arg_count; i++)
                s += count_refs(node->as.prefix_op.args[i], name);
            return s;
        }
        case AST_LET:
            return count_refs(node->as.let.value, name);
        case AST_SET:
            return count_refs(node->as.set.value, name)
                 + (strcmp(node->as.set.name, name) == 0 ? 1 : 0);
        case AST_RETURN:
            return count_refs(node->as.return_stmt.value, name);
        case AST_IF:
            return count_refs(node->as.if_stmt.condition,   name)
                 + count_refs(node->as.if_stmt.then_branch,  name)
                 + count_refs(node->as.if_stmt.else_branch,  name);
        case AST_WHILE:
            return count_refs(node->as.while_stmt.condition, name)
                 + count_refs(node->as.while_stmt.body,      name);
        case AST_FOR:
            return count_refs(node->as.for_stmt.range_expr, name)
                 + count_refs(node->as.for_stmt.body,        name);
        case AST_BLOCK: {
            int s = 0;
            for (int i = 0; i < node->as.block.count; i++)
                s += count_refs(node->as.block.statements[i], name);
            return s;
        }
        case AST_PROGRAM: {
            int s = 0;
            for (int i = 0; i < node->as.program.count; i++)
                s += count_refs(node->as.program.items[i], name);
            return s;
        }
        case AST_FUNCTION:
            return count_refs(node->as.function.body, name);
        case AST_CALL: {
            int s = count_refs(node->as.call.func_expr, name);
            for (int i = 0; i < node->as.call.arg_count; i++)
                s += count_refs(node->as.call.args[i], name);
            return s;
        }
        case AST_MODULE_QUALIFIED_CALL: {
            int s = 0;
            for (int i = 0; i < node->as.module_qualified_call.arg_count; i++)
                s += count_refs(node->as.module_qualified_call.args[i], name);
            return s;
        }
        case AST_PRINT:
            return count_refs(node->as.print.expr, name);
        case AST_ASSERT:
            return count_refs(node->as.assert.condition, name);
        case AST_COND: {
            int s = count_refs(node->as.cond_expr.else_value, name);
            for (int i = 0; i < node->as.cond_expr.clause_count; i++) {
                s += count_refs(node->as.cond_expr.conditions[i], name);
                s += count_refs(node->as.cond_expr.values[i],     name);
            }
            return s;
        }
        case AST_STRUCT_LITERAL: {
            int s = 0;
            for (int i = 0; i < node->as.struct_literal.field_count; i++)
                s += count_refs(node->as.struct_literal.field_values[i], name);
            return s;
        }
        case AST_ARRAY_LITERAL: {
            int s = 0;
            for (int i = 0; i < node->as.array_literal.element_count; i++)
                s += count_refs(node->as.array_literal.elements[i], name);
            return s;
        }
        case AST_TUPLE_LITERAL: {
            int s = 0;
            for (int i = 0; i < node->as.tuple_literal.element_count; i++)
                s += count_refs(node->as.tuple_literal.elements[i], name);
            return s;
        }
        case AST_FIELD_ACCESS:
            return count_refs(node->as.field_access.object, name);
        case AST_TUPLE_INDEX:
            return count_refs(node->as.tuple_index.tuple, name);
        case AST_TRY_OP:
            return count_refs(node->as.try_op.operand, name);
        case AST_MATCH: {
            int s = count_refs(node->as.match_expr.expr, name);
            for (int i = 0; i < node->as.match_expr.arm_count; i++) {
                s += count_refs(node->as.match_expr.arm_bodies[i],  name);
                s += count_refs(node->as.match_expr.guard_exprs[i], name);
            }
            return s;
        }
        case AST_UNION_CONSTRUCT: {
            int s = 0;
            for (int i = 0; i < node->as.union_construct.field_count; i++)
                s += count_refs(node->as.union_construct.field_values[i], name);
            return s;
        }
        default:
            return 0;
    }
}

/* ── if(bool) elimination ────────────────────────────────────────────────── */
/*
 * Replace `if (true/false) A else B` in-place.
 *
 * Strategy: copy the chosen branch's ASTNode struct into *node* using memcpy,
 * then free the shallow wrapper pointer (not its children — we just moved them).
 * Free the discarded branch via free_ast.
 */
static void try_elim_if(ASTNode *node) {
    if (node->as.if_stmt.condition->type != AST_BOOL) return;

    bool cond_val    = node->as.if_stmt.condition->as.bool_val;
    ASTNode *kept    = cond_val ? node->as.if_stmt.then_branch
                                : node->as.if_stmt.else_branch;
    ASTNode *dropped = cond_val ? node->as.if_stmt.else_branch
                                : node->as.if_stmt.then_branch;
    ASTNode *cond    = node->as.if_stmt.condition;

    if (!kept) {
        /* if(false) A  (no else): keep nothing — replace with a no-op bool */
        free_ast(cond);
        free_ast(node->as.if_stmt.then_branch);
        node->type        = AST_BOOL;
        node->as.bool_val = false;
        g_elim_count++;
        note("if(false) with no else → removed");
        return;
    }

    /* Free what we don't need */
    free_ast(cond);
    if (dropped) free_ast(dropped);

    /* Move kept's content into node, then free the (now empty) kept pointer */
    memcpy(node, kept, sizeof(ASTNode));
    free(kept);   /* shallow free: children now owned by *node */

    g_elim_count++;
    note(cond_val ? "if(true)  → kept then-branch"
                  : "if(false) → kept else-branch");
}

/* ── Dead let elimination from a block ───────────────────────────────────── */
/*
 * Walk block->statements[].  For each AST_LET binding that:
 *   - has a name (not a pattern-style let)
 *   - whose value has no observable side effects
 *   - is referenced 0 times in all *subsequent* statements of the same block
 * → free it and compact the array.
 *
 * We do NOT look across function boundaries.
 */
static void elim_dead_lets(ASTNode *block) {
    ASTNode **stmts = block->as.block.statements;
    int count       = block->as.block.count;

    int i = 0;
    while (i < count) {
        ASTNode *s = stmts[i];
        if (s->type == AST_LET && s->as.let.name && !has_side_effect(s->as.let.value)) {
            const char *vname = s->as.let.name;
            /* Count refs in statements i+1 .. count-1 */
            int refs = 0;
            for (int j = i + 1; j < count; j++)
                refs += count_refs(stmts[j], vname);

            if (refs == 0) {
                note("removed dead let binding");
                free_ast(s);
                /* Compact: shift left */
                memmove(&stmts[i], &stmts[i + 1],
                        (size_t)(count - i - 1) * sizeof(ASTNode *));
                count--;
                block->as.block.count = count;
                g_elim_count++;
                /* Don't advance i — recheck same slot */
                continue;
            }
        }
        i++;
    }
}

/* ── Recursive walker ────────────────────────────────────────────────────── */

static void dce_walk_arr(ASTNode **arr, int count) {
    for (int i = 0; i < count; i++) {
        if (arr[i]) dce_walk(arr[i]);
    }
}

static void dce_walk(ASTNode *node) {
    if (!node) return;

    switch (node->type) {
        /* Leaves */
        case AST_NUMBER: case AST_FLOAT: case AST_STRING: case AST_BOOL:
        case AST_IDENTIFIER: case AST_BREAK: case AST_CONTINUE:
        case AST_IMPORT: case AST_MODULE_DECL: case AST_OPAQUE_TYPE:
        case AST_ENUM_DEF: case AST_STRUCT_DEF: case AST_UNION_DEF:
        case AST_QUALIFIED_NAME:
            return;

        case AST_IF:
            /* Recurse first so nested folds happen bottom-up */
            dce_walk(node->as.if_stmt.condition);
            dce_walk(node->as.if_stmt.then_branch);
            dce_walk(node->as.if_stmt.else_branch);
            /* Now check if condition became a bool literal */
            if (node->type == AST_IF)   /* guard: might have been rewritten */
                try_elim_if(node);
            return;

        case AST_BLOCK:
            dce_walk_arr(node->as.block.statements, node->as.block.count);
            elim_dead_lets(node);
            return;

        case AST_PROGRAM:
            dce_walk_arr(node->as.program.items, node->as.program.count);
            return;

        case AST_FUNCTION:
            dce_walk(node->as.function.body);
            return;

        case AST_LET:
            dce_walk(node->as.let.value);
            return;

        case AST_SET:
            dce_walk(node->as.set.value);
            return;

        case AST_RETURN:
            dce_walk(node->as.return_stmt.value);
            return;

        case AST_WHILE:
            dce_walk(node->as.while_stmt.condition);
            dce_walk(node->as.while_stmt.body);
            return;

        case AST_FOR:
            dce_walk(node->as.for_stmt.range_expr);
            dce_walk(node->as.for_stmt.body);
            return;

        case AST_CALL:
            if (node->as.call.func_expr) dce_walk(node->as.call.func_expr);
            dce_walk_arr(node->as.call.args, node->as.call.arg_count);
            return;

        case AST_MODULE_QUALIFIED_CALL:
            dce_walk_arr(node->as.module_qualified_call.args,
                         node->as.module_qualified_call.arg_count);
            return;

        case AST_PRINT:
            dce_walk(node->as.print.expr);
            return;

        case AST_ASSERT:
            dce_walk(node->as.assert.condition);
            return;

        case AST_PREFIX_OP:
            dce_walk_arr(node->as.prefix_op.args, node->as.prefix_op.arg_count);
            return;

        case AST_COND:
            dce_walk_arr(node->as.cond_expr.conditions, node->as.cond_expr.clause_count);
            dce_walk_arr(node->as.cond_expr.values,     node->as.cond_expr.clause_count);
            dce_walk(node->as.cond_expr.else_value);
            return;

        case AST_STRUCT_LITERAL:
            dce_walk_arr(node->as.struct_literal.field_values,
                         node->as.struct_literal.field_count);
            return;

        case AST_ARRAY_LITERAL:
            dce_walk_arr(node->as.array_literal.elements,
                         node->as.array_literal.element_count);
            return;

        case AST_TUPLE_LITERAL:
            dce_walk_arr(node->as.tuple_literal.elements,
                         node->as.tuple_literal.element_count);
            return;

        case AST_UNION_CONSTRUCT:
            dce_walk_arr(node->as.union_construct.field_values,
                         node->as.union_construct.field_count);
            return;

        case AST_MATCH:
            dce_walk(node->as.match_expr.expr);
            dce_walk_arr(node->as.match_expr.arm_bodies,  node->as.match_expr.arm_count);
            dce_walk_arr(node->as.match_expr.guard_exprs, node->as.match_expr.arm_count);
            return;

        case AST_FIELD_ACCESS:
            dce_walk(node->as.field_access.object);
            return;

        case AST_TUPLE_INDEX:
            dce_walk(node->as.tuple_index.tuple);
            return;

        case AST_TRY_OP:
            dce_walk(node->as.try_op.operand);
            return;

        case AST_SHADOW:
            dce_walk(node->as.shadow.body);
            return;

        case AST_UNSAFE_BLOCK:
            dce_walk_arr(node->as.unsafe_block.statements, node->as.unsafe_block.count);
            return;

        case AST_PAR_BLOCK:
            dce_walk_arr(node->as.par_block.bindings, node->as.par_block.count);
            return;

        case AST_EFFECT_DECL:
        case AST_EFFECT_HANDLER:
        case AST_EFFECT_OP:
            return;

        default:
            return;
    }
}

/* ── Public API ──────────────────────────────────────────────────────────── */

int dce_pass(ASTNode *program, bool verbose) {
    g_elim_count = 0;
    g_verbose    = verbose;
    dce_walk(program);
    return g_elim_count;
}
