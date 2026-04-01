/*
 * par_let_pass.c — AST pass for par-let concurrent binding validation and annotation
 *
 * This pass walks the AST looking for AST_PAR_LET nodes and:
 *   1. Warns when a par-let has only a single binding (suggest regular let).
 *   2. Validates that no binding's RHS references a sibling binding name
 *      (which would create a data dependency incompatible with parallel evaluation).
 *   3. Allocates and populates the independent[] flag array on each par-let node
 *      so that downstream codegen passes can annotate or schedule accordingly.
 *
 * Dependency detection uses a simple recursive name-scan over the RHS expression.
 * This is conservative (it may flag shadowed names), which is safe for a concurrent
 * evaluation primitive — false positives fall back to sequential let.
 */

#include "par_let_pass.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* ── Name reference scan ───────────────────────────────────────────────── */

/* Returns true if the AST subtree rooted at node contains a free reference to name. */
static bool expr_refs_name(ASTNode *node, const char *name) {
    if (!node || !name) return false;
    switch (node->type) {
        case AST_IDENTIFIER:
            return strcmp(node->as.identifier, name) == 0;
        case AST_PREFIX_OP:
            for (int i = 0; i < node->as.prefix_op.arg_count; i++)
                if (expr_refs_name(node->as.prefix_op.args[i], name)) return true;
            return false;
        case AST_CALL:
            for (int i = 0; i < node->as.call.arg_count; i++)
                if (expr_refs_name(node->as.call.args[i], name)) return true;
            return false;
        case AST_IF:
            return expr_refs_name(node->as.if_stmt.condition, name) ||
                   expr_refs_name(node->as.if_stmt.then_branch, name) ||
                   expr_refs_name(node->as.if_stmt.else_branch, name);
        case AST_BLOCK:
            for (int i = 0; i < node->as.block.count; i++)
                if (expr_refs_name(node->as.block.statements[i], name)) return true;
            return false;
        case AST_FIELD_ACCESS:
            return expr_refs_name(node->as.field_access.object, name);
        case AST_ARRAY_LITERAL:
            for (int i = 0; i < node->as.array_literal.element_count; i++)
                if (expr_refs_name(node->as.array_literal.elements[i], name)) return true;
            return false;
        case AST_LET:
            return expr_refs_name(node->as.let.value, name);
        case AST_RETURN:
            return expr_refs_name(node->as.return_stmt.value, name);
        case AST_PAR_LET:
            for (int i = 0; i < node->as.par_let.count; i++)
                if (expr_refs_name(node->as.par_let.values[i], name)) return true;
            return expr_refs_name(node->as.par_let.body, name);
        default:
            return false;
    }
}

/* ── Recursive AST walker ──────────────────────────────────────────────── */

static int pass_walk(ASTNode *node, int *errors) {
    if (!node) return 0;

    switch (node->type) {
        case AST_PAR_LET: {
            int n = node->as.par_let.count;

            /* Warn on single-binding par-let */
            if (n == 1) {
                fprintf(stderr,
                    "Warning at line %d, column %d: par-let with a single binding; "
                    "consider using regular let\n",
                    node->line, node->column);
            }

            /* Allocate independence flags (all true by default) */
            if (!node->as.par_let.independent) {
                node->as.par_let.independent = malloc(sizeof(bool) * (n > 0 ? n : 1));
                for (int i = 0; i < n; i++)
                    node->as.par_let.independent[i] = true;
            }

            /* Validate: no binding may reference a sibling binding name */
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (j == i) continue;
                    if (expr_refs_name(node->as.par_let.values[i],
                                       node->as.par_let.names[j])) {
                        fprintf(stderr,
                            "Error at line %d, column %d: par-let binding '%s' depends on "
                            "sibling binding '%s' — use sequential let for dependent bindings\n",
                            node->line, node->column,
                            node->as.par_let.names[i], node->as.par_let.names[j]);
                        node->as.par_let.independent[i] = false;
                        (*errors)++;
                    }
                }
            }

            /* Recurse into binding values and body */
            for (int i = 0; i < n; i++)
                pass_walk(node->as.par_let.values[i], errors);
            pass_walk(node->as.par_let.body, errors);
            break;
        }

        /* Recurse into compound nodes */
        case AST_PROGRAM:
            for (int i = 0; i < node->as.program.count; i++)
                pass_walk(node->as.program.items[i], errors);
            break;
        case AST_BLOCK:
            for (int i = 0; i < node->as.block.count; i++)
                pass_walk(node->as.block.statements[i], errors);
            break;
        case AST_FUNCTION:
            pass_walk(node->as.function.body, errors);
            break;
        case AST_IF:
            pass_walk(node->as.if_stmt.condition, errors);
            pass_walk(node->as.if_stmt.then_branch, errors);
            pass_walk(node->as.if_stmt.else_branch, errors);
            break;
        case AST_WHILE:
            pass_walk(node->as.while_stmt.condition, errors);
            pass_walk(node->as.while_stmt.body, errors);
            break;
        case AST_LET:
            pass_walk(node->as.let.value, errors);
            break;
        case AST_RETURN:
            pass_walk(node->as.return_stmt.value, errors);
            break;
        case AST_PAR_BLOCK:
            for (int i = 0; i < node->as.par_block.count; i++)
                pass_walk(node->as.par_block.bindings[i], errors);
            break;
        default:
            break;
    }
    return 0;
}

/* ── Public API ────────────────────────────────────────────────────────── */

int par_let_pass(ASTNode *program) {
    int errors = 0;
    pass_walk(program, &errors);
    return errors;
}
