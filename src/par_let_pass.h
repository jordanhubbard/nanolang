/*
 * par_let_pass.h — AST pass for par-let concurrent binding validation and annotation
 *
 * Validates that par-let bindings have no cross-dependencies, emits a warning
 * for single-binding par-let expressions, and annotates each binding with an
 * independence flag for downstream codegen hints.
 */

#ifndef PAR_LET_PASS_H
#define PAR_LET_PASS_H

#include "nanolang.h"

/*
 * Run the par-let pass over the entire program AST.
 *
 * For each AST_PAR_LET node:
 *   - Emits a warning if count == 1 (suggest regular let).
 *   - Validates that no binding's RHS references a sibling binding name.
 *   - Allocates and populates node->as.par_let.independent[i]:
 *       true  — binding is independent of all siblings (always true if no error)
 *       false — dependency detected (error reported, flag still set for recovery)
 *
 * Returns the number of dependency errors found (0 = clean).
 */
int par_let_pass(ASTNode *program);

#endif /* PAR_LET_PASS_H */
