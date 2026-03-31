/*
 * par_let_pass.c — AST pass for par-block binding validation (stub)
 *
 * The original par-let AST node was merged into AST_PAR_BLOCK.
 * This pass is currently a no-op; validation happens in the typechecker.
 */
#include "par_let_pass.h"
#include "nanolang.h"

int par_let_pass(ASTNode *program) {
    (void)program;
    return 0; /* success, no errors */
}
