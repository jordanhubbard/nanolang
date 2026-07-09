/*
 * dce_pass.h — Dead Code Elimination pass for nanolang.
 *
 * Walks the AST after typecheck(), before codegen/transpile.
 *
 * Eliminations performed:
 *   1. if(true)  A else B  → A  (dead else branch removed)
 *   2. if(false) A else B  → B  (dead then branch removed)
 *   3. let x = <pure-expr> where x is never referenced in its scope → removed
 *      Bindings with side effects (print, assert, any call) are kept even if
 *      the bound name is unused.
 *
 * Nodes are mutated in-place using memcpy so parent pointers are unchanged.
 * Block statement arrays are compacted when dead lets are removed.
 */
#ifndef DCE_PASS_H
#define DCE_PASS_H

#include "nanolang.h"

/*
 * dce_pass() — run the dead-code elimination pass on *program*.
 *
 * verbose: if true, emit a debug note to stderr for each elimination.
 * Returns the number of eliminations performed.
 */
int dce_pass(ASTNode *program, bool verbose);

#endif /* DCE_PASS_H */
