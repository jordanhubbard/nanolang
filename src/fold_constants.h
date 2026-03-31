/*
 * fold_constants.h — Compile-time constant folding pass for nanolang.
 *
 * Walks the AST after parse(), before typecheck().
 * Folds binary/unary ops whose operands are all literal integers, floats,
 * or booleans.  Nodes are mutated in-place; parents need no pointer updates.
 *
 * Supported folds:
 *   Arithmetic (+, -, *, /, %)  on int/int or float/float literals
 *   Comparisons (==, !=, <, >, <=, >=) on numeric or bool literals → bool
 *   Boolean   (&& and ||) on bool literals (short-circuit fold)
 *   Unary minus  on int/float literal → negated literal
 *   Unary not    on bool literal → bool
 *
 * Side-effect-free only: variable references stop propagation.
 */
#ifndef FOLD_CONSTANTS_H
#define FOLD_CONSTANTS_H

#include "nanolang.h"

/*
 * fold_constants() — run the constant-folding pass on *program*.
 *
 * verbose: if true, emit a debug note to stderr for each fold.
 * Returns the number of nodes folded (0 if nothing changed).
 */
int fold_constants(ASTNode *program, bool verbose);

#endif /* FOLD_CONSTANTS_H */
