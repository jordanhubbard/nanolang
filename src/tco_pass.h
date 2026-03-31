/*
 * tco_pass.h — nanolang tail-call optimization pass
 *
 * Detects syntactic tail-position recursive calls in function bodies and
 * rewrites them to loop-based iteration in the AST before code-gen.
 *
 * A call is in tail position if its result is directly returned (no
 * further computation after the call). This pass handles:
 *   - Direct tail recursion:  fn fact(n, acc) => if n == 0 then acc else fact(n-1, n*acc)
 *   - Tail calls in if/else branches
 *   - Tail calls in block last-statement position
 *
 * After the pass, tail-recursive functions gain a TCO body that uses
 * AST_WHILE + AST_SET nodes instead of recursion.
 */
#pragma once
#ifndef TCO_PASS_H
#define TCO_PASS_H

#include "nanolang.h"
#include <stdbool.h>

/*
 * Run the TCO pass over an entire program (AST_PROGRAM node).
 * Modifies the AST in-place: tail-recursive functions are rewritten to
 * iterative form using while loops and parameter reassignment.
 *
 * Returns the number of functions transformed.
 */
int tco_pass_run(ASTNode *program, bool verbose);

/*
 * Convenience wrapper (non-verbose) — called from main.c as tco_pass(program).
 */
void tco_pass(ASTNode *program);

#endif /* TCO_PASS_H */
