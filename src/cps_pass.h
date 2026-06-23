/*
 * cps_pass.h — nanolang async/await CPS transform pass
 *
 * Transforms async function bodies into continuation-passing style:
 *   - async fn becomes a regular fn that returns a Future/Promise value
 *   - await expr becomes a yield-then-resume with the CPS continuation
 *   - The interpreter (eval.c) handles cooperative scheduling
 */
#pragma once
#include "nanolang.h"

/* Run the CPS transformation pass over the entire program AST.
 * Rewrites AST_ASYNC_FN and AST_AWAIT nodes into CPS form.
 * Returns number of async functions transformed.
 */
int cps_pass(ASTNode *program);

/* Verbose version for diagnostics */
int cps_pass_run(ASTNode *program, bool verbose);
