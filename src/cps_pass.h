/*
 * cps_pass.h — nanolang async/await CPS transform pass
 *
 * I walk selected async/await syntax and report selected context errors.
 * I do not construct continuations or Future/Promise values. My interpreter
 * evaluates scalar awaits transparently and delegates coroutine handles to
 * its run-to-completion scheduler. Resumable async execution remains work.
 */
#pragma once
#include "nanolang.h"

/* I walk the supported AST cases without rewriting async functions.
 * I return the number of async declarations visited, not a transform count.
 */
int cps_pass(ASTNode *program);

/* Verbose version for diagnostics */
int cps_pass_run(ASTNode *program, bool verbose);
