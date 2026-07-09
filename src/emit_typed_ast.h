#ifndef EMIT_TYPED_AST_H
#define EMIT_TYPED_AST_H

#include "nanolang.h"

/* Emit the full typed AST as JSON to stdout.
 * Called after typechecking completes successfully.
 * Produces a machine-readable representation useful for LLM introspection,
 * IDE tooling, documentation generators, and refactoring tools.
 */
void emit_typed_ast_json(const char *input_file, ASTNode *program, Environment *env);

#endif /* EMIT_TYPED_AST_H */
