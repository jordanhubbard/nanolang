#ifndef NANOCORE_EXPORT_H
#define NANOCORE_EXPORT_H

#include "nanolang.h"

/* I export an AST node as a heap-allocated NanoCore S-expression.
 * I return NULL for unsupported nodes or allocation/formatting failure.
 * My caller owns and frees a successful result. */
char *nanocore_export_sexpr(ASTNode *node, Environment *env);

/* Run the Coq-extracted reference interpreter on an S-expression.
 * Returns a heap-allocated result string, or NULL on error.
 * Requires 'nanocore-ref' binary to be in PATH or adjacent to the compiler. */
char *nanocore_reference_eval(const char *sexpr, const char *compiler_path);

#endif /* NANOCORE_EXPORT_H */
