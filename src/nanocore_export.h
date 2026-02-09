#ifndef NANOCORE_EXPORT_H
#define NANOCORE_EXPORT_H

#include "nanolang.h"

/* Export a NanoLang AST node to NanoCore S-expression format.
 * Returns a heap-allocated string, or NULL if the node is outside the NanoCore subset.
 * The caller must free the returned string. */
char *nanocore_export_sexpr(ASTNode *node, Environment *env);

/* Run the Coq-extracted reference interpreter on an S-expression.
 * Returns a heap-allocated result string, or NULL on error.
 * Requires 'nanocore-ref' binary to be in PATH or adjacent to the compiler. */
char *nanocore_reference_eval(const char *sexpr, const char *compiler_path);

#endif /* NANOCORE_EXPORT_H */
