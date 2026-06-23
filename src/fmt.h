/*
 * fmt.h — nano-fmt: canonical code formatter for .nano source files
 *
 * Reformats nanolang source to canonical style:
 *   - 4-space indentation (configurable)
 *   - Spaces around binary operators
 *   - No trailing whitespace
 *   - Single blank line between top-level declarations
 *   - Trailing newline at end of file
 *   - (let name: type = value) let bindings: normalized spacing
 *   - Consistent spacing in function signatures and calls
 *
 * Usage:
 *   nano-fmt file.nano          — print formatted output to stdout
 *   nano-fmt --write file.nano  — reformat in place
 *   nano-fmt --check file.nano  — exit 1 if file would change
 *
 * The formatter is token-based (like gofmt): it does not use the AST,
 * so comments are preserved and output is deterministic.
 *
 * LSP integration: textDocument/formatting calls fmt_source().
 */
#pragma once
#ifndef FMT_H
#define FMT_H

#include "nanolang.h"
#include <stdio.h>
#include <stdbool.h>

/* Format options */
typedef struct {
    int  indent_size;     /* Spaces per indent level (default: 4) */
    bool write_in_place;  /* --write: overwrite the source file */
    bool check_only;      /* --check: exit 1 if output != input */
    bool verbose;
} FmtOptions;

/*
 * Format a nanolang source string.
 * Returns a heap-allocated formatted string (caller must free()),
 * or NULL on parse error.
 */
char *fmt_source(const char *source, const char *filename, const FmtOptions *opts);

/*
 * Format a file.
 * Returns 0 on success, 1 on error, 2 if --check and file would change.
 */
int fmt_file(const char *path, const FmtOptions *opts);

#endif /* FMT_H */
