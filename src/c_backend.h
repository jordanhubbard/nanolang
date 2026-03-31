/*
 * c_backend.h — nanolang C transpiler backend
 *
 * Emits readable, self-contained C source from the nanolang AST.
 * Designed for embedding nanolang logic in agentOS PD binaries on seL4
 * without a WASM interpreter or LLVM toolchain dependency.
 *
 * Type mapping:
 *   int      → int64_t
 *   float    → double
 *   bool     → bool
 *   string   → const char* (static strings only; no GC in PD context)
 *   record   → struct { ... } (row-polymorphic records become C structs)
 *   closure  → function pointer struct { fn_ptr fn; void *env; }
 *   effects  → setjmp/longjmp continuation passing
 *
 * Usage:
 *   nanoc --target c input.nano [-o output.c]
 *
 * Generated output compiles with: gcc -std=c11 output.c -o output
 */

#pragma once
#ifndef C_BACKEND_H
#define C_BACKEND_H

#include "nanolang.h"
#include <stdio.h>
#include <stdbool.h>

/*
 * Emit a C source file from the given AST root.
 *
 * output_path:  path to write the .c file (or NULL to write to stdout)
 * source_file:  path to the .nano input (used in comments/line directives)
 * verbose:      print progress messages to stderr
 *
 * Returns 0 on success, non-zero on error.
 */
int c_backend_emit(ASTNode *root, const char *output_path,
                   const char *source_file, bool verbose);

/*
 * Emit to an already-open FILE*.
 */
int c_backend_emit_fp(ASTNode *root, FILE *out, const char *source_file,
                      bool verbose);

#endif /* C_BACKEND_H */
