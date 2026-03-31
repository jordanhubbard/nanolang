/*
 * c_backend.h — nanolang C source emit backend
 *
 * Emits readable, self-contained C source from the nanolang AST.
 * The resulting .c file can be compiled with any C99/C11-compatible compiler:
 *   gcc -o prog prog.c
 *   clang -o prog prog.c
 *
 * Type mapping:
 *   int      → int64_t
 *   float    → double
 *   bool     → int
 *   string   → const char*
 *   struct   → typedef struct { ... } NanoStruct_Name;
 *   effects  → setjmp/longjmp stubs
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
 * c_backend_emit — generate C source from a nanolang AST.
 *
 * root        — parsed + type-checked AST
 * output_path — output .c file path
 * source_file — original .nano path (used in comments)
 * verbose     — print progress to stderr
 *
 * Returns 0 on success, non-zero on error.
 */
int c_backend_emit(ASTNode *root, const char *output_path,
                   const char *source_file, bool verbose);

/* Write to an already-open FILE*. source_file may be NULL. */
int c_backend_emit_fp(ASTNode *root, FILE *out, const char *source_file,
                      bool verbose);

#endif /* C_BACKEND_H */
