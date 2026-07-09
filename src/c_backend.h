/*
 * c_backend.h — nanolang → readable C transpiler (seL4 / bare-metal target)
 *
 * Emits clean, portable C99 from the nanolang AST. Designed for embedding
 * in agentOS Protection Domain (PD) code on seL4/Microkit without any
 * WASM interpreter, LLVM toolchain, or GC runtime dependency.
 *
 * Type mappings
 * ─────────────
 *   nano int    → int64_t
 *   nano float  → double
 *   nano bool   → bool
 *   nano string → const char *  (static strings; no heap allocation)
 *   nano void   → void
 *   nano struct → C struct (typedef'd)
 *
 * Control flow
 * ────────────
 *   if/else    → C if/else
 *   while      → C while
 *   return     → C return
 *   let        → local variable declaration
 *
 * Effects (algebraic / error handling)
 * ─────────────────────────────────────
 *   raise      → longjmp (via NL_EFFECT_RAISE macro)
 *   handle     → setjmp  (via NL_EFFECT_HANDLE macro)
 *   A minimal effects.h header is emitted in the output preamble.
 *
 * Closures
 * ────────
 *   Anonymous functions / lambdas are emitted as static helper functions
 *   with a captured environment struct passed as an extra void* parameter.
 *
 * Usage
 * ─────
 *   nanoc --target c input.nano -o output.c
 *   cc -std=c99 -I. output.c -o output
 *
 * For seL4 / Microkit:
 *   nanoc --target c --no-stdlib input.nano -o pd_nano.c
 *   # Include pd_nano.c in your PD build; no external runtime required.
 */
#pragma once
#ifndef C_BACKEND_H
#define C_BACKEND_H

#include "nanolang.h"
#include <stdio.h>
#include <stdbool.h>

/* Options for C backend emission */
typedef struct {
    bool no_stdlib;      /* Omit #include <stdio.h> etc. (for seL4 bare-metal) */
    bool no_main;        /* Omit the main() entry point wrapper */
    bool static_strings; /* Emit string literals as static const char* (default) */
    bool verbose;
} CBOptions;

/* Emit C source to a file path.
 * Returns 0 on success, non-zero on error.
 */
int c_backend_emit(ASTNode *root, const char *output_path,
                   const char *source_file, const CBOptions *opts);

/* Emit C source to an already-open FILE*. */
int c_backend_emit_fp(ASTNode *root, FILE *out,
                      const char *source_file, const CBOptions *opts);

#endif /* C_BACKEND_H */
