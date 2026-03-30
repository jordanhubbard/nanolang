/*
 * wasm_backend.h — nanolang WASM binary emit backend
 *
 * Emits WebAssembly binary format (.wasm) from the nanolang AST.
 * Supports a pure-function subset: numeric types, arithmetic, comparisons,
 * function definitions, and basic control flow.
 *
 * Usage:
 *   nanoc --target wasm input.nano [-o output.wasm]
 *
 * WASM binary format spec: https://webassembly.github.io/spec/core/binary/
 */

#pragma once
#ifndef WASM_BACKEND_H
#define WASM_BACKEND_H

#include "nanolang.h"
#include <stdio.h>
#include <stdbool.h>

/* Emit a WASM binary file from the given AST root.
 * source_file:    path to the .nano input file (used for source map "sources" field)
 * sourcemap_path: path for the .wasm.map output; NULL = suppress source map
 * Returns 0 on success, non-zero on error. */
int wasm_backend_emit(ASTNode *root, const char *output_path,
                      const char *source_file, const char *sourcemap_path,
                      bool verbose);

/* Write to an already-open FILE* (no source map support). */
int wasm_backend_emit_fp(ASTNode *root, FILE *out, bool verbose);

/* Write to an already-open FILE* with optional source map support.
 * wasm_path and source_file are needed only when sourcemap_path != NULL. */
int wasm_backend_emit_fp_ex(ASTNode *root, FILE *out, bool verbose,
                             const char *wasm_path,
                             const char *source_file,
                             const char *sourcemap_path);

#endif /* WASM_BACKEND_H */
