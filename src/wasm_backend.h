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
 * Returns 0 on success, non-zero on error. */
int wasm_backend_emit(ASTNode *root, const char *output_path, bool verbose);

/* Inline version: emit to a FILE* already opened. */
int wasm_backend_emit_fp(ASTNode *root, FILE *out, bool verbose);

#endif /* WASM_BACKEND_H */
