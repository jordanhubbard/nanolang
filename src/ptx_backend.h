/*
 * ptx_backend.h — nanolang PTX assembly emit backend
 *
 * Emits NVIDIA PTX assembly (.ptx) from the nanolang AST.
 * Supports functions annotated with `gpu fn`: these become PTX kernels.
 *
 * Supported subset:
 *   - Types: int (.s64 / .s32), float (.f64), bool (.pred)
 *   - Arithmetic: + - * / %
 *   - Comparisons: == != < <= > >=
 *   - Control flow: if/else, return
 *   - Variables: let
 *   - Thread indexing builtins: (thread_id_x), (block_id_x), (block_dim_x),
 *                               (thread_id_y), (block_id_y), (thread_id_z)
 *
 * Usage:
 *   nanoc --target ptx input.nano [-o output.ptx]
 *
 * The emitted .ptx file can be:
 *   1. Compiled with: nvcc -ptx output.ptx -o output.ptxo
 *   2. JIT-loaded via CUDA driver API: cuModuleLoadData() + cuModuleGetFunction()
 *
 * PTX ISA reference: https://docs.nvidia.com/cuda/parallel-thread-execution/
 */
#pragma once
#ifndef PTX_BACKEND_H
#define PTX_BACKEND_H

#include "nanolang.h"
#include <stdio.h>
#include <stdbool.h>

/* Emit PTX text to a file.
 * Returns 0 on success, non-zero on error. */
int ptx_backend_emit(ASTNode *root, const char *output_path,
                     const char *source_file, bool verbose);

/* Emit PTX to an already-open FILE*. */
int ptx_backend_emit_fp(ASTNode *root, FILE *out,
                        const char *source_file, bool verbose);

#endif /* PTX_BACKEND_H */
