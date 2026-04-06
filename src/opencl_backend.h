/*
 * opencl_backend.h — nanolang OpenCL C kernel emit backend
 *
 * Translates `gpu fn` functions to OpenCL C kernel source (.cl).
 * The emitted .cl file is JIT-compiled at runtime by the OpenCL driver
 * via clCreateProgramWithSource / clBuildProgram.
 *
 * Key differences from ptx_backend:
 *   - Parameters are `__global char*` (byte-pointer) for device-buffer args
 *     and `long` for scalar args; pointer params are detected by static analysis
 *     (which params feed gpu_load / gpu_store).
 *   - Temporaries are named _t<n> (long) / _f<n> (double) instead of PTX registers.
 *   - Thread indexing uses get_global_id() / get_local_id() / get_group_id().
 *   - gpu_load/gpu_store emit pointer-cast derefs:
 *       *((__global long*)((ulong)ptr_expr))
 *   - Targets OpenCL C 1.2; works with POCL for CPU execution.
 *
 * Usage:
 *   nanoc --target opencl input.nano [-o output.cl]
 *
 * The emitted .cl file is used by opencl_runtime.c at launch time.
 * Pair with --target ptx for CUDA machines; opencl_runtime.c falls back
 * to .cl when no CUDA GPU is found (replacing .ptx extension with .cl).
 */
#pragma once
#ifndef OPENCL_BACKEND_H
#define OPENCL_BACKEND_H

#include "nanolang.h"
#include <stdio.h>
#include <stdbool.h>

/* Emit OpenCL C source to a file.
 * Returns 0 on success, non-zero on error. */
int ocl_backend_emit(ASTNode *root, const char *output_path,
                     const char *source_file, bool verbose);

/* Emit OpenCL C source to an already-open FILE*. */
int ocl_backend_emit_fp(ASTNode *root, FILE *out,
                        const char *source_file, bool verbose);

#endif /* OPENCL_BACKEND_H */
