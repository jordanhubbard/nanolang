/*
 * wasm_simd.h — WASM SIMD128 auto-vectorization for nanolang
 *
 * Detects numeric array reduction patterns in the nanolang AST and emits
 * WASM SIMD128 v128 opcodes instead of scalar f64/i64 ops.
 *
 * WASM SIMD128 (WebAssembly 2.0 v128 proposal, §5.4):
 *   Prefix opcode: 0xFD (simd_prefix)
 *   Lane operations: v128.load/store, f64x2.add/sub/mul, i32x4.add/mul, etc.
 *
 * Vectorization patterns detected:
 *   1. Elementwise loop:  for i in 0..n: c[i] = a[i] op b[i]
 *      → f64x2.load × 2, f64x2.add/sub/mul/div, f64x2.store
 *      → 2× throughput on 2-lane f64x2, 4× on i32x4
 *
 *   2. Reduction (fold):  let sum = 0.0; for i in 0..n: sum += a[i]
 *      → f64x2.load, f64x2.add to accumulator, horizontal add at end
 *
 *   3. Map (scalar fn over array elements):
 *      → if fn is {neg, abs, sqrt}: emit corresponding v128 op
 *
 * Usage: call wasm_simd_try_vectorize() from emit_stmt() on AST_FOR nodes.
 * Returns true and emits SIMD opcodes if the pattern matches; caller should
 * skip the scalar fallback. Returns false if pattern doesn't match (scalar).
 *
 * Copyright 2026 nanolang Project (MIT)
 */

#pragma once
#ifndef WASM_SIMD_H
#define WASM_SIMD_H

#include "nanolang.h"
#include <stdbool.h>
#include <stdint.h>

/* ── WASM SIMD opcode prefix and immediate opcodes ─────────────────── */

/* Prefix byte for all SIMD instructions */
#define SIMD_PREFIX     0xFD

/* SIMD immediate opcodes (u32 LEB128 after 0xFD prefix) */
#define SIMD_V128_LOAD          0x00  /* v128.load memarg */
#define SIMD_V128_STORE         0x0B  /* v128.store memarg */
#define SIMD_I8X16_SPLAT        0x0F
#define SIMD_I32X4_SPLAT        0x11
#define SIMD_F64X2_SPLAT        0x13
#define SIMD_I32X4_ADD          0xAE  /* i32x4.add */
#define SIMD_I32X4_SUB          0xAF
#define SIMD_I32X4_MUL          0xB5
#define SIMD_F64X2_ADD          0xF0  /* f64x2.add */
#define SIMD_F64X2_SUB          0xF1
#define SIMD_F64X2_MUL          0xF2
#define SIMD_F64X2_DIV          0xF3
#define SIMD_F64X2_ABS          0xEC
#define SIMD_F64X2_NEG          0xED
#define SIMD_F64X2_SQRT         0xEF
#define SIMD_F64X2_EXTRACT_LANE 0x1F  /* f64x2.extract_lane imm */
#define SIMD_F64X2_REPLACE_LANE 0x21

/* ── Vectorization context (per-loop analysis) ──────────────────────── */

typedef enum {
    SIMD_PATTERN_NONE        = 0,
    SIMD_PATTERN_ELEMENTWISE = 1,  /* c[i] = a[i] op b[i] */
    SIMD_PATTERN_REDUCTION   = 2,  /* acc += a[i]          */
    SIMD_PATTERN_MAP         = 3,  /* c[i] = fn(a[i])      */
} SimdPattern;

typedef struct {
    SimdPattern pattern;
    Type        elem_type;      /* TYPE_FLOAT or TYPE_INT */
    int         lanes;          /* 2 for f64x2, 4 for i32x4 */
    const char *arr_a;          /* source array local name */
    const char *arr_b;          /* second source (elementwise only) */
    const char *arr_out;        /* output array local name */
    const char *acc_var;        /* accumulator variable (reduction) */
    const char *map_fn;         /* function name (map only) */
    TokenType   op;             /* arithmetic operator */
} SimdInfo;

/* ── API ─────────────────────────────────────────────────────────────── */

/*
 * Analyse a for-loop AST node and determine if it can be vectorized.
 * Returns true and fills *info if the loop matches a SIMD pattern.
 */
bool wasm_simd_analyze(ASTNode *for_node, SimdInfo *info);

/*
 * Emit SIMD opcodes for a vectorizable for-loop.
 * buf: the WasmBuf to write into.
 * info: analysis result from wasm_simd_analyze().
 * array_base_local: local index for the array base pointer.
 * Returns 0 on success, -1 on error.
 *
 * NOTE: This function emits the SIMD instructions into a raw byte buffer.
 * Caller is responsible for providing the correct local variable indices.
 */
int wasm_simd_emit_loop(void *wasm_buf, const SimdInfo *info,
                         uint32_t idx_local, uint32_t len_local,
                         uint32_t acc_local);

/*
 * Check if the WASM runtime target supports SIMD128.
 * Returns true for WebAssembly 2.0 targets (V8 ≥9.0, wasmtime ≥0.26, etc.)
 */
bool wasm_simd_supported(const char *target);

#endif /* WASM_SIMD_H */
