/*
 * wasm_simd.h — WASM SIMD128 vectorization for nanolang
 *
 * Detects vectorizable patterns in the nanolang AST and emits
 * WASM SIMD128 (v128) opcodes for numeric array operations.
 *
 * Supported patterns:
 *   - map(fn, array)  where fn is a pure elementwise numeric function
 *   - elementwise arithmetic: (+ a b) where a,b are Float/Int arrays
 *   - reduce/fold with commutative ops (+, *, min, max)
 *
 * WASM SIMD128 spec: https://github.com/WebAssembly/simd/blob/master/proposals/simd/SIMD.md
 * All SIMD opcodes use the 0xFD prefix followed by a u32 LEB128 opcode.
 */

#pragma once
#ifndef WASM_SIMD_H
#define WASM_SIMD_H

#include "nanolang.h"
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

/* ── WASM SIMD128 type ────────────────────────────────────────────────── */
#define WASM_V128      0x7B  /* v128 value type */

/* ── SIMD prefix byte ─────────────────────────────────────────────────── */
#define WASM_SIMD_PREFIX 0xFD

/* ── SIMD128 opcodes (after 0xFD prefix, LEB128-encoded) ──────────────── */

/* v128 memory ops */
#define SIMD_V128_LOAD            0   /* v128.load         memarg */
#define SIMD_V128_LOAD8X8_S       1   /* v128.load8x8_s    memarg */
#define SIMD_V128_LOAD8X8_U       2
#define SIMD_V128_LOAD16X4_S      3
#define SIMD_V128_LOAD16X4_U      4
#define SIMD_V128_LOAD32X2_S      5
#define SIMD_V128_LOAD32X2_U      6
#define SIMD_V128_LOAD8_SPLAT     7
#define SIMD_V128_LOAD16_SPLAT    8
#define SIMD_V128_LOAD32_SPLAT    9
#define SIMD_V128_LOAD64_SPLAT    10
#define SIMD_V128_STORE           11  /* v128.store        memarg */

/* v128 constants */
#define SIMD_V128_CONST           12  /* v128.const        i8x16 immediate */

/* Shuffle/swizzle */
#define SIMD_I8X16_SHUFFLE        13
#define SIMD_I8X16_SWIZZLE        14

/* Splat (broadcast scalar → v128) */
#define SIMD_I8X16_SPLAT          15
#define SIMD_I16X8_SPLAT          16
#define SIMD_I32X4_SPLAT          17
#define SIMD_I64X2_SPLAT          18  /* i64x2.splat  i64 → v128 */
#define SIMD_F32X4_SPLAT          19
#define SIMD_F64X2_SPLAT          20  /* f64x2.splat  f64 → v128 */

/* Lane extract/replace */
#define SIMD_I64X2_EXTRACT_LANE   19  /* i64x2.extract_lane laneimm */
#define SIMD_F64X2_EXTRACT_LANE   32  /* f64x2.extract_lane laneimm */

/* f64x2 arithmetic */
#define SIMD_F64X2_ABS            236 /* 0xEC */
#define SIMD_F64X2_NEG            237
#define SIMD_F64X2_SQRT           239
#define SIMD_F64X2_ADD            240 /* 0xF0 */
#define SIMD_F64X2_SUB            241
#define SIMD_F64X2_MUL            242
#define SIMD_F64X2_DIV            243
#define SIMD_F64X2_MIN            244
#define SIMD_F64X2_MAX            245
#define SIMD_F64X2_PMIN           246
#define SIMD_F64X2_PMAX           247

/* i64x2 arithmetic */
#define SIMD_I64X2_NEG            193
#define SIMD_I64X2_ALL_TRUE       195
#define SIMD_I64X2_EQ             214
#define SIMD_I64X2_NE             215
#define SIMD_I64X2_LT_S           216
#define SIMD_I64X2_GT_S           217
#define SIMD_I64X2_LE_S           218
#define SIMD_I64X2_GE_S           219
#define SIMD_I64X2_ADD            197 /* 0xC5 */
#define SIMD_I64X2_SUB            198
#define SIMD_I64X2_MUL            221

/* v128 bitwise */
#define SIMD_V128_AND             78
#define SIMD_V128_ANDNOT          79
#define SIMD_V128_OR              80
#define SIMD_V128_XOR             81
#define SIMD_V128_NOT             77

/* ── Vectorization context ────────────────────────────────────────────── */

/* A vectorizable loop/map/fold pattern detected in the AST */
typedef enum {
    VECT_MAP_FLOAT,   /* map(fn, float_array)  → f64x2 lanes */
    VECT_MAP_INT,     /* map(fn, int_array)    → i64x2 lanes */
    VECT_REDUCE_FLOAT,/* reduce(op, float_array) with +,* */
    VECT_REDUCE_INT,  /* reduce(op, int_array)   with +,* */
    VECT_ELEMENTWISE, /* (+ a b) where a,b are arrays */
} VectPattern;

typedef struct {
    VectPattern pattern;
    ASTNode    *node;         /* the original AST node */
    int         elem_type;    /* TYPE_INT or TYPE_FLOAT */
    int         simd_op;      /* SIMD opcode for the inner op */
    bool        is_float;
} VectCandidate;

/* ── Public API ───────────────────────────────────────────────────────── */

/* Scan program AST for vectorizable patterns.
 * Returns a malloc'd array of VectCandidate (count in *out_count).
 * Caller must free() the returned array. */
VectCandidate *wasm_simd_detect(ASTNode *program, int *out_count);

/* Emit WASM SIMD code for a single candidate into a WasmBuf.
 * Used by wasm_backend.c when --simd is enabled.
 * buf:    byte buffer being written
 * cand:   vectorization candidate
 * Returns 0 on success, -1 on error. */
int wasm_simd_emit_candidate(void *buf /*WasmBuf*/, VectCandidate *cand);

/* Print a summary of detected candidates to stderr (for --verbose). */
void wasm_simd_print_summary(VectCandidate *cands, int count, FILE *out);

/* Free candidates array */
void wasm_simd_free(VectCandidate *cands);

#endif /* WASM_SIMD_H */
