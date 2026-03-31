/*
 * wasm_simd.c — WASM SIMD128 auto-vectorization for nanolang
 *
 * Pattern detection + SIMD opcode emission for numeric array loops.
 *
 * Copyright 2026 nanolang Project (MIT)
 */

#include "wasm_simd.h"
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

/* ── Tiny byte buffer helpers (mirrors wasm_backend.c WasmBuf) ───────── */

typedef struct {
    uint8_t *data;
    size_t   len, cap;
} SIMDBuf;

static void sbuf_byte(SIMDBuf *b, uint8_t v) {
    if (b->len >= b->cap) {
        b->cap = b->cap ? b->cap * 2 : 64;
        b->data = realloc(b->data, b->cap);
    }
    b->data[b->len++] = v;
}

static void sbuf_u32_leb(SIMDBuf *b, uint32_t v) {
    do {
        uint8_t byte = v & 0x7F;
        v >>= 7;
        if (v) byte |= 0x80;
        sbuf_byte(b, byte);
    } while (v);
}

/* Emit SIMD prefix + opcode */
static void sbuf_simd(SIMDBuf *b, uint32_t op) {
    sbuf_byte(b, SIMD_PREFIX);
    sbuf_u32_leb(b, op);
}

/* ── Pattern detection helpers ────────────────────────────────────────── */

/*
 * Detect: for i in range: arr_out[i] = arr_a[i] op arr_b[i]
 * We look for:
 *   AST_FOR with body = single AST_SET (or block with single set)
 *   The set's value is a binary op between two array index expressions.
 */
static bool detect_elementwise(ASTNode *for_node, SimdInfo *info) {
    if (!for_node || for_node->type != AST_FOR) return false;

    ASTNode *body = for_node->as.for_stmt.body;
    if (!body) return false;

    /* Unwrap single-statement block */
    ASTNode *stmt = body;
    if (body->type == AST_BLOCK && body->as.block.count == 1)
        stmt = body->as.block.statements[0];

    /* We expect: arr_out[i] = arr_a[i] op arr_b[i]
     * In current AST: AST_SET with name being an index expression,
     * or an AST_CALL to array_set.
     * Simplified: detect AST_SET where value is AST_PREFIX_OP (binary) */
    if (stmt->type != AST_SET) return false;

    ASTNode *val = stmt->as.set.value;
    if (!val || val->type != AST_PREFIX_OP) return false;
    if (val->as.prefix_op.arg_count != 2) return false;

    TokenType op = val->as.prefix_op.op;
    if (op != TOKEN_PLUS && op != TOKEN_MINUS &&
        op != TOKEN_STAR && op != TOKEN_SLASH) return false;

    /* Check args are identifier references (simplified: any identifier) */
    ASTNode *lhs = val->as.prefix_op.args[0];
    ASTNode *rhs = val->as.prefix_op.args[1];
    if (!lhs || !rhs) return false;

    info->pattern  = SIMD_PATTERN_ELEMENTWISE;
    info->op       = op;
    info->elem_type= TYPE_FLOAT; /* default: f64 */
    info->lanes    = 2;          /* f64x2: 2 lanes × 64-bit */
    info->arr_a    = (lhs->type == AST_IDENTIFIER) ? lhs->as.identifier : "a";
    info->arr_b    = (rhs->type == AST_IDENTIFIER) ? rhs->as.identifier : "b";
    info->arr_out  = stmt->as.set.name;
    info->acc_var  = NULL;
    info->map_fn   = NULL;
    return true;
}

/*
 * Detect: let acc = 0.0; for i in range: acc += arr[i]
 * We look for AST_FOR body = single AST_SET: acc = acc + arr[i]
 */
static bool detect_reduction(ASTNode *for_node, SimdInfo *info) {
    if (!for_node || for_node->type != AST_FOR) return false;

    ASTNode *body = for_node->as.for_stmt.body;
    ASTNode *stmt = body;
    if (body && body->type == AST_BLOCK && body->as.block.count == 1)
        stmt = body->as.block.statements[0];
    if (!stmt || stmt->type != AST_SET) return false;

    ASTNode *val = stmt->as.set.value;
    if (!val || val->type != AST_PREFIX_OP || val->as.prefix_op.arg_count != 2)
        return false;

    TokenType op = val->as.prefix_op.op;
    if (op != TOKEN_PLUS && op != TOKEN_STAR) return false;

    /* Check: one of the args is the accumulator variable itself */
    ASTNode *arg0 = val->as.prefix_op.args[0];
    ASTNode *arg1 = val->as.prefix_op.args[1];
    if (!arg0 || !arg1) return false;

    const char *acc = stmt->as.set.name;
    bool acc_in_arg = (arg0->type == AST_IDENTIFIER &&
                       strcmp(arg0->as.identifier, acc) == 0) ||
                      (arg1->type == AST_IDENTIFIER &&
                       strcmp(arg1->as.identifier, acc) == 0);
    if (!acc_in_arg) return false;

    ASTNode *arr_arg = (arg0->type == AST_IDENTIFIER &&
                        strcmp(arg0->as.identifier, acc) == 0) ? arg1 : arg0;

    info->pattern  = SIMD_PATTERN_REDUCTION;
    info->op       = op;
    info->elem_type= TYPE_FLOAT;
    info->lanes    = 2;
    info->arr_a    = (arr_arg->type == AST_IDENTIFIER) ? arr_arg->as.identifier : "a";
    info->arr_b    = NULL;
    info->arr_out  = NULL;
    info->acc_var  = acc;
    info->map_fn   = NULL;
    return true;
}

/*
 * Detect: for i in range: out[i] = fn(a[i])  where fn is neg/abs/sqrt
 */
static bool detect_map(ASTNode *for_node, SimdInfo *info) {
    if (!for_node || for_node->type != AST_FOR) return false;

    ASTNode *body = for_node->as.for_stmt.body;
    ASTNode *stmt = body;
    if (body && body->type == AST_BLOCK && body->as.block.count == 1)
        stmt = body->as.block.statements[0];
    if (!stmt || stmt->type != AST_SET) return false;

    ASTNode *val = stmt->as.set.value;
    if (!val || val->type != AST_CALL) return false;

    const char *fn = val->as.call.name;
    if (!fn) return false;
    if (strcmp(fn, "neg") != 0 && strcmp(fn, "abs") != 0 &&
        strcmp(fn, "sqrt") != 0 && strcmp(fn, "f64_neg") != 0)
        return false;

    info->pattern  = SIMD_PATTERN_MAP;
    info->op       = TOKEN_PLUS; /* unused */
    info->elem_type= TYPE_FLOAT;
    info->lanes    = 2;
    info->map_fn   = fn;
    info->arr_a    = (val->as.call.arg_count > 0 &&
                      val->as.call.args[0]->type == AST_IDENTIFIER)
                     ? val->as.call.args[0]->as.identifier : "a";
    info->arr_b    = NULL;
    info->arr_out  = stmt->as.set.name;
    info->acc_var  = NULL;
    return true;
}

/* ── Public API ───────────────────────────────────────────────────────── */

bool wasm_simd_analyze(ASTNode *for_node, SimdInfo *info) {
    if (!for_node || !info) return false;
    memset(info, 0, sizeof(SimdInfo));
    return detect_elementwise(for_node, info) ||
           detect_reduction(for_node, info)   ||
           detect_map(for_node, info);
}

int wasm_simd_emit_loop(void *wasm_buf, const SimdInfo *info,
                         uint32_t idx_local, uint32_t len_local,
                         uint32_t acc_local) {
    if (!wasm_buf || !info) return -1;
    SIMDBuf *b = (SIMDBuf *)wasm_buf;

    /*
     * Emit a SIMD-vectorized loop skeleton.
     * The exact offsets for local_get/local_set depend on how the caller
     * has laid out locals in the WASM function.  We emit the pattern as
     * a comment-annotated byte sequence; full integration requires the
     * caller to provide the correct local indices.
     *
     * Pattern (elementwise f64x2, 2 elements per iteration):
     *
     *   block
     *     loop
     *       ;; Check: idx + 2 <= len
     *       local.get $idx
     *       local.get $len
     *       i64.ge_s
     *       br_if 1  ;; exit loop
     *
     *       ;; Load 2 × f64 from a[idx] and b[idx]
     *       local.get $a_ptr
     *       local.get $idx
     *       i64.const 8
     *       i64.mul
     *       i64.add
     *       i32.wrap_i64  ;; WASM memory is i32 address
     *       v128.load 0:16  ;; align=16
     *
     *       ;; (same for b)
     *       ...
     *
     *       f64x2.add
     *
     *       ;; Store result to c[idx]
     *       ...
     *       v128.store 0:16
     *
     *       ;; idx += 2
     *       local.get $idx
     *       i64.const 2
     *       i64.add
     *       local.set $idx
     *       br 0
     *     end
     *   end
     */

    /* WASM block/loop opcodes */
    sbuf_byte(b, 0x02); /* block void */
    sbuf_byte(b, 0x40);
    sbuf_byte(b, 0x03); /* loop void */
    sbuf_byte(b, 0x40);

    /* Bounds check: idx >= len → break */
    sbuf_byte(b, 0x20); sbuf_u32_leb(b, idx_local); /* local.get idx */
    sbuf_byte(b, 0x20); sbuf_u32_leb(b, len_local); /* local.get len */
    sbuf_byte(b, 0x59);                              /* i64.ge_s */
    sbuf_byte(b, 0x0D); sbuf_byte(b, 0x01);          /* br_if 1 (block) */

    /* Load first vector (a[idx]) */
    sbuf_byte(b, 0x20); sbuf_u32_leb(b, idx_local);  /* local.get idx */
    sbuf_byte(b, 0x42); sbuf_byte(b, 0x08);           /* i64.const 8 */
    sbuf_byte(b, 0x7E);                               /* i64.mul */
    sbuf_byte(b, 0xA7);                               /* i32.wrap_i64 */
    sbuf_simd(b, SIMD_V128_LOAD);                     /* v128.load */
    sbuf_byte(b, 0x04); sbuf_byte(b, 0x00);           /* align=4, offset=0 */

    if (info->pattern == SIMD_PATTERN_ELEMENTWISE) {
        /* Load second vector (b[idx]) — same ptr logic */
        sbuf_byte(b, 0x20); sbuf_u32_leb(b, idx_local);
        sbuf_byte(b, 0x42); sbuf_byte(b, 0x08);
        sbuf_byte(b, 0x7E);
        sbuf_byte(b, 0xA7);
        sbuf_simd(b, SIMD_V128_LOAD);
        sbuf_byte(b, 0x04); sbuf_byte(b, 0x00);

        /* Apply f64x2 operation */
        uint32_t vec_op = SIMD_F64X2_ADD;
        switch (info->op) {
            case TOKEN_PLUS:  vec_op = SIMD_F64X2_ADD; break;
            case TOKEN_MINUS: vec_op = SIMD_F64X2_SUB; break;
            case TOKEN_STAR:  vec_op = SIMD_F64X2_MUL; break;
            case TOKEN_SLASH: vec_op = SIMD_F64X2_DIV; break;
            default:          vec_op = SIMD_F64X2_ADD; break;
        }
        sbuf_simd(b, vec_op);

        /* Store result (drop for now — full integration needs out ptr) */
        sbuf_byte(b, 0x1A); /* drop */

    } else if (info->pattern == SIMD_PATTERN_REDUCTION) {
        /* acc_v128 += v128.load(a[idx]) */
        sbuf_byte(b, 0x20); sbuf_u32_leb(b, acc_local); /* local.get acc */
        sbuf_simd(b, SIMD_F64X2_ADD);
        sbuf_byte(b, 0x21); sbuf_u32_leb(b, acc_local); /* local.set acc */

    } else if (info->pattern == SIMD_PATTERN_MAP) {
        uint32_t map_op = SIMD_F64X2_ABS;
        if (strcmp(info->map_fn, "neg") == 0 ||
            strcmp(info->map_fn, "f64_neg") == 0) map_op = SIMD_F64X2_NEG;
        else if (strcmp(info->map_fn, "sqrt") == 0) map_op = SIMD_F64X2_SQRT;
        sbuf_simd(b, map_op);
        sbuf_byte(b, 0x1A); /* drop */
    }

    /* idx += 2 (stride = 2 lanes) */
    sbuf_byte(b, 0x20); sbuf_u32_leb(b, idx_local); /* local.get idx */
    sbuf_byte(b, 0x42); sbuf_byte(b, 0x02);          /* i64.const 2 */
    sbuf_byte(b, 0x7C);                               /* i64.add */
    sbuf_byte(b, 0x21); sbuf_u32_leb(b, idx_local);  /* local.set idx */
    sbuf_byte(b, 0x0C); sbuf_byte(b, 0x00);           /* br 0 (loop) */

    sbuf_byte(b, 0x0B); /* end loop */
    sbuf_byte(b, 0x0B); /* end block */

    return 0;
}

bool wasm_simd_supported(const char *target) {
    /* SIMD128 is in WebAssembly 2.0 — supported by all modern runtimes */
    if (!target) return true;
    /* Explicitly disable for older embeddings */
    if (strcmp(target, "wasm1") == 0) return false;
    return true;
}
