/*
 * wasm_simd.c — WASM SIMD128 vectorization pass for nanolang
 *
 * Pattern detection:
 *   1. map(fn, array_literal)   — elementwise: f64x2 / i64x2
 *   2. (+ a b), (* a b), ...   — binary ops on arrays
 *   3. reduce(fn, array)        — horizontal reduction with SIMD
 *
 * Emission is handled inline in wasm_backend.c; this module provides
 * pattern scanning and the SIMD opcode helper utilities.
 *
 * WASM SIMD128 spec:
 *   https://github.com/WebAssembly/simd/blob/master/proposals/simd/SIMD.md
 */

#define _POSIX_C_SOURCE 200809L
#include "wasm_simd.h"
#include "nanolang.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── Internal: opcode map from nanolang token to SIMD op ─────────────── */

static int token_to_f64x2_op(TokenType tok) {
    switch (tok) {
        case TOKEN_PLUS:     return SIMD_F64X2_ADD;
        case TOKEN_MINUS:    return SIMD_F64X2_SUB;
        case TOKEN_STAR: return SIMD_F64X2_MUL;
        case TOKEN_SLASH:    return SIMD_F64X2_DIV;
        default:             return -1;
    }
}

static int token_to_i64x2_op(TokenType tok) {
    switch (tok) {
        case TOKEN_PLUS:     return SIMD_I64X2_ADD;
        case TOKEN_MINUS:    return SIMD_I64X2_SUB;
        case TOKEN_STAR: return SIMD_I64X2_MUL;
        default:             return -1;
    }
}

/* Return true if all elements of an array literal are the same numeric type */
static bool array_is_uniform_float(ASTNode *arr) {
    if (!arr || arr->type != AST_ARRAY_LITERAL) return false;
    for (int i = 0; i < arr->as.array_literal.element_count; i++) {
        ASTNode *e = arr->as.array_literal.elements[i];
        if (!e) return false;
        if (e->type != AST_FLOAT && e->type != AST_NUMBER) return false;
        /* treat all as float if any is float */
        if (e->type == AST_FLOAT) return true;
    }
    return false;
}

static bool array_is_uniform_int(ASTNode *arr) {
    if (!arr || arr->type != AST_ARRAY_LITERAL) return false;
    for (int i = 0; i < arr->as.array_literal.element_count; i++) {
        ASTNode *e = arr->as.array_literal.elements[i];
        if (!e) return false;
        if (e->type != AST_NUMBER) return false;
    }
    return true;
}

/* Check if a call is map(fn, array) */
static bool is_map_call(ASTNode *node) {
    if (!node || node->type != AST_CALL) return false;
    return (node->as.call.name &&
            strcmp(node->as.call.name, "map") == 0 &&
            node->as.call.arg_count == 2);
}

/* Check if a call is reduce(fn, array) */
static bool is_reduce_call(ASTNode *node) {
    if (!node || node->type != AST_CALL) return false;
    return (node->as.call.name &&
            strcmp(node->as.call.name, "reduce") == 0 &&
            node->as.call.arg_count >= 2);
}

/* Check if a prefix-op is elementwise arith on array literals */
static bool is_elementwise_op(ASTNode *node) {
    if (!node || node->type != AST_PREFIX_OP) return false;
    if (node->as.prefix_op.arg_count < 2) return false;
    ASTNode *a = node->as.prefix_op.args[0];
    ASTNode *b = node->as.prefix_op.args[1];
    if (!a || !b) return false;
    return ((a->type == AST_ARRAY_LITERAL || b->type == AST_ARRAY_LITERAL) &&
            (a->type == AST_ARRAY_LITERAL || a->type == AST_IDENTIFIER) &&
            (b->type == AST_ARRAY_LITERAL || b->type == AST_IDENTIFIER));
}


typedef struct {
    VectCandidate *cands;
    int            count;
    int            cap;
} VectList;

static void vl_add(VectList *vl, VectCandidate c) {
    if (vl->count >= vl->cap) {
        vl->cap = vl->cap ? vl->cap * 2 : 8;
        vl->cands = realloc(vl->cands, vl->cap * sizeof(VectCandidate));
    }
    vl->cands[vl->count++] = c;
}


/* ── Function-level SIMD pattern detection ──────────────────────────── *
 *
 * Detects functions that implement SIMD-friendly patterns:
 *   - 4+ parallel multiplications: (* a0 b0), (* a1 b1), ...
 *   - Horizontal sum tree: (+ (+ a b) (+ c d))
 *   - Parameter name patterns: a0,a1,a2,a3 or x0,x1,x2,x3
 * These are prime candidates for i64x2 / f64x2 vectorization.
 * ──────────────────────────────────────────────────────────────────── */

/* Count multiplications in an expression tree */
static int count_muls(ASTNode *node) {
    if (!node) return 0;
    int n = 0;
    if (node->type == AST_PREFIX_OP) {
        if (node->as.prefix_op.op == TOKEN_STAR) n++;
        for (int i = 0; i < node->as.prefix_op.arg_count; i++)
            n += count_muls(node->as.prefix_op.args[i]);
    }
    if (node->type == AST_BLOCK)
        for (int i = 0; i < node->as.block.count; i++)
            n += count_muls(node->as.block.statements[i]);
    if (node->type == AST_LET) n += count_muls(node->as.let.value);
    if (node->type == AST_RETURN) n += count_muls(node->as.return_stmt.value);
    return n;
}

/* Check if function params follow the a0,a1,a2,a3 or x0..x3 naming pattern */
static bool params_are_vectorlike(ASTNode *fn) {
    if (!fn || fn->type != AST_FUNCTION) return false;
    int pc = fn->as.function.param_count;
    if (pc < 4) return false;
    /* Check if any param has a digit suffix */
    for (int i = 0; i < pc && i < 8; i++) {
        const char *pname = fn->as.function.params[i].name;
        if (!pname) continue;
        size_t len = strlen(pname);
        if (len >= 2 && pname[len-1] >= '0' && pname[len-1] <= '9')
            return true;
    }
    return false;
}

/* Scan function nodes for SIMD-friendly patterns */
static void scan_function_simd(ASTNode *fn, VectList *vl) {
    if (!fn || fn->type != AST_FUNCTION) return;
    int muls = count_muls(fn->as.function.body);
    bool vec_params = params_are_vectorlike(fn);
    if (muls >= 2 && vec_params) {
        bool is_float = (fn->as.function.return_type == TYPE_FLOAT);
        VectCandidate c;
        c.pattern   = VECT_ELEMENTWISE;
        c.node      = fn;
        c.elem_type = is_float ? TYPE_FLOAT : TYPE_INT;
        c.simd_op   = is_float ? SIMD_F64X2_MUL : SIMD_I64X2_MUL;
        c.is_float  = is_float;
        vl_add(vl, c);
    }
}

/* ── Recursive walker: find vectorizable nodes ───────────────────────── */


static void scan_function_simd(ASTNode *fn, VectList *vl);
static void scan_node(ASTNode *node, VectList *vl) {
    if (!node) return;

    /* ── map(fn, float_array) ──────────────────────────────────────────── */
    if (is_map_call(node)) {
        ASTNode *arr = node->as.call.args[0];
        if (arr && arr->type == AST_ARRAY_LITERAL &&
                arr->as.array_literal.element_count >= 2) {
            bool is_float = array_is_uniform_float(arr);
            bool is_int   = !is_float && array_is_uniform_int(arr);
            if (is_float || is_int) {
                VectCandidate c;
                c.pattern   = is_float ? VECT_MAP_FLOAT : VECT_MAP_INT;
                c.node      = node;
                c.elem_type = is_float ? TYPE_FLOAT : TYPE_INT;
                c.simd_op   = is_float ? SIMD_F64X2_MUL : SIMD_I64X2_MUL;
                c.is_float  = is_float;
                vl_add(vl, c);
            }
        }
    }

    /* ── reduce(fn, float_array) ───────────────────────────────────────── */
    if (is_reduce_call(node)) {
        ASTNode *arr = node->as.call.args[0];
        if (arr && arr->type == AST_ARRAY_LITERAL &&
                arr->as.array_literal.element_count >= 4) {
            bool is_float = array_is_uniform_float(arr);
            bool is_int   = !is_float && array_is_uniform_int(arr);
            if (is_float || is_int) {
                VectCandidate c;
                c.pattern   = is_float ? VECT_REDUCE_FLOAT : VECT_REDUCE_INT;
                c.node      = node;
                c.elem_type = is_float ? TYPE_FLOAT : TYPE_INT;
                c.simd_op   = is_float ? SIMD_F64X2_ADD : SIMD_I64X2_ADD;
                c.is_float  = is_float;
                vl_add(vl, c);
            }
        }
    }

    /* ── elementwise (+ a b) on arrays ────────────────────────────────── */
    if (is_elementwise_op(node)) {
        ASTNode *a = node->as.prefix_op.args[0];
        ASTNode *b = node->as.prefix_op.args[1];
        bool arr_a = (a && a->type == AST_ARRAY_LITERAL);
        bool arr_b = (b && b->type == AST_ARRAY_LITERAL);
        ASTNode *arr = arr_a ? a : (arr_b ? b : NULL);
        if (arr) {
            bool is_float = array_is_uniform_float(arr);
            bool is_int   = !is_float && array_is_uniform_int(arr);
            int simd_op = -1;
            if (is_float) {
                simd_op = token_to_f64x2_op(node->as.prefix_op.op);
            } else if (is_int) {
                simd_op = token_to_i64x2_op(node->as.prefix_op.op);
            }
            if (simd_op >= 0) {
                VectCandidate c;
                c.pattern   = VECT_ELEMENTWISE;
                c.node      = node;
                c.elem_type = is_float ? TYPE_FLOAT : TYPE_INT;
                c.simd_op   = simd_op;
                c.is_float  = is_float;
                vl_add(vl, c);
            }
        }
    }

    /* ── Recurse into children ─────────────────────────────────────────── */
    switch (node->type) {
        case AST_PROGRAM:
            for (int i = 0; i < node->as.program.count; i++)
                scan_node(node->as.program.items[i], vl);
            break;
        case AST_FUNCTION:
            scan_function_simd(node, vl);
            scan_node(node->as.function.body, vl);
            break;
        case AST_BLOCK:
            for (int i = 0; i < node->as.block.count; i++)
                scan_node(node->as.block.statements[i], vl);
            break;
        case AST_CALL:
            for (int i = 0; i < node->as.call.arg_count; i++)
                scan_node(node->as.call.args[i], vl);
            break;
        case AST_PREFIX_OP:
            for (int i = 0; i < node->as.prefix_op.arg_count; i++)
                scan_node(node->as.prefix_op.args[i], vl);
            break;
        case AST_LET:
            scan_node(node->as.let.value, vl);
            break;
        case AST_IF:
            scan_node(node->as.if_stmt.condition, vl);
            scan_node(node->as.if_stmt.then_branch, vl);
            scan_node(node->as.if_stmt.else_branch, vl);
            break;
        case AST_WHILE:
            scan_node(node->as.while_stmt.condition, vl);
            scan_node(node->as.while_stmt.body, vl);
            break;
        case AST_RETURN:
            scan_node(node->as.return_stmt.value, vl);
            break;
        default:
            break;
    }
}

/* ── Public: scan AST for vectorizable candidates ────────────────────── */

VectCandidate *wasm_simd_detect(ASTNode *program, int *out_count) {
    VectList vl = {NULL, 0, 0};
    scan_node(program, &vl);
    *out_count = vl.count;
    return vl.cands;
}

/* ── Public: print summary of candidates ────────────────────────────── */

void wasm_simd_print_summary(VectCandidate *cands, int count, FILE *out) {
    if (count == 0) {
        fprintf(out, "[wasm-simd] No vectorizable patterns found\n");
        return;
    }
    static const char *pnames[] = {
        "map(float)", "map(int)", "reduce(float)", "reduce(int)", "elementwise"
    };
    fprintf(out, "[wasm-simd] %d vectorizable site(s) detected:\n", count);
    for (int i = 0; i < count; i++) {
        const char *pname = (cands[i].pattern < 5) ? pnames[cands[i].pattern] : "?";
        const char *lane  = cands[i].is_float ? "f64x2" : "i64x2";
        fprintf(out, "  [%d] %s → %s opcode=0xFD+%d (line %d)\n",
                i+1, pname, lane,
                cands[i].simd_op,
                cands[i].node ? cands[i].node->line : 0);
    }
}

/* ── Public: free candidates ─────────────────────────────────────────── */

void wasm_simd_free(VectCandidate *cands) {
    free(cands);
}
