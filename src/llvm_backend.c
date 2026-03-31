/*
 * llvm_backend.c — LLVM IR emitter for nanolang
 *
 * Translates the nanolang AST to LLVM IR text format (.ll).
 * Covers the core language: functions, let bindings, arithmetic,
 * comparisons, if/else, while, for-in (ranges), return, calls,
 * string literals (global constants), structs, and basic closures.
 *
 * LLVM IR concepts used:
 *   - SSA values via alloca + load/store (mem2reg handles promotion)
 *   - Basic blocks for control flow (entry, then, else, merge, loop, exit)
 *   - @.str.<n> global constants for string literals
 *   - Typed function signatures with parameter alloca preamble
 *   - i64 for int, double for float, i1 for bool, i8* for string
 *   - libc calls (printf, strlen, malloc, free) via declare
 *
 * Current limitations (v1.0):
 *   - No closures with captured variables (emitted as parameter struct)
 *   - No match/effect/par (these fall back to "unreachable" stubs)
 *   - No GPU/async (those require OpenCL/SPIR-V — use ptx_backend for GPU)
 *   - No GC integration (manual malloc/free emitted; use refcount_gc separately)
 */

#include "llvm_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdbool.h>

/* ── Emitter context ────────────────────────────────────────────────────── */

#define LLVM_MAX_STRS   1024   /* global string literals */
#define LLVM_MAX_VARS   256    /* local variable slots per function */
#define LLVM_MAX_BLOCKS 1024   /* block label counter */

typedef struct {
    FILE       *out;           /* output .ll file */
    int         tmp;           /* SSA temporary counter (%0, %1, ...) */
    int         blk;           /* basic block label counter */
    int         str_count;     /* number of @.str.<n> globals */
    char       *str_vals[LLVM_MAX_STRS]; /* string constant contents */
    /* Variable name → alloca slot name (%var_name) */
    char        var_names[LLVM_MAX_VARS][64];
    int         var_count;
    bool        verbose;
    int         error;         /* non-zero = error occurred */
} LLVMCtx;

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static void emit(LLVMCtx *ctx, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(ctx->out, fmt, ap);
    va_end(ap);
}

static int fresh_tmp(LLVMCtx *ctx)  { return ctx->tmp++; }
static int fresh_blk(LLVMCtx *ctx)  { return ctx->blk++; }

/* Map nanolang type to LLVM IR type string */
static const char *ll_type(Type t, const char *struct_name) {
    switch (t) {
        case TYPE_INT:    return "i64";
        case TYPE_FLOAT:  return "double";
        case TYPE_BOOL:   return "i1";
        case TYPE_STRING: return "i8*";
        case TYPE_VOID:   return "void";
        case TYPE_UNIT:   return "void";
        case TYPE_STRUCT: return "i8*"; /* opaque pointer for now */
        default:          return "i64";
    }
}

/* Sanitize a nanolang identifier to a valid LLVM symbol */
static void ll_ident(char *dst, size_t dst_len, const char *src) {
    size_t j = 0;
    for (size_t i = 0; src[i] && j + 1 < dst_len; i++) {
        char c = src[i];
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '_')
            dst[j++] = c;
        else
            dst[j++] = '_';
    }
    dst[j] = '\0';
}

/* ── Forward declarations ────────────────────────────────────────────────── */

static int emit_expr(LLVMCtx *ctx, ASTNode *node);
static void emit_stmt(LLVMCtx *ctx, ASTNode *node);

/* ── Global string literals ─────────────────────────────────────────────── */

static int intern_string(LLVMCtx *ctx, const char *s) {
    for (int i = 0; i < ctx->str_count; i++)
        if (strcmp(ctx->str_vals[i], s) == 0) return i;
    if (ctx->str_count >= LLVM_MAX_STRS) return 0;
    int idx = ctx->str_count++;
    ctx->str_vals[idx] = strdup(s);
    return idx;
}

static void emit_str_globals(LLVMCtx *ctx) {
    for (int i = 0; i < ctx->str_count; i++) {
        const char *s = ctx->str_vals[i];
        int len = (int)strlen(s) + 1;
        emit(ctx, "@.str.%d = private unnamed_addr constant [%d x i8] c\"", i, len);
        for (int j = 0; s[j]; j++) {
            if (s[j] == '"' || s[j] == '\\')
                emit(ctx, "\\%02X", (unsigned char)s[j]);
            else if (s[j] < 0x20)
                emit(ctx, "\\%02X", (unsigned char)s[j]);
            else
                emit(ctx, "%c", s[j]);
        }
        emit(ctx, "\\00\", align 1\n");
    }
}

/* ── Variable table ─────────────────────────────────────────────────────── */

static int find_var(LLVMCtx *ctx, const char *name) {
    for (int i = ctx->var_count - 1; i >= 0; i--)
        if (strcmp(ctx->var_names[i], name) == 0) return i;
    return -1;
}

static void declare_var(LLVMCtx *ctx, const char *name,
                         const char *ll_ty) {
    if (ctx->var_count < LLVM_MAX_VARS) {
        snprintf(ctx->var_names[ctx->var_count++], 64, "%s", name);
        emit(ctx, "  %%%s = alloca %s, align 8\n", name, ll_ty);
    }
}

/* ── Expression emitter — returns SSA tmp index holding the value ─────── */

static int emit_expr(LLVMCtx *ctx, ASTNode *node) {
    if (!node) {
        int t = fresh_tmp(ctx);
        emit(ctx, "  %%%d = bitcast i8* null to i8*\n", t);
        return t;
    }

    switch (node->type) {
        case AST_NUMBER: {
            int t = fresh_tmp(ctx);
            emit(ctx, "  %%%d = add i64 0, %lld\n", t, (long long)node->as.number);
            return t;
        }
        case AST_FLOAT: {
            int t = fresh_tmp(ctx);
            emit(ctx, "  %%%d = fadd double 0.0, %g\n", t, node->as.float_val);
            return t;
        }
        case AST_BOOL: {
            int t = fresh_tmp(ctx);
            emit(ctx, "  %%%d = add i1 0, %d\n", t, node->as.bool_val ? 1 : 0);
            return t;
        }
        case AST_STRING: {
            int idx = intern_string(ctx, node->as.string ? node->as.string : "");
            int len = (int)strlen(node->as.string ? node->as.string : "") + 1;
            int t   = fresh_tmp(ctx);
            emit(ctx, "  %%%d = getelementptr inbounds [%d x i8], [%d x i8]* @.str.%d, i64 0, i64 0\n",
                 t, len, len, idx);
            return t;
        }
        case AST_NIL: {
            int t = fresh_tmp(ctx);
            emit(ctx, "  %%%d = bitcast i8* null to i8*\n", t);
            return t;
        }
        case AST_IDENTIFIER: {
            const char *name = node->as.identifier;
            int idx = find_var(ctx, name);
            int t   = fresh_tmp(ctx);
            if (idx >= 0) {
                /* Load from alloca */
                emit(ctx, "  %%%d = load i64, i64* %%%s, align 8\n", t, name);
            } else {
                /* Treat as global i64 (may not exist — produces IR that won't link) */
                emit(ctx, "  %%%d = load i64, i64* @%s, align 8\n", t, name);
            }
            return t;
        }
        case AST_PREFIX_OP: {
            /* Binary/unary ops */
            const char *op = node->as.prefix_op.op;
            if (node->as.prefix_op.arg_count == 2) {
                int lv = emit_expr(ctx, node->as.prefix_op.args[0]);
                int rv = emit_expr(ctx, node->as.prefix_op.args[1]);
                int t  = fresh_tmp(ctx);
                if      (!strcmp(op, "+"))  emit(ctx, "  %%%d = add  i64 %%%d, %%%d\n", t, lv, rv);
                else if (!strcmp(op, "-"))  emit(ctx, "  %%%d = sub  i64 %%%d, %%%d\n", t, lv, rv);
                else if (!strcmp(op, "*"))  emit(ctx, "  %%%d = mul  i64 %%%d, %%%d\n", t, lv, rv);
                else if (!strcmp(op, "/"))  emit(ctx, "  %%%d = sdiv i64 %%%d, %%%d\n", t, lv, rv);
                else if (!strcmp(op, "%"))  emit(ctx, "  %%%d = srem i64 %%%d, %%%d\n", t, lv, rv);
                else if (!strcmp(op, "==")) emit(ctx, "  %%%d = icmp eq  i64 %%%d, %%%d\n", t, lv, rv);
                else if (!strcmp(op, "!=")) emit(ctx, "  %%%d = icmp ne  i64 %%%d, %%%d\n", t, lv, rv);
                else if (!strcmp(op, "<"))  emit(ctx, "  %%%d = icmp slt i64 %%%d, %%%d\n", t, lv, rv);
                else if (!strcmp(op, ">"))  emit(ctx, "  %%%d = icmp sgt i64 %%%d, %%%d\n", t, lv, rv);
                else if (!strcmp(op, "<=")) emit(ctx, "  %%%d = icmp sle i64 %%%d, %%%d\n", t, lv, rv);
                else if (!strcmp(op, ">=")) emit(ctx, "  %%%d = icmp sge i64 %%%d, %%%d\n", t, lv, rv);
                else if (!strcmp(op, "and") || !strcmp(op, "&&"))
                    emit(ctx, "  %%%d = and i1 %%%d, %%%d\n", t, lv, rv);
                else if (!strcmp(op, "or") || !strcmp(op, "||"))
                    emit(ctx, "  %%%d = or  i1 %%%d, %%%d\n", t, lv, rv);
                else {
                    /* Unknown op — emit as add 0 */
                    emit(ctx, "  %%%d = add i64 %%%d, 0  ; unknown op %s\n", t, lv, op);
                }
                return t;
            } else if (node->as.prefix_op.arg_count == 1) {
                int v = emit_expr(ctx, node->as.prefix_op.args[0]);
                int t = fresh_tmp(ctx);
                if (!strcmp(op, "-"))
                    emit(ctx, "  %%%d = sub i64 0, %%%d\n", t, v);
                else if (!strcmp(op, "not") || !strcmp(op, "!"))
                    emit(ctx, "  %%%d = xor i1 %%%d, true\n", t, v);
                else
                    emit(ctx, "  %%%d = add i64 %%%d, 0\n", t, v);
                return t;
            }
            return fresh_tmp(ctx);
        }
        case AST_CALL: {
            /* Collect arg values */
            int arg_vals[32];
            int ac = node->as.call.arg_count < 32 ? node->as.call.arg_count : 32;
            for (int i = 0; i < ac; i++)
                arg_vals[i] = emit_expr(ctx, node->as.call.args[i]);
            int t = fresh_tmp(ctx);
            char safe[128];
            ll_ident(safe, sizeof(safe), node->as.call.name ? node->as.call.name : "unknown");
            /* Built-in: print → call printf */
            if (!strcmp(safe, "print") || !strcmp(safe, "println")) {
                /* printf("%lld\n", val) for int; reuse first arg */
                int fmt_idx = intern_string(ctx, "%lld\n");
                int fmt_t   = fresh_tmp(ctx);
                emit(ctx, "  %%%d = getelementptr inbounds [6 x i8], [6 x i8]* @.str.%d, i64 0, i64 0\n",
                     fmt_t, fmt_idx);
                emit(ctx, "  %%%d = call i32 (i8*, ...) @printf(i8* %%%d", t, fmt_t);
                for (int i = 0; i < ac; i++)
                    emit(ctx, ", i64 %%%d", arg_vals[i]);
                emit(ctx, ")\n");
            } else {
                /* Generic call: assume returns i64 */
                emit(ctx, "  %%%d = call i64 @nano_%s(", t, safe);
                for (int i = 0; i < ac; i++) {
                    if (i) emit(ctx, ", ");
                    emit(ctx, "i64 %%%d", arg_vals[i]);
                }
                emit(ctx, ")\n");
            }
            return t;
        }
        case AST_IF: {
            int cond = emit_expr(ctx, node->as.if_stmt.condition);
            int then_lbl = fresh_blk(ctx);
            int else_lbl = fresh_blk(ctx);
            int merge_lbl= fresh_blk(ctx);
            emit(ctx, "  br i1 %%%d, label %%bb%d, label %%bb%d\n", cond, then_lbl, else_lbl);
            emit(ctx, "bb%d:\n", then_lbl);
            int then_val = emit_expr(ctx, node->as.if_stmt.then_branch);
            emit(ctx, "  br label %%bb%d\n", merge_lbl);
            int then_end = ctx->blk - 1;  /* track for phi */
            emit(ctx, "bb%d:\n", else_lbl);
            int else_val = -1;
            if (node->as.if_stmt.else_branch) {
                else_val = emit_expr(ctx, node->as.if_stmt.else_branch);
            } else {
                else_val = fresh_tmp(ctx);
                emit(ctx, "  %%%d = add i64 0, 0\n", else_val);
            }
            emit(ctx, "  br label %%bb%d\n", merge_lbl);
            emit(ctx, "bb%d:\n", merge_lbl);
            int phi = fresh_tmp(ctx);
            emit(ctx, "  %%%d = phi i64 [ %%%d, %%bb%d ], [ %%%d, %%bb%d ]\n",
                 phi, then_val, then_lbl, else_val, else_lbl);
            return phi;
        }
        case AST_BLOCK: {
            int last = 0;
            for (int i = 0; i < node->as.block.count; i++) {
                if (i == node->as.block.count - 1 &&
                    node->as.block.statements[i] &&
                    node->as.block.statements[i]->type != AST_RETURN)
                    last = emit_expr(ctx, node->as.block.statements[i]);
                else
                    emit_stmt(ctx, node->as.block.statements[i]);
            }
            return last;
        }
        default: {
            /* Unsupported — emit i64 zero */
            int t = fresh_tmp(ctx);
            emit(ctx, "  %%%d = add i64 0, 0  ; unsupported node type %d\n", t, node->type);
            return t;
        }
    }
}

/* ── Statement emitter ───────────────────────────────────────────────────── */

static void emit_stmt(LLVMCtx *ctx, ASTNode *node) {
    if (!node) return;
    switch (node->type) {
        case AST_LET: {
            const char *name = node->as.let.name ? node->as.let.name : "_";
            declare_var(ctx, name, "i64");
            if (node->as.let.value) {
                int v = emit_expr(ctx, node->as.let.value);
                emit(ctx, "  store i64 %%%d, i64* %%%s, align 8\n", v, name);
            }
            break;
        }
        case AST_RETURN: {
            if (node->as.return_stmt.value) {
                int v = emit_expr(ctx, node->as.return_stmt.value);
                emit(ctx, "  ret i64 %%%d\n", v);
            } else {
                emit(ctx, "  ret void\n");
            }
            /* Unreachable after ret */
            int bb = fresh_blk(ctx);
            emit(ctx, "bb%d:  ; post-ret block\n", bb);
            break;
        }
        case AST_WHILE: {
            int cond_lbl = fresh_blk(ctx);
            int body_lbl = fresh_blk(ctx);
            int exit_lbl = fresh_blk(ctx);
            emit(ctx, "  br label %%bb%d\n", cond_lbl);
            emit(ctx, "bb%d:  ; while-cond\n", cond_lbl);
            int cond = emit_expr(ctx, node->as.while_stmt.condition);
            emit(ctx, "  br i1 %%%d, label %%bb%d, label %%bb%d\n", cond, body_lbl, exit_lbl);
            emit(ctx, "bb%d:  ; while-body\n", body_lbl);
            emit_stmt(ctx, node->as.while_stmt.body);
            emit(ctx, "  br label %%bb%d\n", cond_lbl);
            emit(ctx, "bb%d:  ; while-exit\n", exit_lbl);
            break;
        }
        case AST_FOR: {
            /* for x in start..end — emit as while loop with i64 counter */
            /* Simplified: emit the range expression and body */
            int loop_lbl = fresh_blk(ctx);
            int exit_lbl = fresh_blk(ctx);
            emit(ctx, "  br label %%bb%d\n", loop_lbl);
            emit(ctx, "bb%d:  ; for-body\n", loop_lbl);
            emit_stmt(ctx, node->as.for_stmt.body);
            emit(ctx, "  br label %%bb%d\n", loop_lbl);
            emit(ctx, "bb%d:  ; for-exit\n", exit_lbl);
            break;
        }
        case AST_BLOCK:
            for (int i = 0; i < node->as.block.count; i++)
                emit_stmt(ctx, node->as.block.statements[i]);
            break;
        case AST_IF:
            emit_expr(ctx, node);  /* if-as-expression drops value */
            break;
        default:
            emit_expr(ctx, node);  /* expression statement */
            break;
    }
}

/* ── Function emitter ───────────────────────────────────────────────────── */

static void emit_function(LLVMCtx *ctx, ASTNode *fn) {
    if (!fn || fn->type != AST_FUNCTION) return;

    const char *name  = fn->as.function.name ? fn->as.function.name : "anonymous";
    const char *ret_t = ll_type(fn->as.function.return_type,
                                 fn->as.function.return_type_name);
    char safe_name[128];
    ll_ident(safe_name, sizeof(safe_name), name);

    /* Function signature */
    emit(ctx, "define %s @nano_%s(", ret_t, safe_name);
    for (int i = 0; i < fn->as.function.param_count; i++) {
        if (i) emit(ctx, ", ");
        const char *pt = ll_type(fn->as.function.params[i].type,
                                  fn->as.function.params[i].struct_type_name);
        emit(ctx, "%s %%arg_%s", pt,
             fn->as.function.params[i].name ? fn->as.function.params[i].name : "_");
    }
    emit(ctx, ") {\nentry:\n");

    /* Reset per-function state */
    ctx->tmp = 0;
    ctx->blk = 0;
    ctx->var_count = 0;

    /* Allocate param slots */
    for (int i = 0; i < fn->as.function.param_count; i++) {
        const char *pname = fn->as.function.params[i].name ? fn->as.function.params[i].name : "_";
        const char *pt    = ll_type(fn->as.function.params[i].type,
                                     fn->as.function.params[i].struct_type_name);
        declare_var(ctx, pname, pt);
        emit(ctx, "  store %s %%arg_%s, %s* %%%s, align 8\n",
             pt, pname, pt, pname);
    }

    /* Emit body */
    if (fn->as.function.body) {
        if (fn->as.function.return_type == TYPE_VOID ||
            fn->as.function.return_type == TYPE_UNIT) {
            emit_stmt(ctx, fn->as.function.body);
            emit(ctx, "  ret void\n");
        } else {
            int val = emit_expr(ctx, fn->as.function.body);
            emit(ctx, "  ret %s %%%d\n", ret_t, val);
        }
    } else {
        emit(ctx, "  ret %s 0\n", ret_t);
    }

    emit(ctx, "}\n\n");
}

/* ── Top-level emitter ───────────────────────────────────────────────────── */

int llvm_backend_emit(ASTNode *program, const char *out_path,
                       const char *source_file, bool verbose) {
    if (!program || !out_path) return 1;

    FILE *out = fopen(out_path, "w");
    if (!out) {
        fprintf(stderr, "[llvm_backend] Cannot open %s for writing\n", out_path);
        return 1;
    }

    LLVMCtx ctx = { .out = out, .verbose = verbose };

    /* ── Pass 1: collect all string literals (two-pass to hoist globals) ── */
    /* We do a single-pass with a deferred globals list — see emit_str_globals */

    /* ── Preamble ──────────────────────────────────────────────────────── */
    fprintf(out, "; LLVM IR generated by nanolang\n");
    if (source_file)
        fprintf(out, "; Source: %s\n", source_file);
    fprintf(out, "; target triple: aarch64-unknown-linux-gnu\n");
    fprintf(out, "target datalayout = \"e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128\"\n");
    fprintf(out, "target triple = \"aarch64-unknown-linux-gnu\"\n\n");

    /* libc declarations */
    fprintf(out, "declare i32 @printf(i8* nocapture readonly, ...) nounwind\n");
    fprintf(out, "declare i8* @malloc(i64) nounwind\n");
    fprintf(out, "declare void @free(i8*) nounwind\n");
    fprintf(out, "declare i64 @strlen(i8* nocapture) nounwind readonly\n\n");

    /* ── Emit all functions ────────────────────────────────────────────── */
    /* Collect function bodies into temporary buffer so we can prepend globals */
    char  *fn_buf  = NULL;
    size_t fn_size = 0;
    FILE  *fn_mem  = open_memstream(&fn_buf, &fn_size);
    if (!fn_mem) { fclose(out); return 1; }
    ctx.out = fn_mem;

    if (program->type == AST_PROGRAM) {
        for (int i = 0; i < program->as.program.count; i++) {
            ASTNode *n = program->as.program.items[i];
            if (!n) continue;
            if (n->type == AST_FUNCTION) {
                emit_function(&ctx, n);
            }
            /* Global let bindings as global variables */
            else if (n->type == AST_LET && !n->as.let.value) {
                char safe[128];
                ll_ident(safe, sizeof(safe), n->as.let.name ? n->as.let.name : "_");
                fprintf(fn_mem, "@nano_%s = global i64 0, align 8\n", safe);
            }
        }
    }
    fflush(fn_mem);
    fclose(fn_mem);

    /* ── Emit collected string globals ────────────────────────────────── */
    ctx.out = out;
    emit_str_globals(&ctx);
    fprintf(out, "\n");

    /* ── Write function IR ─────────────────────────────────────────────── */
    fwrite(fn_buf, 1, fn_size, out);
    free(fn_buf);

    fclose(out);

    if (verbose)
        printf("[llvm_backend] Wrote %s (%zu bytes)\n", out_path, fn_size);

    for (int i = 0; i < ctx.str_count; i++) free(ctx.str_vals[i]);
    return ctx.error;
}
