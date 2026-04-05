/*
 * ptx_backend.c — nanolang PTX assembly emit backend
 *
 * Translates `gpu fn` functions to NVIDIA PTX text format.
 * Kernels are emitted as .visible .entry; can be JIT-loaded via CUDA driver API.
 *
 * Register convention:
 *   %rd<n>  — .s64  (int / bool as 64-bit signed)
 *   %fd<n>  — .f64  (float)
 *   %p<n>   — .pred (comparison predicate)
 *
 * All ops use AST_PREFIX_OP with arg_count 1 (unary) or 2 (binary).
 * Token types determine the operation.
 */

#include "ptx_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>
#include <math.h>

/* ── String builder ─────────────────────────────────────────────────── */
typedef struct { char *buf; size_t len; size_t cap; } PtxSB;

static void psb_init(PtxSB *s) { s->buf = NULL; s->len = 0; s->cap = 0; }
static void psb_free(PtxSB *s) { free(s->buf); s->buf = NULL; s->len = s->cap = 0; }

static void psb_grow(PtxSB *s, size_t need) {
    if (s->len + need + 1 <= s->cap) return;
    size_t nc = s->cap ? s->cap * 2 : 4096;
    while (nc < s->len + need + 1) nc *= 2;
    s->buf = realloc(s->buf, nc);
    s->cap = nc;
}
static void psb_append(PtxSB *s, const char *str) {
    size_t n = strlen(str);
    psb_grow(s, n);
    memcpy(s->buf + s->len, str, n + 1);
    s->len += n;
}
static void psb_appendf(PtxSB *s, const char *fmt, ...) {
    char tmp[1024];
    va_list ap; va_start(ap, fmt);
    vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    psb_append(s, tmp);
}

/* ── Codegen context ────────────────────────────────────────────────── */
typedef enum { VK_INT, VK_FLOAT, VK_PRED } VK;

typedef struct {
    char  name[64];
    int   reg;
    VK    kind;
} VLocal;

#define MAX_LOCALS 128

typedef struct {
    VLocal  v[MAX_LOCALS];
    int     n;
} VLocals;

typedef struct {
    PtxSB    sb;
    int      next_reg;
    int      next_lbl;
    bool     verbose;
    const char *src;
    bool     err;
    char     errmsg[512];
} PCtx;

static void ptx_err(PCtx *ctx, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    vsnprintf(ctx->errmsg, sizeof(ctx->errmsg), fmt, ap);
    va_end(ap);
    ctx->err = true;
}

static int  new_reg(PCtx *ctx)  { return ctx->next_reg++; }
static int  new_lbl(PCtx *ctx)  { return ctx->next_lbl++; }

static void loc_init(VLocals *l) { l->n = 0; }
static VLocal *loc_find(VLocals *l, const char *name) {
    for (int i = 0; i < l->n; i++)
        if (strcmp(l->v[i].name, name) == 0) return &l->v[i];
    return NULL;
}
static void loc_add(VLocals *l, const char *name, int reg, VK kind) {
    if (l->n >= MAX_LOCALS) return;
    VLocal *v = &l->v[l->n++];
    strncpy(v->name, name, sizeof(v->name)-1);
    v->name[sizeof(v->name)-1] = '\0';
    v->reg = reg; v->kind = kind;
}

/* Type suffix and prefix helpers */
static const char *vk_ptx(VK k) {
    return (k == VK_FLOAT) ? ".f64" : (k == VK_PRED) ? ".pred" : ".s64";
}
static const char *vk_pfx(VK k) {
    return (k == VK_FLOAT) ? "%fd" : (k == VK_PRED) ? "%p" : "%rd";
}

/* ── Expression emit ────────────────────────────────────────────────── */
static int emit_expr(PCtx *ctx, VLocals *l, ASTNode *node, VK *ok);

static int emit_expr(PCtx *ctx, VLocals *l, ASTNode *node, VK *ok) {
    if (!node || ctx->err) return -1;
    *ok = VK_INT;

    switch (node->type) {

    case AST_NUMBER: {
        int r = new_reg(ctx); *ok = VK_INT;
        psb_appendf(&ctx->sb, "    mov.s64 %%rd%d, %lld;\n", r, (long long)node->as.number);
        return r;
    }
    case AST_FLOAT: {
        int r = new_reg(ctx); *ok = VK_FLOAT;
        /* Emit as hex double for exact representation */
        union { double d; unsigned long long u; } cv;
        cv.d = node->as.float_val;
        psb_appendf(&ctx->sb, "    mov.f64 %%fd%d, 0d%016llx;\n", r, (unsigned long long)cv.u);
        return r;
    }
    case AST_BOOL: {
        int r = new_reg(ctx); *ok = VK_INT;
        psb_appendf(&ctx->sb, "    mov.s64 %%rd%d, %d;\n", r, node->as.bool_val ? 1 : 0);
        return r;
    }
    case AST_IDENTIFIER: {
        VLocal *v = loc_find(l, node->as.identifier);
        if (!v) { ptx_err(ctx, "undefined '%s'", node->as.identifier); return -1; }
        *ok = v->kind;
        return v->reg;
    }

    case AST_CALL: {
        const char *fn = node->as.call.name;

        /* Thread indexing built-ins → PTX special registers */
        struct { const char *name; const char *ptx_reg; } builtins[] = {
            {"thread_id_x", "%tid.x"},
            {"thread_id_y", "%tid.y"},
            {"thread_id_z", "%tid.z"},
            {"block_id_x",  "%ctaid.x"},
            {"block_id_y",  "%ctaid.y"},
            {"block_id_z",  "%ctaid.z"},
            {"block_dim_x", "%ntid.x"},
            {"block_dim_y", "%ntid.y"},
            {"block_dim_z", "%ntid.z"},
            {"grid_dim_x",  "%nctaid.x"},
            {"grid_dim_y",  "%nctaid.y"},
            {"grid_dim_z",  "%nctaid.z"},
            {NULL, NULL}
        };
        for (int i = 0; builtins[i].name; i++) {
            if (strcmp(fn, builtins[i].name) == 0) {
                int tmp = new_reg(ctx);
                int r   = new_reg(ctx);
                *ok = VK_INT;
                psb_appendf(&ctx->sb,
                    "    mov.u32 %%rd%d, %s;\n"
                    "    cvt.s64.u32 %%rd%d, %%rd%d;\n",
                    tmp, builtins[i].ptx_reg, r, tmp);
                return r;
            }
        }

        /* global_id_x = block_id_x * block_dim_x + thread_id_x */
        if (strcmp(fn, "global_id_x") == 0 || strcmp(fn, "global_id_y") == 0) {
            int is_y = (fn[9] == 'y');
            const char *bid_reg  = is_y ? "%ctaid.y" : "%ctaid.x";
            const char *bdim_reg = is_y ? "%ntid.y"  : "%ntid.x";
            const char *tid_reg  = is_y ? "%tid.y"   : "%tid.x";
            int bid_raw  = new_reg(ctx); int bid  = new_reg(ctx);
            int bdim_raw = new_reg(ctx); int bdim = new_reg(ctx);
            int tid_raw  = new_reg(ctx); int tid  = new_reg(ctx);
            int mul_r    = new_reg(ctx); int r    = new_reg(ctx);
            *ok = VK_INT;
            psb_appendf(&ctx->sb,
                "    mov.u32 %%rd%d, %s;\n"
                "    cvt.s64.u32 %%rd%d, %%rd%d;\n"
                "    mov.u32 %%rd%d, %s;\n"
                "    cvt.s64.u32 %%rd%d, %%rd%d;\n"
                "    mov.u32 %%rd%d, %s;\n"
                "    cvt.s64.u32 %%rd%d, %%rd%d;\n"
                "    mul.lo.s64 %%rd%d, %%rd%d, %%rd%d;\n"
                "    add.s64 %%rd%d, %%rd%d, %%rd%d;\n",
                bid_raw, bid_reg, bid, bid_raw,
                bdim_raw, bdim_reg, bdim, bdim_raw,
                tid_raw, tid_reg, tid, tid_raw,
                mul_r, bid, bdim,
                r, mul_r, tid);
            return r;
        }

        /* gpu_barrier() → bar.sync 0; */
        if (strcmp(fn, "gpu_barrier") == 0) {
            *ok = VK_INT;
            psb_append(&ctx->sb, "    bar.sync 0;\n");
            int r = new_reg(ctx);
            psb_appendf(&ctx->sb, "    mov.s64 %%rd%d, 0;\n", r);
            return r;
        }

        ptx_err(ctx, "unsupported call '%s' in gpu fn", fn);
        return -1;
    }

    case AST_PREFIX_OP: {
        int nargs = node->as.prefix_op.arg_count;
        TokenType op = node->as.prefix_op.op;

        /* Unary minus */
        if (nargs == 1 && op == TOKEN_MINUS) {
            VK k; int v = emit_expr(ctx, l, node->as.prefix_op.args[0], &k);
            if (v < 0) return -1;
            int r = new_reg(ctx); *ok = k;
            const char *pt = (k == VK_FLOAT) ? "f64" : "s64";
            psb_appendf(&ctx->sb, "    neg.%s %s%d, %s%d;\n",
                        pt, vk_pfx(k), r, vk_pfx(k), v);
            return r;
        }
        /* Unary not */
        if (nargs == 1 && op == TOKEN_NOT) {
            VK k; int v = emit_expr(ctx, l, node->as.prefix_op.args[0], &k);
            if (v < 0) return -1;
            int r = new_reg(ctx); *ok = VK_PRED;
            if (k == VK_PRED) {
                psb_appendf(&ctx->sb, "    not.pred %%p%d, %%p%d;\n", r, v);
            } else {
                int p = new_reg(ctx);
                psb_appendf(&ctx->sb,
                    "    setp.eq.s64 %%p%d, %%rd%d, 0;\n"
                    "    not.pred %%p%d, %%p%d;\n",   /* invert: ne 0 → not(eq 0) */
                    p, v, r, p);
            }
            return r;
        }

        if (nargs != 2) {
            ptx_err(ctx, "unsupported prefix arity %d", nargs);
            return -1;
        }

        /* Binary: emit both operands */
        VK lk, rk;
        int lv = emit_expr(ctx, l, node->as.prefix_op.args[0], &lk);
        int rv = emit_expr(ctx, l, node->as.prefix_op.args[1], &rk);
        if (lv < 0 || rv < 0) return -1;

        /* Determine result kind */
        VK reskind = (lk == VK_FLOAT || rk == VK_FLOAT) ? VK_FLOAT : VK_INT;

        /* Logical and/or — both operands become predicates */
        if (op == TOKEN_AND || op == TOKEN_OR) {
            /* Convert each operand to predicate if not already */
            int lp, rp;
            if (lk == VK_PRED) { lp = lv; }
            else { lp = new_reg(ctx); psb_appendf(&ctx->sb, "    setp.ne.s64 %%p%d, %%rd%d, 0;\n", lp, lv); }
            if (rk == VK_PRED) { rp = rv; }
            else { rp = new_reg(ctx); psb_appendf(&ctx->sb, "    setp.ne.s64 %%p%d, %%rd%d, 0;\n", rp, rv); }
            int pr = new_reg(ctx); *ok = VK_PRED;
            const char *logop = (op == TOKEN_AND) ? "and" : "or";
            psb_appendf(&ctx->sb, "    %s.pred %%p%d, %%p%d, %%p%d;\n", logop, pr, lp, rp);
            return pr;
        }

        /* Comparison → predicate */
        bool is_cmp = (op == TOKEN_EQ || op == TOKEN_NE ||
                       op == TOKEN_LT || op == TOKEN_LE ||
                       op == TOKEN_GT || op == TOKEN_GE);
        if (is_cmp) {
            int pr = new_reg(ctx); *ok = VK_PRED;
            const char *cmpop;
            switch (op) {
                case TOKEN_EQ: cmpop = "eq"; break;
                case TOKEN_NE: cmpop = "ne"; break;
                case TOKEN_LT: cmpop = "lt"; break;
                case TOKEN_LE: cmpop = "le"; break;
                case TOKEN_GT: cmpop = "gt"; break;
                default:       cmpop = "ge"; break;
            }
            const char *ct = (lk == VK_FLOAT) ? "f64" : "s64";
            psb_appendf(&ctx->sb, "    setp.%s.%s %%p%d, %s%d, %s%d;\n",
                        cmpop, ct, pr, vk_pfx(lk), lv, vk_pfx(rk), rv);
            return pr;
        }

        /* Arithmetic */
        int res = new_reg(ctx); *ok = reskind;
        const char *pt = (reskind == VK_FLOAT) ? "f64" : "s64";

        switch (op) {
            case TOKEN_PLUS:
                psb_appendf(&ctx->sb, "    add.%s %s%d, %s%d, %s%d;\n",
                    pt, vk_pfx(reskind), res, vk_pfx(lk), lv, vk_pfx(rk), rv);
                break;
            case TOKEN_MINUS:
                psb_appendf(&ctx->sb, "    sub.%s %s%d, %s%d, %s%d;\n",
                    pt, vk_pfx(reskind), res, vk_pfx(lk), lv, vk_pfx(rk), rv);
                break;
            case TOKEN_STAR:
                psb_appendf(&ctx->sb, "    mul%s.%s %s%d, %s%d, %s%d;\n",
                    (reskind == VK_INT) ? ".lo" : "",
                    pt, vk_pfx(reskind), res, vk_pfx(lk), lv, vk_pfx(rk), rv);
                break;
            case TOKEN_SLASH:
                if (reskind == VK_FLOAT)
                    psb_appendf(&ctx->sb, "    div.rn.f64 %%fd%d, %%fd%d, %%fd%d;\n", res, lv, rv);
                else
                    psb_appendf(&ctx->sb, "    div.s64 %%rd%d, %%rd%d, %%rd%d;\n", res, lv, rv);
                break;
            case TOKEN_PERCENT:
                psb_appendf(&ctx->sb, "    rem.s64 %%rd%d, %%rd%d, %%rd%d;\n", res, lv, rv);
                break;
            default:
                ptx_err(ctx, "unsupported binary token type %d", (int)op);
                return -1;
        }
        return res;
    }

    case AST_IF: {
        int lbl_else  = new_lbl(ctx);
        int lbl_endif = new_lbl(ctx);
        bool has_else = (node->as.if_stmt.else_branch != NULL);

        VK ck; int cv = emit_expr(ctx, l, node->as.if_stmt.condition, &ck);
        if (cv < 0) return -1;

        int pred;
        if (ck == VK_PRED) {
            pred = cv;
        } else {
            pred = new_reg(ctx);
            psb_appendf(&ctx->sb, "    setp.ne.s64 %%p%d, %%rd%d, 0;\n", pred, cv);
        }
        int skip_lbl = has_else ? lbl_else : lbl_endif;
        psb_appendf(&ctx->sb, "    @!%%p%d bra L%d;\n", pred, skip_lbl);

        int res = new_reg(ctx); *ok = VK_INT;
        VK tk; int tv = emit_expr(ctx, l, node->as.if_stmt.then_branch, &tk);
        if (tv >= 0 && tk != VK_PRED)
            psb_appendf(&ctx->sb, "    mov.%s %s%d, %s%d;\n",
                        (tk==VK_FLOAT)?"f64":"s64", vk_pfx(tk), res, vk_pfx(tk), tv);

        if (has_else) {
            psb_appendf(&ctx->sb, "    bra L%d;\n", lbl_endif);
            psb_appendf(&ctx->sb, "L%d:\n", lbl_else);
            VK ek; int ev = emit_expr(ctx, l, node->as.if_stmt.else_branch, &ek);
            if (ev >= 0 && ek != VK_PRED)
                psb_appendf(&ctx->sb, "    mov.%s %s%d, %s%d;\n",
                            (ek==VK_FLOAT)?"f64":"s64", vk_pfx(ek), res, vk_pfx(ek), ev);
        }
        psb_appendf(&ctx->sb, "L%d:\n", lbl_endif);
        return res;
    }

    case AST_BLOCK: {
        int last_r = -1; VK last_k = VK_INT;
        for (int i = 0; i < node->as.block.count; i++) {
            ASTNode *s = node->as.block.statements[i];
            if (s->type == AST_RETURN) {
                VK vk; int vv = -1;
                if (s->as.return_stmt.value)
                    vv = emit_expr(ctx, l, s->as.return_stmt.value, &vk);
                psb_append(&ctx->sb, "    ret;\n");
                *ok = (vv >= 0) ? vk : VK_INT;
                return vv;
            }
            if (s->type == AST_LET) {
                VK ek = VK_INT; int ev = -1;
                if (s->as.let.value) {
                    ev = emit_expr(ctx, l, s->as.let.value, &ek);
                } else {
                    ev = new_reg(ctx); ek = VK_INT;
                    psb_appendf(&ctx->sb, "    mov.s64 %%rd%d, 0;\n", ev);
                }
                if (ev < 0) return -1;
                loc_add(l, s->as.let.name, ev, ek);
                last_r = ev; last_k = ek;
                continue;
            }
            last_r = emit_expr(ctx, l, s, &last_k);
            if (ctx->err) return -1;
        }
        *ok = last_k;
        return last_r;
    }

    default:
        ptx_err(ctx, "unsupported AST node %d in gpu fn", (int)node->type);
        return -1;
    }
}

/* ── Emit a gpu fn ──────────────────────────────────────────────────── */
static void emit_kernel(PCtx *ctx, ASTNode *fn) {
    if (!fn || fn->type != AST_FUNCTION || !fn->as.function.is_gpu) return;

    ctx->next_reg = 0;
    ctx->next_lbl = 0;

    const char *name = fn->as.function.name;
    psb_appendf(&ctx->sb, "\n.visible .entry %s(\n", name);

    for (int i = 0; i < fn->as.function.param_count; i++) {
        Parameter *p = &fn->as.function.params[i];
        const char *pt = (p->type == TYPE_FLOAT) ? ".f64" : ".s64";
        psb_appendf(&ctx->sb, "    .param %s param_%s%s\n",
                    pt, p->name,
                    (i+1 < fn->as.function.param_count) ? "," : "");
    }
    psb_append(&ctx->sb, ")\n{\n");

    /* Declare register pools upfront */
    psb_append(&ctx->sb, "    .reg .s64  %rd<128>;\n");
    psb_append(&ctx->sb, "    .reg .f64  %fd<64>;\n");
    psb_append(&ctx->sb, "    .reg .pred  %p<32>;\n");
    psb_append(&ctx->sb, "    .reg .u32   %r<64>;\n");

    /* Load parameters */
    VLocals l; loc_init(&l);
    for (int i = 0; i < fn->as.function.param_count; i++) {
        Parameter *p = &fn->as.function.params[i];
        VK k = (p->type == TYPE_FLOAT) ? VK_FLOAT : VK_INT;
        int r = new_reg(ctx);
        psb_appendf(&ctx->sb, "    ld.param%s %s%d, [param_%s];\n",
                    vk_ptx(k), vk_pfx(k), r, p->name);
        loc_add(&l, p->name, r, k);
    }

    if (fn->as.function.body) {
        VK rk;
        emit_expr(ctx, &l, fn->as.function.body, &rk);
    }

    /* Ensure ret at end for void kernels */
    if (fn->as.function.return_type == TYPE_VOID)
        psb_append(&ctx->sb, "    ret;\n");

    psb_append(&ctx->sb, "}\n");
}

/* ── Public API ─────────────────────────────────────────────────────── */
int ptx_backend_emit_fp(ASTNode *root, FILE *out,
                        const char *source_file, bool verbose) {
    if (!root || !out) return 1;

    PCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    psb_init(&ctx.sb);
    ctx.verbose = verbose;
    ctx.src = source_file ? source_file : "<unknown>";

    /* Header */
    psb_appendf(&ctx.sb, "// nanolang PTX output: %s\n", ctx.src);
    psb_append(&ctx.sb,  "// Generated by nanoc --target ptx\n");
    psb_append(&ctx.sb,  "// Load: cuModuleLoadData() + cuModuleGetFunction()\n\n");
    psb_append(&ctx.sb,  ".version 8.0\n");
    psb_append(&ctx.sb,  ".target sm_90\n");   /* sm_90: H100/GB10 Blackwell */
    psb_append(&ctx.sb,  ".address_size 64\n");

    int count = 0;
    if (root->type == AST_PROGRAM) {
        for (int i = 0; i < root->as.program.count && !ctx.err; i++) {
            ASTNode *item = root->as.program.items[i];
            if (item && item->type == AST_FUNCTION && item->as.function.is_gpu) {
                emit_kernel(&ctx, item);
                count++;
            }
        }
    }

    if (ctx.err) {
        fprintf(stderr, "PTX backend error: %s\n", ctx.errmsg);
        psb_free(&ctx.sb);
        return 1;
    }
    if (count == 0 && verbose) {
        fprintf(stderr, "[ptx] warning: no `gpu fn` found in %s\n", ctx.src);
    }

    fputs(ctx.sb.buf, out);
    if (verbose)
        fprintf(stderr, "[ptx] emitted %d kernel(s)\n", count);

    psb_free(&ctx.sb);
    return 0;
}

int ptx_backend_emit(ASTNode *root, const char *output_path,
                     const char *source_file, bool verbose) {
    if (!output_path) return ptx_backend_emit_fp(root, stdout, source_file, verbose);
    FILE *f = fopen(output_path, "w");
    if (!f) { fprintf(stderr, "PTX: cannot open %s\n", output_path); return 1; }
    int rc = ptx_backend_emit_fp(root, f, source_file, verbose);
    fclose(f);
    return rc;
}
