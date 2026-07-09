/*
 * opencl_backend.c — nanolang OpenCL C kernel emit backend
 *
 * Translates `gpu fn` AST functions into OpenCL C kernel source.
 * Mirrors the structure of ptx_backend.c but emits high-level C instead
 * of PTX assembly.
 *
 * Pointer-param detection:
 *   Before emitting, we walk the kernel body to find which `int`-typed
 *   parameters flow into the address argument of gpu_load / gpu_store.
 *   Those are emitted as `__global char*`; the rest as `long` scalars.
 *   This matches how opencl_runtime.c dispatches clSetKernelArg.
 *
 * Temp variable naming:
 *   Integer / bool / predicate results → _t<n>  (long)
 *   Float results                      → _f<n>  (double)
 */

#include "opencl_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>

/* ── String builder ──────────────────────────────────────────────────────── */
typedef struct { char *buf; size_t len; size_t cap; } OclSB;

static void osb_init(OclSB *s)  { s->buf = NULL; s->len = 0; s->cap = 0; }
static void osb_free(OclSB *s)  { free(s->buf); s->buf = NULL; s->len = s->cap = 0; }

static void osb_grow(OclSB *s, size_t need) {
    if (s->len + need + 1 <= s->cap) return;
    size_t nc = s->cap ? s->cap * 2 : 4096;
    while (nc < s->len + need + 1) nc *= 2;
    s->buf = realloc(s->buf, nc);
    s->cap = nc;
}
static void osb_append(OclSB *s, const char *str) {
    size_t n = strlen(str);
    osb_grow(s, n);
    memcpy(s->buf + s->len, str, n + 1);
    s->len += n;
}
static void osb_appendf(OclSB *s, const char *fmt, ...) {
    char tmp[512];
    va_list ap; va_start(ap, fmt);
    vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    osb_append(s, tmp);
}

/* ── Value kinds ─────────────────────────────────────────────────────────── */
typedef enum { OVK_INT, OVK_FLOAT } OVK;

/* ── Local variable map ──────────────────────────────────────────────────── */
typedef struct { char name[64]; int reg; OVK kind; } OLocal;
#define MAX_OCL_LOCALS 128
typedef struct { OLocal v[MAX_OCL_LOCALS]; int n; } OLocals;

static void oloc_init(OLocals *l) { l->n = 0; }
static OLocal *oloc_find(OLocals *l, const char *nm) {
    for (int i = 0; i < l->n; i++)
        if (strcmp(l->v[i].name, nm) == 0) return &l->v[i];
    return NULL;
}
static void oloc_add(OLocals *l, const char *nm, int reg, OVK kind) {
    if (l->n >= MAX_OCL_LOCALS) return;
    OLocal *v = &l->v[l->n++];
    strncpy(v->name, nm, sizeof(v->name)-1);
    v->name[sizeof(v->name)-1] = '\0';
    v->reg = reg; v->kind = kind;
}

/* ── Codegen context ─────────────────────────────────────────────────────── */
typedef struct {
    OclSB   sb;
    int     next_t;     /* _t<n> integer/bool temps */
    int     next_f;     /* _f<n> double temps */
    bool    verbose;
    const char *src;
    bool    err;
    char    errmsg[512];
} OCtx;

static void ocl_err(OCtx *ctx, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    vsnprintf(ctx->errmsg, sizeof(ctx->errmsg), fmt, ap);
    va_end(ap);
    ctx->err = true;
}
static int new_t(OCtx *ctx) { return ctx->next_t++; }
static int new_f(OCtx *ctx) { return ctx->next_f++; }

/* ── Pointer-param static analysis ──────────────────────────────────────── */
/* Mark params that appear (directly or in sub-expressions) as the pointer
 * argument of any gpu_load / gpu_store / gpu_*_float / gpu_atomic_add. */

static void mark_ptr_in_expr(ASTNode *n, bool *is_ptr, const char **pnames, int np);

static void collect_ptr_params(ASTNode *n, bool *is_ptr, const char **pnames, int np) {
    if (!n) return;
    switch (n->type) {
        case AST_CALL: {
            bool mem = (strcmp(n->as.call.name, "gpu_load")        == 0 ||
                        strcmp(n->as.call.name, "gpu_store")       == 0 ||
                        strcmp(n->as.call.name, "gpu_load_float")  == 0 ||
                        strcmp(n->as.call.name, "gpu_store_float") == 0 ||
                        strcmp(n->as.call.name, "gpu_atomic_add")  == 0);
            if (mem && n->as.call.arg_count > 0)
                mark_ptr_in_expr(n->as.call.args[0], is_ptr, pnames, np);
            for (int i = 0; i < n->as.call.arg_count; i++)
                collect_ptr_params(n->as.call.args[i], is_ptr, pnames, np);
            break;
        }
        case AST_PREFIX_OP:
            for (int i = 0; i < n->as.prefix_op.arg_count; i++)
                collect_ptr_params(n->as.prefix_op.args[i], is_ptr, pnames, np);
            break;
        case AST_IF:
            collect_ptr_params(n->as.if_stmt.condition,   is_ptr, pnames, np);
            collect_ptr_params(n->as.if_stmt.then_branch, is_ptr, pnames, np);
            collect_ptr_params(n->as.if_stmt.else_branch, is_ptr, pnames, np);
            break;
        case AST_BLOCK:
            for (int i = 0; i < n->as.block.count; i++)
                collect_ptr_params(n->as.block.statements[i], is_ptr, pnames, np);
            break;
        case AST_LET:
            collect_ptr_params(n->as.let.value, is_ptr, pnames, np);
            break;
        case AST_RETURN:
            collect_ptr_params(n->as.return_stmt.value, is_ptr, pnames, np);
            break;
        default: break;
    }
}

static void mark_ptr_in_expr(ASTNode *n, bool *is_ptr, const char **pnames, int np) {
    if (!n) return;
    if (n->type == AST_IDENTIFIER) {
        for (int i = 0; i < np; i++)
            if (strcmp(n->as.identifier, pnames[i]) == 0) is_ptr[i] = true;
        return;
    }
    switch (n->type) {
        case AST_PREFIX_OP:
            for (int i = 0; i < n->as.prefix_op.arg_count; i++)
                mark_ptr_in_expr(n->as.prefix_op.args[i], is_ptr, pnames, np);
            break;
        case AST_CALL:
            for (int i = 0; i < n->as.call.arg_count; i++)
                mark_ptr_in_expr(n->as.call.args[i], is_ptr, pnames, np);
            break;
        default: break;
    }
}

/* ── Expression emitter ──────────────────────────────────────────────────── */
typedef struct { int reg; OVK kind; } OVal;
static const OVal OVAL_ERR = {-1, OVK_INT};

/* Forward declaration */
static OVal emit_expr(OCtx *ctx, OLocals *l, ASTNode *node);

static OVal emit_expr(OCtx *ctx, OLocals *l, ASTNode *node) {
    if (!node || ctx->err) return OVAL_ERR;

    switch (node->type) {

    case AST_NUMBER: {
        int r = new_t(ctx);
        osb_appendf(&ctx->sb, "    long _t%d = %lldLL;\n", r, (long long)node->as.number);
        return (OVal){r, OVK_INT};
    }
    case AST_FLOAT: {
        int r = new_f(ctx);
        osb_appendf(&ctx->sb, "    double _f%d = %.17g;\n", r, node->as.float_val);
        return (OVal){r, OVK_FLOAT};
    }
    case AST_BOOL: {
        int r = new_t(ctx);
        osb_appendf(&ctx->sb, "    long _t%d = %dLL;\n", r, node->as.bool_val ? 1 : 0);
        return (OVal){r, OVK_INT};
    }
    case AST_IDENTIFIER: {
        OLocal *v = oloc_find(l, node->as.identifier);
        if (!v) { ocl_err(ctx, "undefined '%s'", node->as.identifier); return OVAL_ERR; }
        return (OVal){v->reg, v->kind};
    }

    case AST_CALL: {
        const char *fn = node->as.call.name;

        /* Thread / block / grid built-ins */
        struct { const char *name; const char *ocl_fn; int dim; } builtins[] = {
            {"thread_id_x",  "get_local_id",   0},
            {"thread_id_y",  "get_local_id",   1},
            {"thread_id_z",  "get_local_id",   2},
            {"block_id_x",   "get_group_id",   0},
            {"block_id_y",   "get_group_id",   1},
            {"block_id_z",   "get_group_id",   2},
            {"block_dim_x",  "get_local_size", 0},
            {"block_dim_y",  "get_local_size", 1},
            {"block_dim_z",  "get_local_size", 2},
            {"grid_dim_x",   "get_num_groups", 0},
            {"grid_dim_y",   "get_num_groups", 1},
            {"grid_dim_z",   "get_num_groups", 2},
            {NULL, NULL, 0}
        };
        for (int i = 0; builtins[i].name; i++) {
            if (strcmp(fn, builtins[i].name) == 0) {
                int r = new_t(ctx);
                osb_appendf(&ctx->sb, "    long _t%d = (long)%s(%d);\n",
                    r, builtins[i].ocl_fn, builtins[i].dim);
                return (OVal){r, OVK_INT};
            }
        }

        /* global_id_x/y = get_global_id(0/1) */
        if (strcmp(fn, "global_id_x") == 0 || strcmp(fn, "global_id_y") == 0) {
            int r = new_t(ctx);
            /* fn = "global_id_x/y"; fn[10] is 'x' or 'y' */
            osb_appendf(&ctx->sb, "    long _t%d = (long)get_global_id(%d);\n",
                        r, (fn[10] == 'y') ? 1 : 0);
            return (OVal){r, OVK_INT};
        }

        /* gpu_barrier */
        if (strcmp(fn, "gpu_barrier") == 0) {
            osb_append(&ctx->sb, "    barrier(CLK_LOCAL_MEM_FENCE);\n");
            int r = new_t(ctx);
            osb_appendf(&ctx->sb, "    long _t%d = 0LL;\n", r);
            return (OVal){r, OVK_INT};
        }

        /* gpu_load(ptr) → *((__global long*)((ulong)ptr)) */
        if (strcmp(fn, "gpu_load") == 0) {
            if (node->as.call.arg_count != 1) {
                ocl_err(ctx, "gpu_load: 1 arg"); return OVAL_ERR;
            }
            OVal pv = emit_expr(ctx, l, node->as.call.args[0]);
            if (pv.reg < 0) return OVAL_ERR;
            int r = new_t(ctx);
            osb_appendf(&ctx->sb,
                "    long _t%d = *((__global long*)((ulong)_t%d));\n", r, pv.reg);
            return (OVal){r, OVK_INT};
        }

        /* gpu_store(ptr, val) → *((__global long*)((ulong)ptr)) = val */
        if (strcmp(fn, "gpu_store") == 0) {
            if (node->as.call.arg_count != 2) {
                ocl_err(ctx, "gpu_store: 2 args"); return OVAL_ERR;
            }
            OVal pv = emit_expr(ctx, l, node->as.call.args[0]);
            OVal vv = emit_expr(ctx, l, node->as.call.args[1]);
            if (pv.reg < 0 || vv.reg < 0) return OVAL_ERR;
            osb_appendf(&ctx->sb,
                "    *((__global long*)((ulong)_t%d)) = _t%d;\n", pv.reg, vv.reg);
            int r = new_t(ctx);
            osb_appendf(&ctx->sb, "    long _t%d = 0LL;\n", r);
            return (OVal){r, OVK_INT};
        }

        /* gpu_load_float(ptr) → *((__global double*)((ulong)ptr)) */
        if (strcmp(fn, "gpu_load_float") == 0) {
            if (node->as.call.arg_count != 1) {
                ocl_err(ctx, "gpu_load_float: 1 arg"); return OVAL_ERR;
            }
            OVal pv = emit_expr(ctx, l, node->as.call.args[0]);
            if (pv.reg < 0) return OVAL_ERR;
            int r = new_f(ctx);
            osb_appendf(&ctx->sb,
                "    double _f%d = *((__global double*)((ulong)_t%d));\n", r, pv.reg);
            return (OVal){r, OVK_FLOAT};
        }

        /* gpu_store_float(ptr, val) → *((__global double*)((ulong)ptr)) = val */
        if (strcmp(fn, "gpu_store_float") == 0) {
            if (node->as.call.arg_count != 2) {
                ocl_err(ctx, "gpu_store_float: 2 args"); return OVAL_ERR;
            }
            OVal pv = emit_expr(ctx, l, node->as.call.args[0]);
            OVal vv = emit_expr(ctx, l, node->as.call.args[1]);
            if (pv.reg < 0 || vv.reg < 0) return OVAL_ERR;
            /* coerce val to double if needed */
            char val_name[16];
            if (vv.kind != OVK_FLOAT) {
                int fr = new_f(ctx);
                osb_appendf(&ctx->sb, "    double _f%d = (double)_t%d;\n", fr, vv.reg);
                snprintf(val_name, sizeof(val_name), "_f%d", fr);
            } else {
                snprintf(val_name, sizeof(val_name), "_f%d", vv.reg);
            }
            osb_appendf(&ctx->sb,
                "    *((__global double*)((ulong)_t%d)) = %s;\n", pv.reg, val_name);
            int r = new_t(ctx);
            osb_appendf(&ctx->sb, "    long _t%d = 0LL;\n", r);
            return (OVal){r, OVK_INT};
        }

        /* gpu_atomic_add(ptr, delta) → atom_add((__global long*)ptr, delta) */
        if (strcmp(fn, "gpu_atomic_add") == 0) {
            if (node->as.call.arg_count != 2) {
                ocl_err(ctx, "gpu_atomic_add: 2 args"); return OVAL_ERR;
            }
            OVal pv = emit_expr(ctx, l, node->as.call.args[0]);
            OVal dv = emit_expr(ctx, l, node->as.call.args[1]);
            if (pv.reg < 0 || dv.reg < 0) return OVAL_ERR;
            int r = new_t(ctx);
            osb_appendf(&ctx->sb,
                "    long _t%d = atom_add((__global long*)((ulong)_t%d), _t%d);\n",
                r, pv.reg, dv.reg);
            return (OVal){r, OVK_INT};
        }

        ocl_err(ctx, "unsupported call '%s' in gpu fn", fn);
        return OVAL_ERR;
    }

    case AST_PREFIX_OP: {
        int nargs = node->as.prefix_op.arg_count;
        TokenType op = node->as.prefix_op.op;

        /* Unary minus */
        if (nargs == 1 && op == TOKEN_MINUS) {
            OVal v = emit_expr(ctx, l, node->as.prefix_op.args[0]);
            if (v.reg < 0) return OVAL_ERR;
            if (v.kind == OVK_FLOAT) {
                int r = new_f(ctx);
                osb_appendf(&ctx->sb, "    double _f%d = -_f%d;\n", r, v.reg);
                return (OVal){r, OVK_FLOAT};
            } else {
                int r = new_t(ctx);
                osb_appendf(&ctx->sb, "    long _t%d = -_t%d;\n", r, v.reg);
                return (OVal){r, OVK_INT};
            }
        }

        /* Unary not */
        if (nargs == 1 && op == TOKEN_NOT) {
            OVal v = emit_expr(ctx, l, node->as.prefix_op.args[0]);
            if (v.reg < 0) return OVAL_ERR;
            int r = new_t(ctx);
            if (v.kind == OVK_FLOAT)
                osb_appendf(&ctx->sb, "    long _t%d = (_f%d == 0.0) ? 1LL : 0LL;\n", r, v.reg);
            else
                osb_appendf(&ctx->sb, "    long _t%d = (_t%d == 0LL) ? 1LL : 0LL;\n", r, v.reg);
            return (OVal){r, OVK_INT};
        }

        if (nargs != 2) { ocl_err(ctx, "unsupported prefix arity %d", nargs); return OVAL_ERR; }

        OVal lv = emit_expr(ctx, l, node->as.prefix_op.args[0]);
        OVal rv = emit_expr(ctx, l, node->as.prefix_op.args[1]);
        if (lv.reg < 0 || rv.reg < 0) return OVAL_ERR;

        bool is_float = (lv.kind == OVK_FLOAT || rv.kind == OVK_FLOAT);

        /* Coerce int operand to float if mixed */
        int li = lv.reg, ri = rv.reg;
        if (is_float && lv.kind != OVK_FLOAT) {
            li = new_f(ctx);
            osb_appendf(&ctx->sb, "    double _f%d = (double)_t%d;\n", li, lv.reg);
        }
        if (is_float && rv.kind != OVK_FLOAT) {
            ri = new_f(ctx);
            osb_appendf(&ctx->sb, "    double _f%d = (double)_t%d;\n", ri, rv.reg);
        }

        /* Logical and/or — convert both to bool long, then combine */
        if (op == TOKEN_AND || op == TOKEN_OR) {
            int lb = new_t(ctx), rb = new_t(ctx);
            if (lv.kind == OVK_FLOAT)
                osb_appendf(&ctx->sb, "    long _t%d = (_f%d != 0.0) ? 1LL : 0LL;\n", lb, lv.reg);
            else
                osb_appendf(&ctx->sb, "    long _t%d = (_t%d != 0LL) ? 1LL : 0LL;\n", lb, lv.reg);
            if (rv.kind == OVK_FLOAT)
                osb_appendf(&ctx->sb, "    long _t%d = (_f%d != 0.0) ? 1LL : 0LL;\n", rb, rv.reg);
            else
                osb_appendf(&ctx->sb, "    long _t%d = (_t%d != 0LL) ? 1LL : 0LL;\n", rb, rv.reg);
            int r = new_t(ctx);
            const char *logop = (op == TOKEN_AND) ? "&&" : "||";
            osb_appendf(&ctx->sb, "    long _t%d = (_t%d %s _t%d) ? 1LL : 0LL;\n",
                        r, lb, logop, rb);
            return (OVal){r, OVK_INT};
        }

        /* Comparison → 1LL or 0LL */
        bool is_cmp = (op == TOKEN_EQ || op == TOKEN_NE ||
                       op == TOKEN_LT || op == TOKEN_LE ||
                       op == TOKEN_GT || op == TOKEN_GE);
        if (is_cmp) {
            const char *cmp_op;
            switch (op) {
                case TOKEN_EQ: cmp_op = "=="; break;
                case TOKEN_NE: cmp_op = "!="; break;
                case TOKEN_LT: cmp_op = "<";  break;
                case TOKEN_LE: cmp_op = "<="; break;
                case TOKEN_GT: cmp_op = ">";  break;
                default:       cmp_op = ">="; break;
            }
            int r = new_t(ctx);
            if (is_float)
                osb_appendf(&ctx->sb, "    long _t%d = (_f%d %s _f%d) ? 1LL : 0LL;\n",
                            r, li, cmp_op, ri);
            else
                osb_appendf(&ctx->sb, "    long _t%d = (_t%d %s _t%d) ? 1LL : 0LL;\n",
                            r, li, cmp_op, ri);
            return (OVal){r, OVK_INT};
        }

        /* Arithmetic */
        if (is_float) {
            const char *arith;
            switch (op) {
                case TOKEN_PLUS:  arith = "+"; break;
                case TOKEN_MINUS: arith = "-"; break;
                case TOKEN_STAR:  arith = "*"; break;
                case TOKEN_SLASH: arith = "/"; break;
                default:
                    ocl_err(ctx, "unsupported float op %d", (int)op);
                    return OVAL_ERR;
            }
            int r = new_f(ctx);
            osb_appendf(&ctx->sb, "    double _f%d = _f%d %s _f%d;\n", r, li, arith, ri);
            return (OVal){r, OVK_FLOAT};
        } else {
            const char *arith;
            switch (op) {
                case TOKEN_PLUS:    arith = "+"; break;
                case TOKEN_MINUS:   arith = "-"; break;
                case TOKEN_STAR:    arith = "*"; break;
                case TOKEN_SLASH:   arith = "/"; break;
                case TOKEN_PERCENT: arith = "%"; break;
                default:
                    ocl_err(ctx, "unsupported int op %d", (int)op);
                    return OVAL_ERR;
            }
            int r = new_t(ctx);
            osb_appendf(&ctx->sb, "    long _t%d = _t%d %s _t%d;\n", r, li, arith, ri);
            return (OVal){r, OVK_INT};
        }
    }

    case AST_IF: {
        bool has_else = (node->as.if_stmt.else_branch != NULL);

        OVal cv = emit_expr(ctx, l, node->as.if_stmt.condition);
        if (cv.reg < 0) return OVAL_ERR;

        int r = new_t(ctx);
        osb_appendf(&ctx->sb, "    long _t%d = 0LL;\n", r);
        osb_appendf(&ctx->sb, "    if (_t%d) {\n", cv.reg);

        OVal tv = emit_expr(ctx, l, node->as.if_stmt.then_branch);
        if (tv.reg >= 0 && tv.kind == OVK_INT)
            osb_appendf(&ctx->sb, "    _t%d = _t%d;\n", r, tv.reg);

        if (has_else) {
            osb_append(&ctx->sb, "    } else {\n");
            OVal ev = emit_expr(ctx, l, node->as.if_stmt.else_branch);
            if (ev.reg >= 0 && ev.kind == OVK_INT)
                osb_appendf(&ctx->sb, "    _t%d = _t%d;\n", r, ev.reg);
        }
        osb_append(&ctx->sb, "    }\n");
        return (OVal){r, OVK_INT};
    }

    case AST_BLOCK: {
        int last_r = -1; OVK last_k = OVK_INT;
        for (int i = 0; i < node->as.block.count; i++) {
            ASTNode *s = node->as.block.statements[i];
            if (s->type == AST_RETURN) {
                if (s->as.return_stmt.value)
                    emit_expr(ctx, l, s->as.return_stmt.value);
                osb_append(&ctx->sb, "    return;\n");
                return (OVal){-1, OVK_INT};
            }
            if (s->type == AST_LET) {
                OVal ev;
                if (s->as.let.value) {
                    ev = emit_expr(ctx, l, s->as.let.value);
                } else {
                    ev.reg = new_t(ctx); ev.kind = OVK_INT;
                    osb_appendf(&ctx->sb, "    long _t%d = 0LL;\n", ev.reg);
                }
                if (ev.reg < 0) return OVAL_ERR;
                oloc_add(l, s->as.let.name, ev.reg, ev.kind);
                last_r = ev.reg; last_k = ev.kind;
                continue;
            }
            OVal sv = emit_expr(ctx, l, s);
            if (ctx->err) return OVAL_ERR;
            last_r = sv.reg; last_k = sv.kind;
        }
        return (OVal){last_r, last_k};
    }

    default:
        ocl_err(ctx, "unsupported AST node %d in gpu fn", (int)node->type);
        return OVAL_ERR;
    }
}

/* ── Emit a single gpu fn as an OpenCL kernel ────────────────────────────── */
static void emit_kernel(OCtx *ctx, ASTNode *fn) {
    if (!fn || fn->type != AST_FUNCTION || !fn->as.function.is_gpu) return;

    ctx->next_t = 0;
    ctx->next_f = 0;

    const char *name = fn->as.function.name;
    int np = fn->as.function.param_count;

    /* Pointer-param detection */
    bool is_ptr[64];
    const char *pnames[64];
    for (int i = 0; i < np && i < 64; i++) {
        is_ptr[i] = false;
        pnames[i] = fn->as.function.params[i].name;
    }
    if (fn->as.function.body)
        collect_ptr_params(fn->as.function.body, is_ptr, pnames, np < 64 ? np : 64);

    /* Kernel signature */
    osb_appendf(&ctx->sb, "\n__kernel void %s(", name);
    for (int i = 0; i < np; i++) {
        Parameter *p = &fn->as.function.params[i];
        if (i > 0) osb_append(&ctx->sb, ",\n    ");
        if (p->type == TYPE_FLOAT) {
            osb_appendf(&ctx->sb, "double %s", p->name);
        } else if (is_ptr[i]) {
            osb_appendf(&ctx->sb, "__global char* %s", p->name);
        } else {
            osb_appendf(&ctx->sb, "long %s", p->name);
        }
    }
    osb_append(&ctx->sb, ")\n{\n");

    /* Bind params into local variable map */
    OLocals l; oloc_init(&l);
    for (int i = 0; i < np; i++) {
        Parameter *p = &fn->as.function.params[i];
        OVK k = (p->type == TYPE_FLOAT) ? OVK_FLOAT : OVK_INT;
        if (k == OVK_FLOAT) {
            int r = new_f(ctx);
            osb_appendf(&ctx->sb, "    double _f%d = %s;\n", r, p->name);
            oloc_add(&l, p->name, r, OVK_FLOAT);
        } else if (is_ptr[i]) {
            /* __global char* → store as long for byte-offset arithmetic */
            int r = new_t(ctx);
            osb_appendf(&ctx->sb, "    long _t%d = (long)(ulong)%s;\n", r, p->name);
            oloc_add(&l, p->name, r, OVK_INT);
        } else {
            int r = new_t(ctx);
            osb_appendf(&ctx->sb, "    long _t%d = %s;\n", r, p->name);
            oloc_add(&l, p->name, r, OVK_INT);
        }
    }

    if (fn->as.function.body)
        emit_expr(ctx, &l, fn->as.function.body);

    osb_append(&ctx->sb, "}\n");
}

/* ── Public API ──────────────────────────────────────────────────────────── */
int ocl_backend_emit_fp(ASTNode *root, FILE *out,
                        const char *source_file, bool verbose) {
    if (!root || !out) return 1;

    OCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    osb_init(&ctx.sb);
    ctx.verbose = verbose;
    ctx.src = source_file ? source_file : "<unknown>";

    /* File header */
    osb_appendf(&ctx.sb, "/* nanolang OpenCL C output: %s */\n", ctx.src);
    osb_append(&ctx.sb,  "/* Generated by nanoc --target opencl */\n");
    osb_append(&ctx.sb,  "/* Load: clCreateProgramWithSource + clBuildProgram */\n\n");
    osb_append(&ctx.sb,  "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics    : enable\n");
    osb_append(&ctx.sb,  "#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable\n");
    osb_append(&ctx.sb,  "#pragma OPENCL EXTENSION cl_khr_fp64                   : enable\n\n");

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
        fprintf(stderr, "OpenCL backend error: %s\n", ctx.errmsg);
        osb_free(&ctx.sb);
        return 1;
    }
    if (count == 0 && verbose)
        fprintf(stderr, "[opencl] warning: no `gpu fn` found in %s\n", ctx.src);

    fputs(ctx.sb.buf, out);
    if (verbose)
        fprintf(stderr, "[opencl] emitted %d kernel(s)\n", count);

    osb_free(&ctx.sb);
    return 0;
}

int ocl_backend_emit(ASTNode *root, const char *output_path,
                     const char *source_file, bool verbose) {
    if (!output_path) return ocl_backend_emit_fp(root, stdout, source_file, verbose);
    FILE *f = fopen(output_path, "w");
    if (!f) { fprintf(stderr, "OpenCL: cannot open %s\n", output_path); return 1; }
    int rc = ocl_backend_emit_fp(root, f, source_file, verbose);
    fclose(f);
    return rc;
}
