/*
 * Nano Shell — bounded laboratory frontend. I emit verified NanoISA
 * function bodies and run typed i64 pipelines in the host. See
 * docs/SHELL.md.
 */

#include "shell.h"

#include "nanoisa/assembler.h"
#include "nanoisa/frontend.h"
#include "nanoisa/isa.h"
#include "nanoisa/nvm_format.h"
#include "nanovm/heap.h"
#include "nanovm/vm.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SH_NAME 64
#define SH_MAX 24
#define SH_ERR NL_SH_ERR_SIZE
#define SH_STMT 32
#define SH_ARG 8
#define SH_STAGE 8

typedef enum {
    TK_EOF, TK_INT, TK_ID, TK_STRING,
    TK_LPAREN, TK_RPAREN, TK_LBRACE, TK_RBRACE, TK_EQ, TK_BAR,
    TK_PLUS, TK_MINUS,
    TK_NEED, TK_FILES, TK_PROC, TK_NET, TK_SERVICE, TK_STREAM, TK_REMOTE,
    TK_FN, TK_MAIN, TK_PARSE, TK_READ, TK_RUN, TK_CONNECT, TK_CANCEL
} TkKind;

typedef struct {
    TkKind kind;
    int64_t i;
    char name[SH_NAME];
} Tok;

typedef enum {
    E_INT, E_TEXT, E_VAR, E_BIN, E_PARSE,
    E_READ, E_RUN, E_CONNECT, E_SERVICE, E_STREAM, E_REMOTE
} EKind;

typedef enum { ST_BIND, ST_PIPE, ST_CANCEL } SKind;
typedef enum { V_INT, V_TEXT } VKind;

typedef struct Expr Expr;
struct Expr {
    EKind kind;
    int64_t i;
    char name[SH_NAME];
    int bin;
    Expr *x, *y;
};

typedef struct {
    char fn[SH_NAME];
    Expr *args[SH_ARG];
    int nargs;
} Stage;

typedef struct {
    Expr *left;
    Stage stages[SH_STAGE];
    int nstage;
} Pipe;

typedef struct {
    SKind kind;
    char name[SH_NAME];
    Pipe pipe;
} Stmt;

typedef struct {
    char name[SH_NAME];
    char params[SH_ARG][SH_NAME];
    int nparam;
    Expr *body;
    char asm_name[SH_NAME];
    int fn_idx;
} FnDef;

typedef struct {
    char *p;
    size_t n, cap;
} Buf;

typedef struct {
    const char *src;
    const char *lx;
    Tok tok;
    char err[SH_ERR];
    void **heap;
    uint32_t nheap, capheap;
    int cap_files, cap_proc, cap_net, cap_service, cap_stream, cap_remote;
    FnDef fns[SH_MAX];
    int nfn_def;
    Stmt stmts[SH_STMT];
    int nstmt;
    Buf code[SH_MAX];
    int nfn;
} Cc;

typedef struct {
    VKind kind;
    int64_t i;
    char text[SH_NAME];
} Val;

typedef struct {
    char names[SH_STMT][SH_NAME];
    Val vals[SH_STMT];
    int n;
    int cancelled;
    Cc *cc;
    NvmModule *mod;
    VmState vm;
    int fn_idx[SH_MAX];
} Rt;

static NlShellServiceFn g_service;

void nl_shell_set_service(NlShellServiceFn fn) {
    g_service = fn;
}

#ifdef __GNUC__
__attribute__((format(printf, 2, 3)))
#endif
static int cc_fail(Cc *cc, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(cc->err, sizeof cc->err, fmt, ap);
    va_end(ap);
    return -1;
}

static void *cc_alloc(Cc *cc, size_t n) {
    void *p = calloc(1, n);
    void **h;
    uint32_t cap;
    if (!p) return NULL;
    if (cc->nheap >= cc->capheap) {
        cap = cc->capheap ? cc->capheap * 2 : 64;
        h = realloc(cc->heap, cap * sizeof(void *));
        if (!h) { free(p); return NULL; }
        cc->heap = h;
        cc->capheap = cap;
    }
    cc->heap[cc->nheap++] = p;
    return p;
}

static void cc_free(Cc *cc) {
    uint32_t i;
    for (i = 0; i < cc->nheap; i++) free(cc->heap[i]);
    free(cc->heap);
    for (i = 0; i < (uint32_t)cc->nfn; i++) free(cc->code[i].p);
    memset(cc, 0, sizeof *cc);
}

static int buf_grow(Buf *b, size_t need) {
    char *p;
    size_t cap;
    if (need <= b->cap) return 0;
    cap = b->cap ? b->cap : 256;
    while (cap < need) cap *= 2;
    p = realloc(b->p, cap);
    if (!p) return -1;
    b->p = p;
    b->cap = cap;
    return 0;
}

#ifdef __GNUC__
__attribute__((format(printf, 2, 3)))
#endif
static int buf_printf(Buf *b, const char *fmt, ...) {
    va_list ap;
    int need;
    va_start(ap, fmt);
    need = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (need < 0) return -1;
    if (buf_grow(b, b->n + (size_t)need + 1) < 0) return -1;
    va_start(ap, fmt);
    vsnprintf(b->p + b->n, b->cap - b->n, fmt, ap);
    va_end(ap);
    b->n += (size_t)need;
    return 0;
}

static Expr *ex_new(Cc *cc, EKind k) {
    Expr *e = cc_alloc(cc, sizeof *e);
    if (e) e->kind = k;
    return e;
}

static void skip_ws(Cc *cc) {
    for (;;) {
        while (*cc->lx && isspace((unsigned char)*cc->lx)) cc->lx++;
        if (cc->lx[0] == '/' && cc->lx[1] == '/') {
            while (*cc->lx && *cc->lx != '\n') cc->lx++;
            continue;
        }
        break;
    }
}

static int kw_eq(const char *s, const char *k) { return strcmp(s, k) == 0; }

static void lex(Cc *cc) {
    const char *p;
    size_t n;
    skip_ws(cc);
    p = cc->lx;
    memset(&cc->tok, 0, sizeof cc->tok);
    if (!*p) { cc->tok.kind = TK_EOF; return; }
    if (*p == '"') {
        p++;
        n = 0;
        while (*p && *p != '"') {
            if (n + 1 >= SH_NAME) { cc->tok.kind = TK_EOF; cc->lx = p; return; }
            cc->tok.name[n++] = *p++;
        }
        cc->tok.name[n] = '\0';
        if (*p == '"') p++;
        cc->tok.kind = TK_STRING;
        cc->lx = p;
        return;
    }
    if (*p == '-' && isdigit((unsigned char)p[1])) {
        cc->tok.i = strtoll(p, (char **)&cc->lx, 10);
        cc->tok.kind = TK_INT;
        return;
    }
    if (isdigit((unsigned char)*p)) {
        cc->tok.i = strtoll(p, (char **)&cc->lx, 10);
        cc->tok.kind = TK_INT;
        return;
    }
    if (isalpha((unsigned char)*p) || *p == '_') {
        n = 0;
        while (isalnum((unsigned char)*p) || *p == '_') {
            if (n + 1 < SH_NAME) cc->tok.name[n++] = *p;
            p++;
        }
        cc->tok.name[n] = '\0';
        cc->lx = p;
        if (kw_eq(cc->tok.name, "need")) cc->tok.kind = TK_NEED;
        else if (kw_eq(cc->tok.name, "files")) cc->tok.kind = TK_FILES;
        else if (kw_eq(cc->tok.name, "proc")) cc->tok.kind = TK_PROC;
        else if (kw_eq(cc->tok.name, "net")) cc->tok.kind = TK_NET;
        else if (kw_eq(cc->tok.name, "service")) cc->tok.kind = TK_SERVICE;
        else if (kw_eq(cc->tok.name, "stream")) cc->tok.kind = TK_STREAM;
        else if (kw_eq(cc->tok.name, "remote")) cc->tok.kind = TK_REMOTE;
        else if (kw_eq(cc->tok.name, "fn")) cc->tok.kind = TK_FN;
        else if (kw_eq(cc->tok.name, "main")) cc->tok.kind = TK_MAIN;
        else if (kw_eq(cc->tok.name, "parse")) cc->tok.kind = TK_PARSE;
        else if (kw_eq(cc->tok.name, "read")) cc->tok.kind = TK_READ;
        else if (kw_eq(cc->tok.name, "run")) cc->tok.kind = TK_RUN;
        else if (kw_eq(cc->tok.name, "connect")) cc->tok.kind = TK_CONNECT;
        else if (kw_eq(cc->tok.name, "cancel")) cc->tok.kind = TK_CANCEL;
        else cc->tok.kind = TK_ID;
        return;
    }
    cc->lx = p + 1;
    switch (*p) {
    case '(': cc->tok.kind = TK_LPAREN; return;
    case ')': cc->tok.kind = TK_RPAREN; return;
    case '{': cc->tok.kind = TK_LBRACE; return;
    case '}': cc->tok.kind = TK_RBRACE; return;
    case '=': cc->tok.kind = TK_EQ; return;
    case '|': cc->tok.kind = TK_BAR; return;
    case '+': cc->tok.kind = TK_PLUS; return;
    case '-': cc->tok.kind = TK_MINUS; return;
    default:
        cc->tok.kind = TK_EOF;
        snprintf(cc->err, sizeof cc->err, "I do not know '%c'", *p);
        return;
    }
}

static int have(Cc *cc, TkKind k) { return cc->tok.kind == k; }

static int eat(Cc *cc, TkKind k) {
    if (!have(cc, k)) return 0;
    lex(cc);
    return 1;
}

static Expr *parse_expr(Cc *cc);

static Expr *parse_unary(Cc *cc) {
    Expr *e, *inner;
    EKind k;
    if (have(cc, TK_INT)) {
        e = ex_new(cc, E_INT);
        if (!e) return NULL;
        e->i = cc->tok.i;
        lex(cc);
        return e;
    }
    if (have(cc, TK_STRING)) {
        e = ex_new(cc, E_TEXT);
        if (!e) return NULL;
        snprintf(e->name, sizeof e->name, "%s", cc->tok.name);
        lex(cc);
        return e;
    }
    if (have(cc, TK_ID)) {
        e = ex_new(cc, E_VAR);
        if (!e) return NULL;
        snprintf(e->name, sizeof e->name, "%s", cc->tok.name);
        lex(cc);
        return e;
    }
    if (eat(cc, TK_LPAREN)) {
        e = parse_expr(cc);
        if (!e) return NULL;
        if (!eat(cc, TK_RPAREN)) {
            cc_fail(cc, "I expected ')'");
            return NULL;
        }
        return e;
    }
    if (eat(cc, TK_PARSE)) k = E_PARSE;
    else if (eat(cc, TK_READ)) k = E_READ;
    else if (eat(cc, TK_RUN)) k = E_RUN;
    else if (eat(cc, TK_CONNECT)) k = E_CONNECT;
    else if (eat(cc, TK_SERVICE)) k = E_SERVICE;
    else if (eat(cc, TK_STREAM)) k = E_STREAM;
    else if (eat(cc, TK_REMOTE)) k = E_REMOTE;
    else {
        cc_fail(cc, "I expected an expression");
        return NULL;
    }
    inner = parse_unary(cc);
    if (!inner) return NULL;
    e = ex_new(cc, k);
    if (!e) return NULL;
    e->x = inner;
    return e;
}

static Expr *parse_expr(Cc *cc) {
    Expr *left = parse_unary(cc);
    if (!left) return NULL;
    while (have(cc, TK_PLUS) || have(cc, TK_MINUS)) {
        Expr *e;
        int bin = cc->tok.kind;
        lex(cc);
        e = ex_new(cc, E_BIN);
        if (!e) return NULL;
        e->bin = bin;
        e->x = left;
        e->y = parse_unary(cc);
        if (!e->y) return NULL;
        left = e;
    }
    return left;
}

static int parse_pipe_from(Cc *cc, Pipe *pipe, Expr *left) {
    memset(pipe, 0, sizeof *pipe);
    pipe->left = left;
    while (eat(cc, TK_BAR)) {
        Stage *st;
        if (pipe->nstage >= SH_STAGE) return cc_fail(cc, "I refuse too many pipeline stages");
        st = &pipe->stages[pipe->nstage++];
        memset(st, 0, sizeof *st);
        if (have(cc, TK_ID)) {
            snprintf(st->fn, sizeof st->fn, "%s", cc->tok.name);
            lex(cc);
        } else if (eat(cc, TK_PARSE)) {
            snprintf(st->fn, sizeof st->fn, "parse");
        } else {
            return cc_fail(cc, "I expected a pipeline stage");
        }
        while (have(cc, TK_INT) || have(cc, TK_STRING) || have(cc, TK_ID) ||
               have(cc, TK_LPAREN) || have(cc, TK_PARSE) || have(cc, TK_READ) ||
               have(cc, TK_RUN) || have(cc, TK_CONNECT) || have(cc, TK_SERVICE) ||
               have(cc, TK_STREAM) || have(cc, TK_REMOTE)) {
            if (st->nargs >= SH_ARG) return cc_fail(cc, "I refuse too many arguments");
            st->args[st->nargs] = parse_unary(cc);
            if (!st->args[st->nargs]) return -1;
            st->nargs++;
        }
    }
    return 0;
}

static int parse_pipe(Cc *cc, Pipe *pipe) {
    Expr *left = parse_expr(cc);
    if (!left) return -1;
    return parse_pipe_from(cc, pipe, left);
}

static int fn_lookup(Cc *cc, const char *n) {
    int i;
    for (i = 0; i < cc->nfn_def; i++) {
        if (strcmp(cc->fns[i].name, n) == 0) return i;
    }
    return -1;
}

static int parse_need(Cc *cc) {
    if (eat(cc, TK_FILES)) { cc->cap_files = 1; return 0; }
    if (eat(cc, TK_PROC)) { cc->cap_proc = 1; return 0; }
    if (eat(cc, TK_NET)) { cc->cap_net = 1; return 0; }
    if (eat(cc, TK_SERVICE)) { cc->cap_service = 1; return 0; }
    if (eat(cc, TK_STREAM)) { cc->cap_stream = 1; return 0; }
    if (eat(cc, TK_REMOTE)) { cc->cap_remote = 1; return 0; }
    return cc_fail(cc, "I expected files, proc, net, service, stream, or remote");
}

static int parse_program(Cc *cc) {
    int saw_main = 0;
    cc->lx = cc->src;
    lex(cc);
    if (cc->err[0]) return -1;
    while (!have(cc, TK_EOF)) {
        if (eat(cc, TK_NEED)) {
            if (parse_need(cc) < 0) return -1;
            continue;
        }
        if (eat(cc, TK_FN)) {
            FnDef *f;
            size_t function_name_len;
            if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a function name");
            if (cc->nfn_def >= SH_MAX) return cc_fail(cc, "I refuse too many functions");
            function_name_len = strlen(cc->tok.name);
            if (function_name_len > SH_NAME - 4)
                return cc_fail(cc, "I refuse a function name longer than %d bytes", SH_NAME - 4);
            f = &cc->fns[cc->nfn_def++];
            memset(f, 0, sizeof *f);
            snprintf(f->name, sizeof f->name, "%s", cc->tok.name);
            memcpy(f->asm_name, "sh_", 3);
            memcpy(f->asm_name + 3, f->name, function_name_len + 1);
            lex(cc);
            while (have(cc, TK_ID)) {
                if (f->nparam >= SH_ARG) return cc_fail(cc, "I refuse too many parameters");
                snprintf(f->params[f->nparam], SH_NAME, "%s", cc->tok.name);
                f->nparam++;
                lex(cc);
            }
            if (!eat(cc, TK_EQ)) return cc_fail(cc, "I expected '='");
            f->body = parse_expr(cc);
            if (!f->body) return -1;
            continue;
        }
        if (eat(cc, TK_MAIN)) {
            if (saw_main) return cc_fail(cc, "I refuse a second main");
            saw_main = 1;
            if (!eat(cc, TK_LBRACE)) return cc_fail(cc, "I expected '{'");
            while (!have(cc, TK_RBRACE) && !have(cc, TK_EOF)) {
                Stmt *s;
                if (cc->nstmt >= SH_STMT) return cc_fail(cc, "I refuse too many statements");
                if (eat(cc, TK_CANCEL)) {
                    s = &cc->stmts[cc->nstmt++];
                    memset(s, 0, sizeof *s);
                    s->kind = ST_CANCEL;
                    continue;
                }
                if (have(cc, TK_ID)) {
                    char nm[SH_NAME];
                    snprintf(nm, sizeof nm, "%s", cc->tok.name);
                    lex(cc);
                    if (eat(cc, TK_EQ)) {
                        s = &cc->stmts[cc->nstmt++];
                        memset(s, 0, sizeof *s);
                        s->kind = ST_BIND;
                        snprintf(s->name, sizeof s->name, "%s", nm);
                        if (parse_pipe(cc, &s->pipe) < 0) return -1;
                        continue;
                    }
                    {
                        Expr *e = ex_new(cc, E_VAR);
                        if (!e) return -1;
                        snprintf(e->name, sizeof e->name, "%s", nm);
                        s = &cc->stmts[cc->nstmt++];
                        memset(s, 0, sizeof *s);
                        s->kind = ST_PIPE;
                        if (parse_pipe_from(cc, &s->pipe, e) < 0) return -1;
                        continue;
                    }
                }
                s = &cc->stmts[cc->nstmt++];
                memset(s, 0, sizeof *s);
                s->kind = ST_PIPE;
                if (parse_pipe(cc, &s->pipe) < 0) return -1;
            }
            if (!eat(cc, TK_RBRACE)) return cc_fail(cc, "I expected '}'");
            continue;
        }
        return cc_fail(cc, "I expected need, fn, or main");
    }
    if (!saw_main) return cc_fail(cc, "I expected main");
    return 0;
}

#ifdef __GNUC__
__attribute__((format(printf, 3, 4)))
#endif
static int emit_line(Buf *b, int indent, const char *fmt, ...) {
    va_list ap;
    int need;
    va_start(ap, fmt);
    need = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (need < 0) return -1;
    if (buf_grow(b, b->n + (size_t)need + 4) < 0) return -1;
    if (indent) {
        b->p[b->n++] = ' ';
        b->p[b->n++] = ' ';
        b->p[b->n] = '\0';
    }
    va_start(ap, fmt);
    vsnprintf(b->p + b->n, b->cap - b->n, fmt, ap);
    va_end(ap);
    b->n += (size_t)need;
    return buf_printf(b, "\n");
}

static int local_of(FnDef *f, const char *n) {
    int i;
    for (i = 0; i < f->nparam; i++) {
        if (strcmp(f->params[i], n) == 0) return i;
    }
    return -1;
}

static int compile_value(Cc *cc, Buf *b, FnDef *f, Expr *e) {
    int i;
    if (!e) return cc_fail(cc, "I expected an expression");
    switch (e->kind) {
    case E_INT:
        return emit_line(b, 1, "PUSH_I64 %lld", (long long)e->i);
    case E_VAR:
        i = local_of(f, e->name);
        if (i < 0) return cc_fail(cc, "I do not know %s", e->name);
        return emit_line(b, 1, "LOAD_LOCAL %d", i);
    case E_BIN:
        if (compile_value(cc, b, f, e->x) < 0) return -1;
        if (compile_value(cc, b, f, e->y) < 0) return -1;
        return emit_line(b, 1, "%s", e->bin == TK_MINUS ? "I64_SUB" : "I64_ADD");
    case E_TEXT:
        return cc_fail(cc, "text stays in the host until parse");
    case E_PARSE:
    case E_READ:
    case E_RUN:
    case E_CONNECT:
    case E_SERVICE:
    case E_STREAM:
    case E_REMOTE:
        return cc_fail(cc, "host adapters stay out of compiled functions");
    }
    return cc_fail(cc, "I cannot compile that");
}

static int compile_all(Cc *cc) {
    int i;
    for (i = 0; i < cc->nfn_def; i++) {
        FnDef *f = &cc->fns[i];
        if (cc->nfn >= SH_MAX) return cc_fail(cc, "I refuse too many functions");
        if (compile_value(cc, &cc->code[cc->nfn], f, f->body) < 0) return -1;
        if (emit_line(&cc->code[cc->nfn], 1, "RET") < 0) return -1;
        f->fn_idx = cc->nfn;
        cc->nfn++;
    }
    return 0;
}

static int build_asm(Cc *cc, Buf *out) {
    int i;
    if (buf_printf(out, ".flag has_main\n.entry _sh_main\n") < 0) return -1;
    if (buf_printf(out, ".string \"nano shell\"\n") < 0) return -1;
    for (i = 0; i < cc->nfn_def; i++) {
        FnDef *f = &cc->fns[i];
        int locals = f->nparam + 4;
        if (buf_printf(out, ".function %s %d %d 0 int 1\n",
                       f->asm_name, f->nparam, locals) < 0)
            return -1;
        if (cc->code[i].p && buf_printf(out, "%s", cc->code[i].p) < 0) return -1;
        if (buf_printf(out, ".end\n") < 0) return -1;
    }
    if (buf_printf(out, ".function _sh_main 0 2 0 int 1\n  PUSH_I64 0\n  RET\n.end\n") < 0)
        return -1;
    return 0;
}

static int bind_val(Rt *rt, const char *n, Val v) {
    if (rt->n >= SH_STMT) return -1;
    snprintf(rt->names[rt->n], SH_NAME, "%s", n);
    rt->vals[rt->n] = v;
    rt->n++;
    return 0;
}

static int lookup_val(Rt *rt, const char *n, Val *out) {
    int i;
    for (i = rt->n - 1; i >= 0; i--) {
        if (strcmp(rt->names[i], n) == 0) { *out = rt->vals[i]; return 0; }
    }
    return -1;
}

static int lookup_fn_idx(NvmModule *mod, const char *name) {
    uint32_t f;
    for (f = 0; f < mod->function_count; f++) {
        const char *nm = nvm_get_string(mod, mod->functions[f].name_idx);
        if (nm && strcmp(nm, name) == 0) return (int)f;
    }
    return -1;
}

static int eval_expr(Rt *rt, Expr *e, Val *out);

static int as_int(Rt *rt, Val v, int64_t *out) {
    if (v.kind != V_INT)
        return cc_fail(rt->cc, "text cannot travel a typed pipeline without parse");
    *out = v.i;
    return 0;
}

static int require_cap(Rt *rt, int granted, const char *cap, const char *what) {
    if (!granted) return cc_fail(rt->cc, "I refuse %s without cap:%s", what, cap);
    return cc_fail(rt->cc, "I refuse a host %s in this subset", what);
}

static int eval_expr(Rt *rt, Expr *e, Val *out) {
    Val a, b;
    int64_t ia, ib;
    char *end;
    if (!e) return cc_fail(rt->cc, "I expected an expression");
    if (rt->cancelled) return cc_fail(rt->cc, "cancel stopped the pipeline");
    memset(out, 0, sizeof *out);
    switch (e->kind) {
    case E_INT:
        out->kind = V_INT;
        out->i = e->i;
        return 0;
    case E_TEXT:
        out->kind = V_TEXT;
        snprintf(out->text, sizeof out->text, "%s", e->name);
        return 0;
    case E_VAR:
        if (lookup_val(rt, e->name, out) < 0)
            return cc_fail(rt->cc, "I do not know %s", e->name);
        return 0;
    case E_BIN:
        if (eval_expr(rt, e->x, &a) < 0) return -1;
        if (eval_expr(rt, e->y, &b) < 0) return -1;
        if (as_int(rt, a, &ia) < 0) return -1;
        if (as_int(rt, b, &ib) < 0) return -1;
        out->kind = V_INT;
        out->i = e->bin == TK_MINUS ? ia - ib : ia + ib;
        return 0;
    case E_PARSE:
        if (eval_expr(rt, e->x, &a) < 0) return -1;
        if (a.kind != V_TEXT)
            return cc_fail(rt->cc, "parse is a text adapter");
        out->kind = V_INT;
        out->i = strtoll(a.text, &end, 10);
        if (end == a.text || *end)
            return cc_fail(rt->cc, "I cannot parse '%s' as int", a.text);
        return 0;
    case E_READ:
        if (eval_expr(rt, e->x, &a) < 0) return -1;
        return require_cap(rt, rt->cc->cap_files, "files", "file");
    case E_RUN:
        if (eval_expr(rt, e->x, &a) < 0) return -1;
        return require_cap(rt, rt->cc->cap_proc, "proc", "process");
    case E_CONNECT:
        if (eval_expr(rt, e->x, &a) < 0) return -1;
        return require_cap(rt, rt->cc->cap_net, "net", "network");
    case E_SERVICE: {
        int64_t id = 0, hv = 0;
        char hookerr[NL_SH_ERR_SIZE];
        if (eval_expr(rt, e->x, &a) < 0) return -1;
        if (!rt->cc->cap_service)
            return cc_fail(rt->cc, "I refuse service without cap:service");
        if (!g_service)
            return cc_fail(rt->cc, "I refuse a host service in this subset");
        if (as_int(rt, a, &id) < 0) return -1;
        hookerr[0] = '\0';
        if (!g_service(id, &hv, hookerr, sizeof hookerr))
            return cc_fail(rt->cc, "%s", hookerr[0] ? hookerr : "service failed");
        out->kind = V_INT;
        out->i = hv;
        return 0;
    }
    case E_STREAM:
        if (eval_expr(rt, e->x, &a) < 0) return -1;
        return require_cap(rt, rt->cc->cap_stream, "stream", "stream");
    case E_REMOTE:
        if (eval_expr(rt, e->x, &a) < 0) return -1;
        return require_cap(rt, rt->cc->cap_remote, "remote", "remote");
    }
    return cc_fail(rt->cc, "I cannot evaluate that");
}

static int call_fn(Rt *rt, int def, Val *args, int nargs, Val *out) {
    FnDef *f = &rt->cc->fns[def];
    NanoValue av[SH_ARG];
    NanoValue ret;
    VmResult r;
    int i;
    if (nargs != f->nparam)
        return cc_fail(rt->cc, "%s takes %d arguments", f->name, f->nparam);
    for (i = 0; i < nargs; i++) {
        int64_t v = 0;
        if (as_int(rt, args[i], &v) < 0) return -1;
        av[i] = val_int(v);
    }
    memset(&ret, 0, sizeof ret);
    r = vm_invoke(&rt->vm, (uint32_t)rt->fn_idx[def], av, (uint16_t)nargs, &ret);
    if (r != VM_OK) return cc_fail(rt->cc, "function trapped");
    if (ret.tag != TAG_INT) {
        vm_release(&rt->vm.heap, ret);
        return cc_fail(rt->cc, "function did not return int");
    }
    out->kind = V_INT;
    out->i = ret.as.i64;
    vm_release(&rt->vm.heap, ret);
    return 0;
}

static int eval_pipe(Rt *rt, Pipe *pipe, Val *out) {
    Val cur;
    int i, j;
    if (eval_expr(rt, pipe->left, &cur) < 0) return -1;
    for (i = 0; i < pipe->nstage; i++) {
        Stage *st = &pipe->stages[i];
        Val args[SH_ARG + 1];
        int def;
        if (rt->cancelled) return cc_fail(rt->cc, "cancel stopped the pipeline");
        if (strcmp(st->fn, "parse") == 0) {
            Expr fake;
            memset(&fake, 0, sizeof fake);
            fake.kind = E_PARSE;
            if (st->nargs == 0) {
                Expr hold;
                memset(&hold, 0, sizeof hold);
                if (cur.kind == V_TEXT) {
                    hold.kind = E_TEXT;
                    snprintf(hold.name, sizeof hold.name, "%s", cur.text);
                } else {
                    hold.kind = E_INT;
                    hold.i = cur.i;
                }
                fake.x = &hold;
                if (eval_expr(rt, &fake, &cur) < 0) return -1;
            } else if (st->nargs == 1) {
                fake.x = st->args[0];
                if (eval_expr(rt, &fake, &cur) < 0) return -1;
            } else {
                return cc_fail(rt->cc, "parse takes one value");
            }
            continue;
        }
        def = fn_lookup(rt->cc, st->fn);
        if (def < 0) return cc_fail(rt->cc, "I do not know function %s", st->fn);
        args[0] = cur;
        for (j = 0; j < st->nargs; j++) {
            if (eval_expr(rt, st->args[j], &args[j + 1]) < 0) return -1;
        }
        if (call_fn(rt, def, args, st->nargs + 1, &cur) < 0) return -1;
    }
    *out = cur;
    return 0;
}

static int run_rt(Cc *cc, NvmModule *mod, int64_t *out) {
    Rt rt;
    int i;
    Val last;
    int have_last = 0;
    memset(&rt, 0, sizeof rt);
    memset(&last, 0, sizeof last);
    rt.cc = cc;
    rt.mod = mod;
    (void)rt.mod;
    for (i = 0; i < cc->nfn_def; i++) {
        int fn = lookup_fn_idx(mod, cc->fns[i].asm_name);
        if (fn < 0) return cc_fail(cc, "I lost function %s", cc->fns[i].asm_name);
        cc->fns[i].fn_idx = fn;
        rt.fn_idx[i] = fn;
    }
    vm_init(&rt.vm, mod);
    for (i = 0; i < cc->nstmt; i++) {
        Stmt *s = &cc->stmts[i];
        Val v;
        memset(&v, 0, sizeof v);
        if (s->kind == ST_CANCEL) {
            rt.cancelled = 1;
            continue;
        }
        if (eval_pipe(&rt, &s->pipe, &v) < 0) {
            vm_destroy(&rt.vm);
            return -1;
        }
        if (s->kind == ST_BIND) {
            if (bind_val(&rt, s->name, v) < 0) {
                vm_destroy(&rt.vm);
                return cc_fail(cc, "too many names");
            }
        }
        last = v;
        have_last = 1;
    }
    if (!have_last) {
        vm_destroy(&rt.vm);
        return cc_fail(cc, rt.cancelled ? "cancel stopped the pipeline"
                                       : "I expected a result");
    }
    if (as_int(&rt, last, out) < 0) {
        vm_destroy(&rt.vm);
        return -1;
    }
    vm_destroy(&rt.vm);
    return 0;
}

static void attach_debug(NvmModule *mod) {
    if (!mod) return;
    mod->header.flags |= NVM_FLAG_DEBUG_INFO;
    nvm_add_debug_entry(mod, 0, 1, 1);
}

static int prepare(Cc *cc, const char *src) {
    memset(cc, 0, sizeof *cc);
    cc->src = src;
    if (parse_program(cc) < 0) return -1;
    if (compile_all(cc) < 0) return -1;
    return 0;
}

static NvmModule *assemble_cc(Cc *cc, char *err, size_t errlen) {
    Buf asmbuf;
    AsmResult ar;
    NvmModule *mod;
    memset(&asmbuf, 0, sizeof asmbuf);
    if (build_asm(cc, &asmbuf) < 0) {
        if (err && errlen) snprintf(err, errlen, "I ran out of memory while assembling");
        free(asmbuf.p);
        return NULL;
    }
    if (getenv("NL_SH_TRACE")) fprintf(stderr, "%s", asmbuf.p ? asmbuf.p : "");
    memset(&ar, 0, sizeof ar);
    mod = asm_assemble(asmbuf.p, &ar);
    if (!mod) {
        if (err && errlen)
            snprintf(err, errlen, "assembler: %s (line %u)", ar.message, ar.line);
        free(asmbuf.p);
        return NULL;
    }
    attach_debug(mod);
    free(asmbuf.p);
    return mod;
}

NvmModule *nl_shell_compile(const char *src, const char *path,
                            char *err, size_t errlen) {
    Cc cc;
    NvmModule *mod;
    (void)path;
    if (!src) {
        if (err && errlen) snprintf(err, errlen, "source is null");
        return NULL;
    }
    if (prepare(&cc, src) < 0) {
        if (err && errlen) snprintf(err, errlen, "%s", cc.err[0] ? cc.err : "compile failed");
        cc_free(&cc);
        return NULL;
    }
    mod = assemble_cc(&cc, err, errlen);
    cc_free(&cc);
    return mod;
}

NlFrontendResult nl_shell_accept(const NvmModule *mod, const char *path) {
    NlFrontendFacts f;
    memset(&f, 0, sizeof f);
    f.language = NL_FE_SHELL;
    f.source_path = (path && path[0]) ? path : "<shell>";
    f.purity = 0;
    f.exhaustiveness = 0;
    f.affine_use = -1;
    f.diagnostics_shared = 1;
    return nl_frontend_accept(mod, &f);
}

int nl_shell_eval_i64(const char *src, int64_t *out,
                      char *err, size_t errlen) {
    Cc cc;
    NvmModule *mod;
    NlFrontendResult acc;
    int64_t v = 0;
    if (!src) {
        if (err && errlen) snprintf(err, errlen, "source is null");
        return 0;
    }
    if (prepare(&cc, src) < 0) {
        if (err && errlen) snprintf(err, errlen, "%s", cc.err[0] ? cc.err : "compile failed");
        cc_free(&cc);
        return 0;
    }
    mod = assemble_cc(&cc, err, errlen);
    if (!mod) { cc_free(&cc); return 0; }
    acc = nl_shell_accept(mod, "<eval>");
    if (!acc.ok) {
        if (err && errlen) snprintf(err, errlen, "%s", acc.error);
        nvm_module_free(mod);
        cc_free(&cc);
        return 0;
    }
    if (run_rt(&cc, mod, &v) < 0) {
        if (err && errlen) snprintf(err, errlen, "%s", cc.err[0] ? cc.err : "run failed");
        nvm_module_free(mod);
        cc_free(&cc);
        return 0;
    }
    if (out) *out = v;
    nvm_module_free(mod);
    cc_free(&cc);
    return 1;
}
