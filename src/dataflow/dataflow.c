/*
 * Nano Dataflow — bounded laboratory frontend. I emit verified NanoISA
 * node bodies and run a deterministic DAG. See docs/DATAFLOW.md.
 */

#include "dataflow.h"

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

#define DF_NAME 64
#define DF_MAX 24
#define DF_ERR NL_DF_ERR_SIZE
#define DF_BUF 8
#define DF_FEED 32
#define DF_LOCAL 16

typedef enum {
    TK_EOF, TK_INT, TK_ID,
    TK_LPAREN, TK_RPAREN, TK_LBRACE, TK_RBRACE, TK_EQ, TK_EQEQ,
    TK_PLUS, TK_MINUS, TK_STAR, TK_SLASH,
    TK_NODE, TK_GRAPH, TK_MAIN, TK_IN, TK_OUT, TK_FEED, TK_DRAIN,
    TK_CANCEL, TK_RETRY, TK_EFFECT, TK_BUFFER, TK_PLACE, TK_REMOTE,
    TK_LOCAL, TK_IF, TK_THEN, TK_ELSE, TK_FAIL, TK_ATTEMPT
} TkKind;

typedef struct {
    TkKind kind;
    int64_t i;
    char name[DF_NAME];
} Tok;

typedef enum { E_INT, E_VAR, E_BIN, E_IF, E_FAIL } EKind;
typedef enum { ST_FEED, ST_DRAIN, ST_CANCEL } SKind;

typedef struct Expr Expr;
struct Expr {
    EKind kind;
    int64_t i;
    char name[DF_NAME];
    int bin;
    Expr *x, *y, *z;
};

typedef struct {
    char name[DF_NAME];
    char params[DF_LOCAL][DF_NAME];
    int nparam;
    int retry;
    char effect[DF_NAME];
    Expr *body;
    char asm_name[DF_NAME];
    int fn_idx;
} NodeDef;

typedef struct {
    char name[DF_NAME];
    int inst;
    int is_in;
    int is_sink;
} StreamDef;

typedef struct {
    int def;
    int in_st[DF_LOCAL];
    int nin;
    int out_st;
    int cancelled;
} InstDef;

typedef struct {
    SKind kind;
    char name[DF_NAME];
    int64_t val;
} Stmt;

typedef struct {
    char *p;
    size_t n, cap;
} Buf;

typedef struct {
    const char *src;
    const char *lx;
    Tok tok;
    char err[DF_ERR];
    void **heap;
    uint32_t nheap, capheap;
    NodeDef nodes[DF_MAX];
    int nnode;
    StreamDef streams[DF_MAX];
    int nstream;
    InstDef insts[DF_MAX];
    int ninst;
    Stmt stmts[DF_FEED];
    int nstmt;
    int sink;
    int bufcap;
    int has_io;
    Buf code[DF_MAX];
    int nfn;
    int label;
} Cc;

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
    skip_ws(cc);
    p = cc->lx;
    memset(&cc->tok, 0, sizeof cc->tok);
    if (!*p) { cc->tok.kind = TK_EOF; return; }
    if (isdigit((unsigned char)*p) || (*p == '-' && isdigit((unsigned char)p[1]))) {
        int64_t v = 0;
        int neg = 0;
        if (*p == '-') { neg = 1; p++; }
        while (isdigit((unsigned char)*p)) { v = v * 10 + (*p - '0'); p++; }
        cc->tok.kind = TK_INT;
        cc->tok.i = neg ? -v : v;
        cc->lx = p;
        return;
    }
    if (isalpha((unsigned char)*p) || *p == '_') {
        int n = 0;
        while ((isalnum((unsigned char)*p) || *p == '_') && n < DF_NAME - 1)
            cc->tok.name[n++] = *p++;
        cc->lx = p;
        if (kw_eq(cc->tok.name, "node")) cc->tok.kind = TK_NODE;
        else if (kw_eq(cc->tok.name, "graph")) cc->tok.kind = TK_GRAPH;
        else if (kw_eq(cc->tok.name, "main")) cc->tok.kind = TK_MAIN;
        else if (kw_eq(cc->tok.name, "in")) cc->tok.kind = TK_IN;
        else if (kw_eq(cc->tok.name, "out")) cc->tok.kind = TK_OUT;
        else if (kw_eq(cc->tok.name, "feed")) cc->tok.kind = TK_FEED;
        else if (kw_eq(cc->tok.name, "drain")) cc->tok.kind = TK_DRAIN;
        else if (kw_eq(cc->tok.name, "cancel")) cc->tok.kind = TK_CANCEL;
        else if (kw_eq(cc->tok.name, "retry")) cc->tok.kind = TK_RETRY;
        else if (kw_eq(cc->tok.name, "effect")) cc->tok.kind = TK_EFFECT;
        else if (kw_eq(cc->tok.name, "buffer")) cc->tok.kind = TK_BUFFER;
        else if (kw_eq(cc->tok.name, "place")) cc->tok.kind = TK_PLACE;
        else if (kw_eq(cc->tok.name, "remote")) cc->tok.kind = TK_REMOTE;
        else if (kw_eq(cc->tok.name, "local")) cc->tok.kind = TK_LOCAL;
        else if (kw_eq(cc->tok.name, "if")) cc->tok.kind = TK_IF;
        else if (kw_eq(cc->tok.name, "then")) cc->tok.kind = TK_THEN;
        else if (kw_eq(cc->tok.name, "else")) cc->tok.kind = TK_ELSE;
        else if (kw_eq(cc->tok.name, "fail")) cc->tok.kind = TK_FAIL;
        else if (kw_eq(cc->tok.name, "attempt")) cc->tok.kind = TK_ATTEMPT;
        else cc->tok.kind = TK_ID;
        return;
    }
    if (p[0] == '=' && p[1] == '=') { cc->tok.kind = TK_EQEQ; cc->lx = p + 2; return; }
    cc->lx = p + 1;
    switch (*p) {
    case '(': cc->tok.kind = TK_LPAREN; break;
    case ')': cc->tok.kind = TK_RPAREN; break;
    case '{': cc->tok.kind = TK_LBRACE; break;
    case '}': cc->tok.kind = TK_RBRACE; break;
    case '=': cc->tok.kind = TK_EQ; break;
    case '+': cc->tok.kind = TK_PLUS; break;
    case '-': cc->tok.kind = TK_MINUS; break;
    case '*': cc->tok.kind = TK_STAR; break;
    case '/': cc->tok.kind = TK_SLASH; break;
    default:
        cc->tok.kind = TK_EOF;
        snprintf(cc->err, sizeof cc->err, "I refuse character '%c'", *p);
        break;
    }
}

static int have(Cc *cc, TkKind k) { return cc->tok.kind == k; }
static int eat(Cc *cc, TkKind k) {
    if (!have(cc, k)) return 0;
    lex(cc);
    return 1;
}

static int node_lookup(Cc *cc, const char *n) {
    int i;
    for (i = 0; i < cc->nnode; i++) {
        if (strcmp(cc->nodes[i].name, n) == 0) return i;
    }
    return -1;
}

static int stream_lookup(Cc *cc, const char *n) {
    int i;
    for (i = 0; i < cc->nstream; i++) {
        if (strcmp(cc->streams[i].name, n) == 0) return i;
    }
    return -1;
}

static int add_stream(Cc *cc, const char *n) {
    int i = stream_lookup(cc, n);
    StreamDef *s;
    if (i >= 0) return i;
    if (cc->nstream >= DF_MAX) return cc_fail(cc, "I refuse too many streams");
    i = cc->nstream++;
    s = &cc->streams[i];
    memset(s, 0, sizeof *s);
    snprintf(s->name, sizeof s->name, "%s", n);
    s->inst = -1;
    return i;
}

static Expr *parse_expr(Cc *cc);

static Expr *parse_atom(Cc *cc) {
    Expr *e;
    if (eat(cc, TK_FAIL)) {
        e = ex_new(cc, E_FAIL);
        return e;
    }
    if (have(cc, TK_INT)) {
        e = ex_new(cc, E_INT);
        if (!e) return NULL;
        e->i = cc->tok.i;
        lex(cc);
        return e;
    }
    if (eat(cc, TK_ATTEMPT)) {
        e = ex_new(cc, E_VAR);
        if (!e) return NULL;
        snprintf(e->name, sizeof e->name, "%s", "attempt");
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
        if (!eat(cc, TK_RPAREN)) { cc_fail(cc, "I expected ')'"); return NULL; }
        return e;
    }
    cc_fail(cc, "I expected an expression");
    return NULL;
}

static Expr *parse_mul(Cc *cc) {
    Expr *left = parse_atom(cc);
    if (!left) return NULL;
    while (have(cc, TK_STAR) || have(cc, TK_SLASH)) {
        Expr *n = ex_new(cc, E_BIN);
        Expr *right;
        if (!n) return NULL;
        n->bin = (int)cc->tok.kind;
        lex(cc);
        right = parse_atom(cc);
        if (!right) return NULL;
        n->x = left;
        n->y = right;
        left = n;
    }
    return left;
}

static Expr *parse_add(Cc *cc) {
    Expr *left = parse_mul(cc);
    if (!left) return NULL;
    while (have(cc, TK_PLUS) || have(cc, TK_MINUS)) {
        Expr *n = ex_new(cc, E_BIN);
        Expr *right;
        if (!n) return NULL;
        n->bin = (int)cc->tok.kind;
        lex(cc);
        right = parse_mul(cc);
        if (!right) return NULL;
        n->x = left;
        n->y = right;
        left = n;
    }
    return left;
}

static Expr *parse_eq(Cc *cc) {
    Expr *left = parse_add(cc);
    if (!left) return NULL;
    if (have(cc, TK_EQEQ)) {
        Expr *n = ex_new(cc, E_BIN);
        Expr *right;
        if (!n) return NULL;
        n->bin = TK_EQEQ;
        lex(cc);
        right = parse_add(cc);
        if (!right) return NULL;
        n->x = left;
        n->y = right;
        return n;
    }
    return left;
}

static Expr *parse_expr(Cc *cc) {
    Expr *c, *t, *f, *n;
    if (!eat(cc, TK_IF)) return parse_eq(cc);
    c = parse_expr(cc);
    if (!c) return NULL;
    if (!eat(cc, TK_THEN)) { cc_fail(cc, "I expected then"); return NULL; }
    t = parse_expr(cc);
    if (!t) return NULL;
    if (!eat(cc, TK_ELSE)) { cc_fail(cc, "I expected else"); return NULL; }
    f = parse_expr(cc);
    if (!f) return NULL;
    n = ex_new(cc, E_IF);
    if (!n) return NULL;
    n->x = c;
    n->y = t;
    n->z = f;
    return n;
}

static int parse_effect_name(Cc *cc, char *dst, size_t dstlen) {
    if (!have(cc, TK_ID)) return cc_fail(cc, "I expected an effect name");
    if (strcmp(cc->tok.name, "IO") != 0 && strcmp(cc->tok.name, "Err") != 0
        && strcmp(cc->tok.name, "State") != 0)
        return cc_fail(cc, "I refuse unknown effect %s", cc->tok.name);
    snprintf(dst, dstlen, "%s", cc->tok.name);
    lex(cc);
    return 0;
}

static int parse_program(Cc *cc) {
    cc->lx = cc->src;
    cc->bufcap = 1;
    cc->sink = -1;
    lex(cc);
    while (!have(cc, TK_EOF)) {
        if (eat(cc, TK_NODE)) {
            NodeDef *n;
            size_t node_name_len;
            if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a node name");
            if (cc->nnode >= DF_MAX) return cc_fail(cc, "I refuse too many nodes");
            node_name_len = strlen(cc->tok.name);
            if (node_name_len > DF_NAME - 4)
                return cc_fail(cc, "I refuse a node name longer than %d bytes", DF_NAME - 4);
            n = &cc->nodes[cc->nnode++];
            memset(n, 0, sizeof *n);
            snprintf(n->name, sizeof n->name, "%s", cc->tok.name);
            memcpy(n->asm_name, "df_", 3);
            memcpy(n->asm_name + 3, cc->tok.name, node_name_len + 1);
            lex(cc);
            while (have(cc, TK_ID)) {
                if (n->nparam >= DF_LOCAL) return cc_fail(cc, "I refuse too many parameters");
                snprintf(n->params[n->nparam], DF_NAME, "%s", cc->tok.name);
                n->nparam++;
                lex(cc);
            }
            if (eat(cc, TK_RETRY)) {
                if (!have(cc, TK_INT) || cc->tok.i < 0)
                    return cc_fail(cc, "I expected a retry count");
                n->retry = (int)cc->tok.i;
                lex(cc);
            }
            if (eat(cc, TK_EFFECT)) {
                if (parse_effect_name(cc, n->effect, sizeof n->effect) < 0) return -1;
                if (strcmp(n->effect, "IO") == 0) cc->has_io = 1;
            }
            if (!eat(cc, TK_EQ)) return cc_fail(cc, "I expected '='");
            n->body = parse_expr(cc);
            if (!n->body) return -1;
            continue;
        }
        if (eat(cc, TK_BUFFER)) {
            if (!have(cc, TK_INT) || cc->tok.i < 1 || cc->tok.i > DF_BUF)
                return cc_fail(cc, "I expected a buffer size from 1 to %d", DF_BUF);
            cc->bufcap = (int)cc->tok.i;
            lex(cc);
            continue;
        }
        if (eat(cc, TK_GRAPH)) {
            if (!eat(cc, TK_LBRACE)) return cc_fail(cc, "I expected '{'");
            while (!have(cc, TK_RBRACE) && !have(cc, TK_EOF)) {
                if (eat(cc, TK_OUT)) {
                    int s;
                    if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a stream");
                    s = add_stream(cc, cc->tok.name);
                    if (s < 0) return -1;
                    cc->streams[s].is_sink = 1;
                    cc->sink = s;
                    lex(cc);
                    continue;
                }
                if (eat(cc, TK_PLACE)) {
                    if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a stream");
                    lex(cc);
                    if (eat(cc, TK_REMOTE))
                        return cc_fail(cc, "I refuse remote placement; Phase 18 process boundaries are not wired");
                    if (!eat(cc, TK_LOCAL))
                        return cc_fail(cc, "I expected local or remote");
                    continue;
                }
                if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a stream binding");
                {
                    char nm[DF_NAME];
                    int s;
                    snprintf(nm, sizeof nm, "%s", cc->tok.name);
                    lex(cc);
                    if (!eat(cc, TK_EQ)) return cc_fail(cc, "I expected '='");
                    s = add_stream(cc, nm);
                    if (s < 0) return -1;
                    if (eat(cc, TK_IN)) {
                        cc->streams[s].is_in = 1;
                        continue;
                    }
                    if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a node");
                    {
                        int d = node_lookup(cc, cc->tok.name);
                        InstDef *it;
                        int i;
                        if (d < 0) return cc_fail(cc, "I do not know node %s", cc->tok.name);
                        lex(cc);
                        if (cc->ninst >= DF_MAX) return cc_fail(cc, "I refuse too many instances");
                        it = &cc->insts[cc->ninst];
                        memset(it, 0, sizeof *it);
                        it->def = d;
                        it->out_st = s;
                        for (i = 0; i < cc->nodes[d].nparam; i++) {
                            int in;
                            if (!have(cc, TK_ID))
                                return cc_fail(cc, "I expected input %s", cc->nodes[d].params[i]);
                            in = stream_lookup(cc, cc->tok.name);
                            if (in < 0) return cc_fail(cc, "I do not know stream %s", cc->tok.name);
                            it->in_st[i] = in;
                            it->nin++;
                            lex(cc);
                        }
                        cc->streams[s].inst = cc->ninst;
                        cc->ninst++;
                    }
                }
            }
            if (!eat(cc, TK_RBRACE)) return cc_fail(cc, "I expected '}'");
            continue;
        }
        if (eat(cc, TK_MAIN)) {
            if (!eat(cc, TK_LBRACE)) return cc_fail(cc, "I expected '{'");
            while (!have(cc, TK_RBRACE) && !have(cc, TK_EOF)) {
                Stmt *s;
                if (cc->nstmt >= DF_FEED) return cc_fail(cc, "I refuse too many statements");
                s = &cc->stmts[cc->nstmt];
                memset(s, 0, sizeof *s);
                if (eat(cc, TK_FEED)) {
                    s->kind = ST_FEED;
                    if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a stream");
                    snprintf(s->name, sizeof s->name, "%s", cc->tok.name);
                    lex(cc);
                    if (!have(cc, TK_INT)) return cc_fail(cc, "I expected an integer");
                    s->val = cc->tok.i;
                    lex(cc);
                    cc->nstmt++;
                    continue;
                }
                if (eat(cc, TK_DRAIN)) {
                    s->kind = ST_DRAIN;
                    cc->nstmt++;
                    continue;
                }
                if (eat(cc, TK_CANCEL)) {
                    s->kind = ST_CANCEL;
                    if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a stream");
                    snprintf(s->name, sizeof s->name, "%s", cc->tok.name);
                    lex(cc);
                    cc->nstmt++;
                    continue;
                }
                return cc_fail(cc, "I expected feed, drain, or cancel");
            }
            if (!eat(cc, TK_RBRACE)) return cc_fail(cc, "I expected '}'");
            continue;
        }
        return cc_fail(cc, "I expected node, buffer, graph, or main");
    }
    if (cc->sink < 0) return cc_fail(cc, "I expected an out stream");
    if (cc->nnode < 1) return cc_fail(cc, "I expected a node");
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
    if (buf_printf(b, "\n") < 0) return -1;
    return 0;
}

static int local_of(NodeDef *n, const char *name) {
    int i;
    if (strcmp(name, "attempt") == 0) return n->nparam;
    for (i = 0; i < n->nparam; i++) {
        if (strcmp(n->params[i], name) == 0) return i;
    }
    return -1;
}

static int compile_value(Cc *cc, Buf *b, NodeDef *n, Expr *e);

static int compile_result(Cc *cc, Buf *b, NodeDef *n, Expr *e) {
    int el, join;
    if (!e) return cc_fail(cc, "I expected an expression");
    if (e->kind == E_FAIL) {
        if (emit_line(b, 1, "PUSH_I64 0") < 0) return -1;
        if (emit_line(b, 1, "PUSH_I64 0") < 0) return -1;
        return emit_line(b, 1, "TUPLE_NEW 2");
    }
    if (e->kind == E_IF) {
        el = cc->label++;
        join = cc->label++;
        if (compile_value(cc, b, n, e->x) < 0) return -1;
        if (emit_line(b, 1, "JMP_FALSE lf%u", (unsigned)el) < 0) return -1;
        if (compile_result(cc, b, n, e->y) < 0) return -1;
        if (emit_line(b, 1, "JMP lf%u", (unsigned)join) < 0) return -1;
        if (emit_line(b, 0, "lf%u:", (unsigned)el) < 0) return -1;
        if (compile_result(cc, b, n, e->z) < 0) return -1;
        return emit_line(b, 0, "lf%u:", (unsigned)join);
    }
    if (emit_line(b, 1, "PUSH_I64 1") < 0) return -1;
    if (compile_value(cc, b, n, e) < 0) return -1;
    return emit_line(b, 1, "TUPLE_NEW 2");
}

static int compile_value(Cc *cc, Buf *b, NodeDef *n, Expr *e) {
    int i;
    const char *op;
    if (!e) return cc_fail(cc, "I expected an expression");
    switch (e->kind) {
    case E_INT:
        return emit_line(b, 1, "PUSH_I64 %lld", (long long)e->i);
    case E_VAR:
        i = local_of(n, e->name);
        if (i < 0) return cc_fail(cc, "I do not know %s", e->name);
        return emit_line(b, 1, "LOAD_LOCAL %d", i);
    case E_BIN:
        if (compile_value(cc, b, n, e->x) < 0) return -1;
        if (compile_value(cc, b, n, e->y) < 0) return -1;
        if (e->bin == TK_PLUS) op = "I64_ADD";
        else if (e->bin == TK_MINUS) op = "I64_SUB";
        else if (e->bin == TK_STAR) op = "I64_MUL";
        else if (e->bin == TK_SLASH) op = "I64_DIV_S";
        else op = "I64_EQ";
        return emit_line(b, 1, "%s", op);
    case E_FAIL:
        return cc_fail(cc, "fail is not a value");
    case E_IF:
        return cc_fail(cc, "if is not a value");
    }
    return cc_fail(cc, "I cannot compile that");
}

static int compile_all(Cc *cc) {
    int i;
    for (i = 0; i < cc->nnode; i++) {
        NodeDef *n = &cc->nodes[i];
        if (cc->nfn >= DF_MAX) return cc_fail(cc, "I refuse too many functions");
        if (compile_result(cc, &cc->code[cc->nfn], n, n->body) < 0) return -1;
        if (emit_line(&cc->code[cc->nfn], 1, "RET") < 0) return -1;
        n->fn_idx = cc->nfn;
        cc->nfn++;
    }
    return 0;
}

static int build_asm(Cc *cc, Buf *out) {
    int i;
    if (buf_printf(out, ".flag has_main\n.entry _df_main\n") < 0) return -1;
    for (i = 0; i < cc->nnode; i++) {
        int arity = cc->nodes[i].nparam + 1;
        int locals = arity + 4;
        if (buf_printf(out, ".function %s %d %d 0 tuple 1\n",
                       cc->nodes[i].asm_name, arity, locals) < 0)
            return -1;
        if (cc->code[i].p && buf_printf(out, "%s", cc->code[i].p) < 0) return -1;
        if (buf_printf(out, ".end\n") < 0) return -1;
    }
    for (i = 0; i < cc->nstmt; i++) {
        if (cc->stmts[i].kind == ST_FEED) {
            if (buf_printf(out, ".string \"feed %s %lld\"\n",
                           cc->stmts[i].name, (long long)cc->stmts[i].val) < 0)
                return -1;
        }
    }
    if (buf_printf(out, ".function _df_main 0 2 0 int 1\n  PUSH_I64 0\n  RET\n.end\n") < 0)
        return -1;
    return 0;
}

typedef struct {
    int64_t v[DF_BUF];
    int n;
} Edge;

typedef struct {
    Edge edges[DF_MAX];
    int attempt[DF_MAX];
    int reverse;
    Cc *cc;
    NvmModule *mod;
    VmState vm;
    int vm_on;
} Rt;

static int sys_fail(Cc *cc, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(cc->err, sizeof cc->err, fmt, ap);
    va_end(ap);
    return -1;
}

static int edge_push(Cc *cc, Edge *e, int cap, int64_t v) {
    if (e->n >= cap) return sys_fail(cc, "I refuse a full buffer");
    e->v[e->n++] = v;
    return 0;
}

static int edge_pop(Edge *e, int64_t *out) {
    int i;
    if (e->n <= 0) return -1;
    *out = e->v[0];
    for (i = 1; i < e->n; i++) e->v[i - 1] = e->v[i];
    e->n--;
    return 0;
}

static int inst_ready(Rt *rt, int i) {
    InstDef *it = &rt->cc->insts[i];
    int j;
    if (it->nin <= 0) return 0;
    for (j = 0; j < it->nin; j++) {
        if (rt->edges[it->in_st[j]].n <= 0) return 0;
    }
    return 1;
}

static int pick_ready(Rt *rt) {
    int i, best = -1;
    for (i = 0; i < rt->cc->ninst; i++) {
        if (!inst_ready(rt, i)) continue;
        if (best < 0) best = i;
        else if (rt->reverse && i > best) best = i;
        else if (!rt->reverse && i < best) best = i;
    }
    return best;
}

static int fire(Rt *rt, int i) {
    InstDef *it = &rt->cc->insts[i];
    NodeDef *d = &rt->cc->nodes[it->def];
    NanoValue args[DF_LOCAL];
    NanoValue out;
    VmResult r;
    int64_t ins[DF_LOCAL];
    int64_t ok, val;
    int j, arity = d->nparam + 1;
    if (it->cancelled)
        return sys_fail(rt->cc, "I refuse a cancelled node");
    for (j = 0; j < d->nparam; j++) {
        if (edge_pop(&rt->edges[it->in_st[j]], &ins[j]) < 0)
            return sys_fail(rt->cc, "I expected an input token");
        args[j] = val_int(ins[j]);
    }
    args[d->nparam] = val_int(rt->attempt[i]);
    memset(&out, 0, sizeof out);
    r = vm_invoke(&rt->vm, (uint32_t)d->fn_idx, args, (uint16_t)arity, &out);
    if (r != VM_OK) {
        return sys_fail(rt->cc, "node %s trapped", d->name);
    }
    if (out.tag != TAG_TUPLE || !out.as.tuple || out.as.tuple->count != 2) {
        vm_release(&rt->vm.heap, out);
        return sys_fail(rt->cc, "node did not return a 2-tuple");
    }
    ok = out.as.tuple->elements[0].as.i64;
    val = out.as.tuple->elements[1].as.i64;
    vm_release(&rt->vm.heap, out);
    if (ok == 0) {
        if (rt->attempt[i] < d->retry) {
            for (j = 0; j < d->nparam; j++) {
                if (edge_push(rt->cc, &rt->edges[it->in_st[j]], rt->cc->bufcap, ins[j]) < 0)
                    return -1;
            }
            rt->attempt[i]++;
            return 0;
        }
        return sys_fail(rt->cc, "node %s failed", d->name);
    }
    rt->attempt[i] = 0;
    return edge_push(rt->cc, &rt->edges[it->out_st], rt->cc->bufcap, val);
}

static int drain_rt(Rt *rt) {
    int guard = 0;
    while (rt->edges[rt->cc->sink].n == 0 && guard++ < 10000) {
        int i = pick_ready(rt);
        if (i < 0) return sys_fail(rt->cc, "I am waiting with no ready node");
        if (fire(rt, i) < 0) return -1;
    }
    if (rt->edges[rt->cc->sink].n == 0)
        return sys_fail(rt->cc, "I am waiting with no ready node");
    return 0;
}

static int run_rt(Cc *cc, NvmModule *mod, int reverse, int64_t *out) {
    Rt rt;
    int i;
    int64_t last = 0;
    int have_last = 0;
    memset(&rt, 0, sizeof rt);
    rt.cc = cc;
    rt.mod = mod;
    rt.reverse = reverse;
    for (i = 0; i < cc->nnode; i++) {
        uint32_t f;
        cc->nodes[i].fn_idx = -1;
        for (f = 0; f < mod->function_count; f++) {
            const char *nm = nvm_get_string(mod, mod->functions[f].name_idx);
            if (nm && strcmp(nm, cc->nodes[i].asm_name) == 0) {
                cc->nodes[i].fn_idx = (int)f;
                break;
            }
        }
        if (cc->nodes[i].fn_idx < 0)
            return sys_fail(cc, "I lost function %s", cc->nodes[i].asm_name);
    }
    vm_init(&rt.vm, mod);
    rt.vm_on = 1;
    for (i = 0; i < cc->nstmt; i++) {
        Stmt *s = &cc->stmts[i];
        if (s->kind == ST_FEED) {
            int st = stream_lookup(cc, s->name);
            if (st < 0 || !cc->streams[st].is_in)
                return sys_fail(cc, "I do not know input %s", s->name);
            if (edge_push(cc, &rt.edges[st], cc->bufcap, s->val) < 0) return -1;
            continue;
        }
        if (s->kind == ST_CANCEL) {
            int st = stream_lookup(cc, s->name);
            int inst;
            if (st < 0) return sys_fail(cc, "I do not know stream %s", s->name);
            inst = cc->streams[st].inst;
            if (inst < 0) return sys_fail(cc, "I cannot cancel an input");
            cc->insts[inst].cancelled = 1;
            continue;
        }
        if (s->kind == ST_DRAIN) {
            if (drain_rt(&rt) < 0) {
                if (rt.vm_on) vm_destroy(&rt.vm);
                return -1;
            }
            last = rt.edges[cc->sink].v[0];
            have_last = 1;
        }
    }
    if (!have_last) {
        if (rt.vm_on) vm_destroy(&rt.vm);
        return sys_fail(cc, "I expected drain");
    }
    *out = last;
    if (rt.vm_on) vm_destroy(&rt.vm);
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
    if (getenv("NL_DF_TRACE")) fprintf(stderr, "%s", asmbuf.p ? asmbuf.p : "");
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

NvmModule *nl_dataflow_compile(const char *src, const char *path,
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

NlFrontendResult nl_dataflow_accept(const NvmModule *mod, const char *path) {
    NlFrontendFacts f;
    memset(&f, 0, sizeof f);
    f.language = NL_FE_DATAFLOW;
    f.source_path = (path && path[0]) ? path : "<dataflow>";
    f.purity = 1;
    f.exhaustiveness = 0;
    f.affine_use = -1;
    f.diagnostics_shared = 1;
    return nl_frontend_accept(mod, &f);
}

static int eval_src(const char *src, int reverse, int64_t *out, char *err, size_t errlen) {
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
    acc = nl_dataflow_accept(mod, "<eval>");
    if (!acc.ok) {
        if (err && errlen) snprintf(err, errlen, "%s", acc.error);
        nvm_module_free(mod);
        cc_free(&cc);
        return 0;
    }
    if (run_rt(&cc, mod, reverse, &v) < 0) {
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

int nl_dataflow_eval_i64(const char *src, int64_t *out, char *err, size_t errlen) {
    return eval_src(src, 0, out, err, errlen);
}

int nl_dataflow_eval_i64_sched(const char *src, int reverse, int64_t *out,
                               char *err, size_t errlen) {
    return eval_src(src, reverse, out, err, errlen);
}
