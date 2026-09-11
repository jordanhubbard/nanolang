/*
 * Nano Object — bounded laboratory frontend. I emit verified NanoISA
 * methods and dispatch in the host. See docs/OBJECT.md.
 */

#include "object.h"

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

#define OB_NAME 64
#define OB_MAX 24
#define OB_ERR NL_OBJ_ERR_SIZE
#define OB_SLOT 8
#define OB_METH 12
#define OB_STMT 32

typedef enum {
    TK_EOF, TK_INT, TK_ID,
    TK_LPAREN, TK_RPAREN, TK_LBRACE, TK_RBRACE, TK_EQ,
    TK_PLUS, TK_MINUS,
    TK_CLASS, TK_METHOD, TK_MAIN, TK_NEW, TK_SEND, TK_HANDLE,
    TK_SENDVIA, TK_REPLACE, TK_EXTEND, TK_CLASSOF, TK_SLOTS
} TkKind;

typedef struct {
    TkKind kind;
    int64_t i;
    char name[OB_NAME];
} Tok;

typedef enum { E_INT, E_VAR, E_BIN, E_CLASSOF, E_SLOTS } EKind;
typedef enum {
    ST_NEW, ST_SEND, ST_HANDLE, ST_SENDVIA, ST_REPLACE, ST_EXTEND
} SKind;

typedef struct Expr Expr;
struct Expr {
    EKind kind;
    int64_t i;
    char name[OB_NAME];
    int bin;
    Expr *x, *y;
};

typedef struct {
    char slot[OB_NAME];
    Expr *e;
} Assign;

typedef struct {
    char sel[OB_NAME];
    char params[OB_SLOT][OB_NAME];
    int nparam;
    Assign as[OB_METH];
    int na;
    Expr *result;
    char asm_name[OB_NAME];
    int fn_idx;
} Meth;

typedef struct {
    char name[OB_NAME];
    char slots[OB_SLOT][OB_NAME];
    int nslot;
    Meth meths[OB_METH];
    int nmeth;
} ClassDef;

typedef struct {
    char cls[OB_NAME];
    char sel[OB_NAME];
    Assign as[OB_METH];
    int na;
    Expr *result;
    char asm_name[OB_NAME];
    int fn_idx;
    int nparam;
} Repl;

typedef struct {
    SKind kind;
    char name[OB_NAME];
    char cls[OB_NAME];
    char sel[OB_NAME];
    Expr *recv;
    Expr *args[OB_SLOT];
    int nargs;
    int repl;
    int ic;
} Stmt;

typedef struct {
    char *p;
    size_t n, cap;
} Buf;

typedef struct {
    const char *src;
    const char *lx;
    Tok tok;
    char err[OB_ERR];
    void **heap;
    uint32_t nheap, capheap;
    ClassDef classes[OB_MAX];
    int nclass;
    Repl repls[OB_MAX];
    int nrepl;
    Stmt stmts[OB_STMT];
    int nstmt;
    Expr *result;
    Buf code[OB_MAX];
    int nfn;
} Cc;

static int g_ic_hits;
static int g_ic_misses;
static char g_image[NL_OBJ_IMAGE_SIZE];

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

static int set_symbol(char *dst, size_t cap, const char *first,
                      const char *second, const char *third) {
    char symbol[OB_NAME];
    size_t first_len = strlen(first);
    size_t second_len = strlen(second);
    size_t third_len = strlen(third);
    size_t total = first_len + second_len + third_len;
    if (cap > sizeof symbol || total >= cap) return -1;
    memcpy(symbol, first, first_len);
    memcpy(symbol + first_len, second, second_len);
    memcpy(symbol + first_len + second_len, third, third_len + 1);
    memmove(dst, symbol, total + 1);
    return 0;
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
        while ((isalnum((unsigned char)*p) || *p == '_') && n < OB_NAME - 1)
            cc->tok.name[n++] = *p++;
        cc->lx = p;
        if (kw_eq(cc->tok.name, "class")) cc->tok.kind = TK_CLASS;
        else if (kw_eq(cc->tok.name, "method")) cc->tok.kind = TK_METHOD;
        else if (kw_eq(cc->tok.name, "main")) cc->tok.kind = TK_MAIN;
        else if (kw_eq(cc->tok.name, "new")) cc->tok.kind = TK_NEW;
        else if (kw_eq(cc->tok.name, "send")) cc->tok.kind = TK_SEND;
        else if (kw_eq(cc->tok.name, "handle")) cc->tok.kind = TK_HANDLE;
        else if (kw_eq(cc->tok.name, "sendvia")) cc->tok.kind = TK_SENDVIA;
        else if (kw_eq(cc->tok.name, "replace")) cc->tok.kind = TK_REPLACE;
        else if (kw_eq(cc->tok.name, "extend")) cc->tok.kind = TK_EXTEND;
        else if (kw_eq(cc->tok.name, "classof")) cc->tok.kind = TK_CLASSOF;
        else if (kw_eq(cc->tok.name, "slots")) cc->tok.kind = TK_SLOTS;
        else cc->tok.kind = TK_ID;
        return;
    }
    cc->lx = p + 1;
    switch (*p) {
    case '(': cc->tok.kind = TK_LPAREN; break;
    case ')': cc->tok.kind = TK_RPAREN; break;
    case '{': cc->tok.kind = TK_LBRACE; break;
    case '}': cc->tok.kind = TK_RBRACE; break;
    case '=': cc->tok.kind = TK_EQ; break;
    case '+': cc->tok.kind = TK_PLUS; break;
    case '-': cc->tok.kind = TK_MINUS; break;
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

static int class_lookup(Cc *cc, const char *n) {
    int i;
    for (i = 0; i < cc->nclass; i++) {
        if (strcmp(cc->classes[i].name, n) == 0) return i;
    }
    return -1;
}

static int meth_lookup(ClassDef *c, const char *s) {
    int i;
    for (i = 0; i < c->nmeth; i++) {
        if (strcmp(c->meths[i].sel, s) == 0) return i;
    }
    return -1;
}

static int slot_lookup(ClassDef *c, const char *s) {
    int i;
    for (i = 0; i < c->nslot; i++) {
        if (strcmp(c->slots[i], s) == 0) return i;
    }
    return -1;
}

static Expr *parse_expr(Cc *cc);

static Expr *parse_atom(Cc *cc) {
    Expr *e;
    if (have(cc, TK_INT)) {
        e = ex_new(cc, E_INT);
        if (!e) return NULL;
        e->i = cc->tok.i;
        lex(cc);
        return e;
    }
    if (eat(cc, TK_CLASSOF)) {
        e = ex_new(cc, E_CLASSOF);
        if (!e) return NULL;
        if (!have(cc, TK_ID)) { cc_fail(cc, "I expected an object"); return NULL; }
        snprintf(e->name, sizeof e->name, "%s", cc->tok.name);
        lex(cc);
        return e;
    }
    if (eat(cc, TK_SLOTS)) {
        e = ex_new(cc, E_SLOTS);
        if (!e) return NULL;
        if (!have(cc, TK_ID)) { cc_fail(cc, "I expected an object"); return NULL; }
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
        if (!eat(cc, TK_RPAREN)) { cc_fail(cc, "I expected ')'"); return NULL; }
        return e;
    }
    cc_fail(cc, "I expected an expression");
    return NULL;
}

static Expr *parse_expr(Cc *cc) {
    Expr *left = parse_atom(cc);
    if (!left) return NULL;
    while (have(cc, TK_PLUS) || have(cc, TK_MINUS)) {
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

static int parse_body(Cc *cc, Assign *as, int *na, Expr **result) {
    *na = 0;
    *result = NULL;
    while (!have(cc, TK_RBRACE) && !have(cc, TK_EOF)) {
        if (have(cc, TK_ID)) {
            char nm[OB_NAME];
            snprintf(nm, sizeof nm, "%s", cc->tok.name);
            lex(cc);
            if (eat(cc, TK_EQ)) {
                if (*na >= OB_METH) return cc_fail(cc, "I refuse too many assignments");
                snprintf(as[*na].slot, OB_NAME, "%s", nm);
                as[*na].e = parse_expr(cc);
                if (!as[*na].e) return -1;
                (*na)++;
                continue;
            }
            {
                Expr *e = ex_new(cc, E_VAR);
                if (!e) return -1;
                snprintf(e->name, sizeof e->name, "%s", nm);
                while (have(cc, TK_PLUS) || have(cc, TK_MINUS)) {
                    Expr *n = ex_new(cc, E_BIN);
                    Expr *right;
                    if (!n) return -1;
                    n->bin = (int)cc->tok.kind;
                    lex(cc);
                    right = parse_atom(cc);
                    if (!right) return -1;
                    n->x = e;
                    n->y = right;
                    e = n;
                }
                *result = e;
                break;
            }
        }
        *result = parse_expr(cc);
        if (!*result) return -1;
        break;
    }
    if (!*result) {
        if (*na <= 0) return cc_fail(cc, "I expected a method body");
        *result = ex_new(cc, E_VAR);
        if (!*result) return -1;
        snprintf((*result)->name, OB_NAME, "%s", as[*na - 1].slot);
    }
    return 0;
}

static int parse_program(Cc *cc) {
    cc->lx = cc->src;
    lex(cc);
    while (!have(cc, TK_EOF)) {
        if (eat(cc, TK_CLASS)) {
            ClassDef *c;
            if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a class name");
            if (cc->nclass >= OB_MAX) return cc_fail(cc, "I refuse too many classes");
            c = &cc->classes[cc->nclass++];
            memset(c, 0, sizeof *c);
            snprintf(c->name, sizeof c->name, "%s", cc->tok.name);
            lex(cc);
            if (!eat(cc, TK_LBRACE)) return cc_fail(cc, "I expected '{'");
            while (have(cc, TK_ID)) {
                if (c->nslot >= OB_SLOT) return cc_fail(cc, "I refuse more than eight slots");
                snprintf(c->slots[c->nslot], OB_NAME, "%s", cc->tok.name);
                c->nslot++;
                lex(cc);
            }
            while (eat(cc, TK_METHOD)) {
                Meth *m;
                if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a selector");
                if (c->nmeth >= OB_METH) return cc_fail(cc, "I refuse too many methods");
                m = &c->meths[c->nmeth++];
                memset(m, 0, sizeof *m);
                snprintf(m->sel, sizeof m->sel, "%s", cc->tok.name);
                if (set_symbol(m->asm_name, sizeof m->asm_name, "obj_", c->name, "_") < 0
                    || set_symbol(m->asm_name, sizeof m->asm_name, m->asm_name, m->sel, "") < 0)
                    return cc_fail(cc, "I refuse class and selector names that exceed my symbol boundary");
                lex(cc);
                while (have(cc, TK_ID)) {
                    if (m->nparam >= OB_SLOT) return cc_fail(cc, "I refuse too many parameters");
                    snprintf(m->params[m->nparam], OB_NAME, "%s", cc->tok.name);
                    m->nparam++;
                    lex(cc);
                }
                if (!eat(cc, TK_LBRACE)) return cc_fail(cc, "I expected '{'");
                if (parse_body(cc, m->as, &m->na, &m->result) < 0) return -1;
                if (!eat(cc, TK_RBRACE)) return cc_fail(cc, "I expected '}'");
            }
            if (!eat(cc, TK_RBRACE)) return cc_fail(cc, "I expected '}'");
            continue;
        }
        if (eat(cc, TK_MAIN)) {
            if (!eat(cc, TK_LBRACE)) return cc_fail(cc, "I expected '{'");
            while (!have(cc, TK_RBRACE) && !have(cc, TK_EOF)) {
                Stmt *s;
                if (have(cc, TK_ID)) {
                    char nm[OB_NAME];
                    snprintf(nm, sizeof nm, "%s", cc->tok.name);
                    lex(cc);
                    if (eat(cc, TK_EQ)) {
                        if (cc->nstmt >= OB_STMT) return cc_fail(cc, "I refuse too many statements");
                        s = &cc->stmts[cc->nstmt++];
                        memset(s, 0, sizeof *s);
                        snprintf(s->name, sizeof s->name, "%s", nm);
                        if (eat(cc, TK_NEW)) {
                            s->kind = ST_NEW;
                            if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a class");
                            snprintf(s->cls, sizeof s->cls, "%s", cc->tok.name);
                            lex(cc);
                            continue;
                        }
                        if (eat(cc, TK_HANDLE)) {
                            s->kind = ST_HANDLE;
                            if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a class");
                            snprintf(s->cls, sizeof s->cls, "%s", cc->tok.name);
                            lex(cc);
                            if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a selector");
                            snprintf(s->sel, sizeof s->sel, "%s", cc->tok.name);
                            lex(cc);
                            continue;
                        }
                        if (eat(cc, TK_SEND)) {
                            s->kind = ST_SEND;
                            s->ic = cc->nstmt - 1;
                            s->recv = parse_atom(cc);
                            if (!s->recv) return -1;
                            if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a selector");
                            snprintf(s->sel, sizeof s->sel, "%s", cc->tok.name);
                            lex(cc);
                            while (have(cc, TK_INT) || have(cc, TK_ID) || have(cc, TK_LPAREN)) {
                                if (s->nargs >= OB_SLOT) return cc_fail(cc, "too many args");
                                s->args[s->nargs] = parse_atom(cc);
                                if (!s->args[s->nargs]) return -1;
                                s->nargs++;
                            }
                            continue;
                        }
                        return cc_fail(cc, "I expected new, handle, or send");
                    }
                    {
                        Expr *e = ex_new(cc, E_VAR);
                        if (!e) return -1;
                        snprintf(e->name, sizeof e->name, "%s", nm);
                        cc->result = e;
                        break;
                    }
                }
                if (eat(cc, TK_SEND)) {
                    if (cc->nstmt >= OB_STMT) return cc_fail(cc, "I refuse too many statements");
                    s = &cc->stmts[cc->nstmt++];
                    memset(s, 0, sizeof *s);
                    s->kind = ST_SEND;
                    s->ic = cc->nstmt - 1;
                    s->recv = parse_atom(cc);
                    if (!s->recv) return -1;
                    if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a selector");
                    snprintf(s->sel, sizeof s->sel, "%s", cc->tok.name);
                    lex(cc);
                    while (have(cc, TK_INT) || have(cc, TK_ID) || have(cc, TK_LPAREN)) {
                        if (s->nargs >= OB_SLOT) return cc_fail(cc, "too many args");
                        s->args[s->nargs] = parse_atom(cc);
                        if (!s->args[s->nargs]) return -1;
                        s->nargs++;
                    }
                    continue;
                }
                if (eat(cc, TK_SENDVIA)) {
                    if (cc->nstmt >= OB_STMT) return cc_fail(cc, "I refuse too many statements");
                    s = &cc->stmts[cc->nstmt++];
                    memset(s, 0, sizeof *s);
                    s->kind = ST_SENDVIA;
                    s->recv = parse_atom(cc);
                    if (!s->recv) return -1;
                    s->args[0] = parse_atom(cc);
                    if (!s->args[0]) return -1;
                    s->nargs = 1;
                    continue;
                }
                if (eat(cc, TK_REPLACE)) {
                    Repl *r;
                    ClassDef *c;
                    Meth *om;
                    int mi;
                    if (cc->nrepl >= OB_MAX) return cc_fail(cc, "I refuse too many replacements");
                    r = &cc->repls[cc->nrepl];
                    memset(r, 0, sizeof *r);
                    if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a class");
                    snprintf(r->cls, sizeof r->cls, "%s", cc->tok.name);
                    lex(cc);
                    if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a selector");
                    snprintf(r->sel, sizeof r->sel, "%s", cc->tok.name);
                    lex(cc);
                    c = class_lookup(cc, r->cls) >= 0 ? &cc->classes[class_lookup(cc, r->cls)] : NULL;
                    if (!c) return cc_fail(cc, "I do not know class %s", r->cls);
                    mi = meth_lookup(c, r->sel);
                    if (mi < 0) return cc_fail(cc, "I do not know method %s", r->sel);
                    om = &c->meths[mi];
                    r->nparam = om->nparam;
                    {
                        char suffix[16];
                        snprintf(suffix, sizeof suffix, "_r%d", cc->nrepl);
                        if (set_symbol(r->asm_name, sizeof r->asm_name, om->asm_name, suffix, "") < 0)
                            return cc_fail(cc, "I refuse a replacement name that exceeds my symbol boundary");
                    }
                    if (!eat(cc, TK_LBRACE)) return cc_fail(cc, "I expected '{'");
                    if (parse_body(cc, r->as, &r->na, &r->result) < 0) return -1;
                    if (!eat(cc, TK_RBRACE)) return cc_fail(cc, "I expected '}'");
                    if (cc->nstmt >= OB_STMT) return cc_fail(cc, "I refuse too many statements");
                    s = &cc->stmts[cc->nstmt++];
                    memset(s, 0, sizeof *s);
                    s->kind = ST_REPLACE;
                    s->repl = cc->nrepl;
                    memmove(s->cls, r->cls, sizeof s->cls);
                    memmove(s->sel, r->sel, sizeof s->sel);
                    cc->nrepl++;
                    continue;
                }
                if (eat(cc, TK_EXTEND)) {
                    if (cc->nstmt >= OB_STMT) return cc_fail(cc, "I refuse too many statements");
                    s = &cc->stmts[cc->nstmt++];
                    memset(s, 0, sizeof *s);
                    s->kind = ST_EXTEND;
                    if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a class");
                    snprintf(s->cls, sizeof s->cls, "%s", cc->tok.name);
                    lex(cc);
                    if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a slot");
                    snprintf(s->sel, sizeof s->sel, "%s", cc->tok.name);
                    lex(cc);
                    continue;
                }
                if (have(cc, TK_CLASSOF) || have(cc, TK_SLOTS) || have(cc, TK_INT)) {
                    cc->result = parse_expr(cc);
                    if (!cc->result) return -1;
                    break;
                }
                return cc_fail(cc, "I expected a statement");
            }
            if (!eat(cc, TK_RBRACE)) return cc_fail(cc, "I expected '}'");
            continue;
        }
        return cc_fail(cc, "I expected class or main");
    }
    if (cc->nclass < 1) return cc_fail(cc, "I expected a class");
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

static int local_of(ClassDef *c, Meth *m, const char *n) {
    int i;
    i = slot_lookup(c, n);
    if (i >= 0) return i;
    for (i = 0; i < m->nparam; i++) {
        if (strcmp(m->params[i], n) == 0) return OB_SLOT + i;
    }
    return -1;
}

static int compile_value(Cc *cc, Buf *b, ClassDef *c, Meth *m, Expr *e) {
    int i;
    if (!e) return cc_fail(cc, "I expected an expression");
    switch (e->kind) {
    case E_INT:
        return emit_line(b, 1, "PUSH_I64 %lld", (long long)e->i);
    case E_VAR:
        i = local_of(c, m, e->name);
        if (i < 0) return cc_fail(cc, "I do not know %s", e->name);
        return emit_line(b, 1, "LOAD_LOCAL %d", i);
    case E_BIN:
        if (compile_value(cc, b, c, m, e->x) < 0) return -1;
        if (compile_value(cc, b, c, m, e->y) < 0) return -1;
        return emit_line(b, 1, "%s", e->bin == TK_MINUS ? "I64_SUB" : "I64_ADD");
    case E_CLASSOF:
    case E_SLOTS:
        return cc_fail(cc, "classof and slots are host expressions");
    }
    return cc_fail(cc, "I cannot compile that");
}

static int compile_meth_body(Cc *cc, Buf *b, ClassDef *c, Meth *m,
                             Assign *as, int na, Expr *result) {
    int i;
    Meth tmp = {0};
    Meth *use = m ? m : &tmp;
    if (m) use = m;
    for (i = 0; i < na; i++) {
        int sl = slot_lookup(c, as[i].slot);
        if (sl < 0) return cc_fail(cc, "I do not know slot %s", as[i].slot);
        if (compile_value(cc, b, c, use, as[i].e) < 0) return -1;
        if (emit_line(b, 1, "STORE_LOCAL %d", sl) < 0) return -1;
    }
    if (compile_value(cc, b, c, use, result) < 0) return -1;
    for (i = 0; i < OB_SLOT; i++) {
        if (emit_line(b, 1, "LOAD_LOCAL %d", i) < 0) return -1;
    }
    if (emit_line(b, 1, "TUPLE_NEW 9") < 0) return -1;
    return emit_line(b, 1, "RET");
}

static int compile_all(Cc *cc) {
    int i, j;
    for (i = 0; i < cc->nclass; i++) {
        ClassDef *c = &cc->classes[i];
        for (j = 0; j < c->nmeth; j++) {
            Meth *m = &c->meths[j];
            if (cc->nfn >= OB_MAX) return cc_fail(cc, "I refuse too many functions");
            if (compile_meth_body(cc, &cc->code[cc->nfn], c, m, m->as, m->na, m->result) < 0)
                return -1;
            m->fn_idx = cc->nfn;
            cc->nfn++;
        }
    }
    for (i = 0; i < cc->nrepl; i++) {
        Repl *r = &cc->repls[i];
        ClassDef *c = &cc->classes[class_lookup(cc, r->cls)];
        Meth *om = &c->meths[meth_lookup(c, r->sel)];
        if (cc->nfn >= OB_MAX) return cc_fail(cc, "I refuse too many functions");
        if (compile_meth_body(cc, &cc->code[cc->nfn], c, om, r->as, r->na, r->result) < 0)
            return -1;
        r->fn_idx = cc->nfn;
        cc->nfn++;
    }
    return 0;
}

static int build_asm(Cc *cc, Buf *out) {
    int i, j, k = 0;
    if (buf_printf(out, ".flag has_main\n.entry _obj_main\n") < 0) return -1;
    for (i = 0; i < cc->nclass; i++) {
        if (buf_printf(out, ".string \"class %s\"\n", cc->classes[i].name) < 0)
            return -1;
        for (j = 0; j < cc->classes[i].nmeth; j++, k++) {
            int arity = OB_SLOT + cc->classes[i].meths[j].nparam;
            if (buf_printf(out, ".function %s %d %d 0 tuple 1\n",
                           cc->classes[i].meths[j].asm_name, arity, arity + 4) < 0)
                return -1;
            if (cc->code[k].p && buf_printf(out, "%s", cc->code[k].p) < 0) return -1;
            if (buf_printf(out, ".end\n") < 0) return -1;
        }
    }
    for (i = 0; i < cc->nrepl; i++, k++) {
        int arity = OB_SLOT + cc->repls[i].nparam;
        if (buf_printf(out, ".function %s %d %d 0 tuple 1\n",
                       cc->repls[i].asm_name, arity, arity + 4) < 0)
            return -1;
        if (cc->code[k].p && buf_printf(out, "%s", cc->code[k].p) < 0) return -1;
        if (buf_printf(out, ".end\n") < 0) return -1;
    }
    if (buf_printf(out, ".function _obj_main 0 2 0 int 1\n  PUSH_I64 0\n  RET\n.end\n") < 0)
        return -1;
    return 0;
}

typedef struct {
    int used, cls;
    int64_t slots[OB_SLOT];
} Obj;

typedef struct {
    int class_id, fn_idx, armed, hits, misses;
} IC;

typedef struct {
    Obj objs[OB_MAX];
    int nobj;
    int64_t locals[32];
    char lnames[32][OB_NAME];
    int nl;
    IC ics[OB_MAX][OB_METH];
    int meth_fn[OB_MAX][OB_METH];
    Cc *cc;
    NvmModule *mod;
    VmState vm;
} Rt;

static int sys_fail(Cc *cc, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(cc->err, sizeof cc->err, fmt, ap);
    va_end(ap);
    return -1;
}

static int bind_local(Rt *rt, const char *n, int64_t v) {
    if (rt->nl >= 32) return -1;
    snprintf(rt->lnames[rt->nl], OB_NAME, "%s", n);
    rt->locals[rt->nl] = v;
    rt->nl++;
    return 0;
}

static int lookup_local(Rt *rt, const char *n, int64_t *out) {
    int i;
    for (i = rt->nl - 1; i >= 0; i--) {
        if (strcmp(rt->lnames[i], n) == 0) { *out = rt->locals[i]; return 0; }
    }
    return -1;
}

static int eval_host_expr(Cc *cc, Rt *rt, Expr *e, int64_t *out) {
    int64_t a, b, oid;
    if (!e) return sys_fail(cc, "I expected an expression");
    switch (e->kind) {
    case E_INT: *out = e->i; return 0;
    case E_VAR:
        if (lookup_local(rt, e->name, out) < 0)
            return sys_fail(cc, "I do not know %s", e->name);
        return 0;
    case E_BIN:
        if (eval_host_expr(cc, rt, e->x, &a) < 0) return -1;
        if (eval_host_expr(cc, rt, e->y, &b) < 0) return -1;
        *out = e->bin == TK_MINUS ? a - b : a + b;
        return 0;
    case E_CLASSOF:
        if (lookup_local(rt, e->name, &oid) < 0)
            return sys_fail(cc, "I do not know %s", e->name);
        if (oid <= 0 || oid >= rt->nobj) return sys_fail(cc, "I do not know that object");
        *out = rt->objs[oid].cls;
        return 0;
    case E_SLOTS:
        if (lookup_local(rt, e->name, &oid) < 0)
            return sys_fail(cc, "I do not know %s", e->name);
        if (oid <= 0 || oid >= rt->nobj) return sys_fail(cc, "I do not know that object");
        *out = cc->classes[rt->objs[oid].cls].nslot;
        return 0;
    }
    return sys_fail(cc, "I cannot evaluate that");
}

static int invoke_meth(Rt *rt, int oid, int fn_idx, Expr **args, int nargs, int64_t *out) {
    Obj *o;
    ClassDef *c;
    NanoValue av[OB_SLOT + OB_SLOT];
    NanoValue ret;
    VmResult r;
    int i, arity;
    if (oid <= 0 || oid >= rt->nobj) return sys_fail(rt->cc, "I do not know that object");
    o = &rt->objs[oid];
    c = &rt->cc->classes[o->cls];
    arity = OB_SLOT + nargs;
    for (i = 0; i < OB_SLOT; i++) av[i] = val_int(o->slots[i]);
    for (i = 0; i < nargs; i++) {
        int64_t v = 0;
        if (eval_host_expr(rt->cc, rt, args[i], &v) < 0) return -1;
        av[OB_SLOT + i] = val_int(v);
    }
    memset(&ret, 0, sizeof ret);
    r = vm_invoke(&rt->vm, (uint32_t)fn_idx, av, (uint16_t)arity, &ret);
    if (r != VM_OK) return sys_fail(rt->cc, "method trapped");
    if (ret.tag != TAG_TUPLE || !ret.as.tuple || ret.as.tuple->count != 9) {
        vm_release(&rt->vm.heap, ret);
        return sys_fail(rt->cc, "method did not return a 9-tuple");
    }
    *out = ret.as.tuple->elements[0].as.i64;
    for (i = 0; i < c->nslot && i < OB_SLOT; i++)
        o->slots[i] = ret.as.tuple->elements[i + 1].as.i64;
    vm_release(&rt->vm.heap, ret);
    return 0;
}

static int lookup_fn(NvmModule *mod, const char *name) {
    uint32_t f;
    for (f = 0; f < mod->function_count; f++) {
        const char *nm = nvm_get_string(mod, mod->functions[f].name_idx);
        if (nm && strcmp(nm, name) == 0) return (int)f;
    }
    return -1;
}

static int send_sel(Rt *rt, int oid, const char *sel, Expr **args, int nargs,
                    int64_t *out) {
    Obj *o;
    ClassDef *c;
    int mi, fn;
    IC *ic;
    if (oid <= 0 || oid >= rt->nobj) return sys_fail(rt->cc, "I do not know that object");
    o = &rt->objs[oid];
    c = &rt->cc->classes[o->cls];
    mi = meth_lookup(c, sel);
    if (mi < 0) return sys_fail(rt->cc, "I do not know method %s", sel);
    fn = rt->meth_fn[o->cls][mi];
    ic = &rt->ics[o->cls][mi];
    if (ic->armed && ic->class_id == o->cls && ic->fn_idx == fn) {
        ic->hits++;
        g_ic_hits++;
    } else {
        ic->misses++;
        ic->armed = 1;
        ic->class_id = o->cls;
        ic->fn_idx = fn;
        g_ic_misses++;
    }
    return invoke_meth(rt, oid, fn, args, nargs, out);
}

static void write_image(Rt *rt) {
    int i, j, n = 0;
    g_image[0] = '\0';
    for (i = 1; i < rt->nobj; i++) {
        ClassDef *c;
        if (!rt->objs[i].used) continue;
        c = &rt->cc->classes[rt->objs[i].cls];
        n += snprintf(g_image + n, (n < NL_OBJ_IMAGE_SIZE) ? (size_t)(NL_OBJ_IMAGE_SIZE - n) : 0,
                      "%s#%d", c->name, i);
        for (j = 0; j < c->nslot; j++) {
            n += snprintf(g_image + n, (n < NL_OBJ_IMAGE_SIZE) ? (size_t)(NL_OBJ_IMAGE_SIZE - n) : 0,
                          " %s=%lld", c->slots[j], (long long)rt->objs[i].slots[j]);
        }
        n += snprintf(g_image + n, (n < NL_OBJ_IMAGE_SIZE) ? (size_t)(NL_OBJ_IMAGE_SIZE - n) : 0, ";");
        if (n >= NL_OBJ_IMAGE_SIZE - 1) break;
    }
}

static int run_rt(Cc *cc, NvmModule *mod, int64_t *out) {
    Rt rt;
    int i, j;
    int64_t last = 0;
    int have_last = 0;
    memset(&rt, 0, sizeof rt);
    rt.cc = cc;
    rt.mod = mod;
    rt.nobj = 1;
    g_ic_hits = g_ic_misses = 0;
    g_image[0] = '\0';
    for (i = 0; i < cc->nclass; i++) {
        for (j = 0; j < cc->classes[i].nmeth; j++) {
            int fn = lookup_fn(mod, cc->classes[i].meths[j].asm_name);
            if (fn < 0) return sys_fail(cc, "I lost function %s", cc->classes[i].meths[j].asm_name);
            cc->classes[i].meths[j].fn_idx = fn;
            rt.meth_fn[i][j] = fn;
        }
    }
    for (i = 0; i < cc->nrepl; i++) {
        int fn = lookup_fn(mod, cc->repls[i].asm_name);
        if (fn < 0) return sys_fail(cc, "I lost function %s", cc->repls[i].asm_name);
        cc->repls[i].fn_idx = fn;
    }
    vm_init(&rt.vm, mod);
    for (i = 0; i < cc->nstmt; i++) {
        Stmt *s = &cc->stmts[i];
        int64_t v = 0, oid = 0;
        if (s->kind == ST_NEW) {
            int cls = class_lookup(cc, s->cls);
            if (cls < 0) { vm_destroy(&rt.vm); return sys_fail(cc, "I do not know class %s", s->cls); }
            if (rt.nobj >= OB_MAX) { vm_destroy(&rt.vm); return sys_fail(cc, "I refuse too many objects"); }
            oid = rt.nobj;
            rt.objs[rt.nobj].used = 1;
            rt.objs[rt.nobj].cls = cls;
            rt.nobj++;
            if (bind_local(&rt, s->name, oid) < 0) { vm_destroy(&rt.vm); return sys_fail(cc, "too many names"); }
            last = oid;
            have_last = 1;
            continue;
        }
        if (s->kind == ST_HANDLE) {
            int cls = class_lookup(cc, s->cls);
            int mi;
            if (cls < 0) { vm_destroy(&rt.vm); return sys_fail(cc, "I do not know class %s", s->cls); }
            mi = meth_lookup(&cc->classes[cls], s->sel);
            if (mi < 0) { vm_destroy(&rt.vm); return sys_fail(cc, "I do not know method %s", s->sel); }
            v = rt.meth_fn[cls][mi];
            if (bind_local(&rt, s->name, v) < 0) { vm_destroy(&rt.vm); return sys_fail(cc, "too many names"); }
            last = v;
            have_last = 1;
            continue;
        }
        if (s->kind == ST_SEND) {
            if (eval_host_expr(cc, &rt, s->recv, &oid) < 0) { vm_destroy(&rt.vm); return -1; }
            if (send_sel(&rt, (int)oid, s->sel, s->args, s->nargs, &v) < 0) {
                vm_destroy(&rt.vm);
                return -1;
            }
            if (s->name[0] && bind_local(&rt, s->name, v) < 0) {
                vm_destroy(&rt.vm);
                return sys_fail(cc, "too many names");
            }
            last = v;
            have_last = 1;
            continue;
        }
        if (s->kind == ST_SENDVIA) {
            int64_t h = 0;
            if (eval_host_expr(cc, &rt, s->recv, &h) < 0) { vm_destroy(&rt.vm); return -1; }
            if (eval_host_expr(cc, &rt, s->args[0], &oid) < 0) { vm_destroy(&rt.vm); return -1; }
            if (invoke_meth(&rt, (int)oid, (int)h, NULL, 0, &v) < 0) {
                vm_destroy(&rt.vm);
                return -1;
            }
            last = v;
            have_last = 1;
            continue;
        }
        if (s->kind == ST_REPLACE) {
            Repl *r = &cc->repls[s->repl];
            int cls = class_lookup(cc, r->cls);
            int mi = meth_lookup(&cc->classes[cls], r->sel);
            rt.meth_fn[cls][mi] = r->fn_idx;
            continue;
        }
        if (s->kind == ST_EXTEND) {
            int cls = class_lookup(cc, s->cls);
            ClassDef *c;
            if (cls < 0) { vm_destroy(&rt.vm); return sys_fail(cc, "I do not know class %s", s->cls); }
            c = &cc->classes[cls];
            if (c->nslot >= OB_SLOT) { vm_destroy(&rt.vm); return sys_fail(cc, "I refuse more than eight slots"); }
            snprintf(c->slots[c->nslot], OB_NAME, "%s", s->sel);
            c->nslot++;
            continue;
        }
    }
    if (cc->result) {
        if (eval_host_expr(cc, &rt, cc->result, &last) < 0) {
            vm_destroy(&rt.vm);
            return -1;
        }
        have_last = 1;
    }
    if (!have_last) {
        vm_destroy(&rt.vm);
        return sys_fail(cc, "I expected a result");
    }
    *out = last;
    write_image(&rt);
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
    if (getenv("NL_OBJ_TRACE")) fprintf(stderr, "%s", asmbuf.p ? asmbuf.p : "");
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

NvmModule *nl_object_compile(const char *src, const char *path,
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

NlFrontendResult nl_object_accept(const NvmModule *mod, const char *path) {
    NlFrontendFacts f;
    memset(&f, 0, sizeof f);
    f.language = NL_FE_OBJECT;
    f.source_path = (path && path[0]) ? path : "<object>";
    f.purity = 0;
    f.exhaustiveness = 0;
    f.affine_use = -1;
    f.diagnostics_shared = 1;
    return nl_frontend_accept(mod, &f);
}

int nl_object_eval_i64(const char *src, int64_t *out, char *err, size_t errlen) {
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
    acc = nl_object_accept(mod, "<eval>");
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

int nl_object_last_ic_hits(void) { return g_ic_hits; }
int nl_object_last_ic_misses(void) { return g_ic_misses; }

int nl_object_last_image(char *out, size_t outlen) {
    if (!out || !outlen) return 0;
    snprintf(out, outlen, "%s", g_image);
    return 1;
}
