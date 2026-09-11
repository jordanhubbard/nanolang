/*
 * Nano ML — bounded laboratory frontend. I emit verified NanoISA.
 * C host compiler; no src_nano twin. See docs/ML.md.
 */

#include "ml.h"

#include "nanoisa/assembler.h"
#include "nanoisa/frontend.h"
#include "nanoisa/isa.h"
#include "nanoisa/nvm_format.h"
#include "nanovm/vm.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ML_NAME 64
#define ML_MAX_FN 128
#define ML_MAX_BIND 24
#define ML_MAX_UP 16
#define ML_MAX_DECL 128
#define ML_MAX_CTOR 64
#define ML_MAX_ADT 16
#define ML_MAX_ARM 16
#define ML_ERR NL_ML_ERR_SIZE

typedef enum {
    TK_EOF, TK_INT, TK_ID, TK_TYVAR,
    TK_LPAREN, TK_RPAREN, TK_COMMA, TK_BAR, TK_COLON,
    TK_EQ, TK_DARROW, TK_ARROW, TK_ASSIGN,
    TK_PLUS, TK_MINUS, TK_STAR, TK_SLASH, TK_LT, TK_GT, TK_NE,
    TK_TRUE, TK_FALSE, TK_FUN, TK_VAL, TK_FN, TK_LET, TK_IN, TK_END,
    TK_IF, TK_THEN, TK_ELSE, TK_CASE, TK_OF, TK_DATATYPE, TK_SIGNATURE,
    TK_SIG, TK_AND, TK_REF, TK_EXCEPTION, TK_UNDERSCORE
} TkKind;

typedef struct {
    TkKind kind;
    int64_t i;
    char name[ML_NAME];
    int after_nl;
} Tok;

typedef enum {
    E_INT, E_BOOL, E_VAR, E_TUPLE, E_APP, E_FN, E_IF, E_LET,
    E_CASE, E_CON, E_BIN
} EKind;

typedef enum { P_WILD, P_VAR, P_TUPLE, P_CON } PKind;

typedef struct Ty Ty;
typedef struct Expr Expr;
typedef struct Pat Pat;

struct Ty {
    enum { TY_INT, TY_BOOL, TY_VAR, TY_FUN, TY_PROD, TY_ADT } kind;
    int id;
    int adt;
    Ty *a, *b;
    Ty *link;
};

struct Pat {
    PKind kind;
    char name[ML_NAME];
    int ctor;
    Pat *x, *y;
};

typedef struct {
    Pat *pat;
    Expr *body;
} Arm;

struct Expr {
    EKind kind;
    int64_t i;
    int b;
    char name[ML_NAME];
    int ctor;
    int bin;
    Expr *x, *y, *z;
    Expr *fnbody;
    char fnparam[ML_NAME];
    char letname[ML_NAME];
    Arm arms[ML_MAX_ARM];
    int narm;
    Ty *ty;
    int fn_idx;
};

typedef struct {
    char name[ML_NAME];
    char params[ML_MAX_BIND][ML_NAME];
    int nparam;
    Expr *body;
    Ty *ann;
    int is_fun;
} ValFun;

typedef struct {
    char name[ML_NAME];
    int nty;
    char typaram[ML_NAME];
    int nctor;
    char ctor[8][ML_NAME];
    int has_payload[8];
    Ty *payload[8];
} Adt;

typedef struct {
    char name[ML_NAME];
    Ty *ty;
} SigVal;

typedef struct {
    enum { D_VALFUN, D_ADT, D_SIG, D_EXPR } kind;
    ValFun vf;
    Adt adt;
    char signame[ML_NAME];
    SigVal sigs[8];
    int nsig;
    Expr *expr;
} Decl;

typedef struct {
    char *p;
    size_t n, cap;
} Buf;

typedef struct {
    char asm_name[ML_NAME];
    char src_name[ML_NAME];
    int arity, nup, max_local;
    char params[ML_MAX_BIND][ML_NAME];
    char upnames[ML_MAX_UP][ML_NAME];
    Expr *body;
    Buf code;
    uint8_t result_tag;
    Ty *ty;
} MlFn;

typedef struct {
    int n;
    char names[ML_MAX_BIND][ML_NAME];
    int kind[ML_MAX_BIND];
    int index[ML_MAX_BIND];
} Env;

typedef struct TEnv {
    struct TEnv *parent;
    char names[ML_MAX_BIND][ML_NAME];
    Ty *tys[ML_MAX_BIND];
    int gen[ML_MAX_BIND];
    int n;
} TEnv;

typedef struct {
    const char *src;
    const char *lx;
    Tok tok;
    char err[ML_ERR];
    Decl decls[ML_MAX_DECL];
    int ndecl;
    Expr *result;
    Adt adts[ML_MAX_ADT];
    int nadt;
    struct {
        char name[ML_NAME];
        int adt, tag, has_payload;
        Ty *payload;
    } ctors[ML_MAX_CTOR];
    int nctor;
    MlFn fns[ML_MAX_FN];
    int nfn, cur, label, terminated, next_ty, exhaustive, saw_nl;
    void **heap;
    uint32_t nheap, capheap;
    SigVal specs[32];
    int nspec;
    char typebuf[ML_MAX_FN][ML_NAME];
    char typenames[ML_MAX_FN][ML_NAME];
    int ntypes;
} Cc;

static int cc_fail(Cc *cc, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(cc->err, sizeof cc->err, fmt, ap);
    va_end(ap);
    return -1;
}

static void *cc_alloc(Cc *cc, size_t n) {
    void *p = calloc(1, n);
    if (!p) return NULL;
    if (cc->nheap >= cc->capheap) {
        uint32_t cap = cc->capheap ? cc->capheap * 2 : 64;
        void **h = realloc(cc->heap, cap * sizeof(void *));
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
    for (i = 0; i < (uint32_t)cc->nfn; i++) free(cc->fns[i].code.p);
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

static Ty *ty_new(Cc *cc, int kind) {
    Ty *t = cc_alloc(cc, sizeof *t);
    if (t) t->kind = kind;
    return t;
}

static Ty *ty_int(Cc *cc) { return ty_new(cc, TY_INT); }
static Ty *ty_bool(Cc *cc) { return ty_new(cc, TY_BOOL); }

static Ty *ty_var(Cc *cc) {
    Ty *t = ty_new(cc, TY_VAR);
    if (t) t->id = cc->next_ty++;
    return t;
}

static Ty *ty_fun(Cc *cc, Ty *a, Ty *b) {
    Ty *t = ty_new(cc, TY_FUN);
    if (t) { t->a = a; t->b = b; }
    return t;
}

static Ty *ty_prod(Cc *cc, Ty *a, Ty *b) {
    Ty *t = ty_new(cc, TY_PROD);
    if (t) { t->a = a; t->b = b; }
    return t;
}

static Ty *ty_adt(Cc *cc, int adt) {
    Ty *t = ty_new(cc, TY_ADT);
    if (t) t->adt = adt;
    return t;
}

static Ty *prune(Ty *t) {
    if (!t) return NULL;
    if (t->kind == TY_VAR && t->link) {
        t->link = prune(t->link);
        return t->link;
    }
    return t;
}

static int occurs(Ty *v, Ty *t) {
    t = prune(t);
    v = prune(v);
    if (!t || !v) return 0;
    if (v->kind != TY_VAR) return 0;
    if (t == v) return 1;
    if (t->kind == TY_FUN || t->kind == TY_PROD)
        return occurs(v, t->a) || occurs(v, t->b);
    return 0;
}

static int unify(Cc *cc, Ty *a, Ty *b) {
    a = prune(a); b = prune(b);
    if (!a || !b) return cc_fail(cc, "I cannot unify a missing type");
    if (a == b) return 0;
    if (a->kind == TY_VAR) {
        if (occurs(a, b))
            return cc_fail(cc, "I refuse an infinite type (%d into %d)", a->kind, prune(b)->kind);
        a->link = b;
        return 0;
    }
    if (b->kind == TY_VAR) return unify(cc, b, a);
    if (a->kind != b->kind) return cc_fail(cc, "I cannot unify those types");
    if (a->kind == TY_INT || a->kind == TY_BOOL) return 0;
    if (a->kind == TY_ADT) {
        if (a->adt != b->adt) return cc_fail(cc, "I cannot unify distinct datatypes");
        return 0;
    }
    if (unify(cc, a->a, b->a) < 0) return -1;
    return unify(cc, a->b, b->b);
}

static Ty *fresh_copy(Cc *cc, Ty *t, Ty **map, int nmap, int *nmapw);

static Ty *inst(Cc *cc, Ty *t) {
    Ty *map[64];
    int n = 0;
    memset(map, 0, sizeof map);
    return fresh_copy(cc, t, map, 64, &n);
}

static Ty *fresh_copy(Cc *cc, Ty *t, Ty **map, int nmap, int *nmapw) {
    int i;
    t = prune(t);
    if (!t) return NULL;
    if (t->kind == TY_VAR) {
        for (i = 0; i < *nmapw; i += 2) {
            if (map[i] == t) return map[i + 1];
        }
        if (*nmapw + 1 >= nmap) return t;
        map[*nmapw] = t;
        map[*nmapw + 1] = ty_var(cc);
        *nmapw += 2;
        return map[*nmapw - 1];
    }
    if (t->kind == TY_FUN) return ty_fun(cc, fresh_copy(cc, t->a, map, nmap, nmapw),
                                        fresh_copy(cc, t->b, map, nmap, nmapw));
    if (t->kind == TY_PROD) return ty_prod(cc, fresh_copy(cc, t->a, map, nmap, nmapw),
                                          fresh_copy(cc, t->b, map, nmap, nmapw));
    if (t->kind == TY_ADT) return ty_adt(cc, t->adt);
    if (t->kind == TY_BOOL) return ty_bool(cc);
    return ty_int(cc);
}

static Expr *ex_new(Cc *cc, EKind k) {
    Expr *e = cc_alloc(cc, sizeof *e);
    if (e) { e->kind = k; e->fn_idx = -1; e->ctor = -1; }
    return e;
}

static Pat *pat_new(Cc *cc, PKind k) {
    Pat *p = cc_alloc(cc, sizeof *p);
    if (p) { p->kind = k; p->ctor = -1; }
    return p;
}

static void skip_ws(Cc *cc) {
    cc->saw_nl = 0;
    for (;;) {
        while (*cc->lx && isspace((unsigned char)*cc->lx)) {
            if (*cc->lx == '\n' || *cc->lx == '\r') cc->saw_nl = 1;
            cc->lx++;
        }
        if (cc->lx[0] == '(' && cc->lx[1] == '*') {
            cc->lx += 2;
            while (*cc->lx && !(cc->lx[0] == '*' && cc->lx[1] == ')')) {
                if (*cc->lx == '\n' || *cc->lx == '\r') cc->saw_nl = 1;
                cc->lx++;
            }
            if (*cc->lx) cc->lx += 2;
            continue;
        }
        break;
    }
}

static int kw_eq(const char *s, const char *k) {
    return strcmp(s, k) == 0;
}

static void lex(Cc *cc) {
    const char *p;
    int after_nl;
    skip_ws(cc);
    p = cc->lx;
    after_nl = cc->saw_nl;
    memset(&cc->tok, 0, sizeof cc->tok);
    cc->tok.after_nl = after_nl;
    if (!*p) { cc->tok.kind = TK_EOF; return; }
    if (isdigit((unsigned char)*p) || (*p == '-' && isdigit((unsigned char)p[1]))) {
        int64_t v = 0; int neg = 0;
        if (*p == '-') { neg = 1; p++; }
        while (isdigit((unsigned char)*p)) { v = v * 10 + (*p - '0'); p++; }
        cc->tok.kind = TK_INT;
        cc->tok.i = neg ? -v : v;
        cc->lx = p;
        return;
    }
    if (*p == '\'') {
        int n = 0;
        p++;
        cc->tok.kind = TK_TYVAR;
        while (isalnum((unsigned char)*p) && n < ML_NAME - 1)
            cc->tok.name[n++] = *p++;
        cc->lx = p;
        return;
    }
    if (isalpha((unsigned char)*p) || *p == '_') {
        int n = 0;
        if (*p == '_' && !isalnum((unsigned char)p[1])) {
            cc->tok.kind = TK_UNDERSCORE;
            cc->lx = p + 1;
            return;
        }
        while ((isalnum((unsigned char)*p) || *p == '_') && n < ML_NAME - 1)
            cc->tok.name[n++] = *p++;
        cc->lx = p;
        if (kw_eq(cc->tok.name, "true")) cc->tok.kind = TK_TRUE;
        else if (kw_eq(cc->tok.name, "false")) cc->tok.kind = TK_FALSE;
        else if (kw_eq(cc->tok.name, "fun")) cc->tok.kind = TK_FUN;
        else if (kw_eq(cc->tok.name, "val")) cc->tok.kind = TK_VAL;
        else if (kw_eq(cc->tok.name, "fn")) cc->tok.kind = TK_FN;
        else if (kw_eq(cc->tok.name, "let")) cc->tok.kind = TK_LET;
        else if (kw_eq(cc->tok.name, "in")) cc->tok.kind = TK_IN;
        else if (kw_eq(cc->tok.name, "end")) cc->tok.kind = TK_END;
        else if (kw_eq(cc->tok.name, "if")) cc->tok.kind = TK_IF;
        else if (kw_eq(cc->tok.name, "then")) cc->tok.kind = TK_THEN;
        else if (kw_eq(cc->tok.name, "else")) cc->tok.kind = TK_ELSE;
        else if (kw_eq(cc->tok.name, "case")) cc->tok.kind = TK_CASE;
        else if (kw_eq(cc->tok.name, "of")) cc->tok.kind = TK_OF;
        else if (kw_eq(cc->tok.name, "datatype")) cc->tok.kind = TK_DATATYPE;
        else if (kw_eq(cc->tok.name, "signature")) cc->tok.kind = TK_SIGNATURE;
        else if (kw_eq(cc->tok.name, "sig")) cc->tok.kind = TK_SIG;
        else if (kw_eq(cc->tok.name, "and")) cc->tok.kind = TK_AND;
        else if (kw_eq(cc->tok.name, "ref")) cc->tok.kind = TK_REF;
        else if (kw_eq(cc->tok.name, "exception")) cc->tok.kind = TK_EXCEPTION;
        else cc->tok.kind = TK_ID;
        return;
    }
    if (p[0] == '=' && p[1] == '>') { cc->tok.kind = TK_DARROW; cc->lx = p + 2; return; }
    if (p[0] == '-' && p[1] == '>') { cc->tok.kind = TK_ARROW; cc->lx = p + 2; return; }
    if (p[0] == ':' && p[1] == '=') { cc->tok.kind = TK_ASSIGN; cc->lx = p + 2; return; }
    if (p[0] == '<' && p[1] == '>') { cc->tok.kind = TK_NE; cc->lx = p + 2; return; }
    cc->lx = p + 1;
    switch (*p) {
    case '(': cc->tok.kind = TK_LPAREN; break;
    case ')': cc->tok.kind = TK_RPAREN; break;
    case ',': cc->tok.kind = TK_COMMA; break;
    case '|': cc->tok.kind = TK_BAR; break;
    case ':': cc->tok.kind = TK_COLON; break;
    case '=': cc->tok.kind = TK_EQ; break;
    case '+': cc->tok.kind = TK_PLUS; break;
    case '-': cc->tok.kind = TK_MINUS; break;
    case '*': cc->tok.kind = TK_STAR; break;
    case '/': cc->tok.kind = TK_SLASH; break;
    case '<': cc->tok.kind = TK_LT; break;
    case '>': cc->tok.kind = TK_GT; break;
    default: cc->tok.kind = TK_EOF; snprintf(cc->err, sizeof cc->err, "I refuse character '%c'", *p); break;
    }
}

static int have(Cc *cc, TkKind k) { return cc->tok.kind == k; }
static int eat(Cc *cc, TkKind k) {
    if (!have(cc, k)) return 0;
    lex(cc);
    return 1;
}

static Expr *parse_expr(Cc *cc);
static Ty *parse_type(Cc *cc);
static Pat *parse_pat(Cc *cc);

static int ctor_lookup(Cc *cc, const char *n) {
    int i;
    for (i = 0; i < cc->nctor; i++) {
        if (strcmp(cc->ctors[i].name, n) == 0) return i;
    }
    return -1;
}

static int is_ctor_name(const char *n) {
    return n[0] >= 'A' && n[0] <= 'Z';
}

static Expr *parse_atom(Cc *cc) {
    Expr *e;
    if (have(cc, TK_REF)) { cc_fail(cc, "I refuse ref; mutation is out of scope"); return NULL; }
    if (have(cc, TK_EXCEPTION)) { cc_fail(cc, "I refuse exception; exceptions are out of scope"); return NULL; }
    if (have(cc, TK_ASSIGN)) { cc_fail(cc, "I refuse :=; assignment is out of scope"); return NULL; }
    if (have(cc, TK_INT)) {
        e = ex_new(cc, E_INT); if (!e) return NULL;
        e->i = cc->tok.i; lex(cc); return e;
    }
    if (have(cc, TK_TRUE) || have(cc, TK_FALSE)) {
        e = ex_new(cc, E_BOOL); if (!e) return NULL;
        e->b = have(cc, TK_TRUE); lex(cc); return e;
    }
    if (have(cc, TK_ID)) {
        e = ex_new(cc, is_ctor_name(cc->tok.name) ? E_CON : E_VAR);
        if (!e) return NULL;
        snprintf(e->name, sizeof e->name, "%s", cc->tok.name);
        lex(cc);
        return e;
    }
    if (eat(cc, TK_LPAREN)) {
        Expr *a, *b;
        if (have(cc, TK_RPAREN)) { cc_fail(cc, "I refuse an empty tuple"); return NULL; }
        a = parse_expr(cc); if (!a) return NULL;
        if (eat(cc, TK_COMMA)) {
            b = parse_expr(cc); if (!b) return NULL;
            if (!eat(cc, TK_RPAREN)) { cc_fail(cc, "I expected ')' after a pair"); return NULL; }
            e = ex_new(cc, E_TUPLE); if (!e) return NULL;
            e->x = a; e->y = b; return e;
        }
        if (!eat(cc, TK_RPAREN)) { cc_fail(cc, "I expected ')'"); return NULL; }
        return a;
    }
    cc_fail(cc, "I expected an expression");
    return NULL;
}

static int atom_start(Cc *cc) {
    return have(cc, TK_INT) || have(cc, TK_TRUE) || have(cc, TK_FALSE)
        || have(cc, TK_ID) || have(cc, TK_LPAREN);
}

static Expr *parse_app(Cc *cc) {
    Expr *e = parse_atom(cc);
    if (!e) return NULL;
    /* Juxtaposition does not cross a newline, so
       `fun id x = x` / `id 41` is a body and a program, not `x id 41`. */
    while (atom_start(cc) && !cc->tok.after_nl) {
        Expr *arg = parse_atom(cc);
        Expr *app;
        if (!arg) return NULL;
        app = ex_new(cc, E_APP); if (!app) return NULL;
        app->x = e; app->y = arg;
        e = app;
    }
    return e;
}

static int bin_prec(TkKind k) {
    switch (k) {
    case TK_STAR: case TK_SLASH: return 3;
    case TK_PLUS: case TK_MINUS: return 2;
    case TK_EQ: case TK_LT: case TK_GT: case TK_NE: return 1;
    default: return -1;
    }
}

static Expr *parse_bin(Cc *cc, int minp) {
    Expr *left = parse_app(cc);
    if (!left) return NULL;
    for (;;) {
        int p = bin_prec(cc->tok.kind);
        Expr *right, *n;
        TkKind op;
        if (p < minp) break;
        op = cc->tok.kind;
        lex(cc);
        right = parse_bin(cc, p + 1);
        if (!right) return NULL;
        n = ex_new(cc, E_BIN); if (!n) return NULL;
        n->bin = (int)op;
        n->x = left; n->y = right;
        left = n;
    }
    return left;
}

static Pat *parse_pat(Cc *cc) {
    Pat *p;
    if (eat(cc, TK_UNDERSCORE)) return pat_new(cc, P_WILD);
    if (have(cc, TK_ID)) {
        if (is_ctor_name(cc->tok.name)) {
            p = pat_new(cc, P_CON); if (!p) return NULL;
            snprintf(p->name, sizeof p->name, "%s", cc->tok.name);
            lex(cc);
            if (atom_start(cc) || have(cc, TK_UNDERSCORE) || have(cc, TK_ID) || have(cc, TK_LPAREN)) {
                p->x = parse_pat(cc);
                if (!p->x) return NULL;
            }
            return p;
        }
        p = pat_new(cc, P_VAR); if (!p) return NULL;
        snprintf(p->name, sizeof p->name, "%s", cc->tok.name);
        lex(cc);
        return p;
    }
    if (eat(cc, TK_LPAREN)) {
        Pat *a = parse_pat(cc); Pat *b;
        if (!a) return NULL;
        if (!eat(cc, TK_COMMA)) { cc_fail(cc, "I expected a pair pattern"); return NULL; }
        b = parse_pat(cc); if (!b) return NULL;
        if (!eat(cc, TK_RPAREN)) { cc_fail(cc, "I expected ')' in a pattern"); return NULL; }
        p = pat_new(cc, P_TUPLE); if (!p) return NULL;
        p->x = a; p->y = b; return p;
    }
    cc_fail(cc, "I expected a pattern");
    return NULL;
}

static Expr *parse_fn(Cc *cc) {
    Expr *e;
    if (!eat(cc, TK_FN)) return NULL;
    if (!have(cc, TK_ID)) { cc_fail(cc, "I expected a parameter after fn"); return NULL; }
    e = ex_new(cc, E_FN); if (!e) return NULL;
    snprintf(e->fnparam, sizeof e->fnparam, "%s", cc->tok.name);
    lex(cc);
    if (have(cc, TK_ID)) {
        /* fn a b => e  desugars to fn a => fn b => e */
        Expr *inner;
        char restname[ML_NAME];
        snprintf(restname, sizeof restname, "%s", cc->tok.name);
        lex(cc);
        while (have(cc, TK_ID)) {
            cc_fail(cc, "I only take two fn parameters in this subset");
            return NULL;
        }
        if (!eat(cc, TK_DARROW)) { cc_fail(cc, "I expected => after fn"); return NULL; }
        inner = ex_new(cc, E_FN); if (!inner) return NULL;
        snprintf(inner->fnparam, sizeof inner->fnparam, "%s", restname);
        inner->fnbody = parse_expr(cc);
        if (!inner->fnbody) return NULL;
        e->fnbody = inner;
        return e;
    }
    if (!eat(cc, TK_DARROW)) { cc_fail(cc, "I expected => after fn"); return NULL; }
    e->fnbody = parse_expr(cc);
    return e->fnbody ? e : NULL;
}

static Expr *parse_expr(Cc *cc) {
    if (have(cc, TK_ASSIGN)) { cc_fail(cc, "I refuse :=; assignment is out of scope"); return NULL; }
    if (have(cc, TK_REF)) { cc_fail(cc, "I refuse ref; mutation is out of scope"); return NULL; }
    if (have(cc, TK_EXCEPTION)) { cc_fail(cc, "I refuse exception; exceptions are out of scope"); return NULL; }
    if (have(cc, TK_FN)) return parse_fn(cc);
    if (eat(cc, TK_IF)) {
        Expr *e = ex_new(cc, E_IF); if (!e) return NULL;
        e->x = parse_expr(cc); if (!e->x) return NULL;
        if (!eat(cc, TK_THEN)) { cc_fail(cc, "I expected then"); return NULL; }
        e->y = parse_expr(cc); if (!e->y) return NULL;
        if (!eat(cc, TK_ELSE)) { cc_fail(cc, "I expected else"); return NULL; }
        e->z = parse_expr(cc); return e->z ? e : NULL;
    }
    if (eat(cc, TK_LET)) {
        Expr *e = ex_new(cc, E_LET); if (!e) return NULL;
        if (!eat(cc, TK_VAL)) { cc_fail(cc, "I expected val after let"); return NULL; }
        if (!have(cc, TK_ID)) { cc_fail(cc, "I expected a name after val"); return NULL; }
        snprintf(e->letname, sizeof e->letname, "%s", cc->tok.name);
        lex(cc);
        if (!eat(cc, TK_EQ)) { cc_fail(cc, "I expected = in let"); return NULL; }
        e->x = parse_expr(cc); if (!e->x) return NULL;
        if (!eat(cc, TK_IN)) { cc_fail(cc, "I expected in"); return NULL; }
        e->y = parse_expr(cc); if (!e->y) return NULL;
        if (!eat(cc, TK_END)) { cc_fail(cc, "I expected end"); return NULL; }
        return e;
    }
    if (eat(cc, TK_CASE)) {
        Expr *e = ex_new(cc, E_CASE); if (!e) return NULL;
        e->x = parse_expr(cc); if (!e->x) return NULL;
        if (!eat(cc, TK_OF)) { cc_fail(cc, "I expected of"); return NULL; }
        do {
            if (e->narm >= ML_MAX_ARM) { cc_fail(cc, "I refuse too many case arms"); return NULL; }
            e->arms[e->narm].pat = parse_pat(cc);
            if (!e->arms[e->narm].pat) return NULL;
            if (!eat(cc, TK_DARROW)) { cc_fail(cc, "I expected => in case"); return NULL; }
            e->arms[e->narm].body = parse_bin(cc, 0);
            if (!e->arms[e->narm].body) return NULL;
            e->narm++;
        } while (eat(cc, TK_BAR));
        return e;
    }
    return parse_bin(cc, 0);
}

static Ty *parse_type_atom(Cc *cc) {
    if (have(cc, TK_TYVAR)) { lex(cc); return ty_var(cc); }
    if (have(cc, TK_ID)) {
        if (strcmp(cc->tok.name, "int") == 0) { lex(cc); return ty_int(cc); }
        if (strcmp(cc->tok.name, "bool") == 0) { lex(cc); return ty_bool(cc); }
        {
            int i;
            char n[ML_NAME];
            snprintf(n, sizeof n, "%s", cc->tok.name);
            lex(cc);
            for (i = 0; i < cc->nadt; i++) {
                if (strcmp(cc->adts[i].name, n) == 0) return ty_adt(cc, i);
            }
            cc_fail(cc, "I do not know type %s", n);
            return NULL;
        }
    }
    if (eat(cc, TK_LPAREN)) {
        Ty *t = parse_type(cc);
        if (!t) return NULL;
        if (!eat(cc, TK_RPAREN)) { cc_fail(cc, "I expected ')' in a type"); return NULL; }
        return t;
    }
    cc_fail(cc, "I expected a type");
    return NULL;
}

static Ty *parse_type_prod(Cc *cc) {
    Ty *t = parse_type_atom(cc);
    if (!t) return NULL;
    while (eat(cc, TK_STAR)) {
        Ty *r = parse_type_atom(cc);
        if (!r) return NULL;
        t = ty_prod(cc, t, r);
        if (!t) return NULL;
    }
    return t;
}

static Ty *parse_type(Cc *cc) {
    Ty *t = parse_type_prod(cc);
    if (!t) return NULL;
    if (eat(cc, TK_ARROW)) {
        Ty *r = parse_type(cc);
        if (!r) return NULL;
        return ty_fun(cc, t, r);
    }
    return t;
}

static int parse_datatype(Cc *cc, Decl *d) {
    Adt *a = &d->adt;
    d->kind = D_ADT;
    if (have(cc, TK_TYVAR)) {
        snprintf(a->typaram, sizeof a->typaram, "%s", cc->tok.name);
        a->nty = 1;
        lex(cc);
    }
    if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a datatype name");
    snprintf(a->name, sizeof a->name, "%s", cc->tok.name);
    lex(cc);
    if (!eat(cc, TK_EQ)) return cc_fail(cc, "I expected = in datatype");
    do {
        if (a->nctor >= 8) return cc_fail(cc, "I refuse too many constructors");
        if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a constructor");
        snprintf(a->ctor[a->nctor], sizeof a->ctor[0], "%s", cc->tok.name);
        lex(cc);
        if (eat(cc, TK_OF)) {
            a->has_payload[a->nctor] = 1;
            a->payload[a->nctor] = parse_type(cc);
            if (!a->payload[a->nctor]) return -1;
        }
        a->nctor++;
    } while (eat(cc, TK_BAR));
    return 0;
}

static int parse_signature(Cc *cc, Decl *d) {
    d->kind = D_SIG;
    if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a signature name");
    snprintf(d->signame, sizeof d->signame, "%s", cc->tok.name);
    lex(cc);
    if (!eat(cc, TK_EQ) || !eat(cc, TK_SIG))
        return cc_fail(cc, "I expected = sig");
    while (eat(cc, TK_VAL)) {
        if (d->nsig >= 8) return cc_fail(cc, "I refuse too many signature vals");
        if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a val name");
        snprintf(d->sigs[d->nsig].name, sizeof d->sigs[0].name, "%s", cc->tok.name);
        lex(cc);
        if (!eat(cc, TK_COLON)) return cc_fail(cc, "I expected : in signature");
        d->sigs[d->nsig].ty = parse_type(cc);
        if (!d->sigs[d->nsig].ty) return -1;
        d->nsig++;
    }
    if (!eat(cc, TK_END)) return cc_fail(cc, "I expected end of signature");
    return 0;
}

static int parse_fun(Cc *cc, Decl *d) {
    d->kind = D_VALFUN;
    d->vf.is_fun = 1;
    if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a function name");
    snprintf(d->vf.name, sizeof d->vf.name, "%s", cc->tok.name);
    lex(cc);
    while (have(cc, TK_ID) || have(cc, TK_LPAREN)) {
        if (d->vf.nparam >= ML_MAX_BIND) return cc_fail(cc, "I refuse too many parameters");
        if (eat(cc, TK_LPAREN)) {
            if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a parameter");
            snprintf(d->vf.params[d->vf.nparam], ML_NAME, "%s", cc->tok.name);
            d->vf.nparam++;
            lex(cc);
            if (eat(cc, TK_COMMA)) {
                /* (a, b) is one tuple parameter — keep as single param named by first
                   and rewrite body... simpler: treat as two params via tuple sugar.
                   Store as one param "_p" and wrap body in case. */
                return cc_fail(cc, "I take tuple parameters as 'p' and match with case");
            }
            if (!eat(cc, TK_RPAREN)) return cc_fail(cc, "I expected ')'");
        } else {
            snprintf(d->vf.params[d->vf.nparam], ML_NAME, "%s", cc->tok.name);
            d->vf.nparam++;
            lex(cc);
        }
    }
    if (d->vf.nparam < 1) return cc_fail(cc, "I expected a parameter");
    if (!eat(cc, TK_EQ)) return cc_fail(cc, "I expected = after fun");
    d->vf.body = parse_expr(cc);
    return d->vf.body ? 0 : -1;
}

static int parse_program(Cc *cc) {
    cc->lx = cc->src;
    lex(cc);
    while (!have(cc, TK_EOF)) {
        Decl *d;
        if (cc->ndecl >= ML_MAX_DECL) return cc_fail(cc, "I refuse too many declarations");
        if (eat(cc, TK_DATATYPE)) {
            d = &cc->decls[cc->ndecl];
            memset(d, 0, sizeof *d);
            if (parse_datatype(cc, d) < 0) return -1;
            {
                int i, ai = cc->nadt;
                if (ai >= ML_MAX_ADT) return cc_fail(cc, "I refuse too many datatypes");
                cc->adts[ai] = d->adt;
                cc->nadt++;
                for (i = 0; i < d->adt.nctor; i++) {
                    if (cc->nctor >= ML_MAX_CTOR) return cc_fail(cc, "I refuse too many constructors");
                    snprintf(cc->ctors[cc->nctor].name, ML_NAME, "%s", d->adt.ctor[i]);
                    cc->ctors[cc->nctor].adt = ai;
                    cc->ctors[cc->nctor].tag = i;
                    cc->ctors[cc->nctor].has_payload = d->adt.has_payload[i];
                    cc->ctors[cc->nctor].payload = d->adt.payload[i];
                    cc->nctor++;
                }
            }
            cc->ndecl++;
            continue;
        }
        if (eat(cc, TK_SIGNATURE)) {
            d = &cc->decls[cc->ndecl];
            memset(d, 0, sizeof *d);
            if (parse_signature(cc, d) < 0) return -1;
            {
                int i;
                for (i = 0; i < d->nsig; i++) {
                    if (cc->nspec >= 32) return cc_fail(cc, "I refuse too many specs");
                    cc->specs[cc->nspec++] = d->sigs[i];
                }
            }
            cc->ndecl++;
            continue;
        }
        if (eat(cc, TK_FUN)) {
            d = &cc->decls[cc->ndecl];
            memset(d, 0, sizeof *d);
            if (parse_fun(cc, d) < 0) return -1;
            cc->ndecl++;
            continue;
        }
        if (have(cc, TK_AND)) return cc_fail(cc, "I refuse mutually recursive and");
        if (eat(cc, TK_VAL)) {
            if (have(cc, TK_ID)) lex(cc);
            if (eat(cc, TK_EQ)) {
                if (!parse_expr(cc)) return -1;
            }
            continue;
        }
        /* trailing expression */
        cc->result = parse_expr(cc);
        if (!cc->result) return -1;
        if (have(cc, TK_ASSIGN))
            return cc_fail(cc, "I refuse :=; assignment is out of scope");
        if (!have(cc, TK_EOF)) return cc_fail(cc, "I expected the program to end");
        break;
    }
    return 0;
}

/* The rest of the compiler continues below. */
static Expr *wrap_fun_body(Cc *cc, ValFun *vf) {
    Expr *body = vf->body;
    int i;
    for (i = vf->nparam - 1; i >= 1; i--) {
        Expr *fn = ex_new(cc, E_FN);
        if (!fn) return NULL;
        snprintf(fn->fnparam, sizeof fn->fnparam, "%s", vf->params[i]);
        fn->fnbody = body;
        body = fn;
    }
    return body;
}

static int find_fn_src(Cc *cc, const char *name) {
    int i;
    for (i = 0; i < cc->nfn; i++) {
        if (cc->fns[i].src_name[0] && strcmp(cc->fns[i].src_name, name) == 0)
            return i;
    }
    return -1;
}

static int env_find(const Env *env, const char *name) {
    int i;
    for (i = env->n - 1; i >= 0; i--) {
        if (strcmp(env->names[i], name) == 0) return i;
    }
    return -1;
}

static int tenv_find(TEnv *e, const char *name, Ty **out, int *gen) {
    for (; e; e = e->parent) {
        int i;
        for (i = e->n - 1; i >= 0; i--) {
            if (strcmp(e->names[i], name) == 0) {
                *out = e->tys[i];
                *gen = e->gen[i];
                return 0;
            }
        }
    }
    return -1;
}

static int tenv_bind(TEnv *e, const char *name, Ty *t, int gen) {
    if (e->n >= ML_MAX_BIND) return -1;
    snprintf(e->names[e->n], ML_NAME, "%s", name);
    e->tys[e->n] = t;
    e->gen[e->n] = gen;
    e->n++;
    return 0;
}

static int is_bound_name(const char *n, const char *params[], int np,
                         const char *ups[], int nu) {
    int i;
    for (i = 0; i < np; i++) if (params[i] && strcmp(params[i], n) == 0) return 1;
    for (i = 0; i < nu; i++) if (ups[i] && strcmp(ups[i], n) == 0) return 1;
    return 0;
}

static int collect_free(Cc *cc, Expr *e, const char *params[], int np,
                        char ups[][ML_NAME], int *nu, int maxu);

static int collect_free_pat(Pat *p, char locals[][ML_NAME], int *nl, int maxl) {
    if (!p) return 0;
    if (p->kind == P_VAR) {
        if (*nl >= maxl) return -1;
        snprintf(locals[*nl], ML_NAME, "%s", p->name);
        (*nl)++;
        return 0;
    }
    if (p->kind == P_TUPLE) {
        if (collect_free_pat(p->x, locals, nl, maxl) < 0) return -1;
        return collect_free_pat(p->y, locals, nl, maxl);
    }
    if (p->kind == P_CON) return collect_free_pat(p->x, locals, nl, maxl);
    return 0;
}

static int add_up(char ups[][ML_NAME], int *nu, int maxu, const char *n) {
    int i;
    for (i = 0; i < *nu; i++) if (strcmp(ups[i], n) == 0) return 0;
    if (*nu >= maxu) return -1;
    snprintf(ups[*nu], ML_NAME, "%s", n);
    (*nu)++;
    return 0;
}

static int collect_free(Cc *cc, Expr *e, const char *params[], int np,
                        char ups[][ML_NAME], int *nu, int maxu) {
    int i;
    if (!e) return 0;
    switch (e->kind) {
    case E_INT: case E_BOOL: case E_CON: return 0;
    case E_VAR:
        if (find_fn_src(cc, e->name) >= 0) return 0;
        if (ctor_lookup(cc, e->name) >= 0) return 0;
        if (!is_bound_name(e->name, params, np, NULL, 0))
            return add_up(ups, nu, maxu, e->name);
        return 0;
    case E_TUPLE: case E_APP: case E_BIN:
        if (collect_free(cc, e->x, params, np, ups, nu, maxu) < 0) return -1;
        return collect_free(cc, e->y, params, np, ups, nu, maxu);
    case E_IF:
        if (collect_free(cc, e->x, params, np, ups, nu, maxu) < 0) return -1;
        if (collect_free(cc, e->y, params, np, ups, nu, maxu) < 0) return -1;
        return collect_free(cc, e->z, params, np, ups, nu, maxu);
    case E_LET: {
        const char *p2[ML_MAX_BIND];
        int n2 = np;
        for (i = 0; i < np; i++) p2[i] = params[i];
        if (collect_free(cc, e->x, params, np, ups, nu, maxu) < 0) return -1;
        if (n2 < ML_MAX_BIND) p2[n2++] = e->letname;
        return collect_free(cc, e->y, p2, n2, ups, nu, maxu);
    }
    case E_FN: {
        char inner_ups[ML_MAX_UP][ML_NAME];
        const char *p2[1];
        int ni = 0, j;
        p2[0] = e->fnparam;
        if (collect_free(cc, e->fnbody, p2, 1, inner_ups, &ni, ML_MAX_UP) < 0) return -1;
        for (j = 0; j < ni; j++) {
            if (!is_bound_name(inner_ups[j], params, np, NULL, 0)) {
                if (add_up(ups, nu, maxu, inner_ups[j]) < 0) return -1;
            }
        }
        return 0;
    }
    case E_CASE: {
        if (collect_free(cc, e->x, params, np, ups, nu, maxu) < 0) return -1;
        for (i = 0; i < e->narm; i++) {
            char loc[ML_MAX_BIND][ML_NAME];
            const char *p2[ML_MAX_BIND];
            int nl = 0, n2 = 0, j;
            if (collect_free_pat(e->arms[i].pat, loc, &nl, ML_MAX_BIND) < 0) return -1;
            for (j = 0; j < np; j++) p2[n2++] = params[j];
            for (j = 0; j < nl && n2 < ML_MAX_BIND; j++) p2[n2++] = loc[j];
            if (collect_free(cc, e->arms[i].body, p2, n2, ups, nu, maxu) < 0) return -1;
        }
        return 0;
    }
    }
    return 0;
}

static int register_one_fn(Cc *cc, const char *src_name, const char *param,
                           Expr *body, int *out_idx) {
    MlFn *fn;
    const char *ps[1];
    if (cc->nfn >= ML_MAX_FN) return cc_fail(cc, "I refuse more than %d functions", ML_MAX_FN);
    fn = &cc->fns[cc->nfn];
    memset(fn, 0, sizeof *fn);
    fn->result_tag = 0xFF;
    if (src_name && src_name[0])
        snprintf(fn->src_name, sizeof fn->src_name, "%s", src_name);
    if (src_name && src_name[0])
        snprintf(fn->asm_name, sizeof fn->asm_name, "ml_%s", src_name);
    else
        snprintf(fn->asm_name, sizeof fn->asm_name, "_L%d", cc->nfn);
    fn->arity = 1;
    snprintf(fn->params[0], ML_NAME, "%s", param);
    fn->body = body;
    ps[0] = param;
    if (collect_free(cc, body, ps, 1, fn->upnames, &fn->nup, ML_MAX_UP) < 0)
        return cc_fail(cc, "I refuse too many captures");
    if (out_idx) *out_idx = cc->nfn;
    cc->nfn++;
    return 0;
}

static int walk_expr_fns(Cc *cc, Expr *e);

static int walk_expr_fns(Cc *cc, Expr *e) {
    int i;
    if (!e) return 0;
    switch (e->kind) {
    case E_INT: case E_BOOL: case E_VAR: case E_CON: return 0;
    case E_TUPLE: case E_APP: case E_BIN:
        if (walk_expr_fns(cc, e->x) < 0) return -1;
        return walk_expr_fns(cc, e->y);
    case E_IF:
        if (walk_expr_fns(cc, e->x) < 0) return -1;
        if (walk_expr_fns(cc, e->y) < 0) return -1;
        return walk_expr_fns(cc, e->z);
    case E_LET:
        if (walk_expr_fns(cc, e->x) < 0) return -1;
        return walk_expr_fns(cc, e->y);
    case E_CASE:
        if (walk_expr_fns(cc, e->x) < 0) return -1;
        for (i = 0; i < e->narm; i++) {
            if (walk_expr_fns(cc, e->arms[i].body) < 0) return -1;
        }
        return 0;
    case E_FN:
        if (walk_expr_fns(cc, e->fnbody) < 0) return -1;
        if (register_one_fn(cc, NULL, e->fnparam, e->fnbody, &e->fn_idx) < 0)
            return -1;
        return 0;
    }
    return 0;
}

static uint8_t ty_tag(Ty *t);

static Ty *infer(Cc *cc, TEnv *env, Expr *e);

static int bind_pat(Cc *cc, TEnv *env, Pat *p, Ty *t) {
    int c;
    t = prune(t);
    if (!p) return cc_fail(cc, "I expected a pattern");
    if (p->kind == P_WILD) return 0;
    if (p->kind == P_VAR) return tenv_bind(env, p->name, t, 0) < 0
        ? cc_fail(cc, "I refuse too many bindings") : 0;
    if (p->kind == P_TUPLE) {
        if (t->kind == TY_VAR) {
            if (unify(cc, t, ty_prod(cc, ty_var(cc), ty_var(cc))) < 0) return -1;
            t = prune(t);
        }
        if (!t || t->kind != TY_PROD) return cc_fail(cc, "I expected a pair pattern on a pair");
        if (bind_pat(cc, env, p->x, t->a) < 0) return -1;
        return bind_pat(cc, env, p->y, t->b);
    }
    c = ctor_lookup(cc, p->name);
    if (c < 0) return cc_fail(cc, "I do not know constructor %s", p->name);
    p->ctor = c;
    if (unify(cc, t, ty_adt(cc, cc->ctors[c].adt)) < 0) return -1;
    if (cc->ctors[c].has_payload) {
        if (!p->x) return cc_fail(cc, "I expected a payload for %s", p->name);
        return bind_pat(cc, env, p->x, cc->ctors[c].payload ? cc->ctors[c].payload : ty_int(cc));
    }
    if (p->x) return cc_fail(cc, "%s takes no payload", p->name);
    return 0;
}

static int case_exhaustive(Cc *cc, Expr *e, Ty *st) {
    int i, adt = -1, seen = 0, has_wild = 0, nctor = 0;
    st = prune(st);
    if (st && st->kind == TY_PROD) {
        for (i = 0; i < e->narm; i++) {
            if (e->arms[i].pat->kind == P_TUPLE || e->arms[i].pat->kind == P_WILD
                || e->arms[i].pat->kind == P_VAR) return 1;
        }
        return 0;
    }
    if (st && st->kind == TY_ADT) adt = st->adt;
    if (adt < 0) return 1;
    nctor = cc->adts[adt].nctor;
    for (i = 0; i < e->narm; i++) {
        Pat *p = e->arms[i].pat;
        if (p->kind == P_WILD || p->kind == P_VAR) { has_wild = 1; continue; }
        if (p->kind == P_CON) {
            int c = ctor_lookup(cc, p->name);
            if (c >= 0) seen |= (1 << cc->ctors[c].tag);
        }
    }
    if (has_wild) return 1;
    return seen == ((1 << nctor) - 1);
}

static Ty *infer(Cc *cc, TEnv *env, Expr *e) {
    Ty *t, *ta, *tb;
    int gen, i, c;
    if (!e) { cc_fail(cc, "I expected an expression"); return NULL; }
    switch (e->kind) {
    case E_INT: e->ty = ty_int(cc); return e->ty;
    case E_BOOL: e->ty = ty_bool(cc); return e->ty;
    case E_VAR:
        if (tenv_find(env, e->name, &t, &gen) == 0) {
            e->ty = gen ? inst(cc, t) : t;
            return e->ty;
        }
        i = find_fn_src(cc, e->name);
        if (i >= 0 && cc->fns[i].ty) {
            e->ty = inst(cc, cc->fns[i].ty);
            return e->ty;
        }
        cc_fail(cc, "I do not know %s", e->name);
        return NULL;
    case E_CON:
        c = ctor_lookup(cc, e->name);
        if (c < 0) { cc_fail(cc, "I do not know constructor %s", e->name); return NULL; }
        e->ctor = c;
        if (cc->ctors[c].has_payload) {
            Ty *pay = cc->ctors[c].payload ? cc->ctors[c].payload : ty_int(cc);
            e->ty = ty_fun(cc, pay, ty_adt(cc, cc->ctors[c].adt));
        } else {
            e->ty = ty_adt(cc, cc->ctors[c].adt);
        }
        return e->ty;
    case E_TUPLE:
        ta = infer(cc, env, e->x); if (!ta) return NULL;
        tb = infer(cc, env, e->y); if (!tb) return NULL;
        e->ty = ty_prod(cc, ta, tb);
        return e->ty;
    case E_APP:
        ta = infer(cc, env, e->x); if (!ta) return NULL;
        tb = infer(cc, env, e->y); if (!tb) return NULL;
        e->ty = ty_var(cc);
        if (unify(cc, ta, ty_fun(cc, tb, e->ty)) < 0) return NULL;
        return prune(e->ty);
    case E_BIN: {
        const char *op = "I64_ADD";
        (void)op;
        ta = infer(cc, env, e->x); if (!ta) return NULL;
        tb = infer(cc, env, e->y); if (!tb) return NULL;
        if (e->bin == TK_EQ || e->bin == TK_LT || e->bin == TK_GT || e->bin == TK_NE) {
            if (unify(cc, ta, ty_int(cc)) < 0 || unify(cc, tb, ty_int(cc)) < 0) return NULL;
            e->ty = ty_bool(cc);
        } else {
            if (unify(cc, ta, ty_int(cc)) < 0 || unify(cc, tb, ty_int(cc)) < 0) return NULL;
            e->ty = ty_int(cc);
        }
        return e->ty;
    }
    case E_IF:
        ta = infer(cc, env, e->x); if (!ta) return NULL;
        if (unify(cc, ta, ty_bool(cc)) < 0) return NULL;
        ta = infer(cc, env, e->y); if (!ta) return NULL;
        tb = infer(cc, env, e->z); if (!tb) return NULL;
        if (unify(cc, ta, tb) < 0) return NULL;
        e->ty = ta;
        return e->ty;
    case E_LET: {
        TEnv inner;
        memset(&inner, 0, sizeof inner);
        inner.parent = env;
        ta = infer(cc, env, e->x); if (!ta) return NULL;
        if (tenv_bind(&inner, e->letname, ta, 1) < 0) {
            cc_fail(cc, "I refuse too many bindings");
            return NULL;
        }
        e->ty = infer(cc, &inner, e->y);
        return e->ty;
    }
    case E_FN: {
        TEnv inner;
        memset(&inner, 0, sizeof inner);
        inner.parent = env;
        ta = ty_var(cc);
        if (tenv_bind(&inner, e->fnparam, ta, 0) < 0) {
            cc_fail(cc, "I refuse too many bindings");
            return NULL;
        }
        tb = infer(cc, &inner, e->fnbody); if (!tb) return NULL;
        e->ty = ty_fun(cc, ta, tb);
        if (e->fn_idx >= 0) {
            cc->fns[e->fn_idx].ty = e->ty;
            cc->fns[e->fn_idx].result_tag = ty_tag(tb);
        }
        return e->ty;
    }
    case E_CASE: {
        TEnv inner;
        Ty *res = NULL;
        ta = infer(cc, env, e->x); if (!ta) return NULL;
        for (i = 0; i < e->narm; i++) {
            Pat *p = e->arms[i].pat;
            int c;
            if (p->kind == P_TUPLE) {
                if (unify(cc, ta, ty_prod(cc, ty_var(cc), ty_var(cc))) < 0) return NULL;
            } else if (p->kind == P_CON) {
                c = ctor_lookup(cc, p->name);
                if (c < 0) { cc_fail(cc, "I do not know constructor %s", p->name); return NULL; }
                if (unify(cc, ta, ty_adt(cc, cc->ctors[c].adt)) < 0) return NULL;
            }
        }
        ta = prune(ta);
        if (!case_exhaustive(cc, e, ta)) {
            cc_fail(cc, "I refuse a case that is not exhaustive");
            return NULL;
        }
        cc->exhaustive = 1;
        for (i = 0; i < e->narm; i++) {
            memset(&inner, 0, sizeof inner);
            inner.parent = env;
            if (bind_pat(cc, &inner, e->arms[i].pat, ta) < 0) return NULL;
            tb = infer(cc, &inner, e->arms[i].body); if (!tb) return NULL;
            if (!res) res = tb;
            else if (unify(cc, res, tb) < 0) return NULL;
        }
        e->ty = res;
        return e->ty;
    }
    }
    cc_fail(cc, "I cannot infer that");
    return NULL;
}

static uint8_t ty_tag(Ty *t) {
    t = prune(t);
    if (!t) return TAG_INT;
    if (t->kind == TY_BOOL) return TAG_BOOL;
    if (t->kind == TY_FUN) return TAG_FUNCTION;
    if (t->kind == TY_PROD || t->kind == TY_ADT) return TAG_TUPLE;
    return TAG_INT;
}

static void type_text_append(char *out, size_t n, const char *text) {
    size_t used, available, length;
    if (!out || !text || n == 0) return;
    used = strlen(out);
    if (used >= n - 1) return;
    available = n - used - 1;
    length = strlen(text);
    if (length > available) length = available;
    memcpy(out + used, text, length);
    out[used + length] = '\0';
}

static void print_ty(Cc *cc, Ty *t, char *out, size_t n, int *map, int *nm) {
    char a[64], b[64];
    int i;
    t = prune(t);
    if (!t || n < 2) { if (n) out[0] = '\0'; return; }
    switch (t->kind) {
    case TY_INT: snprintf(out, n, "int"); return;
    case TY_BOOL: snprintf(out, n, "bool"); return;
    case TY_VAR:
        for (i = 0; i < *nm; i++) if (map[i] == t->id) {
            snprintf(out, n, "%c", (char)('a' + i));
            return;
        }
        map[*nm] = t->id;
        snprintf(out, n, "%c", (char)('a' + *nm));
        (*nm)++;
        return;
    case TY_FUN:
        print_ty(cc, t->a, a, sizeof a, map, nm);
        print_ty(cc, t->b, b, sizeof b, map, nm);
        out[0] = '\0';
        if (prune(t->a) && prune(t->a)->kind == TY_FUN) type_text_append(out, n, "(");
        type_text_append(out, n, a);
        if (prune(t->a) && prune(t->a)->kind == TY_FUN) type_text_append(out, n, ")");
        type_text_append(out, n, " -> ");
        type_text_append(out, n, b);
        return;
    case TY_PROD:
        print_ty(cc, t->a, a, sizeof a, map, nm);
        print_ty(cc, t->b, b, sizeof b, map, nm);
        out[0] = '\0';
        type_text_append(out, n, a);
        type_text_append(out, n, " * ");
        type_text_append(out, n, b);
        return;
    case TY_ADT:
        if (cc && t->adt >= 0 && t->adt < cc->nadt)
            snprintf(out, n, "%s", cc->adts[t->adt].name);
        else
            snprintf(out, n, "t%d", t->adt);
        return;
    }
}

static int typecheck_program(Cc *cc) {
    TEnv env;
    int i;
    memset(&env, 0, sizeof env);
    for (i = 0; i < cc->ndecl; i++) {
        if (cc->decls[i].kind != D_VALFUN) continue;
        {
            ValFun *vf = &cc->decls[i].vf;
            Expr *body = wrap_fun_body(cc, vf);
            int idx = -1;
            Ty *t;
            if (!body) return cc_fail(cc, "I could not build that function");
            vf->body = body;
            if (register_one_fn(cc, vf->name, vf->params[0], body, &idx) < 0) return -1;
            if (walk_expr_fns(cc, body) < 0) return -1;
            t = ty_var(cc);
            cc->fns[idx].ty = t;
            if (tenv_bind(&env, vf->name, t, 0) < 0)
                return cc_fail(cc, "I refuse too many bindings");
            {
                TEnv inner;
                Ty *res, *paramty, *want;
                memset(&inner, 0, sizeof inner);
                inner.parent = &env;
                paramty = ty_var(cc);
                if (tenv_bind(&inner, vf->params[0], paramty, 0) < 0)
                    return cc_fail(cc, "I refuse too many bindings");
                res = infer(cc, &inner, body);
                if (!res) return -1;
                want = ty_fun(cc, paramty, res);
                if (unify(cc, t, want) < 0) return -1;
                cc->fns[idx].ty = prune(t);
                cc->fns[idx].result_tag = ty_tag(res);
                env.gen[env.n - 1] = 1;
            }
            {
                int map[32], nm = 0;
                snprintf(cc->typenames[cc->ntypes], ML_NAME, "%s", vf->name);
                print_ty(cc, cc->fns[idx].ty, cc->typebuf[cc->ntypes], ML_NAME, map, &nm);
                cc->ntypes++;
            }
        }
    }
    if (cc->result) {
        if (walk_expr_fns(cc, cc->result) < 0) return -1;
        if (!infer(cc, &env, cc->result)) return -1;
    }
    for (i = 0; i < cc->nspec; i++) {
        Ty *have = NULL;
        int gen = 0, m[32], nm = 0;
        char got[ML_NAME], want[ML_NAME];
        if (tenv_find(&env, cc->specs[i].name, &have, &gen) < 0)
            return cc_fail(cc, "signature names %s but I have no such value", cc->specs[i].name);
        if (unify(cc, inst(cc, have), cc->specs[i].ty) < 0) {
            snprintf(cc->err, sizeof cc->err, "I refuse a signature mismatch for %s",
                     cc->specs[i].name);
            return -1;
        }
        (void)got; (void)want; (void)m; (void)nm;
    }
    return 0;
}

#ifdef __GNUC__
__attribute__((format(printf, 3, 4)))
#endif
static int emit_line(Cc *cc, int indent, const char *fmt, ...) {
    va_list ap;
    int need;
    Buf *b;
    if (cc->cur < 0 || cc->cur >= cc->nfn) return cc_fail(cc, "internal emit");
    b = &cc->fns[cc->cur].code;
    va_start(ap, fmt);
    need = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (need < 0) return -1;
    if (buf_grow(b, b->n + (size_t)need + 4) < 0)
        return cc_fail(cc, "I ran out of memory while emitting");
    if (indent) {
        b->p[b->n++] = ' ';
        b->p[b->n++] = ' ';
        b->p[b->n] = '\0';
    }
    va_start(ap, fmt);
    vsnprintf(b->p + b->n, b->cap - b->n, fmt, ap);
    va_end(ap);
    b->n += (size_t)need;
    if (buf_printf(b, "\n") < 0) return cc_fail(cc, "I ran out of memory while emitting");
    return 0;
}

static int finish_tail(Cc *cc, int tail) {
    if (tail) {
        if (emit_line(cc, 1, "RET") < 0) return -1;
        cc->terminated = 1;
    }
    return 0;
}

static int alloc_local(Cc *cc) {
    MlFn *fn = &cc->fns[cc->cur];
    int i = fn->max_local;
    fn->max_local++;
    return i;
}

static int compile_expr(Cc *cc, Env *env, Expr *e, int tail);

static int compile_ref(Cc *cc, Env *env, const char *name, int tail) {
    int i = env_find(env, name);
    int g;
    if (i >= 0) {
        if (env->kind[i] == 0) {
            if (emit_line(cc, 1, "LOAD_LOCAL %d", env->index[i]) < 0) return -1;
        } else {
            if (emit_line(cc, 1, "LOAD_UPVALUE 0 %d", env->index[i]) < 0) return -1;
        }
        return finish_tail(cc, tail);
    }
    g = find_fn_src(cc, name);
    if (g >= 0) {
        if (cc->fns[g].nup > 0) {
            int u;
            for (u = 0; u < cc->fns[g].nup; u++) {
                if (compile_ref(cc, env, cc->fns[g].upnames[u], 0) < 0) return -1;
            }
            if (emit_line(cc, 1, "CLOSURE_NEW %s %d", cc->fns[g].asm_name, cc->fns[g].nup) < 0)
                return -1;
        } else if (emit_line(cc, 1, "FUNCREF %s", cc->fns[g].asm_name) < 0) return -1;
        return finish_tail(cc, tail);
    }
    return cc_fail(cc, "I do not know %s", name);
}

static int emit_upvalues(Cc *cc, Env *env, MlFn *fn) {
    int u;
    for (u = 0; u < fn->nup; u++) {
        if (compile_ref(cc, env, fn->upnames[u], 0) < 0) return -1;
    }
    return 0;
}

static int compile_expr(Cc *cc, Env *env, Expr *e, int tail) {
    int i;
    cc->terminated = 0;
    if (!e) return cc_fail(cc, "I expected an expression");
    switch (e->kind) {
    case E_INT:
        if (emit_line(cc, 1, "PUSH_I64 %lld", (long long)e->i) < 0) return -1;
        return finish_tail(cc, tail);
    case E_BOOL:
        if (emit_line(cc, 1, "PUSH_BOOL %u", e->b ? 1u : 0u) < 0) return -1;
        return finish_tail(cc, tail);
    case E_VAR:
        return compile_ref(cc, env, e->name, tail);
    case E_CON: {
        int c = e->ctor >= 0 ? e->ctor : ctor_lookup(cc, e->name);
        if (c < 0) return cc_fail(cc, "I do not know constructor %s", e->name);
        if (cc->ctors[c].has_payload)
            return cc_fail(cc, "%s needs a payload", e->name);
        if (emit_line(cc, 1, "PUSH_I64 %d", cc->ctors[c].tag) < 0) return -1;
        if (emit_line(cc, 1, "PUSH_VOID") < 0) return -1;
        if (emit_line(cc, 1, "TUPLE_NEW 2") < 0) return -1;
        return finish_tail(cc, tail);
    }
    case E_TUPLE:
        if (compile_expr(cc, env, e->x, 0) < 0) return -1;
        if (compile_expr(cc, env, e->y, 0) < 0) return -1;
        if (emit_line(cc, 1, "TUPLE_NEW 2") < 0) return -1;
        return finish_tail(cc, tail);
    case E_BIN: {
        const char *ins = "I64_ADD";
        if (e->bin == TK_MINUS) ins = "I64_SUB";
        else if (e->bin == TK_STAR) ins = "I64_MUL";
        else if (e->bin == TK_SLASH) ins = "I64_DIV_S";
        else if (e->bin == TK_EQ) ins = "I64_EQ";
        else if (e->bin == TK_LT) ins = "I64_LT_S";
        else if (e->bin == TK_GT) ins = "I64_GT_S";
        else if (e->bin == TK_NE) ins = "I64_NE";
        if (compile_expr(cc, env, e->x, 0) < 0) return -1;
        if (compile_expr(cc, env, e->y, 0) < 0) return -1;
        if (emit_line(cc, 1, "%s", ins) < 0) return -1;
        return finish_tail(cc, tail);
    }
    case E_IF: {
        int l_else = cc->label++, l_end = cc->label++;
        if (compile_expr(cc, env, e->x, 0) < 0) return -1;
        if (emit_line(cc, 1, "JMP_FALSE lf%u", (unsigned)l_else) < 0) return -1;
        if (compile_expr(cc, env, e->y, tail) < 0) return -1;
        if (!cc->terminated && emit_line(cc, 1, "JMP le%u", (unsigned)l_end) < 0) return -1;
        if (emit_line(cc, 0, "lf%u:", (unsigned)l_else) < 0) return -1;
        if (compile_expr(cc, env, e->z, tail) < 0) return -1;
        if (emit_line(cc, 0, "le%u:", (unsigned)l_end) < 0) return -1;
        return 0;
    }
    case E_LET: {
        int loc = alloc_local(cc);
        Env inner = *env;
        if (compile_expr(cc, env, e->x, 0) < 0) return -1;
        if (emit_line(cc, 1, "STORE_LOCAL %d", loc) < 0) return -1;
        if (inner.n >= ML_MAX_BIND) return cc_fail(cc, "I refuse too many locals");
        snprintf(inner.names[inner.n], ML_NAME, "%s", e->letname);
        inner.kind[inner.n] = 0;
        inner.index[inner.n] = loc;
        inner.n++;
        return compile_expr(cc, &inner, e->y, tail);
    }
    case E_FN: {
        MlFn *fn;
        if (e->fn_idx < 0) return cc_fail(cc, "internal lambda");
        fn = &cc->fns[e->fn_idx];
        if (emit_upvalues(cc, env, fn) < 0) return -1;
        if (emit_line(cc, 1, "CLOSURE_NEW %s %d", fn->asm_name, fn->nup) < 0) return -1;
        return finish_tail(cc, tail);
    }
    case E_APP:
        if (e->x && e->x->kind == E_CON) {
            int c = e->x->ctor >= 0 ? e->x->ctor : ctor_lookup(cc, e->x->name);
            if (c < 0) return cc_fail(cc, "I do not know constructor %s", e->x->name);
            if (emit_line(cc, 1, "PUSH_I64 %d", cc->ctors[c].tag) < 0) return -1;
            if (compile_expr(cc, env, e->y, 0) < 0) return -1;
            if (emit_line(cc, 1, "TUPLE_NEW 2") < 0) return -1;
            return finish_tail(cc, tail);
        }
        if (e->x && e->x->kind == E_VAR) {
            int g = find_fn_src(cc, e->x->name);
            if (g >= 0 && env_find(env, e->x->name) < 0) {
                if (compile_expr(cc, env, e->y, 0) < 0) return -1;
                if (tail && strcmp(cc->fns[cc->cur].src_name, e->x->name) == 0
                    && cc->fns[g].arity == 1) {
                    if (emit_line(cc, 1, "TAIL_CALL %s", cc->fns[g].asm_name) < 0) return -1;
                    cc->terminated = 1;
                    return 0;
                }
                if (emit_line(cc, 1, "CALL %s", cc->fns[g].asm_name) < 0) return -1;
                return finish_tail(cc, tail);
            }
        }
        if (compile_expr(cc, env, e->y, 0) < 0) return -1;
        if (compile_expr(cc, env, e->x, 0) < 0) return -1;
        if (emit_line(cc, 1, "CALL_INDIRECT %d 1", 1) < 0) return -1;
        return finish_tail(cc, tail);
    case E_CASE: {
        int scrut = alloc_local(cc);
        int l_end = cc->label++;
        int l_next[ML_MAX_ARM];
        if (compile_expr(cc, env, e->x, 0) < 0) return -1;
        if (emit_line(cc, 1, "STORE_LOCAL %d", scrut) < 0) return -1;
        for (i = 0; i < e->narm; i++) l_next[i] = cc->label++;
        for (i = 0; i < e->narm; i++) {
            Pat *p = e->arms[i].pat;
            Env inner = *env;
            int fail = (i + 1 < e->narm) ? l_next[i] : l_next[e->narm - 1];
            if (i > 0 && emit_line(cc, 0, "lf%u:", (unsigned)l_next[i - 1]) < 0) return -1;
            if (p->kind == P_TUPLE) {
                int la = alloc_local(cc), lb = alloc_local(cc);
                if (emit_line(cc, 1, "LOAD_LOCAL %d", scrut) < 0) return -1;
                if (emit_line(cc, 1, "TUPLE_GET 0") < 0) return -1;
                if (emit_line(cc, 1, "STORE_LOCAL %d", la) < 0) return -1;
                if (emit_line(cc, 1, "LOAD_LOCAL %d", scrut) < 0) return -1;
                if (emit_line(cc, 1, "TUPLE_GET 1") < 0) return -1;
                if (emit_line(cc, 1, "STORE_LOCAL %d", lb) < 0) return -1;
                if (p->x && p->x->kind == P_VAR) {
                    snprintf(inner.names[inner.n], ML_NAME, "%s", p->x->name);
                    inner.kind[inner.n] = 0; inner.index[inner.n] = la; inner.n++;
                }
                if (p->y && p->y->kind == P_VAR) {
                    snprintf(inner.names[inner.n], ML_NAME, "%s", p->y->name);
                    inner.kind[inner.n] = 0; inner.index[inner.n] = lb; inner.n++;
                }
                if (compile_expr(cc, &inner, e->arms[i].body, tail) < 0) return -1;
                if (!cc->terminated && emit_line(cc, 1, "JMP le%u", (unsigned)l_end) < 0)
                    return -1;
                (void)fail;
            } else if (p->kind == P_CON) {
                int c = p->ctor >= 0 ? p->ctor : ctor_lookup(cc, p->name);
                if (emit_line(cc, 1, "LOAD_LOCAL %d", scrut) < 0) return -1;
                if (emit_line(cc, 1, "TUPLE_GET 0") < 0) return -1;
                if (emit_line(cc, 1, "PUSH_I64 %d", cc->ctors[c].tag) < 0) return -1;
                if (emit_line(cc, 1, "I64_EQ") < 0) return -1;
                if (emit_line(cc, 1, "JMP_FALSE lf%u", (unsigned)l_next[i]) < 0) return -1;
                if (p->x && p->x->kind == P_VAR) {
                    int lp = alloc_local(cc);
                    if (emit_line(cc, 1, "LOAD_LOCAL %d", scrut) < 0) return -1;
                    if (emit_line(cc, 1, "TUPLE_GET 1") < 0) return -1;
                    if (emit_line(cc, 1, "STORE_LOCAL %d", lp) < 0) return -1;
                    snprintf(inner.names[inner.n], ML_NAME, "%s", p->x->name);
                    inner.kind[inner.n] = 0; inner.index[inner.n] = lp; inner.n++;
                }
                if (compile_expr(cc, &inner, e->arms[i].body, tail) < 0) return -1;
                if (!cc->terminated && emit_line(cc, 1, "JMP le%u", (unsigned)l_end) < 0)
                    return -1;
            } else if (p->kind == P_VAR) {
                snprintf(inner.names[inner.n], ML_NAME, "%s", p->name);
                inner.kind[inner.n] = 0; inner.index[inner.n] = scrut; inner.n++;
                if (compile_expr(cc, &inner, e->arms[i].body, tail) < 0) return -1;
                if (!cc->terminated && emit_line(cc, 1, "JMP le%u", (unsigned)l_end) < 0)
                    return -1;
            } else {
                if (compile_expr(cc, &inner, e->arms[i].body, tail) < 0) return -1;
                if (!cc->terminated && emit_line(cc, 1, "JMP le%u", (unsigned)l_end) < 0)
                    return -1;
            }
        }
        if (emit_line(cc, 0, "lf%u:", (unsigned)l_next[e->narm - 1]) < 0) return -1;
        if (emit_line(cc, 1, "PUSH_I64 0") < 0) return -1;
        if (tail && emit_line(cc, 1, "RET") < 0) return -1;
        if (emit_line(cc, 0, "le%u:", (unsigned)l_end) < 0) return -1;
        if (tail) cc->terminated = 1;
        return 0;
    }
    }
    return cc_fail(cc, "I cannot compile that");
}

static int compile_function(Cc *cc, int idx) {
    Env env;
    MlFn *fn = &cc->fns[idx];
    int i;
    cc->cur = idx;
    cc->terminated = 0;
    memset(&env, 0, sizeof env);
    fn->max_local = fn->arity;
    for (i = 0; i < fn->arity; i++) {
        snprintf(env.names[env.n], ML_NAME, "%s", fn->params[i]);
        env.kind[env.n] = 0;
        env.index[env.n] = i;
        env.n++;
    }
    for (i = 0; i < fn->nup; i++) {
        snprintf(env.names[env.n], ML_NAME, "%s", fn->upnames[i]);
        env.kind[env.n] = 1;
        env.index[env.n] = i;
        env.n++;
    }
    return compile_expr(cc, &env, fn->body, 1);
}

static const char *result_tag_name(uint8_t tag) {
    const char *n = isa_tag_name(tag);
    return n ? n : "int";
}

static int compile_main(Cc *cc) {
    MlFn *fn;
    Env env;
    if (cc->nfn >= ML_MAX_FN) return cc_fail(cc, "I refuse more than %d functions", ML_MAX_FN);
    fn = &cc->fns[cc->nfn];
    memset(fn, 0, sizeof *fn);
    snprintf(fn->asm_name, sizeof fn->asm_name, "_ml_main");
    fn->result_tag = cc->result && cc->result->ty ? ty_tag(cc->result->ty) : TAG_INT;
    cc->cur = cc->nfn;
    cc->nfn++;
    memset(&env, 0, sizeof env);
    fn->max_local = 0;
    if (!cc->result) {
        if (emit_line(cc, 1, "PUSH_I64 0") < 0) return -1;
        return emit_line(cc, 1, "RET");
    }
    return compile_expr(cc, &env, cc->result, 1);
}

static int build_asm(Cc *cc, Buf *out) {
    int i;
    if (buf_printf(out, ".flag has_main\n.entry _ml_main\n") < 0) return -1;
    for (i = 0; i < cc->ntypes; i++) {
        if (buf_printf(out, ".string \"%s : %s\"\n",
                       cc->typenames[i], cc->typebuf[i]) < 0)
            return -1;
    }
    for (i = 0; i < cc->nfn; i++) {
        MlFn *fn = &cc->fns[i];
        uint32_t locals = (uint32_t)(fn->max_local > fn->arity ? fn->max_local : fn->arity);
        if (locals < 2) locals = 2;
        uint8_t tag = fn->result_tag == 0xFF ? TAG_INT : fn->result_tag;
        if (tag == TAG_VOID) tag = TAG_INT;
        if (buf_printf(out, ".function %s %d %u %d %s 1\n",
                       fn->asm_name, fn->arity, locals, fn->nup,
                       result_tag_name(tag)) < 0)
            return -1;
        if (fn->code.p && buf_printf(out, "%s", fn->code.p) < 0) return -1;
        if (buf_printf(out, ".end\n") < 0) return -1;
    }
    return 0;
}

static int prepare(Cc *cc, const char *src) {
    int i;
    memset(cc, 0, sizeof *cc);
    cc->src = src;
    cc->cur = -1;
    if (parse_program(cc) < 0) return -1;
    if (typecheck_program(cc) < 0) return -1;
    for (i = 0; i < cc->nfn; i++) {
        if (compile_function(cc, i) < 0) return -1;
    }
    return compile_main(cc);
}

static void attach_debug(NvmModule *mod) {
    if (!mod) return;
    mod->header.flags |= NVM_FLAG_DEBUG_INFO;
    nvm_add_debug_entry(mod, 0, 1, 1);
}

NvmModule *nl_ml_compile(const char *src, const char *path,
                         char *err, size_t errlen) {
    Cc cc;
    Buf asmbuf;
    AsmResult ar;
    NvmModule *mod;
    (void)path;
    memset(&asmbuf, 0, sizeof asmbuf);
    if (!src) {
        if (err && errlen) snprintf(err, errlen, "source is null");
        return NULL;
    }
    if (prepare(&cc, src) < 0) {
        if (err && errlen) snprintf(err, errlen, "%s", cc.err[0] ? cc.err : "compile failed");
        cc_free(&cc);
        return NULL;
    }
    if (build_asm(&cc, &asmbuf) < 0) {
        if (err && errlen) snprintf(err, errlen, "I ran out of memory while assembling");
        free(asmbuf.p);
        cc_free(&cc);
        return NULL;
    }
    if (getenv("NL_ML_TRACE")) fprintf(stderr, "%s", asmbuf.p ? asmbuf.p : "");
    memset(&ar, 0, sizeof ar);
    mod = asm_assemble(asmbuf.p, &ar);
    if (!mod) {
        if (err && errlen)
            snprintf(err, errlen, "assembler: %s (line %u)", ar.message, ar.line);
        free(asmbuf.p);
        cc_free(&cc);
        return NULL;
    }
    attach_debug(mod);
    free(asmbuf.p);
    cc_free(&cc);
    return mod;
}

NlFrontendResult nl_ml_accept(const NvmModule *mod, const char *path) {
    NlFrontendFacts f;
    memset(&f, 0, sizeof f);
    f.language = NL_FE_ML;
    f.source_path = (path && path[0]) ? path : "<ml>";
    f.purity = 1;
    f.exhaustiveness = 1;
    f.affine_use = -1;
    f.diagnostics_shared = 1;
    return nl_frontend_accept(mod, &f);
}

VmResult nl_ml_execute(const NvmModule *mod, NanoValue *out,
                       char *err, size_t errlen) {
    VmState vm;
    VmResult r;
    NanoValue v;
    if (!mod) {
        if (err && errlen) snprintf(err, errlen, "module is null");
        return VM_ERR_TYPE_ERROR;
    }
    vm_init(&vm, mod);
    r = vm_execute(&vm);
    if (r != VM_OK) {
        if (err && errlen) snprintf(err, errlen, "%s", vm.error_msg[0] ? vm.error_msg : vm_error_string(r));
        vm_destroy(&vm);
        return r;
    }
    v = vm_get_result(&vm);
    vm_retain(&vm.heap, v);
    if (out) *out = v;
    vm_destroy(&vm);
    return VM_OK;
}

int nl_ml_eval_i64(const char *src, int64_t *out, char *err, size_t errlen) {
    NvmModule *mod;
    NlFrontendResult acc;
    NanoValue v;
    VmResult r;
    mod = nl_ml_compile(src, "<eval>", err, errlen);
    if (!mod) return 0;
    acc = nl_ml_accept(mod, "<eval>");
    if (!acc.ok) {
        if (err && errlen) snprintf(err, errlen, "%s", acc.error);
        nvm_module_free(mod);
        return 0;
    }
    memset(&v, 0, sizeof v);
    r = nl_ml_execute(mod, &v, err, errlen);
    nvm_module_free(mod);
    if (r != VM_OK) return 0;
    if (v.tag != TAG_INT) {
        if (err && errlen)
            snprintf(err, errlen, "result is %s, not int", isa_tag_name(v.tag));
        return 0;
    }
    if (out) *out = v.as.i64;
    return 1;
}

int nl_ml_type_of(const char *src, const char *name, char *out, size_t outlen,
                  char *err, size_t errlen) {
    Cc cc;
    int i;
    if (!src || !name || !out) {
        if (err && errlen) snprintf(err, errlen, "type_of needs source and a name");
        return 0;
    }
    if (prepare(&cc, src) < 0) {
        if (err && errlen) snprintf(err, errlen, "%s", cc.err[0] ? cc.err : "compile failed");
        cc_free(&cc);
        return 0;
    }
    for (i = 0; i < cc.ntypes; i++) {
        if (strcmp(cc.typenames[i], name) == 0) {
            snprintf(out, outlen, "%s", cc.typebuf[i]);
            cc_free(&cc);
            return 1;
        }
    }
    if (err && errlen) snprintf(err, errlen, "I have no type for %s", name);
    cc_free(&cc);
    return 0;
}

uint32_t nl_ml_fun_index(const char *src, const char *name, char *err,
                         size_t errlen) {
    NvmModule *mod;
    uint32_t i;
    char want[ML_NAME];
    mod = nl_ml_compile(src, "<index>", err, errlen);
    if (!mod) return (uint32_t)-1;
    snprintf(want, sizeof want, "ml_%s", name);
    for (i = 0; i < mod->function_count; i++) {
        const char *fn = nvm_get_string(mod, mod->functions[i].name_idx);
        if (fn && strcmp(fn, want) == 0) {
            nvm_module_free(mod);
            return i;
        }
    }
    if (err && errlen) snprintf(err, errlen, "I have no function %s", name);
    nvm_module_free(mod);
    return (uint32_t)-1;
}
