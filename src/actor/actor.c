/*
 * Nano Actor — bounded laboratory frontend. I emit verified NanoISA
 * handlers and run them in isolated NanoVM contexts. See docs/ACTOR.md.
 */

#include "actor.h"

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

#define AC_NAME 64
#define AC_MAX 24
#define AC_ARM 12
#define AC_ERR NL_ACTOR_ERR_SIZE
#define AC_BOX 32
#define AC_LOCAL 16

typedef enum {
    TK_EOF, TK_INT, TK_ID,
    TK_LPAREN, TK_RPAREN, TK_LBRACE, TK_RBRACE, TK_BAR, TK_EQ, TK_DARROW,
    TK_PLUS, TK_MINUS, TK_COLON,
    TK_MESSAGE, TK_ACTOR, TK_STATE, TK_RECEIVE, TK_REPLY, TK_BECOME,
    TK_CRASH, TK_SPAWN, TK_SEND, TK_RECV, TK_AFTER, TK_CANCEL,
    TK_MONITOR, TK_LINK, TK_SUPERVISE, TK_ONE_FOR_ONE, TK_CHILD,
    TK_REPLACE, TK_MAIN, TK_OF, TK_REMOTE, TK_CAP
} TkKind;

typedef struct {
    TkKind kind;
    int64_t i;
    char name[AC_NAME];
    int after_nl;
} Tok;

typedef enum { E_INT, E_VAR, E_STATE, E_BIN, E_CON } EKind;
typedef enum { P_VAR, P_CON, P_WILD } PKind;
typedef enum { ST_SPAWN, ST_SEND, ST_RECV, ST_CANCEL, ST_MONITOR,
               ST_LINK, ST_REPLACE } SKind;
typedef enum { B_REPLY, B_BECOME, B_CRASH } BKind;

typedef struct Expr Expr;
struct Expr {
    EKind kind;
    int64_t i;
    char name[AC_NAME];
    int bin;
    int ctor;
    Expr *x, *y;
};

typedef struct {
    PKind kind;
    char name[AC_NAME];
    int ctor;
    char bind[AC_NAME];
} Pat;

typedef struct {
    Pat pat;
    BKind bk;
    Expr *e;
} Arm;

typedef struct {
    char name[AC_NAME];
    int64_t init;
    Arm arms[AC_ARM];
    int narm;
    char asm_name[AC_NAME];
    int fn_idx;
    uint8_t handled[64];
} ActorDef;

typedef struct {
    char child[AC_NAME];
    char type[AC_NAME];
} ChildSpec;

typedef struct {
    SKind kind;
    char name[AC_NAME];
    char actor[AC_NAME];
    Expr *pid;
    int ctor;
    Expr *payload;
    Arm arms[AC_ARM];
    int narm;
    int has_after;
    Expr *result;
} Stmt;

typedef struct {
    char *p;
    size_t n, cap;
} Buf;

typedef struct {
    const char *src;
    const char *lx;
    Tok tok;
    int saw_nl;
    char err[AC_ERR];
    void **heap;
    uint32_t nheap, capheap;
    struct { char name[AC_NAME]; int tag, has_payload; } ctors[64];
    int nctor;
    ActorDef actors[AC_MAX];
    int nactor;
    ChildSpec kids[AC_MAX];
    int nkid;
    Stmt stmts[64];
    int nstmt;
    Expr *result;
    Buf code[AC_MAX];
    int nfn;
    int label;
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
    if (e) { e->kind = k; e->ctor = -1; }
    return e;
}

static void skip_ws(Cc *cc) {
    cc->saw_nl = 0;
    for (;;) {
        while (*cc->lx && isspace((unsigned char)*cc->lx)) {
            if (*cc->lx == '\n' || *cc->lx == '\r') cc->saw_nl = 1;
            cc->lx++;
        }
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
    if (isalpha((unsigned char)*p) || *p == '_') {
        int n = 0;
        while ((isalnum((unsigned char)*p) || *p == '_') && n < AC_NAME - 1)
            cc->tok.name[n++] = *p++;
        cc->lx = p;
        if (kw_eq(cc->tok.name, "message")) cc->tok.kind = TK_MESSAGE;
        else if (kw_eq(cc->tok.name, "actor")) cc->tok.kind = TK_ACTOR;
        else if (kw_eq(cc->tok.name, "state")) cc->tok.kind = TK_STATE;
        else if (kw_eq(cc->tok.name, "receive")) cc->tok.kind = TK_RECEIVE;
        else if (kw_eq(cc->tok.name, "reply")) cc->tok.kind = TK_REPLY;
        else if (kw_eq(cc->tok.name, "become")) cc->tok.kind = TK_BECOME;
        else if (kw_eq(cc->tok.name, "crash")) cc->tok.kind = TK_CRASH;
        else if (kw_eq(cc->tok.name, "spawn")) cc->tok.kind = TK_SPAWN;
        else if (kw_eq(cc->tok.name, "send")) cc->tok.kind = TK_SEND;
        else if (kw_eq(cc->tok.name, "recv")) cc->tok.kind = TK_RECV;
        else if (kw_eq(cc->tok.name, "after")) cc->tok.kind = TK_AFTER;
        else if (kw_eq(cc->tok.name, "cancel")) cc->tok.kind = TK_CANCEL;
        else if (kw_eq(cc->tok.name, "monitor")) cc->tok.kind = TK_MONITOR;
        else if (kw_eq(cc->tok.name, "link")) cc->tok.kind = TK_LINK;
        else if (kw_eq(cc->tok.name, "supervise")) cc->tok.kind = TK_SUPERVISE;
        else if (kw_eq(cc->tok.name, "one_for_one")) cc->tok.kind = TK_ONE_FOR_ONE;
        else if (kw_eq(cc->tok.name, "child")) cc->tok.kind = TK_CHILD;
        else if (kw_eq(cc->tok.name, "replace")) cc->tok.kind = TK_REPLACE;
        else if (kw_eq(cc->tok.name, "main")) cc->tok.kind = TK_MAIN;
        else if (kw_eq(cc->tok.name, "of")) cc->tok.kind = TK_OF;
        else if (kw_eq(cc->tok.name, "remote")) cc->tok.kind = TK_REMOTE;
        else if (kw_eq(cc->tok.name, "cap")) cc->tok.kind = TK_CAP;
        else cc->tok.kind = TK_ID;
        return;
    }
    if (p[0] == '=' && p[1] == '>') { cc->tok.kind = TK_DARROW; cc->lx = p + 2; return; }
    cc->lx = p + 1;
    switch (*p) {
    case '(': cc->tok.kind = TK_LPAREN; break;
    case ')': cc->tok.kind = TK_RPAREN; break;
    case '{': cc->tok.kind = TK_LBRACE; break;
    case '}': cc->tok.kind = TK_RBRACE; break;
    case '|': cc->tok.kind = TK_BAR; break;
    case '=': cc->tok.kind = TK_EQ; break;
    case '+': cc->tok.kind = TK_PLUS; break;
    case '-': cc->tok.kind = TK_MINUS; break;
    case ':': cc->tok.kind = TK_COLON; break;
    default: cc->tok.kind = TK_EOF;
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

static int is_ctor(const char *n) { return n[0] >= 'A' && n[0] <= 'Z'; }

static int ctor_lookup(Cc *cc, const char *n) {
    int i;
    for (i = 0; i < cc->nctor; i++) {
        if (strcmp(cc->ctors[i].name, n) == 0) return i;
    }
    return -1;
}

static int actor_lookup(Cc *cc, const char *n) {
    int i;
    for (i = 0; i < cc->nactor; i++) {
        if (strcmp(cc->actors[i].name, n) == 0) return i;
    }
    return -1;
}

static int kid_lookup(Cc *cc, const char *n) {
    int i;
    for (i = 0; i < cc->nkid; i++) {
        if (strcmp(cc->kids[i].child, n) == 0) return i;
    }
    return -1;
}

static Expr *parse_expr(Cc *cc);

static Expr *parse_atom(Cc *cc) {
    Expr *e;
    if (have(cc, TK_CAP)) return cc_fail(cc, "I refuse cap:; capabilities do not cross restarts"), NULL;
    if (have(cc, TK_INT)) {
        e = ex_new(cc, E_INT); if (!e) return NULL;
        e->i = cc->tok.i; lex(cc); return e;
    }
    if (have(cc, TK_STATE)) {
        e = ex_new(cc, E_STATE); if (!e) return NULL;
        lex(cc); return e;
    }
    if (have(cc, TK_ID)) {
        if (is_ctor(cc->tok.name)) {
            int c = ctor_lookup(cc, cc->tok.name);
            if (c < 0) { cc_fail(cc, "I do not know message %s", cc->tok.name); return NULL; }
            e = ex_new(cc, E_CON); if (!e) return NULL;
            snprintf(e->name, sizeof e->name, "%s", cc->tok.name);
            e->ctor = c;
            lex(cc);
            if (!cc->tok.after_nl && (have(cc, TK_INT) || have(cc, TK_ID) || have(cc, TK_STATE) || have(cc, TK_LPAREN))) {
                e->x = parse_expr(cc);
                if (!e->x) return NULL;
            }
            return e;
        }
        e = ex_new(cc, E_VAR); if (!e) return NULL;
        snprintf(e->name, sizeof e->name, "%s", cc->tok.name);
        lex(cc);
        return e;
    }
    if (eat(cc, TK_LPAREN)) {
        e = parse_expr(cc); if (!e) return NULL;
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
        Expr *n = ex_new(cc, E_BIN); Expr *right;
        if (!n) return NULL;
        n->bin = (int)cc->tok.kind;
        lex(cc);
        right = parse_atom(cc); if (!right) return NULL;
        n->x = left; n->y = right;
        left = n;
    }
    return left;
}

static int parse_pat(Cc *cc, Pat *p) {
    memset(p, 0, sizeof *p);
    p->ctor = -1;
    if (have(cc, TK_ID) && strcmp(cc->tok.name, "_") == 0) {
        p->kind = P_WILD; lex(cc); return 0;
    }
    if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a pattern");
    if (is_ctor(cc->tok.name)) {
        p->kind = P_CON;
        snprintf(p->name, sizeof p->name, "%s", cc->tok.name);
        p->ctor = ctor_lookup(cc, p->name);
        if (p->ctor < 0) return cc_fail(cc, "I do not know message %s", p->name);
        lex(cc);
        if (have(cc, TK_ID) && !is_ctor(cc->tok.name) && !cc->tok.after_nl) {
            snprintf(p->bind, sizeof p->bind, "%s", cc->tok.name);
            lex(cc);
        }
        return 0;
    }
    p->kind = P_VAR;
    snprintf(p->name, sizeof p->name, "%s", cc->tok.name);
    lex(cc);
    return 0;
}

static int parse_arms(Cc *cc, Arm *arms, int *narm, int bodies) {
    *narm = 0;
    do {
        if (*narm >= AC_ARM) return cc_fail(cc, "I refuse too many receive arms");
        if (!eat(cc, TK_BAR) && *narm > 0) break;
        if (*narm == 0) eat(cc, TK_BAR);
        if (parse_pat(cc, &arms[*narm].pat) < 0) return -1;
        if (!eat(cc, TK_DARROW)) return cc_fail(cc, "I expected =>");
        if (bodies) {
            if (eat(cc, TK_CRASH)) {
                arms[*narm].bk = B_CRASH;
            } else if (eat(cc, TK_REPLY)) {
                arms[*narm].bk = B_REPLY;
                arms[*narm].e = parse_expr(cc);
                if (!arms[*narm].e) return -1;
            } else if (eat(cc, TK_BECOME)) {
                arms[*narm].bk = B_BECOME;
                arms[*narm].e = parse_expr(cc);
                if (!arms[*narm].e) return -1;
            } else {
                return cc_fail(cc, "I expected reply, become, or crash");
            }
        } else {
            arms[*narm].bk = B_REPLY;
            arms[*narm].e = parse_expr(cc);
            if (!arms[*narm].e) return -1;
        }
        (*narm)++;
    } while (have(cc, TK_BAR));
    return *narm > 0 ? 0 : cc_fail(cc, "I expected a receive arm");
}

static int add_ctor(Cc *cc, const char *name, int has_payload) {
    int i = ctor_lookup(cc, name);
    if (i >= 0) return i;
    if (cc->nctor >= 64) return cc_fail(cc, "I refuse too many messages");
    i = cc->nctor++;
    snprintf(cc->ctors[i].name, AC_NAME, "%s", name);
    cc->ctors[i].tag = i;
    cc->ctors[i].has_payload = has_payload;
    return i;
}

static int parse_program(Cc *cc) {
    cc->lx = cc->src;
    lex(cc);
    add_ctor(cc, "Down", 1);
    add_ctor(cc, "Timeout", 0);
    while (!have(cc, TK_EOF)) {
        if (eat(cc, TK_MESSAGE)) {
            if (!have(cc, TK_ID) || !is_ctor(cc->tok.name))
                return cc_fail(cc, "I expected a message name");
            {
                char n[AC_NAME];
                int pay = 0;
                snprintf(n, sizeof n, "%s", cc->tok.name);
                lex(cc);
                if (eat(cc, TK_OF)) {
                    if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a payload type");
                    lex(cc);
                    pay = 1;
                }
                if (add_ctor(cc, n, pay) < 0) return -1;
            }
            continue;
        }
        if (eat(cc, TK_ACTOR)) {
            ActorDef *a;
            size_t actor_name_len;
            if (!have(cc, TK_ID)) return cc_fail(cc, "I expected an actor name");
            if (cc->nactor >= AC_MAX) return cc_fail(cc, "I refuse too many actors");
            actor_name_len = strlen(cc->tok.name);
            if (actor_name_len > AC_NAME - 5)
                return cc_fail(cc, "I refuse an actor name longer than %d bytes", AC_NAME - 5);
            a = &cc->actors[cc->nactor++];
            memset(a, 0, sizeof *a);
            snprintf(a->name, sizeof a->name, "%s", cc->tok.name);
            memcpy(a->asm_name, "act_", 4);
            memcpy(a->asm_name + 4, cc->tok.name, actor_name_len + 1);
            lex(cc);
            if (!eat(cc, TK_LBRACE)) return cc_fail(cc, "I expected '{'");
            if (eat(cc, TK_STATE)) {
                if (!have(cc, TK_INT)) return cc_fail(cc, "I expected an integer state");
                a->init = cc->tok.i;
                lex(cc);
            }
            if (!eat(cc, TK_RECEIVE)) return cc_fail(cc, "I expected receive");
            if (parse_arms(cc, a->arms, &a->narm, 1) < 0) return -1;
            if (!eat(cc, TK_RBRACE)) return cc_fail(cc, "I expected '}'");
            {
                int i;
                for (i = 0; i < a->narm; i++) {
                    if (a->arms[i].pat.kind == P_CON && a->arms[i].pat.ctor >= 0)
                        a->handled[a->arms[i].pat.ctor] = 1;
                    if (a->arms[i].pat.kind == P_VAR || a->arms[i].pat.kind == P_WILD) {
                        int c;
                        for (c = 0; c < 64; c++) a->handled[c] = 1;
                    }
                }
            }
            continue;
        }
        if (eat(cc, TK_SUPERVISE)) {
            if (!eat(cc, TK_ONE_FOR_ONE)) return cc_fail(cc, "I only take one_for_one supervision");
            if (!eat(cc, TK_LBRACE)) return cc_fail(cc, "I expected '{'");
            while (eat(cc, TK_CHILD)) {
                ChildSpec *k;
                if (cc->nkid >= AC_MAX) return cc_fail(cc, "I refuse too many children");
                k = &cc->kids[cc->nkid++];
                if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a child name");
                snprintf(k->child, sizeof k->child, "%s", cc->tok.name);
                lex(cc);
                if (!eat(cc, TK_EQ) || !have(cc, TK_ID))
                    return cc_fail(cc, "I expected = Actor");
                snprintf(k->type, sizeof k->type, "%s", cc->tok.name);
                lex(cc);
            }
            if (!eat(cc, TK_RBRACE)) return cc_fail(cc, "I expected '}'");
            continue;
        }
        if (eat(cc, TK_MAIN)) {
            if (!eat(cc, TK_LBRACE)) return cc_fail(cc, "I expected '{'");
            while (!have(cc, TK_RBRACE) && !have(cc, TK_EOF)) {
                Stmt *s;
                if (have(cc, TK_ID) && !is_ctor(cc->tok.name)) {
                    char nm[AC_NAME];
                    snprintf(nm, sizeof nm, "%s", cc->tok.name);
                    lex(cc);
                    if (eat(cc, TK_EQ)) {
                        if (cc->nstmt >= 64) return cc_fail(cc, "I refuse too many statements");
                        s = &cc->stmts[cc->nstmt++];
                        memset(s, 0, sizeof *s);
                        snprintf(s->name, sizeof s->name, "%s", nm);
                        if (eat(cc, TK_SPAWN)) {
                            s->kind = ST_SPAWN;
                            if (have(cc, TK_REMOTE))
                                return cc_fail(cc, "I refuse remote spawn; Phase 18 process boundaries are not wired");
                            if (!have(cc, TK_ID)) return cc_fail(cc, "I expected an actor name");
                            snprintf(s->actor, sizeof s->actor, "%s", cc->tok.name);
                            lex(cc);
                            continue;
                        }
                        if (have(cc, TK_RECV)) {
                            s->kind = ST_RECV;
                            lex(cc);
                            if (eat(cc, TK_AFTER)) {
                                if (!have(cc, TK_INT) || cc->tok.i != 0)
                                    return cc_fail(cc, "I only take after 0 in this subset");
                                lex(cc);
                                s->has_after = 1;
                            }
                            if (parse_arms(cc, s->arms, &s->narm, 0) < 0) return -1;
                            continue;
                        }
                        return cc_fail(cc, "I expected spawn or recv");
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
                        cc->result = e;
                        break;
                    }
                }
                if (eat(cc, TK_SEND)) {
                    if (cc->nstmt >= 64) return cc_fail(cc, "I refuse too many statements");
                    s = &cc->stmts[cc->nstmt++];
                    memset(s, 0, sizeof *s);
                    s->kind = ST_SEND;
                    if (have(cc, TK_CAP))
                        return cc_fail(cc, "I refuse cap:; capabilities do not cross restarts");
                    s->pid = parse_expr(cc); if (!s->pid) return -1;
                    if (have(cc, TK_CAP) || (have(cc, TK_ID) && strcmp(cc->tok.name, "cap") == 0))
                        return cc_fail(cc, "I refuse cap:; capabilities do not cross restarts");
                    if (eat(cc, TK_LPAREN)) {
                        if (!have(cc, TK_ID) || !is_ctor(cc->tok.name))
                            return cc_fail(cc, "I expected a message");
                        s->ctor = ctor_lookup(cc, cc->tok.name);
                        if (s->ctor < 0) return cc_fail(cc, "I do not know message %s", cc->tok.name);
                        lex(cc);
                        if (!have(cc, TK_RPAREN)) {
                            s->payload = parse_expr(cc);
                            if (!s->payload) return -1;
                        }
                        if (!eat(cc, TK_RPAREN)) return cc_fail(cc, "I expected ')'");
                    } else if (have(cc, TK_ID) && is_ctor(cc->tok.name)) {
                        s->ctor = ctor_lookup(cc, cc->tok.name);
                        if (s->ctor < 0) return cc_fail(cc, "I do not know message %s", cc->tok.name);
                        lex(cc);
                    } else {
                        return cc_fail(cc, "I expected a message after send");
                    }
                    continue;
                }
                if (have(cc, TK_CANCEL) || have(cc, TK_MONITOR) || have(cc, TK_LINK)
                    || have(cc, TK_REPLACE) || have(cc, TK_RECV) || have(cc, TK_INT)
                    || have(cc, TK_ID)) {
                    if (have(cc, TK_CANCEL) || have(cc, TK_MONITOR) || have(cc, TK_LINK)
                        || have(cc, TK_REPLACE)) {
                        if (cc->nstmt >= 64) return cc_fail(cc, "I refuse too many statements");
                        s = &cc->stmts[cc->nstmt++];
                        memset(s, 0, sizeof *s);
                        if (eat(cc, TK_CANCEL)) s->kind = ST_CANCEL;
                        else if (eat(cc, TK_MONITOR)) s->kind = ST_MONITOR;
                        else if (eat(cc, TK_LINK)) s->kind = ST_LINK;
                        else {
                            eat(cc, TK_REPLACE);
                            s->kind = ST_REPLACE;
                        }
                        s->pid = parse_expr(cc); if (!s->pid) return -1;
                        if (s->kind == ST_REPLACE) {
                            if (!have(cc, TK_ID)) return cc_fail(cc, "I expected an actor name");
                            snprintf(s->actor, sizeof s->actor, "%s", cc->tok.name);
                            lex(cc);
                        }
                        continue;
                    }
                    if (have(cc, TK_RECV)) {
                        if (cc->nstmt >= 64) return cc_fail(cc, "I refuse too many statements");
                        s = &cc->stmts[cc->nstmt++];
                        memset(s, 0, sizeof *s);
                        s->kind = ST_RECV;
                        lex(cc);
                        if (eat(cc, TK_AFTER)) {
                            if (!have(cc, TK_INT) || cc->tok.i != 0)
                                return cc_fail(cc, "I only take after 0 in this subset");
                            lex(cc);
                            s->has_after = 1;
                        }
                        if (parse_arms(cc, s->arms, &s->narm, 0) < 0) return -1;
                        continue;
                    }
                    cc->result = parse_expr(cc);
                    if (!cc->result) return -1;
                    break;
                }
                return cc_fail(cc, "I expected a statement");
            }
            if (!eat(cc, TK_RBRACE)) return cc_fail(cc, "I expected '}'");
            continue;
        }
        return cc_fail(cc, "I expected message, actor, supervise, or main");
    }
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

typedef struct {
    char names[AC_LOCAL][AC_NAME];
    int index[AC_LOCAL];
    int n;
} Env;

static int env_find(const Env *e, const char *n) {
    int i;
    for (i = e->n - 1; i >= 0; i--) {
        if (strcmp(e->names[i], n) == 0) return i;
    }
    return -1;
}

static int compile_expr(Cc *cc, Buf *b, Env *env, Expr *e) {
    int i;
    if (!e) return cc_fail(cc, "I expected an expression");
    switch (e->kind) {
    case E_INT:
        return emit_line(b, 1, "PUSH_I64 %lld", (long long)e->i);
    case E_STATE:
        return emit_line(b, 1, "LOAD_LOCAL 0");
    case E_VAR:
        i = env_find(env, e->name);
        if (i < 0) return cc_fail(cc, "I do not know %s", e->name);
        return emit_line(b, 1, "LOAD_LOCAL %d", env->index[i]);
    case E_BIN:
        if (compile_expr(cc, b, env, e->x) < 0) return -1;
        if (compile_expr(cc, b, env, e->y) < 0) return -1;
        return emit_line(b, 1, "%s", e->bin == TK_MINUS ? "I64_SUB" : "I64_ADD");
    case E_CON:
        if (e->x) {
            if (compile_expr(cc, b, env, e->x) < 0) return -1;
        } else if (emit_line(b, 1, "PUSH_I64 0") < 0) return -1;
        return 0;
    }
    return cc_fail(cc, "I cannot compile that");
}

static int compile_body(Cc *cc, Buf *b, Env *env, Arm *a) {
    if (a->bk == B_CRASH) {
        if (emit_line(b, 1, "PUSH_I64 2") < 0) return -1;
        if (emit_line(b, 1, "PUSH_I64 0") < 0) return -1;
        if (emit_line(b, 1, "PUSH_I64 0") < 0) return -1;
        if (emit_line(b, 1, "PUSH_I64 0") < 0) return -1;
        return emit_line(b, 1, "TUPLE_NEW 4");
    }
    if (a->bk == B_BECOME) {
        if (emit_line(b, 1, "PUSH_I64 1") < 0) return -1;
        if (compile_expr(cc, b, env, a->e) < 0) return -1;
        if (emit_line(b, 1, "PUSH_I64 0") < 0) return -1;
        if (emit_line(b, 1, "PUSH_I64 0") < 0) return -1;
        return emit_line(b, 1, "TUPLE_NEW 4");
    }
    /* reply */
    if (!a->e) return cc_fail(cc, "I expected a reply value");
    if (a->e->kind != E_CON)
        return cc_fail(cc, "I expected a message constructor in reply");
    if (emit_line(b, 1, "PUSH_I64 0") < 0) return -1;
    if (emit_line(b, 1, "LOAD_LOCAL 0") < 0) return -1;
    if (emit_line(b, 1, "PUSH_I64 %d", a->e->ctor) < 0) return -1;
    if (compile_expr(cc, b, env, a->e) < 0) return -1;
    return emit_line(b, 1, "TUPLE_NEW 4");
}

static int compile_actor(Cc *cc, ActorDef *a, Buf *b) {
    Env env;
    int i;
    memset(&env, 0, sizeof env);
    snprintf(env.names[0], AC_NAME, "state");
    env.index[0] = 0;
    env.n = 1;
    for (i = 0; i < a->narm; i++) {
        int next = cc->label++;
        Pat *p = &a->arms[i].pat;
        Env inner = env;
        if (p->kind == P_CON) {
            if (emit_line(b, 1, "LOAD_LOCAL 1") < 0) return -1;
            if (emit_line(b, 1, "PUSH_I64 %d", cc->ctors[p->ctor].tag) < 0) return -1;
            if (emit_line(b, 1, "I64_EQ") < 0) return -1;
            if (emit_line(b, 1, "JMP_FALSE lf%u", (unsigned)next) < 0) return -1;
            if (p->bind[0]) {
                snprintf(inner.names[inner.n], AC_NAME, "%s", p->bind);
                inner.index[inner.n] = 2;
                inner.n++;
            }
            if (compile_body(cc, b, &inner, &a->arms[i]) < 0) return -1;
            if (emit_line(b, 1, "RET") < 0) return -1;
            if (emit_line(b, 0, "lf%u:", (unsigned)next) < 0) return -1;
        } else {
            if (p->kind == P_VAR) {
                snprintf(inner.names[inner.n], AC_NAME, "%s", p->name);
                inner.index[inner.n] = 2;
                inner.n++;
            }
            if (compile_body(cc, b, &inner, &a->arms[i]) < 0) return -1;
            if (emit_line(b, 1, "RET") < 0) return -1;
        }
    }
    if (emit_line(b, 1, "PUSH_I64 2") < 0) return -1;
    if (emit_line(b, 1, "PUSH_I64 0") < 0) return -1;
    if (emit_line(b, 1, "PUSH_I64 0") < 0) return -1;
    if (emit_line(b, 1, "PUSH_I64 0") < 0) return -1;
    if (emit_line(b, 1, "TUPLE_NEW 4") < 0) return -1;
    return emit_line(b, 1, "RET");
}

static int compile_all(Cc *cc) {
    int i;
    for (i = 0; i < cc->nactor; i++) {
        if (cc->nfn >= AC_MAX) return cc_fail(cc, "I refuse too many functions");
        if (compile_actor(cc, &cc->actors[i], &cc->code[cc->nfn]) < 0) return -1;
        cc->actors[i].fn_idx = cc->nfn;
        cc->nfn++;
    }
    return 0;
}

static int build_asm(Cc *cc, Buf *out) {
    int i;
    if (buf_printf(out, ".flag has_main\n.entry _act_main\n") < 0) return -1;
    for (i = 0; i < cc->nactor; i++) {
        if (buf_printf(out, ".function %s 3 8 0 tuple 1\n", cc->actors[i].asm_name) < 0)
            return -1;
        if (cc->code[i].p && buf_printf(out, "%s", cc->code[i].p) < 0) return -1;
        if (buf_printf(out, ".end\n") < 0) return -1;
    }
    if (buf_printf(out, ".function _act_main 0 2 0 int 1\n  PUSH_I64 0\n  RET\n.end\n") < 0)
        return -1;
    return 0;
}

typedef struct {
    int from, tag;
    int64_t payload;
} Msg;

typedef struct {
    int used, alive, cancelled, linked_from_main;
    int def, fn_idx, generation, supervised, spec;
    int64_t state;
    Msg box[AC_BOX];
    int bhead, bcount;
    int monitors[8];
    int nmon;
    int links[8];
    int nlink;
    VmState vm;
    int vm_on;
} Act;

typedef struct {
    Act acts[AC_MAX];
    int nact;
    int64_t locals[32];
    char lnames[32][AC_NAME];
    int nl;
    int main_dead;
    Cc *cc;
    NvmModule *mod;
} Sys;

static int sys_fail(Cc *cc, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(cc->err, sizeof cc->err, fmt, ap);
    va_end(ap);
    return -1;
}

static int enqueue(Act *a, int from, int tag, int64_t payload) {
    int i;
    if (!a->alive || a->cancelled) return -1;
    if (a->bcount >= AC_BOX) return -1;
    i = (a->bhead + a->bcount) % AC_BOX;
    a->box[i].from = from;
    a->box[i].tag = tag;
    a->box[i].payload = payload;
    a->bcount++;
    return 0;
}

static int dequeue(Act *a, Msg *m) {
    if (a->bcount <= 0) return -1;
    *m = a->box[a->bhead];
    a->bhead = (a->bhead + 1) % AC_BOX;
    a->bcount--;
    return 0;
}

static int lookup_local(Sys *sy, const char *n, int64_t *out) {
    int i;
    for (i = sy->nl - 1; i >= 0; i--) {
        if (strcmp(sy->lnames[i], n) == 0) { *out = sy->locals[i]; return 0; }
    }
    return -1;
}

static int eval_expr(Cc *cc, Sys *sy, Expr *e, int64_t *out) {
    int64_t a, b;
    if (!e) return sys_fail(cc, "I expected an expression");
    switch (e->kind) {
    case E_INT: *out = e->i; return 0;
    case E_VAR:
        if (lookup_local(sy, e->name, out) < 0)
            return sys_fail(cc, "I do not know %s", e->name);
        return 0;
    case E_BIN:
        if (eval_expr(cc, sy, e->x, &a) < 0) return -1;
        if (eval_expr(cc, sy, e->y, &b) < 0) return -1;
        *out = e->bin == TK_MINUS ? a - b : a + b;
        return 0;
    case E_CON:
        if (e->x) return eval_expr(cc, sy, e->x, out);
        *out = 0;
        return 0;
    case E_STATE:
        return sys_fail(cc, "state is only for actors");
    }
    return sys_fail(cc, "I cannot evaluate that");
}

static int bind_local(Sys *sy, const char *n, int64_t v) {
    if (sy->nl >= 32) return -1;
    snprintf(sy->lnames[sy->nl], AC_NAME, "%s", n);
    sy->locals[sy->nl] = v;
    sy->nl++;
    return 0;
}

static void kill_actor(Sys *sy, int pid, int from_crash);

static int restart_actor(Sys *sy, int pid) {
    Act *a = &sy->acts[pid];
    ActorDef *d;
    if (pid <= 0 || pid >= sy->nact) return -1;
    d = &sy->cc->actors[a->def];
    if (a->vm_on) { vm_destroy(&a->vm); a->vm_on = 0; }
    a->alive = 1;
    a->cancelled = 0;
    a->state = d->init;
    a->bhead = a->bcount = 0;
    a->generation++;
    a->fn_idx = d->fn_idx;
    vm_init(&a->vm, sy->mod);
    a->vm_on = 1;
    return 0;
}

static void notify_down(Sys *sy, int pid) {
    Act *a = &sy->acts[pid];
    int i, down = ctor_lookup(sy->cc, "Down");
    for (i = 0; i < a->nmon; i++) {
        int m = a->monitors[i];
        if (m >= 0 && m < sy->nact && sy->acts[m].alive)
            enqueue(&sy->acts[m], pid, down, pid);
    }
    for (i = 0; i < a->nlink; i++) {
        int l = a->links[i];
        if (l == 0) sy->main_dead = 1;
        else if (l > 0) kill_actor(sy, l, 0);
    }
    if (a->linked_from_main) sy->main_dead = 1;
}

static void kill_actor(Sys *sy, int pid, int from_crash) {
    Act *a;
    (void)from_crash;
    if (pid <= 0 || pid >= sy->nact) return;
    a = &sy->acts[pid];
    if (!a->alive) return;
    a->alive = 0;
    notify_down(sy, pid);
    if (a->supervised) restart_actor(sy, pid);
}

static int deliver_handler(Sys *sy, int pid, Msg *msg) {
    Act *a = &sy->acts[pid];
    NanoValue args[3], out;
    VmResult r;
    int64_t op, st, tag, pay;
    args[0] = val_int(a->state);
    args[1] = val_int(msg->tag);
    args[2] = val_int(msg->payload);
    memset(&out, 0, sizeof out);
    r = vm_invoke(&a->vm, (uint32_t)a->fn_idx, args, 3, &out);
    if (r != VM_OK) {
        kill_actor(sy, pid, 1);
        return 0;
    }
    if (out.tag != TAG_TUPLE || !out.as.tuple || out.as.tuple->count != 4) {
        vm_release(&a->vm.heap, out);
        return sys_fail(sy->cc, "handler did not return a 4-tuple");
    }
    op = out.as.tuple->elements[0].as.i64;
    st = out.as.tuple->elements[1].as.i64;
    tag = out.as.tuple->elements[2].as.i64;
    pay = out.as.tuple->elements[3].as.i64;
    vm_release(&a->vm.heap, out);
    if (op == 2) {
        kill_actor(sy, pid, 1);
        return 0;
    }
    a->state = st;
    if (op == 0) {
        int dest = msg->from;
        if (dest >= 0 && dest < sy->nact)
            enqueue(&sy->acts[dest], pid, (int)tag, pay);
    }
    return 0;
}

static int drain(Sys *sy) {
    int progress = 1, guard = 0;
    while (progress && guard++ < 10000) {
        int i;
        progress = 0;
        for (i = 1; i < sy->nact; i++) {
            Msg m;
            if (!sy->acts[i].alive || sy->acts[i].cancelled || sy->acts[i].bcount == 0)
                continue;
            if (dequeue(&sy->acts[i], &m) < 0) continue;
            progress = 1;
            if (deliver_handler(sy, i, &m) < 0) return -1;
        }
    }
    return 0;
}

static int spawn_actor(Sys *sy, const char *name, int64_t *pid_out) {
    int def = actor_lookup(sy->cc, name);
    int kid = kid_lookup(sy->cc, name);
    int spec = -1;
    Act *a;
    if (kid >= 0) {
        spec = kid;
        def = actor_lookup(sy->cc, sy->cc->kids[kid].type);
    }
    if (def < 0) return sys_fail(sy->cc, "I do not know actor %s", name);
    if (sy->nact >= AC_MAX) return sys_fail(sy->cc, "I refuse too many pids");
    a = &sy->acts[sy->nact];
    memset(a, 0, sizeof *a);
    a->used = a->alive = 1;
    a->def = def;
    a->fn_idx = sy->cc->actors[def].fn_idx;
    a->state = sy->cc->actors[def].init;
    a->supervised = spec >= 0;
    a->spec = spec;
    vm_init(&a->vm, sy->mod);
    a->vm_on = 1;
    *pid_out = sy->nact;
    sy->nact++;
    return 0;
}

static int match_recv(Cc *cc, Sys *sy, Stmt *s, Msg *m, int64_t *out) {
    int i, saved = sy->nl;
    for (i = 0; i < s->narm; i++) {
        Pat *p = &s->arms[i].pat;
        sy->nl = saved;
        if (p->kind == P_WILD || p->kind == P_VAR) {
            if (p->kind == P_VAR && bind_local(sy, p->name, m->payload) < 0)
                return sys_fail(cc, "I refuse too many bindings");
            return eval_expr(cc, sy, s->arms[i].e, out);
        }
        if (p->kind == P_CON && p->ctor == m->tag) {
            if (p->bind[0] && bind_local(sy, p->bind, m->payload) < 0)
                return sys_fail(cc, "I refuse too many bindings");
            return eval_expr(cc, sy, s->arms[i].e, out);
        }
    }
    sy->nl = saved;
    return 1;
}

static int do_recv(Cc *cc, Sys *sy, Stmt *s, int64_t *out) {
    int timeout = ctor_lookup(cc, "Timeout");
    for (;;) {
        Msg m;
        int mr;
        if (drain(sy) < 0) return -1;
        if (sy->acts[0].bcount > 0) {
            if (dequeue(&sy->acts[0], &m) < 0) return -1;
            mr = match_recv(cc, sy, s, &m, out);
            if (mr < 0) return -1;
            if (mr == 0) return 0;
            continue;
        }
        if (s->has_after) {
            m.from = 0; m.tag = timeout; m.payload = 0;
            mr = match_recv(cc, sy, s, &m, out);
            if (mr < 0) return -1;
            if (mr == 0) return 0;
            return sys_fail(cc, "I had a deadline and no matching arm");
        }
        return sys_fail(cc, "I am waiting with an empty mailbox");
    }
}

static int run_sys(Cc *cc, NvmModule *mod, int64_t *out) {
    Sys *sy;
    int i;
    int64_t last = 0;
    int have_last = 0;
    sy = calloc(1, sizeof *sy);
    if (!sy) return sys_fail(cc, "I ran out of memory while starting actors");
    sy->cc = cc;
    sy->mod = mod;
    sy->nact = 1;
    sy->acts[0].used = sy->acts[0].alive = 1;
    for (i = 0; i < cc->nactor; i++) {
        uint32_t f;
        cc->actors[i].fn_idx = -1;
        for (f = 0; f < mod->function_count; f++) {
            const char *nm = nvm_get_string(mod, mod->functions[f].name_idx);
            if (nm && strcmp(nm, cc->actors[i].asm_name) == 0) {
                cc->actors[i].fn_idx = (int)f;
                break;
            }
        }
        if (cc->actors[i].fn_idx < 0) {
            sys_fail(cc, "I lost function %s", cc->actors[i].asm_name);
            goto fail;
        }
    }
    for (i = 0; i < cc->nstmt; i++) {
        Stmt *s = &cc->stmts[i];
        int64_t pid = 0, v = 0;
        if (s->kind == ST_SPAWN) {
            if (spawn_actor(sy, s->actor, &pid) < 0) goto fail;
            if (bind_local(sy, s->name, pid) < 0) {
                sys_fail(cc, "too many names");
                goto fail;
            }
            continue;
        }
        if (s->kind == ST_SEND) {
            int64_t dest;
            int64_t pay = 0;
            ActorDef *d;
            Act *a;
            if (eval_expr(cc, sy, s->pid, &dest) < 0) goto fail;
            if (s->payload && eval_expr(cc, sy, s->payload, &pay) < 0) goto fail;
            if (dest < 0 || dest >= sy->nact) {
                sys_fail(cc, "I do not know that pid");
                goto fail;
            }
            a = &sy->acts[(int)dest];
            if (a->cancelled) {
                sys_fail(cc, "I refuse send to a cancelled actor");
                goto fail;
            }
            if (!a->alive) {
                sys_fail(cc, "I refuse send to a dead actor");
                goto fail;
            }
            if ((int)dest > 0) {
                d = &cc->actors[a->def];
                if (!d->handled[s->ctor]) {
                    sys_fail(cc, "I refuse a mailbox type that actor does not receive");
                    goto fail;
                }
            }
            if (enqueue(a, 0, s->ctor, pay) < 0) {
                sys_fail(cc, "mailbox is full");
                goto fail;
            }
            if (drain(sy) < 0) goto fail;
            continue;
        }
        if (s->kind == ST_RECV) {
            if (do_recv(cc, sy, s, &v) < 0) goto fail;
            last = v;
            have_last = 1;
            if (s->name[0] && bind_local(sy, s->name, v) < 0) {
                sys_fail(cc, "too many names");
                goto fail;
            }
            continue;
        }
        if (s->kind == ST_CANCEL) {
            if (eval_expr(cc, sy, s->pid, &pid) < 0) goto fail;
            if (pid <= 0 || pid >= sy->nact) {
                sys_fail(cc, "I do not know that pid");
                goto fail;
            }
            sy->acts[(int)pid].cancelled = 1;
            sy->acts[(int)pid].alive = 0;
            continue;
        }
        if (s->kind == ST_MONITOR) {
            if (eval_expr(cc, sy, s->pid, &pid) < 0) goto fail;
            if (pid <= 0 || pid >= sy->nact) {
                sys_fail(cc, "I do not know that pid");
                goto fail;
            }
            {
                Act *a = &sy->acts[(int)pid];
                if (a->nmon >= 8) {
                    sys_fail(cc, "too many monitors");
                    goto fail;
                }
                a->monitors[a->nmon++] = 0;
            }
            continue;
        }
        if (s->kind == ST_LINK) {
            if (eval_expr(cc, sy, s->pid, &pid) < 0) goto fail;
            if (pid <= 0 || pid >= sy->nact) {
                sys_fail(cc, "I do not know that pid");
                goto fail;
            }
            {
                Act *a = &sy->acts[(int)pid];
                a->linked_from_main = 1;
                if (a->nlink < 8) a->links[a->nlink++] = 0;
            }
            continue;
        }
        if (s->kind == ST_REPLACE) {
            int def = actor_lookup(cc, s->actor);
            if (eval_expr(cc, sy, s->pid, &pid) < 0) goto fail;
            if (def < 0) {
                sys_fail(cc, "I do not know actor %s", s->actor);
                goto fail;
            }
            if (pid <= 0 || pid >= sy->nact) {
                sys_fail(cc, "I do not know that pid");
                goto fail;
            }
            sy->acts[(int)pid].def = def;
            sy->acts[(int)pid].fn_idx = cc->actors[def].fn_idx;
            continue;
        }
    }
    if (cc->result) {
        if (eval_expr(cc, sy, cc->result, out) < 0) goto fail;
    } else if (have_last) {
        *out = last;
    } else if (sy->nl > 0) {
        *out = sy->locals[sy->nl - 1];
    } else {
        *out = 0;
    }
    for (i = 1; i < sy->nact; i++) {
        if (sy->acts[i].vm_on) vm_destroy(&sy->acts[i].vm);
    }
    free(sy);
    return 0;

fail:
    for (i = 1; i < sy->nact; i++) {
        if (sy->acts[i].vm_on) vm_destroy(&sy->acts[i].vm);
    }
    free(sy);
    return -1;
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
    if (getenv("NL_ACTOR_TRACE")) fprintf(stderr, "%s", asmbuf.p ? asmbuf.p : "");
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

NvmModule *nl_actor_compile(const char *src, const char *path,
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

NlFrontendResult nl_actor_accept(const NvmModule *mod, const char *path) {
    NlFrontendFacts f;
    memset(&f, 0, sizeof f);
    f.language = NL_FE_ACTOR;
    f.source_path = (path && path[0]) ? path : "<actor>";
    f.purity = 0;
    f.exhaustiveness = 1;
    f.affine_use = -1;
    f.diagnostics_shared = 1;
    return nl_frontend_accept(mod, &f);
}

int nl_actor_eval_i64(const char *src, int64_t *out, char *err, size_t errlen) {
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
    acc = nl_actor_accept(mod, "<eval>");
    if (!acc.ok) {
        if (err && errlen) snprintf(err, errlen, "%s", acc.error);
        nvm_module_free(mod);
        cc_free(&cc);
        return 0;
    }
    if (run_sys(&cc, mod, &v) < 0) {
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
