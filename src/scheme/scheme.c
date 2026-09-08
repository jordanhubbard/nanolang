/*
 * Nano Scheme — bounded laboratory frontend. I emit verified NanoISA.
 * C host compiler; no src_nano twin. See docs/SCHEME.md.
 */

#include "scheme.h"

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

#define SC_NAME 64
#define SC_MAX_FN 128
#define SC_MAX_BIND 32
#define SC_MAX_UP 16
#define SC_MAX_FORMS 256
#define SC_ERR NL_SCHEME_ERR_SIZE

typedef enum { SC_INT, SC_BOOL, SC_NIL, SC_SYM, SC_PAIR } ScKind;

typedef struct ScObj ScObj;
struct ScObj {
    ScKind kind;
    union {
        int64_t i;
        int b;
        char *sym;
        struct { ScObj *car; ScObj *cdr; } pair;
    } u;
};

typedef struct {
    char *p;
    size_t n, cap;
} Buf;

typedef struct {
    ScObj *form;
    char asm_name[SC_NAME];
    char scheme_name[SC_NAME];
    int is_define;
    int arity;
    int nup;
    char params[SC_MAX_BIND][SC_NAME];
    char upnames[SC_MAX_UP][SC_NAME];
    ScObj *body;
    Buf code;
    uint8_t result_tag;
} ScFn;

typedef struct NameEnv {
    struct NameEnv *parent;
    int n;
    char names[SC_MAX_BIND][SC_NAME];
} NameEnv;

typedef struct {
    int n;
    char names[SC_MAX_BIND][SC_NAME];
    int kind[SC_MAX_BIND]; /* 0 local, 1 upvalue */
    int index[SC_MAX_BIND];
} Env;

typedef struct {
    ScObj **objs;
    uint32_t nobj, capobj;
    char **strs;
    uint32_t nstr, capstr;
    ScObj *forms[SC_MAX_FORMS];
    int nforms;
    ScFn fns[SC_MAX_FN];
    int nfn;
    int cur;
    int label;
    int terminated;
    char err[SC_ERR];
} Cc;

struct NlScheme {
    struct {
        char name[SC_NAME];
        char *src;
    } defs[64];
    uint32_t ndef;
};

static const char *const k_prim[] = {
    "quote", "if", "begin", "lambda", "define", "let", "set!", "call/cc",
    "cons", "car", "cdr", "null?", "pair?", "+", "-", "*", "/", "=", "<", ">",
    "eq?", "not", NULL
};

static int cc_fail(Cc *cc, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(cc->err, sizeof cc->err, fmt, ap);
    va_end(ap);
    return -1;
}

static int buf_grow(Buf *b, size_t need) {
    size_t cap = b->cap ? b->cap : 256;
    char *p;
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

static ScObj *sc_alloc(Cc *cc, ScKind kind) {
    ScObj *o;
    if (cc->nobj >= cc->capobj) {
        uint32_t cap = cc->capobj ? cc->capobj * 2 : 64;
        ScObj **p = realloc(cc->objs, cap * sizeof(*p));
        if (!p) return NULL;
        cc->objs = p;
        cc->capobj = cap;
    }
    o = calloc(1, sizeof(*o));
    if (!o) return NULL;
    o->kind = kind;
    cc->objs[cc->nobj++] = o;
    return o;
}

static char *sc_strdup(Cc *cc, const char *s) {
    char *d;
    if (cc->nstr >= cc->capstr) {
        uint32_t cap = cc->capstr ? cc->capstr * 2 : 64;
        char **p = realloc(cc->strs, cap * sizeof(*p));
        if (!p) return NULL;
        cc->strs = p;
        cc->capstr = cap;
    }
    d = strdup(s ? s : "");
    if (!d) return NULL;
    cc->strs[cc->nstr++] = d;
    return d;
}

static ScObj *sc_nil(Cc *cc) {
    return sc_alloc(cc, SC_NIL);
}

static ScObj *sc_int(Cc *cc, int64_t v) {
    ScObj *o = sc_alloc(cc, SC_INT);
    if (o) o->u.i = v;
    return o;
}

static ScObj *sc_bool(Cc *cc, int v) {
    ScObj *o = sc_alloc(cc, SC_BOOL);
    if (o) o->u.b = v ? 1 : 0;
    return o;
}

static ScObj *sc_sym(Cc *cc, const char *s) {
    ScObj *o = sc_alloc(cc, SC_SYM);
    if (o) o->u.sym = sc_strdup(cc, s);
    return o;
}

static ScObj *sc_cons(Cc *cc, ScObj *a, ScObj *d) {
    ScObj *o = sc_alloc(cc, SC_PAIR);
    if (o) {
        o->u.pair.car = a;
        o->u.pair.cdr = d;
    }
    return o;
}

static void cc_free(Cc *cc) {
    uint32_t i;
    int f;
    for (i = 0; i < cc->nobj; i++) free(cc->objs[i]);
    free(cc->objs);
    for (i = 0; i < cc->nstr; i++) free(cc->strs[i]);
    free(cc->strs);
    for (f = 0; f < cc->nfn; f++) free(cc->fns[f].code.p);
}

static int is_pair(const ScObj *e) { return e && e->kind == SC_PAIR; }
static int is_nil(const ScObj *e) { return e && e->kind == SC_NIL; }
static int is_sym(const ScObj *e) { return e && e->kind == SC_SYM; }

static ScObj *car(ScObj *e) { return is_pair(e) ? e->u.pair.car : NULL; }
static ScObj *cdr(ScObj *e) { return is_pair(e) ? e->u.pair.cdr : NULL; }

static int sym_is(const ScObj *e, const char *s) {
    return is_sym(e) && e->u.sym && strcmp(e->u.sym, s) == 0;
}

static int proper_len(ScObj *e) {
    int n = 0;
    while (is_pair(e)) {
        n++;
        e = e->u.pair.cdr;
    }
    if (e && e->kind != SC_NIL) return -1;
    return n;
}

static ScObj *nth(ScObj *e, int i) {
    while (i > 0 && is_pair(e)) {
        e = e->u.pair.cdr;
        i--;
    }
    return is_pair(e) ? e->u.pair.car : NULL;
}

static int is_primitive(const char *s) {
    int i;
    for (i = 0; k_prim[i]; i++) {
        if (strcmp(k_prim[i], s) == 0) return 1;
    }
    return 0;
}

static void skip_ws(const char **p) {
    for (;;) {
        while (**p == ' ' || **p == '\t' || **p == '\n' || **p == '\r') (*p)++;
        if (**p == ';') {
            while (**p && **p != '\n') (*p)++;
            continue;
        }
        break;
    }
}

static int is_ident_char(char c) {
    if (isalnum((unsigned char)c)) return 1;
    return strchr("!$%&*+-./:<=>?@^_~", c) != NULL;
}

static ScObj *parse_expr(Cc *cc, const char **p);

static ScObj *parse_list(Cc *cc, const char **p) {
    ScObj *head = NULL, *tail = NULL, *item;
    skip_ws(p);
    if (**p == ')') {
        (*p)++;
        return sc_nil(cc);
    }
    while (**p && **p != ')') {
        skip_ws(p);
        if (**p == ')') break;
        if (**p == '.' && !is_ident_char((*p)[1])) {
            (*p)++;
            skip_ws(p);
            item = parse_expr(cc, p);
            if (!item) return NULL;
            skip_ws(p);
            if (**p != ')') {
                cc_fail(cc, "I expected ')' after a dotted pair");
                return NULL;
            }
            (*p)++;
            if (!head) return item;
            tail->u.pair.cdr = item;
            return head;
        }
        item = parse_expr(cc, p);
        if (!item) return NULL;
        item = sc_cons(cc, item, sc_nil(cc));
        if (!item) return NULL;
        if (!head) head = tail = item;
        else {
            tail->u.pair.cdr = item;
            tail = item;
        }
        skip_ws(p);
    }
    if (**p != ')') {
        cc_fail(cc, "I expected ')' to close a list");
        return NULL;
    }
    (*p)++;
    return head ? head : sc_nil(cc);
}

static ScObj *parse_expr(Cc *cc, const char **p) {
    skip_ws(p);
    if (**p == '\0') {
        cc_fail(cc, "I expected an expression");
        return NULL;
    }
    if (**p == '\'') {
        ScObj *inner;
        (*p)++;
        inner = parse_expr(cc, p);
        if (!inner) return NULL;
        return sc_cons(cc, sc_sym(cc, "quote"), sc_cons(cc, inner, sc_nil(cc)));
    }
    if (**p == '(') {
        (*p)++;
        return parse_list(cc, p);
    }
    if (**p == '#') {
        (*p)++;
        if (**p == 't' || **p == 'T') { (*p)++; return sc_bool(cc, 1); }
        if (**p == 'f' || **p == 'F') { (*p)++; return sc_bool(cc, 0); }
        cc_fail(cc, "I do not read that # token in this subset");
        return NULL;
    }
    if (((**p == '+' || **p == '-') && isdigit((unsigned char)(*p)[1]))
            || isdigit((unsigned char)**p)) {
        char *end = NULL;
        long long v = strtoll(*p, &end, 10);
        if (end == *p) {
            cc_fail(cc, "I could not read that integer");
            return NULL;
        }
        *p = end;
        return sc_int(cc, (int64_t)v);
    }
    if (is_ident_char(**p) && **p != '(' && **p != ')') {
        char buf[SC_NAME];
        size_t n = 0;
        while (is_ident_char(**p) && n + 1 < sizeof buf) buf[n++] = *(*p)++;
        buf[n] = '\0';
        if (n == 0) {
            cc_fail(cc, "I expected an identifier");
            return NULL;
        }
        return sc_sym(cc, buf);
    }
    cc_fail(cc, "I do not read that token in this subset");
    return NULL;
}

static int parse_program(Cc *cc, const char *src) {
    const char *p = src;
    skip_ws(&p);
    while (*p) {
        ScObj *e;
        if (cc->nforms >= SC_MAX_FORMS)
            return cc_fail(cc, "I refuse a program with more than %d forms", SC_MAX_FORMS);
        e = parse_expr(cc, &p);
        if (!e) return -1;
        cc->forms[cc->nforms++] = e;
        skip_ws(&p);
    }
    return 0;
}

static int mangle_into(const char *in, char *out, size_t n) {
    size_t j = 0;
    size_t i;
    if (n < 4) return -1;
    out[j++] = 's';
    out[j++] = 'c';
    out[j++] = '_';
    for (i = 0; in[i]; i++) {
        unsigned char c = (unsigned char)in[i];
        if (j + 3 >= n) return -1;
        if (isalnum(c) || c == '_') out[j++] = (char)c;
        else if (c == '-') out[j++] = '_';
        else if (c == '?') { out[j++] = '_'; out[j++] = 'p'; }
        else if (c == '!') { out[j++] = '_'; out[j++] = 'b'; }
        else out[j++] = '_';
    }
    out[j] = '\0';
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

static int global_fn(Cc *cc, const char *name) {
    int i;
    for (i = 0; i < cc->nfn; i++) {
        if (cc->fns[i].scheme_name[0] && strcmp(cc->fns[i].scheme_name, name) == 0)
            return i;
    }
    return -1;
}

static int fn_for_form(Cc *cc, ScObj *form) {
    int i;
    for (i = 0; i < cc->nfn; i++) {
        if (cc->fns[i].form == form) return i;
    }
    return -1;
}

static int nameenv_has(const NameEnv *e, const char *n) {
    while (e) {
        int i;
        for (i = 0; i < e->n; i++) {
            if (strcmp(e->names[i], n) == 0) return 1;
        }
        e = e->parent;
    }
    return 0;
}

static int add_free(ScFn *fn, const char *name) {
    int i;
    for (i = 0; i < fn->nup; i++) {
        if (strcmp(fn->upnames[i], name) == 0) return 0;
    }
    if (fn->nup >= SC_MAX_UP) return -1;
    snprintf(fn->upnames[fn->nup], SC_NAME, "%s", name);
    fn->nup++;
    return 0;
}

static int in_params(const ScFn *fn, const char *name) {
    int i;
    for (i = 0; i < fn->arity; i++) {
        if (strcmp(fn->params[i], name) == 0) return 1;
    }
    return 0;
}

static int collect_frees(Cc *cc, ScObj *e, ScFn *fn, const NameEnv *parent);

static int collect_frees(Cc *cc, ScObj *e, ScFn *fn, const NameEnv *parent) {
    int inner, u;
    if (!e) return 0;
    if (e->kind == SC_SYM) {
        const char *n = e->u.sym;
        if (in_params(fn, n) || is_primitive(n) || global_fn(cc, n) >= 0) return 0;
        if (nameenv_has(parent, n)) {
            if (add_free(fn, n) < 0)
                return cc_fail(cc, "I refuse more than %d free variables", SC_MAX_UP);
            return 0;
        }
        return cc_fail(cc, "I do not know '%s'", n);
    }
    if (!is_pair(e)) return 0;
    if (sym_is(car(e), "quote")) return 0;
    if (sym_is(car(e), "lambda")) {
        inner = fn_for_form(cc, e);
        if (inner < 0) return cc_fail(cc, "internal: nested lambda was not registered");
        for (u = 0; u < cc->fns[inner].nup; u++) {
            if (!in_params(fn, cc->fns[inner].upnames[u])) {
                if (add_free(fn, cc->fns[inner].upnames[u]) < 0)
                    return cc_fail(cc, "I refuse more than %d free variables", SC_MAX_UP);
            }
        }
        return 0;
    }
    if (collect_frees(cc, car(e), fn, parent) < 0) return -1;
    return collect_frees(cc, cdr(e), fn, parent);
}

static int parse_params(Cc *cc, ScObj *args, ScFn *fn) {
    fn->arity = 0;
    while (is_pair(args)) {
        ScObj *a = car(args);
        if (!is_sym(a)) return cc_fail(cc, "I only bind symbols as parameters");
        if (fn->arity >= SC_MAX_BIND)
            return cc_fail(cc, "I refuse more than %d parameters", SC_MAX_BIND);
        snprintf(fn->params[fn->arity], SC_NAME, "%s", a->u.sym);
        fn->arity++;
        args = cdr(args);
    }
    if (args && !is_nil(args))
        return cc_fail(cc, "I do not compile rest arguments in this subset");
    return 0;
}

static int walk_nested(Cc *cc, ScObj *e, NameEnv *parent);

static int register_fn(Cc *cc, ScObj *form, NameEnv *parent, int is_define,
                       const char *scheme_name) {
    ScFn *fn;
    NameEnv plus;
    ScObj *body;
    int i;

    if (cc->nfn >= SC_MAX_FN)
        return cc_fail(cc, "I refuse more than %d functions", SC_MAX_FN);
    fn = &cc->fns[cc->nfn];
    memset(fn, 0, sizeof(*fn));
    fn->form = form;
    fn->is_define = is_define;
    fn->result_tag = 0xFF;
    (void)scheme_name;
    if (is_define) {
        ScObj *spec = nth(form, 1);
        if (is_sym(spec) && is_pair(nth(form, 2)) && sym_is(car(nth(form, 2)), "lambda")) {
            ScObj *lam = nth(form, 2);
            snprintf(fn->scheme_name, SC_NAME, "%s", spec->u.sym);
            if (parse_params(cc, nth(lam, 1), fn) < 0) return -1;
            body = cdr(cdr(lam));
        } else if (is_pair(spec)) {
            if (!is_sym(car(spec)))
                return cc_fail(cc, "I expected a procedure name after define");
            snprintf(fn->scheme_name, SC_NAME, "%s", car(spec)->u.sym);
            if (parse_params(cc, cdr(spec), fn) < 0) return -1;
            body = cdr(cdr(form));
        } else {
            return cc_fail(cc, "I only bind procedures at top level in this subset");
        }
        if (mangle_into(fn->scheme_name, fn->asm_name, sizeof fn->asm_name) < 0)
            return cc_fail(cc, "I could not mangle '%s'", fn->scheme_name);
        if (scheme_name && scheme_name[0] && strcmp(scheme_name, fn->scheme_name) != 0)
            return cc_fail(cc, "internal define name");
    } else {
        if (parse_params(cc, nth(form, 1), fn) < 0) return -1;
        body = cdr(cdr(form));
        snprintf(fn->asm_name, sizeof fn->asm_name, "_L%d", cc->nfn);
    }
    fn->body = body ? body : sc_nil(cc);
    if (proper_len(fn->body) < 1)
        return cc_fail(cc, "I refuse an empty procedure body");
    cc->nfn++;

    memset(&plus, 0, sizeof plus);
    plus.parent = parent;
    plus.n = fn->arity;
    for (i = 0; i < fn->arity; i++)
        snprintf(plus.names[i], SC_NAME, "%s", fn->params[i]);
    if (walk_nested(cc, fn->body, &plus) < 0) return -1;
    if (collect_frees(cc, fn->body, fn, parent) < 0) return -1;
    return 0;
}

static int walk_nested(Cc *cc, ScObj *e, NameEnv *parent) {
    if (!e) return 0;
    if (!is_pair(e)) return 0;
    if (sym_is(car(e), "quote")) return 0;
    if (sym_is(car(e), "lambda"))
        return register_fn(cc, e, parent, 0, "");
    if (walk_nested(cc, car(e), parent) < 0) return -1;
    return walk_nested(cc, cdr(e), parent);
}

static ScObj *map_desugar(Cc *cc, ScObj *e);

static ScObj *desugar(Cc *cc, ScObj *e) {
    int n, i, nb;
    ScObj *bindings, *body, *vars, *vals, *lam;
    ScObj *var[SC_MAX_BIND], *val[SC_MAX_BIND];
    if (!e || e->kind != SC_PAIR) return e;
    if (sym_is(car(e), "quote")) return e;
    if (sym_is(car(e), "if")) {
        n = proper_len(e);
        if (n == 3) {
            ScObj *t = desugar(cc, nth(e, 1));
            ScObj *a = desugar(cc, nth(e, 2));
            return sc_cons(cc, sc_sym(cc, "if"),
                           sc_cons(cc, t,
                                   sc_cons(cc, a,
                                           sc_cons(cc, sc_bool(cc, 0), sc_nil(cc)))));
        }
        return map_desugar(cc, e);
    }
    if (sym_is(car(e), "let")) {
        bindings = nth(e, 1);
        body = cdr(cdr(e));
        nb = proper_len(bindings);
        if (nb < 0) {
            cc_fail(cc, "I expected a proper binding list in let");
            return NULL;
        }
        if (nb > SC_MAX_BIND) {
            cc_fail(cc, "I refuse more than %d let bindings", SC_MAX_BIND);
            return NULL;
        }
        if (proper_len(body) < 1) {
            cc_fail(cc, "I refuse an empty let body");
            return NULL;
        }
        for (i = 0; i < nb; i++) {
            ScObj *b = nth(bindings, i);
            if (proper_len(b) != 2 || !is_sym(car(b))) {
                cc_fail(cc, "I expected (name expr) in let");
                return NULL;
            }
            var[i] = car(b);
            val[i] = desugar(cc, nth(b, 1));
            if (!val[i]) return NULL;
        }
        vars = sc_nil(cc);
        vals = sc_nil(cc);
        for (i = nb - 1; i >= 0; i--) {
            vars = sc_cons(cc, var[i], vars);
            vals = sc_cons(cc, val[i], vals);
        }
        body = map_desugar(cc, body);
        if (!body) return NULL;
        lam = sc_cons(cc, sc_sym(cc, "lambda"), sc_cons(cc, vars, body));
        return sc_cons(cc, lam, vals);
    }
    return map_desugar(cc, e);
}

static ScObj *map_desugar(Cc *cc, ScObj *e) {
    if (!e) return NULL;
    if (!is_pair(e)) return e;
    if (sym_is(car(e), "quote")) return e;
    {
        ScObj *a = desugar(cc, car(e));
        ScObj *d = map_desugar(cc, cdr(e));
        if (!a || (!d && cdr(e))) return NULL;
        return sc_cons(cc, a, d ? d : sc_nil(cc));
    }
}

static int lookup_env(const Env *env, const char *name, int *kind, int *idx) {
    int i;
    for (i = 0; i < env->n; i++) {
        if (strcmp(env->names[i], name) == 0) {
            *kind = env->kind[i];
            *idx = env->index[i];
            return 1;
        }
    }
    return 0;
}

static int compile_expr(Cc *cc, const Env *env, ScObj *e, int tail);

static int emit_nil(Cc *cc) {
    if (emit_line(cc, 1, "PUSH_VOID") < 0) return -1;
    if (emit_line(cc, 1, "PUSH_VOID") < 0) return -1;
    return emit_line(cc, 1, "TUPLE_NEW 2");
}

static int compile_datum(Cc *cc, ScObj *e) {
    if (!e) return cc_fail(cc, "I cannot quote that");
    if (e->kind == SC_INT) return emit_line(cc, 1, "PUSH_I64 %lld", (long long)e->u.i);
    if (e->kind == SC_BOOL) return emit_line(cc, 1, "PUSH_BOOL %u", e->u.b ? 1u : 0u);
    if (e->kind == SC_NIL) return emit_nil(cc);
    if (e->kind == SC_PAIR) {
        if (compile_datum(cc, car(e)) < 0) return -1;
        if (compile_datum(cc, cdr(e)) < 0) return -1;
        return emit_line(cc, 1, "TUPLE_NEW 2");
    }
    return cc_fail(cc, "I do not quote symbols in this subset");
}

static int compile_ref(Cc *cc, const Env *env, const char *name, int tail) {
    int kind, idx, g;
    if (lookup_env(env, name, &kind, &idx)) {
        if (kind == 0) {
            if (emit_line(cc, 1, "LOAD_LOCAL %d", idx) < 0) return -1;
        } else {
            if (emit_line(cc, 1, "LOAD_UPVALUE 0 %d", idx) < 0) return -1;
        }
        return finish_tail(cc, tail);
    }
    g = global_fn(cc, name);
    if (g >= 0) {
        if (emit_line(cc, 1, "FUNCREF %s", cc->fns[g].asm_name) < 0) return -1;
        return finish_tail(cc, tail);
    }
    if (is_primitive(name))
        return cc_fail(cc, "I do not lift '%s' to a first-class value in this subset", name);
    return cc_fail(cc, "I do not know '%s'", name);
}

static int compile_begin(Cc *cc, const Env *env, ScObj *forms, int tail) {
    int n = proper_len(forms);
    int i;
    if (n < 0) return cc_fail(cc, "I expected a proper begin list");
    if (n == 0) {
        if (emit_line(cc, 1, "PUSH_VOID") < 0) return -1;
        return finish_tail(cc, tail);
    }
    for (i = 0; i < n - 1; i++) {
        if (compile_expr(cc, env, nth(forms, i), 0) < 0) return -1;
        if (emit_line(cc, 1, "POP") < 0) return -1;
    }
    return compile_expr(cc, env, nth(forms, n - 1), tail);
}

static int compile_lambda_value(Cc *cc, const Env *env, ScObj *form, int tail) {
    int idx = fn_for_form(cc, form);
    ScFn *fn;
    int i;
    if (idx < 0) return cc_fail(cc, "internal: lambda was not registered");
    fn = &cc->fns[idx];
    for (i = 0; i < fn->nup; i++) {
        if (compile_ref(cc, env, fn->upnames[i], 0) < 0) return -1;
    }
    if (emit_line(cc, 1, "CLOSURE_NEW %s %d", fn->asm_name, fn->nup) < 0) return -1;
    return finish_tail(cc, tail);
}

static int compile_app(Cc *cc, const Env *env, ScObj *e, int tail) {
    ScObj *op = car(e);
    ScObj *args = cdr(e);
    int n = proper_len(args);
    int i, g;
    if (n < 0) return cc_fail(cc, "I expected a proper argument list");
    if (is_sym(op) && (g = global_fn(cc, op->u.sym)) >= 0) {
        if (n != cc->fns[g].arity)
            return cc_fail(cc, "'%s' takes %d arguments", op->u.sym, cc->fns[g].arity);
        for (i = 0; i < n; i++) {
            if (compile_expr(cc, env, nth(args, i), 0) < 0) return -1;
        }
        if (tail) {
            if (emit_line(cc, 1, "TAIL_CALL %s", cc->fns[g].asm_name) < 0) return -1;
            cc->terminated = 1;
            return 0;
        }
        return emit_line(cc, 1, "CALL %s", cc->fns[g].asm_name);
    }
    if (is_pair(op) && sym_is(car(op), "lambda")) {
        int idx = fn_for_form(cc, op);
        if (idx < 0) return cc_fail(cc, "internal: lambda application");
        if (n != cc->fns[idx].arity)
            return cc_fail(cc, "that lambda takes %d arguments", cc->fns[idx].arity);
        for (i = 0; i < n; i++) {
            if (compile_expr(cc, env, nth(args, i), 0) < 0) return -1;
        }
        if (cc->fns[idx].nup == 0) {
            if (tail) {
                if (emit_line(cc, 1, "TAIL_CALL %s", cc->fns[idx].asm_name) < 0) return -1;
                cc->terminated = 1;
                return 0;
            }
            return emit_line(cc, 1, "CALL %s", cc->fns[idx].asm_name);
        }
        for (i = 0; i < cc->fns[idx].nup; i++) {
            if (compile_ref(cc, env, cc->fns[idx].upnames[i], 0) < 0) return -1;
        }
        if (emit_line(cc, 1, "CLOSURE_NEW %s %d", cc->fns[idx].asm_name, cc->fns[idx].nup) < 0)
            return -1;
        if (emit_line(cc, 1, "CALL_INDIRECT %d 1", n) < 0) return -1;
        return finish_tail(cc, tail);
    }
    for (i = 0; i < n; i++) {
        if (compile_expr(cc, env, nth(args, i), 0) < 0) return -1;
    }
    if (compile_expr(cc, env, op, 0) < 0) return -1;
    if (emit_line(cc, 1, "CALL_INDIRECT %d 1", n) < 0) return -1;
    return finish_tail(cc, tail);
}

static int fold_i64(Cc *cc, const Env *env, ScObj *args, const char *op,
                    int unary_ok, int tail) {
    int n = proper_len(args);
    int i;
    if (n < 0) return cc_fail(cc, "I expected a proper argument list");
    if (n == 0) {
        if (strcmp(op, "I64_ADD") == 0) {
            if (emit_line(cc, 1, "PUSH_I64 0") < 0) return -1;
            return finish_tail(cc, tail);
        }
        if (strcmp(op, "I64_MUL") == 0) {
            if (emit_line(cc, 1, "PUSH_I64 1") < 0) return -1;
            return finish_tail(cc, tail);
        }
        return cc_fail(cc, "I need arguments for that operator");
    }
    if (n == 1 && unary_ok && strcmp(op, "I64_SUB") == 0) {
        if (compile_expr(cc, env, car(args), 0) < 0) return -1;
        if (emit_line(cc, 1, "I64_NEG") < 0) return -1;
        return finish_tail(cc, tail);
    }
    if (compile_expr(cc, env, car(args), 0) < 0) return -1;
    for (i = 1; i < n; i++) {
        if (compile_expr(cc, env, nth(args, i), 0) < 0) return -1;
        if (emit_line(cc, 1, "%s", op) < 0) return -1;
    }
    return finish_tail(cc, tail);
}

static int compile_form(Cc *cc, const Env *env, ScObj *e, int tail) {
    ScObj *op = car(e);
    ScObj *args = cdr(e);
    int n = proper_len(e);
    int l_else, l_end;

    if (n < 0) return cc_fail(cc, "I expected a proper list");
    if (!is_sym(op) && !is_pair(op))
        return cc_fail(cc, "I cannot call that");

    if (sym_is(op, "quote")) {
        if (n != 2) return cc_fail(cc, "quote takes one operand");
        if (compile_datum(cc, nth(e, 1)) < 0) return -1;
        return finish_tail(cc, tail);
    }
    if (sym_is(op, "if")) {
        if (n != 4) return cc_fail(cc, "if takes a test, a then, and an else");
        l_else = cc->label++;
        l_end = cc->label++;
        if (compile_expr(cc, env, nth(e, 1), 0) < 0) return -1;
        if (emit_line(cc, 1, "PUSH_BOOL 0") < 0) return -1;
        if (emit_line(cc, 1, "EQ") < 0) return -1;
        if (emit_line(cc, 1, "JMP_TRUE lf%u", (unsigned)l_else) < 0) return -1;
        if (tail) {
            if (compile_expr(cc, env, nth(e, 2), 1) < 0) return -1;
            if (emit_line(cc, 0, "lf%u:", (unsigned)l_else) < 0) return -1;
            return compile_expr(cc, env, nth(e, 3), 1);
        }
        if (compile_expr(cc, env, nth(e, 2), 0) < 0) return -1;
        if (emit_line(cc, 1, "JMP le%u", (unsigned)l_end) < 0) return -1;
        if (emit_line(cc, 0, "lf%u:", (unsigned)l_else) < 0) return -1;
        if (compile_expr(cc, env, nth(e, 3), 0) < 0) return -1;
        if (emit_line(cc, 0, "le%u:", (unsigned)l_end) < 0) return -1;
        return 0;
    }
    if (sym_is(op, "begin")) return compile_begin(cc, env, args, tail);
    if (sym_is(op, "lambda")) return compile_lambda_value(cc, env, e, tail);
    if (sym_is(op, "define"))
        return cc_fail(cc, "I only accept define at top level");
    if (sym_is(op, "set!"))
        return cc_fail(cc, "I refuse set!; mutation waits with exceptions");
    if (sym_is(op, "call/cc"))
        return cc_fail(cc, "I refuse call/cc; continuations are out of scope");
    if (sym_is(op, "cons")) {
        if (n != 3) return cc_fail(cc, "cons takes two operands");
        if (compile_expr(cc, env, nth(e, 1), 0) < 0) return -1;
        if (compile_expr(cc, env, nth(e, 2), 0) < 0) return -1;
        if (emit_line(cc, 1, "TUPLE_NEW 2") < 0) return -1;
        return finish_tail(cc, tail);
    }
    if (sym_is(op, "car") || sym_is(op, "cdr")) {
        if (n != 2) return cc_fail(cc, "%s takes one operand", op->u.sym);
        if (compile_expr(cc, env, nth(e, 1), 0) < 0) return -1;
        if (emit_line(cc, 1, "TUPLE_GET %u", sym_is(op, "car") ? 0u : 1u) < 0) return -1;
        return finish_tail(cc, tail);
    }
    if (sym_is(op, "null?")) {
        int l_no, l_end;
        if (n != 2) return cc_fail(cc, "null? takes one operand");
        l_no = cc->label++;
        l_end = cc->label++;
        if (compile_expr(cc, env, nth(e, 1), 0) < 0) return -1;
        if (emit_line(cc, 1, "DUP") < 0) return -1;
        if (emit_line(cc, 1, "TYPE_CHECK 12") < 0) return -1;
        if (emit_line(cc, 1, "JMP_FALSE lf%u", (unsigned)l_no) < 0) return -1;
        if (emit_line(cc, 1, "TUPLE_GET 0") < 0) return -1;
        if (emit_line(cc, 1, "TYPE_CHECK 0") < 0) return -1;
        if (emit_line(cc, 1, "JMP le%u", (unsigned)l_end) < 0) return -1;
        if (emit_line(cc, 0, "lf%u:", (unsigned)l_no) < 0) return -1;
        if (emit_line(cc, 1, "POP") < 0) return -1;
        if (emit_line(cc, 1, "PUSH_BOOL 0") < 0) return -1;
        if (emit_line(cc, 0, "le%u:", (unsigned)l_end) < 0) return -1;
        return finish_tail(cc, tail);
    }
    if (sym_is(op, "pair?")) {
        int l_no, l_end;
        if (n != 2) return cc_fail(cc, "pair? takes one operand");
        l_no = cc->label++;
        l_end = cc->label++;
        if (compile_expr(cc, env, nth(e, 1), 0) < 0) return -1;
        if (emit_line(cc, 1, "DUP") < 0) return -1;
        if (emit_line(cc, 1, "TYPE_CHECK 12") < 0) return -1;
        if (emit_line(cc, 1, "JMP_FALSE lf%u", (unsigned)l_no) < 0) return -1;
        if (emit_line(cc, 1, "TUPLE_GET 0") < 0) return -1;
        if (emit_line(cc, 1, "TYPE_CHECK 0") < 0) return -1;
        if (emit_line(cc, 1, "PUSH_BOOL 0") < 0) return -1;
        if (emit_line(cc, 1, "EQ") < 0) return -1;
        if (emit_line(cc, 1, "JMP le%u", (unsigned)l_end) < 0) return -1;
        if (emit_line(cc, 0, "lf%u:", (unsigned)l_no) < 0) return -1;
        if (emit_line(cc, 1, "POP") < 0) return -1;
        if (emit_line(cc, 1, "PUSH_BOOL 0") < 0) return -1;
        if (emit_line(cc, 0, "le%u:", (unsigned)l_end) < 0) return -1;
        return finish_tail(cc, tail);
    }
    if (sym_is(op, "not")) {
        if (n != 2) return cc_fail(cc, "not takes one operand");
        if (compile_expr(cc, env, nth(e, 1), 0) < 0) return -1;
        if (emit_line(cc, 1, "PUSH_BOOL 0") < 0) return -1;
        if (emit_line(cc, 1, "EQ") < 0) return -1;
        return finish_tail(cc, tail);
    }
    if (sym_is(op, "+")) return fold_i64(cc, env, args, "I64_ADD", 0, tail);
    if (sym_is(op, "-")) return fold_i64(cc, env, args, "I64_SUB", 1, tail);
    if (sym_is(op, "*")) return fold_i64(cc, env, args, "I64_MUL", 0, tail);
    if (sym_is(op, "/")) return fold_i64(cc, env, args, "I64_DIV_S", 0, tail);
    if (sym_is(op, "=") || sym_is(op, "<") || sym_is(op, ">") || sym_is(op, "eq?")) {
        const char *ins = "EQ";
        if (n != 3) return cc_fail(cc, "%s takes two operands", op->u.sym);
        if (sym_is(op, "=")) ins = "I64_EQ";
        else if (sym_is(op, "<")) ins = "I64_LT_S";
        else if (sym_is(op, ">")) ins = "I64_GT_S";
        if (compile_expr(cc, env, nth(e, 1), 0) < 0) return -1;
        if (compile_expr(cc, env, nth(e, 2), 0) < 0) return -1;
        if (emit_line(cc, 1, "%s", ins) < 0) return -1;
        return finish_tail(cc, tail);
    }
    return compile_app(cc, env, e, tail);
}

static int compile_expr(Cc *cc, const Env *env, ScObj *e, int tail) {
    cc->terminated = 0;
    if (!e) return cc_fail(cc, "I expected an expression");
    switch (e->kind) {
    case SC_INT:
        if (emit_line(cc, 1, "PUSH_I64 %lld", (long long)e->u.i) < 0) return -1;
        return finish_tail(cc, tail);
    case SC_BOOL:
        if (emit_line(cc, 1, "PUSH_BOOL %u", e->u.b ? 1u : 0u) < 0) return -1;
        return finish_tail(cc, tail);
    case SC_NIL:
        if (emit_nil(cc) < 0) return -1;
        return finish_tail(cc, tail);
    case SC_SYM:
        return compile_ref(cc, env, e->u.sym, tail);
    case SC_PAIR:
        return compile_form(cc, env, e, tail);
    }
    return cc_fail(cc, "I cannot compile that");
}

static void env_from_fn(const ScFn *fn, Env *env) {
    int i;
    memset(env, 0, sizeof *env);
    for (i = 0; i < fn->arity; i++) {
        snprintf(env->names[env->n], SC_NAME, "%s", fn->params[i]);
        env->kind[env->n] = 0;
        env->index[env->n] = i;
        env->n++;
    }
    for (i = 0; i < fn->nup; i++) {
        snprintf(env->names[env->n], SC_NAME, "%s", fn->upnames[i]);
        env->kind[env->n] = 1;
        env->index[env->n] = i;
        env->n++;
    }
}

static uint8_t merge_tag(uint8_t a, uint8_t b) {
    if (a == 0xFF) return b;
    if (b == 0xFF) return a;
    if (a == b) return a;
    if (a == TAG_TUPLE || b == TAG_TUPLE) return TAG_TUPLE;
    if (a == TAG_FUNCTION || b == TAG_FUNCTION) return TAG_FUNCTION;
    return a;
}

static uint8_t infer_expr(Cc *cc, ScFn *self, ScObj *e);

static uint8_t infer_body(Cc *cc, ScFn *self, ScObj *forms) {
    int n = proper_len(forms);
    if (n < 1) return 0xFF;
    return infer_expr(cc, self, nth(forms, n - 1));
}

static uint8_t infer_expr(Cc *cc, ScFn *self, ScObj *e) {
    ScObj *op, *args;
    int n, i, g;
    uint8_t t;
    if (!e) return 0xFF;
    if (e->kind == SC_INT) return TAG_INT;
    if (e->kind == SC_BOOL) return TAG_BOOL;
    if (e->kind == SC_NIL) return TAG_TUPLE;
    if (e->kind == SC_SYM) {
        g = global_fn(cc, e->u.sym);
        if (g >= 0) return TAG_FUNCTION;
        return 0xFF;
    }
    if (!is_pair(e)) return 0xFF;
    op = car(e);
    args = cdr(e);
    n = proper_len(e);
    if (sym_is(op, "quote")) {
        ScObj *d = nth(e, 1);
        if (d && (d->kind == SC_NIL || d->kind == SC_PAIR)) return TAG_TUPLE;
        return infer_expr(cc, self, d);
    }
    if (sym_is(op, "if") && n == 4)
        return merge_tag(infer_expr(cc, self, nth(e, 2)), infer_expr(cc, self, nth(e, 3)));
    if (sym_is(op, "begin")) return infer_body(cc, self, args);
    if (sym_is(op, "lambda")) return TAG_FUNCTION;
    if (sym_is(op, "cons")) return TAG_TUPLE;
    if (sym_is(op, "car")) return 0xFF;
    if (sym_is(op, "cdr")) return 0xFF;
    if (sym_is(op, "+") || sym_is(op, "-") || sym_is(op, "*") || sym_is(op, "/"))
        return TAG_INT;
    if (sym_is(op, "=") || sym_is(op, "<") || sym_is(op, ">") || sym_is(op, "eq?")
            || sym_is(op, "not") || sym_is(op, "null?") || sym_is(op, "pair?"))
        return TAG_BOOL;
    if (is_sym(op) && (g = global_fn(cc, op->u.sym)) >= 0) {
        t = cc->fns[g].result_tag;
        if (self && g == (int)(self - cc->fns)) {
            for (i = 0; i < proper_len(args); i++) {
                if (infer_expr(cc, self, nth(args, i)) == TAG_TUPLE)
                    t = TAG_TUPLE;
            }
        }
        return t;
    }
    if (is_pair(op) && sym_is(car(op), "lambda")) {
        g = fn_for_form(cc, op);
        if (g >= 0) return cc->fns[g].result_tag;
    }
    return 0xFF;
}

static void infer_results(Cc *cc) {
    int pass, i;
    for (pass = 0; pass < cc->nfn + 2; pass++) {
        int changed = 0;
        for (i = 0; i < cc->nfn; i++) {
            uint8_t t = infer_body(cc, &cc->fns[i], cc->fns[i].body);
            uint8_t merged = merge_tag(cc->fns[i].result_tag, t);
            if (merged != cc->fns[i].result_tag && merged != 0xFF) {
                cc->fns[i].result_tag = merged;
                changed = 1;
            }
        }
        if (!changed) break;
    }
    for (i = 0; i < cc->nfn; i++) {
        if (cc->fns[i].result_tag == 0xFF) cc->fns[i].result_tag = TAG_INT;
    }
}

static const char *result_tag_name(uint8_t tag) {
    const char *n = isa_tag_name(tag);
    return n ? n : "int";
}

static int compile_function(Cc *cc, int idx) {
    Env env;
    ScFn *fn = &cc->fns[idx];
    cc->cur = idx;
    cc->terminated = 0;
    if (strncmp(fn->asm_name, "_dead", 5) == 0) {
        if (emit_line(cc, 1, "PUSH_I64 0") < 0) return -1;
        return emit_line(cc, 1, "RET");
    }
    env_from_fn(fn, &env);
    return compile_begin(cc, &env, fn->body, 1);
}

static int compile_main(Cc *cc, ScObj **exprs, int nexpr) {
    ScFn *fn;
    Env env;
    int i;
    if (cc->nfn >= SC_MAX_FN)
        return cc_fail(cc, "I refuse more than %d functions", SC_MAX_FN);
    fn = &cc->fns[cc->nfn];
    memset(fn, 0, sizeof(*fn));
    snprintf(fn->asm_name, sizeof fn->asm_name, "_sc_main");
    fn->body = sc_nil(cc);
    fn->result_tag = TAG_INT;
    cc->cur = cc->nfn;
    cc->nfn++;
    memset(&env, 0, sizeof env);
    if (nexpr == 0) {
        if (emit_line(cc, 1, "PUSH_I64 0") < 0) return -1;
        return emit_line(cc, 1, "RET");
    }
    {
        uint8_t t = infer_expr(cc, NULL, exprs[nexpr - 1]);
        fn->result_tag = (t == 0xFF) ? TAG_INT : t;
    }
    for (i = 0; i < nexpr - 1; i++) {
        if (compile_expr(cc, &env, exprs[i], 0) < 0) return -1;
        if (emit_line(cc, 1, "POP") < 0) return -1;
    }
    return compile_expr(cc, &env, exprs[nexpr - 1], 1);
}

static int build_asm(Cc *cc, Buf *out) {
    int i;
    if (buf_printf(out, ".flag has_main\n.entry _sc_main\n") < 0) return -1;
    for (i = 0; i < cc->nfn; i++) {
        ScFn *fn = &cc->fns[i];
        uint32_t locals = (uint32_t)fn->arity;
        if (buf_printf(out, ".function %s %d %u %d %s 1\n",
                       fn->asm_name, fn->arity, locals, fn->nup,
                       result_tag_name(fn->result_tag ? fn->result_tag : TAG_INT)) < 0)
            return -1;
        if (fn->code.p && buf_printf(out, "%s", fn->code.p) < 0) return -1;
        if (buf_printf(out, ".end\n") < 0) return -1;
    }
    return 0;
}

static int prepare(Cc *cc, const char *src) {
    int i, nexpr = 0;
    ScObj *exprs[SC_MAX_FORMS];

    memset(cc, 0, sizeof *cc);
    cc->cur = -1;
    if (parse_program(cc, src) < 0) return -1;
    for (i = 0; i < cc->nforms; i++) {
        cc->forms[i] = desugar(cc, cc->forms[i]);
        if (!cc->forms[i]) return cc->err[0] ? -1 : cc_fail(cc, "desugar failed");
    }
    for (i = 0; i < cc->nforms; i++) {
        if (is_pair(cc->forms[i]) && sym_is(car(cc->forms[i]), "define")) {
            {
                ScObj *spec = nth(cc->forms[i], 1);
                const char *nm = NULL;
                int g;
                if (is_sym(spec)) nm = spec->u.sym;
                else if (is_pair(spec) && is_sym(car(spec))) nm = car(spec)->u.sym;
                if (nm && (g = global_fn(cc, nm)) >= 0) {
                    cc->fns[g].scheme_name[0] = '\0';
                    snprintf(cc->fns[g].asm_name, sizeof cc->fns[g].asm_name,
                             "_dead%d", g);
                }
            }
            if (register_fn(cc, cc->forms[i], NULL, 1, NULL) < 0) return -1;
        } else {
            exprs[nexpr++] = cc->forms[i];
        }
    }
    for (i = 0; i < nexpr; i++) {
        if (walk_nested(cc, exprs[i], NULL) < 0) return -1;
    }
    infer_results(cc);
    for (i = 0; i < cc->nfn; i++) {
        if (compile_function(cc, i) < 0) return -1;
    }
    return compile_main(cc, exprs, nexpr);
}

static void attach_debug(NvmModule *mod) {
    if (!mod) return;
    mod->header.flags |= NVM_FLAG_DEBUG_INFO;
    nvm_add_debug_entry(mod, 0, 1, 1);
}

NvmModule *nl_scheme_compile(const char *src, const char *path,
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
    if (getenv("NL_SCHEME_TRACE")) fprintf(stderr, "%s", asmbuf.p ? asmbuf.p : "");
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

NlFrontendResult nl_scheme_accept(const NvmModule *mod, const char *path) {
    NlFrontendFacts f;
    memset(&f, 0, sizeof f);
    f.language = NL_FE_SCHEME;
    f.source_path = (path && path[0]) ? path : "<scheme>";
    f.purity = -1;
    f.exhaustiveness = -1;
    f.affine_use = -1;
    f.diagnostics_shared = 1;
    return nl_frontend_accept(mod, &f);
}

VmResult nl_scheme_execute(const NvmModule *mod, NanoValue *out,
                           uint32_t *max_frame_depth, char *err, size_t errlen) {
    VmState vm;
    VmResult r;
    NanoValue v;
    if (!mod) {
        if (err && errlen) snprintf(err, errlen, "module is null");
        return VM_ERR_TYPE_ERROR;
    }
    vm_init(&vm, mod);
    vm_profile_enable(&vm, true);
    r = vm_execute(&vm);
    if (r != VM_OK) {
        if (err && errlen)
            snprintf(err, errlen, "%s", vm_error_string(r));
        vm_destroy(&vm);
        return r;
    }
    v = vm_get_result(&vm);
    vm_retain(&vm.heap, v);
    if (out) *out = v;
    if (max_frame_depth) *max_frame_depth = vm.profile.max_frame_depth;
    vm_destroy(&vm);
    return VM_OK;
}

int nl_scheme_eval_i64(const char *src, int64_t *out, char *err, size_t errlen) {
    NvmModule *mod;
    NlFrontendResult acc;
    NanoValue v;
    VmResult r;
    uint32_t depth = 0;
    mod = nl_scheme_compile(src, "<eval>", err, errlen);
    if (!mod) return 0;
    acc = nl_scheme_accept(mod, "<eval>");
    if (!acc.ok) {
        if (err && errlen) snprintf(err, errlen, "%s", acc.error);
        nvm_module_free(mod);
        return 0;
    }
    memset(&v, 0, sizeof v);
    r = nl_scheme_execute(mod, &v, &depth, err, errlen);
    nvm_module_free(mod);
    if (r != VM_OK) return 0;
    if (v.tag != TAG_INT) {
        if (err && errlen) snprintf(err, errlen, "result is %s, not int", isa_tag_name(v.tag));
        return 0;
    }
    if (out) *out = v.as.i64;
    return 1;
}

static char *join_session(NlScheme *s, const char *extra) {
    Buf b;
    uint32_t i;
    memset(&b, 0, sizeof b);
    for (i = 0; i < s->ndef; i++) {
        if (buf_printf(&b, "%s\n", s->defs[i].src) < 0) {
            free(b.p);
            return NULL;
        }
    }
    if (extra && extra[0] && buf_printf(&b, "%s\n", extra) < 0) {
        free(b.p);
        return NULL;
    }
    if (!b.p) {
        b.p = strdup("");
    }
    return b.p;
}

static int extract_define_name(const char *src, char *name, size_t nlen) {
    Cc cc;
    ScObj *form, *spec;
    memset(&cc, 0, sizeof cc);
    if (parse_program(&cc, src) < 0 || cc.nforms != 1) {
        cc_free(&cc);
        return 0;
    }
    form = cc.forms[0];
    if (!is_pair(form) || !sym_is(car(form), "define")) {
        cc_free(&cc);
        return 0;
    }
    spec = nth(form, 1);
    if (is_sym(spec)) snprintf(name, nlen, "%s", spec->u.sym);
    else if (is_pair(spec) && is_sym(car(spec))) snprintf(name, nlen, "%s", car(spec)->u.sym);
    else {
        cc_free(&cc);
        return 0;
    }
    cc_free(&cc);
    return 1;
}

static int session_upsert(NlScheme *s, const char *src) {
    char name[SC_NAME];
    uint32_t i;
    char *copy;
    if (!extract_define_name(src, name, sizeof name)) return 0;
    copy = strdup(src);
    if (!copy) return -1;
    for (i = 0; i < s->ndef; i++) {
        if (strcmp(s->defs[i].name, name) == 0) {
            free(s->defs[i].src);
            s->defs[i].src = copy;
            return 1;
        }
    }
    if (s->ndef >= 64) {
        free(copy);
        return -1;
    }
    snprintf(s->defs[s->ndef].name, SC_NAME, "%s", name);
    s->defs[s->ndef].src = copy;
    s->ndef++;
    return 1;
}

NlScheme *nl_scheme_open(void) {
    return calloc(1, sizeof(NlScheme));
}

void nl_scheme_close(NlScheme *session) {
    uint32_t i;
    if (!session) return;
    for (i = 0; i < session->ndef; i++) free(session->defs[i].src);
    free(session);
}

int nl_scheme_eval(NlScheme *session, const char *src, NanoValue *out,
                   char *err, size_t errlen) {
    char *joined;
    NvmModule *mod;
    NlFrontendResult acc;
    VmResult r;
    uint32_t depth = 0;
    int stored;
    if (!session || !src) {
        if (err && errlen) snprintf(err, errlen, "session or source is null");
        return 0;
    }
    stored = session_upsert(session, src);
    if (stored < 0) {
        if (err && errlen) snprintf(err, errlen, "I cannot store that definition");
        return 0;
    }
    joined = join_session(session, stored ? NULL : src);
    if (!joined) {
        if (err && errlen) snprintf(err, errlen, "I ran out of memory");
        return 0;
    }
    mod = nl_scheme_compile(joined, "<session>", err, errlen);
    free(joined);
    if (!mod) return 0;
    acc = nl_scheme_accept(mod, "<session>");
    if (!acc.ok) {
        if (err && errlen) snprintf(err, errlen, "%s", acc.error);
        nvm_module_free(mod);
        return 0;
    }
    r = nl_scheme_execute(mod, out, &depth, err, errlen);
    nvm_module_free(mod);
    return r == VM_OK;
}

int nl_scheme_eval_i64_session(NlScheme *session, const char *src, int64_t *out,
                               char *err, size_t errlen) {
    NanoValue v;
    memset(&v, 0, sizeof v);
    if (!nl_scheme_eval(session, src, &v, err, errlen)) return 0;
    if (v.tag != TAG_INT) {
        if (err && errlen) snprintf(err, errlen, "result is %s, not int", isa_tag_name(v.tag));
        return 0;
    }
    if (out) *out = v.as.i64;
    return 1;
}
