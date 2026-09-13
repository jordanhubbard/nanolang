/*
 * Nano Logic — bounded laboratory frontend. I emit verified NanoISA
 * unification and evaluate Datalog in the host. See docs/LOGIC.md.
 */

#include "logic.h"

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

#define LG_NAME 64
#define LG_MAX 24
#define LG_ERR NL_LG_ERR_SIZE
#define LG_ARITY 3
#define LG_TUP 48
#define LG_BODY 6
#define LG_REL 12
#define LG_RULE 16

typedef enum {
    TK_EOF, TK_INT, TK_ID,
    TK_COMMA, TK_TURN,
    TK_FACT, TK_RULE, TK_QUERY
} TkKind;

typedef struct {
    TkKind kind;
    int64_t i;
    char name[LG_NAME];
} Tok;

typedef struct {
    int is_var;
    int64_t i;
    char name[LG_NAME];
} Term;

typedef struct {
    char pred[LG_NAME];
    Term terms[LG_ARITY];
    int arity;
} Atom;

typedef struct {
    Atom head;
    Atom body[LG_BODY];
    int nbody;
} Rule;

typedef struct {
    char name[LG_NAME];
    int arity;
    int64_t tups[LG_TUP][LG_ARITY];
    int ntup;
} Rel;

typedef struct {
    char *p;
    size_t n, cap;
} Buf;

typedef struct {
    const char *src;
    const char *lx;
    Tok tok;
    char err[LG_ERR];
    Rel rels[LG_REL];
    int nrel;
    Rule rules[LG_RULE];
    int nrule;
    Atom query;
    int has_query;
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
    if (*p == ':' && p[1] == '-') {
        cc->lx = p + 2;
        cc->tok.kind = TK_TURN;
        return;
    }
    if (*p == ',') {
        cc->lx = p + 1;
        cc->tok.kind = TK_COMMA;
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
            if (n + 1 < LG_NAME) cc->tok.name[n++] = *p;
            p++;
        }
        cc->tok.name[n] = '\0';
        cc->lx = p;
        if (kw_eq(cc->tok.name, "fact")) cc->tok.kind = TK_FACT;
        else if (kw_eq(cc->tok.name, "rule")) cc->tok.kind = TK_RULE;
        else if (kw_eq(cc->tok.name, "query")) cc->tok.kind = TK_QUERY;
        else cc->tok.kind = TK_ID;
        return;
    }
    snprintf(cc->err, sizeof cc->err, "I do not know '%c'", *p);
    cc->lx = p + 1;
    cc->tok.kind = TK_EOF;
}

static int have(Cc *cc, TkKind k) { return cc->tok.kind == k; }

static int eat(Cc *cc, TkKind k) {
    if (!have(cc, k)) return 0;
    lex(cc);
    return 1;
}

static int parse_term(Cc *cc, Term *t) {
    memset(t, 0, sizeof *t);
    if (have(cc, TK_INT)) {
        t->is_var = 0;
        t->i = cc->tok.i;
        lex(cc);
        return 0;
    }
    if (have(cc, TK_ID)) {
        t->is_var = 1;
        snprintf(t->name, sizeof t->name, "%s", cc->tok.name);
        lex(cc);
        return 0;
    }
    return cc_fail(cc, "I expected a term");
}

static int parse_atom_named(Cc *cc, Atom *a, const char *name) {
    memset(a, 0, sizeof *a);
    snprintf(a->pred, sizeof a->pred, "%s", name);
    while (have(cc, TK_INT) || have(cc, TK_ID)) {
        if (a->arity >= LG_ARITY) return cc_fail(cc, "I refuse arity above 3");
        if (parse_term(cc, &a->terms[a->arity]) < 0) return -1;
        a->arity++;
    }
    if (a->arity < 1) return cc_fail(cc, "I expected at least one term");
    return 0;
}

static int parse_atom(Cc *cc, Atom *a) {
    char name[LG_NAME];
    if (!have(cc, TK_ID)) return cc_fail(cc, "I expected a predicate");
    snprintf(name, sizeof name, "%s", cc->tok.name);
    lex(cc);
    return parse_atom_named(cc, a, name);
}

static int rel_lookup(Cc *cc, const char *n, int arity) {
    int i;
    for (i = 0; i < cc->nrel; i++) {
        if (strcmp(cc->rels[i].name, n) == 0) {
            if (cc->rels[i].arity != arity)
                return cc_fail(cc, "arity of %s does not match", n);
            return i;
        }
    }
    if (cc->nrel >= LG_REL) return cc_fail(cc, "I refuse too many relations");
    snprintf(cc->rels[cc->nrel].name, LG_NAME, "%s", n);
    cc->rels[cc->nrel].arity = arity;
    cc->rels[cc->nrel].ntup = 0;
    cc->nrel++;
    return cc->nrel - 1;
}

static int tup_cmp(const int64_t *a, const int64_t *b, int arity) {
    int i;
    for (i = 0; i < arity; i++) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

static int rel_add(Rel *r, const int64_t *t) {
    int lo = 0, hi = r->ntup, mid, c;
    while (lo < hi) {
        mid = (lo + hi) / 2;
        c = tup_cmp(r->tups[mid], t, r->arity);
        if (c == 0) return 0;
        if (c < 0) lo = mid + 1;
        else hi = mid;
    }
    if (r->ntup >= LG_TUP) return -2;
    memmove(r->tups[lo + 1], r->tups[lo],
            (size_t)(r->ntup - lo) * sizeof r->tups[0]);
    memcpy(r->tups[lo], t, (size_t)r->arity * sizeof(int64_t));
    r->ntup++;
    return 1;
}

static int parse_program(Cc *cc) {
    cc->lx = cc->src;
    lex(cc);
    if (cc->err[0]) return -1;
    while (!have(cc, TK_EOF)) {
        if (eat(cc, TK_FACT)) {
            Atom a;
            int rid, i;
            int64_t t[LG_ARITY];
            if (parse_atom(cc, &a) < 0) return -1;
            for (i = 0; i < a.arity; i++) {
                if (a.terms[i].is_var)
                    return cc_fail(cc, "facts must be ground");
                t[i] = a.terms[i].i;
            }
            rid = rel_lookup(cc, a.pred, a.arity);
            if (rid < 0) return -1;
            if (rel_add(&cc->rels[rid], t) == -2)
                return cc_fail(cc, "I refuse too many tuples");
            continue;
        }
        if (eat(cc, TK_RULE)) {
            Rule *r;
            if (cc->nrule >= LG_RULE) return cc_fail(cc, "I refuse too many rules");
            r = &cc->rules[cc->nrule];
            memset(r, 0, sizeof *r);
            if (parse_atom(cc, &r->head) < 0) return -1;
            if (!eat(cc, TK_TURN)) return cc_fail(cc, "I expected ':-'");
            if (parse_atom(cc, &r->body[0]) < 0) return -1;
            r->nbody = 1;
            while (eat(cc, TK_COMMA)) {
                if (r->nbody >= LG_BODY) return cc_fail(cc, "I refuse too many body atoms");
                if (parse_atom(cc, &r->body[r->nbody]) < 0) return -1;
                r->nbody++;
            }
            if (rel_lookup(cc, r->head.pred, r->head.arity) < 0) return -1;
            cc->nrule++;
            continue;
        }
        if (eat(cc, TK_QUERY)) {
            int i;
            if (cc->has_query) return cc_fail(cc, "I refuse a second query");
            if (parse_atom(cc, &cc->query) < 0) return -1;
            for (i = 0; i < cc->query.arity; i++) {
                if (cc->query.terms[i].is_var)
                    return cc_fail(cc, "queries must be ground");
            }
            cc->has_query = 1;
            continue;
        }
        return cc_fail(cc, "I expected fact, rule, or query");
    }
    if (!cc->has_query) return cc_fail(cc, "I expected a query");
    return 0;
}

static int build_asm(Cc *cc, Buf *out) {
    int i, j, k;
    if (buf_printf(out, ".flag has_main\n.entry _lg_main\n") < 0) return -1;
    for (i = 0; i < cc->nrel; i++) {
        Rel *r = &cc->rels[i];
        for (j = 0; j < r->ntup; j++) {
            if (buf_printf(out, ".string \"fact %s", r->name) < 0) return -1;
            for (k = 0; k < r->arity; k++) {
                if (buf_printf(out, " %lld", (long long)r->tups[j][k]) < 0)
                    return -1;
            }
            if (buf_printf(out, "\"\n") < 0) return -1;
        }
    }
    if (buf_printf(out,
                   ".function lg_unify 2 6 0 bool 1\n"
                   "  LOAD_LOCAL 0\n"
                   "  LOAD_LOCAL 1\n"
                   "  I64_EQ\n"
                   "  RET\n"
                   ".end\n"
                   ".function _lg_main 0 2 0 int 1\n"
                   "  PUSH_I64 0\n"
                   "  RET\n"
                   ".end\n") < 0)
        return -1;
    return 0;
}

typedef struct {
    char name[LG_NAME];
    int64_t v;
    int used;
} Bind;

static int bind_lookup(Bind *b, int n, const char *name, int64_t *out) {
    int i;
    for (i = 0; i < n; i++) {
        if (b[i].used && strcmp(b[i].name, name) == 0) {
            *out = b[i].v;
            return 1;
        }
    }
    return 0;
}

static int bind_set(Bind *b, int *n, const char *name, int64_t v) {
    int64_t old;
    if (bind_lookup(b, *n, name, &old)) return old == v;
    if (*n >= 32) return 0;
    snprintf(b[*n].name, LG_NAME, "%s", name);
    b[*n].v = v;
    b[*n].used = 1;
    (*n)++;
    return 1;
}

static int unify_vm(VmState *vm, int fn, int64_t a, int64_t b) {
    NanoValue av[2];
    NanoValue ret;
    VmResult r;
    int ok;
    av[0] = val_int(a);
    av[1] = val_int(b);
    memset(&ret, 0, sizeof ret);
    r = vm_invoke(vm, (uint32_t)fn, av, 2, &ret);
    if (r != VM_OK) return -1;
    ok = (ret.tag == TAG_BOOL && ret.as.boolean) ||
         (ret.tag == TAG_INT && ret.as.i64 != 0);
    vm_release(&vm->heap, ret);
    return ok ? 1 : 0;
}

static int match_atom(Cc *cc, VmState *vm, int fn, const Atom *atom,
                      const int64_t *tup, Bind *binds, int *nb) {
    int i;
    (void)cc;
    for (i = 0; i < atom->arity; i++) {
        if (atom->terms[i].is_var) {
            if (!bind_set(binds, nb, atom->terms[i].name, tup[i])) return 0;
        } else {
            int u = unify_vm(vm, fn, atom->terms[i].i, tup[i]);
            if (u < 0) return -1;
            if (!u) return 0;
        }
    }
    return 1;
}

static int rel_find(Cc *cc, const char *n) {
    int i;
    for (i = 0; i < cc->nrel; i++) {
        if (strcmp(cc->rels[i].name, n) == 0) return i;
    }
    return -1;
}

static int fire_rule(Cc *cc, VmState *vm, int fn, Rule *rule, int *added) {
    int idx[LG_BODY];
    int pos[LG_BODY];
    int snap[LG_BODY];
    int64_t pending[LG_TUP][LG_ARITY];
    int npend = 0;
    int i, p, hid;
    *added = 0;
    for (i = 0; i < rule->nbody; i++) {
        idx[i] = rel_find(cc, rule->body[i].pred);
        if (idx[i] < 0) return cc_fail(cc, "I do not know %s", rule->body[i].pred);
        if (cc->rels[idx[i]].arity != rule->body[i].arity)
            return cc_fail(cc, "arity of %s does not match", rule->body[i].pred);
        snap[i] = cc->rels[idx[i]].ntup;
        if (snap[i] == 0) return 0;
        pos[i] = 0;
    }
    hid = rel_find(cc, rule->head.pred);
    if (hid < 0) return cc_fail(cc, "I do not know %s", rule->head.pred);
    for (;;) {
        Bind binds[32];
        int nb = 0, ok = 1, k;
        int64_t head[LG_ARITY];
        memset(binds, 0, sizeof binds);
        for (i = 0; i < rule->nbody && ok; i++) {
            int m = match_atom(cc, vm, fn, &rule->body[i],
                               cc->rels[idx[i]].tups[pos[i]], binds, &nb);
            if (m < 0) return -1;
            if (!m) ok = 0;
        }
        if (ok) {
            for (k = 0; k < rule->head.arity; k++) {
                Term *t = &rule->head.terms[k];
                if (t->is_var) {
                    if (!bind_lookup(binds, nb, t->name, &head[k]))
                        return cc_fail(cc, "unbound %s", t->name);
                } else {
                    head[k] = t->i;
                }
            }
            if (npend >= LG_TUP) return cc_fail(cc, "I refuse an unbounded derivation");
            memcpy(pending[npend], head, sizeof head);
            npend++;
        }
        for (i = rule->nbody - 1; i >= 0; i--) {
            pos[i]++;
            if (pos[i] < snap[i]) break;
            pos[i] = 0;
        }
        if (i < 0) break;
    }
    for (p = 0; p < npend; p++) {
        int k = rel_add(&cc->rels[hid], pending[p]);
        if (k == -2) return cc_fail(cc, "I refuse an unbounded derivation");
        if (k == 1) *added = 1;
    }
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

static int ask_query(Cc *cc, VmState *vm, int fn, int64_t *out) {
    int rid = rel_find(cc, cc->query.pred);
    int i, k;
    int64_t want[LG_ARITY];
    if (rid < 0) {
        *out = 0;
        return 0;
    }
    if (cc->rels[rid].arity != cc->query.arity)
        return cc_fail(cc, "query arity does not match");
    for (k = 0; k < cc->query.arity; k++) want[k] = cc->query.terms[k].i;
    for (i = 0; i < cc->rels[rid].ntup; i++) {
        int ok = 1;
        for (k = 0; k < cc->query.arity && ok; k++) {
            int u = unify_vm(vm, fn, want[k], cc->rels[rid].tups[i][k]);
            if (u < 0) return -1;
            if (!u) ok = 0;
        }
        if (ok) { *out = 1; return 0; }
    }
    *out = 0;
    return 0;
}

static int run_rt(Cc *cc, NvmModule *mod, int64_t *out) {
    VmState vm;
    int fn, guard, r, added, any;
    fn = lookup_fn(mod, "lg_unify");
    if (fn < 0) return cc_fail(cc, "I lost lg_unify");
    vm_init(&vm, mod);
    for (guard = 0; guard < LG_TUP + 2; guard++) {
        any = 0;
        for (r = 0; r < cc->nrule; r++) {
            if (fire_rule(cc, &vm, fn, &cc->rules[r], &added) < 0) {
                vm_destroy(&vm);
                return -1;
            }
            if (added) any = 1;
        }
        if (!any) break;
    }
    if (guard >= LG_TUP + 2) {
        vm_destroy(&vm);
        return cc_fail(cc, "I refuse an unbounded derivation");
    }
    if (ask_query(cc, &vm, fn, out) < 0) {
        vm_destroy(&vm);
        return -1;
    }
    vm_destroy(&vm);
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
    return parse_program(cc);
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
    if (getenv("NL_LG_TRACE")) fprintf(stderr, "%s", asmbuf.p ? asmbuf.p : "");
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

NvmModule *nl_logic_compile(const char *src, const char *path,
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
        return NULL;
    }
    mod = assemble_cc(&cc, err, errlen);
    return mod;
}

NlFrontendResult nl_logic_accept(const NvmModule *mod, const char *path) {
    NlFrontendFacts f;
    memset(&f, 0, sizeof f);
    f.language = NL_FE_LOGIC;
    f.source_path = (path && path[0]) ? path : "<logic>";
    f.purity = 1;
    f.exhaustiveness = 0;
    f.affine_use = -1;
    f.diagnostics_shared = 1;
    return nl_frontend_accept(mod, &f);
}

int nl_logic_eval_i64(const char *src, int64_t *out,
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
        return 0;
    }
    mod = assemble_cc(&cc, err, errlen);
    if (!mod) return 0;
    acc = nl_logic_accept(mod, "<eval>");
    if (!acc.ok) {
        if (err && errlen) snprintf(err, errlen, "%s", acc.error);
        nvm_module_free(mod);
        return 0;
    }
    if (run_rt(&cc, mod, &v) < 0) {
        if (err && errlen) snprintf(err, errlen, "%s", cc.err[0] ? cc.err : "run failed");
        nvm_module_free(mod);
        return 0;
    }
    if (out) *out = v;
    nvm_module_free(mod);
    return 1;
}
