/*
 * peg2.c — Packrat PEG Parser Engine for NanoLang
 *
 * Implements a full PEG grammar engine with:
 *   - O(n) Packrat memoization
 *   - Named rules and rule references
 *   - Named captures (name:expr)
 *   - Ordered choice (e1 / e2)
 *   - Sequence (e1 e2)
 *   - Repetition (e*, e+, e?)
 *   - Lookahead (!e, &e)
 *   - Character classes ([a-z], [^abc])
 *   - String literals ("text")
 *   - Any-char (.)
 *   - Furthest-failure error reporting
 *
 * Grammar format:
 *   rule_name <- expr
 *   rule_name <- expr
 *   ...
 * The first rule is the start rule.
 *
 * Operator precedence (high to low):
 *   1. Atoms: "lit", [cls], ., (e), rule_ref
 *   2. Postfix: e*, e+, e?
 *   3. Prefix: !e, &e, name:e
 *   4. Sequence: e1 e2  (space-separated)
 *   5. Choice: e1 / e2
 */

#include "peg2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>

/* ============================================================
 * Expression AST
 * ============================================================ */

typedef enum {
    EXPR_LITERAL,   /* "text" */
    EXPR_CLASS,     /* [a-z0-9] or [^...] */
    EXPR_DOT,       /* . */
    EXPR_SEQ,       /* e1 e2 ... */
    EXPR_CHOICE,    /* e1 / e2 / ... */
    EXPR_STAR,      /* e* */
    EXPR_PLUS,      /* e+ */
    EXPR_OPT,       /* e? */
    EXPR_NOT,       /* !e */
    EXPR_AND,       /* &e */
    EXPR_RULE_REF,  /* rule_name */
    EXPR_CAPTURE,   /* name:e */
    EXPR_GROUP      /* (e) */
} ExprType;

typedef struct Expr {
    ExprType type;
    /* LITERAL */
    char *literal;
    int literal_len;
    /* CLASS */
    unsigned char class_bits[32]; /* 256-bit set for char membership */
    int class_negated;
    /* SEQ / CHOICE: children array */
    struct Expr **children;
    int child_count;
    int child_cap;
    /* STAR/PLUS/OPT/NOT/AND/GROUP/CAPTURE: single operand */
    struct Expr *operand;
    /* RULE_REF */
    char *rule_name;
    int rule_index; /* resolved after parsing all rules */
    /* CAPTURE */
    char *capture_name;
} Expr;

typedef struct {
    char *name;
    Expr *expr;
} Rule;

struct Peg2Grammar {
    Rule *rules;
    int rule_count;
    int rule_cap;
};

/* ============================================================
 * Expression construction helpers
 * ============================================================ */

static Expr *expr_new(ExprType type) {
    Expr *e = calloc(1, sizeof(Expr));
    e->type = type;
    e->rule_index = -1;
    return e;
}

static void expr_add_child(Expr *parent, Expr *child) {
    if (parent->child_count >= parent->child_cap) {
        parent->child_cap = parent->child_cap ? parent->child_cap * 2 : 4;
        parent->children = realloc(parent->children,
                                   parent->child_cap * sizeof(Expr *));
    }
    parent->children[parent->child_count++] = child;
}

static void expr_free(Expr *e) {
    if (!e) return;
    free(e->literal);
    free(e->rule_name);
    free(e->capture_name);
    for (int i = 0; i < e->child_count; i++) expr_free(e->children[i]);
    free(e->children);
    expr_free(e->operand);
    free(e);
}

/* ============================================================
 * Grammar string parser
 * ============================================================ */

typedef struct {
    const char *src;
    int pos;
    int len;
    char *error;
    /* Rule names for forward reference resolution */
    char **rule_names;
    int rule_name_count;
} GParser;

static void gp_error(GParser *p, const char *fmt, ...) {
    if (p->error) return; /* keep first error */
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    p->error = strdup(buf);
}

static void gp_skip_spaces(GParser *p) {
    while (p->pos < p->len &&
           (p->src[p->pos] == ' ' || p->src[p->pos] == '\t'))
        p->pos++;
}

static int gp_at_end(GParser *p) { return p->pos >= p->len; }
static char gp_peek(GParser *p) { return p->pos < p->len ? p->src[p->pos] : 0; }
static char gp_next(GParser *p) { return p->pos < p->len ? p->src[p->pos++] : 0; }

/* Parse a string literal "..." into an EXPR_LITERAL */
static Expr *gp_parse_literal(GParser *p) {
    /* p->pos points at '"' */
    p->pos++; /* consume '"' */
    int start = p->pos;
    int len = 0;
    char buf[1024];
    while (p->pos < p->len && p->src[p->pos] != '"') {
        if (len >= (int)sizeof(buf) - 2) { gp_error(p, "literal too long"); return NULL; }
        if (p->src[p->pos] == '\\') {
            p->pos++;
            char c = gp_next(p);
            switch (c) {
                case 'n': buf[len++] = '\n'; break;
                case 't': buf[len++] = '\t'; break;
                case 'r': buf[len++] = '\r'; break;
                case '"': buf[len++] = '"'; break;
                case '\\': buf[len++] = '\\'; break;
                default: buf[len++] = c; break;
            }
        } else {
            buf[len++] = gp_next(p);
        }
    }
    if (gp_at_end(p)) { gp_error(p, "unterminated string literal"); return NULL; }
    p->pos++; /* consume closing '"' */
    buf[len] = '\0';
    Expr *e = expr_new(EXPR_LITERAL);
    e->literal = strndup(buf, len);
    e->literal_len = len;
    return e;
}

/* Set a character in a class bit array */
static void class_set(unsigned char *bits, unsigned char c) {
    bits[c / 8] |= (1u << (c % 8));
}
static void class_set_range(unsigned char *bits, unsigned char lo, unsigned char hi) {
    for (int c = lo; c <= hi; c++) class_set(bits, (unsigned char)c);
}
static int class_test(const unsigned char *bits, unsigned char c) {
    return (bits[c / 8] >> (c % 8)) & 1;
}

/* Parse a character class [...] into EXPR_CLASS */
static Expr *gp_parse_class(GParser *p) {
    p->pos++; /* consume '[' */
    Expr *e = expr_new(EXPR_CLASS);
    if (gp_peek(p) == '^') { e->class_negated = 1; p->pos++; }
    /* First char can be ']' without closing */
    while (!gp_at_end(p) && gp_peek(p) != ']') {
        unsigned char c;
        if (p->src[p->pos] == '\\') {
            p->pos++;
            char esc = gp_next(p);
            switch (esc) {
                case 'n': c = '\n'; break;
                case 't': c = '\t'; break;
                case 'r': c = '\r'; break;
                case 's': /* \s = whitespace */
                    class_set(e->class_bits, ' ');
                    class_set(e->class_bits, '\t');
                    class_set(e->class_bits, '\n');
                    class_set(e->class_bits, '\r');
                    continue;
                case 'd': /* \d = digits */
                    class_set_range(e->class_bits, '0', '9');
                    continue;
                case 'w': /* \w = word chars */
                    class_set_range(e->class_bits, 'a', 'z');
                    class_set_range(e->class_bits, 'A', 'Z');
                    class_set_range(e->class_bits, '0', '9');
                    class_set(e->class_bits, '_');
                    continue;
                default: c = (unsigned char)esc; break;
            }
        } else {
            c = (unsigned char)gp_next(p);
        }
        /* Check for range a-z */
        if (!gp_at_end(p) && p->src[p->pos] == '-' &&
            p->pos + 1 < p->len && p->src[p->pos + 1] != ']') {
            p->pos++; /* consume '-' */
            unsigned char hi;
            if (p->src[p->pos] == '\\') {
                p->pos++;
                hi = (unsigned char)gp_next(p);
            } else {
                hi = (unsigned char)gp_next(p);
            }
            class_set_range(e->class_bits, c, hi);
        } else {
            class_set(e->class_bits, c);
        }
    }
    if (gp_at_end(p)) { gp_error(p, "unterminated character class"); expr_free(e); return NULL; }
    p->pos++; /* consume ']' */
    return e;
}

static int is_rule_start_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}
static int is_rule_char(char c) {
    return is_rule_start_char(c) || (c >= '0' && c <= '9');
}

static int gp_rule_index(GParser *p, const char *name) {
    for (int i = 0; i < p->rule_name_count; i++)
        if (strcmp(p->rule_names[i], name) == 0) return i;
    return -1;
}

/* Forward declaration */
static Expr *gp_parse_choice(GParser *p);

/* Parse an atom: literal, class, dot, rule_ref, or (group) */
static Expr *gp_parse_atom(GParser *p) {
    gp_skip_spaces(p);
    if (gp_at_end(p)) return NULL;

    char c = gp_peek(p);

    if (c == '"') return gp_parse_literal(p);
    if (c == '[') return gp_parse_class(p);
    if (c == '.') { p->pos++; return expr_new(EXPR_DOT); }

    if (c == '(') {
        p->pos++;
        Expr *inner = gp_parse_choice(p);
        if (!inner) return NULL;
        gp_skip_spaces(p);
        if (gp_peek(p) != ')') { gp_error(p, "expected ')'"); expr_free(inner); return NULL; }
        p->pos++;
        Expr *g = expr_new(EXPR_GROUP);
        g->operand = inner;
        return g;
    }

    if (is_rule_start_char(c)) {
        int start = p->pos;
        while (p->pos < p->len && is_rule_char(p->src[p->pos])) p->pos++;
        int end = p->pos;
        /* Check it's not '<-' (rule definition arrow) */
        int saved = p->pos;
        gp_skip_spaces(p);
        if (p->pos + 1 < p->len && p->src[p->pos] == '<' && p->src[p->pos+1] == '-') {
            /* This is a rule definition start, not an atom — backtrack */
            p->pos = start;
            return NULL;
        }
        p->pos = saved;

        /* Check for name:expr capture syntax */
        if (p->pos < p->len && p->src[p->pos] == ':') {
            char name[256];
            int namelen = end - start;
            if (namelen >= (int)sizeof(name)) { gp_error(p, "capture name too long"); return NULL; }
            memcpy(name, p->src + start, namelen);
            name[namelen] = '\0';
            p->pos++; /* consume ':' */
            Expr *inner = gp_parse_atom(p);
            if (!inner) { gp_error(p, "expected expression after capture '%s:'", name); return NULL; }
            /* Wrap in repetition/prefix first, then capture */
            Expr *cap = expr_new(EXPR_CAPTURE);
            cap->capture_name = strdup(name);
            cap->operand = inner;
            return cap;
        }

        char rname[256];
        int rlen = end - start;
        if (rlen >= (int)sizeof(rname)) { gp_error(p, "rule name too long"); return NULL; }
        memcpy(rname, p->src + start, rlen);
        rname[rlen] = '\0';

        Expr *e = expr_new(EXPR_RULE_REF);
        e->rule_name = strdup(rname);
        e->rule_index = gp_rule_index(p, rname); /* may be -1 for forward refs */
        return e;
    }

    return NULL;
}

/* Parse a postfix expression: atom (*|+|?)* */
static Expr *gp_parse_postfix(GParser *p) {
    Expr *base = gp_parse_atom(p);
    if (!base) return NULL;
    for (;;) {
        /* No space before postfix operator */
        if (p->pos >= p->len) break;
        char c = p->src[p->pos];
        ExprType t;
        if (c == '*') t = EXPR_STAR;
        else if (c == '+') t = EXPR_PLUS;
        else if (c == '?') t = EXPR_OPT;
        else break;
        p->pos++;
        Expr *e = expr_new(t);
        e->operand = base;
        base = e;
    }
    return base;
}

/* Parse a prefix expression: (!|&)* postfix */
static Expr *gp_parse_prefix(GParser *p) {
    gp_skip_spaces(p);
    if (gp_at_end(p)) return NULL;
    char c = gp_peek(p);
    if (c == '!') { p->pos++; Expr *e = expr_new(EXPR_NOT); e->operand = gp_parse_prefix(p); if (!e->operand) { expr_free(e); return NULL; } return e; }
    if (c == '&') { p->pos++; Expr *e = expr_new(EXPR_AND); e->operand = gp_parse_prefix(p); if (!e->operand) { expr_free(e); return NULL; } return e; }
    return gp_parse_postfix(p);
}

/* Check if we're at the start of a new top-level rule definition */
static int gp_at_rule_def(GParser *p) {
    int saved = p->pos;
    gp_skip_spaces(p);
    /* skip over optional newlines/whitespace to next non-blank */
    while (p->pos < p->len && (p->src[p->pos] == '\n' || p->src[p->pos] == '\r')) p->pos++;
    gp_skip_spaces(p);
    if (gp_at_end(p)) { p->pos = saved; return 1; }
    int start = p->pos;
    while (p->pos < p->len && is_rule_char(p->src[p->pos])) p->pos++;
    if (p->pos == start) { p->pos = saved; return 0; }
    gp_skip_spaces(p);
    int result = (p->pos + 1 < p->len && p->src[p->pos] == '<' && p->src[p->pos+1] == '-');
    p->pos = saved;
    return result;
}

/* Parse a sequence: prefix+ (space-separated, stops at '/' or newline or ')' or end or next rule) */
static Expr *gp_parse_seq(GParser *p) {
    Expr *first = gp_parse_prefix(p);
    if (!first) return NULL;

    /* Peek ahead: if next non-space char is '/' or ')' or '\n' or end-of-rule, no sequence */
    gp_skip_spaces(p);
    if (gp_at_end(p) || p->src[p->pos] == '/' || p->src[p->pos] == ')'
        || p->src[p->pos] == '\n' || p->src[p->pos] == '\r') {
        return first;
    }
    if (gp_at_rule_def(p)) return first;

    Expr *seq = expr_new(EXPR_SEQ);
    expr_add_child(seq, first);

    for (;;) {
        gp_skip_spaces(p);
        if (gp_at_end(p) || p->src[p->pos] == '/' || p->src[p->pos] == ')'
            || p->src[p->pos] == '\n' || p->src[p->pos] == '\r') break;
        if (gp_at_rule_def(p)) break;
        Expr *next = gp_parse_prefix(p);
        if (!next) break;
        expr_add_child(seq, next);
    }

    if (seq->child_count == 1) {
        Expr *only = seq->children[0];
        free(seq->children);
        free(seq);
        return only;
    }
    return seq;
}

/* Parse ordered choice: seq (/ seq)* */
static Expr *gp_parse_choice(GParser *p) {
    Expr *first = gp_parse_seq(p);
    if (!first) return NULL;

    gp_skip_spaces(p);
    if (gp_at_end(p) || p->src[p->pos] != '/') return first;

    Expr *choice = expr_new(EXPR_CHOICE);
    expr_add_child(choice, first);

    while (!gp_at_end(p) && p->src[p->pos] == '/') {
        p->pos++; /* consume '/' */
        Expr *alt = gp_parse_seq(p);
        if (!alt) { gp_error(p, "expected expression after '/'"); break; }
        expr_add_child(choice, alt);
        gp_skip_spaces(p);
    }

    if (choice->child_count == 1) {
        Expr *only = choice->children[0];
        free(choice->children);
        free(choice);
        return only;
    }
    return choice;
}

/* Skip blank lines and comments */
static void gp_skip_lines(GParser *p) {
    while (p->pos < p->len) {
        if (p->src[p->pos] == ' ' || p->src[p->pos] == '\t') { p->pos++; continue; }
        if (p->src[p->pos] == '\n' || p->src[p->pos] == '\r') { p->pos++; continue; }
        if (p->src[p->pos] == '#') { /* line comment */
            while (p->pos < p->len && p->src[p->pos] != '\n') p->pos++;
            continue;
        }
        break;
    }
}

/* First pass: collect rule names */
static int gp_collect_rule_names(GParser *p, char ***names_out, int *count_out) {
    int saved = p->pos;
    int cap = 8, count = 0;
    char **names = malloc(cap * sizeof(char *));

    while (!gp_at_end(p)) {
        gp_skip_lines(p);
        if (gp_at_end(p)) break;
        if (!is_rule_start_char(p->src[p->pos])) { p->pos++; continue; }
        int start = p->pos;
        while (p->pos < p->len && is_rule_char(p->src[p->pos])) p->pos++;
        int end = p->pos;
        gp_skip_spaces(p);
        if (p->pos + 1 < p->len && p->src[p->pos] == '<' && p->src[p->pos+1] == '-') {
            char *name = strndup(p->src + start, end - start);
            if (count >= cap) { cap *= 2; names = realloc(names, cap * sizeof(char *)); }
            names[count++] = name;
        }
        /* advance past this line */
        while (p->pos < p->len && p->src[p->pos] != '\n') p->pos++;
    }

    p->pos = saved;
    *names_out = names;
    *count_out = count;
    return 1;
}

/* Resolve forward references in an expression tree */
static void resolve_refs(Expr *e, char **rule_names, int rule_count) {
    if (!e) return;
    if (e->type == EXPR_RULE_REF && e->rule_index < 0) {
        for (int i = 0; i < rule_count; i++)
            if (strcmp(rule_names[i], e->rule_name) == 0) { e->rule_index = i; break; }
    }
    for (int i = 0; i < e->child_count; i++) resolve_refs(e->children[i], rule_names, rule_count);
    resolve_refs(e->operand, rule_names, rule_count);
}

/* ============================================================
 * Compile a grammar string
 * ============================================================ */

Peg2Grammar *nl_peg2_compile(const char *src, char **error_out) {
    if (error_out) *error_out = NULL;

    GParser p = { .src = src, .pos = 0, .len = (int)strlen(src) };

    /* First pass: collect all rule names for forward-reference resolution */
    gp_collect_rule_names(&p, &p.rule_names, &p.rule_name_count);

    Peg2Grammar *g = calloc(1, sizeof(Peg2Grammar));
    g->rule_cap = 8;
    g->rules = malloc(g->rule_cap * sizeof(Rule));

    while (!gp_at_end(&p)) {
        gp_skip_lines(&p);
        if (gp_at_end(&p)) break;
        if (p.error) break;

        /* Parse rule name */
        if (!is_rule_start_char(p.src[p.pos])) {
            gp_error(&p, "expected rule name at position %d", p.pos);
            break;
        }
        int name_start = p.pos;
        while (p.pos < p.len && is_rule_char(p.src[p.pos])) p.pos++;
        char *rule_name = strndup(p.src + name_start, p.pos - name_start);

        gp_skip_spaces(&p);
        if (p.pos + 1 >= p.len || p.src[p.pos] != '<' || p.src[p.pos+1] != '-') {
            gp_error(&p, "expected '<-' after rule name '%s'", rule_name);
            free(rule_name);
            break;
        }
        p.pos += 2; /* consume '<-' */
        gp_skip_spaces(&p);

        Expr *expr = gp_parse_choice(&p);
        if (!expr) {
            if (!p.error) gp_error(&p, "expected expression for rule '%s'", rule_name);
            free(rule_name);
            break;
        }

        if (g->rule_count >= g->rule_cap) {
            g->rule_cap *= 2;
            g->rules = realloc(g->rules, g->rule_cap * sizeof(Rule));
        }
        g->rules[g->rule_count].name = rule_name;
        g->rules[g->rule_count].expr = expr;
        g->rule_count++;
    }

    /* Free name list */
    for (int i = 0; i < p.rule_name_count; i++) free(p.rule_names[i]);
    free(p.rule_names);

    if (p.error) {
        if (error_out) *error_out = p.error; else free(p.error);
        nl_peg2_free(g);
        return NULL;
    }

    if (g->rule_count == 0) {
        if (error_out) *error_out = strdup("grammar has no rules");
        nl_peg2_free(g);
        return NULL;
    }

    /* Resolve all forward rule references */
    for (int i = 0; i < g->rule_count; i++)
        resolve_refs(g->rules[i].expr, /* rule names list */
                     (char **)NULL, 0);

    /* Second resolve pass: build a local name array */
    char **rnames = malloc(g->rule_count * sizeof(char *));
    for (int i = 0; i < g->rule_count; i++) rnames[i] = g->rules[i].name;
    for (int i = 0; i < g->rule_count; i++) resolve_refs(g->rules[i].expr, rnames, g->rule_count);
    free(rnames);

    return g;
}

void nl_peg2_free(Peg2Grammar *g) {
    if (!g) return;
    for (int i = 0; i < g->rule_count; i++) {
        free(g->rules[i].name);
        expr_free(g->rules[i].expr);
    }
    free(g->rules);
    free(g);
}

/* ============================================================
 * Parse tree construction
 * ============================================================ */

static Peg2Node *node_new(const char *rule, const char *cap, int start, int end) {
    Peg2Node *n = calloc(1, sizeof(Peg2Node));
    n->rule_name = rule ? strdup(rule) : strdup("");
    n->capture_name = cap ? strdup(cap) : NULL;
    n->start = start;
    n->end = end;
    return n;
}

static void node_add_child(Peg2Node *parent, Peg2Node *child) {
    if (!child) return;
    int cap = parent->child_count + 1;
    parent->children = realloc(parent->children, cap * sizeof(Peg2Node *));
    parent->children[parent->child_count++] = child;
}

void nl_peg2_node_free(Peg2Node *node) {
    if (!node) return;
    free((char *)node->rule_name);
    free((char *)node->capture_name);
    for (int i = 0; i < node->child_count; i++) nl_peg2_node_free(node->children[i]);
    free(node->children);
    free(node);
}

/* ============================================================
 * Packrat memoization
 * ============================================================ */

typedef struct {
    int valid;
    int end_pos;    /* -1 = failure, >= 0 = success */
    Peg2Node *tree;
} MemoCell;

typedef struct {
    MemoCell *cells;    /* [input_len+1][rule_count] */
    int input_len;
    int rule_count;
    /* Furthest failure tracking */
    int fail_pos;
    char fail_expected[512];
} Memo;

static MemoCell *memo_get(Memo *m, int pos, int rule) {
    return &m->cells[pos * m->rule_count + rule];
}

static void memo_set_fail(Memo *m, int pos, const char *expected) {
    if (pos > m->fail_pos) {
        m->fail_pos = pos;
        if (expected)
            snprintf(m->fail_expected, sizeof(m->fail_expected), "%s", expected);
        else
            m->fail_expected[0] = '\0';
    }
}

/* Forward declaration */
static int peg_exec(Peg2Grammar *g, const Expr *e, const char *input, int pos,
                    int input_len, Memo *m, Peg2Node **tree_out,
                    const char *rule_ctx);

/* Execute with Packrat memoization at the rule level */
static int peg_rule(Peg2Grammar *g, int rule_idx, const char *input,
                    int pos, int input_len, Memo *m, Peg2Node **tree_out) {
    if (rule_idx < 0 || rule_idx >= g->rule_count) return -1;

    MemoCell *cell = memo_get(m, pos, rule_idx);
    if (cell->valid) {
        if (tree_out) *tree_out = cell->tree ? /* share */ cell->tree : NULL;
        return cell->end_pos;
    }

    cell->valid = 1;
    cell->end_pos = -1;
    cell->tree = NULL;

    Peg2Node *subtree = NULL;
    int end = peg_exec(g, g->rules[rule_idx].expr, input, pos, input_len, m, &subtree,
                       g->rules[rule_idx].name);

    if (end >= 0) {
        Peg2Node *rnode = node_new(g->rules[rule_idx].name, NULL, pos, end);
        if (subtree) node_add_child(rnode, subtree);
        cell->end_pos = end;
        cell->tree = rnode;
    } else {
        cell->end_pos = -1;
    }

    if (tree_out) *tree_out = cell->tree;
    return cell->end_pos;
}

/* Core PEG executor */
static int peg_exec(Peg2Grammar *g, const Expr *e, const char *input, int pos,
                    int input_len, Memo *m, Peg2Node **tree_out,
                    const char *rule_ctx) {
    if (!e) return -1;
    if (tree_out) *tree_out = NULL;

    switch (e->type) {
        case EXPR_LITERAL: {
            if (pos + e->literal_len > input_len) {
                char msg[128];
                snprintf(msg, sizeof(msg), "\"%s\"", e->literal);
                memo_set_fail(m, pos, msg);
                return -1;
            }
            if (memcmp(input + pos, e->literal, e->literal_len) != 0) {
                char msg[128];
                snprintf(msg, sizeof(msg), "\"%s\"", e->literal);
                memo_set_fail(m, pos, msg);
                return -1;
            }
            return pos + e->literal_len;
        }

        case EXPR_DOT: {
            if (pos >= input_len) { memo_set_fail(m, pos, "any character"); return -1; }
            return pos + 1;
        }

        case EXPR_CLASS: {
            if (pos >= input_len) { memo_set_fail(m, pos, "character class"); return -1; }
            unsigned char c = (unsigned char)input[pos];
            int member = class_test(e->class_bits, c);
            if (e->class_negated) member = !member;
            if (!member) { memo_set_fail(m, pos, "character class"); return -1; }
            return pos + 1;
        }

        case EXPR_SEQ: {
            int cur = pos;
            Peg2Node *seq_node = node_new("", NULL, pos, pos);
            for (int i = 0; i < e->child_count; i++) {
                Peg2Node *child_tree = NULL;
                int next = peg_exec(g, e->children[i], input, cur, input_len, m,
                                    &child_tree, rule_ctx);
                if (next < 0) { nl_peg2_node_free(seq_node); return -1; }
                if (child_tree) node_add_child(seq_node, child_tree);
                cur = next;
            }
            seq_node->end = cur;
            /* Flatten single-child seq nodes */
            if (seq_node->child_count == 1 && strcmp(seq_node->rule_name, "") == 0) {
                Peg2Node *only = seq_node->children[0];
                free(seq_node->children);
                free((char*)seq_node->rule_name);
                free(seq_node);
                if (tree_out) *tree_out = only;
            } else {
                if (tree_out) *tree_out = seq_node;
                else nl_peg2_node_free(seq_node);
            }
            return cur;
        }

        case EXPR_CHOICE: {
            for (int i = 0; i < e->child_count; i++) {
                Peg2Node *child_tree = NULL;
                int end = peg_exec(g, e->children[i], input, pos, input_len, m,
                                   &child_tree, rule_ctx);
                if (end >= 0) {
                    if (tree_out) *tree_out = child_tree; else nl_peg2_node_free(child_tree);
                    return end;
                }
            }
            return -1;
        }

        case EXPR_STAR: {
            int cur = pos;
            Peg2Node *rep = node_new("", NULL, pos, pos);
            for (;;) {
                Peg2Node *child_tree = NULL;
                int next = peg_exec(g, e->operand, input, cur, input_len, m,
                                    &child_tree, rule_ctx);
                if (next < 0 || next == cur) break; /* avoid infinite loops on empty matches */
                if (child_tree) node_add_child(rep, child_tree);
                cur = next;
            }
            rep->end = cur;
            if (tree_out) *tree_out = rep; else nl_peg2_node_free(rep);
            return cur; /* always succeeds */
        }

        case EXPR_PLUS: {
            Peg2Node *child_tree = NULL;
            int first = peg_exec(g, e->operand, input, pos, input_len, m, &child_tree, rule_ctx);
            if (first < 0) return -1;
            /* Now do star from first */
            Peg2Node *rep = node_new("", NULL, pos, pos);
            if (child_tree) node_add_child(rep, child_tree);
            int cur = first;
            for (;;) {
                child_tree = NULL;
                int next = peg_exec(g, e->operand, input, cur, input_len, m,
                                    &child_tree, rule_ctx);
                if (next < 0 || next == cur) break;
                if (child_tree) node_add_child(rep, child_tree);
                cur = next;
            }
            rep->end = cur;
            if (tree_out) *tree_out = rep; else nl_peg2_node_free(rep);
            return cur;
        }

        case EXPR_OPT: {
            Peg2Node *child_tree = NULL;
            int end = peg_exec(g, e->operand, input, pos, input_len, m, &child_tree, rule_ctx);
            if (end >= 0) {
                if (tree_out) *tree_out = child_tree; else nl_peg2_node_free(child_tree);
                return end;
            }
            return pos; /* always succeeds, consuming nothing on failure */
        }

        case EXPR_NOT: {
            /* Negative lookahead: succeeds iff operand fails; consumes nothing */
            int end = peg_exec(g, e->operand, input, pos, input_len, m, NULL, rule_ctx);
            if (end >= 0) { memo_set_fail(m, pos, "negative lookahead"); return -1; }
            return pos;
        }

        case EXPR_AND: {
            /* Positive lookahead: succeeds iff operand succeeds; consumes nothing */
            int end = peg_exec(g, e->operand, input, pos, input_len, m, NULL, rule_ctx);
            if (end < 0) return -1;
            return pos;
        }

        case EXPR_RULE_REF: {
            if (e->rule_index < 0) {
                char msg[128];
                snprintf(msg, sizeof(msg), "undefined rule '%s'", e->rule_name);
                memo_set_fail(m, pos, msg);
                return -1;
            }
            return peg_rule(g, e->rule_index, input, pos, input_len, m, tree_out);
        }

        case EXPR_CAPTURE: {
            Peg2Node *inner_tree = NULL;
            int end = peg_exec(g, e->operand, input, pos, input_len, m, &inner_tree, rule_ctx);
            if (end < 0) return -1;
            Peg2Node *cap = node_new("", e->capture_name, pos, end);
            if (inner_tree) node_add_child(cap, inner_tree);
            if (tree_out) *tree_out = cap; else nl_peg2_node_free(cap);
            return end;
        }

        case EXPR_GROUP: {
            return peg_exec(g, e->operand, input, pos, input_len, m, tree_out, rule_ctx);
        }
    }
    return -1;
}

/* ============================================================
 * Public matching API
 * ============================================================ */

static Peg2Result run_match(Peg2Grammar *g, const char *input, int input_len, int start_pos) {
    Peg2Result r = {0};
    if (!g || g->rule_count == 0) {
        r.error_msg = strdup("null or empty grammar");
        return r;
    }

    Memo m = {0};
    m.input_len = input_len;
    m.rule_count = g->rule_count;
    m.fail_pos = -1;
    m.cells = calloc((size_t)(input_len + 1) * g->rule_count, sizeof(MemoCell));

    Peg2Node *tree = NULL;
    int end = peg_rule(g, 0, input, start_pos, input_len, &m, &tree);

    free(m.cells);

    if (end >= 0) {
        r.ok = 1;
        r.tree = tree;
    } else {
        /* Generate error message */
        int ep = m.fail_pos >= 0 ? m.fail_pos : start_pos;
        /* Count line/col */
        int line = 1, col = 1;
        for (int i = 0; i < ep && i < input_len; i++) {
            if (input[i] == '\n') { line++; col = 1; } else col++;
        }
        char buf[512];
        if (m.fail_expected[0])
            snprintf(buf, sizeof(buf), "parse error at line %d col %d: expected %s",
                     line, col, m.fail_expected);
        else
            snprintf(buf, sizeof(buf), "parse error at line %d col %d", line, col);
        r.error_msg = strdup(buf);
        r.error_pos = ep;
    }
    return r;
}

Peg2Result nl_peg2_parse(Peg2Grammar *g, const char *input, int input_len) {
    Peg2Result r = run_match(g, input, input_len, 0);
    /* Require full match */
    if (r.ok && r.tree && r.tree->end != input_len) {
        int end = r.tree->end;
        nl_peg2_node_free(r.tree);
        r.ok = 0;
        r.tree = NULL;
        int line = 1, col = 1;
        for (int i = 0; i < end && i < input_len; i++) {
            if (input[i] == '\n') { line++; col = 1; } else col++;
        }
        char buf[256];
        snprintf(buf, sizeof(buf), "incomplete match: stopped at line %d col %d", line, col);
        free(r.error_msg);
        r.error_msg = strdup(buf);
        r.error_pos = end;
    }
    return r;
}

Peg2Result nl_peg2_find(Peg2Grammar *g, const char *input, int input_len) {
    if (!g) { Peg2Result r = {0}; r.error_msg = strdup("null grammar"); return r; }
    for (int i = 0; i < input_len; i++) {
        Peg2Result r = run_match(g, input, input_len, i);
        if (r.ok) return r;
        free(r.error_msg);
    }
    Peg2Result fail = {0};
    fail.error_msg = strdup("no match found");
    fail.error_pos = input_len;
    return fail;
}

Peg2Node **nl_peg2_find_all(Peg2Grammar *g, const char *input, int input_len, int *count_out) {
    int cap = 8, count = 0;
    Peg2Node **results = malloc(cap * sizeof(Peg2Node *));
    int pos = 0;
    while (pos < input_len) {
        Peg2Result r = run_match(g, input, input_len, pos);
        if (!r.ok) { free(r.error_msg); pos++; continue; }
        if (count >= cap) { cap *= 2; results = realloc(results, cap * sizeof(Peg2Node *)); }
        results[count++] = r.tree;
        int next = r.tree->end;
        if (next <= pos) next = pos + 1; /* avoid infinite loop on empty match */
        pos = next;
    }
    *count_out = count;
    return results;
}

int nl_peg2_matches(Peg2Grammar *g, const char *input, int input_len) {
    Peg2Result r = nl_peg2_parse(g, input, input_len);
    int ok = r.ok;
    nl_peg2_result_free(&r);
    return ok;
}

/* ============================================================
 * Tree navigation
 * ============================================================ */

char *nl_peg2_node_text(const Peg2Node *node, const char *input) {
    if (!node) return strdup("");
    int len = node->end - node->start;
    if (len < 0) len = 0;
    return strndup(input + node->start, len);
}

Peg2Node *nl_peg2_node_get(const Peg2Node *node, const char *rule_name) {
    if (!node) return NULL;
    for (int i = 0; i < node->child_count; i++) {
        Peg2Node *c = node->children[i];
        if (c && c->rule_name && strcmp(c->rule_name, rule_name) == 0) return c;
        /* Recurse into anonymous nodes */
        if (c && c->rule_name && c->rule_name[0] == '\0') {
            Peg2Node *found = nl_peg2_node_get(c, rule_name);
            if (found) return found;
        }
    }
    return NULL;
}

Peg2Node *nl_peg2_node_capture(const Peg2Node *node, const char *cap_name) {
    if (!node) return NULL;
    for (int i = 0; i < node->child_count; i++) {
        Peg2Node *c = node->children[i];
        if (c && c->capture_name && strcmp(c->capture_name, cap_name) == 0) return c;
        Peg2Node *found = nl_peg2_node_capture(c, cap_name);
        if (found) return found;
    }
    return NULL;
}

/* ============================================================
 * Cleanup
 * ============================================================ */

void nl_peg2_result_free(Peg2Result *r) {
    if (!r) return;
    nl_peg2_node_free(r->tree);
    free(r->error_msg);
    r->tree = NULL;
    r->error_msg = NULL;
}

void nl_peg2_free_all(Peg2Node **nodes, int count) {
    for (int i = 0; i < count; i++) nl_peg2_node_free(nodes[i]);
    free(nodes);
}

/* ============================================================
 * NanoLang FFI entry points
 * Thread-local state holds the last parse result for tree access.
 * ============================================================ */

#include <stdint.h>

/* Thread-local last-result state */
static __thread Peg2Result tls_last_result = {0};
static __thread const Peg2Node *tls_current_node = NULL;
static __thread const char *tls_last_input = NULL;

void *nl_peg2_ffi_compile(const char *grammar_src) {
    char *err = NULL;
    Peg2Grammar *g = nl_peg2_compile(grammar_src, &err);
    if (!g) { if (err) free(err); }
    return (void *)g;
}

static void tls_set_result(Peg2Result r, const char *input) {
    nl_peg2_result_free(&tls_last_result);
    tls_last_result = r;
    tls_current_node = r.tree;
    tls_last_input = input;
}

int64_t nl_peg2_ffi_parse(void *g, const char *input) {
    int len = input ? (int)strlen(input) : 0;
    Peg2Result r = nl_peg2_parse((Peg2Grammar *)g, input, len);
    tls_set_result(r, input);
    return r.ok ? 1 : 0;
}

int64_t nl_peg2_ffi_find(void *g, const char *input) {
    int len = input ? (int)strlen(input) : 0;
    Peg2Result r = nl_peg2_find((Peg2Grammar *)g, input, len);
    tls_set_result(r, input);
    return r.ok ? 1 : 0;
}

int64_t nl_peg2_ffi_matches(void *g, const char *input) {
    int len = input ? (int)strlen(input) : 0;
    return nl_peg2_matches((Peg2Grammar *)g, input, len) ? 1 : 0;
}

void nl_peg2_ffi_free(void *g) {
    nl_peg2_free((Peg2Grammar *)g);
}

const char *nl_peg2_ffi_tree_rule(void) {
    return tls_current_node ? tls_current_node->rule_name : "";
}

const char *nl_peg2_ffi_tree_capture(void) {
    return tls_current_node ? (tls_current_node->capture_name ? tls_current_node->capture_name : "") : "";
}

int64_t nl_peg2_ffi_tree_start(void) {
    return tls_current_node ? tls_current_node->start : -1;
}

int64_t nl_peg2_ffi_tree_end(void) {
    return tls_current_node ? tls_current_node->end : -1;
}

int64_t nl_peg2_ffi_tree_child_count(void) {
    return tls_current_node ? tls_current_node->child_count : 0;
}

void *nl_peg2_ffi_tree_child(int64_t idx) {
    if (!tls_current_node || idx < 0 || idx >= tls_current_node->child_count) return NULL;
    return (void *)tls_current_node->children[idx];
}

const char *nl_peg2_ffi_tree_text(const char *input) {
    if (!tls_current_node || !input) return "";
    char *t = nl_peg2_node_text(tls_current_node, input);
    /* Store in a small static buffer — caller should use quickly */
    static __thread char text_buf[4096];
    snprintf(text_buf, sizeof(text_buf), "%s", t);
    free(t);
    return text_buf;
}

void *nl_peg2_ffi_tree_get(const char *rule_name) {
    return (void *)nl_peg2_node_get(tls_current_node, rule_name);
}

void *nl_peg2_ffi_tree_capture_get(const char *cap_name) {
    return (void *)nl_peg2_node_capture(tls_current_node, cap_name);
}

const char *nl_peg2_ffi_last_error(void) {
    return tls_last_result.error_msg ? tls_last_result.error_msg : "";
}

int64_t nl_peg2_ffi_last_error_pos(void) {
    return tls_last_result.error_pos;
}
