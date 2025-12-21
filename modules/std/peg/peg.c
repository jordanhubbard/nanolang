#include "peg.h"

#include <assert.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    N_LITERAL,
    N_DOT,
    N_CLASS,
    N_SEQ,
    N_CHOICE,
    N_STAR,
    N_PLUS,
    N_OPT,
    N_GROUP,
} NodeType;

typedef struct Node Node;

struct Node {
    NodeType type;
    union {
        struct {
            char *bytes;
            int64_t len;
        } lit;
        struct {
            bool inverted;
            uint8_t table[256];
        } cls;
        struct {
            Node **items;
            int64_t count;
        } list;
        struct {
            Node *child;
        } unary;
        struct {
            Node *child;
        } group;
    } as;
};

typedef struct {
    const char *src;
    int64_t len;
    int64_t pos;
    bool ok;
} PegParser;

typedef struct {
    int64_t start;
    int64_t end;
} CaptureSpan;

typedef struct {
    CaptureSpan *spans;
    int64_t count;
    int64_t cap;
} CaptureState;

typedef struct {
    Node *root;
} PEG;

static void cap_init(CaptureState *cs) {
    cs->spans = NULL;
    cs->count = 0;
    cs->cap = 0;
}

static void cap_free(CaptureState *cs) {
    free(cs->spans);
    cs->spans = NULL;
    cs->count = 0;
    cs->cap = 0;
}

static void cap_push(CaptureState *cs, int64_t start, int64_t end) {
    if (cs->count >= cs->cap) {
        int64_t new_cap = cs->cap == 0 ? 8 : cs->cap * 2;
        if (new_cap > INT64_MAX / (int64_t)sizeof(CaptureSpan)) {
            fprintf(stderr, "Error: PEG capture overflow\n");
            exit(1);
        }
        CaptureSpan *ns = (CaptureSpan*)realloc(cs->spans, (size_t)new_cap * sizeof(CaptureSpan));
        if (!ns) {
            fprintf(stderr, "Error: Out of memory in PEG captures\n");
            exit(1);
        }
        cs->spans = ns;
        cs->cap = new_cap;
    }
    cs->spans[cs->count++] = (CaptureSpan){.start = start, .end = end};
}

static Node *node_new(NodeType t) {
    Node *n = (Node*)calloc(1, sizeof(Node));
    if (!n) return NULL;
    n->type = t;
    return n;
}

static void node_free(Node *n) {
    if (!n) return;
    switch (n->type) {
        case N_LITERAL:
            free(n->as.lit.bytes);
            break;
        case N_SEQ:
        case N_CHOICE:
            for (int64_t i = 0; i < n->as.list.count; i++) {
                node_free(n->as.list.items[i]);
            }
            free(n->as.list.items);
            break;
        case N_STAR:
        case N_PLUS:
        case N_OPT:
            node_free(n->as.unary.child);
            break;
        case N_GROUP:
            node_free(n->as.group.child);
            break;
        case N_CLASS:
        case N_DOT:
            break;
    }
    free(n);
}

static void pp_skip_ws(PegParser *p) {
    while (p->pos < p->len) {
        char c = p->src[p->pos];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            p->pos++;
        } else {
            break;
        }
    }
}

static char pp_peek(PegParser *p) {
    if (p->pos >= p->len) return '\0';
    return p->src[p->pos];
}

static bool pp_consume(PegParser *p, char c) {
    if (pp_peek(p) == c) {
        p->pos++;
        return true;
    }
    return false;
}

static int64_t pp_read_escape(PegParser *p) {
    if (p->pos >= p->len) return -1;
    char c = p->src[p->pos++];
    switch (c) {
        case 'n': return '\n';
        case 'r': return '\r';
        case 't': return '\t';
        case '\\': return '\\';
        case '"': return '"';
        case ']': return ']';
        case '[': return '[';
        case '-': return '-';
        case '(': return '(';
        case ')': return ')';
        case '*': return '*';
        case '+': return '+';
        case '?': return '?';
        case '/': return '/';
        case '.': return '.';
        default: return (unsigned char)c;
    }
}

static Node *parse_expr(PegParser *p);

static Node *parse_literal(PegParser *p) {
    if (!pp_consume(p, '"')) return NULL;

    char *buf = NULL;
    int64_t cap = 0;
    int64_t len = 0;

    while (p->pos < p->len) {
        char c = p->src[p->pos++];
        if (c == '"') {
            Node *n = node_new(N_LITERAL);
            if (!n) {
                free(buf);
                return NULL;
            }
            n->as.lit.bytes = buf;
            n->as.lit.len = len;
            return n;
        }
        int64_t outc = (unsigned char)c;
        if (c == '\\') {
            outc = pp_read_escape(p);
            if (outc < 0) break;
        }

        if (len + 1 >= cap) {
            int64_t new_cap = cap == 0 ? 16 : cap * 2;
            char *nb = (char*)realloc(buf, (size_t)new_cap);
            if (!nb) {
                free(buf);
                return NULL;
            }
            buf = nb;
            cap = new_cap;
        }
        buf[len++] = (char)outc;
    }

    free(buf);
    p->ok = false;
    return NULL;
}

static Node *parse_class(PegParser *p) {
    if (!pp_consume(p, '[')) return NULL;
    Node *n = node_new(N_CLASS);
    if (!n) return NULL;
    memset(n->as.cls.table, 0, sizeof(n->as.cls.table));
    n->as.cls.inverted = false;

    if (pp_peek(p) == '^') {
        p->pos++;
        n->as.cls.inverted = true;
    }

    bool first = true;
    int last_ch = -1;
    bool in_range = false;

    while (p->pos < p->len) {
        char c = p->src[p->pos++];
        if (c == ']' && !first) {
            return n;
        }
        first = false;

        int ch = (unsigned char)c;
        if (c == '\\') {
            int64_t esc = pp_read_escape(p);
            if (esc < 0) {
                node_free(n);
                p->ok = false;
                return NULL;
            }
            ch = (int)esc;
        }

        if (c == '-' && last_ch >= 0 && !in_range && pp_peek(p) != ']' ) {
            in_range = true;
            continue;
        }

        if (in_range) {
            int start = last_ch;
            int end = ch;
            if (start > end) {
                int tmp = start;
                start = end;
                end = tmp;
            }
            for (int i = start; i <= end; i++) {
                n->as.cls.table[(unsigned char)i] = 1;
            }
            last_ch = -1;
            in_range = false;
        } else {
            n->as.cls.table[(unsigned char)ch] = 1;
            last_ch = ch;
        }
    }

    node_free(n);
    p->ok = false;
    return NULL;
}

static Node *parse_primary(PegParser *p) {
    pp_skip_ws(p);
    char c = pp_peek(p);
    if (c == '\0') return NULL;

    if (c == '"') {
        return parse_literal(p);
    }
    if (c == '[') {
        return parse_class(p);
    }
    if (c == '.') {
        p->pos++;
        return node_new(N_DOT);
    }
    if (c == '(') {
        p->pos++;
        Node *inner = parse_expr(p);
        pp_skip_ws(p);
        if (!inner || !pp_consume(p, ')')) {
            node_free(inner);
            p->ok = false;
            return NULL;
        }
        Node *g = node_new(N_GROUP);
        if (!g) {
            node_free(inner);
            return NULL;
        }
        g->as.group.child = inner;
        return g;
    }

    p->ok = false;
    return NULL;
}

static Node *parse_postfix(PegParser *p) {
    Node *n = parse_primary(p);
    if (!n) return NULL;

    while (true) {
        pp_skip_ws(p);
        char c = pp_peek(p);
        if (c == '*' || c == '+' || c == '?') {
            p->pos++;
            Node *u = node_new(c == '*' ? N_STAR : c == '+' ? N_PLUS : N_OPT);
            if (!u) {
                node_free(n);
                return NULL;
            }
            u->as.unary.child = n;
            n = u;
            continue;
        }
        break;
    }

    return n;
}

static Node *parse_sequence(PegParser *p) {
    pp_skip_ws(p);

    Node **items = NULL;
    int64_t cap = 0;
    int64_t count = 0;

    while (true) {
        pp_skip_ws(p);
        char c = pp_peek(p);
        if (c == '\0' || c == ')' || c == '/') {
            break;
        }

        Node *part = parse_postfix(p);
        if (!part) {
            p->ok = false;
            break;
        }

        if (count >= cap) {
            int64_t new_cap = cap == 0 ? 4 : cap * 2;
            Node **ni = (Node**)realloc(items, (size_t)new_cap * sizeof(Node*));
            if (!ni) {
                node_free(part);
                p->ok = false;
                break;
            }
            items = ni;
            cap = new_cap;
        }
        items[count++] = part;
    }

    if (!p->ok) {
        for (int64_t i = 0; i < count; i++) node_free(items[i]);
        free(items);
        return NULL;
    }

    if (count == 0) {
        free(items);
        p->ok = false;
        return NULL;
    }
    if (count == 1) {
        Node *only = items[0];
        free(items);
        return only;
    }

    Node *seq = node_new(N_SEQ);
    if (!seq) {
        for (int64_t i = 0; i < count; i++) node_free(items[i]);
        free(items);
        return NULL;
    }
    seq->as.list.items = items;
    seq->as.list.count = count;
    return seq;
}

static Node *parse_expr(PegParser *p) {
    Node *first = parse_sequence(p);
    if (!first) return NULL;

    Node **alts = NULL;
    int64_t cap = 0;
    int64_t count = 0;

    /* collect choices */
    alts = (Node**)malloc(sizeof(Node*) * 4);
    if (!alts) {
        node_free(first);
        return NULL;
    }
    cap = 4;
    alts[count++] = first;

    while (true) {
        pp_skip_ws(p);
        if (!pp_consume(p, '/')) break;
        Node *rhs = parse_sequence(p);
        if (!rhs) {
            p->ok = false;
            break;
        }
        if (count >= cap) {
            int64_t new_cap = cap * 2;
            Node **na = (Node**)realloc(alts, (size_t)new_cap * sizeof(Node*));
            if (!na) {
                node_free(rhs);
                p->ok = false;
                break;
            }
            alts = na;
            cap = new_cap;
        }
        alts[count++] = rhs;
    }

    if (!p->ok) {
        for (int64_t i = 0; i < count; i++) node_free(alts[i]);
        free(alts);
        return NULL;
    }

    if (count == 1) {
        Node *only = alts[0];
        free(alts);
        return only;
    }

    Node *ch = node_new(N_CHOICE);
    if (!ch) {
        for (int64_t i = 0; i < count; i++) node_free(alts[i]);
        free(alts);
        return NULL;
    }
    ch->as.list.items = alts;
    ch->as.list.count = count;
    return ch;
}

static bool match_node(Node *n, const char *in, int64_t in_len, int64_t pos, int64_t *out_pos, CaptureState *caps);

static bool match_list(Node **items, int64_t count, const char *in, int64_t in_len, int64_t pos, int64_t *out_pos, CaptureState *caps) {
    int64_t p0 = pos;
    int64_t cap0 = caps ? caps->count : 0;
    for (int64_t i = 0; i < count; i++) {
        int64_t next = p0;
        if (!match_node(items[i], in, in_len, p0, &next, caps)) {
            if (caps) caps->count = cap0;
            return false;
        }
        p0 = next;
    }
    *out_pos = p0;
    return true;
}

static bool match_node(Node *n, const char *in, int64_t in_len, int64_t pos, int64_t *out_pos, CaptureState *caps) {
    if (!n) return false;
    switch (n->type) {
        case N_LITERAL: {
            if (pos + n->as.lit.len > in_len) return false;
            if (memcmp(in + pos, n->as.lit.bytes, (size_t)n->as.lit.len) != 0) return false;
            *out_pos = pos + n->as.lit.len;
            return true;
        }
        case N_DOT: {
            if (pos >= in_len) return false;
            *out_pos = pos + 1;
            return true;
        }
        case N_CLASS: {
            if (pos >= in_len) return false;
            unsigned char c = (unsigned char)in[pos];
            bool hit = n->as.cls.table[c] != 0;
            if (n->as.cls.inverted) hit = !hit;
            if (!hit) return false;
            *out_pos = pos + 1;
            return true;
        }
        case N_SEQ:
            return match_list(n->as.list.items, n->as.list.count, in, in_len, pos, out_pos, caps);
        case N_CHOICE: {
            int64_t cap0 = caps ? caps->count : 0;
            for (int64_t i = 0; i < n->as.list.count; i++) {
                int64_t next = pos;
                if (caps) caps->count = cap0;
                if (match_node(n->as.list.items[i], in, in_len, pos, &next, caps)) {
                    *out_pos = next;
                    return true;
                }
            }
            if (caps) caps->count = cap0;
            return false;
        }
        case N_OPT: {
            int64_t cap0 = caps ? caps->count : 0;
            int64_t next = pos;
            if (match_node(n->as.unary.child, in, in_len, pos, &next, caps)) {
                *out_pos = next;
                return true;
            }
            if (caps) caps->count = cap0;
            *out_pos = pos;
            return true;
        }
        case N_STAR: {
            int64_t p0 = pos;
            while (true) {
                int64_t cap0 = caps ? caps->count : 0;
                int64_t next = p0;
                if (!match_node(n->as.unary.child, in, in_len, p0, &next, caps)) {
                    if (caps) caps->count = cap0;
                    break;
                }
                if (next == p0) {
                    /* empty match; avoid infinite loop */
                    break;
                }
                p0 = next;
            }
            *out_pos = p0;
            return true;
        }
        case N_PLUS: {
            int64_t first = pos;
            if (!match_node(n->as.unary.child, in, in_len, pos, &first, caps)) return false;
            if (first == pos) {
                /* '+' requires progress */
                return false;
            }
            /* then behave like '*' */
            int64_t p0 = first;
            while (true) {
                int64_t cap0 = caps ? caps->count : 0;
                int64_t next = p0;
                if (!match_node(n->as.unary.child, in, in_len, p0, &next, caps)) {
                    if (caps) caps->count = cap0;
                    break;
                }
                if (next == p0) break;
                p0 = next;
            }
            *out_pos = p0;
            return true;
        }
        case N_GROUP: {
            int64_t cap0 = caps ? caps->count : 0;
            int64_t start = pos;
            int64_t next = pos;
            if (!match_node(n->as.group.child, in, in_len, pos, &next, caps)) {
                if (caps) caps->count = cap0;
                return false;
            }
            if (caps) cap_push(caps, start, next);
            *out_pos = next;
            return true;
        }
    }
    return false;
}

void* nl_peg_compile(const char* pattern) {
    if (!pattern) return NULL;

    /* Nanolang string literals do not currently unescape \" sequences, so PEG
     * patterns arrive with backslashes still present. We only unescape \" and \\\
     * here so the PEG syntax can use quoted literals like "a".
     */
    int64_t raw_len = (int64_t)strnlen(pattern, 1024ULL * 1024ULL);
    char *unescaped = (char*)malloc((size_t)raw_len + 1);
    if (!unescaped) return NULL;
    int64_t ulen = 0;
    for (int64_t i = 0; i < raw_len; i++) {
        char c = pattern[i];
        if (c == '\\' && (i + 1) < raw_len) {
            char n = pattern[i + 1];
            if (n == '"' || n == '\\') {
                unescaped[ulen++] = n;
                i++;
                continue;
            }
        }
        unescaped[ulen++] = c;
    }
    unescaped[ulen] = '\0';

    PegParser p = {.src = unescaped, .len = ulen, .pos = 0, .ok = true};
    Node *root = parse_expr(&p);
    pp_skip_ws(&p);
    if (!p.ok || !root || p.pos != p.len) {
        node_free(root);
        free(unescaped);
        return NULL;
    }

    PEG *peg = (PEG*)calloc(1, sizeof(PEG));
    if (!peg) {
        node_free(root);
        free(unescaped);
        return NULL;
    }
    peg->root = root;
    free(unescaped);
    return peg;
}

int64_t nl_peg_match(void* peg_ptr, const char* input) {
    PEG *peg = (PEG*)peg_ptr;
    if (!peg || !peg->root || !input) return -1;
    int64_t in_len = (int64_t)strnlen(input, 1024ULL * 1024ULL);
    int64_t out = 0;
    bool ok = match_node(peg->root, input, in_len, 0, &out, NULL);
    return (ok && out == in_len) ? 1 : 0;
}

DynArray* nl_peg_captures(void* peg_ptr, const char* input) {
    PEG *peg = (PEG*)peg_ptr;
    DynArray *out_arr = dyn_array_new(ELEM_STRING);
    if (!peg || !peg->root || !input) return out_arr;

    int64_t in_len = (int64_t)strnlen(input, 1024ULL * 1024ULL);
    CaptureState caps;
    cap_init(&caps);

    int64_t out_pos = 0;
    bool ok = match_node(peg->root, input, in_len, 0, &out_pos, &caps);
    if (!(ok && out_pos == in_len)) {
        cap_free(&caps);
        return out_arr;
    }

    for (int64_t i = 0; i < caps.count; i++) {
        int64_t s = caps.spans[i].start;
        int64_t e = caps.spans[i].end;
        if (s < 0) s = 0;
        if (e < s) e = s;
        if (e > in_len) e = in_len;
        int64_t n = e - s;
        char *buf = (char*)malloc((size_t)n + 1);
        if (!buf) continue;
        memcpy(buf, input + s, (size_t)n);
        buf[n] = '\0';
        dyn_array_push_string(out_arr, buf);
    }

    cap_free(&caps);
    return out_arr;
}

void nl_peg_free(void* peg_ptr) {
    PEG *peg = (PEG*)peg_ptr;
    if (!peg) return;
    node_free(peg->root);
    free(peg);
}
