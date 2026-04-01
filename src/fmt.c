/*
 * fmt.c — nano-fmt: canonical code formatter for .nano source files
 *
 * Token-based formatter. Reconstructs canonical spacing from the token stream.
 *
 * Style rules:
 *   1. 4-space indent (configurable)
 *   2. One blank line between top-level declarations
 *   3. No trailing whitespace; trailing newline at EOF
 *   4. Space around binary operators in call expressions
 *   5. `fn name(param: Type, ...) -> RetType`
 *   6. `let name: Type = value`
 *   7. Opening `{` on same line; body indented; closing `}` on own line
 *
 * Note: comments are stripped by the nanolang lexer and are not preserved.
 */

#include "fmt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>
#include <ctype.h>

/* ── String builder ──────────────────────────────────────────────────── */
typedef struct { char *buf; size_t len; size_t cap; } FSB;

static void fsb_init(FSB *s) { s->buf = NULL; s->len = 0; s->cap = 0; }
static void fsb_free(FSB *s) { free(s->buf); s->buf = NULL; s->len = s->cap = 0; }
static void fsb_grow(FSB *s, size_t need) {
    if (s->len + need + 1 <= s->cap) return;
    size_t nc = s->cap ? s->cap * 2 : 4096;
    while (nc < s->len + need + 1) nc *= 2;
    s->buf = realloc(s->buf, nc);
    s->cap = nc;
}
static void fsb_append(FSB *s, const char *str) {
    size_t n = strlen(str);
    fsb_grow(s, n);
    memcpy(s->buf + s->len, str, n + 1);
    s->len += n;
}
static void fsb_append_char(FSB *s, char c) {
    fsb_grow(s, 1);
    s->buf[s->len++] = c;
    s->buf[s->len]   = '\0';
}

/* ── Spacing decision table ──────────────────────────────────────────── */

/* Should we put a space BEFORE this token, given the previous token type? */
static bool space_before(int prev, int cur_tok) {
    /* Never space before these */
    if (cur_tok == TOKEN_COMMA  || cur_tok == TOKEN_RPAREN ||
        cur_tok == TOKEN_RBRACE || cur_tok == TOKEN_EOF)
        return false;
    /* Never space after open delimiters */
    if (prev == TOKEN_LPAREN || prev == TOKEN_LBRACE)
        return false;
    /* Space after most declaration keywords, but NOT fn (fn name( is correct) */
    if (prev == TOKEN_STRUCT || prev == TOKEN_ENUM  || prev == TOKEN_UNION  ||
        prev == TOKEN_SHADOW || prev == TOKEN_PUB   || prev == TOKEN_EXTERN ||
        prev == TOKEN_MODULE || prev == TOKEN_FROM  || prev == TOKEN_IMPORT ||
        prev == TOKEN_NOT)
        return true;
    /* fn: space before identifier (fn add), but NOT before ( (fn add( not fn add () ) */
    if (prev == TOKEN_FN)
        return (cur_tok == TOKEN_IDENTIFIER);
    /* Space after keywords */
    if (prev == TOKEN_LET    || prev == TOKEN_SET    || prev == TOKEN_RETURN ||
        prev == TOKEN_IF     || prev == TOKEN_ELSE   || prev == TOKEN_WHILE  ||
        prev == TOKEN_ASSERT || prev == TOKEN_MUT    || prev == TOKEN_AND   ||
        prev == TOKEN_OR)
        return true;
    /* Colon: space after (for param: Type and let x: T), no space before */
    if (prev == TOKEN_COLON) return true;
    if (cur_tok == TOKEN_COLON) return false;
    /* Arrow ->: space before and after */
    if (prev == TOKEN_ARROW || cur_tok == TOKEN_ARROW) return true;
    /* Binary operators: space before (but not immediately after open paren) */
    if ((cur_tok == TOKEN_PLUS  || cur_tok == TOKEN_MINUS ||
         cur_tok == TOKEN_STAR  || cur_tok == TOKEN_SLASH ||
         cur_tok == TOKEN_PERCENT ||
         cur_tok == TOKEN_EQ    || cur_tok == TOKEN_NE    ||
         cur_tok == TOKEN_LT    || cur_tok == TOKEN_LE    ||
         cur_tok == TOKEN_GT    || cur_tok == TOKEN_GE    ||
         cur_tok == TOKEN_AND   || cur_tok == TOKEN_OR    ||
         cur_tok == TOKEN_ASSIGN) && prev != TOKEN_LPAREN)
        return true;
    /* Space after operators (prev=op, cur=operand) */
    if ((prev == TOKEN_PLUS  || prev == TOKEN_MINUS ||
         prev == TOKEN_STAR  || prev == TOKEN_SLASH ||
         prev == TOKEN_PERCENT ||
         prev == TOKEN_EQ    || prev == TOKEN_NE    ||
         prev == TOKEN_LT    || prev == TOKEN_LE    ||
         prev == TOKEN_GT    || prev == TOKEN_GE    ||
         prev == TOKEN_ASSIGN) &&
        cur_tok != TOKEN_RPAREN && cur_tok != TOKEN_COMMA)
        return true;
    /* Space after comma */
    if (prev == TOKEN_COMMA) return true;
    /* Space before { (block open) */
    if (cur_tok == TOKEN_LBRACE) return true;
    /* No space before ( after identifier (fn name( not fn name () */
    if (cur_tok == TOKEN_LPAREN && (prev == TOKEN_IDENTIFIER))
        return false;
    /* Space before ( after ) — nested call: (+ a (- b 1)) */
    if (cur_tok == TOKEN_LPAREN && prev == TOKEN_RPAREN)
        return true;
    /* Space before ( after number/bool/string */
    if (cur_tok == TOKEN_LPAREN && (prev == TOKEN_NUMBER || prev == TOKEN_FLOAT ||
        prev == TOKEN_TRUE || prev == TOKEN_FALSE))
        return true;
    /* Space before string literals (but not at start of expression) */
    if (cur_tok == TOKEN_STRING && prev != TOKEN_LPAREN && prev != TOKEN_EOF)
        return true;
    /* Space between identifiers, numbers, etc. */
    if (prev == TOKEN_IDENTIFIER || prev == TOKEN_NUMBER ||
        prev == TOKEN_FLOAT || prev == TOKEN_STRING || prev == TOKEN_TRUE ||
        prev == TOKEN_FALSE || prev == TOKEN_RPAREN) {
        if (cur_tok == TOKEN_IDENTIFIER || cur_tok == TOKEN_NUMBER)
            return true;
    }
    return false;
}

/* ── Token canonical string ─────────────────────────────────────────── */
static const char *token_canonical_str(Token *t) {
    if (t->value) return t->value;
    switch (t->token_type) {
        case TOKEN_LPAREN:   return "(";
        case TOKEN_RPAREN:   return ")";
        case TOKEN_LBRACE:   return "{";
        case TOKEN_RBRACE:   return "}";
        case TOKEN_LBRACKET: return "[";
        case TOKEN_RBRACKET: return "]";
        case TOKEN_COLON:    return ":";
        case TOKEN_COMMA:    return ",";
        case TOKEN_ARROW:    return "->";
        case TOKEN_ASSIGN:   return "=";
        case TOKEN_PLUS:     return "+";
        case TOKEN_MINUS:    return "-";
        case TOKEN_STAR:     return "*";
        case TOKEN_SLASH:    return "/";
        case TOKEN_PERCENT:  return "%";
        case TOKEN_EQ:       return "==";
        case TOKEN_NE:       return "!=";
        case TOKEN_LT:       return "<";
        case TOKEN_LE:       return "<=";
        case TOKEN_GT:       return ">";
        case TOKEN_GE:       return ">=";
        case TOKEN_AND:      return "and";
        case TOKEN_OR:       return "or";
        case TOKEN_NOT:      return "not";
        case TOKEN_DOT:      return ".";
        default:             return "";
    }
}

/* ── Formatter state ─────────────────────────────────────────────────── */
typedef struct {
    Token *tokens;
    int    count;
    int    pos;
    FSB    out;
    int    indent;
    int    indent_size;
    bool   at_line_start;
    int    brace_depth;
    int    paren_depth;
    int    prev_type;
} FCtx;

static void emit_indent_spaces(FCtx *ctx) {
    for (int i = 0; i < ctx->indent * ctx->indent_size; i++)
        fsb_append_char(&ctx->out, ' ');
}
static void emit_newline(FCtx *ctx) {
    /* Trim trailing spaces */
    while (ctx->out.len > 0 && ctx->out.buf[ctx->out.len - 1] == ' ')
        ctx->out.buf[--ctx->out.len] = '\0';
    fsb_append_char(&ctx->out, '\n');
    ctx->at_line_start = true;
}

/* ── Single-pass formatter ───────────────────────────────────────────── */
static void fmt_pass(FCtx *ctx) {
    int prev = TOKEN_EOF;

    while (ctx->pos < ctx->count) {
        Token *t = &ctx->tokens[ctx->pos];
        if (!t || t->token_type == TOKEN_EOF) break;
        int tt = t->token_type;
        /* String literals need re-quoting (lexer strips quotes from value) */
        char str_buf[4096];
        const char *val;
        if (tt == TOKEN_STRING && t->value) {
            /* Escape and re-quote */
            size_t vi = 0;
            str_buf[vi++] = '"';
            for (const char *p = t->value; *p && vi < sizeof(str_buf) - 4; p++) {
                if (*p == '"') { str_buf[vi++] = '\\'; str_buf[vi++] = '"'; }
                else             { str_buf[vi++] = *p; }
            }
            str_buf[vi++] = '"';
            str_buf[vi] = '\0';
            val = str_buf;
        } else if (0 /* TOKEN_FSTRING */ && t->value) {
        } else {
            val = token_canonical_str(t);
        }

        /* ── Newline before closing brace ──────────────────────────── */
        if (tt == TOKEN_RBRACE && !ctx->at_line_start) {
            emit_newline(ctx);
        }

        /* ── Blank line before top-level keywords ───────────────────── */
        if (ctx->brace_depth == 0 && ctx->out.len > 1 &&
            (tt == TOKEN_FN || tt == TOKEN_STRUCT || tt == TOKEN_ENUM ||
             tt == TOKEN_UNION || tt == TOKEN_SHADOW || tt == TOKEN_PUB ||
             tt == TOKEN_EXTERN || tt == TOKEN_FROM || tt == TOKEN_IMPORT)) {
            if (!ctx->at_line_start) emit_newline(ctx);
            /* Emit extra blank line if not already at start of file */
            if (ctx->out.len > 1) fsb_append_char(&ctx->out, '\n');
        }

        /* ── Indent at line start ───────────────────────────────────── */
        if (ctx->at_line_start) {
            if (tt == TOKEN_RBRACE && ctx->indent > 0)
                ctx->indent--;
            emit_indent_spaces(ctx);
            ctx->at_line_start = false;
        } else {
            /* Adjust indent for close brace (already on same line — handled above) */
            if (tt == TOKEN_RBRACE && ctx->indent > 0)
                ctx->indent--;
        }

        /* ── Space before token ─────────────────────────────────────── */
        if (ctx->out.len > 0 && !ctx->at_line_start && space_before(prev, tt)) {
            /* Only add space if not already at start of a line indent */
            char last = ctx->out.buf[ctx->out.len - 1];
            if (last != ' ' && last != '\n')
                fsb_append_char(&ctx->out, ' ');
        }

        /* ── Emit the token ─────────────────────────────────────────── */
        fsb_append(&ctx->out, val);
        prev = tt;
        ctx->pos++;

        /* ── Post-emit: newline / indent adjustments ────────────────── */

        /* If next token is a statement keyword at same indent (not in paren),
         * emit a newline so each statement is on its own line */
        if (ctx->brace_depth > 0 && 1 == 1  /* paren tracking removed */ &&
            !ctx->at_line_start) {
            Token *nx = (ctx->pos < ctx->count) ? &ctx->tokens[ctx->pos] : NULL;
            if (nx) {
                int nt = nx->token_type;
                if (nt == TOKEN_LET || nt == TOKEN_SET || nt == TOKEN_RETURN ||
                    nt == TOKEN_IF  || nt == TOKEN_WHILE || nt == TOKEN_ASSERT) {
                    emit_newline(ctx);
                }
            }
        }

        if (tt == TOKEN_LPAREN) {
            ctx->paren_depth++;
        } else if (tt == TOKEN_RPAREN) {
            if (ctx->paren_depth > 0) ctx->paren_depth--;
        } else if (tt == TOKEN_LBRACE) {
            ctx->brace_depth++;
            ctx->indent++;
            emit_newline(ctx);
        } else if (tt == TOKEN_RBRACE) {
            ctx->brace_depth--;
            emit_newline(ctx);
        }
    }

    /* Trailing newline */
    if (ctx->out.len > 0 && ctx->out.buf[ctx->out.len - 1] != '\n')
        emit_newline(ctx);
}

/* ── Public API ──────────────────────────────────────────────────────── */
char *fmt_source(const char *source, const char *filename, const FmtOptions *opts) {
    (void)filename;
    if (!source) return NULL;

    static const FmtOptions default_opts = { 4, false, false, false };
    if (!opts) opts = &default_opts;

    int count = 0;
    Token *tokens = tokenize(source, &count);
    if (!tokens || count == 0) return NULL;

    FCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    fsb_init(&ctx.out);
    ctx.tokens       = tokens;
    ctx.count        = count;
    ctx.indent_size  = opts->indent_size > 0 ? opts->indent_size : 4;
    ctx.at_line_start = true;
    ctx.prev_type    = TOKEN_EOF;

    fmt_pass(&ctx);

    char *result = strdup(ctx.out.buf ? ctx.out.buf : "");
    fsb_free(&ctx.out);
    free_tokens(tokens, count);
    return result;
}

int fmt_file(const char *path, const FmtOptions *opts) {
    if (!path) return 1;

    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "nano-fmt: cannot open %s\n", path); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *src = malloc((size_t)sz + 1);
    if (!src) { fclose(f); return 1; }
    if (fread(src, 1, (size_t)sz, f) != (size_t)sz) { /* partial ok */ }
    src[sz] = '\0'; fclose(f);

    char *formatted = fmt_source(src, path, opts);
    free(src);
    if (!formatted) { fprintf(stderr, "nano-fmt: formatting failed: %s\n", path); return 1; }

    bool changed = false;
    {
        FILE *f2 = fopen(path, "r");
        if (f2) {
            fseek(f2, 0, SEEK_END); long sz2 = ftell(f2); fseek(f2, 0, SEEK_SET);
            char *orig = malloc((size_t)sz2 + 1);
            if (orig) {
                if (fread(orig, 1, (size_t)sz2, f2) != (size_t)sz2) { /* ok */ }
                orig[sz2] = '\0';
                changed = (strcmp(orig, formatted) != 0);
                free(orig);
            }
            fclose(f2);
        }
    }

    if (opts && opts->check_only) {
        if (changed) {
            fprintf(stderr, "nano-fmt: %s would be reformatted\n", path);
            free(formatted); return 2;
        }
        free(formatted); return 0;
    }

    if (opts && opts->write_in_place) {
        if (changed) {
            FILE *fw = fopen(path, "w");
            if (!fw) { fprintf(stderr, "nano-fmt: cannot write %s\n", path); free(formatted); return 1; }
            fputs(formatted, fw); fclose(fw);
            if (opts->verbose) fprintf(stderr, "nano-fmt: reformatted %s\n", path);
        }
    } else {
        fputs(formatted, stdout);
    }

    free(formatted);
    return 0;
}
