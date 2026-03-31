/*
 * docgen.c — HTML API documentation generator for nanolang
 *
 * Extracts /// doc comments associated with exported declarations and emits
 * a self-contained single-page HTML document with:
 *   - Inline CSS (no external dependencies)
 *   - Sidebar TOC with anchored section links
 *   - @example blocks rendered as syntax-highlighted <code>
 *   - @param / @returns / @throws annotation tags
 */

#include "docgen.h"
#include "nanolang.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>

/* ── Buffer helpers ──────────────────────────────────────────────────────── */

static void buf_appendf(char *buf, size_t cap, size_t *off, const char *fmt, ...) {
    if (!buf || cap == 0 || !off || *off + 1 >= cap) return;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf + *off, cap - *off, fmt, ap);
    va_end(ap);
    if (n > 0) {
        size_t nn = (size_t)n;
        *off += (nn < cap - *off) ? nn : cap - *off - 1;
    }
}

/* ── Type rendering (mirrors reflection.c helpers) ───────────────────────── */

static void append_typeinfo_str(char *buf, size_t cap, size_t *off, const TypeInfo *ti);

static void append_type_str(char *buf, size_t cap, size_t *off,
                            Type t, const char *sname, Type elt,
                            const TypeInfo *ti) {
    switch (t) {
        case TYPE_INT:    buf_appendf(buf, cap, off, "int");    return;
        case TYPE_U8:     buf_appendf(buf, cap, off, "u8");     return;
        case TYPE_FLOAT:  buf_appendf(buf, cap, off, "float");  return;
        case TYPE_BOOL:   buf_appendf(buf, cap, off, "bool");   return;
        case TYPE_STRING: buf_appendf(buf, cap, off, "string"); return;
        case TYPE_VOID:   buf_appendf(buf, cap, off, "void");   return;
        case TYPE_ARRAY:
            buf_appendf(buf, cap, off, "array<");
            if (ti && ti->element_type)
                append_typeinfo_str(buf, cap, off, ti->element_type);
            else
                append_type_str(buf, cap, off, elt, NULL, TYPE_UNKNOWN, NULL);
            buf_appendf(buf, cap, off, ">");
            return;
        case TYPE_HASHMAP:
            if (ti) append_typeinfo_str(buf, cap, off, ti);
            else     buf_appendf(buf, cap, off, "HashMap");
            return;
        case TYPE_STRUCT:
        case TYPE_ENUM:
        case TYPE_UNION:
        case TYPE_OPAQUE:
            if (sname && sname[0])
                buf_appendf(buf, cap, off, "%s", sname);
            else if (ti && ti->generic_name)
                buf_appendf(buf, cap, off, "%s", ti->generic_name);
            else
                buf_appendf(buf, cap, off, "unknown");
            return;
        default:
            buf_appendf(buf, cap, off, "unknown");
            return;
    }
}

static void append_typeinfo_str(char *buf, size_t cap, size_t *off,
                                 const TypeInfo *ti) {
    if (!ti) { buf_appendf(buf, cap, off, "unknown"); return; }

    if (ti->base_type == TYPE_HASHMAP && ti->generic_name &&
        strcmp(ti->generic_name, "HashMap") == 0 && ti->type_param_count == 2) {
        buf_appendf(buf, cap, off, "HashMap<");
        append_typeinfo_str(buf, cap, off, ti->type_params[0]);
        buf_appendf(buf, cap, off, ",");
        append_typeinfo_str(buf, cap, off, ti->type_params[1]);
        buf_appendf(buf, cap, off, ">");
        return;
    }
    if (ti->base_type == TYPE_ARRAY && ti->element_type) {
        buf_appendf(buf, cap, off, "array<");
        append_typeinfo_str(buf, cap, off, ti->element_type);
        buf_appendf(buf, cap, off, ">");
        return;
    }
    if ((ti->base_type == TYPE_STRUCT || ti->base_type == TYPE_UNION) &&
        ti->generic_name) {
        buf_appendf(buf, cap, off, "%s", ti->generic_name);
        if (ti->type_param_count > 0 && ti->type_params) {
            buf_appendf(buf, cap, off, "<");
            for (int i = 0; i < ti->type_param_count; i++) {
                if (i > 0) buf_appendf(buf, cap, off, ",");
                append_typeinfo_str(buf, cap, off, ti->type_params[i]);
            }
            buf_appendf(buf, cap, off, ">");
        }
        return;
    }
    append_type_str(buf, cap, off, ti->base_type, ti->generic_name,
                    TYPE_UNKNOWN, NULL);
}

/* ── Doc comment extraction ──────────────────────────────────────────────── */

typedef struct {
    int assoc_line; /* First non-/// non-blank line after this block (1-based) */
    char *text;     /* Concatenated comment text with '/// ' prefix stripped   */
} DocBlock;

typedef struct {
    DocBlock *blocks;
    int count;
    int cap;
} DocMap;

static void docmap_push(DocMap *m, int assoc_line, const char *text) {
    if (m->count >= m->cap) {
        m->cap = m->cap ? m->cap * 2 : 16;
        m->blocks = realloc(m->blocks, sizeof(DocBlock) * (size_t)m->cap);
    }
    m->blocks[m->count].assoc_line = assoc_line;
    m->blocks[m->count].text = strdup(text ? text : "");
    m->count++;
}

static void docmap_free(DocMap *m) {
    for (int i = 0; i < m->count; i++) free(m->blocks[i].text);
    free(m->blocks);
    m->blocks = NULL;
    m->count = m->cap = 0;
}

/*
 * Scan source text line-by-line.
 * Consecutive lines starting with /// form a doc block.
 * The block is associated with the first non-blank, non-/// line that follows.
 * A blank line between the comment block and the declaration discards the block.
 */
static DocMap build_doc_map(const char *src) {
    DocMap m = {NULL, 0, 0};
    if (!src) return m;

    char accum[16384];
    size_t accum_len = 0;
    int in_doc = 0;
    int line = 1;
    const char *p = src;

    while (*p) {
        const char *eol = p;
        while (*eol && *eol != '\n') eol++;

        /* Skip leading whitespace to find /// */
        const char *lp = p;
        while (lp < eol && (*lp == ' ' || *lp == '\t')) lp++;

        int is_doc_line = (lp + 2 < eol &&
                           lp[0] == '/' && lp[1] == '/' && lp[2] == '/');

        if (is_doc_line) {
            const char *text = lp + 3;
            if (text < eol && *text == ' ') text++; /* strip one leading space */
            size_t tlen = (size_t)(eol - text);
            if (accum_len + tlen + 2 < sizeof(accum)) {
                if (accum_len > 0) accum[accum_len++] = '\n';
                memcpy(accum + accum_len, text, tlen);
                accum_len += tlen;
                accum[accum_len] = '\0';
            }
            in_doc = 1;
        } else if (in_doc) {
            /* Check if blank line */
            int blank = 1;
            for (const char *c = p; c < eol; c++) {
                if (*c != ' ' && *c != '\t') { blank = 0; break; }
            }
            if (!blank) {
                docmap_push(&m, line, accum);
            }
            /* Either way, reset accumulator */
            accum_len = 0;
            accum[0] = '\0';
            in_doc = 0;
        }

        p = (*eol == '\n') ? eol + 1 : eol;
        line++;
    }
    return m;
}

/* Find a doc block associated with a declaration at decl_line.
 * Allow up to 2 lines of offset (e.g., for annotations before the keyword). */
static const char *docmap_find(const DocMap *m, int decl_line) {
    for (int i = 0; i < m->count; i++) {
        int d = m->blocks[i].assoc_line - decl_line;
        if (d >= 0 && d <= 2) return m->blocks[i].text;
    }
    return NULL;
}

/* ── HTML helpers ────────────────────────────────────────────────────────── */

static void html_escape(FILE *out, const char *s) {
    if (!s) return;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        switch (*p) {
            case '&': fputs("&amp;",  out); break;
            case '<': fputs("&lt;",   out); break;
            case '>': fputs("&gt;",   out); break;
            case '"': fputs("&quot;", out); break;
            default:  fputc((int)*p,  out); break;
        }
    }
}

/* ── Syntax highlighter for @example blocks ──────────────────────────────── */

static const char *NANO_KW[] = {
    "fn", "pub", "let", "mut", "if", "else", "while", "for", "return",
    "struct", "enum", "union", "match", "import", "from", "as", "module",
    "true", "false", "extern", "effect", "handle", "async", "await",
    "shadow", "assert", "par",
    NULL
};

static int is_nano_kw(const char *w, size_t len) {
    for (int i = 0; NANO_KW[i]; i++) {
        if (strlen(NANO_KW[i]) == len && memcmp(NANO_KW[i], w, len) == 0)
            return 1;
    }
    return 0;
}

static void emit_char_escaped(FILE *out, char c) {
    switch ((unsigned char)c) {
        case '&': fputs("&amp;",  out); break;
        case '<': fputs("&lt;",   out); break;
        case '>': fputs("&gt;",   out); break;
        default:  fputc((int)c,   out); break;
    }
}

static void emit_code_line(FILE *out, const char *line, size_t len) {
    size_t i = 0;
    while (i < len) {
        unsigned char c = (unsigned char)line[i];

        /* Line comment: # to end of line */
        if (c == '#') {
            fputs("<span class=\"cm\">", out);
            for (; i < len; i++) emit_char_escaped(out, line[i]);
            fputs("</span>", out);
            break;
        }

        /* String literal */
        if (c == '"') {
            fputs("<span class=\"st\">", out);
            emit_char_escaped(out, '"');
            i++;
            while (i < len) {
                char sc = line[i];
                if (sc == '\\' && i + 1 < len) {
                    emit_char_escaped(out, sc);
                    i++;
                    emit_char_escaped(out, line[i]);
                    i++;
                    continue;
                }
                emit_char_escaped(out, sc);
                i++;
                if (sc == '"') break;
            }
            fputs("</span>", out);
            continue;
        }

        /* Number */
        if (isdigit(c)) {
            fputs("<span class=\"nm\">", out);
            while (i < len &&
                   (isdigit((unsigned char)line[i]) || line[i] == '.' ||
                    line[i] == 'e' || line[i] == 'E')) {
                emit_char_escaped(out, line[i]);
                i++;
            }
            fputs("</span>", out);
            continue;
        }

        /* Identifier or keyword */
        if (isalpha(c) || c == '_') {
            size_t start = i;
            while (i < len && (isalnum((unsigned char)line[i]) || line[i] == '_'))
                i++;
            size_t wlen = i - start;
            if (is_nano_kw(line + start, wlen)) {
                fputs("<span class=\"kw\">", out);
                fwrite(line + start, 1, wlen, out);
                fputs("</span>", out);
            } else {
                fwrite(line + start, 1, wlen, out);
            }
            continue;
        }

        emit_char_escaped(out, (char)c);
        i++;
    }
}

/* ── Doc comment body emitter ────────────────────────────────────────────── */

/*
 * Render the doc comment text as HTML.
 * Recognises:
 *   @example ... @end  — code block with syntax highlighting
 *   @param <name> ...  — parameter description
 *   @returns ...       — return value description
 *   @throws ...        — exception/effect description
 *   Blank lines        — paragraph breaks
 */
static void emit_doc_body(FILE *out, const char *doc) {
    if (!doc || !*doc) return;

    fputs("<div class=\"doc\">", out);

    const char *p = doc;
    int in_example = 0;
    int para_open = 0;

    while (*p) {
        const char *eol = p;
        while (*eol && *eol != '\n') eol++;
        size_t llen = (size_t)(eol - p);

        /* Skip leading whitespace for tag detection */
        const char *lp = p;
        while (lp < eol && (*lp == ' ' || *lp == '\t')) lp++;
        size_t ltrimlen = (size_t)(eol - lp);

        if (!in_example && ltrimlen >= 8 &&
            strncmp(lp, "@example", 8) == 0) {
            if (para_open) { fputs("</p>", out); para_open = 0; }
            in_example = 1;
            fputs("<pre class=\"example\"><code>", out);
        } else if (in_example &&
                   ltrimlen >= 4 && strncmp(lp, "@end", 4) == 0) {
            in_example = 0;
            fputs("</code></pre>", out);
        } else if (in_example) {
            emit_code_line(out, p, llen);
            fputc('\n', out);
        } else if (llen == 0) {
            if (para_open) { fputs("</p>", out); para_open = 0; }
        } else {
            if (!para_open) { fputs("<p>", out); para_open = 1; }
            if (ltrimlen >= 6 && strncmp(lp, "@param", 6) == 0) {
                fputs("<span class=\"tag\">@param</span>", out);
                html_escape(out, lp + 6);
            } else if (ltrimlen >= 8 && strncmp(lp, "@returns", 8) == 0) {
                fputs("<span class=\"tag\">@returns</span>", out);
                html_escape(out, lp + 8);
            } else if (ltrimlen >= 7 && strncmp(lp, "@throws", 7) == 0) {
                fputs("<span class=\"tag\">@throws</span>", out);
                html_escape(out, lp + 7);
            } else {
                for (size_t j = 0; j < llen; j++)
                    emit_char_escaped(out, p[j]);
            }
            fputs("</p>", out);
            para_open = 0;
        }

        p = (*eol == '\n') ? eol + 1 : eol;
    }

    if (in_example) fputs("</code></pre>", out);
    if (para_open)  fputs("</p>", out);
    fputs("</div>\n", out);
}

/* ── Inline CSS ──────────────────────────────────────────────────────────── */

static void emit_css(FILE *out) {
    fputs(
"*{box-sizing:border-box;margin:0;padding:0}\n"
"body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
      "color:#1a1a2e;background:#f8f9fa;display:flex;min-height:100vh}\n"
"#sidebar{width:230px;min-width:180px;background:#1a1a2e;color:#e0e0e0;"
          "padding:24px 16px;position:sticky;top:0;height:100vh;"
          "overflow-y:auto;flex-shrink:0}\n"
"#sidebar h2{font-size:.75rem;text-transform:uppercase;letter-spacing:.1em;"
              "color:#888;margin-bottom:10px}\n"
"#sidebar ul{list-style:none}\n"
"#sidebar li{margin:3px 0}\n"
"#sidebar a{color:#c9d1d9;text-decoration:none;font-size:.83rem;"
            "display:block;padding:3px 8px;border-radius:4px}\n"
"#sidebar a:hover{background:#2d3748;color:#fff}\n"
".toc-kind{font-size:.68rem!important;color:#6e7681!important;"
           "font-weight:700!important;text-transform:uppercase!important;"
           "margin-top:14px!important;margin-bottom:3px!important;"
           "padding:0!important;pointer-events:none!important}\n"
"main{flex:1;max-width:880px;padding:40px 48px}\n"
"h1{font-size:1.8rem;margin-bottom:6px}\n"
".module-path{color:#6e7681;font-size:.9rem;margin-bottom:36px}\n"
"section{margin-bottom:52px;border-top:1px solid #e1e4e8;padding-top:32px}\n"
"section:first-of-type{border-top:none;padding-top:0}\n"
"h2.decl{font-size:.92rem;font-family:'SFMono-Regular',Consolas,monospace;"
          "background:#f1f3f5;border:1px solid #dee2e6;border-radius:6px;"
          "padding:10px 14px;margin-bottom:14px;overflow-x:auto;white-space:pre-wrap}\n"
".kw{color:#d73a49;font-weight:700}\n"
".nm{color:#005cc5;font-weight:600}\n"
".ty{color:#6f42c1}\n"
".doc p{margin:6px 0;line-height:1.65;color:#444;font-size:.88rem}\n"
".tag{color:#e36209;font-weight:600}\n"
".example{background:#1a1a2e;border-radius:8px;padding:16px 20px;"
           "font-family:'SFMono-Regular',Consolas,monospace;font-size:.82rem;"
           "color:#e0e0e0;overflow-x:auto;margin:12px 0;white-space:pre}\n"
".example .kw{color:#ff7b72;font-weight:700}\n"
".example .st{color:#a5d6ff}\n"
".example .nm{color:#79c0ff}\n"
".example .cm{color:#8b949e;font-style:italic}\n"
".fields{margin:10px 0;border:1px solid #e1e4e8;border-radius:6px;"
          "overflow:hidden}\n"
".field{display:flex;gap:14px;font-family:'SFMono-Regular',Consolas,monospace;"
         "font-size:.83rem;padding:6px 12px;border-bottom:1px solid #f0f0f0}\n"
".field:last-child{border-bottom:none}\n"
".fn{color:#1a1a2e;font-weight:600;min-width:110px}\n"
".ft{color:#6f42c1}\n"
".variants{margin:10px 0;font-family:'SFMono-Regular',Consolas,monospace;"
            "font-size:.83rem}\n"
".variant{padding:4px 12px;border-bottom:1px solid #f0f0f0;color:#1a1a2e}\n"
".variants{border:1px solid #e1e4e8;border-radius:6px;overflow:hidden}\n"
".variant:last-child{border-bottom:none}\n",
    out);
}

/* ── TOC entry ───────────────────────────────────────────────────────────── */

typedef struct {
    char kind;       /* 'f'=fn, 's'=struct, 'e'=enum, 'u'=union */
    const char *name;
} TocEntry;

/* ── Main emitter ────────────────────────────────────────────────────────── */

bool emit_doc_html(const char *output_path, ASTNode *program,
                   const char *source_text, const char *module_name) {
    if (!program || program->type != AST_PROGRAM) return false;

    DocMap dmap = build_doc_map(source_text);

    FILE *out = fopen(output_path, "w");
    if (!out) {
        fprintf(stderr, "docgen: cannot open '%s' for writing: ", output_path);
        perror(NULL);
        docmap_free(&dmap);
        return false;
    }

    /* ── Collect TOC entries ── */
    TocEntry toc[1024];
    int toc_count = 0;
    for (int i = 0; i < program->as.program.count && toc_count < 1024; i++) {
        ASTNode *item = program->as.program.items[i];
        switch (item->type) {
            case AST_FUNCTION:
                if (item->as.function.is_pub && !item->as.function.is_extern &&
                    item->as.function.name) {
                    toc[toc_count].kind = 'f';
                    toc[toc_count++].name = item->as.function.name;
                }
                break;
            case AST_STRUCT_DEF:
                if (item->as.struct_def.is_pub && item->as.struct_def.name) {
                    toc[toc_count].kind = 's';
                    toc[toc_count++].name = item->as.struct_def.name;
                }
                break;
            case AST_ENUM_DEF:
                if (item->as.enum_def.is_pub && item->as.enum_def.name) {
                    toc[toc_count].kind = 'e';
                    toc[toc_count++].name = item->as.enum_def.name;
                }
                break;
            case AST_UNION_DEF:
                if (item->as.union_def.is_pub && item->as.union_def.name) {
                    toc[toc_count].kind = 'u';
                    toc[toc_count++].name = item->as.union_def.name;
                }
                break;
            default:
                break;
        }
    }

    /* ── HTML head ── */
    fprintf(out, "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
    fprintf(out, "<meta charset=\"utf-8\">\n");
    fprintf(out,
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n");
    fprintf(out, "<title>");
    html_escape(out, module_name ? module_name : "module");
    fprintf(out, " \xe2\x80\x94 nanolang API</title>\n");
    fprintf(out, "<style>\n");
    emit_css(out);
    fprintf(out, "</style>\n</head>\n<body>\n");

    /* ── Sidebar ── */
    fprintf(out, "<nav id=\"sidebar\">\n<h2>Contents</h2>\n<ul>\n");

    static const struct { char kind; const char *label; } groups[] = {
        {'f', "Functions"}, {'s', "Structs"},
        {'e', "Enums"},     {'u', "Unions"},
        {0, NULL}
    };
    for (int g = 0; groups[g].label; g++) {
        int found = 0;
        for (int j = 0; j < toc_count; j++) {
            if (toc[j].kind != groups[g].kind) continue;
            if (!found) {
                fprintf(out, "<li class=\"toc-kind\">%s</li>\n",
                        groups[g].label);
                found = 1;
            }
            fprintf(out, "<li><a href=\"#");
            html_escape(out, toc[j].name);
            fprintf(out, "\">");
            html_escape(out, toc[j].name);
            fprintf(out, "</a></li>\n");
        }
    }
    fprintf(out, "</ul>\n</nav>\n");

    /* ── Main content ── */
    fprintf(out, "<main id=\"content\">\n");
    fprintf(out, "<h1>");
    html_escape(out, module_name ? module_name : "module");
    fprintf(out, "</h1>\n");
    fprintf(out, "<p class=\"module-path\">module <code>");
    html_escape(out, module_name ? module_name : "module");
    fprintf(out, "</code></p>\n");

    /* ── Emit each exported declaration ── */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];

        /* ── Function ── */
        if (item->type == AST_FUNCTION &&
            item->as.function.is_pub &&
            !item->as.function.is_extern &&
            item->as.function.name) {
            const char *name = item->as.function.name;
            const char *doc  = docmap_find(&dmap, item->line);

            fprintf(out, "<section id=\"");
            html_escape(out, name);
            fprintf(out, "\">\n");

            /* Signature */
            fprintf(out, "<h2 class=\"decl\">"
                         "<span class=\"kw\">pub fn</span> "
                         "<span class=\"nm\">");
            html_escape(out, name);
            fprintf(out, "</span>(");
            for (int j = 0; j < item->as.function.param_count; j++) {
                Parameter *pp = &item->as.function.params[j];
                if (j > 0) fputs(", ", out);
                html_escape(out, pp->name ? pp->name : "_");
                fputs(": <span class=\"ty\">", out);
                char tbuf[256]; size_t toff = 0; tbuf[0] = '\0';
                append_type_str(tbuf, sizeof(tbuf), &toff,
                                pp->type, pp->struct_type_name,
                                pp->element_type, pp->type_info);
                html_escape(out, tbuf);
                fputs("</span>", out);
            }
            fputs(") -&gt; <span class=\"ty\">", out);
            char rbuf[256]; size_t roff = 0; rbuf[0] = '\0';
            append_type_str(rbuf, sizeof(rbuf), &roff,
                            item->as.function.return_type,
                            item->as.function.return_struct_type_name,
                            item->as.function.return_element_type,
                            item->as.function.return_type_info);
            html_escape(out, rbuf);
            fputs("</span></h2>\n", out);

            emit_doc_body(out, doc);
            fprintf(out, "</section>\n");

        /* ── Struct ── */
        } else if (item->type == AST_STRUCT_DEF &&
                   item->as.struct_def.is_pub &&
                   item->as.struct_def.name) {
            const char *name = item->as.struct_def.name;
            const char *doc  = docmap_find(&dmap, item->line);

            fprintf(out, "<section id=\"");
            html_escape(out, name);
            fprintf(out, "\">\n");
            fprintf(out, "<h2 class=\"decl\">"
                         "<span class=\"kw\">pub struct</span> "
                         "<span class=\"nm\">");
            html_escape(out, name);
            fputs("</span></h2>\n", out);

            if (item->as.struct_def.field_count > 0) {
                fputs("<div class=\"fields\">\n", out);
                for (int j = 0; j < item->as.struct_def.field_count; j++) {
                    fputs("<div class=\"field\">"
                          "<span class=\"fn\">", out);
                    html_escape(out, item->as.struct_def.field_names[j]);
                    fputs("</span><span class=\"ft\">", out);
                    char tbuf[256]; size_t toff = 0; tbuf[0] = '\0';
                    append_type_str(tbuf, sizeof(tbuf), &toff,
                                    item->as.struct_def.field_types[j],
                                    item->as.struct_def.field_type_names[j],
                                    item->as.struct_def.field_element_types[j],
                                    NULL);
                    html_escape(out, tbuf);
                    fputs("</span></div>\n", out);
                }
                fputs("</div>\n", out);
            }
            emit_doc_body(out, doc);
            fprintf(out, "</section>\n");

        /* ── Enum ── */
        } else if (item->type == AST_ENUM_DEF &&
                   item->as.enum_def.is_pub &&
                   item->as.enum_def.name) {
            const char *name = item->as.enum_def.name;
            const char *doc  = docmap_find(&dmap, item->line);

            fprintf(out, "<section id=\"");
            html_escape(out, name);
            fprintf(out, "\">\n");
            fprintf(out, "<h2 class=\"decl\">"
                         "<span class=\"kw\">pub enum</span> "
                         "<span class=\"nm\">");
            html_escape(out, name);
            fputs("</span></h2>\n", out);

            fputs("<div class=\"variants\">\n", out);
            for (int j = 0; j < item->as.enum_def.variant_count; j++) {
                fputs("<div class=\"variant\">", out);
                html_escape(out, item->as.enum_def.variant_names[j]);
                fputs("</div>\n", out);
            }
            fputs("</div>\n", out);
            emit_doc_body(out, doc);
            fprintf(out, "</section>\n");

        /* ── Union ── */
        } else if (item->type == AST_UNION_DEF &&
                   item->as.union_def.is_pub &&
                   item->as.union_def.name) {
            const char *name = item->as.union_def.name;
            const char *doc  = docmap_find(&dmap, item->line);

            fprintf(out, "<section id=\"");
            html_escape(out, name);
            fprintf(out, "\">\n");
            fprintf(out, "<h2 class=\"decl\">"
                         "<span class=\"kw\">pub union</span> "
                         "<span class=\"nm\">");
            html_escape(out, name);
            fputs("</span></h2>\n", out);

            fputs("<div class=\"variants\">\n", out);
            for (int j = 0; j < item->as.union_def.variant_count; j++) {
                fputs("<div class=\"variant\">", out);
                html_escape(out, item->as.union_def.variant_names[j]);
                if (item->as.union_def.variant_field_counts &&
                    item->as.union_def.variant_field_counts[j] > 0) {
                    fputs(" { ", out);
                    for (int k = 0;
                         k < item->as.union_def.variant_field_counts[j]; k++) {
                        if (k > 0) fputs(", ", out);
                        html_escape(out,
                            item->as.union_def.variant_field_names[j][k]);
                        fputs(": <span class=\"ty\">", out);
                        char tbuf[256]; size_t toff = 0; tbuf[0] = '\0';
                        append_type_str(tbuf, sizeof(tbuf), &toff,
                            item->as.union_def.variant_field_types[j][k],
                            item->as.union_def.variant_field_type_names[j][k],
                            TYPE_UNKNOWN, NULL);
                        html_escape(out, tbuf);
                        fputs("</span>", out);
                    }
                    fputs(" }", out);
                }
                fputs("</div>\n", out);
            }
            fputs("</div>\n", out);
            emit_doc_body(out, doc);
            fprintf(out, "</section>\n");
        }
    }

    fprintf(out, "</main>\n</body>\n</html>\n");
    fclose(out);
    docmap_free(&dmap);
    return true;
}
