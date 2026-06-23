/*
 * docgen_md.c — GitHub-flavored Markdown API doc generator for nanolang
 *
 * Extracts /// doc comments associated with exported declarations and emits
 * a GFM Markdown document suitable for GitHub wiki pages, README sections,
 * or the stdlib docs site.
 *
 * Output format:
 *   # nano stdlib — ModuleName
 *
 *   ## `pub fn name(param: Type) -> ReturnType`
 *   Doc text here.
 *
 *   **Parameters:**
 *   | Name | Type | Description |
 *   |------|------|-------------|
 *   | param | `Type` | description |
 *
 *   **Returns:** description
 *
 *   **Example:**
 *   ```nano
 *   code here
 *   ```
 *
 *   ---
 */

#include "docgen_md.h"
#include "nanolang.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>

/* ── Buffer helpers ──────────────────────────────────────────────────────── */

static void md_buf_appendf(char *buf, size_t cap, size_t *off,
                           const char *fmt, ...) {
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

/* ── Type rendering ──────────────────────────────────────────────────────── */

static void md_append_typeinfo_str(char *buf, size_t cap, size_t *off,
                                   const TypeInfo *ti);

static void md_append_type_str(char *buf, size_t cap, size_t *off,
                               Type t, const char *sname, Type elt,
                               const TypeInfo *ti) {
    switch (t) {
        case TYPE_INT:    md_buf_appendf(buf, cap, off, "int");    return;
        case TYPE_U8:     md_buf_appendf(buf, cap, off, "u8");     return;
        case TYPE_FLOAT:  md_buf_appendf(buf, cap, off, "float");  return;
        case TYPE_BOOL:   md_buf_appendf(buf, cap, off, "bool");   return;
        case TYPE_STRING: md_buf_appendf(buf, cap, off, "string"); return;
        case TYPE_VOID:   md_buf_appendf(buf, cap, off, "void");   return;
        case TYPE_ARRAY:
            md_buf_appendf(buf, cap, off, "array<");
            if (ti && ti->element_type)
                md_append_typeinfo_str(buf, cap, off, ti->element_type);
            else
                md_append_type_str(buf, cap, off, elt, NULL, TYPE_UNKNOWN, NULL);
            md_buf_appendf(buf, cap, off, ">");
            return;
        case TYPE_HASHMAP:
            if (ti) md_append_typeinfo_str(buf, cap, off, ti);
            else     md_buf_appendf(buf, cap, off, "HashMap");
            return;
        case TYPE_STRUCT:
        case TYPE_ENUM:
        case TYPE_UNION:
        case TYPE_OPAQUE:
            if (sname && sname[0])
                md_buf_appendf(buf, cap, off, "%s", sname);
            else if (ti && ti->generic_name)
                md_buf_appendf(buf, cap, off, "%s", ti->generic_name);
            else
                md_buf_appendf(buf, cap, off, "unknown");
            return;
        default:
            md_buf_appendf(buf, cap, off, "unknown");
            return;
    }
}

static void md_append_typeinfo_str(char *buf, size_t cap, size_t *off,
                                   const TypeInfo *ti) {
    if (!ti) { md_buf_appendf(buf, cap, off, "unknown"); return; }

    if (ti->base_type == TYPE_HASHMAP && ti->generic_name &&
        strcmp(ti->generic_name, "HashMap") == 0 && ti->type_param_count == 2) {
        md_buf_appendf(buf, cap, off, "HashMap<");
        md_append_typeinfo_str(buf, cap, off, ti->type_params[0]);
        md_buf_appendf(buf, cap, off, ",");
        md_append_typeinfo_str(buf, cap, off, ti->type_params[1]);
        md_buf_appendf(buf, cap, off, ">");
        return;
    }
    if (ti->base_type == TYPE_ARRAY && ti->element_type) {
        md_buf_appendf(buf, cap, off, "array<");
        md_append_typeinfo_str(buf, cap, off, ti->element_type);
        md_buf_appendf(buf, cap, off, ">");
        return;
    }
    if ((ti->base_type == TYPE_STRUCT || ti->base_type == TYPE_UNION) &&
        ti->generic_name) {
        md_buf_appendf(buf, cap, off, "%s", ti->generic_name);
        if (ti->type_param_count > 0 && ti->type_params) {
            md_buf_appendf(buf, cap, off, "<");
            for (int i = 0; i < ti->type_param_count; i++) {
                if (i > 0) md_buf_appendf(buf, cap, off, ",");
                md_append_typeinfo_str(buf, cap, off, ti->type_params[i]);
            }
            md_buf_appendf(buf, cap, off, ">");
        }
        return;
    }
    md_append_type_str(buf, cap, off, ti->base_type, ti->generic_name,
                       TYPE_UNKNOWN, NULL);
}

/* ── Doc comment extraction ──────────────────────────────────────────────── */

typedef struct {
    int assoc_line; /* First non-/// non-blank line after this block (1-based) */
    char *text;     /* Concatenated comment text with '/// ' prefix stripped   */
} MdDocBlock;

typedef struct {
    MdDocBlock *blocks;
    int count;
    int cap;
} MdDocMap;

static void mddocmap_push(MdDocMap *m, int assoc_line, const char *text) {
    if (m->count >= m->cap) {
        m->cap = m->cap ? m->cap * 2 : 16;
        m->blocks = realloc(m->blocks, sizeof(MdDocBlock) * (size_t)m->cap);
    }
    m->blocks[m->count].assoc_line = assoc_line;
    m->blocks[m->count].text = strdup(text ? text : "");
    m->count++;
}

static void mddocmap_free(MdDocMap *m) {
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
static MdDocMap md_build_doc_map(const char *src) {
    MdDocMap m = {NULL, 0, 0};
    if (!src) return m;

    char accum[16384];
    size_t accum_len = 0;
    int in_doc = 0;
    int line = 1;
    const char *p = src;

    while (*p) {
        const char *eol = p;
        while (*eol && *eol != '\n') eol++;

        const char *lp = p;
        while (lp < eol && (*lp == ' ' || *lp == '\t')) lp++;

        int is_doc_line = (lp + 2 < eol &&
                           lp[0] == '/' && lp[1] == '/' && lp[2] == '/');

        if (is_doc_line) {
            const char *text = lp + 3;
            if (text < eol && *text == ' ') text++;
            size_t tlen = (size_t)(eol - text);
            if (accum_len + tlen + 2 < sizeof(accum)) {
                if (accum_len > 0) accum[accum_len++] = '\n';
                memcpy(accum + accum_len, text, tlen);
                accum_len += tlen;
                accum[accum_len] = '\0';
            }
            in_doc = 1;
        } else if (in_doc) {
            int blank = 1;
            for (const char *c = p; c < eol; c++) {
                if (*c != ' ' && *c != '\t') { blank = 0; break; }
            }
            if (!blank) {
                mddocmap_push(&m, line, accum);
            }
            accum_len = 0;
            accum[0] = '\0';
            in_doc = 0;
        }

        p = (*eol == '\n') ? eol + 1 : eol;
        line++;
    }
    return m;
}

static const char *mddocmap_find(const MdDocMap *m, int decl_line) {
    for (int i = 0; i < m->count; i++) {
        int d = m->blocks[i].assoc_line - decl_line;
        if (d >= 0 && d <= 2) return m->blocks[i].text;
    }
    return NULL;
}

/* ── Param table accumulation ────────────────────────────────────────────── */

typedef struct {
    char name[64];
    char desc[512];
} MdParam;

/* ── Doc body emitter ────────────────────────────────────────────────────── */

/*
 * Render the doc comment text as GFM Markdown.
 * Recognises:
 *   @example ... @end  — fenced ```nano code block
 *   @param <name> ...  — row in a **Parameters:** table
 *   @returns ...       — **Returns:** line
 *   @throws ...        — **Throws:** line
 *   Blank lines        — paragraph separators (pass through as blank lines)
 */
static void emit_md_doc_body(FILE *out, const char *doc) {
    if (!doc || !*doc) return;

    /* Two-pass: first pass collects @param lines, second emits everything.
     * To keep it simple and single-pass, we accumulate @param into a small
     * array and flush the table at the first non-@param line after params. */

    MdParam params[64];
    int param_count = 0;
    int in_example = 0;
    int params_flushed = 0;

    const char *p = doc;

    while (*p) {
        const char *eol = p;
        while (*eol && *eol != '\n') eol++;
        size_t llen = (size_t)(eol - p);

        const char *lp = p;
        while (lp < eol && (*lp == ' ' || *lp == '\t')) lp++;
        size_t ltrimlen = (size_t)(eol - lp);

        if (!in_example && ltrimlen >= 8 && strncmp(lp, "@example", 8) == 0) {
            /* Flush any pending param table */
            if (param_count > 0 && !params_flushed) {
                fputs("\n**Parameters:**\n\n", out);
                fputs("| Name | Type | Description |\n", out);
                fputs("|------|------|-------------|\n", out);
                for (int i = 0; i < param_count; i++) {
                    fprintf(out, "| `%s` | | %s |\n",
                            params[i].name, params[i].desc);
                }
                fputc('\n', out);
                params_flushed = 1;
            }
            in_example = 1;
            fputs("\n**Example:**\n````nano\n", out);

        } else if (in_example &&
                   ltrimlen >= 4 && strncmp(lp, "@end", 4) == 0) {
            in_example = 0;
            fputs("````\n", out);

        } else if (in_example) {
            fwrite(p, 1, llen, out);
            fputc('\n', out);

        } else if (ltrimlen >= 6 && strncmp(lp, "@param", 6) == 0) {
            /* Accumulate param for table */
            if (param_count < 64) {
                const char *rest = lp + 6;
                while (*rest == ' ' || *rest == '\t') rest++;
                /* param name is first token */
                const char *name_end = rest;
                while (*name_end && *name_end != ' ' && *name_end != '\t' &&
                       name_end < eol)
                    name_end++;
                size_t name_len = (size_t)(name_end - rest);
                if (name_len >= sizeof(params[param_count].name))
                    name_len = sizeof(params[param_count].name) - 1;
                memcpy(params[param_count].name, rest, name_len);
                params[param_count].name[name_len] = '\0';
                /* rest is description */
                const char *desc = name_end;
                while (*desc == ' ' || *desc == '\t') desc++;
                size_t desc_len = (size_t)(eol - desc);
                if (desc_len >= sizeof(params[param_count].desc))
                    desc_len = sizeof(params[param_count].desc) - 1;
                memcpy(params[param_count].desc, desc, desc_len);
                params[param_count].desc[desc_len] = '\0';
                param_count++;
            }

        } else if (ltrimlen >= 8 && strncmp(lp, "@returns", 8) == 0) {
            /* Flush param table first */
            if (param_count > 0 && !params_flushed) {
                fputs("\n**Parameters:**\n\n", out);
                fputs("| Name | Type | Description |\n", out);
                fputs("|------|------|-------------|\n", out);
                for (int i = 0; i < param_count; i++) {
                    fprintf(out, "| `%s` | | %s |\n",
                            params[i].name, params[i].desc);
                }
                fputc('\n', out);
                params_flushed = 1;
            }
            const char *rest = lp + 8;
            while (*rest == ' ' || *rest == '\t') rest++;
            size_t rlen = (size_t)(eol - rest);
            fputs("\n**Returns:** ", out);
            fwrite(rest, 1, rlen, out);
            fputc('\n', out);

        } else if (ltrimlen >= 7 && strncmp(lp, "@throws", 7) == 0) {
            if (param_count > 0 && !params_flushed) {
                fputs("\n**Parameters:**\n\n", out);
                fputs("| Name | Type | Description |\n", out);
                fputs("|------|------|-------------|\n", out);
                for (int i = 0; i < param_count; i++) {
                    fprintf(out, "| `%s` | | %s |\n",
                            params[i].name, params[i].desc);
                }
                fputc('\n', out);
                params_flushed = 1;
            }
            const char *rest = lp + 7;
            while (*rest == ' ' || *rest == '\t') rest++;
            size_t rlen = (size_t)(eol - rest);
            fputs("\n**Throws:** ", out);
            fwrite(rest, 1, rlen, out);
            fputc('\n', out);

        } else if (llen == 0) {
            /* Flush param table on blank line if not yet flushed */
            if (param_count > 0 && !params_flushed) {
                fputs("\n**Parameters:**\n\n", out);
                fputs("| Name | Type | Description |\n", out);
                fputs("|------|------|-------------|\n", out);
                for (int i = 0; i < param_count; i++) {
                    fprintf(out, "| `%s` | | %s |\n",
                            params[i].name, params[i].desc);
                }
                fputc('\n', out);
                params_flushed = 1;
            }
            fputc('\n', out);

        } else {
            /* Plain text line */
            if (param_count > 0 && !params_flushed) {
                fputs("\n**Parameters:**\n\n", out);
                fputs("| Name | Type | Description |\n", out);
                fputs("|------|------|-------------|\n", out);
                for (int i = 0; i < param_count; i++) {
                    fprintf(out, "| `%s` | | %s |\n",
                            params[i].name, params[i].desc);
                }
                fputc('\n', out);
                params_flushed = 1;
            }
            fwrite(p, 1, llen, out);
            fputc('\n', out);
        }

        p = (*eol == '\n') ? eol + 1 : eol;
    }

    /* Flush any trailing param table */
    if (param_count > 0 && !params_flushed) {
        fputs("\n**Parameters:**\n\n", out);
        fputs("| Name | Type | Description |\n", out);
        fputs("|------|------|-------------|\n", out);
        for (int i = 0; i < param_count; i++) {
            fprintf(out, "| `%s` | | %s |\n",
                    params[i].name, params[i].desc);
        }
        fputc('\n', out);
    }

    if (in_example) fputs("````\n", out);
}

/* ── Main emitter ────────────────────────────────────────────────────────── */

bool emit_doc_md(const char *output_path, ASTNode *program,
                 const char *source_text, const char *module_name) {
    if (!program || program->type != AST_PROGRAM) return false;

    MdDocMap dmap = md_build_doc_map(source_text);

    FILE *out = fopen(output_path, "w");
    if (!out) {
        fprintf(stderr, "docgen_md: cannot open '%s' for writing: ", output_path);
        perror(NULL);
        mddocmap_free(&dmap);
        return false;
    }

    const char *mod = module_name ? module_name : "module";

    /* Top-level heading */
    fprintf(out, "# nano stdlib \xe2\x80\x94 %s\n\n", mod);

    int first_decl = 1;

    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];

        /* ── Function ── */
        if (item->type == AST_FUNCTION &&
            item->as.function.is_pub &&
            !item->as.function.is_extern &&
            item->as.function.name) {

            const char *name = item->as.function.name;
            const char *doc  = mddocmap_find(&dmap, item->line);

            if (!first_decl) fputs("\n---\n\n", out);
            first_decl = 0;

            /* Build signature string */
            char sig[1024];
            size_t soff = 0;
            sig[0] = '\0';
            md_buf_appendf(sig, sizeof(sig), &soff, "pub fn %s(", name);
            for (int j = 0; j < item->as.function.param_count; j++) {
                Parameter *pp = &item->as.function.params[j];
                if (j > 0) md_buf_appendf(sig, sizeof(sig), &soff, ", ");
                md_buf_appendf(sig, sizeof(sig), &soff, "%s: ",
                               pp->name ? pp->name : "_");
                char tbuf[256]; size_t toff = 0; tbuf[0] = '\0';
                md_append_type_str(tbuf, sizeof(tbuf), &toff,
                                   pp->type, pp->struct_type_name,
                                   pp->element_type, pp->type_info);
                md_buf_appendf(sig, sizeof(sig), &soff, "%s", tbuf);
            }
            md_buf_appendf(sig, sizeof(sig), &soff, ") -> ");
            char rbuf[256]; size_t roff = 0; rbuf[0] = '\0';
            md_append_type_str(rbuf, sizeof(rbuf), &roff,
                               item->as.function.return_type,
                               item->as.function.return_struct_type_name,
                               item->as.function.return_element_type,
                               item->as.function.return_type_info);
            md_buf_appendf(sig, sizeof(sig), &soff, "%s", rbuf);

            fprintf(out, "## `%s`\n\n", sig);
            emit_md_doc_body(out, doc);
            fputc('\n', out);

        /* ── Struct ── */
        } else if (item->type == AST_STRUCT_DEF &&
                   item->as.struct_def.is_pub &&
                   item->as.struct_def.name) {

            const char *name = item->as.struct_def.name;
            const char *doc  = mddocmap_find(&dmap, item->line);

            if (!first_decl) fputs("\n---\n\n", out);
            first_decl = 0;

            fprintf(out, "## `pub struct %s`\n\n", name);
            emit_md_doc_body(out, doc);

            if (item->as.struct_def.field_count > 0) {
                fputs("\n**Fields:**\n\n", out);
                fputs("| Name | Type |\n", out);
                fputs("|------|------|\n", out);
                for (int j = 0; j < item->as.struct_def.field_count; j++) {
                    char tbuf[256]; size_t toff = 0; tbuf[0] = '\0';
                    md_append_type_str(tbuf, sizeof(tbuf), &toff,
                                      item->as.struct_def.field_types[j],
                                      item->as.struct_def.field_type_names[j],
                                      item->as.struct_def.field_element_types[j],
                                      NULL);
                    fprintf(out, "| `%s` | `%s` |\n",
                            item->as.struct_def.field_names[j], tbuf);
                }
                fputc('\n', out);
            }

        /* ── Enum ── */
        } else if (item->type == AST_ENUM_DEF &&
                   item->as.enum_def.is_pub &&
                   item->as.enum_def.name) {

            const char *name = item->as.enum_def.name;
            const char *doc  = mddocmap_find(&dmap, item->line);

            if (!first_decl) fputs("\n---\n\n", out);
            first_decl = 0;

            fprintf(out, "## `pub enum %s`\n\n", name);
            emit_md_doc_body(out, doc);

            fputs("\n**Variants:**\n\n", out);
            for (int j = 0; j < item->as.enum_def.variant_count; j++) {
                fprintf(out, "- `%s`\n", item->as.enum_def.variant_names[j]);
            }
            fputc('\n', out);

        /* ── Union ── */
        } else if (item->type == AST_UNION_DEF &&
                   item->as.union_def.is_pub &&
                   item->as.union_def.name) {

            const char *name = item->as.union_def.name;
            const char *doc  = mddocmap_find(&dmap, item->line);

            if (!first_decl) fputs("\n---\n\n", out);
            first_decl = 0;

            fprintf(out, "## `pub union %s`\n\n", name);
            emit_md_doc_body(out, doc);

            fputs("\n**Variants:**\n\n", out);
            for (int j = 0; j < item->as.union_def.variant_count; j++) {
                fprintf(out, "- `%s", item->as.union_def.variant_names[j]);
                if (item->as.union_def.variant_field_counts &&
                    item->as.union_def.variant_field_counts[j] > 0) {
                    fputs(" { ", out);
                    for (int k = 0;
                         k < item->as.union_def.variant_field_counts[j]; k++) {
                        if (k > 0) fputs(", ", out);
                        fputs(item->as.union_def.variant_field_names[j][k], out);
                        fputs(": ", out);
                        char tbuf[256]; size_t toff = 0; tbuf[0] = '\0';
                        md_append_type_str(
                            tbuf, sizeof(tbuf), &toff,
                            item->as.union_def.variant_field_types[j][k],
                            item->as.union_def.variant_field_type_names[j][k],
                            TYPE_UNKNOWN, NULL);
                        fputs(tbuf, out);
                    }
                    fputs(" }", out);
                }
                fputs("`\n", out);
            }
            fputc('\n', out);
        }
    }

    fclose(out);
    mddocmap_free(&dmap);
    return true;
}
