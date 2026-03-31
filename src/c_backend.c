/*
 * c_backend.c — nanolang → readable C99 transpiler (seL4 / bare-metal target)
 *
 * Produces clean, portable C source from nanolang AST.
 * See c_backend.h for design notes.
 */

#include "c_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>

/* ── String builder ─────────────────────────────────────────────────── */
typedef struct { char *buf; size_t len; size_t cap; } CSB;

static void csb_init(CSB *s) { s->buf = NULL; s->len = 0; s->cap = 0; }
static void csb_free(CSB *s) { free(s->buf); s->buf = NULL; s->len = s->cap = 0; }

static void csb_grow(CSB *s, size_t need) {
    if (s->len + need + 1 <= s->cap) return;
    size_t nc = s->cap ? s->cap * 2 : 8192;
    while (nc < s->len + need + 1) nc *= 2;
    s->buf = realloc(s->buf, nc);
    s->cap = nc;
}
static void csb_append(CSB *s, const char *str) {
    size_t n = strlen(str);
    csb_grow(s, n);
    memcpy(s->buf + s->len, str, n + 1);
    s->len += n;
}
static void csb_appendf(CSB *s, const char *fmt, ...) {
    char tmp[2048];
    va_list ap; va_start(ap, fmt);
    vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    csb_append(s, tmp);
}
static void csb_indent(CSB *s, int depth) {
    for (int i = 0; i < depth * 4; i++) csb_append(s, " ");
}

/* ── Type mapping ────────────────────────────────────────────────────── */
static const char *nano_type_to_c(Type t, const char *struct_name) {
    switch (t) {
        case TYPE_INT:    return "int64_t";
        case TYPE_FLOAT:  return "double";
        case TYPE_BOOL:   return "bool";
        case TYPE_STRING: return "const char*";
        case TYPE_VOID:   return "void";
        case TYPE_STRUCT:
        case TYPE_UNION:
        case TYPE_ENUM:
            return struct_name ? struct_name : "void*";
        default:          return "int64_t";  /* fallback */
    }
}

/* ── Codegen context ─────────────────────────────────────────────────── */
typedef struct {
    CSB          sb;        /* output buffer */
    int          indent;    /* current indentation depth */
    int          tmp_count; /* for generated temporaries */
    int          lbl_count; /* for effect labels */
    bool         err;
    char         errmsg[512];
    const CBOptions *opts;
    const char  *source_file;
} CCtx;

static void c_error(CCtx *ctx, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    vsnprintf(ctx->errmsg, sizeof(ctx->errmsg), fmt, ap);
    va_end(ap);
    ctx->err = true;
}

static void __attribute__((unused)) emit_line(CCtx *ctx, const char *fmt, ...) {
    csb_indent(&ctx->sb, ctx->indent);
    char tmp[2048]; va_list ap; va_start(ap, fmt);
    vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    csb_append(&ctx->sb, tmp);
    csb_append(&ctx->sb, "\n");
}

/* ── Forward declarations ────────────────────────────────────────────── */
static void emit_expr(CCtx *ctx, ASTNode *node);
static void emit_stmt(CCtx *ctx, ASTNode *node);
static void emit_block(CCtx *ctx, ASTNode *block);

/* ── Expression emit ─────────────────────────────────────────────────── */
static void emit_expr(CCtx *ctx, ASTNode *node) {
    if (!node || ctx->err) return;

    switch (node->type) {

    case AST_NUMBER:
        csb_appendf(&ctx->sb, "INT64_C(%lld)", (long long)node->as.number);
        break;

    case AST_FLOAT: {
        /* Use hex float for exact representation */
        char fbuf[64];
        snprintf(fbuf, sizeof(fbuf), "%a", node->as.float_val);
        csb_append(&ctx->sb, fbuf);
        break;
    }

    case AST_BOOL:
        csb_append(&ctx->sb, node->as.bool_val ? "true" : "false");
        break;

    case AST_STRING:
        /* Emit as a C string literal (escape special chars) */
        csb_append(&ctx->sb, "\"");
        for (const char *p = node->as.string_val; p && *p; p++) {
            if (*p == '"')       csb_append(&ctx->sb, "\\\"");
            else if (*p == '\\') csb_append(&ctx->sb, "\\\\");
            else if (*p == '\n') csb_append(&ctx->sb, "\\n");
            else if (*p == '\t') csb_append(&ctx->sb, "\\t");
            else if (*p == '\r') csb_append(&ctx->sb, "\\r");
            else { char ch[2] = {*p, 0}; csb_append(&ctx->sb, ch); }
        }
        csb_append(&ctx->sb, "\"");
        break;

    case AST_IDENTIFIER:
        /* Prefix with nl_ to avoid C keyword collisions */
        csb_appendf(&ctx->sb, "nl_%s", node->as.identifier);
        break;

    case AST_PREFIX_OP: {
        int nargs = node->as.prefix_op.arg_count;
        TokenType op = node->as.prefix_op.op;

        /* Unary ops */
        if (nargs == 1) {
            if (op == TOKEN_MINUS) {
                csb_append(&ctx->sb, "(-");
                emit_expr(ctx, node->as.prefix_op.args[0]);
                csb_append(&ctx->sb, ")");
            } else if (op == TOKEN_NOT) {
                csb_append(&ctx->sb, "(!");
                emit_expr(ctx, node->as.prefix_op.args[0]);
                csb_append(&ctx->sb, ")");
            } else {
                c_error(ctx, "C backend: unsupported unary op %d", (int)op);
            }
            break;
        }

        /* Binary ops */
        if (nargs != 2) { c_error(ctx, "C backend: prefix_op with %d args", nargs); break; }

        const char *cop = NULL;
        switch (op) {
            case TOKEN_PLUS:   cop = "+";  break;
            case TOKEN_MINUS:  cop = "-";  break;
            case TOKEN_STAR:   cop = "*";  break;
            case TOKEN_SLASH:  cop = "/";  break;
            case TOKEN_PERCENT:cop = "%";  break;
            case TOKEN_EQ:     cop = "=="; break;
            case TOKEN_NE:     cop = "!="; break;
            case TOKEN_LT:     cop = "<";  break;
            case TOKEN_LE:     cop = "<="; break;
            case TOKEN_GT:     cop = ">";  break;
            case TOKEN_GE:     cop = ">="; break;
            case TOKEN_AND:    cop = "&&"; break;
            case TOKEN_OR:     cop = "||"; break;
            default:
                c_error(ctx, "C backend: unsupported binary op token %d", (int)op);
                break;
        }
        if (cop) {
            csb_append(&ctx->sb, "(");
            emit_expr(ctx, node->as.prefix_op.args[0]);
            csb_appendf(&ctx->sb, " %s ", cop);
            emit_expr(ctx, node->as.prefix_op.args[1]);
            csb_append(&ctx->sb, ")");
        }
        break;
    }

    case AST_CALL: {
        const char *fname = node->as.call.name;

        /* Built-in mappings */
        if (strcmp(fname, "println") == 0) {
            csb_append(&ctx->sb, "printf(");
            if (node->as.call.arg_count > 0) {
                emit_expr(ctx, node->as.call.args[0]);
                csb_append(&ctx->sb, "\"\\n\"");
            } else {
                csb_append(&ctx->sb, "\"\\n\"");
            }
            csb_append(&ctx->sb, ")");
            break;
        }
        if (strcmp(fname, "print") == 0) {
            csb_append(&ctx->sb, "printf(");
            if (node->as.call.arg_count > 0)
                emit_expr(ctx, node->as.call.args[0]);
            else
                csb_append(&ctx->sb, "\"\"");
            csb_append(&ctx->sb, ")");
            break;
        }
        if (strcmp(fname, "assert") == 0 && node->as.call.arg_count > 0) {
            csb_append(&ctx->sb, "NL_ASSERT(");
            emit_expr(ctx, node->as.call.args[0]);
            csb_append(&ctx->sb, ")");
            break;
        }
        if (strcmp(fname, "int_to_string") == 0 && node->as.call.arg_count > 0) {
            csb_append(&ctx->sb, "nl_int_to_string(");
            emit_expr(ctx, node->as.call.args[0]);
            csb_append(&ctx->sb, ")");
            break;
        }
        if (strcmp(fname, "str_length") == 0 && node->as.call.arg_count > 0) {
            csb_append(&ctx->sb, "((int64_t)strlen(");
            emit_expr(ctx, node->as.call.args[0]);
            csb_append(&ctx->sb, "))");
            break;
        }
        if (strcmp(fname, "str_concat") == 0 && node->as.call.arg_count == 2) {
            csb_append(&ctx->sb, "nl_str_concat(");
            emit_expr(ctx, node->as.call.args[0]);
            csb_append(&ctx->sb, ", ");
            emit_expr(ctx, node->as.call.args[1]);
            csb_append(&ctx->sb, ")");
            break;
        }
        if (strcmp(fname, "str_equals") == 0 && node->as.call.arg_count == 2) {
            csb_append(&ctx->sb, "(strcmp(");
            emit_expr(ctx, node->as.call.args[0]);
            csb_append(&ctx->sb, ", ");
            emit_expr(ctx, node->as.call.args[1]);
            csb_append(&ctx->sb, ") == 0)");
            break;
        }

        /* User-defined / unknown function call */
        csb_appendf(&ctx->sb, "nl_%s(", fname);
        for (int i = 0; i < node->as.call.arg_count; i++) {
            if (i > 0) csb_append(&ctx->sb, ", ");
            emit_expr(ctx, node->as.call.args[i]);
        }
        csb_append(&ctx->sb, ")");
        break;
    }

    case AST_FIELD_ACCESS: {
        /* struct.field → nl_struct.field */
        emit_expr(ctx, node->as.field_access.object);
        csb_appendf(&ctx->sb, ".%s", node->as.field_access.field_name);
        break;
    }

    case AST_STRUCT_LITERAL: {
        /* { field: val, ... } → (StructType){ .field = val, ... } */
        const char *sname = node->as.struct_literal.struct_name;
        csb_appendf(&ctx->sb, "(%s){", sname ? sname : "/*struct*/");
        for (int i = 0; i < node->as.struct_literal.field_count; i++) {
            if (i > 0) csb_append(&ctx->sb, ", ");
            csb_appendf(&ctx->sb, " .%s = ", node->as.struct_literal.field_names[i]);
            emit_expr(ctx, node->as.struct_literal.field_values[i]);
        }
        csb_append(&ctx->sb, " }");
        break;
    }

    case AST_IF: {
        /* Ternary if as expression */
        csb_append(&ctx->sb, "(");
        emit_expr(ctx, node->as.if_stmt.condition);
        csb_append(&ctx->sb, " ? ");
        emit_expr(ctx, node->as.if_stmt.then_branch);
        csb_append(&ctx->sb, " : ");
        if (node->as.if_stmt.else_branch)
            emit_expr(ctx, node->as.if_stmt.else_branch);
        else
            csb_append(&ctx->sb, "0");
        csb_append(&ctx->sb, ")");
        break;
    }

    default:
        c_error(ctx, "C backend: unsupported expr AST type %d", (int)node->type);
        break;
    }
}

/* ── Statement emit ──────────────────────────────────────────────────── */
static void emit_stmt(CCtx *ctx, ASTNode *node) {
    if (!node || ctx->err) return;

    switch (node->type) {

    case AST_LET: {
        const char *ctype = nano_type_to_c(node->as.let.var_type,
                                           node->as.let.type_name);
        csb_indent(&ctx->sb, ctx->indent);
        csb_appendf(&ctx->sb, "%s nl_%s", ctype, node->as.let.name);
        if (node->as.let.value) {
            csb_append(&ctx->sb, " = ");
            emit_expr(ctx, node->as.let.value);
        } else {
            /* Zero-initialise */
            csb_append(&ctx->sb, " = 0");
        }
        csb_append(&ctx->sb, ";\n");
        break;
    }

    case AST_SET: {
        csb_indent(&ctx->sb, ctx->indent);
        csb_appendf(&ctx->sb, "nl_%s = ", node->as.set.name);
        emit_expr(ctx, node->as.set.value);
        csb_append(&ctx->sb, ";\n");
        break;
    }

    case AST_RETURN: {
        csb_indent(&ctx->sb, ctx->indent);
        csb_append(&ctx->sb, "return");
        if (node->as.return_stmt.value) {
            csb_append(&ctx->sb, " ");
            emit_expr(ctx, node->as.return_stmt.value);
        }
        csb_append(&ctx->sb, ";\n");
        break;
    }

    case AST_IF: {
        csb_indent(&ctx->sb, ctx->indent);
        csb_append(&ctx->sb, "if (");
        emit_expr(ctx, node->as.if_stmt.condition);
        csb_append(&ctx->sb, ") {\n");
        ctx->indent++;
        emit_block(ctx, node->as.if_stmt.then_branch);
        ctx->indent--;
        if (node->as.if_stmt.else_branch) {
            csb_indent(&ctx->sb, ctx->indent);
            csb_append(&ctx->sb, "} else {\n");
            ctx->indent++;
            emit_block(ctx, node->as.if_stmt.else_branch);
            ctx->indent--;
        }
        csb_indent(&ctx->sb, ctx->indent);
        csb_append(&ctx->sb, "}\n");
        break;
    }

    case AST_WHILE: {
        csb_indent(&ctx->sb, ctx->indent);
        csb_append(&ctx->sb, "while (");
        emit_expr(ctx, node->as.while_stmt.condition);
        csb_append(&ctx->sb, ") {\n");
        ctx->indent++;
        emit_block(ctx, node->as.while_stmt.body);
        ctx->indent--;
        csb_indent(&ctx->sb, ctx->indent);
        csb_append(&ctx->sb, "}\n");
        break;
    }

    case AST_BLOCK:
        emit_block(ctx, node);
        break;

    case AST_ASSERT: {
        csb_indent(&ctx->sb, ctx->indent);
        csb_append(&ctx->sb, "NL_ASSERT(");
        emit_expr(ctx, node->as.assert.condition);
        csb_append(&ctx->sb, ");\n");
        break;
    }

    default:
        /* Treat as expression statement */
        csb_indent(&ctx->sb, ctx->indent);
        emit_expr(ctx, node);
        csb_append(&ctx->sb, ";\n");
        break;
    }
}

static void emit_block(CCtx *ctx, ASTNode *block) {
    if (!block || ctx->err) return;
    if (block->type == AST_BLOCK) {
        for (int i = 0; i < block->as.block.count && !ctx->err; i++)
            emit_stmt(ctx, block->as.block.statements[i]);
    } else {
        emit_stmt(ctx, block);
    }
}

/* ── Struct/enum definitions ─────────────────────────────────────────── */
static void emit_struct_def(CCtx *ctx, ASTNode *node) {
    if (!node || node->type != AST_STRUCT_DEF) return;
    const char *name = node->as.struct_def.name;
    csb_appendf(&ctx->sb, "typedef struct %s {\n", name);
    for (int i = 0; i < node->as.struct_def.field_count; i++) {
        const char *fname = node->as.struct_def.field_names
                          ? node->as.struct_def.field_names[i] : "?";
        Type ftype = node->as.struct_def.field_types
                   ? node->as.struct_def.field_types[i] : TYPE_INT;
        const char *ftname = (node->as.struct_def.field_type_names &&
                              node->as.struct_def.field_type_names[i])
                           ? node->as.struct_def.field_type_names[i] : NULL;
        const char *ft = nano_type_to_c(ftype, ftname);
        csb_appendf(&ctx->sb, "    %s %s;\n", ft, fname);
    }
    csb_appendf(&ctx->sb, "} %s;\n\n", name);
}

/* ── Function definition ─────────────────────────────────────────────── */
static void emit_function(CCtx *ctx, ASTNode *node) {
    if (!node || node->type != AST_FUNCTION) return;
    if (node->as.function.is_extern) return;  /* Skip extern (C FFI) functions */

    const char *name = node->as.function.name;
    const char *rettype = nano_type_to_c(node->as.function.return_type,
                                          node->as.function.return_struct_type_name);

    /* Function signature */
    csb_appendf(&ctx->sb, "static %s nl_%s(", rettype, name);
    for (int i = 0; i < node->as.function.param_count; i++) {
        if (i > 0) csb_append(&ctx->sb, ", ");
        const char *pt = nano_type_to_c(node->as.function.params[i].type,
                                         node->as.function.params[i].struct_type_name);
        csb_appendf(&ctx->sb, "%s nl_%s", pt, node->as.function.params[i].name);
    }
    csb_append(&ctx->sb, ") {\n");

    /* Body */
    ctx->indent = 1;
    emit_block(ctx, node->as.function.body);
    ctx->indent = 0;

    /* Ensure non-void functions have a fallback return */
    if (node->as.function.return_type != TYPE_VOID) {
        csb_append(&ctx->sb, "    /* fallback */\n");
        if (node->as.function.return_type == TYPE_INT)
            csb_append(&ctx->sb, "    return INT64_C(0);\n");
        else if (node->as.function.return_type == TYPE_FLOAT)
            csb_append(&ctx->sb, "    return 0.0;\n");
        else if (node->as.function.return_type == TYPE_BOOL)
            csb_append(&ctx->sb, "    return false;\n");
        else if (node->as.function.return_type == TYPE_STRING)
            csb_append(&ctx->sb, "    return \"\";\n");
    }

    csb_append(&ctx->sb, "}\n\n");
}

/* ── Program-level emitter ───────────────────────────────────────────── */
static void emit_preamble(CCtx *ctx) {
    csb_appendf(&ctx->sb, "/* nanolang C backend output — %s */\n", ctx->source_file);
    csb_append(&ctx->sb,  "/* Generated by nanoc --target c */\n\n");

    if (!ctx->opts->no_stdlib) {
        csb_append(&ctx->sb,
            "#include <stdint.h>\n"
            "#include <inttypes.h>\n"
            "#include <stdbool.h>\n"
            "#include <string.h>\n"
            "#include <stdio.h>\n"
            "#include <stdlib.h>\n"
            "#include <setjmp.h>\n"
            "\n");
    } else {
        csb_append(&ctx->sb,
            "/* seL4 / bare-metal: no stdlib headers */\n"
            "#include <stdint.h>\n"
            "#include <inttypes.h>\n"
            "#include <stdbool.h>\n"
            "\n");
    }

    /* Minimal runtime macros */
    csb_append(&ctx->sb,
        "#ifndef NL_ASSERT\n"
        "#  ifdef NL_NO_ASSERT\n"
        "#    define NL_ASSERT(x) ((void)(x))\n"
        "#  else\n"
        "#    include <assert.h>\n"
        "#    define NL_ASSERT(x) assert(x)\n"
        "#  endif\n"
        "#endif\n"
        "\n"
        "/* Effect / error handling via setjmp/longjmp */\n"
        "typedef struct { jmp_buf _jb; int64_t _code; } NlEffect;\n"
        "#define NL_EFFECT_HANDLE(e)  (setjmp((e)._jb) == 0)\n"
        "#define NL_EFFECT_RAISE(e,c) do { (e)._code = (c); longjmp((e)._jb, 1); } while(0)\n"
        "\n"
        "/* String helpers */\n"
        "#ifndef nl_str_concat\n"
        "static char *nl_str_concat(const char *a, const char *b) {\n"
        "    size_t la = strlen(a), lb = strlen(b);\n"
        "    char *out = (char*)malloc(la + lb + 1);\n"
        "    if (!out) return \"\";\n"
        "    memcpy(out, a, la); memcpy(out + la, b, lb); out[la+lb] = '\\0';\n"
        "    return out;\n"
        "}\n"
        "#endif\n"
        "#ifndef nl_int_to_string\n"
        "static char *nl_int_to_string(int64_t v) {\n"
        "    char *buf = (char*)malloc(32);\n"
        "    if (buf) snprintf(buf, 32, \"%\" PRId64, v);\n"
        "    return buf ? buf : \"\";\n"
        "}\n"
        "#endif\n"
        "\n"
    );
}

/* ── Public API ──────────────────────────────────────────────────────── */
int c_backend_emit_fp(ASTNode *root, FILE *out,
                      const char *source_file, const CBOptions *opts) {
    if (!root || !out) return 1;

    static const CBOptions default_opts = { false, false, true, false };
    if (!opts) opts = &default_opts;

    CCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    csb_init(&ctx.sb);
    ctx.opts        = opts;
    ctx.source_file = source_file ? source_file : "<unknown>";

    emit_preamble(&ctx);

    /* Two-pass: structs first, then functions (forward compatibility) */
    if (root->type == AST_PROGRAM) {
        /* Pass 1: struct/enum/union definitions */
        for (int i = 0; i < root->as.program.count && !ctx.err; i++) {
            ASTNode *item = root->as.program.items[i];
            if (item && item->type == AST_STRUCT_DEF)
                emit_struct_def(&ctx, item);
        }

        /* Pass 2: function forward declarations */
        for (int i = 0; i < root->as.program.count && !ctx.err; i++) {
            ASTNode *item = root->as.program.items[i];
            if (!item || item->type != AST_FUNCTION) continue;
            if (item->as.function.is_extern) continue;
            const char *name = item->as.function.name;
            const char *rettype = nano_type_to_c(item->as.function.return_type,
                                                  item->as.function.return_struct_type_name);
            csb_appendf(&ctx.sb, "static %s nl_%s(", rettype, name);
            for (int j = 0; j < item->as.function.param_count; j++) {
                if (j > 0) csb_append(&ctx.sb, ", ");
                const char *pt = nano_type_to_c(item->as.function.params[j].type,
                                                 item->as.function.params[j].struct_type_name);
                csb_appendf(&ctx.sb, "%s nl_%s", pt,
                            item->as.function.params[j].name ? item->as.function.params[j].name : "_");
            }
            csb_append(&ctx.sb, ");\n");
        }
        csb_append(&ctx.sb, "\n");

        /* Pass 3: function bodies (skip shadow tests) */
        for (int i = 0; i < root->as.program.count && !ctx.err; i++) {
            ASTNode *item = root->as.program.items[i];
            if (!item) continue;
            if (item->type == AST_FUNCTION)
                emit_function(&ctx, item);
            /* AST_SHADOW skipped — shadow tests are interpreter-only */
        }

        /* Entry point: emit main() wrapper calling nl_main() */
        if (!ctx.opts->no_main) {
            bool has_main = false;
            for (int i = 0; i < root->as.program.count; i++) {
                ASTNode *item = root->as.program.items[i];
                if (item && item->type == AST_FUNCTION &&
                    item->as.function.name &&
                    strcmp(item->as.function.name, "main") == 0) {
                    has_main = true; break;
                }
            }
            if (has_main) {
                csb_append(&ctx.sb,
                    "int main(int argc, char **argv) {\n"
                    "    (void)argc; (void)argv;\n"
                    "    return (int)nl_main();\n"
                    "}\n");
            }
        }
    }

    if (ctx.err) {
        fprintf(stderr, "C backend error: %s\n", ctx.errmsg);
        csb_free(&ctx.sb);
        return 1;
    }

    fputs(ctx.sb.buf, out);

    if (opts->verbose)
        fprintf(stderr, "[c-backend] emitted %zu bytes of C source\n", ctx.sb.len);

    csb_free(&ctx.sb);
    return 0;
}

int c_backend_emit(ASTNode *root, const char *output_path,
                   const char *source_file, const CBOptions *opts) {
    if (!output_path) return c_backend_emit_fp(root, stdout, source_file, opts);
    FILE *f = fopen(output_path, "w");
    if (!f) { fprintf(stderr, "C backend: cannot open %s\n", output_path); return 1; }
    int rc = c_backend_emit_fp(root, f, source_file, opts);
    fclose(f);
    return rc;
}
