/*
 * c_backend.c — nanolang C source emit backend
 *
 * Emits readable, self-contained C99 source from the nanolang AST.
 * Supports: numeric types, strings, arithmetic, comparisons, logical ops,
 * function definitions, let/set bindings, if/else, while, for, return,
 * print/println, structs, enums, unions, match, and basic effect stubs.
 *
 * Generated output is C99/C11-compliant and compiles with:
 *   gcc -std=c11 output.c -o output
 *
 * Usage:
 *   nanoc --target c input.nano [-o output.c]
 */

#include "c_backend.h"
#include "binary64_arithmetic_source.h"
#include "binary64_bits.h"
#include "binary64_format.h"
#include "string_literal_decode.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

/* ── Symbol table for local type tracking ───────────────────────────────── */
#define CB_MAX_SYMS   512
#define CB_MAX_SCOPES  64

typedef struct {
    const char *name;
    Type        type;
    const char *nominal;
    const char *variant;
} CBSym;

/* ── Emit context ────────────────────────────────────────────────────────── */
typedef struct {
    FILE       *out;
    ASTNode    *root;
    bool        planning;
    char        prefix[64];
    size_t      operand_slots;
    size_t      string_slots;
    Type        return_type;
    bool        has_globals;
    int         indent;
    bool        verbose;
    const char *error;

    /* Scoped symbol table */
    CBSym syms[CB_MAX_SYMS];
    int   sym_count;
    int   scope_marks[CB_MAX_SCOPES];
    int   scope_depth;
} CBCtx;

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static void ctx_error(CBCtx *c, const char *message) {
    if (!c->error) c->error = message;
}

static void ctx_push_scope(CBCtx *c) {
    if (c->scope_depth < CB_MAX_SCOPES)
        c->scope_marks[c->scope_depth++] = c->sym_count;
    else
        ctx_error(c, "I exceeded my C source lexical scope capacity.");
}

static void ctx_pop_scope(CBCtx *c) {
    if (c->scope_depth > 0)
        c->sym_count = c->scope_marks[--c->scope_depth];
}

static void ctx_add_sym(CBCtx *c, const char *name, Type t) {
    if (c->sym_count < CB_MAX_SYMS) {
        c->syms[c->sym_count].name = name;
        c->syms[c->sym_count].type = t;
        c->syms[c->sym_count].nominal = NULL;
        c->syms[c->sym_count].variant = NULL;
        c->sym_count++;
    } else {
        ctx_error(c, "I exceeded my C source binding capacity.");
    }
}

static void ctx_add_nominal(CBCtx *c, const char *name, Type type,
                            const char *nominal, const char *variant) {
    int before = c->sym_count;
    ctx_add_sym(c, name, type);
    if (c->sym_count != before) {
        c->syms[before].nominal = nominal;
        c->syms[before].variant = variant;
    }
}

static Type ctx_lookup_type(CBCtx *c, const char *name) {
    for (int i = c->sym_count - 1; i >= 0; i--) {
        if (strcmp(c->syms[i].name, name) == 0)
            return c->syms[i].type;
    }
    return TYPE_UNKNOWN;
}

/* Emit n*2 spaces of indentation */
static void emit_indent(CBCtx *c) {
    for (int i = 0; i < c->indent * 2; i++)
        fputc(' ', c->out);
}

/* Map nanolang Type → C type string */
static const char *c_type(Type t) {
    switch (t) {
        case TYPE_INT:    return "int64_t";
        case TYPE_U8:     return "uint8_t";
        case TYPE_FLOAT:  return "double";
        case TYPE_BOOL:   return "int";
        case TYPE_STRING: return "const char*";
        case TYPE_VOID:   return "void";
        default:          return "int64_t";
    }
}

/* I resolve declarations before builtin spellings, never from a name fragment. */
static ASTNode *ctx_function(CBCtx *c, const char *name) {
    if (!c->root || !name) return NULL;
    ASTNode **items = c->root->type == AST_PROGRAM ? c->root->as.program.items : &c->root;
    int count = c->root->type == AST_PROGRAM ? c->root->as.program.count : 1;
    for (int i = 0; i < count; ++i) {
        ASTNode *item = items[i];
        if (item && item->type == AST_ASYNC_FN) item = item->as.async_fn.function;
        if (item && item->type == AST_FUNCTION &&
            strcmp(item->as.function.name, name) == 0) return item;
    }
    return NULL;
}

/* I name payload types by exact declaration and variant indexes. */
static ASTNode *ctx_union_variant(CBCtx *c, const char *name, const char *variant,
                                  int *declaration_index, int *variant_index) {
    if (!c->root || !name || !variant) return NULL;
    ASTNode **items = c->root->type == AST_PROGRAM ? c->root->as.program.items : &c->root;
    int count = c->root->type == AST_PROGRAM ? c->root->as.program.count : 1;
    for (int i = 0; i < count; ++i) {
        ASTNode *item = items[i];
        if (!item || item->type != AST_UNION_DEF || item->as.union_def.is_extern ||
            strcmp(item->as.union_def.name, name) != 0) continue;
        for (int v = 0; v < item->as.union_def.variant_count; ++v) {
            if (strcmp(item->as.union_def.variant_names[v], variant) == 0) {
                *declaration_index = i; *variant_index = v;
                return item;
            }
        }
        return NULL;
    }
    return NULL;
}

/* Only unique, unguarded coverage of one exact declaration is exhaustive here. */
static bool ctx_union_match_complete(CBCtx *c, ASTNode *node) {
    if (!node->as.match_expr.union_type_name || node->as.match_expr.arm_count <= 0)
        return false;
    ASTNode *owner = NULL;
    for (int i = 0; i < node->as.match_expr.arm_count; ++i) {
        if (node->as.match_expr.guard_exprs && node->as.match_expr.guard_exprs[i]) return false;
        int declaration_index, variant_index;
        const char *variant = node->as.match_expr.pattern_variants[i];
        ASTNode *declaration = ctx_union_variant(c, node->as.match_expr.union_type_name,
                                                variant, &declaration_index, &variant_index);
        if (!declaration || (owner && owner != declaration)) return false;
        owner = declaration;
        if (owner->as.union_def.variant_count != node->as.match_expr.arm_count) return false;
        for (int j = 0; j < i; ++j)
            if (strcmp(node->as.match_expr.pattern_variants[j], variant) == 0) return false;
    }
    return true;
}

static bool ctx_has_binding(CBCtx *c, const char *name) {
    if (!name) return false;
    for (int i = c->sym_count - 1; i >= 0; --i)
        if (strcmp(c->syms[i].name, name) == 0) return true;
    return false;
}

/* I separate the language's int64 entry from the hosted C int wrapper. */
static void emit_function_name(CBCtx *c, const char *name, bool lexical) {
    if (name && strcmp(name, "main") == 0 && ctx_function(c, name) &&
        (!lexical || !ctx_has_binding(c, name))) fprintf(c->out, "%sentry", c->prefix);
    else if (name) fputs(name, c->out);
}

/* I resolve absent annotations only from exact scoped declaration identities. */
static Type declared_field_type(CBCtx *c, ASTNode *node, const char **nominal,
                                const char **variant) {
    *nominal = NULL; *variant = NULL;
    if (!node) return TYPE_UNKNOWN;
    if (node->type == AST_IDENTIFIER) {
        for (int i = c->sym_count - 1; i >= 0; --i)
            if (strcmp(c->syms[i].name, node->as.identifier) == 0) {
                *nominal = c->syms[i].nominal;
                *variant = c->syms[i].variant;
                return c->syms[i].type;
            }
    } else if (node->type == AST_STRUCT_LITERAL) {
        *nominal = node->as.struct_literal.struct_name;
        return TYPE_STRUCT;
    } else if (node->type == AST_CALL && node->as.call.name &&
               !node->as.call.func_expr && !ctx_has_binding(c, node->as.call.name)) {
        ASTNode *function = ctx_function(c, node->as.call.name);
        if (function) {
            *nominal = function->as.function.return_struct_type_name;
            return function->as.function.return_type;
        }
    } else if (node->type == AST_FIELD_ACCESS) {
        TypeInfo *checked = node->as.field_access.resolved_type_info;
        if (checked) { *nominal = checked->generic_name; return checked->base_type; }
        const char *owner, *selected;
        Type type = declared_field_type(c, node->as.field_access.object, &owner, &selected);
        if (type != TYPE_STRUCT || !owner || !c->root) return TYPE_UNKNOWN;
        ASTNode **items = c->root->type == AST_PROGRAM ? c->root->as.program.items : &c->root;
        int count = c->root->type == AST_PROGRAM ? c->root->as.program.count : 1;
        for (int i = 0; i < count; ++i) {
            ASTNode *item = items[i];
            if (!item) continue;
            if (!selected && item->type == AST_STRUCT_DEF &&
                strcmp(item->as.struct_def.name, owner) == 0) {
                for (int j = 0; j < item->as.struct_def.field_count; ++j)
                    if (strcmp(item->as.struct_def.field_names[j], node->as.field_access.field_name) == 0) {
                        if (item->as.struct_def.field_type_names)
                            *nominal = item->as.struct_def.field_type_names[j];
                        return item->as.struct_def.field_types[j];
                    }
                return TYPE_UNKNOWN;
            }
            if (selected && item->type == AST_UNION_DEF &&
                strcmp(item->as.union_def.name, owner) == 0) {
                /* I require checked annotations for generic substitutions. */
                if (item->as.union_def.generic_param_count) return TYPE_UNKNOWN;
                for (int v = 0; v < item->as.union_def.variant_count; ++v)
                    if (strcmp(item->as.union_def.variant_names[v], selected) == 0) {
                        for (int j = 0; j < item->as.union_def.variant_field_counts[v]; ++j)
                            if (strcmp(item->as.union_def.variant_field_names[v][j], node->as.field_access.field_name) == 0) {
                                if (item->as.union_def.variant_field_type_names &&
                                    item->as.union_def.variant_field_type_names[v])
                                    *nominal = item->as.union_def.variant_field_type_names[v][j];
                                return item->as.union_def.variant_field_types[v][j];
                            }
                        return TYPE_UNKNOWN;
                    }
            }
        }
    }
    return TYPE_UNKNOWN;
}

/* I retain only resolved scalar types; UNKNOWN is not an integer promise. */
static Type infer_expr_type(CBCtx *c, ASTNode *node) {
    if (!node) return TYPE_UNKNOWN;
    switch (node->type) {
        case AST_NUMBER:     return TYPE_INT;
        case AST_FLOAT:      return TYPE_FLOAT;
        case AST_BOOL:       return TYPE_BOOL;
        case AST_STRING:     return TYPE_STRING;
        case AST_IDENTIFIER: return ctx_lookup_type(c, node->as.identifier);
        case AST_FIELD_ACCESS: {
            const char *nominal, *variant;
            return declared_field_type(c, node, &nominal, &variant);
        }
        case AST_LET:        return node->as.let.var_type;
        case AST_RETURN:     return infer_expr_type(c, node->as.return_stmt.value);
        case AST_CALL: {
            const char *name = node->as.call.name;
            if (node->as.call.checked_signature)
                return node->as.call.checked_signature->return_type;
            if (!name || node->as.call.func_expr || ctx_has_binding(c, name)) return TYPE_UNKNOWN;
            ASTNode *function = ctx_function(c, name);
            if (function) return function->as.function.return_type;
            if (strcmp(name, "float_from_bits") == 0)
                return node->as.call.arg_count == 1 &&
                    infer_expr_type(c, node->as.call.args[0]) == TYPE_INT ? TYPE_FLOAT : TYPE_UNKNOWN;
            if (strcmp(name, "float_to_bits") == 0)
                return node->as.call.arg_count == 1 &&
                    infer_expr_type(c, node->as.call.args[0]) == TYPE_FLOAT ? TYPE_INT : TYPE_UNKNOWN;
            if (strcmp(name, "float_to_string") == 0)
                return node->as.call.arg_count == 1 &&
                    infer_expr_type(c, node->as.call.args[0]) == TYPE_FLOAT ? TYPE_STRING : TYPE_UNKNOWN;
            if (strcmp(name, "str_length") == 0)
                return node->as.call.arg_count == 1 &&
                    infer_expr_type(c, node->as.call.args[0]) == TYPE_STRING ? TYPE_INT : TYPE_UNKNOWN;
            if (strcmp(name, "int_to_string") == 0)
                return node->as.call.arg_count == 1 &&
                    infer_expr_type(c, node->as.call.args[0]) == TYPE_INT ? TYPE_STRING : TYPE_UNKNOWN;
            if (strcmp(name, "str_concat") == 0)
                return node->as.call.arg_count == 2 &&
                    infer_expr_type(c, node->as.call.args[0]) == TYPE_STRING &&
                    infer_expr_type(c, node->as.call.args[1]) == TYPE_STRING ? TYPE_STRING : TYPE_UNKNOWN;
            if (strcmp(name, "bool_to_string") == 0)
                return TYPE_STRING;
            if (strcmp(name, "print") == 0 || strcmp(name, "println") == 0) return TYPE_VOID;
            return TYPE_UNKNOWN;
        }
        case AST_IF: {
            Type then_type = infer_expr_type(c, node->as.if_stmt.then_branch);
            Type else_type = infer_expr_type(c, node->as.if_stmt.else_branch);
            return then_type == else_type ? then_type : TYPE_UNKNOWN;
        }
        case AST_PREFIX_OP: {
            TokenType op = node->as.prefix_op.op;
            if (op == TOKEN_EQ || op == TOKEN_NE || op == TOKEN_LT ||
                op == TOKEN_GT || op == TOKEN_LE || op == TOKEN_GE ||
                op == TOKEN_AND || op == TOKEN_OR || op == TOKEN_NOT)
                return TYPE_BOOL;
            int count = node->as.prefix_op.arg_count;
            if (count < 1 || count > 2) return TYPE_UNKNOWN;
            Type left = infer_expr_type(c, node->as.prefix_op.args[0]);
            if (count == 1) return left;
            Type right = infer_expr_type(c, node->as.prefix_op.args[1]);
            if (left == TYPE_STRING && right == TYPE_STRING && op == TOKEN_PLUS)
                return TYPE_STRING;
            if ((left == TYPE_INT || left == TYPE_FLOAT) &&
                (right == TYPE_INT || right == TYPE_FLOAT))
                return left == TYPE_FLOAT || right == TYPE_FLOAT ? TYPE_FLOAT : TYPE_INT;
            return TYPE_UNKNOWN;
        }
        default: return TYPE_UNKNOWN;
    }
}

/* Return printf format specifier for a type */
static const char *fmt_for_type(Type t) {
    switch (t) {
        case TYPE_FLOAT:  return "%f";
        case TYPE_BOOL:   return "%d";
        case TYPE_STRING: return "%s";
        default:          return "%lld";
    }
}

/* Forward declarations */
static int emit_expr(CBCtx *c, ASTNode *node);
static int emit_stmt(CBCtx *c, ASTNode *node);
static int emit_block_body(CBCtx *c, ASTNode *node);

/* I map only my private helper identifiers, leaving operation bodies unchanged. */
static void emit_private_source(CBCtx *c, const char *source) {
    while (*source) {
        const char *replacement = NULL;
        size_t consumed = 0;
        if (strncmp(source, "nano_rt_", 8) == 0) {
            replacement = ""; consumed = 8;
        } else if (strncmp(source, "nl_float_from_bits", 18) == 0) {
            replacement = "from_bits"; consumed = 18;
        } else if (strncmp(source, "nl_float_to_bits", 16) == 0) {
            replacement = "to_bits"; consumed = 16;
        } else if (strncmp(source, "NANOLANG_BINARY64_ARITHMETIC_H", 30) == 0) {
            replacement = "ARITHMETIC_GUARD"; consumed = 30;
        }
        if (replacement) {
            fputs(c->prefix, c->out);
            fputs(replacement, c->out);
            source += consumed;
        } else {
            fputc(*source++, c->out);
        }
    }
}

static void emit_signed_bits(CBCtx *c, uint64_t bits) {
    if (bits <= INT64_MAX) fprintf(c->out, "INT64_C(%llu)", (unsigned long long)bits);
    else fprintf(c->out, "(-INT64_C(1)-INT64_C(%llu))", (unsigned long long)(UINT64_MAX - bits));
}

/* ── Preamble ─────────────────────────────────────────────────────────────── */
static void emit_preamble(CBCtx *c, const char *source_file) {
    fprintf(c->out, "/* Generated by nanolang --target c");
    if (source_file) fprintf(c->out, " from %s", source_file);
    fprintf(c->out, " */\n");
    fprintf(c->out, "#include <stdio.h>\n");
    fprintf(c->out, "#include <stdlib.h>\n");
    fprintf(c->out, "#include <stdint.h>\n");
    fprintf(c->out, "#include <string.h>\n");
    fprintf(c->out, "#include <setjmp.h>\n\n");

    if (!c->planning) {
        emit_private_source(c, nl_binary64_arithmetic_source);
        emit_private_source(c, NL_BINARY64_BITS_SOURCE);
        emit_private_source(c, NL_BINARY64_FORMAT_SOURCE);
        emit_private_source(c,
            "static int nano_rt_string_equal(const char *left, const char *right) {\n"
            "  return left == right || (left && right && strcmp(left, right) == 0);\n"
            "}\n");
        emit_private_source(c,
            "typedef struct nano_rt_float_text { struct nano_rt_float_text *next; char text[]; } nano_rt_float_text;\n"
            "static nano_rt_float_text *nano_rt_float_text_head;\n"
            "static int nano_rt_float_text_registered;\n"
            "static void nano_rt_float_text_cleanup(void) {\n"
            "  while (nano_rt_float_text_head) {\n"
            "    nano_rt_float_text *node = nano_rt_float_text_head;\n"
            "    nano_rt_float_text_head = node->next; free(node);\n"
            "  }\n"
            "}\n"
            "static char *nano_rt_text_allocate(size_t length) {\n"
            "  if (length > SIZE_MAX - sizeof(nano_rt_float_text) - 1) {\n"
            "    fputs(\"I exceeded my C string allocation size.\\n\", stderr); exit(EXIT_FAILURE);\n"
            "  }\n"
            "  nano_rt_float_text *node = (nano_rt_float_text *)malloc(sizeof *node + length + 1);\n"
            "  if (!node) { fputs(\"I could not allocate my C scalar result.\\n\", stderr); exit(EXIT_FAILURE); }\n"
            "  if (!nano_rt_float_text_registered) {\n"
            "    if (atexit(nano_rt_float_text_cleanup) != 0) {\n"
            "      free(node); fputs(\"I could not register my C scalar cleanup.\\n\", stderr); exit(EXIT_FAILURE);\n"
            "    }\n"
            "    nano_rt_float_text_registered = 1;\n"
            "  }\n"
            "  node->text[length] = 0;\n"
            "  node->next = nano_rt_float_text_head; nano_rt_float_text_head = node;\n"
            "  return node->text;\n"
            "}\n"
            "static const char *nano_rt_scalar_text_copy(const char *text, size_t length) {\n"
            "  char *result = nano_rt_text_allocate(length);\n"
            "  memcpy(result, text, length); return result;\n"
            "}\n"
            "static size_t nano_rt_string_length_sum(size_t left, size_t right) {\n"
            "  if (right > SIZE_MAX - left) {\n"
            "    fputs(\"I exceeded my C concatenation length.\\n\", stderr); exit(EXIT_FAILURE);\n"
            "  }\n"
            "  return left + right;\n"
            "}\n"
            "static const char *nano_rt_string_concat(const char *left, const char *right) {\n"
            "  if (!left || !right) {\n"
            "    fputs(\"I require nonnull C concatenation operands.\\n\", stderr); exit(EXIT_FAILURE);\n"
            "  }\n"
            "  size_t a = strlen(left), b = strlen(right);\n"
            "  char *result = nano_rt_text_allocate(nano_rt_string_length_sum(a, b));\n"
            "  memcpy(result, left, a); memcpy(result + a, right, b); return result;\n"
            "}\n"
            "static const char *nano_rt_float_text_new(double value) {\n"
            "  char text[64]; int length = nano_rt_f64_format(text, sizeof text, value);\n"
            "  if (length < 0 || (size_t)length >= sizeof text) {\n"
            "    fputs(\"I could not format my C float result.\\n\", stderr); exit(EXIT_FAILURE);\n"
            "  }\n"
            "  return nano_rt_scalar_text_copy(text, (size_t)length);\n"
            "}\n"
            "static const char *nano_rt_int_text_new(int64_t value) {\n"
            "  char text[32]; int length = snprintf(text, sizeof text, \"%lld\", (long long)value);\n"
            "  if (length < 0 || (size_t)length >= sizeof text) {\n"
            "    fputs(\"I could not format my C integer result.\\n\", stderr); exit(EXIT_FAILURE);\n"
            "  }\n"
            "  return nano_rt_scalar_text_copy(text, (size_t)length);\n"
            "}\n"
            "static int nano_rt_public_float_print(double value, int newline) {\n"
            "  const char *special = nano_rt_f64_nonfinite(value);\n"
            "  return special ? printf(newline ? \"%s\\n\" : \"%s\", special)\n"
            "                 : printf(newline ? \"%g\\n\" : \"%g\", value);\n"
            "}\n");
    }

    /* nano_bool_to_string helper */
    fprintf(c->out,
        "static const char* nano_bool_to_string(int b) {\n"
        "    return b ? \"true\" : \"false\";\n"
        "}\n\n");

    /* Effect stub support */
    fprintf(c->out,
        "/* Effect handler stubs */\n"
        "static jmp_buf _nano_effect_jmp;\n"
        "static int64_t _nano_effect_val;\n\n");
}

/* I reserve an outer slot before nested operands and sequence both evaluations. */
static int emit_string_concat(CBCtx *c, ASTNode *left, ASTNode *right) {
    if (infer_expr_type(c, left) != TYPE_STRING || infer_expr_type(c, right) != TYPE_STRING) {
        ctx_error(c, "I require two exact STRING operands for C concatenation.");
        return -1;
    }
    if (c->string_slots == SIZE_MAX) {
        ctx_error(c, "I exceeded my string concatenation operand slot capacity.");
        return -1;
    }
    size_t slot = c->string_slots++;
    fprintf(c->out, "(%ssl[%zu] = (", c->prefix, slot);
    if (emit_expr(c, left)) return -1;
    fprintf(c->out, "), %ssr[%zu] = (", c->prefix, slot);
    if (emit_expr(c, right)) return -1;
    fprintf(c->out, "), %sstring_concat(%ssl[%zu], %ssr[%zu]))",
            c->prefix, c->prefix, slot, c->prefix, slot);
    return 0;
}

/* ── Expression emitter ───────────────────────────────────────────────────── */
static int emit_expr(CBCtx *c, ASTNode *node) {
    if (!node) {
        if (c->verbose)
            fprintf(stderr, "[c_backend] null AST node in expression\n");
        ctx_error(c, "I require an expression AST node.");
        return -1;
    }

    switch (node->type) {
    case AST_NUMBER:
        emit_signed_bits(c, (uint64_t)node->as.number);
        return 0;

    case AST_FLOAT: {
        if (c->planning) fprintf(c->out, "%a", node->as.float_val);
        else {
            uint64_t bits;
            memcpy(&bits, &node->as.float_val, sizeof bits);
            fprintf(c->out, "%sfrom_bits(", c->prefix);
            emit_signed_bits(c, bits);
            fputc(')', c->out);
        }
        return 0;
    }

    case AST_BOOL:
        fprintf(c->out, "%d", node->as.bool_val ? 1 : 0);
        return 0;

    case AST_STRING: {
        /* I match canonical source decoding and its existing strlen boundary. */
        char *decoded = nl_decode_string_literal(node->as.string_val ? node->as.string_val : "");
        if (!decoded) {
            ctx_error(c, "I could not decode my C string literal.");
            return -1;
        }
        fputc('"', c->out);
        for (const unsigned char *s = (const unsigned char *)decoded; *s; ++s) {
            switch (*s) {
                case '"':  fputs("\\\"", c->out); break;
                case '\\': fputs("\\\\", c->out); break;
                case '?':  fputs("\\?", c->out); break;
                default:
                    if (*s < 32 || *s >= 127) fprintf(c->out, "\\%03o", (unsigned int)*s);
                    else fputc(*s, c->out);
                    break;
            }
        }
        fputc('"', c->out);
        free(decoded);
        return 0;
    }

    case AST_IDENTIFIER:
        emit_function_name(c, node->as.identifier, true);
        return 0;

    case AST_PREFIX_OP: {
        int argc = node->as.prefix_op.arg_count;
        TokenType op = node->as.prefix_op.op;

        /* Unary not */
        if (argc == 1 && op == TOKEN_NOT) {
            fputc('!', c->out);
            fputc('(', c->out);
            if (emit_expr(c, node->as.prefix_op.args[0])) return -1;
            fputc(')', c->out);
            return 0;
        }

        /* Unary minus */
        if (argc == 1 && op == TOKEN_MINUS) {
            fputc('(', c->out);
            fputc('-', c->out);
            if (emit_expr(c, node->as.prefix_op.args[0])) return -1;
            fputc(')', c->out);
            return 0;
        }

        /* Binary op: check for string concatenation */
        if (argc == 2 && op == TOKEN_PLUS) {
            Type lt = infer_expr_type(c, node->as.prefix_op.args[0]);
            Type rt = infer_expr_type(c, node->as.prefix_op.args[1]);
            if (lt == TYPE_STRING || rt == TYPE_STRING)
                return emit_string_concat(c, node->as.prefix_op.args[0], node->as.prefix_op.args[1]);
        }

        if (argc != 2) {
            ctx_error(c, "I do not support this C prefix-operation arity.");
            return -1;
        }

        Type left_type = infer_expr_type(c, node->as.prefix_op.args[0]);
        Type right_type = infer_expr_type(c, node->as.prefix_op.args[1]);
        if ((op == TOKEN_EQ || op == TOKEN_NE) &&
            (left_type == TYPE_STRING || right_type == TYPE_STRING)) {
            if (left_type != TYPE_STRING || right_type != TYPE_STRING) {
                ctx_error(c, "I require two exact STRING operands for C string equality.");
                return -1;
            }
            if (c->string_slots == SIZE_MAX) {
                ctx_error(c, "I exceeded my string comparison operand slot capacity.");
                return -1;
            }
            size_t slot = c->string_slots++;
            fprintf(c->out, "(%ssl[%zu] = (", c->prefix, slot);
            if (emit_expr(c, node->as.prefix_op.args[0])) return -1;
            fprintf(c->out, "), %ssr[%zu] = (", c->prefix, slot);
            if (emit_expr(c, node->as.prefix_op.args[1])) return -1;
            fprintf(c->out, "), %s%sstring_equal(%ssl[%zu], %ssr[%zu]))",
                    op == TOKEN_NE ? "!" : "", c->prefix, c->prefix, slot, c->prefix, slot);
            return 0;
        }
        if (op == TOKEN_PLUS || op == TOKEN_MINUS || op == TOKEN_STAR || op == TOKEN_SLASH) {
            if (left_type == TYPE_UNKNOWN || right_type == TYPE_UNKNOWN) {
                ctx_error(c, "I require resolved scalar arithmetic operand types.");
                return -1;
            }
            if (left_type == TYPE_FLOAT || right_type == TYPE_FLOAT) {
                if ((left_type != TYPE_FLOAT && left_type != TYPE_INT) ||
                    (right_type != TYPE_FLOAT && right_type != TYPE_INT)) {
                    ctx_error(c, "I require exact numeric operands for C binary64 arithmetic.");
                    return -1;
                }
                if (!c->planning) {
                    if (c->operand_slots == SIZE_MAX) {
                        ctx_error(c, "I exceeded my scalar operand slot capacity.");
                        return -1;
                    }
                    size_t slot = c->operand_slots++;
                    const char *operation = op == TOKEN_PLUS ? "add" : op == TOKEN_MINUS ? "sub" :
                                            op == TOKEN_STAR ? "mul" : "div";
                    fprintf(c->out, "(%sl[%zu] = (", c->prefix, slot);
                    if (emit_expr(c, node->as.prefix_op.args[0])) return -1;
                    fprintf(c->out, "), %sr[%zu] = (", c->prefix, slot);
                    if (emit_expr(c, node->as.prefix_op.args[1])) return -1;
                    fprintf(c->out, "), %sf64_%s(%sl[%zu], %sr[%zu]))",
                            c->prefix, operation, c->prefix, slot, c->prefix, slot);
                    return 0;
                }
            }
        }

        const char *op_str = NULL;
        switch (op) {
            case TOKEN_PLUS:    op_str = "+";  break;
            case TOKEN_MINUS:   op_str = "-";  break;
            case TOKEN_STAR:    op_str = "*";  break;
            case TOKEN_SLASH:   op_str = "/";  break;
            case TOKEN_PERCENT: op_str = "%";  break;
            case TOKEN_EQ:      op_str = "=="; break;
            case TOKEN_NE:      op_str = "!="; break;
            case TOKEN_LT:      op_str = "<";  break;
            case TOKEN_LE:      op_str = "<="; break;
            case TOKEN_GT:      op_str = ">";  break;
            case TOKEN_GE:      op_str = ">="; break;
            case TOKEN_AND:     op_str = "&&"; break;
            case TOKEN_OR:      op_str = "||"; break;
            default:
                ctx_error(c, "I do not support this C binary operator.");
                return -1;
        }

        fputc('(', c->out);
        if (emit_expr(c, node->as.prefix_op.args[0])) return -1;
        fprintf(c->out, " %s ", op_str);
        if (emit_expr(c, node->as.prefix_op.args[1])) return -1;
        fputc(')', c->out);
        return 0;
    }

    case AST_CALL: {
        const char *name = node->as.call.name;
        bool builtin = name && !node->as.call.func_expr && !node->as.call.checked_signature &&
                       !ctx_has_binding(c, name) && !ctx_function(c, name);
        if (name && (strcmp(name, "float_from_bits") == 0 || strcmp(name, "float_to_bits") == 0) &&
            !node->as.call.func_expr && !node->as.call.checked_signature &&
            !ctx_has_binding(c, name) && !ctx_function(c, name)) {
            bool from_bits = strcmp(name, "float_from_bits") == 0;
            if (node->as.call.arg_count != 1 ||
                infer_expr_type(c, node->as.call.args[0]) != (from_bits ? TYPE_INT : TYPE_FLOAT)) {
                ctx_error(c, "I require one exactly typed C binary64 transport operand.");
                return -1;
            }
            fprintf(c->out, "%s%s(", c->planning ? "" : c->prefix,
                    c->planning ? name : from_bits ? "from_bits" : "to_bits");
            if (emit_expr(c, node->as.call.args[0])) return -1;
            fputc(')', c->out);
            return 0;
        }
        /* Handle built-in print/println */
        if (builtin && (strcmp(name, "println") == 0 || strcmp(name, "print") == 0) &&
            node->as.call.arg_count == 1) {
            bool is_println = (strcmp(name, "println") == 0);
            Type t = infer_expr_type(c, node->as.call.args[0]);
            if (t == TYPE_FLOAT) {
                fprintf(c->out, "%spublic_float_print(", c->prefix);
                if (emit_expr(c, node->as.call.args[0])) return -1;
                fprintf(c->out, ", %d)", is_println ? 1 : 0);
                return 0;
            }
            if (t == TYPE_STRING) {
                fprintf(c->out, "printf(\"%%s%s\", ", is_println ? "\\n" : "");
            } else if (t == TYPE_BOOL) {
                fprintf(c->out, "printf(\"%%d%s\", ", is_println ? "\\n" : "");
            } else {
                fprintf(c->out, "printf(\"%%lld%s\", (long long)(", is_println ? "\\n" : "");
            }
            if (emit_expr(c, node->as.call.args[0])) return -1;
            if (t != TYPE_STRING && t != TYPE_FLOAT && t != TYPE_BOOL)
                fputc(')', c->out);
            fputc(')', c->out);
            return 0;
        }
        if (builtin && strcmp(name, "str_concat") == 0) {
            if (node->as.call.arg_count != 2) {
                ctx_error(c, "I require two exact STRING operands for C concatenation.");
                return -1;
            }
            return emit_string_concat(c, node->as.call.args[0], node->as.call.args[1]);
        }
        if (builtin && strcmp(name, "int_to_string") == 0) {
            if (node->as.call.arg_count != 1 ||
                infer_expr_type(c, node->as.call.args[0]) != TYPE_INT) {
                ctx_error(c, "I require one exact INT operand for C int_to_string.");
                return -1;
            }
            fprintf(c->out, "%sint_text_new(", c->prefix);
            if (emit_expr(c, node->as.call.args[0])) return -1;
            fputc(')', c->out);
            return 0;
        }
        if (builtin && strcmp(name, "float_to_string") == 0) {
            if (node->as.call.arg_count != 1 ||
                infer_expr_type(c, node->as.call.args[0]) != TYPE_FLOAT) {
                ctx_error(c, "I require one exact FLOAT operand for C float_to_string.");
                return -1;
            }
            fprintf(c->out, "%sfloat_text_new(", c->prefix);
            if (emit_expr(c, node->as.call.args[0])) return -1;
            fputc(')', c->out);
            return 0;
        }
        if (name && strcmp(name, "bool_to_string") == 0 &&
            node->as.call.arg_count == 1) {
            fputs("nano_bool_to_string(", c->out);
            if (emit_expr(c, node->as.call.args[0])) return -1;
            fputc(')', c->out);
            return 0;
        }
        if (builtin && strcmp(name, "str_length") == 0) {
            if (node->as.call.arg_count != 1 ||
                infer_expr_type(c, node->as.call.args[0]) != TYPE_STRING) {
                ctx_error(c, "I require one exact STRING operand for C str_length.");
                return -1;
            }
            fputs("(int64_t)strlen(", c->out);
            if (emit_expr(c, node->as.call.args[0])) return -1;
            fputc(')', c->out);
            return 0;
        }
        /* Regular function call */
        if (node->as.call.func_expr) {
            fputc('(', c->out);
            if (emit_expr(c, node->as.call.func_expr)) return -1;
            fputc(')', c->out);
        } else if (name) {
            emit_function_name(c, name, true);
        } else {
            ctx_error(c, "I require a named or expression callee.");
            return -1;
        }
        fputc('(', c->out);
        for (int i = 0; i < node->as.call.arg_count; i++) {
            if (i > 0) fputs(", ", c->out);
            if (emit_expr(c, node->as.call.args[i])) return -1;
        }
        fputc(')', c->out);
        return 0;
    }

    case AST_MODULE_QUALIFIED_CALL: {
        /* Emit as alias_function(args) */
        if (node->as.module_qualified_call.module_alias &&
            node->as.module_qualified_call.function_name) {
            fprintf(c->out, "%s_%s",
                    node->as.module_qualified_call.module_alias,
                    node->as.module_qualified_call.function_name);
        } else if (node->as.module_qualified_call.function_name) {
            fputs(node->as.module_qualified_call.function_name, c->out);
        }
        fputc('(', c->out);
        for (int i = 0; i < node->as.module_qualified_call.arg_count; i++) {
            if (i > 0) fputs(", ", c->out);
            if (emit_expr(c, node->as.module_qualified_call.args[i])) return -1;
        }
        fputc(')', c->out);
        return 0;
    }

    case AST_IF: {
        /* Ternary for expression contexts */
        fputc('(', c->out);
        if (emit_expr(c, node->as.if_stmt.condition)) return -1;
        fputs(" ? (", c->out);
        if (emit_expr(c, node->as.if_stmt.then_branch)) return -1;
        fputs(") : (", c->out);
        if (node->as.if_stmt.else_branch) {
            if (emit_expr(c, node->as.if_stmt.else_branch)) return -1;
        } else {
            fputs("0", c->out);
        }
        fputs("))", c->out);
        return 0;
    }

    case AST_BLOCK: {
        /* Inline block (GNU statement expression) */
        fputs("({\n", c->out);
        c->indent++;
        ctx_push_scope(c);
        for (int i = 0; i < node->as.block.count; i++) {
            emit_indent(c);
            if (emit_stmt(c, node->as.block.statements[i])) {
                ctx_pop_scope(c);
                c->indent--;
                return -1;
            }
        }
        ctx_pop_scope(c);
        c->indent--;
        emit_indent(c);
        fputs("})", c->out);
        return 0;
    }

    case AST_RETURN:
        /* Should be handled as a statement; if used as expr: */
        if (node->as.return_stmt.value) {
            return emit_expr(c, node->as.return_stmt.value);
        }
        fputs("0", c->out);
        return 0;

    case AST_STRUCT_LITERAL: {
        /* A dotted struct name ("Union.Variant") is a union variant
         * construction written with struct-literal syntax. The typechecker
         * normalizes module-qualified struct names back to their bare form,
         * so any surviving '.' here identifies a union variant. Emit the
         * tagged-union initializer instead of an invalid dotted struct type. */
        const char *dot = node->as.struct_literal.struct_name
            ? strchr(node->as.struct_literal.struct_name, '.') : NULL;
        if (dot) {
            size_t union_name_len = (size_t)(dot - node->as.struct_literal.struct_name);
            const char *variant_name = dot + 1;
            fprintf(c->out, "(NanoUnion_%.*s){ .tag = NanoUnion_%.*s_TAG_%s",
                    (int)union_name_len, node->as.struct_literal.struct_name,
                    (int)union_name_len, node->as.struct_literal.struct_name,
                    variant_name);
            if (node->as.struct_literal.field_count > 0) {
                fprintf(c->out, ", .as.%s = {", variant_name);
                for (int i = 0; i < node->as.struct_literal.field_count; i++) {
                    if (i > 0) fputs(", ", c->out);
                    fprintf(c->out, ".%s = ", node->as.struct_literal.field_names[i]);
                    if (emit_expr(c, node->as.struct_literal.field_values[i])) return -1;
                }
                fputc('}', c->out);
            }
            fputc('}', c->out);
            return 0;
        }

        /* Emit: (NanoStruct_Name){ .field = val, ... } */
        fprintf(c->out, "(NanoStruct_%s){", node->as.struct_literal.struct_name);
        for (int i = 0; i < node->as.struct_literal.field_count; i++) {
            if (i > 0) fputs(", ", c->out);
            fprintf(c->out, ".%s = ", node->as.struct_literal.field_names[i]);
            if (emit_expr(c, node->as.struct_literal.field_values[i])) return -1;
        }
        fputc('}', c->out);
        return 0;
    }

    case AST_FIELD_ACCESS: {
        if (emit_expr(c, node->as.field_access.object)) return -1;
        fprintf(c->out, ".%s", node->as.field_access.field_name);
        return 0;
    }

    case AST_UNION_CONSTRUCT: {
        fprintf(c->out, "(NanoUnion_%s){ .tag = NanoUnion_%s_TAG_%s",
                node->as.union_construct.union_name,
                node->as.union_construct.union_name,
                node->as.union_construct.variant_name);
        if (node->as.union_construct.field_count > 0) {
            fprintf(c->out, ", .as.%s = {",
                    node->as.union_construct.variant_name);
            for (int i = 0; i < node->as.union_construct.field_count; i++) {
                if (i > 0) fputs(", ", c->out);
                fprintf(c->out, ".%s = ",
                        node->as.union_construct.field_names[i]);
                if (emit_expr(c, node->as.union_construct.field_values[i]))
                    return -1;
            }
            fputc('}', c->out);
        }
        fputc('}', c->out);
        return 0;
    }

    case AST_ARRAY_LITERAL: {
        fputc('{', c->out);
        for (int i = 0; i < node->as.array_literal.element_count; i++) {
            if (i > 0) fputs(", ", c->out);
            if (emit_expr(c, node->as.array_literal.elements[i])) return -1;
        }
        fputc('}', c->out);
        return 0;
    }

    case AST_TUPLE_LITERAL: {
        /* Tuples not fully supported; emit first element as fallback */
        if (node->as.tuple_literal.element_count > 0) {
            return emit_expr(c, node->as.tuple_literal.elements[0]);
        }
        fputs("0", c->out);
        return 0;
    }

    case AST_EFFECT_OP: {
        /* Simplified stub: longjmp to nearest handler */
        fputs("(longjmp(_nano_effect_jmp, 1), 0)", c->out);
        return 0;
    }

    case AST_AWAIT: {
        /* Simplified stub: just evaluate the inner expression */
        return emit_expr(c, node->as.await_expr.expr);
    }

    case AST_TRY_OP: {
        /* Simplified stub: evaluate operand */
        return emit_expr(c, node->as.try_op.operand);
    }

    default:
        if (c->verbose)
            fprintf(stderr, "[c_backend] unsupported expr node type %d\n",
                    node->type);
        ctx_error(c, "I do not support this C expression AST node.");
        return -1;
    }
}

/* ── Statement emitter ────────────────────────────────────────────────────── */
static int emit_stmt(CBCtx *c, ASTNode *node) {
    if (!node) return 0;

    switch (node->type) {
    case AST_LET: {
        Type t = node->as.let.var_type;
        if (t == TYPE_FLOAT && node->as.let.value) {
            Type actual = infer_expr_type(c, node->as.let.value);
            if (actual != TYPE_FLOAT && actual != TYPE_INT) {
                ctx_error(c, "I require a resolved numeric C float initializer.");
                return -1;
            }
        }
        ctx_add_nominal(c, node->as.let.name, t, node->as.let.type_name, NULL);

        if (t == TYPE_STRUCT && node->as.let.type_name) {
            fprintf(c->out, "NanoStruct_%s %s", node->as.let.type_name,
                    node->as.let.name);
        } else if (t == TYPE_UNION && node->as.let.type_name) {
            fprintf(c->out, "NanoUnion_%s %s", node->as.let.type_name,
                    node->as.let.name);
        } else if (t == TYPE_ARRAY) {
            fprintf(c->out, "%s* %s", c_type(node->as.let.element_type),
                    node->as.let.name);
        } else {
            fprintf(c->out, "%s %s", c_type(t), node->as.let.name);
        }
        if (node->as.let.value) {
            fputs(" = ", c->out);
            if (emit_expr(c, node->as.let.value)) return -1;
        }
        fputs(";\n", c->out);
        return 0;
    }

    case AST_SET: {
        if (ctx_lookup_type(c, node->as.set.name) == TYPE_FLOAT) {
            Type actual = infer_expr_type(c, node->as.set.value);
            if (actual != TYPE_FLOAT && actual != TYPE_INT) {
                ctx_error(c, "I require a resolved numeric C float assignment."); return -1;
            }
        }
        fputs(node->as.set.name, c->out);
        fputs(" = ", c->out);
        if (emit_expr(c, node->as.set.value)) return -1;
        fputs(";\n", c->out);
        return 0;
    }

    case AST_RETURN:
        if (c->return_type == TYPE_FLOAT) {
            Type actual = infer_expr_type(c, node->as.return_stmt.value);
            if (actual != TYPE_FLOAT && actual != TYPE_INT) {
                ctx_error(c, "I require a resolved numeric C float return."); return -1;
            }
        }
        fputs("return", c->out);
        if (node->as.return_stmt.value) {
            fputc(' ', c->out);
            if (emit_expr(c, node->as.return_stmt.value)) return -1;
        }
        fputs(";\n", c->out);
        return 0;

    case AST_BREAK:
        fputs("break;\n", c->out);
        return 0;

    case AST_CONTINUE:
        fputs("continue;\n", c->out);
        return 0;

    case AST_PRINT: {
        Type t = infer_expr_type(c, node->as.print.expr);
        if (t == TYPE_FLOAT) {
            fprintf(c->out, "%spublic_float_print(", c->prefix);
            if (emit_expr(c, node->as.print.expr)) return -1;
            fprintf(c->out, ", %d);\n", node->as.print.is_println ? 1 : 0);
            return 0;
        }
        const char *fmt = fmt_for_type(t);
        (void)fmt;
        if (node->as.print.is_println) {
            if (t == TYPE_STRING)
                fprintf(c->out, "printf(\"%%s\\n\", ");
            else if (t == TYPE_BOOL)
                fprintf(c->out, "printf(\"%%d\\n\", ");
            else
                fprintf(c->out, "printf(\"%%lld\\n\", (long long)(");
        } else {
            if (t == TYPE_STRING)
                fprintf(c->out, "printf(\"%%s\", ");
            else if (t == TYPE_BOOL)
                fprintf(c->out, "printf(\"%%d\", ");
            else
                fprintf(c->out, "printf(\"%%lld\", (long long)(");
        }
        if (emit_expr(c, node->as.print.expr)) return -1;
        if (t != TYPE_STRING && t != TYPE_FLOAT && t != TYPE_BOOL)
            fputc(')', c->out);
        fputs(");\n", c->out);
        return 0;
    }

    case AST_ASSERT: {
        fputs("if (!(", c->out);
        if (emit_expr(c, node->as.assert.condition)) return -1;
        fprintf(c->out,
                ")) { fprintf(stderr, \"Assertion failed at line %d\\n\"); exit(1); }\n",
                node->line);
        return 0;
    }

    case AST_IF: {
        fputs("if (", c->out);
        if (emit_expr(c, node->as.if_stmt.condition)) return -1;
        fputs(") {\n", c->out);
        c->indent++;
        ctx_push_scope(c);
        if (emit_block_body(c, node->as.if_stmt.then_branch)) {
            ctx_pop_scope(c); c->indent--;
            return -1;
        }
        ctx_pop_scope(c);
        c->indent--;
        emit_indent(c);
        fputc('}', c->out);
        if (node->as.if_stmt.else_branch) {
            fputs(" else {\n", c->out);
            c->indent++;
            ctx_push_scope(c);
            if (emit_block_body(c, node->as.if_stmt.else_branch)) {
                ctx_pop_scope(c); c->indent--;
                return -1;
            }
            ctx_pop_scope(c);
            c->indent--;
            emit_indent(c);
            fputc('}', c->out);
        }
        fputc('\n', c->out);
        return 0;
    }

    case AST_WHILE: {
        fputs("while (", c->out);
        if (emit_expr(c, node->as.while_stmt.condition)) return -1;
        fputs(") {\n", c->out);
        c->indent++;
        ctx_push_scope(c);
        if (emit_block_body(c, node->as.while_stmt.body)) {
            ctx_pop_scope(c); c->indent--;
            return -1;
        }
        ctx_pop_scope(c);
        c->indent--;
        emit_indent(c);
        fputs("}\n", c->out);
        return 0;
    }

    case AST_FOR: {
        ASTNode *range = node->as.for_stmt.range_expr;
        const char *var = node->as.for_stmt.var_name;
        ctx_add_sym(c, var, TYPE_INT);

        if (range && range->type == AST_PREFIX_OP &&
            range->as.prefix_op.op == TOKEN_RANGE &&
            range->as.prefix_op.arg_count == 2) {
            fprintf(c->out, "for (int64_t %s = ", var);
            if (emit_expr(c, range->as.prefix_op.args[0])) return -1;
            fprintf(c->out, "; %s < ", var);
            if (emit_expr(c, range->as.prefix_op.args[1])) return -1;
            fprintf(c->out, "; %s++) {\n", var);
        } else {
            fprintf(c->out, "for (int64_t %s = 0; %s < ", var, var);
            if (emit_expr(c, range)) return -1;
            fprintf(c->out, "; %s++) {\n", var);
        }
        c->indent++;
        ctx_push_scope(c);
        if (emit_block_body(c, node->as.for_stmt.body)) {
            ctx_pop_scope(c); c->indent--;
            return -1;
        }
        ctx_pop_scope(c);
        c->indent--;
        emit_indent(c);
        fputs("}\n", c->out);
        return 0;
    }

    case AST_BLOCK: {
        fputs("{\n", c->out);
        c->indent++;
        ctx_push_scope(c);
        for (int i = 0; i < node->as.block.count; i++) {
            emit_indent(c);
            if (emit_stmt(c, node->as.block.statements[i])) {
                ctx_pop_scope(c); c->indent--;
                return -1;
            }
        }
        ctx_pop_scope(c);
        c->indent--;
        emit_indent(c);
        fputs("}\n", c->out);
        return 0;
    }

    case AST_MATCH: {
        ASTNode *expr = node->as.match_expr.expr;
        fprintf(c->out, "/* match */ {\n");
        c->indent++;
        emit_indent(c);
        const char *utype = node->as.match_expr.union_type_name;
        if (utype) {
            fprintf(c->out, "NanoUnion_%s %smatch_value = ", utype, c->prefix);
        } else {
            fprintf(c->out, "int64_t %smatch_value = ", c->prefix);
        }
        if (emit_expr(c, expr)) { c->indent--; return -1; }
        fputs(";\n", c->out);

        for (int i = 0; i < node->as.match_expr.arm_count; i++) {
            int declaration_index = 0, variant_index = 0;
            ASTNode *declaration = NULL;
            if (utype) {
                declaration = ctx_union_variant(c, utype, node->as.match_expr.pattern_variants[i],
                                                &declaration_index, &variant_index);
                if (!declaration) {
                    ctx_error(c, "I require an exact declared C union variant for match.");
                    c->indent--; return -1;
                }
            }
            emit_indent(c);
            if (i == 0) fputs("if (", c->out);
            else        fputs("} else if (", c->out);

            if (utype) {
                fprintf(c->out, "%smatch_value.tag == NanoUnion_%s_TAG_%s",
                        c->prefix, utype, node->as.match_expr.pattern_variants[i]);
            } else {
                fprintf(c->out, "1 /* %s */",
                        node->as.match_expr.pattern_variants[i]);
            }

            if (node->as.match_expr.guard_exprs &&
                node->as.match_expr.guard_exprs[i]) {
                fputs(" && (", c->out);
                if (emit_expr(c, node->as.match_expr.guard_exprs[i]))
                    { c->indent--; return -1; }
                fputc(')', c->out);
            }

            fputs(") {\n", c->out);
            c->indent++;
            ctx_push_scope(c);

            if (declaration && node->as.match_expr.pattern_bindings &&
                node->as.match_expr.pattern_bindings[i] &&
                strcmp(node->as.match_expr.pattern_bindings[i], "_") != 0 &&
                declaration->as.union_def.variant_field_counts[variant_index] > 0) {
                emit_indent(c);
                fprintf(c->out, "%spayload_%d_%d %s = %smatch_value.as.%s;\n",
                        c->prefix, declaration_index, variant_index,
                        node->as.match_expr.pattern_bindings[i], c->prefix,
                        node->as.match_expr.pattern_variants[i]);
                ctx_add_nominal(c, node->as.match_expr.pattern_bindings[i], TYPE_STRUCT,
                                utype, node->as.match_expr.pattern_variants[i]);
            }

            emit_indent(c);
            if (emit_stmt(c, node->as.match_expr.arm_bodies[i])) {
                ctx_pop_scope(c); c->indent -= 2;
                return -1;
            }
            ctx_pop_scope(c);
            c->indent--;
        }
        if (node->as.match_expr.arm_count > 0) {
            emit_indent(c);
            if (ctx_union_match_complete(c, node)) {
                fputs("} else {\n", c->out);
                c->indent++;
                emit_indent(c);
                fputs("fputs(\"I require a declared C union tag for match.\\n\", stderr); exit(EXIT_FAILURE);\n", c->out);
                c->indent--;
                emit_indent(c);
            }
            fputs("}\n", c->out);
        }
        c->indent--;
        emit_indent(c);
        fputs("}\n", c->out);
        return 0;
    }

    case AST_HANDLE_EXPR: {
        fputs("/* handle */ if (setjmp(_nano_effect_jmp) == 0) {\n", c->out);
        c->indent++;
        if (emit_block_body(c, node->as.handle_expr.body)) {
            c->indent--;
            return -1;
        }
        c->indent--;
        emit_indent(c);
        fputs("}\n", c->out);
        return 0;
    }

    case AST_EFFECT_DECL:
        fputs("/* effect declaration (stub) */\n", c->out);
        return 0;

    case AST_SHADOW:
        fputs("/* shadow test (skipped) */\n", c->out);
        return 0;

    case AST_IMPORT:
    case AST_MODULE_DECL:
    case AST_OPAQUE_TYPE:
        return 0;

    case AST_PAR_BLOCK: {
        ASTNode **bindings = node->as.par_block.bindings;
        int cnt = node->as.par_block.count;
        int *order = node->as.par_block.is_flow ? passive_binding_order(node) : NULL;
        if (node->as.par_block.is_flow && !order) return -1;
        int status = 0;
        for (int i = 0; i < cnt && !status; i++) {
            emit_indent(c);
            status = emit_stmt(c, bindings[order ? order[i] : i]);
        }
        free(order);
        return status;
    }
    case AST_PAR_LET: {
        /* par-let: emit each binding value as a statement, then the body */
        int cnt = node->as.par_let.count;
        for (int i = 0; i < cnt; i++) {
            emit_indent(c);
            if (emit_stmt(c, node->as.par_let.values[i])) return -1;
        }
        if (node->as.par_let.body) {
            emit_indent(c);
            if (emit_stmt(c, node->as.par_let.body)) return -1;
        }
        return 0;
    }

    case AST_UNSAFE_BLOCK: {
        fputs("/* unsafe */ {\n", c->out);
        c->indent++;
        ctx_push_scope(c);
        for (int i = 0; i < node->as.unsafe_block.count; i++) {
            emit_indent(c);
            if (emit_stmt(c, node->as.unsafe_block.statements[i])) {
                ctx_pop_scope(c); c->indent--;
                return -1;
            }
        }
        ctx_pop_scope(c);
        c->indent--;
        emit_indent(c);
        fputs("}\n", c->out);
        return 0;
    }

    case AST_COND: {
        for (int i = 0; i < node->as.cond_expr.clause_count; i++) {
            emit_indent(c);
            if (i == 0) fputs("if (", c->out);
            else        fputs("} else if (", c->out);
            if (emit_expr(c, node->as.cond_expr.conditions[i])) return -1;
            fputs(") {\n", c->out);
            c->indent++;
            ctx_push_scope(c);
            emit_indent(c);
            fputs("return ", c->out);
            if (emit_expr(c, node->as.cond_expr.values[i])) {
                ctx_pop_scope(c); c->indent--;
                return -1;
            }
            fputs(";\n", c->out);
            ctx_pop_scope(c);
            c->indent--;
        }
        if (node->as.cond_expr.clause_count > 0) {
            emit_indent(c);
            fputs("} else {\n", c->out);
            c->indent++;
            emit_indent(c);
            fputs("return ", c->out);
            if (emit_expr(c, node->as.cond_expr.else_value)) {
                c->indent--;
                return -1;
            }
            fputs(";\n", c->out);
            c->indent--;
            emit_indent(c);
            fputs("}\n", c->out);
        }
        return 0;
    }

    /* Expressions used as statements */
    default: {
        if (emit_expr(c, node)) return -1;
        fputs(";\n", c->out);
        return 0;
    }
    }
}

/*
 * emit_block_body — emit the contents of a block node (or single statement)
 * without surrounding braces.
 */
static int emit_block_body(CBCtx *c, ASTNode *node) {
    if (!node) return 0;
    if (node->type == AST_BLOCK) {
        for (int i = 0; i < node->as.block.count; i++) {
            emit_indent(c);
            if (emit_stmt(c, node->as.block.statements[i])) return -1;
        }
        return 0;
    }
    emit_indent(c);
    return emit_stmt(c, node);
}

/* ── Type definitions ─────────────────────────────────────────────────────── */
static void emit_struct_def(CBCtx *c, ASTNode *node) {
    if (node->as.struct_def.is_extern) return;
    fprintf(c->out, "typedef struct {\n");
    for (int i = 0; i < node->as.struct_def.field_count; i++) {
        Type ft = node->as.struct_def.field_types[i];
        const char *fname = node->as.struct_def.field_names[i];
        fputs("  ", c->out);
        if (ft == TYPE_STRUCT && node->as.struct_def.field_type_names &&
            node->as.struct_def.field_type_names[i]) {
            fprintf(c->out, "NanoStruct_%s %s;\n",
                    node->as.struct_def.field_type_names[i], fname);
        } else if (ft == TYPE_ARRAY) {
            Type et = node->as.struct_def.field_element_types
                    ? node->as.struct_def.field_element_types[i]
                    : TYPE_INT;
            fprintf(c->out, "%s* %s;\n", c_type(et), fname);
        } else {
            fprintf(c->out, "%s %s;\n", c_type(ft), fname);
        }
    }
    fprintf(c->out, "} NanoStruct_%s;\n\n", node->as.struct_def.name);
}

static void emit_enum_def(CBCtx *c, ASTNode *node) {
    if (node->as.enum_def.is_extern) return;
    fprintf(c->out, "typedef enum {\n");
    for (int i = 0; i < node->as.enum_def.variant_count; i++) {
        fputs("  ", c->out);
        fprintf(c->out, "NanoEnum_%s_%s",
                node->as.enum_def.name, node->as.enum_def.variant_names[i]);
        if (node->as.enum_def.variant_values) {
            fprintf(c->out, " = %d", node->as.enum_def.variant_values[i]);
        }
        if (i < node->as.enum_def.variant_count - 1) fputc(',', c->out);
        fputc('\n', c->out);
    }
    fprintf(c->out, "} NanoEnum_%s;\n\n", node->as.enum_def.name);
}

static void emit_union_def(CBCtx *c, ASTNode *node) {
    if (node->as.union_def.is_extern) return;
    const char *uname = node->as.union_def.name;
    bool has_payload = false;
    for (int v = 0; v < node->as.union_def.variant_count; v++) {
        int fc = node->as.union_def.variant_field_counts[v];
        if (fc == 0) continue;
        int declaration_index, variant_index;
        if (ctx_union_variant(c, uname, node->as.union_def.variant_names[v],
                              &declaration_index, &variant_index) != node) {
            ctx_error(c, "I require an exact declared C union payload type."); return;
        }
        has_payload = true;
        fprintf(c->out, "typedef struct {\n");
        for (int f = 0; f < fc; f++) {
            Type ft = node->as.union_def.variant_field_types[v][f];
            const char *field = node->as.union_def.variant_field_names[v][f];
            fprintf(c->out, "  %s %s;\n", c_type(ft), field);
        }
        fprintf(c->out, "} %spayload_%d_%d;\n\n", c->prefix, declaration_index, variant_index);
    }
    fprintf(c->out, "typedef enum {\n");
    for (int i = 0; i < node->as.union_def.variant_count; i++) {
        fprintf(c->out, "  NanoUnion_%s_TAG_%s%s\n",
                uname, node->as.union_def.variant_names[i],
                (i < node->as.union_def.variant_count - 1) ? "," : "");
    }
    fprintf(c->out, "} NanoUnion_%s_Tag;\n\n", uname);

    fprintf(c->out, "typedef struct {\n");
    fprintf(c->out, "  NanoUnion_%s_Tag tag;\n", uname);
    fprintf(c->out, "  union {\n");
    for (int v = 0; v < node->as.union_def.variant_count; v++) {
        if (node->as.union_def.variant_field_counts[v] == 0) continue;
        int declaration_index, variant_index;
        if (ctx_union_variant(c, uname, node->as.union_def.variant_names[v],
                              &declaration_index, &variant_index) != node) {
            ctx_error(c, "I require an exact declared C union payload type."); return;
        }
        fprintf(c->out, "    %spayload_%d_%d %s;\n", c->prefix,
                declaration_index, variant_index, node->as.union_def.variant_names[v]);
    }
    if (!has_payload) fprintf(c->out, "    unsigned char %sempty;\n", c->prefix);
    fprintf(c->out, "  } as;\n");
    fprintf(c->out, "} NanoUnion_%s;\n\n", uname);
}

/* ── Function emitter ─────────────────────────────────────────────────────── */
static int emit_staged_body(CBCtx *c, FILE *body, FILE *destination) {
    if (c->operand_slots) fprintf(c->out, "  double %sl[%zu], %sr[%zu];\n",
                                  c->prefix, c->operand_slots, c->prefix, c->operand_slots);
    if (c->string_slots) fprintf(c->out, "  const char *%ssl[%zu], *%ssr[%zu];\n",
                                 c->prefix, c->string_slots, c->prefix, c->string_slots);
    if (fflush(body) != 0 || fseek(body, 0, SEEK_SET) != 0) {
        ctx_error(c, "I could not rewind a C function body."); fclose(body); return -1;
    }
    char chunk[8192];
    size_t got;
    while ((got = fread(chunk, 1, sizeof chunk, body)) != 0) {
        if (fwrite(chunk, 1, got, destination) != got) {
            ctx_error(c, "I could not publish a staged C function body."); break;
        }
    }
    if (ferror(body)) ctx_error(c, "I could not read a staged C function body.");
    if (fclose(body) != 0) ctx_error(c, "I could not close a staged C function body.");
    if (c->error) return -1;
    return 0;
}

static int emit_function(CBCtx *c, ASTNode *node) {
    if (node->as.function.is_extern) return 0;

    Type ret = node->as.function.return_type;
    c->return_type = ret;
    const char *ret_type_str;
    static char struct_ret[128];
    if (ret == TYPE_STRUCT && node->as.function.return_struct_type_name) {
        snprintf(struct_ret, sizeof(struct_ret), "NanoStruct_%s",
                 node->as.function.return_struct_type_name);
        ret_type_str = struct_ret;
    } else {
        ret_type_str = c_type(ret);
    }

    fprintf(c->out, "%s ", ret_type_str);
    emit_function_name(c, node->as.function.name, false);
    fputc('(', c->out);
    for (int i = 0; i < node->as.function.param_count; i++) {
        if (i > 0) fputs(", ", c->out);
        Parameter *p = &node->as.function.params[i];
        if (p->type == TYPE_STRUCT && p->struct_type_name) {
            fprintf(c->out, "NanoStruct_%s %s", p->struct_type_name, p->name);
        } else if (p->type == TYPE_UNION && p->struct_type_name) {
            fprintf(c->out, "NanoUnion_%s %s", p->struct_type_name, p->name);
        } else {
            fprintf(c->out, "%s %s", c_type(p->type), p->name);
        }
    }
    if (node->as.function.param_count == 0) fputs("void", c->out);
    fputs(") {\n", c->out);
    FILE *destination = c->out;
    FILE *body = tmpfile();
    if (!body) { ctx_error(c, "I could not stage a C function body."); return -1; }
    c->out = body;
    c->operand_slots = 0;
    c->string_slots = 0;

    ctx_push_scope(c);
    for (int i = 0; i < node->as.function.param_count; i++) {
        Parameter *p = &node->as.function.params[i];
        ctx_add_nominal(c, p->name, p->type, p->struct_type_name, NULL);
    }

    c->indent = 1;
    if (c->has_globals && strcmp(node->as.function.name, "main") == 0)
        fprintf(c->out, "  %sinit();\n", c->prefix);
    if (node->as.function.body) {
        if (node->as.function.body->type == AST_BLOCK) {
            for (int i = 0; i < node->as.function.body->as.block.count; i++) {
                emit_indent(c);
                if (emit_stmt(c, node->as.function.body->as.block.statements[i])) {
                    ctx_pop_scope(c);
                    fclose(body); c->out = destination;
                    return -1;
                }
            }
        } else {
            emit_indent(c);
            if (emit_stmt(c, node->as.function.body)) {
                ctx_pop_scope(c);
                fclose(body); c->out = destination;
                return -1;
            }
        }
    }

    ctx_pop_scope(c);
    c->out = destination;
    if (emit_staged_body(c, body, destination)) return -1;
    c->indent = 0;
    fputs("}\n\n", c->out);
    return 0;
}

/* ── Forward declarations ─────────────────────────────────────────────────── */
static void emit_forward_decls(CBCtx *c, ASTNode *root) {
    if (!root) return;
    ASTNode **items = NULL;
    int count = 0;
    if (root->type == AST_PROGRAM) {
        items = root->as.program.items;
        count = root->as.program.count;
    } else {
        items = &root;
        count = 1;
    }
    for (int i = 0; i < count; i++) {
        ASTNode *n = items[i];
        if (!n || n->type != AST_FUNCTION) continue;
        /* Emit prototypes for both regular and extern functions.
         * Without this, extern fn declarations (e.g. fs_mkdir_p, path_relpath,
         * file_copy, dir_copy) have no C prototype visible to the compiler,
         * causing -Wimplicit-function-declaration and the cascading
         * -Wint-conversion errors that follow from the defaulted int return. */
        Type ret = n->as.function.return_type;
        const char *ret_str;
        static char sbuf[128];
        if (ret == TYPE_STRUCT && n->as.function.return_struct_type_name) {
            snprintf(sbuf, sizeof(sbuf), "NanoStruct_%s",
                     n->as.function.return_struct_type_name);
            ret_str = sbuf;
        } else {
            ret_str = c_type(ret);
        }
        fprintf(c->out, "%s ", ret_str);
        emit_function_name(c, n->as.function.name, false);
        fputc('(', c->out);
        for (int j = 0; j < n->as.function.param_count; j++) {
            if (j > 0) fputs(", ", c->out);
            Parameter *p = &n->as.function.params[j];
            if (p->type == TYPE_STRUCT && p->struct_type_name)
                fprintf(c->out, "NanoStruct_%s %s", p->struct_type_name, p->name);
            else if (p->type == TYPE_UNION && p->struct_type_name)
                fprintf(c->out, "NanoUnion_%s %s", p->struct_type_name, p->name);
            else
                fprintf(c->out, "%s %s", c_type(p->type), p->name);
        }
        if (n->as.function.param_count == 0) fputs("void", c->out);
        fputs(");\n", c->out);
    }
    if (count > 0) fputc('\n', c->out);
}

static int emit_global_initializer(CBCtx *c, ASTNode **items, int count) {
    FILE *destination = c->out;
    FILE *body = tmpfile();
    if (!body) { ctx_error(c, "I could not stage scalar global initialization."); return -1; }
    c->out = body;
    c->operand_slots = 0;
    c->string_slots = 0;
    c->return_type = TYPE_VOID;
    fprintf(c->out, "  static int %sinitialized;\n  if (%sinitialized) return;\n  %sinitialized = 1;\n",
            c->prefix, c->prefix, c->prefix);
    for (int i = 0; i < count; ++i) {
        ASTNode *item = items[i];
        if (!item || item->type != AST_LET || !item->as.let.value) continue;
        Type expected = item->as.let.var_type;
        Type actual = infer_expr_type(c, item->as.let.value);
        if (actual == TYPE_UNKNOWN || (actual != expected && !(expected == TYPE_FLOAT && actual == TYPE_INT))) {
            ctx_error(c, "I require a resolved compatible scalar global initializer.");
            break;
        }
        fprintf(c->out, "  %s = ", item->as.let.name);
        if (emit_expr(c, item->as.let.value)) break;
        fputs(";\n", c->out);
    }
    c->out = destination;
    if (c->error) { fclose(body); return -1; }
    fprintf(c->out, "static void %sinit(void) {\n", c->prefix);
    if (emit_staged_body(c, body, destination)) return -1;
    fputs("}\n\n", c->out);
    return 0;
}

/* ── Top-level program emitter ───────────────────────────────────────────── */
static int emit_program(CBCtx *c, ASTNode *root) {
    ASTNode **items = NULL;
    int count = 0;

    if (root->type == AST_PROGRAM) {
        items = root->as.program.items;
        count = root->as.program.count;
    } else {
        items = &root;
        count = 1;
    }

    /* Pass 1: type definitions */
    for (int i = 0; i < count; i++) {
        ASTNode *n = items[i];
        if (!n) continue;
        switch (n->type) {
            case AST_STRUCT_DEF: emit_struct_def(c, n); break;
            case AST_ENUM_DEF:   emit_enum_def(c, n);   break;
            case AST_UNION_DEF:  emit_union_def(c, n);  break;
            default: break;
        }
    }

    /* I declare exact scalar storage before prototypes and ordered startup. */
    c->has_globals = false;
    for (int i = 0; i < count; ++i) {
        ASTNode *item = items[i];
        if (!item || item->type != AST_LET) continue;
        Type type = item->as.let.var_type;
        if (type != TYPE_INT && type != TYPE_FLOAT && type != TYPE_BOOL && type != TYPE_STRING) {
            ctx_error(c, "I require a supported exact scalar C global binding."); return -1;
        }
        ctx_add_sym(c, item->as.let.name, type);
        fprintf(c->out, "static %s %s;\n", c_type(type), item->as.let.name);
        c->has_globals = true;
    }
    if (c->has_globals && !ctx_function(c, "main")) {
        ctx_error(c, "I require a hosted main for ordered C global initialization."); return -1;
    }
    ASTNode *main_function = ctx_function(c, "main");
    if (main_function && (main_function->as.function.return_type != TYPE_INT || main_function->as.function.param_count != 0)) {
        ctx_error(c, "I require main()->int for this hosted C entry."); return -1;
    }

    /* Pass 2: forward declarations precede calls in global initializers. */
    emit_forward_decls(c, root);
    if (c->has_globals && emit_global_initializer(c, items, count)) return -1;

    /* Pass 3: function definitions */
    for (int i = 0; i < count; i++) {
        ASTNode *n = items[i];
        if (!n) continue;
        switch (n->type) {
            case AST_ASYNC_FN:
                if (n->as.async_fn.function)
                    n = n->as.async_fn.function;
                /* fall through */
            case AST_FUNCTION:
                if (emit_function(c, n)) return -1;
                break;
            default:
                break;
        }
    }
    if (main_function)
        fprintf(c->out, "int main(void) { return (int)%sentry(); }\n", c->prefix);
    return 0;
}

/* ── Public API ──────────────────────────────────────────────────────────── */
static int render_source(ASTNode *root, FILE *out, const char *source_file,
                         const CBOptions *opts) {
    CBCtx c;
    memset(&c, 0, sizeof(c));
    c.out = out;
    c.root = root;
    c.verbose = opts ? opts->verbose : false;
    snprintf(c.prefix, sizeof c.prefix, "nano_cb_plan_");
    if (!root) ctx_error(&c, "I require a program AST.");
    FILE *plan = NULL;
    char *text = NULL;
    if (!c.error) {
        plan = tmpfile();
        if (!plan) ctx_error(&c, "I could not stage my C namespace plan.");
    }
    if (!c.error) {
        c.out = plan;
        c.planning = true;
        if (emit_program(&c, root) != 0 && !c.error)
            ctx_error(&c, "I could not plan this C program.");
        if (opts && opts->no_main && c.has_globals)
            ctx_error(&c, "I require an initialization entry for scalar C globals.");
        long length = -1;
        if (!c.error && fflush(plan) == 0) length = ftell(plan);
        if (length < 0 || (uintmax_t)length >= SIZE_MAX)
            ctx_error(&c, "I could not measure my C namespace plan.");
        if (!c.error) {
            text = malloc((size_t)length + 1);
            if (!text || fseek(plan, 0, SEEK_SET) != 0)
                ctx_error(&c, "I could not retain my C namespace plan.");
        }
        if (!c.error) {
            if (fread(text, 1, (size_t)length, plan) != (size_t)length)
                ctx_error(&c, "I could not read my C namespace plan.");
            else text[length] = '\0';
        }
    }
    if (plan && fclose(plan) != 0) ctx_error(&c, "I could not close my C namespace plan.");
    if (!c.error) {
        size_t counter = 0;
        for (;;) {
            snprintf(c.prefix, sizeof c.prefix, "nano_cb_%zu_", counter);
            if (!strstr(text, c.prefix)) break;
            if (counter == SIZE_MAX) { ctx_error(&c, "I exhausted private C namespaces."); break; }
            ++counter;
        }
    }
    free(text);
    c.out = out;
    c.planning = false;
    c.indent = 0;
    c.sym_count = 0;
    c.scope_depth = 0;
    c.operand_slots = 0;
    c.string_slots = 0;
    if (!c.error) {
        emit_preamble(&c, source_file);
        if (emit_program(&c, root) != 0 && !c.error)
            ctx_error(&c, "I could not emit this C program.");
    }
    if (ferror(out)) ctx_error(&c, "I could not write staged C source.");
    if (c.error) {
        fprintf(stderr, "[c_backend] %s\n", c.error);
        return 1;
    }
    return 0;
}

int c_backend_emit_fp(ASTNode *root, FILE *out, const char *source_file,
                      const CBOptions *opts) {
    if (!out) { fprintf(stderr, "[c_backend] I require an output stream.\n"); return 1; }
    FILE *staged = tmpfile();
    if (!staged) {
        fprintf(stderr, "[c_backend] I could not stage C source.\n");
        return 1;
    }
    int rc = render_source(root, staged, source_file, opts);
    if (!rc && (fflush(staged) != 0 || fseek(staged, 0, SEEK_SET) != 0)) {
        fprintf(stderr, "[c_backend] I could not rewind staged source.\n"); rc = 1;
    }
    char buffer[8192];
    while (!rc) {
        size_t count = fread(buffer, 1, sizeof buffer, staged);
        if (count && fwrite(buffer, 1, count, out) != count) {
            fprintf(stderr, "[c_backend] I could not write the output stream.\n"); rc = 1;
        }
        if (count < sizeof buffer) {
            if (ferror(staged) && !rc) {
                fprintf(stderr, "[c_backend] I could not read staged source.\n"); rc = 1;
            }
            break;
        }
    }
    if (!rc && fflush(out) != 0) {
        fprintf(stderr, "[c_backend] I could not flush the output stream.\n"); rc = 1;
    }
    if (fclose(staged) != 0 && !rc) {
        fprintf(stderr, "[c_backend] I could not close staged source.\n"); rc = 1;
    }
    return rc;
}

int c_backend_emit(ASTNode *root, const char *output_path,
                   const char *source_file, const CBOptions *opts) {
    if (!output_path) { fprintf(stderr, "[c_backend] I require an output path.\n"); return 1; }
    size_t length = strlen(output_path);
    static const char suffix[] = ".tmp.XXXXXX";
    if (length > SIZE_MAX - sizeof suffix) {
        fprintf(stderr, "[c_backend] I cannot represent the staging path.\n"); return 1;
    }
    char *temporary = malloc(length + sizeof suffix);
    if (!temporary) { fprintf(stderr, "[c_backend] I could not allocate the staging path.\n"); return 1; }
    memcpy(temporary, output_path, length);
    memcpy(temporary + length, suffix, sizeof suffix);
    int descriptor = mkstemp(temporary);
    if (descriptor < 0) {
        fprintf(stderr, "[c_backend] I could not stage output for %s.\n", output_path);
        free(temporary);
        return 1;
    }
    FILE *staged = fdopen(descriptor, "w");
    if (!staged) {
        fprintf(stderr, "[c_backend] I could not open the staged output stream.\n");
        close(descriptor);
        unlink(temporary);
        free(temporary);
        return 1;
    }
    int rc = render_source(root, staged, source_file, opts);
    if (fclose(staged) != 0 && !rc) {
        fprintf(stderr, "[c_backend] I could not close staged output.\n"); rc = 1;
    }
    if (!rc && rename(temporary, output_path) != 0) {
        fprintf(stderr, "[c_backend] I could not publish staged output.\n"); rc = 1;
    }
    if (rc) unlink(temporary);
    free(temporary);
    return rc;
}
