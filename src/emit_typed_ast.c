/* emit_typed_ast.c - Emit typed AST as JSON for LLM introspection
 *
 * Outputs the full type-annotated AST after typechecking, enabling:
 * - LLM reasoning about type flow through programs
 * - IDE tooling and go-to-definition support
 * - Documentation generators
 * - Refactoring and analysis tools
 */

#include "emit_typed_ast.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern const char *get_struct_type_name(ASTNode *expr, Environment *env);

/* ============================================================
 * JSON helpers
 * ============================================================ */

static void jstr(const char *s) {
    if (!s) { fputs("null", stdout); return; }
    fputc('"', stdout);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        unsigned char c = *p;
        if      (c == '\\') fputs("\\\\", stdout);
        else if (c == '"')  fputs("\\\"", stdout);
        else if (c == '\n') fputs("\\n",  stdout);
        else if (c == '\r') fputs("\\r",  stdout);
        else if (c == '\t') fputs("\\t",  stdout);
        else if (c < 0x20)  printf("\\u%04x", (unsigned)c);
        else                fputc((int)c, stdout);
    }
    fputc('"', stdout);
}

static void jkey(const char *k) {
    jstr(k); fputs(": ", stdout);
}

/* Convert Type enum to string */
static const char *type_str(Type t, const char *struct_type_name) {
    switch (t) {
        case TYPE_INT:          return "int";
        case TYPE_FLOAT:        return "float";
        case TYPE_BOOL:         return "bool";
        case TYPE_STRING:       return "string";
        case TYPE_VOID:         return "void";
        case TYPE_LIST_INT:     return "List<int>";
        case TYPE_LIST_STRING:  return "List<string>";
        case TYPE_LIST_TOKEN:   return "List<token>";
        case TYPE_ARRAY:        return "array";
        case TYPE_TUPLE:        return "tuple";
        case TYPE_FUNCTION:     return "function";
        case TYPE_STRUCT:       return struct_type_name ? struct_type_name : "struct";
        case TYPE_UNION:        return struct_type_name ? struct_type_name : "union";
        case TYPE_LIST_GENERIC: return struct_type_name ? struct_type_name : "List<?>";
        default:                return "unknown";
    }
}

/* ============================================================
 * Forward declarations
 * ============================================================ */
static void emit_stmt(ASTNode *stmt, Environment *env, int indent, int *first);

/* ============================================================
 * Indentation
 * ============================================================ */
static void ind(int n) { for (int i = 0; i < n * 2; i++) fputc(' ', stdout); }

/* ============================================================
 * Statement serializer
 * ============================================================ */
static void emit_stmt(ASTNode *stmt, Environment *env, int indent, int *first) {
    if (!stmt) return;

    if (!(*first)) fputs(",\n", stdout);
    *first = 0;

    ind(indent); fputs("{\n", stdout);
    int d = indent + 1;

    switch (stmt->type) {
        case AST_LET: {
            ind(d); jkey("kind"); jstr("let"); fputs(",\n", stdout);
            ind(d); jkey("line"); printf("%d", stmt->line); fputs(",\n", stdout);
            ind(d); jkey("name"); jstr(stmt->as.let.name); fputs(",\n", stdout);
            ind(d); jkey("mutable"); fputs(stmt->as.let.is_mut ? "true" : "false", stdout);
            fputs(",\n", stdout);
            Symbol *sym = env_get_var(env, stmt->as.let.name);
            const char *tname = sym ?
                type_str(sym->type, sym->struct_type_name) :
                type_str(stmt->as.let.var_type, stmt->as.let.type_name);
            ind(d); jkey("type"); jstr(tname); fputc('\n', stdout);
            break;
        }
        case AST_SET: {
            ind(d); jkey("kind"); jstr("set"); fputs(",\n", stdout);
            ind(d); jkey("line"); printf("%d", stmt->line); fputs(",\n", stdout);
            ind(d); jkey("target"); jstr(stmt->as.set.name); fputs(",\n", stdout);
            Symbol *sym = env_get_var(env, stmt->as.set.name);
            ind(d); jkey("type"); jstr(sym ? type_str(sym->type, sym->struct_type_name) : "unknown");
            fputc('\n', stdout);
            break;
        }
        case AST_RETURN: {
            ind(d); jkey("kind"); jstr("return"); fputs(",\n", stdout);
            ind(d); jkey("line"); printf("%d", stmt->line);
            if (stmt->as.return_stmt.value) {
                fputs(",\n", stdout);
                Type ret_type = check_expression(stmt->as.return_stmt.value, env);
                const char *sn = get_struct_type_name(stmt->as.return_stmt.value, env);
                ind(d); jkey("type"); jstr(type_str(ret_type, sn));
            }
            fputc('\n', stdout);
            break;
        }
        case AST_IF: {
            ind(d); jkey("kind"); jstr("if"); fputs(",\n", stdout);
            ind(d); jkey("line"); printf("%d", stmt->line); fputs(",\n", stdout);
            ind(d); jkey("has_else"); fputs(stmt->as.if_stmt.else_branch ? "true" : "false", stdout);
            fputc('\n', stdout);
            break;
        }
        case AST_WHILE: {
            ind(d); jkey("kind"); jstr("while"); fputs(",\n", stdout);
            ind(d); jkey("line"); printf("%d", stmt->line); fputc('\n', stdout);
            break;
        }
        case AST_FOR: {
            ind(d); jkey("kind"); jstr("for"); fputs(",\n", stdout);
            ind(d); jkey("line"); printf("%d", stmt->line); fputs(",\n", stdout);
            ind(d); jkey("var"); jstr(stmt->as.for_stmt.var_name); fputs(",\n", stdout);
            Symbol *sym = env_get_var(env, stmt->as.for_stmt.var_name);
            ind(d); jkey("var_type"); jstr(sym ? type_str(sym->type, sym->struct_type_name) : "int");
            fputc('\n', stdout);
            break;
        }
        case AST_ASSERT: {
            ind(d); jkey("kind"); jstr("assert"); fputs(",\n", stdout);
            ind(d); jkey("line"); printf("%d", stmt->line); fputc('\n', stdout);
            break;
        }
        case AST_BLOCK: {
            ind(d); jkey("kind"); jstr("block"); fputs(",\n", stdout);
            ind(d); jkey("line"); printf("%d", stmt->line); fputs(",\n", stdout);
            ind(d); jkey("statements"); fputs("[\n", stdout);
            int f2 = 1;
            for (int i = 0; i < stmt->as.block.count; i++) {
                emit_stmt(stmt->as.block.statements[i], env, d + 1, &f2);
            }
            fputc('\n', stdout);
            ind(d); fputs("]\n", stdout);
            break;
        }
        case AST_CALL: {
            ind(d); jkey("kind"); jstr("call_stmt"); fputs(",\n", stdout);
            ind(d); jkey("line"); printf("%d", stmt->line); fputs(",\n", stdout);
            ind(d); jkey("func"); jstr(stmt->as.call.name); fputc('\n', stdout);
            break;
        }
        default: {
            ind(d); jkey("kind"); jstr("stmt"); fputs(",\n", stdout);
            ind(d); jkey("line"); printf("%d", stmt->line); fputc('\n', stdout);
            break;
        }
    }

    ind(indent); fputc('}', stdout);
}

/* ============================================================
 * Main entry point
 * ============================================================ */
void emit_typed_ast_json(const char *input_file, ASTNode *program, Environment *env) {
    fputs("{\n", stdout);
    fputs("  ", stdout); jkey("format_version"); jstr("1.0"); fputs(",\n", stdout);
    fputs("  ", stdout); jkey("file"); jstr(input_file); fputs(",\n", stdout);
    fputs("  ", stdout); jkey("functions"); fputs("[\n", stdout);

    int func_first = 1;

    int item_count = 0;
    ASTNode **items = NULL;
    if (program && program->type == AST_BLOCK) {
        item_count = program->as.block.count;
        items = program->as.block.statements;
    } else if (program && program->type == AST_PROGRAM) {
        item_count = program->as.program.count;
        items = program->as.program.items;
    } else {
        fputs("  ]\n}\n", stdout);
        return;
    }

    for (int i = 0; i < item_count; i++) {
        ASTNode *item = items[i];
        if (!item) continue;

        if (item->type == AST_FUNCTION) {
            if (!func_first) fputs(",\n", stdout);
            func_first = 0;

            fputs("    {\n", stdout);
            fputs("      ", stdout); jkey("name"); jstr(item->as.function.name);
            fputs(",\n      ", stdout); jkey("line"); printf("%d", item->line);
            fputs(",\n      ", stdout); jkey("kind"); jstr("function");
            fputs(",\n", stdout);

            /* Parameters */
            fputs("      ", stdout); jkey("params"); fputs("[\n", stdout);
            for (int p = 0; p < item->as.function.param_count; p++) {
                if (p > 0) fputs(",\n", stdout);
                Parameter *param = &item->as.function.params[p];
                fputs("        {", stdout);
                jkey("name"); jstr(param->name); fputs(", ", stdout);
                jkey("type"); jstr(type_str(param->type, param->struct_type_name));
                fputc('}', stdout);
            }
            if (item->as.function.param_count > 0) fputc('\n', stdout);
            fputs("      ]", stdout);

            /* Return type */
            fputs(",\n      ", stdout); jkey("return_type");
            jstr(type_str(item->as.function.return_type,
                          item->as.function.return_struct_type_name));

            /* Body statements */
            fputs(",\n      ", stdout); jkey("body"); fputs(" [\n", stdout);
            if (item->as.function.body && item->as.function.body->type == AST_BLOCK) {
                ASTNode *body = item->as.function.body;
                int bfirst = 1;
                for (int s = 0; s < body->as.block.count; s++) {
                    emit_stmt(body->as.block.statements[s], env, 8, &bfirst);
                }
                if (!bfirst) fputc('\n', stdout);
            }
            fputs("      ]\n", stdout);
            fputs("    }", stdout);
        }
    }

    fputs("\n  ]\n}\n", stdout);
}
