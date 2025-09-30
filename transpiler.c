#include "nanolang.h"
#include <stdarg.h>

/* String builder for C code generation */
typedef struct {
    char *buffer;
    int length;
    int capacity;
} StringBuilder;

static StringBuilder *sb_create(void) {
    StringBuilder *sb = malloc(sizeof(StringBuilder));
    sb->capacity = 1024;
    sb->length = 0;
    sb->buffer = malloc(sb->capacity);
    sb->buffer[0] = '\0';
    return sb;
}

static void sb_append(StringBuilder *sb, const char *str) {
    int len = strlen(str);
    while (sb->length + len >= sb->capacity) {
        sb->capacity *= 2;
        sb->buffer = realloc(sb->buffer, sb->capacity);
    }
    strcpy(sb->buffer + sb->length, str);
    sb->length += len;
}

static void sb_appendf(StringBuilder *sb, const char *fmt, ...) {
    char buffer[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    sb_append(sb, buffer);
}

/* Forward declarations */
static void transpile_expression(StringBuilder *sb, ASTNode *expr);
static void transpile_statement(StringBuilder *sb, ASTNode *stmt, int indent);

/* Generate indentation */
static void emit_indent(StringBuilder *sb, int indent) {
    for (int i = 0; i < indent; i++) {
        sb_append(sb, "    ");
    }
}

/* Transpile type to C type */
static const char *type_to_c(Type type) {
    switch (type) {
        case TYPE_INT: return "int64_t";
        case TYPE_FLOAT: return "double";
        case TYPE_BOOL: return "bool";
        case TYPE_STRING: return "const char*";
        case TYPE_VOID: return "void";
        default: return "void";
    }
}

/* Transpile expression to C */
static void transpile_expression(StringBuilder *sb, ASTNode *expr) {
    if (!expr) return;

    switch (expr->type) {
        case AST_NUMBER:
            sb_appendf(sb, "%lldLL", expr->as.number);
            break;

        case AST_FLOAT:
            sb_appendf(sb, "%g", expr->as.float_val);
            break;

        case AST_STRING:
            sb_appendf(sb, "\"%s\"", expr->as.string_val);
            break;

        case AST_BOOL:
            sb_append(sb, expr->as.bool_val ? "true" : "false");
            break;

        case AST_IDENTIFIER:
            sb_append(sb, expr->as.identifier);
            break;

        case AST_PREFIX_OP: {
            TokenType op = expr->as.prefix_op.op;
            int arg_count = expr->as.prefix_op.arg_count;

            sb_append(sb, "(");

            if (arg_count == 2) {
                transpile_expression(sb, expr->as.prefix_op.args[0]);
                switch (op) {
                    case TOKEN_PLUS: sb_append(sb, " + "); break;
                    case TOKEN_MINUS: sb_append(sb, " - "); break;
                    case TOKEN_STAR: sb_append(sb, " * "); break;
                    case TOKEN_SLASH: sb_append(sb, " / "); break;
                    case TOKEN_PERCENT: sb_append(sb, " % "); break;
                    case TOKEN_EQ: sb_append(sb, " == "); break;
                    case TOKEN_NE: sb_append(sb, " != "); break;
                    case TOKEN_LT: sb_append(sb, " < "); break;
                    case TOKEN_LE: sb_append(sb, " <= "); break;
                    case TOKEN_GT: sb_append(sb, " > "); break;
                    case TOKEN_GE: sb_append(sb, " >= "); break;
                    case TOKEN_AND: sb_append(sb, " && "); break;
                    case TOKEN_OR: sb_append(sb, " || "); break;
                    default: sb_append(sb, " OP "); break;
                }
                transpile_expression(sb, expr->as.prefix_op.args[1]);
            } else if (arg_count == 1 && op == TOKEN_NOT) {
                sb_append(sb, "!");
                transpile_expression(sb, expr->as.prefix_op.args[0]);
            }

            sb_append(sb, ")");
            break;
        }

        case AST_CALL:
            sb_appendf(sb, "%s(", expr->as.call.name);
            for (int i = 0; i < expr->as.call.arg_count; i++) {
                if (i > 0) sb_append(sb, ", ");
                transpile_expression(sb, expr->as.call.args[i]);
            }
            sb_append(sb, ")");
            break;

        case AST_IF:
            sb_append(sb, "(");
            transpile_expression(sb, expr->as.if_stmt.condition);
            sb_append(sb, " ? ");
            /* For if expressions, we need to handle block expressions */
            /* Simplified: just handle single expressions */
            if (expr->as.if_stmt.then_branch->type == AST_BLOCK) {
                sb_append(sb, "/* block */ 0");
            } else {
                transpile_expression(sb, expr->as.if_stmt.then_branch);
            }
            sb_append(sb, " : ");
            if (expr->as.if_stmt.else_branch->type == AST_BLOCK) {
                sb_append(sb, "/* block */ 0");
            } else {
                transpile_expression(sb, expr->as.if_stmt.else_branch);
            }
            sb_append(sb, ")");
            break;

        default:
            sb_append(sb, "/* unknown expr */");
            break;
    }
}

/* Transpile statement to C */
static void transpile_statement(StringBuilder *sb, ASTNode *stmt, int indent) {
    if (!stmt) return;

    switch (stmt->type) {
        case AST_LET:
            emit_indent(sb, indent);
            sb_appendf(sb, "%s %s = ", type_to_c(stmt->as.let.var_type), stmt->as.let.name);
            transpile_expression(sb, stmt->as.let.value);
            sb_append(sb, ";\n");
            break;

        case AST_SET:
            emit_indent(sb, indent);
            sb_appendf(sb, "%s = ", stmt->as.set.name);
            transpile_expression(sb, stmt->as.set.value);
            sb_append(sb, ";\n");
            break;

        case AST_WHILE:
            emit_indent(sb, indent);
            sb_append(sb, "while (");
            transpile_expression(sb, stmt->as.while_stmt.condition);
            sb_append(sb, ") ");
            transpile_statement(sb, stmt->as.while_stmt.body, indent);
            break;

        case AST_FOR: {
            emit_indent(sb, indent);
            /* Extract range bounds */
            ASTNode *range = stmt->as.for_stmt.range_expr;
            sb_appendf(sb, "for (int64_t %s = ", stmt->as.for_stmt.var_name);
            if (range && range->type == AST_CALL && range->as.call.arg_count == 2) {
                transpile_expression(sb, range->as.call.args[0]);
                sb_appendf(sb, "; %s < ", stmt->as.for_stmt.var_name);
                transpile_expression(sb, range->as.call.args[1]);
                sb_appendf(sb, "; %s++) ", stmt->as.for_stmt.var_name);
            } else {
                sb_append(sb, "0; 0; ) ");
            }
            transpile_statement(sb, stmt->as.for_stmt.body, indent);
            break;
        }

        case AST_RETURN:
            emit_indent(sb, indent);
            sb_append(sb, "return");
            if (stmt->as.return_stmt.value) {
                sb_append(sb, " ");
                transpile_expression(sb, stmt->as.return_stmt.value);
            }
            sb_append(sb, ";\n");
            break;

        case AST_BLOCK:
            sb_append(sb, "{\n");
            for (int i = 0; i < stmt->as.block.count; i++) {
                transpile_statement(sb, stmt->as.block.statements[i], indent + 1);
            }
            emit_indent(sb, indent);
            sb_append(sb, "}\n");
            break;

        case AST_PRINT:
            emit_indent(sb, indent);
            /* For now, detect type based on expression */
            if (stmt->as.print.expr->type == AST_STRING) {
                sb_append(sb, "printf(\"%s\\n\", ");
                transpile_expression(sb, stmt->as.print.expr);
                sb_append(sb, ");\n");
            } else if (stmt->as.print.expr->type == AST_BOOL) {
                sb_append(sb, "printf(\"%s\\n\", ");
                transpile_expression(sb, stmt->as.print.expr);
                sb_append(sb, " ? \"true\" : \"false\");\n");
            } else if (stmt->as.print.expr->type == AST_FLOAT) {
                sb_append(sb, "printf(\"%g\\n\", ");
                transpile_expression(sb, stmt->as.print.expr);
                sb_append(sb, ");\n");
            } else {
                sb_append(sb, "printf(\"%lld\\n\", (long long)");
                transpile_expression(sb, stmt->as.print.expr);
                sb_append(sb, ");\n");
            }
            break;

        case AST_IF:
            emit_indent(sb, indent);
            sb_append(sb, "if (");
            transpile_expression(sb, stmt->as.if_stmt.condition);
            sb_append(sb, ") ");
            transpile_statement(sb, stmt->as.if_stmt.then_branch, indent);
            if (stmt->as.if_stmt.else_branch) {
                emit_indent(sb, indent);
                sb_append(sb, "else ");
                transpile_statement(sb, stmt->as.if_stmt.else_branch, indent);
            }
            break;

        default:
            /* Expression statements */
            emit_indent(sb, indent);
            transpile_expression(sb, stmt);
            sb_append(sb, ";\n");
            break;
    }
}

/* Transpile program to C */
char *transpile_to_c(ASTNode *program) {
    if (!program || program->type != AST_PROGRAM) {
        return NULL;
    }

    StringBuilder *sb = sb_create();

    /* C includes and headers */
    sb_append(sb, "#include <stdio.h>\n");
    sb_append(sb, "#include <stdint.h>\n");
    sb_append(sb, "#include <stdbool.h>\n");
    sb_append(sb, "#include <string.h>\n");
    sb_append(sb, "\n");

    /* Forward declare all functions */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_FUNCTION) {
            /* Special case for main - must return int in C */
            const char *ret_type = strcmp(item->as.function.name, "main") == 0 ? "int" : type_to_c(item->as.function.return_type);
            sb_appendf(sb, "%s %s(", ret_type, item->as.function.name);
            for (int j = 0; j < item->as.function.param_count; j++) {
                if (j > 0) sb_append(sb, ", ");
                sb_appendf(sb, "%s %s",
                          type_to_c(item->as.function.params[j].type),
                          item->as.function.params[j].name);
            }
            sb_append(sb, ");\n");
        }
    }
    sb_append(sb, "\n");

    /* Transpile all functions (skip shadow tests) */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_FUNCTION) {
            /* Function signature */
            const char *ret_type = strcmp(item->as.function.name, "main") == 0 ? "int" : type_to_c(item->as.function.return_type);
            sb_appendf(sb, "%s %s(", ret_type, item->as.function.name);
            for (int j = 0; j < item->as.function.param_count; j++) {
                if (j > 0) sb_append(sb, ", ");
                sb_appendf(sb, "%s %s",
                          type_to_c(item->as.function.params[j].type),
                          item->as.function.params[j].name);
            }
            sb_append(sb, ") ");

            /* Function body */
            transpile_statement(sb, item->as.function.body, 0);
            sb_append(sb, "\n");
        }
    }

    char *result = sb->buffer;
    free(sb);
    return result;
}