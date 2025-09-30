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
static void transpile_statement(StringBuilder *sb, ASTNode *stmt, int indent, Environment *env);

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

/* Get C function name with prefix to avoid conflicts with standard library */
static const char *get_c_func_name(const char *nano_name) {
    /* Don't prefix main */
    if (strcmp(nano_name, "main") == 0) {
        return "main";
    }

    /* Prefix user functions to avoid conflicts with C stdlib (abs, min, max, etc.) */
    static char buffer[256];
    snprintf(buffer, sizeof(buffer), "nl_%s", nano_name);
    return buffer;
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

        case AST_CALL: {
            /* Map nanolang OS function names to C implementation names */
            const char *func_name = expr->as.call.name;

            /* File operations */
            if (strcmp(func_name, "file_read") == 0) {
                func_name = "nl_os_file_read";
            } else if (strcmp(func_name, "file_write") == 0) {
                func_name = "nl_os_file_write";
            } else if (strcmp(func_name, "file_append") == 0) {
                func_name = "nl_os_file_append";
            } else if (strcmp(func_name, "file_remove") == 0) {
                func_name = "nl_os_file_remove";
            } else if (strcmp(func_name, "file_rename") == 0) {
                func_name = "nl_os_file_rename";
            } else if (strcmp(func_name, "file_exists") == 0) {
                func_name = "nl_os_file_exists";
            } else if (strcmp(func_name, "file_size") == 0) {
                func_name = "nl_os_file_size";

            /* Directory operations */
            } else if (strcmp(func_name, "dir_create") == 0) {
                func_name = "nl_os_dir_create";
            } else if (strcmp(func_name, "dir_remove") == 0) {
                func_name = "nl_os_dir_remove";
            } else if (strcmp(func_name, "dir_list") == 0) {
                func_name = "nl_os_dir_list";
            } else if (strcmp(func_name, "dir_exists") == 0) {
                func_name = "nl_os_dir_exists";

            /* Working directory operations */
            } else if (strcmp(func_name, "getcwd") == 0) {
                func_name = "nl_os_getcwd";
            } else if (strcmp(func_name, "chdir") == 0) {
                func_name = "nl_os_chdir";

            /* Path operations */
            } else if (strcmp(func_name, "path_isfile") == 0) {
                func_name = "nl_os_path_isfile";
            } else if (strcmp(func_name, "path_isdir") == 0) {
                func_name = "nl_os_path_isdir";
            } else if (strcmp(func_name, "path_join") == 0) {
                func_name = "nl_os_path_join";
            } else if (strcmp(func_name, "path_basename") == 0) {
                func_name = "nl_os_path_basename";
            } else if (strcmp(func_name, "path_dirname") == 0) {
                func_name = "nl_os_path_dirname";

            /* Process operations */
            } else if (strcmp(func_name, "system") == 0) {
                func_name = "nl_os_system";
            } else if (strcmp(func_name, "exit") == 0) {
                func_name = "nl_os_exit";
            } else if (strcmp(func_name, "getenv") == 0) {
                func_name = "nl_os_getenv";
            } else {
                /* User-defined functions get nl_ prefix to avoid C stdlib conflicts */
                func_name = get_c_func_name(func_name);
            }

            sb_appendf(sb, "%s(", func_name);
            for (int i = 0; i < expr->as.call.arg_count; i++) {
                if (i > 0) sb_append(sb, ", ");
                transpile_expression(sb, expr->as.call.args[i]);
            }
            sb_append(sb, ")");
            break;
        }

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
static void transpile_statement(StringBuilder *sb, ASTNode *stmt, int indent, Environment *env) {
    if (!stmt) return;

    switch (stmt->type) {
        case AST_LET:
            emit_indent(sb, indent);
            sb_appendf(sb, "%s %s = ", type_to_c(stmt->as.let.var_type), stmt->as.let.name);
            transpile_expression(sb, stmt->as.let.value);
            sb_append(sb, ";\n");
            /* Register variable in environment for type tracking during transpilation */
            env_define_var(env, stmt->as.let.name, stmt->as.let.var_type, stmt->as.let.is_mut, create_void());
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
            transpile_statement(sb, stmt->as.while_stmt.body, indent, env);
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
            transpile_statement(sb, stmt->as.for_stmt.body, indent, env);
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
                transpile_statement(sb, stmt->as.block.statements[i], indent + 1, env);
            }
            emit_indent(sb, indent);
            sb_append(sb, "}\n");
            break;

        case AST_PRINT:
            emit_indent(sb, indent);
            /* Use semantic type from type checker instead of AST node type */
            Type expr_type = check_expression(stmt->as.print.expr, env);
            if (expr_type == TYPE_STRING) {
                sb_append(sb, "printf(\"%s\\n\", ");
                transpile_expression(sb, stmt->as.print.expr);
                sb_append(sb, ");\n");
            } else if (expr_type == TYPE_BOOL) {
                sb_append(sb, "printf(\"%s\\n\", ");
                transpile_expression(sb, stmt->as.print.expr);
                sb_append(sb, " ? \"true\" : \"false\");\n");
            } else if (expr_type == TYPE_FLOAT) {
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
            transpile_statement(sb, stmt->as.if_stmt.then_branch, indent, env);
            if (stmt->as.if_stmt.else_branch) {
                emit_indent(sb, indent);
                sb_append(sb, "else ");
                transpile_statement(sb, stmt->as.if_stmt.else_branch, indent, env);
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
char *transpile_to_c(ASTNode *program, Environment *env) {
    if (!program || program->type != AST_PROGRAM) {
        return NULL;
    }

    StringBuilder *sb = sb_create();

    /* C includes and headers */
    sb_append(sb, "#include <stdio.h>\n");
    sb_append(sb, "#include <stdint.h>\n");
    sb_append(sb, "#include <stdbool.h>\n");
    sb_append(sb, "#include <string.h>\n");
    sb_append(sb, "#include <stdlib.h>\n");
    sb_append(sb, "#include <sys/stat.h>\n");
    sb_append(sb, "#include <sys/types.h>\n");
    sb_append(sb, "#include <dirent.h>\n");
    sb_append(sb, "#include <unistd.h>\n");
    sb_append(sb, "#include <libgen.h>\n");
    sb_append(sb, "\n");

    /* OS stdlib runtime library */
    sb_append(sb, "/* ========== OS Standard Library ========== */\n\n");

    /* File operations */
    sb_append(sb, "static char* nl_os_file_read(const char* path) {\n");
    sb_append(sb, "    FILE* f = fopen(path, \"r\");\n");
    sb_append(sb, "    if (!f) return \"\";\n");
    sb_append(sb, "    fseek(f, 0, SEEK_END);\n");
    sb_append(sb, "    long size = ftell(f);\n");
    sb_append(sb, "    fseek(f, 0, SEEK_SET);\n");
    sb_append(sb, "    char* buffer = malloc(size + 1);\n");
    sb_append(sb, "    fread(buffer, 1, size, f);\n");
    sb_append(sb, "    buffer[size] = '\\0';\n");
    sb_append(sb, "    fclose(f);\n");
    sb_append(sb, "    return buffer;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static int64_t nl_os_file_write(const char* path, const char* content) {\n");
    sb_append(sb, "    FILE* f = fopen(path, \"w\");\n");
    sb_append(sb, "    if (!f) return -1;\n");
    sb_append(sb, "    fputs(content, f);\n");
    sb_append(sb, "    fclose(f);\n");
    sb_append(sb, "    return 0;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static int64_t nl_os_file_append(const char* path, const char* content) {\n");
    sb_append(sb, "    FILE* f = fopen(path, \"a\");\n");
    sb_append(sb, "    if (!f) return -1;\n");
    sb_append(sb, "    fputs(content, f);\n");
    sb_append(sb, "    fclose(f);\n");
    sb_append(sb, "    return 0;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static int64_t nl_os_file_remove(const char* path) {\n");
    sb_append(sb, "    return remove(path) == 0 ? 0 : -1;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static int64_t nl_os_file_rename(const char* old_path, const char* new_path) {\n");
    sb_append(sb, "    return rename(old_path, new_path) == 0 ? 0 : -1;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static bool nl_os_file_exists(const char* path) {\n");
    sb_append(sb, "    return access(path, F_OK) == 0;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static int64_t nl_os_file_size(const char* path) {\n");
    sb_append(sb, "    struct stat st;\n");
    sb_append(sb, "    if (stat(path, &st) != 0) return -1;\n");
    sb_append(sb, "    return st.st_size;\n");
    sb_append(sb, "}\n\n");

    /* Directory operations */
    sb_append(sb, "static int64_t nl_os_dir_create(const char* path) {\n");
    sb_append(sb, "    return mkdir(path, 0755) == 0 ? 0 : -1;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static int64_t nl_os_dir_remove(const char* path) {\n");
    sb_append(sb, "    return rmdir(path) == 0 ? 0 : -1;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* nl_os_dir_list(const char* path) {\n");
    sb_append(sb, "    DIR* dir = opendir(path);\n");
    sb_append(sb, "    if (!dir) return \"\";\n");
    sb_append(sb, "    char* buffer = malloc(4096);\n");
    sb_append(sb, "    buffer[0] = '\\0';\n");
    sb_append(sb, "    struct dirent* entry;\n");
    sb_append(sb, "    while ((entry = readdir(dir)) != NULL) {\n");
    sb_append(sb, "        if (strcmp(entry->d_name, \".\") == 0 || strcmp(entry->d_name, \"..\") == 0) continue;\n");
    sb_append(sb, "        strcat(buffer, entry->d_name);\n");
    sb_append(sb, "        strcat(buffer, \"\\n\");\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    closedir(dir);\n");
    sb_append(sb, "    return buffer;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static bool nl_os_dir_exists(const char* path) {\n");
    sb_append(sb, "    struct stat st;\n");
    sb_append(sb, "    if (stat(path, &st) != 0) return false;\n");
    sb_append(sb, "    return S_ISDIR(st.st_mode);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* nl_os_getcwd(void) {\n");
    sb_append(sb, "    char* buffer = malloc(1024);\n");
    sb_append(sb, "    if (getcwd(buffer, 1024) == NULL) {\n");
    sb_append(sb, "        buffer[0] = '\\0';\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return buffer;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static int64_t nl_os_chdir(const char* path) {\n");
    sb_append(sb, "    return chdir(path) == 0 ? 0 : -1;\n");
    sb_append(sb, "}\n\n");

    /* Path operations */
    sb_append(sb, "static bool nl_os_path_isfile(const char* path) {\n");
    sb_append(sb, "    struct stat st;\n");
    sb_append(sb, "    if (stat(path, &st) != 0) return false;\n");
    sb_append(sb, "    return S_ISREG(st.st_mode);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static bool nl_os_path_isdir(const char* path) {\n");
    sb_append(sb, "    struct stat st;\n");
    sb_append(sb, "    if (stat(path, &st) != 0) return false;\n");
    sb_append(sb, "    return S_ISDIR(st.st_mode);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* nl_os_path_join(const char* a, const char* b) {\n");
    sb_append(sb, "    char* buffer = malloc(2048);\n");
    sb_append(sb, "    if (strlen(a) == 0) {\n");
    sb_append(sb, "        snprintf(buffer, 2048, \"%s\", b);\n");
    sb_append(sb, "    } else if (a[strlen(a) - 1] == '/') {\n");
    sb_append(sb, "        snprintf(buffer, 2048, \"%s%s\", a, b);\n");
    sb_append(sb, "    } else {\n");
    sb_append(sb, "        snprintf(buffer, 2048, \"%s/%s\", a, b);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return buffer;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* nl_os_path_basename(const char* path) {\n");
    sb_append(sb, "    char* path_copy = strdup(path);\n");
    sb_append(sb, "    char* base = basename(path_copy);\n");
    sb_append(sb, "    char* result = strdup(base);\n");
    sb_append(sb, "    free(path_copy);\n");
    sb_append(sb, "    return result;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* nl_os_path_dirname(const char* path) {\n");
    sb_append(sb, "    char* path_copy = strdup(path);\n");
    sb_append(sb, "    char* dir = dirname(path_copy);\n");
    sb_append(sb, "    char* result = strdup(dir);\n");
    sb_append(sb, "    free(path_copy);\n");
    sb_append(sb, "    return result;\n");
    sb_append(sb, "}\n\n");

    /* Process operations */
    sb_append(sb, "static int64_t nl_os_system(const char* command) {\n");
    sb_append(sb, "    return system(command);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_os_exit(int64_t code) {\n");
    sb_append(sb, "    exit((int)code);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* nl_os_getenv(const char* name) {\n");
    sb_append(sb, "    const char* value = getenv(name);\n");
    sb_append(sb, "    return value ? (char*)value : \"\";\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "/* ========== End OS Standard Library ========== */\n\n");

    /* Forward declare all functions */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_FUNCTION) {
            /* Special case for main - must return int in C */
            const char *ret_type = strcmp(item->as.function.name, "main") == 0 ? "int" : type_to_c(item->as.function.return_type);
            const char *c_func_name = get_c_func_name(item->as.function.name);
            sb_appendf(sb, "%s %s(", ret_type, c_func_name);
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
            const char *c_func_name = get_c_func_name(item->as.function.name);
            sb_appendf(sb, "%s %s(", ret_type, c_func_name);
            for (int j = 0; j < item->as.function.param_count; j++) {
                if (j > 0) sb_append(sb, ", ");
                sb_appendf(sb, "%s %s",
                          type_to_c(item->as.function.params[j].type),
                          item->as.function.params[j].name);
            }
            sb_append(sb, ") ");

            /* Function body */
            transpile_statement(sb, item->as.function.body, 0, env);
            sb_append(sb, "\n");
        }
    }

    char *result = sb->buffer;
    free(sb);
    return result;
}