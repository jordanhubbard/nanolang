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
static void transpile_expression(StringBuilder *sb, ASTNode *expr, Environment *env);
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
        case TYPE_ARRAY: return "int64_t*";  /* Arrays are int64_t* for array<int> */
        case TYPE_STRUCT: return "struct"; /* Will be extended with struct name */
        case TYPE_LIST_INT: return "List_int*";
        case TYPE_LIST_STRING: return "List_string*";
        default: return "void";
    }
}

/* Get C type for struct (returns "struct StructName") */
static void type_to_c_struct(StringBuilder *sb, const char *struct_name) {
    sb_appendf(sb, "struct %s", struct_name);
}

/* Extract struct type name from an expression (returns NULL if not a struct) */
static const char *get_struct_type_from_expr(ASTNode *expr) {
    if (!expr) return NULL;
    
    if (expr->type == AST_STRUCT_LITERAL) {
        return expr->as.struct_literal.struct_name;
    }
    
    /* For identifiers, calls, field access - would need type environment lookup */
    /* For now, return NULL and let caller handle it */
    return NULL;
}

/* Get C function name with prefix to avoid conflicts with standard library */
static const char *get_c_func_name(const char *nano_name) {
    /* Don't prefix main */
    if (strcmp(nano_name, "main") == 0) {
        return "main";
    }
    
    /* Don't prefix list runtime functions */
    if (strncmp(nano_name, "list_int_", 9) == 0 || 
        strncmp(nano_name, "list_string_", 12) == 0) {
        return nano_name;
    }
    
    /* Don't prefix advanced string operations (generated inline) */
    if (strcmp(nano_name, "char_at") == 0 ||
        strcmp(nano_name, "string_from_char") == 0 ||
        strcmp(nano_name, "is_digit") == 0 ||
        strcmp(nano_name, "is_alpha") == 0 ||
        strcmp(nano_name, "is_alnum") == 0 ||
        strcmp(nano_name, "is_whitespace") == 0 ||
        strcmp(nano_name, "is_upper") == 0 ||
        strcmp(nano_name, "is_lower") == 0 ||
        strcmp(nano_name, "int_to_string") == 0 ||
        strcmp(nano_name, "string_to_int") == 0 ||
        strcmp(nano_name, "digit_value") == 0 ||
        strcmp(nano_name, "char_to_lower") == 0 ||
        strcmp(nano_name, "char_to_upper") == 0) {
        return nano_name;
    }

    /* Prefix user functions to avoid conflicts with C stdlib (abs, min, max, etc.) */
    static char buffer[256];
    snprintf(buffer, sizeof(buffer), "nl_%s", nano_name);
    return buffer;
}

/* Transpile expression to C */
static void transpile_expression(StringBuilder *sb, ASTNode *expr, Environment *env) {
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
                transpile_expression(sb, expr->as.prefix_op.args[0], env);
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
                transpile_expression(sb, expr->as.prefix_op.args[1], env);
            } else if (arg_count == 1 && op == TOKEN_NOT) {
                sb_append(sb, "!");
                transpile_expression(sb, expr->as.prefix_op.args[0], env);
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

            /* Math and utility functions */
            } else if (strcmp(func_name, "abs") == 0) {
                func_name = "nl_abs";
            } else if (strcmp(func_name, "min") == 0) {
                func_name = "nl_min";
            } else if (strcmp(func_name, "max") == 0) {
                func_name = "nl_max";
            } else if (strcmp(func_name, "sqrt") == 0) {
                func_name = "sqrt";  /* Use C math library directly */
            } else if (strcmp(func_name, "pow") == 0) {
                func_name = "pow";  /* Use C math library directly */
            } else if (strcmp(func_name, "floor") == 0) {
                func_name = "floor";  /* Use C math library directly */
            } else if (strcmp(func_name, "ceil") == 0) {
                func_name = "ceil";  /* Use C math library directly */
            } else if (strcmp(func_name, "round") == 0) {
                func_name = "round";  /* Use C math library directly */
            } else if (strcmp(func_name, "sin") == 0) {
                func_name = "sin";  /* Use C math library directly */
            } else if (strcmp(func_name, "cos") == 0) {
                func_name = "cos";  /* Use C math library directly */
            } else if (strcmp(func_name, "tan") == 0) {
                func_name = "tan";  /* Use C math library directly */
            
            /* String operations */
            } else if (strcmp(func_name, "str_length") == 0) {
                func_name = "strlen";  /* Use C string library directly */
            } else if (strcmp(func_name, "str_concat") == 0) {
                func_name = "nl_str_concat";
            } else if (strcmp(func_name, "str_substring") == 0) {
                func_name = "nl_str_substring";
            } else if (strcmp(func_name, "str_contains") == 0) {
                func_name = "nl_str_contains";
            } else if (strcmp(func_name, "str_equals") == 0) {
                func_name = "nl_str_equals";
            
            /* Array operations */
            } else if (strcmp(func_name, "at") == 0) {
                func_name = "nl_array_at_int";  /* For now, assume int arrays */
            } else if (strcmp(func_name, "array_length") == 0) {
                func_name = "nl_array_length";
            } else if (strcmp(func_name, "array_new") == 0) {
                func_name = "nl_array_new_int";  /* For now, assume int arrays */
            } else if (strcmp(func_name, "array_set") == 0) {
                func_name = "nl_array_set_int";  /* For now, assume int arrays */
            } else if (strcmp(func_name, "print") == 0) {
                /* Special handling for print - dispatch based on argument type */
                Type arg_type = check_expression(expr->as.call.args[0], env);
                switch (arg_type) {
                    case TYPE_INT:
                        func_name = "nl_print_int";
                        break;
                    case TYPE_FLOAT:
                        func_name = "nl_print_float";
                        break;
                    case TYPE_STRING:
                        func_name = "nl_print_string";
                        break;
                    case TYPE_BOOL:
                        func_name = "nl_print_bool";
                        break;
                    default:
                        func_name = "nl_print_int";  /* Fallback */
                        break;
                }
            } else if (strcmp(func_name, "println") == 0) {
                /* Special handling for println - dispatch based on argument type */
                Type arg_type = check_expression(expr->as.call.args[0], env);
                switch (arg_type) {
                    case TYPE_INT:
                        func_name = "nl_println_int";
                        break;
                    case TYPE_FLOAT:
                        func_name = "nl_println_float";
                        break;
                    case TYPE_STRING:
                        func_name = "nl_println_string";
                        break;
                    case TYPE_BOOL:
                        func_name = "nl_println_bool";
                        break;
                    default:
                        func_name = "nl_println_int";  /* Fallback */
                        break;
                }
            } else {
                /* Check if this is an extern function */
                Function *func_def = env_get_function(env, func_name);
                if (func_def && func_def->is_extern) {
                    /* Extern functions - call directly with original name (no change) */
                } else {
                    /* User-defined functions get nl_ prefix to avoid C stdlib conflicts */
                    func_name = get_c_func_name(func_name);
                }
            }

            sb_appendf(sb, "%s(", func_name);
            for (int i = 0; i < expr->as.call.arg_count; i++) {
                if (i > 0) sb_append(sb, ", ");
                transpile_expression(sb, expr->as.call.args[i], env);
            }
            sb_append(sb, ")");
            break;
        }

        case AST_ARRAY_LITERAL: {
            /* Transpile array literal: [1, 2, 3] -> nl_array_literal_int(3, 1, 2, 3) */
            int count = expr->as.array_literal.element_count;
            sb_appendf(sb, "nl_array_literal_int(%d", count);
            for (int i = 0; i < count; i++) {
                sb_append(sb, ", ");
                transpile_expression(sb, expr->as.array_literal.elements[i], env);
            }
            sb_append(sb, ")");
            break;
        }

        case AST_IF:
            sb_append(sb, "(");
            transpile_expression(sb, expr->as.if_stmt.condition, env);
            sb_append(sb, " ? ");
            /* For if expressions, we need to handle block expressions */
            /* Simplified: just handle single expressions */
            if (expr->as.if_stmt.then_branch->type == AST_BLOCK) {
                sb_append(sb, "/* block */ 0");
            } else {
                transpile_expression(sb, expr->as.if_stmt.then_branch, env);
            }
            sb_append(sb, " : ");
            if (expr->as.if_stmt.else_branch->type == AST_BLOCK) {
                sb_append(sb, "/* block */ 0");
            } else {
                transpile_expression(sb, expr->as.if_stmt.else_branch, env);
            }
            sb_append(sb, ")");
            break;

        case AST_STRUCT_LITERAL: {
            /* Transpile struct literal: Point { x: 10, y: 20 } -> (struct Point){.x = 10, .y = 20} */
            sb_append(sb, "(struct ");
            sb_append(sb, expr->as.struct_literal.struct_name);
            sb_append(sb, "){");
            for (int i = 0; i < expr->as.struct_literal.field_count; i++) {
                if (i > 0) sb_append(sb, ", ");
                sb_append(sb, ".");
                sb_append(sb, expr->as.struct_literal.field_names[i]);
                sb_append(sb, " = ");
                transpile_expression(sb, expr->as.struct_literal.field_values[i], env);
            }
            sb_append(sb, "}");
            break;
        }

        case AST_FIELD_ACCESS: {
            /* Check if this is an enum variant access */
            if (expr->as.field_access.object->type == AST_IDENTIFIER) {
                const char *enum_name = expr->as.field_access.object->as.identifier;
                if (env_get_enum(env, enum_name)) {
                    /* Transpile as: EnumName_VariantName */
                    sb_appendf(sb, "%s_%s",
                              enum_name,
                              expr->as.field_access.field_name);
                    break;
                }
            }
            
            /* Regular struct field access: point.x -> point.x */
            transpile_expression(sb, expr->as.field_access.object, env);
            sb_append(sb, ".");
            sb_append(sb, expr->as.field_access.field_name);
            break;
        }

        default:
            sb_append(sb, "/* unknown expr */");
            break;
    }
}

/* Transpile statement to C */
static void transpile_statement(StringBuilder *sb, ASTNode *stmt, int indent, Environment *env) {
    if (!stmt) return;

    switch (stmt->type) {
        case AST_LET: {
            emit_indent(sb, indent);
            
            /* For struct types, check if it's actually an enum (enums are ints) */
            if (stmt->as.let.var_type == TYPE_STRUCT) {
                /* Check if value is an enum variant access */
                bool is_enum = false;
                if (stmt->as.let.value->type == AST_FIELD_ACCESS &&
                    stmt->as.let.value->as.field_access.object->type == AST_IDENTIFIER) {
                    const char *enum_name = stmt->as.let.value->as.field_access.object->as.identifier;
                    if (env_get_enum(env, enum_name)) {
                        is_enum = true;
                    }
                }
                
                if (is_enum) {
                    /* This is an enum, use int64_t */
                    sb_appendf(sb, "%s %s = ", type_to_c(TYPE_INT), stmt->as.let.name);
                } else {
                    /* Regular struct type */
                    const char *struct_name = get_struct_type_from_expr(stmt->as.let.value);
                    if (struct_name) {
                        sb_appendf(sb, "struct %s %s = ", struct_name, stmt->as.let.name);
                    } else {
                        /* Fallback if we can't determine struct type */
                        sb_appendf(sb, "/* struct */ void* %s = ", stmt->as.let.name);
                    }
                }
            } else {
                sb_appendf(sb, "%s %s = ", type_to_c(stmt->as.let.var_type), stmt->as.let.name);
            }
            
            transpile_expression(sb, stmt->as.let.value, env);
            sb_append(sb, ";\n");
            /* Register variable in environment for type tracking during transpilation */
            env_define_var(env, stmt->as.let.name, stmt->as.let.var_type, stmt->as.let.is_mut, create_void());
            break;
        }

        case AST_SET:
            emit_indent(sb, indent);
            sb_appendf(sb, "%s = ", stmt->as.set.name);
            transpile_expression(sb, stmt->as.set.value, env);
            sb_append(sb, ";\n");
            break;

        case AST_WHILE:
            emit_indent(sb, indent);
            sb_append(sb, "while (");
            transpile_expression(sb, stmt->as.while_stmt.condition, env);
            sb_append(sb, ") ");
            transpile_statement(sb, stmt->as.while_stmt.body, indent, env);
            break;

        case AST_FOR: {
            emit_indent(sb, indent);
            /* Extract range bounds */
            ASTNode *range = stmt->as.for_stmt.range_expr;
            sb_appendf(sb, "for (int64_t %s = ", stmt->as.for_stmt.var_name);
            if (range && range->type == AST_CALL && range->as.call.arg_count == 2) {
                transpile_expression(sb, range->as.call.args[0], env);
                sb_appendf(sb, "; %s < ", stmt->as.for_stmt.var_name);
                transpile_expression(sb, range->as.call.args[1], env);
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
                transpile_expression(sb, stmt->as.return_stmt.value, env);
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
                transpile_expression(sb, stmt->as.print.expr, env);
                sb_append(sb, ");\n");
            } else if (expr_type == TYPE_BOOL) {
                sb_append(sb, "printf(\"%s\\n\", ");
                transpile_expression(sb, stmt->as.print.expr, env);
                sb_append(sb, " ? \"true\" : \"false\");\n");
            } else if (expr_type == TYPE_FLOAT) {
                sb_append(sb, "printf(\"%g\\n\", ");
                transpile_expression(sb, stmt->as.print.expr, env);
                sb_append(sb, ");\n");
            } else {
                sb_append(sb, "printf(\"%lld\\n\", (long long)");
                transpile_expression(sb, stmt->as.print.expr, env);
                sb_append(sb, ");\n");
            }
            break;

        case AST_IF:
            emit_indent(sb, indent);
            sb_append(sb, "if (");
            transpile_expression(sb, stmt->as.if_stmt.condition, env);
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
            transpile_expression(sb, stmt, env);
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
    sb_append(sb, "#include <time.h>\n");
    sb_append(sb, "#include <stdarg.h>\n");
    sb_append(sb, "#include <math.h>\n");
    sb_append(sb, "\n/* nanolang runtime */\n");
    sb_append(sb, "#include \"runtime/list_int.h\"\n");
    sb_append(sb, "#include \"runtime/list_string.h\"\n");
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

    sb_append(sb, "/* ========== Advanced String Operations ========== */\n\n");
    
    /* char_at */
    sb_append(sb, "static int64_t char_at(const char* s, int64_t index) {\n");
    sb_append(sb, "    int len = strlen(s);\n");
    sb_append(sb, "    if (index < 0 || index >= len) {\n");
    sb_append(sb, "        fprintf(stderr, \"Error: Index %lld out of bounds (string length %d)\\n\", index, len);\n");
    sb_append(sb, "        return 0;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return (unsigned char)s[index];\n");
    sb_append(sb, "}\n\n");
    
    /* string_from_char */
    sb_append(sb, "static char* string_from_char(int64_t c) {\n");
    sb_append(sb, "    char* buffer = malloc(2);\n");
    sb_append(sb, "    buffer[0] = (char)c;\n");
    sb_append(sb, "    buffer[1] = '\\0';\n");
    sb_append(sb, "    return buffer;\n");
    sb_append(sb, "}\n\n");
    
    /* Character classification */
    sb_append(sb, "static bool is_digit(int64_t c) {\n");
    sb_append(sb, "    return c >= '0' && c <= '9';\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static bool is_alpha(int64_t c) {\n");
    sb_append(sb, "    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static bool is_alnum(int64_t c) {\n");
    sb_append(sb, "    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static bool is_whitespace(int64_t c) {\n");
    sb_append(sb, "    return c == ' ' || c == '\\t' || c == '\\n' || c == '\\r';\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static bool is_upper(int64_t c) {\n");
    sb_append(sb, "    return c >= 'A' && c <= 'Z';\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static bool is_lower(int64_t c) {\n");
    sb_append(sb, "    return c >= 'a' && c <= 'z';\n");
    sb_append(sb, "}\n\n");
    
    /* Type conversions */
    sb_append(sb, "static char* int_to_string(int64_t n) {\n");
    sb_append(sb, "    char* buffer = malloc(32);\n");
    sb_append(sb, "    snprintf(buffer, 32, \"%lld\", n);\n");
    sb_append(sb, "    return buffer;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static int64_t string_to_int(const char* s) {\n");
    sb_append(sb, "    return strtoll(s, NULL, 10);\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static int64_t digit_value(int64_t c) {\n");
    sb_append(sb, "    if (c >= '0' && c <= '9') {\n");
    sb_append(sb, "        return c - '0';\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return -1;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static int64_t char_to_lower(int64_t c) {\n");
    sb_append(sb, "    if (c >= 'A' && c <= 'Z') {\n");
    sb_append(sb, "        return c + 32;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return c;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static int64_t char_to_upper(int64_t c) {\n");
    sb_append(sb, "    if (c >= 'a' && c <= 'z') {\n");
    sb_append(sb, "        return c - 32;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return c;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "/* ========== End Advanced String Operations ========== */\n\n");

    sb_append(sb, "/* ========== Math and Utility Built-in Functions ========== */\n\n");

    /* abs function - works with int and float via macro */
    sb_append(sb, "#define nl_abs(x) _Generic((x), \\\n");
    sb_append(sb, "    int64_t: (int64_t)((x) < 0 ? -(x) : (x)), \\\n");
    sb_append(sb, "    double: (double)((x) < 0.0 ? -(x) : (x)))\n\n");

    /* min function */
    sb_append(sb, "#define nl_min(a, b) _Generic((a), \\\n");
    sb_append(sb, "    int64_t: (int64_t)((a) < (b) ? (a) : (b)), \\\n");
    sb_append(sb, "    double: (double)((a) < (b) ? (a) : (b)))\n\n");

    /* max function */
    sb_append(sb, "#define nl_max(a, b) _Generic((a), \\\n");
    sb_append(sb, "    int64_t: (int64_t)((a) > (b) ? (a) : (b)), \\\n");
    sb_append(sb, "    double: (double)((a) > (b) ? (a) : (b)))\n\n");

    /* println function - uses _Generic for type dispatch */
    sb_append(sb, "static void nl_println(void* value_ptr) {\n");
    sb_append(sb, "    /* This is a placeholder - actual implementation uses type info from checker */\n");
    sb_append(sb, "}\n\n");

    /* Specialized print functions for each type (no newline) */
    sb_append(sb, "static void nl_print_int(int64_t value) {\n");
    sb_append(sb, "    printf(\"%lld\", value);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_print_float(double value) {\n");
    sb_append(sb, "    printf(\"%g\", value);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_print_string(const char* value) {\n");
    sb_append(sb, "    printf(\"%s\", value);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_print_bool(bool value) {\n");
    sb_append(sb, "    printf(value ? \"true\" : \"false\");\n");
    sb_append(sb, "}\n\n");

    /* Specialized println functions for each type */
    sb_append(sb, "static void nl_println_int(int64_t value) {\n");
    sb_append(sb, "    printf(\"%lld\\n\", value);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_println_float(double value) {\n");
    sb_append(sb, "    printf(\"%g\\n\", value);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_println_string(const char* value) {\n");
    sb_append(sb, "    printf(\"%s\\n\", value);\n");
    sb_append(sb, "}\n\n");
    
    /* String operations */
    sb_append(sb, "/* String concatenation */\n");
    sb_append(sb, "static const char* nl_str_concat(const char* s1, const char* s2) {\n");
    sb_append(sb, "    size_t len1 = strlen(s1);\n");
    sb_append(sb, "    size_t len2 = strlen(s2);\n");
    sb_append(sb, "    char* result = malloc(len1 + len2 + 1);\n");
    sb_append(sb, "    if (!result) return \"\";\n");
    sb_append(sb, "    strcpy(result, s1);\n");
    sb_append(sb, "    strcat(result, s2);\n");
    sb_append(sb, "    return result;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "/* String substring */\n");
    sb_append(sb, "static const char* nl_str_substring(const char* str, int64_t start, int64_t length) {\n");
    sb_append(sb, "    int64_t str_len = strlen(str);\n");
    sb_append(sb, "    if (start < 0 || start >= str_len || length < 0) return \"\";\n");
    sb_append(sb, "    if (start + length > str_len) length = str_len - start;\n");
    sb_append(sb, "    char* result = malloc(length + 1);\n");
    sb_append(sb, "    if (!result) return \"\";\n");
    sb_append(sb, "    strncpy(result, str + start, length);\n");
    sb_append(sb, "    result[length] = '\\0';\n");
    sb_append(sb, "    return result;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "/* String contains */\n");
    sb_append(sb, "static bool nl_str_contains(const char* str, const char* substr) {\n");
    sb_append(sb, "    return strstr(str, substr) != NULL;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "/* String equals */\n");
    sb_append(sb, "static bool nl_str_equals(const char* s1, const char* s2) {\n");
    sb_append(sb, "    return strcmp(s1, s2) == 0;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_println_bool(bool value) {\n");
    sb_append(sb, "    printf(\"%s\\n\", value ? \"true\" : \"false\");\n");
    sb_append(sb, "}\n\n");

    /* Array operations */
    sb_append(sb, "/* ========== Array Operations (With Bounds Checking!) ========== */\n\n");
    
    sb_append(sb, "/* Array struct */\n");
    sb_append(sb, "typedef struct {\n");
    sb_append(sb, "    int64_t length;\n");
    sb_append(sb, "    void* data;\n");
    sb_append(sb, "    size_t element_size;\n");
    sb_append(sb, "} nl_array;\n\n");
    
    sb_append(sb, "/* Array access - BOUNDS CHECKED! */\n");
    sb_append(sb, "static int64_t nl_array_at_int(nl_array* arr, int64_t index) {\n");
    sb_append(sb, "    if (index < 0 || index >= arr->length) {\n");
    sb_append(sb, "        fprintf(stderr, \"Runtime Error: Array index %lld out of bounds [0..%lld)\\n\", index, arr->length);\n");
    sb_append(sb, "        exit(1);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return ((int64_t*)arr->data)[index];\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static double nl_array_at_float(nl_array* arr, int64_t index) {\n");
    sb_append(sb, "    if (index < 0 || index >= arr->length) {\n");
    sb_append(sb, "        fprintf(stderr, \"Runtime Error: Array index %lld out of bounds [0..%lld)\\n\", index, arr->length);\n");
    sb_append(sb, "        exit(1);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return ((double*)arr->data)[index];\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static const char* nl_array_at_string(nl_array* arr, int64_t index) {\n");
    sb_append(sb, "    if (index < 0 || index >= arr->length) {\n");
    sb_append(sb, "        fprintf(stderr, \"Runtime Error: Array index %lld out of bounds [0..%lld)\\n\", index, arr->length);\n");
    sb_append(sb, "        exit(1);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return ((const char**)arr->data)[index];\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "/* Array length */\n");
    sb_append(sb, "static int64_t nl_array_length(nl_array* arr) {\n");
    sb_append(sb, "    return arr->length;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "/* Array creation */\n");
    sb_append(sb, "static nl_array* nl_array_new_int(int64_t size, int64_t default_val) {\n");
    sb_append(sb, "    nl_array* arr = malloc(sizeof(nl_array));\n");
    sb_append(sb, "    arr->length = size;\n");
    sb_append(sb, "    arr->element_size = sizeof(int64_t);\n");
    sb_append(sb, "    arr->data = malloc(size * sizeof(int64_t));\n");
    sb_append(sb, "    for (int64_t i = 0; i < size; i++) {\n");
    sb_append(sb, "        ((int64_t*)arr->data)[i] = default_val;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return arr;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "/* Array set - BOUNDS CHECKED! */\n");
    sb_append(sb, "static void nl_array_set_int(nl_array* arr, int64_t index, int64_t value) {\n");
    sb_append(sb, "    if (index < 0 || index >= arr->length) {\n");
    sb_append(sb, "        fprintf(stderr, \"Runtime Error: Array index %lld out of bounds [0..%lld)\\n\", index, arr->length);\n");
    sb_append(sb, "        exit(1);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    ((int64_t*)arr->data)[index] = value;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "/* Array literal creation helper */\n");
    sb_append(sb, "static nl_array* nl_array_literal_int(int64_t count, ...) {\n");
    sb_append(sb, "    nl_array* arr = malloc(sizeof(nl_array));\n");
    sb_append(sb, "    arr->length = count;\n");
    sb_append(sb, "    arr->element_size = sizeof(int64_t);\n");
    sb_append(sb, "    arr->data = malloc(count * sizeof(int64_t));\n");
    sb_append(sb, "    va_list args;\n");
    sb_append(sb, "    va_start(args, count);\n");
    sb_append(sb, "    for (int64_t i = 0; i < count; i++) {\n");
    sb_append(sb, "        ((int64_t*)arr->data)[i] = va_arg(args, int64_t);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    va_end(args);\n");
    sb_append(sb, "    return arr;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "/* ========== End Array Operations ========== */\n\n");

    sb_append(sb, "/* ========== End Math and Utility Built-in Functions ========== */\n\n");

    /* Generate struct typedefs */
    sb_append(sb, "/* ========== Struct Definitions ========== */\n\n");
    for (int i = 0; i < env->struct_count; i++) {
        StructDef *sdef = &env->structs[i];
        sb_appendf(sb, "struct %s {\n", sdef->name);
        for (int j = 0; j < sdef->field_count; j++) {
            sb_append(sb, "    ");
            if (sdef->field_types[j] == TYPE_STRUCT) {
                /* For struct fields, we'd need the struct type name - for now use void* */
                sb_append(sb, "void* /* struct field */");
            } else {
                sb_append(sb, type_to_c(sdef->field_types[j]));
            }
            sb_appendf(sb, " %s;\n", sdef->field_names[j]);
        }
        sb_append(sb, "};\n\n");
    }
    sb_append(sb, "/* ========== End Struct Definitions ========== */\n\n");

    /* Generate enum typedefs */
    sb_append(sb, "/* ========== Enum Definitions ========== */\n\n");
    for (int i = 0; i < env->enum_count; i++) {
        EnumDef *edef = &env->enums[i];
        sb_appendf(sb, "typedef enum {\n");
        for (int j = 0; j < edef->variant_count; j++) {
            sb_appendf(sb, "    %s_%s = %d",
                      edef->name,
                      edef->variant_names[j],
                      edef->variant_values[j]);
            if (j < edef->variant_count - 1) sb_append(sb, ",\n");
            else sb_append(sb, "\n");
        }
        sb_appendf(sb, "} %s;\n\n", edef->name);
    }
    sb_append(sb, "/* ========== End Enum Definitions ========== */\n\n");

    /* Forward declare all functions */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_FUNCTION) {
            /* Skip extern functions - they're declared in C headers */
            if (item->as.function.is_extern) {
                continue;
            }
            
            /* Regular functions - forward declare with nl_ prefix */
            /* Function return type */
            if (strcmp(item->as.function.name, "main") == 0) {
                sb_append(sb, "int");
            } else if (item->as.function.return_type == TYPE_STRUCT && item->as.function.return_struct_type_name) {
                sb_appendf(sb, "struct %s", item->as.function.return_struct_type_name);
            } else {
                sb_append(sb, type_to_c(item->as.function.return_type));
            }
            
            const char *c_func_name = get_c_func_name(item->as.function.name);
            sb_appendf(sb, " %s(", c_func_name);
            
            /* Function parameters */
            for (int j = 0; j < item->as.function.param_count; j++) {
                if (j > 0) sb_append(sb, ", ");
                
                if (item->as.function.params[j].type == TYPE_STRUCT && item->as.function.params[j].struct_type_name) {
                    sb_appendf(sb, "struct %s %s",
                              item->as.function.params[j].struct_type_name,
                              item->as.function.params[j].name);
                } else {
                    sb_appendf(sb, "%s %s",
                              type_to_c(item->as.function.params[j].type),
                              item->as.function.params[j].name);
                }
            }
            sb_append(sb, ");\n");
        }
    }
    sb_append(sb, "\n");

    /* Transpile all functions (skip shadow tests and extern functions) */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_FUNCTION) {
            /* Skip extern functions - they're declared only, no implementation */
            if (item->as.function.is_extern) {
                continue;
            }
            
            /* Function return type */
            if (strcmp(item->as.function.name, "main") == 0) {
                sb_append(sb, "int");
            } else if (item->as.function.return_type == TYPE_STRUCT && item->as.function.return_struct_type_name) {
                sb_appendf(sb, "struct %s", item->as.function.return_struct_type_name);
            } else {
                sb_append(sb, type_to_c(item->as.function.return_type));
            }
            
            const char *c_func_name = get_c_func_name(item->as.function.name);
            sb_appendf(sb, " %s(", c_func_name);
            
            /* Function parameters */
            for (int j = 0; j < item->as.function.param_count; j++) {
                if (j > 0) sb_append(sb, ", ");
                
                if (item->as.function.params[j].type == TYPE_STRUCT && item->as.function.params[j].struct_type_name) {
                    sb_appendf(sb, "struct %s %s",
                              item->as.function.params[j].struct_type_name,
                              item->as.function.params[j].name);
                } else {
                    sb_appendf(sb, "%s %s",
                              type_to_c(item->as.function.params[j].type),
                              item->as.function.params[j].name);
                }
            }
            sb_append(sb, ") ");

            /* Add parameters to environment for type checking during transpilation */
            int saved_symbol_count = env->symbol_count;
            for (int j = 0; j < item->as.function.param_count; j++) {
                Value dummy_val = create_void();
                env_define_var(env, item->as.function.params[j].name,
                             item->as.function.params[j].type, false, dummy_val);
            }

            /* Function body */
            transpile_statement(sb, item->as.function.body, 0, env);
            sb_append(sb, "\n");

            /* Restore environment (remove parameters) */
            env->symbol_count = saved_symbol_count;
        }
    }

    char *result = sb->buffer;
    free(sb);
    return result;
}