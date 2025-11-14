#include "nanolang.h"
#include "runtime/list_int.h"
#include "runtime/list_string.h"
#include "runtime/list_token.h"
#include "tracing.h"
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <libgen.h>
#include <math.h>

/* Forward declarations */
static Value eval_expression(ASTNode *expr, Environment *env);
static Value eval_statement(ASTNode *stmt, Environment *env);

/* ==========================================================================
 * Built-in OS Functions Implementation
 * ========================================================================== */

/* File Operations */
static Value builtin_file_read(Value *args) {
    const char *path = args[0].as.string_val;
    FILE *f = fopen(path, "r");
    if (!f) return create_string("");

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *buffer = malloc(size + 1);
    fread(buffer, 1, size, f);
    buffer[size] = '\0';
    fclose(f);

    Value result = create_string(buffer);
    free(buffer);
    return result;
}

static Value builtin_file_write(Value *args) {
    const char *path = args[0].as.string_val;
    const char *content = args[1].as.string_val;
    FILE *f = fopen(path, "w");
    if (!f) return create_int(-1);

    fputs(content, f);
    fclose(f);
    return create_int(0);
}

static Value builtin_file_append(Value *args) {
    const char *path = args[0].as.string_val;
    const char *content = args[1].as.string_val;
    FILE *f = fopen(path, "a");
    if (!f) return create_int(-1);

    fputs(content, f);
    fclose(f);
    return create_int(0);
}

static Value builtin_file_remove(Value *args) {
    const char *path = args[0].as.string_val;
    return create_int(remove(path) == 0 ? 0 : -1);
}

static Value builtin_file_rename(Value *args) {
    const char *old_path = args[0].as.string_val;
    const char *new_path = args[1].as.string_val;
    return create_int(rename(old_path, new_path) == 0 ? 0 : -1);
}

static Value builtin_file_exists(Value *args) {
    const char *path = args[0].as.string_val;
    return create_bool(access(path, F_OK) == 0);
}

static Value builtin_file_size(Value *args) {
    const char *path = args[0].as.string_val;
    struct stat st;
    if (stat(path, &st) != 0) return create_int(-1);
    return create_int(st.st_size);
}

/* Directory Operations */
static Value builtin_dir_create(Value *args) {
    const char *path = args[0].as.string_val;
    return create_int(mkdir(path, 0755) == 0 ? 0 : -1);
}

static Value builtin_dir_remove(Value *args) {
    const char *path = args[0].as.string_val;
    return create_int(rmdir(path) == 0 ? 0 : -1);
}

static Value builtin_dir_list(Value *args) {
    const char *path = args[0].as.string_val;
    DIR *dir = opendir(path);
    if (!dir) return create_string("");

    /* Build newline-separated list */
    char buffer[4096] = "";
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        /* Skip . and .. */
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        strcat(buffer, entry->d_name);
        strcat(buffer, "\n");
    }
    closedir(dir);

    return create_string(buffer);
}

static Value builtin_dir_exists(Value *args) {
    const char *path = args[0].as.string_val;
    struct stat st;
    if (stat(path, &st) != 0) return create_bool(false);
    return create_bool(S_ISDIR(st.st_mode));
}

static Value builtin_getcwd(Value *args) {
    (void)args;  /* Unused */
    char buffer[1024];
    if (getcwd(buffer, sizeof(buffer)) == NULL) {
        return create_string("");
    }
    return create_string(buffer);
}

static Value builtin_chdir(Value *args) {
    const char *path = args[0].as.string_val;
    return create_int(chdir(path) == 0 ? 0 : -1);
}

/* Path Operations */
static Value builtin_path_isfile(Value *args) {
    const char *path = args[0].as.string_val;
    struct stat st;
    if (stat(path, &st) != 0) return create_bool(false);
    return create_bool(S_ISREG(st.st_mode));
}

static Value builtin_path_isdir(Value *args) {
    const char *path = args[0].as.string_val;
    struct stat st;
    if (stat(path, &st) != 0) return create_bool(false);
    return create_bool(S_ISDIR(st.st_mode));
}

static Value builtin_path_join(Value *args) {
    const char *a = args[0].as.string_val;
    const char *b = args[1].as.string_val;
    char buffer[2048];

    /* Handle various cases */
    if (strlen(a) == 0) {
        snprintf(buffer, sizeof(buffer), "%s", b);
    } else if (a[strlen(a) - 1] == '/') {
        snprintf(buffer, sizeof(buffer), "%s%s", a, b);
    } else {
        snprintf(buffer, sizeof(buffer), "%s/%s", a, b);
    }

    return create_string(buffer);
}

static Value builtin_path_basename(Value *args) {
    const char *path = args[0].as.string_val;
    char *path_copy = strdup(path);
    char *base = basename(path_copy);
    Value result = create_string(base);
    free(path_copy);
    return result;
}

static Value builtin_path_dirname(Value *args) {
    const char *path = args[0].as.string_val;
    char *path_copy = strdup(path);
    char *dir = dirname(path_copy);
    Value result = create_string(dir);
    free(path_copy);
    return result;
}

/* Process Operations */
static Value builtin_system(Value *args) {
    const char *command = args[0].as.string_val;
    return create_int(system(command));
}

static Value builtin_exit(Value *args) {
    int code = (int)args[0].as.int_val;
    exit(code);
    return create_void();  /* Never reached */
}

static Value builtin_getenv(Value *args) {
    const char *name = args[0].as.string_val;
    const char *value = getenv(name);
    return create_string(value ? value : "");
}

/* ==========================================================================
 * End of Built-in OS Functions
 * ========================================================================== */

/* Print a value (used by println and eval) */
static void print_value(Value val) {
    switch (val.type) {
        case VAL_INT:
            printf("%lld", val.as.int_val);
            break;
        case VAL_FLOAT:
            printf("%g", val.as.float_val);
            break;
        case VAL_BOOL:
            printf("%s", val.as.bool_val ? "true" : "false");
            break;
        case VAL_STRING:
            printf("%s", val.as.string_val);
            break;
        case VAL_ARRAY: {
            /* Print array as [elem1, elem2, ...] */
            Array *arr = val.as.array_val;
            printf("[");
            for (int i = 0; i < arr->length; i++) {
                if (i > 0) printf(", ");
                switch (arr->element_type) {
                    case VAL_INT:
                        printf("%lld", ((long long*)arr->data)[i]);
                        break;
                    case VAL_FLOAT:
                        printf("%g", ((double*)arr->data)[i]);
                        break;
                    case VAL_BOOL:
                        printf("%s", ((bool*)arr->data)[i] ? "true" : "false");
                        break;
                    case VAL_STRING:
                        printf("\"%s\"", ((char**)arr->data)[i]);
                        break;
                    default:
                        break;
                }
            }
            printf("]");
            break;
        }
        case VAL_STRUCT: {
            /* Print struct as StructName { field1: value1, field2: value2 } */
            StructValue *sv = val.as.struct_val;
            printf("%s { ", sv->struct_name);
            for (int i = 0; i < sv->field_count; i++) {
                if (i > 0) printf(", ");
                printf("%s: ", sv->field_names[i]);
                print_value(sv->field_values[i]);
            }
            printf(" }");
            break;
        }
        case VAL_VOID:
            printf("void");
            break;
    }
}

/* ==========================================================================
 * Math and Utility Built-in Functions
 * ========================================================================== */

static Value builtin_abs(Value *args) {
    if (args[0].type == VAL_INT) {
        long long val = args[0].as.int_val;
        return create_int(val < 0 ? -val : val);
    } else if (args[0].type == VAL_FLOAT) {
        double val = args[0].as.float_val;
        return create_float(val < 0 ? -val : val);
    }
    fprintf(stderr, "Error: abs requires int or float argument\n");
    return create_void();
}

static Value builtin_min(Value *args) {
    if (args[0].type == VAL_INT && args[1].type == VAL_INT) {
        long long a = args[0].as.int_val;
        long long b = args[1].as.int_val;
        return create_int(a < b ? a : b);
    } else if (args[0].type == VAL_FLOAT && args[1].type == VAL_FLOAT) {
        double a = args[0].as.float_val;
        double b = args[1].as.float_val;
        return create_float(a < b ? a : b);
    }
    fprintf(stderr, "Error: min requires two arguments of same type (int or float)\n");
    return create_void();
}

static Value builtin_max(Value *args) {
    if (args[0].type == VAL_INT && args[1].type == VAL_INT) {
        long long a = args[0].as.int_val;
        long long b = args[1].as.int_val;
        return create_int(a > b ? a : b);
    } else if (args[0].type == VAL_FLOAT && args[1].type == VAL_FLOAT) {
        double a = args[0].as.float_val;
        double b = args[1].as.float_val;
        return create_float(a > b ? a : b);
    }
    fprintf(stderr, "Error: max requires two arguments of same type (int or float)\n");
    return create_void();
}

/* Advanced Math Functions */
static Value builtin_sqrt(Value *args) {
    if (args[0].type == VAL_FLOAT) {
        return create_float(sqrt(args[0].as.float_val));
    } else if (args[0].type == VAL_INT) {
        return create_float(sqrt((double)args[0].as.int_val));
    }
    fprintf(stderr, "Error: sqrt requires numeric argument\n");
    return create_void();
}

static Value builtin_pow(Value *args) {
    double base, exponent;
    if (args[0].type == VAL_FLOAT) {
        base = args[0].as.float_val;
    } else if (args[0].type == VAL_INT) {
        base = (double)args[0].as.int_val;
    } else {
        fprintf(stderr, "Error: pow requires numeric arguments\n");
        return create_void();
    }
    
    if (args[1].type == VAL_FLOAT) {
        exponent = args[1].as.float_val;
    } else if (args[1].type == VAL_INT) {
        exponent = (double)args[1].as.int_val;
    } else {
        fprintf(stderr, "Error: pow requires numeric arguments\n");
        return create_void();
    }
    
    return create_float(pow(base, exponent));
}

static Value builtin_floor(Value *args) {
    if (args[0].type == VAL_FLOAT) {
        return create_float(floor(args[0].as.float_val));
    } else if (args[0].type == VAL_INT) {
        return create_int(args[0].as.int_val);  /* Already an integer */
    }
    fprintf(stderr, "Error: floor requires numeric argument\n");
    return create_void();
}

static Value builtin_ceil(Value *args) {
    if (args[0].type == VAL_FLOAT) {
        return create_float(ceil(args[0].as.float_val));
    } else if (args[0].type == VAL_INT) {
        return create_int(args[0].as.int_val);  /* Already an integer */
    }
    fprintf(stderr, "Error: ceil requires numeric argument\n");
    return create_void();
}

static Value builtin_round(Value *args) {
    if (args[0].type == VAL_FLOAT) {
        return create_float(round(args[0].as.float_val));
    } else if (args[0].type == VAL_INT) {
        return create_int(args[0].as.int_val);  /* Already an integer */
    }
    fprintf(stderr, "Error: round requires numeric argument\n");
    return create_void();
}

/* Trigonometric Functions */
static Value builtin_sin(Value *args) {
    if (args[0].type == VAL_FLOAT) {
        return create_float(sin(args[0].as.float_val));
    } else if (args[0].type == VAL_INT) {
        return create_float(sin((double)args[0].as.int_val));
    }
    fprintf(stderr, "Error: sin requires numeric argument\n");
    return create_void();
}

static Value builtin_cos(Value *args) {
    if (args[0].type == VAL_FLOAT) {
        return create_float(cos(args[0].as.float_val));
    } else if (args[0].type == VAL_INT) {
        return create_float(cos((double)args[0].as.int_val));
    }
    fprintf(stderr, "Error: cos requires numeric argument\n");
    return create_void();
}

static Value builtin_tan(Value *args) {
    if (args[0].type == VAL_FLOAT) {
        return create_float(tan(args[0].as.float_val));
    } else if (args[0].type == VAL_INT) {
        return create_float(tan((double)args[0].as.int_val));
    }
    fprintf(stderr, "Error: tan requires numeric argument\n");
    return create_void();
}

static Value builtin_print(Value *args) {
    print_value(args[0]);
    return create_void();
}

static Value builtin_println(Value *args) {
    print_value(args[0]);
    printf("\n");
    return create_void();
}

/* ============================================================================
 * String Operations
 * ========================================================================== */

static Value builtin_str_length(Value *args) {
    if (args[0].type != VAL_STRING) {
        fprintf(stderr, "Error: str_length requires string argument\n");
        return create_void();
    }
    return create_int(strlen(args[0].as.string_val));
}

static Value builtin_str_concat(Value *args) {
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING) {
        fprintf(stderr, "Error: str_concat requires two string arguments\n");
        return create_void();
    }
    
    size_t len1 = strlen(args[0].as.string_val);
    size_t len2 = strlen(args[1].as.string_val);
    char *result = malloc(len1 + len2 + 1);
    if (!result) {
        fprintf(stderr, "Error: Memory allocation failed in str_concat\n");
        return create_void();
    }
    
    strcpy(result, args[0].as.string_val);
    strcat(result, args[1].as.string_val);
    
    return create_string(result);
}

static Value builtin_str_substring(Value *args) {
    if (args[0].type != VAL_STRING) {
        fprintf(stderr, "Error: str_substring requires string as first argument\n");
        return create_void();
    }
    if (args[1].type != VAL_INT || args[2].type != VAL_INT) {
        fprintf(stderr, "Error: str_substring requires integer start and length\n");
        return create_void();
    }
    
    const char *str = args[0].as.string_val;
    long long start = args[1].as.int_val;
    long long length = args[2].as.int_val;
    long long str_len = strlen(str);
    
    if (start < 0 || start >= str_len) {
        fprintf(stderr, "Error: str_substring start index out of bounds\n");
        return create_void();
    }
    
    if (length < 0) {
        fprintf(stderr, "Error: str_substring length cannot be negative\n");
        return create_void();
    }
    
    /* Adjust length if it exceeds string bounds */
    if (start + length > str_len) {
        length = str_len - start;
    }
    
    char *result = malloc(length + 1);
    if (!result) {
        fprintf(stderr, "Error: Memory allocation failed in str_substring\n");
        return create_void();
    }
    
    strncpy(result, str + start, length);
    result[length] = '\0';
    
    return create_string(result);
}

static Value builtin_str_contains(Value *args) {
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING) {
        fprintf(stderr, "Error: str_contains requires two string arguments\n");
        return create_void();
    }
    
    const char *str = args[0].as.string_val;
    const char *substr = args[1].as.string_val;
    
    return create_bool(strstr(str, substr) != NULL);
}

static Value builtin_str_equals(Value *args) {
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING) {
        fprintf(stderr, "Error: str_equals requires two string arguments\n");
        return create_void();
    }
    
    return create_bool(strcmp(args[0].as.string_val, args[1].as.string_val) == 0);
}

/* ==========================================================================
 * Array Built-in Functions (With Bounds Checking!)
 * ========================================================================== */

static Value builtin_at(Value *args) {
    /* at(array, index) -> element */
    if (args[0].type != VAL_ARRAY) {
        fprintf(stderr, "Error: at() requires an array as first argument\n");
        return create_void();
    }
    if (args[1].type != VAL_INT) {
        fprintf(stderr, "Error: at() requires an integer index\n");
        return create_void();
    }
    
    Array *arr = args[0].as.array_val;
    long long index = args[1].as.int_val;
    
    /* BOUNDS CHECKING - This is the safety guarantee! */
    if (index < 0 || index >= arr->length) {
        fprintf(stderr, "Runtime Error: Array index %lld out of bounds [0..%d)\n",
                index, arr->length);
        exit(1);  /* Fail fast - no undefined behavior! */
    }
    
    /* Return element based on type */
    switch (arr->element_type) {
        case VAL_INT:
            return create_int(((long long*)arr->data)[index]);
        case VAL_FLOAT:
            return create_float(((double*)arr->data)[index]);
        case VAL_BOOL:
            return create_bool(((bool*)arr->data)[index]);
        case VAL_STRING:
            return create_string(((char**)arr->data)[index]);
        default:
            fprintf(stderr, "Error: Unsupported array element type\n");
            return create_void();
    }
}

static Value builtin_array_length(Value *args) {
    /* array_length(array) -> int */
    if (args[0].type != VAL_ARRAY) {
        fprintf(stderr, "Error: array_length() requires an array argument\n");
        return create_void();
    }
    
    return create_int(args[0].as.array_val->length);
}

static Value builtin_array_new(Value *args) {
    /* array_new(size, default_value) -> array */
    if (args[0].type != VAL_INT) {
        fprintf(stderr, "Error: array_new() requires an integer size\n");
        return create_void();
    }
    
    long long size = args[0].as.int_val;
    if (size < 0) {
        fprintf(stderr, "Error: array_new() size must be non-negative\n");
        return create_void();
    }
    
    ValueType elem_type = args[1].type;
    Value arr = create_array(elem_type, size, size);
    
    /* Initialize all elements with default value */
    for (long long i = 0; i < size; i++) {
        switch (elem_type) {
            case VAL_INT:
                ((long long*)arr.as.array_val->data)[i] = args[1].as.int_val;
                break;
            case VAL_FLOAT:
                ((double*)arr.as.array_val->data)[i] = args[1].as.float_val;
                break;
            case VAL_BOOL:
                ((bool*)arr.as.array_val->data)[i] = args[1].as.bool_val;
                break;
            case VAL_STRING:
                ((char**)arr.as.array_val->data)[i] = strdup(args[1].as.string_val);
                break;
            default:
                break;
        }
    }
    
    return arr;
}

static Value builtin_array_set(Value *args) {
    /* array_set(array, index, value) -> void */
    if (args[0].type != VAL_ARRAY) {
        fprintf(stderr, "Error: array_set() requires an array as first argument\n");
        return create_void();
    }
    if (args[1].type != VAL_INT) {
        fprintf(stderr, "Error: array_set() requires an integer index\n");
        return create_void();
    }
    
    Array *arr = args[0].as.array_val;
    long long index = args[1].as.int_val;
    
    /* BOUNDS CHECKING */
    if (index < 0 || index >= arr->length) {
        fprintf(stderr, "Runtime Error: Array index %lld out of bounds [0..%d)\n",
                index, arr->length);
        exit(1);  /* Fail fast! */
    }
    
    /* Set element based on type */
    switch (arr->element_type) {
        case VAL_INT:
            if (args[2].type != VAL_INT) {
                fprintf(stderr, "Error: Type mismatch in array_set\n");
                return create_void();
            }
            ((long long*)arr->data)[index] = args[2].as.int_val;
            break;
        case VAL_FLOAT:
            if (args[2].type != VAL_FLOAT) {
                fprintf(stderr, "Error: Type mismatch in array_set\n");
                return create_void();
            }
            ((double*)arr->data)[index] = args[2].as.float_val;
            break;
        case VAL_BOOL:
            if (args[2].type != VAL_BOOL) {
                fprintf(stderr, "Error: Type mismatch in array_set\n");
                return create_void();
            }
            ((bool*)arr->data)[index] = args[2].as.bool_val;
            break;
        case VAL_STRING:
            if (args[2].type != VAL_STRING) {
                fprintf(stderr, "Error: Type mismatch in array_set\n");
                return create_void();
            }
            /* Free old string if exists */
            if (((char**)arr->data)[index]) {
                free(((char**)arr->data)[index]);
            }
            ((char**)arr->data)[index] = strdup(args[2].as.string_val);
            break;
        default:
            fprintf(stderr, "Error: Unsupported array element type\n");
            break;
    }
    
    return create_void();
}

/* ==========================================================================
 * End of Math and Utility Built-in Functions
 * ========================================================================== */

/* Helper to convert value to boolean */
static bool is_truthy(Value val) {
    switch (val.type) {
        case VAL_BOOL:
            return val.as.bool_val;
        case VAL_INT:
            return val.as.int_val != 0;
        case VAL_FLOAT:
            return val.as.float_val != 0.0;
        case VAL_VOID:
            return false;
        default:
            return true; /* Strings are truthy if non-null */
    }
}

/* Evaluate prefix operation */
static Value eval_prefix_op(ASTNode *node, Environment *env) {
    TokenType op = node->as.prefix_op.op;
    int arg_count = node->as.prefix_op.arg_count;


    /* Arithmetic operators */
    if (op == TOKEN_PLUS || op == TOKEN_MINUS || op == TOKEN_STAR ||
        op == TOKEN_SLASH || op == TOKEN_PERCENT) {
        if (arg_count != 2) {
            fprintf(stderr, "Error: Arithmetic operators require 2 arguments\n");
            return create_void();
        }
        Value left = eval_expression(node->as.prefix_op.args[0], env);
        Value right = eval_expression(node->as.prefix_op.args[1], env);

        if (left.type == VAL_INT && right.type == VAL_INT) {
            long long result;
            switch (op) {
                case TOKEN_PLUS: result = left.as.int_val + right.as.int_val; break;
                case TOKEN_MINUS: result = left.as.int_val - right.as.int_val; break;
                case TOKEN_STAR: result = left.as.int_val * right.as.int_val; break;
                case TOKEN_SLASH:
                    if (right.as.int_val == 0) {
                        fprintf(stderr, "Error: Division by zero\n");
                        return create_void();
                    }
                    result = left.as.int_val / right.as.int_val;
                    break;
                case TOKEN_PERCENT:
                    if (right.as.int_val == 0) {
                        fprintf(stderr, "Error: Modulo by zero\n");
                        return create_void();
                    }
                    result = left.as.int_val % right.as.int_val;
                    break;
                default: result = 0;
            }
            return create_int(result);
        } else if (left.type == VAL_FLOAT && right.type == VAL_FLOAT) {
            double result;
            switch (op) {
                case TOKEN_PLUS: result = left.as.float_val + right.as.float_val; break;
                case TOKEN_MINUS: result = left.as.float_val - right.as.float_val; break;
                case TOKEN_STAR: result = left.as.float_val * right.as.float_val; break;
                case TOKEN_SLASH:
                    if (right.as.float_val == 0.0) {
                        fprintf(stderr, "Error: Division by zero\n");
                        return create_void();
                    }
                    result = left.as.float_val / right.as.float_val;
                    break;
                default: result = 0.0;
            }
            return create_float(result);
        }
    }

    /* Comparison operators */
    if (op == TOKEN_LT || op == TOKEN_LE || op == TOKEN_GT || op == TOKEN_GE) {
        if (arg_count != 2) {
            fprintf(stderr, "Error: Comparison operators require 2 arguments\n");
            return create_void();
        }
        Value left = eval_expression(node->as.prefix_op.args[0], env);
        Value right = eval_expression(node->as.prefix_op.args[1], env);

        if (left.type == VAL_INT && right.type == VAL_INT) {
            bool result;
            switch (op) {
                case TOKEN_LT: result = left.as.int_val < right.as.int_val; break;
                case TOKEN_LE: result = left.as.int_val <= right.as.int_val; break;
                case TOKEN_GT: result = left.as.int_val > right.as.int_val; break;
                case TOKEN_GE: result = left.as.int_val >= right.as.int_val; break;
                default: result = false;
            }
            return create_bool(result);
        }
    }

    /* Equality operators */
    if (op == TOKEN_EQ || op == TOKEN_NE) {
        if (arg_count != 2) {
            fprintf(stderr, "Error: Equality operators require 2 arguments\n");
            return create_void();
        }
        Value left = eval_expression(node->as.prefix_op.args[0], env);
        Value right = eval_expression(node->as.prefix_op.args[1], env);

        bool equal = false;
        if (left.type == right.type) {
            switch (left.type) {
                case VAL_INT: equal = left.as.int_val == right.as.int_val; break;
                case VAL_FLOAT: equal = left.as.float_val == right.as.float_val; break;
                case VAL_BOOL: equal = left.as.bool_val == right.as.bool_val; break;
                case VAL_STRING: equal = strcmp(left.as.string_val, right.as.string_val) == 0; break;
                case VAL_STRUCT: {
                    /* Structs are equal if they're the same type and all fields are equal */
                    StructValue *left_sv = left.as.struct_val;
                    StructValue *right_sv = right.as.struct_val;
                    if (strcmp(left_sv->struct_name, right_sv->struct_name) != 0 ||
                        left_sv->field_count != right_sv->field_count) {
                        equal = false;
                    } else {
                        equal = true;
                        for (int i = 0; i < left_sv->field_count && equal; i++) {
                            Value left_field = left_sv->field_values[i];
                            Value right_field = right_sv->field_values[i];
                            /* Recursively compare field values (simplified - only int/float/bool/string) */
                            if (left_field.type != right_field.type) {
                                equal = false;
                            } else if (left_field.type == VAL_INT) {
                                equal = left_field.as.int_val == right_field.as.int_val;
                            } else if (left_field.type == VAL_FLOAT) {
                                equal = left_field.as.float_val == right_field.as.float_val;
                            } else if (left_field.type == VAL_BOOL) {
                                equal = left_field.as.bool_val == right_field.as.bool_val;
                            } else if (left_field.type == VAL_STRING) {
                                equal = strcmp(left_field.as.string_val, right_field.as.string_val) == 0;
                            }
                        }
                    }
                    break;
                }
                case VAL_VOID: equal = true; break;  /* void == void */
                case VAL_ARRAY: {
                    /* Arrays are equal if they have same length and all elements equal */
                    Array *left_arr = left.as.array_val;
                    Array *right_arr = right.as.array_val;
                    if (left_arr->length != right_arr->length) {
                        equal = false;
                    } else {
                        equal = true;
                        for (int i = 0; i < left_arr->length && equal; i++) {
                            switch (left_arr->element_type) {
                                case VAL_INT:
                                    equal = ((long long*)left_arr->data)[i] == ((long long*)right_arr->data)[i];
                                    break;
                                case VAL_FLOAT:
                                    equal = ((double*)left_arr->data)[i] == ((double*)right_arr->data)[i];
                                    break;
                                case VAL_BOOL:
                                    equal = ((bool*)left_arr->data)[i] == ((bool*)right_arr->data)[i];
                                    break;
                                case VAL_STRING:
                                    equal = strcmp(((char**)left_arr->data)[i], ((char**)right_arr->data)[i]) == 0;
                                    break;
                                default:
                                    equal = false;
                                    break;
                            }
                        }
                    }
                    break;
                }
            }
        }

        return create_bool(op == TOKEN_EQ ? equal : !equal);
    }

    /* Logical operators */
    if (op == TOKEN_AND || op == TOKEN_OR) {
        if (arg_count != 2) {
            fprintf(stderr, "Error: Logical operators require 2 arguments\n");
            return create_void();
        }
        Value left = eval_expression(node->as.prefix_op.args[0], env);

        if (op == TOKEN_AND) {
            if (!is_truthy(left)) return create_bool(false);
            Value right = eval_expression(node->as.prefix_op.args[1], env);
            return create_bool(is_truthy(right));
        } else { /* OR */
            if (is_truthy(left)) return create_bool(true);
            Value right = eval_expression(node->as.prefix_op.args[1], env);
            return create_bool(is_truthy(right));
        }
    }

    if (op == TOKEN_NOT) {
        if (arg_count != 1) {
            fprintf(stderr, "Error: 'not' requires 1 argument\n");
            return create_void();
        }
        Value arg = eval_expression(node->as.prefix_op.args[0], env);
        return create_bool(!is_truthy(arg));
    }

    return create_void();
}

/* Evaluate function call */
static Value eval_call(ASTNode *node, Environment *env) {
    const char *name = node->as.call.name;


    /* Special built-in: range (used in for loops only) */
    if (strcmp(name, "range") == 0) {
        /* This should not be called directly */
        return create_void();
    }

    /* Check for built-in OS functions */
    /* Evaluate arguments first */
    Value args[16];  /* Max args for function calls */
    for (int i = 0; i < node->as.call.arg_count; i++) {
        args[i] = eval_expression(node->as.call.args[i], env);
    }

    /* File operations */
    if (strcmp(name, "file_read") == 0) return builtin_file_read(args);
    if (strcmp(name, "file_write") == 0) return builtin_file_write(args);
    if (strcmp(name, "file_append") == 0) return builtin_file_append(args);
    if (strcmp(name, "file_remove") == 0) return builtin_file_remove(args);
    if (strcmp(name, "file_rename") == 0) return builtin_file_rename(args);
    if (strcmp(name, "file_exists") == 0) return builtin_file_exists(args);
    if (strcmp(name, "file_size") == 0) return builtin_file_size(args);

    /* Directory operations */
    if (strcmp(name, "dir_create") == 0) return builtin_dir_create(args);
    if (strcmp(name, "dir_remove") == 0) return builtin_dir_remove(args);
    if (strcmp(name, "dir_list") == 0) return builtin_dir_list(args);
    if (strcmp(name, "dir_exists") == 0) return builtin_dir_exists(args);
    if (strcmp(name, "getcwd") == 0) return builtin_getcwd(args);
    if (strcmp(name, "chdir") == 0) return builtin_chdir(args);

    /* Path operations */
    if (strcmp(name, "path_isfile") == 0) return builtin_path_isfile(args);
    if (strcmp(name, "path_isdir") == 0) return builtin_path_isdir(args);
    if (strcmp(name, "path_join") == 0) return builtin_path_join(args);
    if (strcmp(name, "path_basename") == 0) return builtin_path_basename(args);
    if (strcmp(name, "path_dirname") == 0) return builtin_path_dirname(args);

    /* Process operations */
    if (strcmp(name, "system") == 0) return builtin_system(args);
    if (strcmp(name, "exit") == 0) return builtin_exit(args);
    if (strcmp(name, "getenv") == 0) return builtin_getenv(args);

    /* Math and utility functions */
    if (strcmp(name, "abs") == 0) return builtin_abs(args);
    if (strcmp(name, "min") == 0) return builtin_min(args);
    if (strcmp(name, "max") == 0) return builtin_max(args);
    if (strcmp(name, "print") == 0) return builtin_print(args);
    if (strcmp(name, "println") == 0) return builtin_println(args);
    
    /* Advanced math functions */
    if (strcmp(name, "sqrt") == 0) return builtin_sqrt(args);
    if (strcmp(name, "pow") == 0) return builtin_pow(args);
    if (strcmp(name, "floor") == 0) return builtin_floor(args);
    if (strcmp(name, "ceil") == 0) return builtin_ceil(args);
    if (strcmp(name, "round") == 0) return builtin_round(args);
    
    /* Trigonometric functions */
    if (strcmp(name, "sin") == 0) return builtin_sin(args);
    if (strcmp(name, "cos") == 0) return builtin_cos(args);
    if (strcmp(name, "tan") == 0) return builtin_tan(args);
    
    /* String operations */
    if (strcmp(name, "str_length") == 0) return builtin_str_length(args);
    if (strcmp(name, "str_concat") == 0) return builtin_str_concat(args);
    if (strcmp(name, "str_substring") == 0) return builtin_str_substring(args);
    if (strcmp(name, "str_contains") == 0) return builtin_str_contains(args);
    if (strcmp(name, "str_equals") == 0) return builtin_str_equals(args);
    
    /* Advanced string operations */
    if (strcmp(name, "char_at") == 0) {
        if (args[0].type != VAL_STRING || args[1].type != VAL_INT) {
            fprintf(stderr, "Error: char_at requires string and int\n");
            return create_void();
        }
        const char *str = args[0].as.string_val;
        long long index = args[1].as.int_val;
        int len = strlen(str);
        if (index < 0 || index >= len) {
            fprintf(stderr, "Error: Index %lld out of bounds (string length %d)\n", index, len);
            return create_void();
        }
        return create_int((unsigned char)str[index]);
    }
    
    if (strcmp(name, "string_from_char") == 0) {
        if (args[0].type != VAL_INT) {
            fprintf(stderr, "Error: string_from_char requires int\n");
            return create_void();
        }
        char buffer[2];
        buffer[0] = (char)args[0].as.int_val;
        buffer[1] = '\0';
        return create_string(buffer);
    }
    
    /* Character classification */
    if (strcmp(name, "is_digit") == 0) {
        if (args[0].type != VAL_INT) return create_bool(false);
        int c = (int)args[0].as.int_val;
        return create_bool(c >= '0' && c <= '9');
    }
    
    if (strcmp(name, "is_alpha") == 0) {
        if (args[0].type != VAL_INT) return create_bool(false);
        int c = (int)args[0].as.int_val;
        return create_bool((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'));
    }
    
    if (strcmp(name, "is_alnum") == 0) {
        if (args[0].type != VAL_INT) return create_bool(false);
        int c = (int)args[0].as.int_val;
        return create_bool((c >= '0' && c <= '9') || 
                           (c >= 'a' && c <= 'z') || 
                           (c >= 'A' && c <= 'Z'));
    }
    
    if (strcmp(name, "is_whitespace") == 0) {
        if (args[0].type != VAL_INT) return create_bool(false);
        int c = (int)args[0].as.int_val;
        return create_bool(c == ' ' || c == '\t' || c == '\n' || c == '\r');
    }
    
    if (strcmp(name, "is_upper") == 0) {
        if (args[0].type != VAL_INT) return create_bool(false);
        int c = (int)args[0].as.int_val;
        return create_bool(c >= 'A' && c <= 'Z');
    }
    
    if (strcmp(name, "is_lower") == 0) {
        if (args[0].type != VAL_INT) return create_bool(false);
        int c = (int)args[0].as.int_val;
        return create_bool(c >= 'a' && c <= 'z');
    }
    
    /* Type conversions */
    if (strcmp(name, "int_to_string") == 0) {
        if (args[0].type != VAL_INT) {
            return create_string("0");
        }
        char buffer[32];
        snprintf(buffer, sizeof(buffer), "%lld", args[0].as.int_val);
        return create_string(buffer);
    }
    
    if (strcmp(name, "string_to_int") == 0) {
        if (args[0].type != VAL_STRING) {
            return create_int(0);
        }
        long long result = strtoll(args[0].as.string_val, NULL, 10);
        return create_int(result);
    }
    
    if (strcmp(name, "digit_value") == 0) {
        if (args[0].type != VAL_INT) return create_int(-1);
        int c = (int)args[0].as.int_val;
        if (c >= '0' && c <= '9') {
            return create_int(c - '0');
        }
        return create_int(-1);
    }
    
    if (strcmp(name, "char_to_lower") == 0) {
        if (args[0].type != VAL_INT) return create_int(args[0].as.int_val);
        int c = (int)args[0].as.int_val;
        if (c >= 'A' && c <= 'Z') {
            return create_int(c + 32);
        }
        return create_int(c);
    }
    
    if (strcmp(name, "char_to_upper") == 0) {
        if (args[0].type != VAL_INT) return create_int(args[0].as.int_val);
        int c = (int)args[0].as.int_val;
        if (c >= 'a' && c <= 'z') {
            return create_int(c - 32);
        }
        return create_int(c);
    }
    
    /* Array operations */
    if (strcmp(name, "at") == 0) return builtin_at(args);
    if (strcmp(name, "array_length") == 0) return builtin_array_length(args);
    if (strcmp(name, "array_new") == 0) return builtin_array_new(args);
    if (strcmp(name, "array_set") == 0) return builtin_array_set(args);
    
    /* list_int operations - delegate to C runtime */
    if (strcmp(name, "list_int_new") == 0) {
        List_int *list = list_int_new();
        Value result = create_int((long long)list);
        return result;
    }
    if (strcmp(name, "list_int_with_capacity") == 0) {
        List_int *list = list_int_with_capacity(args[0].as.int_val);
        return create_int((long long)list);
    }
    if (strcmp(name, "list_int_push") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        list_int_push(list, args[1].as.int_val);
        return create_void();
    }
    if (strcmp(name, "list_int_pop") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        return create_int(list_int_pop(list));
    }
    if (strcmp(name, "list_int_get") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        return create_int(list_int_get(list, args[1].as.int_val));
    }
    if (strcmp(name, "list_int_set") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        list_int_set(list, args[1].as.int_val, args[2].as.int_val);
        return create_void();
    }
    if (strcmp(name, "list_int_insert") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        list_int_insert(list, args[1].as.int_val, args[2].as.int_val);
        return create_void();
    }
    if (strcmp(name, "list_int_remove") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        return create_int(list_int_remove(list, args[1].as.int_val));
    }
    if (strcmp(name, "list_int_length") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        return create_int(list_int_length(list));
    }
    if (strcmp(name, "list_int_capacity") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        return create_int(list_int_capacity(list));
    }
    if (strcmp(name, "list_int_is_empty") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        return create_bool(list_int_is_empty(list));
    }
    if (strcmp(name, "list_int_clear") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        list_int_clear(list);
        return create_void();
    }
    if (strcmp(name, "list_int_free") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        list_int_free(list);
        return create_void();
    }

    /* list_string operations - delegate to C runtime */
    if (strcmp(name, "list_string_new") == 0) {
        List_string *list = list_string_new();
        Value result = create_int((long long)list);
        return result;
    }
    if (strcmp(name, "list_string_with_capacity") == 0) {
        List_string *list = list_string_with_capacity(args[0].as.int_val);
        return create_int((long long)list);
    }
    if (strcmp(name, "list_string_push") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        list_string_push(list, args[1].as.string_val);
        return create_void();
    }
    if (strcmp(name, "list_string_pop") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        char *str = list_string_pop(list);
        Value result = create_string(str);
        free(str);  /* list_string_pop returns strdup'd string */
        return result;
    }
    if (strcmp(name, "list_string_get") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        char *str = list_string_get(list, args[1].as.int_val);
        return create_string(str);
    }
    if (strcmp(name, "list_string_set") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        list_string_set(list, args[1].as.int_val, args[2].as.string_val);
        return create_void();
    }
    if (strcmp(name, "list_string_insert") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        list_string_insert(list, args[1].as.int_val, args[2].as.string_val);
        return create_void();
    }
    if (strcmp(name, "list_string_remove") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        char *str = list_string_remove(list, args[1].as.int_val);
        Value result = create_string(str);
        free(str);  /* list_string_remove returns strdup'd string */
        return result;
    }
    if (strcmp(name, "list_string_length") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        return create_int(list_string_length(list));
    }
    if (strcmp(name, "list_string_capacity") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        return create_int(list_string_capacity(list));
    }
    if (strcmp(name, "list_string_is_empty") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        return create_bool(list_string_is_empty(list));
    }
    if (strcmp(name, "list_string_clear") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        list_string_clear(list);
        return create_void();
    }
    if (strcmp(name, "list_string_free") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        list_string_free(list);
        return create_void();
    }

    /* list_token operations - delegate to C runtime */
    /* Note: Token structs are stored as pointers for now */
    /* When we rewrite lexer in nanolang, we'll use proper Token struct values */
    if (strcmp(name, "list_token_new") == 0) {
        List_token *list = list_token_new();
        Value result = create_int((long long)list);
        return result;
    }
    if (strcmp(name, "list_token_with_capacity") == 0) {
        List_token *list = list_token_with_capacity(args[0].as.int_val);
        return create_int((long long)list);
    }
    if (strcmp(name, "list_token_push") == 0) {
        List_token *list = (List_token*)args[0].as.int_val;
        /* For now, args[1] should be a Token struct pointer */
        /* When we have proper Token struct support, this will change */
        Token *token = (Token*)args[1].as.int_val;
        if (token) {
            list_token_push(list, *token);
        }
        return create_void();
    }
    if (strcmp(name, "list_token_pop") == 0) {
        List_token *list = (List_token*)args[0].as.int_val;
        Token token = list_token_pop(list);
        /* Return token as struct value - for now return pointer */
        /* TODO: Convert Token to proper struct value when we have Token struct support */
        Token *token_ptr = malloc(sizeof(Token));
        *token_ptr = token;
        return create_int((long long)token_ptr);
    }
    if (strcmp(name, "list_token_get") == 0) {
        List_token *list = (List_token*)args[0].as.int_val;
        Token *token = list_token_get(list, args[1].as.int_val);
        /* Return token pointer for now */
        return create_int((long long)token);
    }
    if (strcmp(name, "list_token_set") == 0) {
        List_token *list = (List_token*)args[0].as.int_val;
        Token *token = (Token*)args[2].as.int_val;
        if (token) {
            list_token_set(list, args[1].as.int_val, *token);
        }
        return create_void();
    }
    if (strcmp(name, "list_token_insert") == 0) {
        List_token *list = (List_token*)args[0].as.int_val;
        Token *token = (Token*)args[2].as.int_val;
        if (token) {
            list_token_insert(list, args[1].as.int_val, *token);
        }
        return create_void();
    }
    if (strcmp(name, "list_token_remove") == 0) {
        List_token *list = (List_token*)args[0].as.int_val;
        Token token = list_token_remove(list, args[1].as.int_val);
        Token *token_ptr = malloc(sizeof(Token));
        *token_ptr = token;
        return create_int((long long)token_ptr);
    }
    if (strcmp(name, "list_token_length") == 0) {
        List_token *list = (List_token*)args[0].as.int_val;
        return create_int(list_token_length(list));
    }
    if (strcmp(name, "list_token_capacity") == 0) {
        List_token *list = (List_token*)args[0].as.int_val;
        return create_int(list_token_capacity(list));
    }
    if (strcmp(name, "list_token_is_empty") == 0) {
        List_token *list = (List_token*)args[0].as.int_val;
        return create_bool(list_token_is_empty(list));
    }
    if (strcmp(name, "list_token_clear") == 0) {
        List_token *list = (List_token*)args[0].as.int_val;
        list_token_clear(list);
        return create_void();
    }
    if (strcmp(name, "list_token_free") == 0) {
        List_token *list = (List_token*)args[0].as.int_val;
        list_token_free(list);
        return create_void();
    }

    /* External C library functions */
    if (strcmp(name, "rand") == 0) {
        return create_int(rand());
    }
    if (strcmp(name, "srand") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_INT) {
            fprintf(stderr, "Error: srand expects 1 int argument\n");
            return create_void();
        }
        srand((unsigned int)args[0].as.int_val);
        return create_void();
    }
    if (strcmp(name, "time") == 0) {
        /* Simplified: ignore the argument, just return current time */
        return create_int((long long)time(NULL));
    }

    /* Get user-defined function */
    Function *func = env_get_function(env, name);
    if (!func) {
        fprintf(stderr, "Error: Undefined function '%s'\n", name);
        return create_void();
    }

    /* If built-in with no body, already handled above */
    if (func->body == NULL) {
        fprintf(stderr, "Error: Built-in function '%s' not implemented in interpreter\n", name);
        return create_void();
    }

    /* Trace function call */
    const char **param_names = NULL;
    if (func->params) {
        param_names = malloc(sizeof(char*) * func->param_count);
        for (int i = 0; i < func->param_count; i++) {
            param_names[i] = func->params[i].name;
        }
    }
    tracing_push_call(name);
    trace_function_call(name, args, node->as.call.arg_count, param_names, 
                        node->line, node->column);
    if (param_names) free(param_names);

    /* Create new environment for function */
    int old_symbol_count = env->symbol_count;

    /* Bind parameters with copies of string values */
    for (int i = 0; i < func->param_count; i++) {
        Value param_value = args[i];

        /* Make a deep copy of string values to avoid memory corruption */
        if (param_value.type == VAL_STRING) {
            param_value = create_string(args[i].as.string_val);
        }

        env_define_var(env, func->params[i].name, func->params[i].type, false, param_value);
    }

    /* Execute function body */
    Value result = create_void();
    for (int i = 0; i < func->body->as.block.count; i++) {
        ASTNode *stmt = func->body->as.block.statements[i];
        if (stmt->type == AST_RETURN) {
            if (stmt->as.return_stmt.value) {
                result = eval_expression(stmt->as.return_stmt.value, env);
            }
            break;
        }
        result = eval_statement(stmt, env);
    }

    /* Pop call stack */
    tracing_pop_call();

    /* Make a copy of the result if it's a string BEFORE cleaning up parameters */
    Value return_value = result;
    if (result.type == VAL_STRING) {
        return_value = create_string(result.as.string_val);
    }

    /* Clean up parameter strings and restore environment */
    for (int i = old_symbol_count; i < env->symbol_count; i++) {
        free(env->symbols[i].name);
        if (env->symbols[i].value.type == VAL_STRING) {
            free(env->symbols[i].value.as.string_val);
        }
    }
    env->symbol_count = old_symbol_count;

    return return_value;
}

/* Evaluate expression */
static Value eval_expression(ASTNode *expr, Environment *env) {
    if (!expr) return create_void();


    switch (expr->type) {
        case AST_NUMBER:
            return create_int(expr->as.number);

        case AST_FLOAT:
            return create_float(expr->as.float_val);

        case AST_STRING:
            return create_string(expr->as.string_val);

        case AST_BOOL:
            return create_bool(expr->as.bool_val);

        case AST_IDENTIFIER: {
            Symbol *sym = env_get_var(env, expr->as.identifier);
            if (!sym) {
                fprintf(stderr, "Error: Undefined variable '%s'\n", expr->as.identifier);
                return create_void();
            }
            
            /* Trace variable read */
#ifdef TRACING_ENABLED
            const char *scope = (g_tracing_config.call_stack_size > 0) ?
                g_tracing_config.call_stack[g_tracing_config.call_stack_size - 1] : NULL;
            trace_var_read(expr->as.identifier, sym->value, expr->line, expr->column, scope);
#else
            trace_var_read(expr->as.identifier, sym->value, expr->line, expr->column, NULL);
#endif
            
            return sym->value;
        }

        case AST_PREFIX_OP:
            return eval_prefix_op(expr, env);

        case AST_CALL:
            return eval_call(expr, env);

        case AST_ARRAY_LITERAL: {
            /* Evaluate array literal: [1, 2, 3] */
            int count = expr->as.array_literal.element_count;
            
            /* Empty array */
            if (count == 0) {
                /* Create empty array - type will be determined by context */
                return create_array(VAL_INT, 0, 0);  /* Default to int for now */
            }
            
            /* Evaluate first element to determine type */
            Value first = eval_expression(expr->as.array_literal.elements[0], env);
            ValueType elem_type = first.type;
            
            /* Create array */
            Value arr = create_array(elem_type, count, count);
            
            /* Set elements */
            for (int i = 0; i < count; i++) {
                Value elem = eval_expression(expr->as.array_literal.elements[i], env);
                
                /* Store element in array data */
                switch (elem_type) {
                    case VAL_INT:
                        ((long long*)arr.as.array_val->data)[i] = elem.as.int_val;
                        break;
                    case VAL_FLOAT:
                        ((double*)arr.as.array_val->data)[i] = elem.as.float_val;
                        break;
                    case VAL_BOOL:
                        ((bool*)arr.as.array_val->data)[i] = elem.as.bool_val;
                        break;
                    case VAL_STRING:
                        ((char**)arr.as.array_val->data)[i] = strdup(elem.as.string_val);
                        break;
                    default:
                        fprintf(stderr, "Error: Unsupported array element type\n");
                        break;
                }
            }
            
            return arr;
        }

        case AST_IF: {
            Value cond = eval_expression(expr->as.if_stmt.condition, env);
            if (is_truthy(cond)) {
                return eval_statement(expr->as.if_stmt.then_branch, env);
            } else {
                return eval_statement(expr->as.if_stmt.else_branch, env);
            }
        }

        case AST_STRUCT_LITERAL: {
            /* Evaluate struct literal: Point { x: 10, y: 20 } */
            
            const char *struct_name = expr->as.struct_literal.struct_name;
            int field_count = expr->as.struct_literal.field_count;
            
            
            /* Get struct definition to verify field order */
            StructDef *struct_def = env_get_struct(env, struct_name);
            if (!struct_def) {
                fprintf(stderr, "Error: Undefined struct '%s'\n", struct_name);
                return create_void();
            }
            
            
            /* Allocate arrays for field names and values */
            char **field_names = malloc(sizeof(char*) * field_count);
            Value *field_values = malloc(sizeof(Value) * field_count);
            
            
            /* Evaluate each field value */
            for (int i = 0; i < field_count; i++) {
                field_names[i] = expr->as.struct_literal.field_names[i];
                field_values[i] = eval_expression(expr->as.struct_literal.field_values[i], env);
            }
            
            
            /* Create struct value */
            Value result = create_struct(struct_name, field_names, field_values, field_count);
            
            
            /* Free temporary arrays (create_struct makes copies) */
            free(field_names);
            free(field_values);
            
            
            return result;
        }

        case AST_FIELD_ACCESS: {
            /* Special case: Check if this is an enum variant access */
            if (expr->as.field_access.object->type == AST_IDENTIFIER) {
                const char *enum_name = expr->as.field_access.object->as.identifier;
                EnumDef *enum_def = env_get_enum(env, enum_name);
                
                if (enum_def) {
                    /* This is an enum variant access (e.g., Color.Red) */
                    const char *variant_name = expr->as.field_access.field_name;
                    
                    /* Lookup variant value */
                    for (int i = 0; i < enum_def->variant_count; i++) {
                        if (strcmp(enum_def->variant_names[i], variant_name) == 0) {
                            return create_int(enum_def->variant_values[i]);
                        }
                    }
                    
                    fprintf(stderr, "Error: Enum '%s' has no variant '%s'\n",
                            enum_name, variant_name);
                    return create_void();
                }
            }
            
            /* Regular struct field access */
            /* Evaluate field access: point.x */
            Value obj = eval_expression(expr->as.field_access.object, env);
            
            if (obj.type != VAL_STRUCT) {
                fprintf(stderr, "Error: Cannot access field on non-struct value\n");
                return create_void();
            }
            
            const char *field_name = expr->as.field_access.field_name;
            StructValue *sv = obj.as.struct_val;
            
            /* Find field in struct */
            for (int i = 0; i < sv->field_count; i++) {
                if (strcmp(sv->field_names[i], field_name) == 0) {
                    return sv->field_values[i];
                }
            }
            
            fprintf(stderr, "Error: Struct '%s' has no field '%s'\n", 
                    sv->struct_name, field_name);
            return create_void();
        }

        case AST_UNION_CONSTRUCT: {
            /* Evaluate union construction: Status.Ok {} or Result.Error { code: 404 } */
            const char *union_name = expr->as.union_construct.union_name;
            const char *variant_name = expr->as.union_construct.variant_name;
            
            /* Get variant index */
            int variant_idx = env_get_union_variant_index(env, union_name, variant_name);
            if (variant_idx < 0) {
                fprintf(stderr, "Error: Unknown variant '%s' for union '%s'\n", variant_name, union_name);
                return create_void();
            }
            
            /* Evaluate field values */
            int field_count = expr->as.union_construct.field_count;
            char **field_names = NULL;
            Value *field_values = NULL;
            
            if (field_count > 0) {
                field_names = malloc(sizeof(char*) * field_count);
                field_values = malloc(sizeof(Value) * field_count);
                
                for (int i = 0; i < field_count; i++) {
                    field_names[i] = expr->as.union_construct.field_names[i];
                    field_values[i] = eval_expression(expr->as.union_construct.field_values[i], env);
                }
            }
            
            Value result = create_union(union_name, variant_idx, variant_name, 
                                       field_names, field_values, field_count);
            
            /* Free temporary arrays (create_union makes copies) */
            if (field_count > 0) {
                free(field_names);
                free(field_values);
            }
            
            return result;
        }

        case AST_MATCH: {
            /* Evaluate match expression: match status { Ok(x) => 1, Error(e) => 0 } */
            Value match_val = eval_expression(expr->as.match_expr.expr, env);
            
            if (match_val.type != VAL_UNION) {
                fprintf(stderr, "Error: Match expression requires a union value\n");
                return create_void();
            }
            
            UnionValue *uval = match_val.as.union_val;
            
            /* Find matching arm by comparing variant names */
            for (int i = 0; i < expr->as.match_expr.arm_count; i++) {
                const char *pattern_variant = expr->as.match_expr.pattern_variants[i];
                
                if (strcmp(uval->variant_name, pattern_variant) == 0) {
                    /* This arm matches! */
                    const char *binding = expr->as.match_expr.pattern_bindings[i];
                    
                    /* Save environment state for scope */
                    int saved_symbol_count = env->symbol_count;
                    
                    /* Bind the pattern variable to the union value (or a placeholder) */
                    env_define_var(env, binding, TYPE_UNION, false, match_val);
                    
                    /* Evaluate arm body */
                    Value result = eval_expression(expr->as.match_expr.arm_bodies[i], env);
                    
                    /* Restore environment */
                    env->symbol_count = saved_symbol_count;
                    
                    return result;
                }
            }
            
            /* No matching arm found - this should be caught by typechecker */
            fprintf(stderr, "Error: No matching arm for variant '%s'\n", uval->variant_name);
            return create_void();
        }

        default:
            return create_void();
    }
}

/* Evaluate statement */
static Value eval_statement(ASTNode *stmt, Environment *env) {
    if (!stmt) return create_void();


    switch (stmt->type) {
        case AST_LET: {
            Value value = eval_expression(stmt->as.let.value, env);
            env_define_var(env, stmt->as.let.name, stmt->as.let.var_type, stmt->as.let.is_mut, value);
            
            /* Trace variable declaration */
#ifdef TRACING_ENABLED
            const char *scope = (g_tracing_config.call_stack_size > 0) ?
                g_tracing_config.call_stack[g_tracing_config.call_stack_size - 1] : NULL;
            trace_var_decl(stmt->as.let.name, stmt->as.let.var_type, value, 
                          stmt->as.let.is_mut, stmt->line, stmt->column, scope);
#else
            trace_var_decl(stmt->as.let.name, stmt->as.let.var_type, value, 
                          stmt->as.let.is_mut, stmt->line, stmt->column, NULL);
#endif
            
            return create_void();
        }

        case AST_SET: {
            /* Get old value before setting */
            Symbol *sym = env_get_var(env, stmt->as.set.name);
            Value old_value = sym ? sym->value : create_void();
            
            Value value = eval_expression(stmt->as.set.value, env);
            env_set_var(env, stmt->as.set.name, value);
            
            /* Trace variable assignment */
#ifdef TRACING_ENABLED
            const char *scope = (g_tracing_config.call_stack_size > 0) ?
                g_tracing_config.call_stack[g_tracing_config.call_stack_size - 1] : NULL;
            trace_var_set(stmt->as.set.name, old_value, value, 
                         stmt->line, stmt->column, scope);
#else
            trace_var_set(stmt->as.set.name, old_value, value, 
                         stmt->line, stmt->column, NULL);
#endif
            
            return create_void();
        }

        case AST_WHILE: {
            Value result = create_void();
            while (is_truthy(eval_expression(stmt->as.while_stmt.condition, env))) {
                result = eval_statement(stmt->as.while_stmt.body, env);
                /* If body returned a value, propagate it immediately */
                if (result.is_return) {
                    return result;
                }
            }
            return result;
        }

        case AST_FOR: {
            /* Evaluate range */
            ASTNode *range_expr = stmt->as.for_stmt.range_expr;
            if (range_expr->type != AST_CALL || strcmp(range_expr->as.call.name, "range") != 0) {
                fprintf(stderr, "Error: for loop requires range expression\n");
                return create_void();
            }

            if (range_expr->as.call.arg_count != 2) {
                fprintf(stderr, "Error: range requires 2 arguments\n");
                return create_void();
            }

            Value start_val = eval_expression(range_expr->as.call.args[0], env);
            Value end_val = eval_expression(range_expr->as.call.args[1], env);

            if (start_val.type != VAL_INT || end_val.type != VAL_INT) {
                fprintf(stderr, "Error: range requires int arguments\n");
                return create_void();
            }

            long long start = start_val.as.int_val;
            long long end = end_val.as.int_val;

            /* Define loop variable before the loop */
            int loop_var_index = env->symbol_count;
            env_define_var(env, stmt->as.for_stmt.var_name, TYPE_INT, false, create_int(start));

            Value result = create_void();
            for (long long i = start; i < end; i++) {
                /* Update loop variable value */
                env->symbols[loop_var_index].value = create_int(i);

                /* Execute loop body */
                result = eval_statement(stmt->as.for_stmt.body, env);
                
                /* If body returned a value, propagate it immediately */
                if (result.is_return) {
                    env->symbol_count = loop_var_index;  /* Clean up before return */
                    return result;
                }
            }

            /* Remove loop variable from scope */
            env->symbol_count = loop_var_index;

            return result;
        }

        case AST_RETURN: {
            Value result;
            if (stmt->as.return_stmt.value) {
                result = eval_expression(stmt->as.return_stmt.value, env);
            } else {
                result = create_void();
            }
            result.is_return = true;  /* Mark as return value */
            return result;
        }

        case AST_BLOCK: {
            Value result = create_void();
            for (int i = 0; i < stmt->as.block.count; i++) {
                result = eval_statement(stmt->as.block.statements[i], env);
                /* If statement returned a value, propagate it immediately */
                if (result.is_return) {
                    return result;
                }
            }
            return result;
        }

        case AST_PRINT: {
            Value value = eval_expression(stmt->as.print.expr, env);
            print_value(value);
            printf("\n");
            return create_void();
        }

        case AST_ASSERT: {
            Value cond = eval_expression(stmt->as.assert.condition, env);
            if (!is_truthy(cond)) {
                fprintf(stderr, "Assertion failed at line %d, column %d\n", stmt->line, stmt->column);
                exit(1);
            }
            return create_void();
        }

        case AST_STRUCT_DEF:
        case AST_ENUM_DEF:
        case AST_FUNCTION:
        case AST_SHADOW:
            /* Struct, enum, function, and shadow definitions are handled at program level */
            /* Just return void if encountered during execution */
            return create_void();

        default:
            /* Expression statements */
            return eval_expression(stmt, env);
    }
}

/* Run shadow tests */
bool run_shadow_tests(ASTNode *program, Environment *env) {
    if (!program || program->type != AST_PROGRAM) {
        fprintf(stderr, "Error: Invalid program for shadow tests\n");
        return false;
    }

    printf("Running shadow tests...\n");

    bool all_passed = true;

    /* Run each shadow test */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        
        if (item->type == AST_SHADOW) {
            printf("Testing %s... ", item->as.shadow.function_name);

            
            /* Execute shadow test */
            eval_statement(item->as.shadow.body, env);

            printf("PASSED\n");
        } else {
            eval_statement(item, env);
        }
    }

    if (all_passed) {
        printf("All shadow tests passed!\n");
    }

    return all_passed;
}

/* Run the entire program (interpreter mode) */
bool run_program(ASTNode *program, Environment *env) {
    if (!program || program->type != AST_PROGRAM) {
        fprintf(stderr, "Error: Invalid program\n");
        return false;
    }

    /* Execute all top-level items (functions, statements, etc.) */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];

        /* Skip shadow tests in interpreter mode - they're for compiler validation */
        if (item->type == AST_SHADOW) {
            continue;
        }

        /* Execute the item */
        eval_statement(item, env);
    }

    return true;
}

/* Call a function by name with arguments */
Value call_function(const char *name, Value *args, int arg_count, Environment *env) {
    Function *func = env_get_function(env, name);
    if (!func) {
        fprintf(stderr, "Error: Function '%s' not found\n", name);
        return create_void();
    }

    /* Check argument count */
    if (arg_count != func->param_count) {
        fprintf(stderr, "Error: Function '%s' expects %d arguments, got %d\n",
                name, func->param_count, arg_count);
        return create_void();
    }

    /* Save original symbol count to restore environment after function call */
    int original_symbol_count = env->symbol_count;

    /* Add function parameters to environment with copies of string values */
    for (int i = 0; i < arg_count; i++) {
        Value param_value = args[i];

        /* Make a deep copy of string values to avoid memory corruption */
        if (param_value.type == VAL_STRING) {
            param_value = create_string(args[i].as.string_val);
        }

        env_define_var(env, func->params[i].name, func->params[i].type, false, param_value);
    }

    /* Execute the function body */
    Value result = eval_statement(func->body, env);

    /* Make a copy of the result if it's a string BEFORE cleaning up parameters */
    Value return_value = result;
    if (result.type == VAL_STRING) {
        return_value = create_string(result.as.string_val);
    }
    
    /* Clear is_return flag - we've exited the function */
    return_value.is_return = false;

    /* Clean up parameter strings and restore environment */
    for (int i = original_symbol_count; i < env->symbol_count; i++) {
        free(env->symbols[i].name);
        if (env->symbols[i].value.type == VAL_STRING) {
            free(env->symbols[i].value.as.string_val);
        }
    }
    env->symbol_count = original_symbol_count;

    return return_value;
}