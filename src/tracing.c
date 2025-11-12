#include "tracing.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <regex.h>

#ifdef TRACING_ENABLED

/* Global tracing configuration */
TracingConfig g_tracing_config = {
    .enabled = false,
    .trace_functions = false,
    .trace_variables = false,
    .trace_function_definitions = false,
    .function_filters = NULL,
    .function_filter_count = 0,
    .variable_filters = NULL,
    .variable_filter_count = 0,
    .regex_patterns = NULL,
    .regex_pattern_count = 0,
    .scope_functions = NULL,
    .scope_function_count = 0,
    .call_stack_depth = 0,
    .in_traced_scope = false,
    .call_stack = NULL,
    .call_stack_size = 0,
    .call_stack_capacity = 0
};

/* Initialize tracing system */
void tracing_init(void) {
    g_tracing_config.call_stack_capacity = 32;
    g_tracing_config.call_stack = malloc(sizeof(char*) * g_tracing_config.call_stack_capacity);
    g_tracing_config.call_stack_size = 0;
}

/* Cleanup tracing system */
void tracing_cleanup(void) {
    /* Free filters */
    if (g_tracing_config.function_filters) {
        for (int i = 0; i < g_tracing_config.function_filter_count; i++) {
            free(g_tracing_config.function_filters[i]);
        }
        free(g_tracing_config.function_filters);
    }
    if (g_tracing_config.variable_filters) {
        for (int i = 0; i < g_tracing_config.variable_filter_count; i++) {
            free(g_tracing_config.variable_filters[i]);
        }
        free(g_tracing_config.variable_filters);
    }
    if (g_tracing_config.regex_patterns) {
        for (int i = 0; i < g_tracing_config.regex_pattern_count; i++) {
            free(g_tracing_config.regex_patterns[i]);
        }
        free(g_tracing_config.regex_patterns);
    }
    if (g_tracing_config.scope_functions) {
        for (int i = 0; i < g_tracing_config.scope_function_count; i++) {
            free(g_tracing_config.scope_functions[i]);
        }
        free(g_tracing_config.scope_functions);
    }
    
    /* Free call stack */
    if (g_tracing_config.call_stack) {
        for (int i = 0; i < g_tracing_config.call_stack_size; i++) {
            free(g_tracing_config.call_stack[i]);
        }
        free(g_tracing_config.call_stack);
    }
}

/* Check if string matches any regex pattern */
static bool matches_regex(const char *str, char **patterns, int count) {
    if (!patterns || count == 0) return false;
    
    for (int i = 0; i < count; i++) {
        regex_t regex;
        if (regcomp(&regex, patterns[i], REG_EXTENDED | REG_NOSUB) == 0) {
            int result = regexec(&regex, str, 0, NULL, 0);
            regfree(&regex);
            if (result == 0) {
                return true;
            }
        }
    }
    return false;
}

/* Check if string is in filter list */
static bool in_filter_list(const char *str, char **filters, int count) {
    if (!filters || count == 0) return false;
    for (int i = 0; i < count; i++) {
        if (filters[i] && strcmp(filters[i], str) == 0) {
            return true;
        }
    }
    return false;
}

/* Check if we should trace a function */
bool should_trace_function(const char *function_name) {
    if (!g_tracing_config.enabled || !g_tracing_config.trace_functions) {
        return false;
    }
    
    /* If no filters, trace all */
    if (g_tracing_config.function_filter_count == 0 && 
        g_tracing_config.regex_pattern_count == 0 &&
        g_tracing_config.scope_function_count == 0) {
        return true;
    }
    
    /* Check function name filters */
    if (in_filter_list(function_name, g_tracing_config.function_filters, 
                       g_tracing_config.function_filter_count)) {
        return true;
    }
    
    /* Check regex patterns */
    if (matches_regex(function_name, g_tracing_config.regex_patterns,
                      g_tracing_config.regex_pattern_count)) {
        return true;
    }
    
    /* Check if we're in a traced scope */
    if (g_tracing_config.scope_function_count > 0 && g_tracing_config.in_traced_scope) {
        return true;
    }
    
    return false;
}

/* Check if we should trace a variable */
bool should_trace_variable(const char *variable_name) {
    if (!g_tracing_config.enabled || !g_tracing_config.trace_variables) {
        return false;
    }
    
    /* If no filters, trace all */
    if (g_tracing_config.variable_filter_count == 0 && 
        g_tracing_config.regex_pattern_count == 0 &&
        g_tracing_config.scope_function_count == 0) {
        return true;
    }
    
    /* Check variable name filters */
    if (in_filter_list(variable_name, g_tracing_config.variable_filters,
                       g_tracing_config.variable_filter_count)) {
        return true;
    }
    
    /* Check regex patterns */
    if (matches_regex(variable_name, g_tracing_config.regex_patterns,
                      g_tracing_config.regex_pattern_count)) {
        return true;
    }
    
    /* Check if we're in a traced scope */
    if (g_tracing_config.scope_function_count > 0 && g_tracing_config.in_traced_scope) {
        return true;
    }
    
    return false;
}

/* Check if we should trace in current scope */
bool should_trace_in_current_scope(void) {
    if (!g_tracing_config.enabled) return false;
    
    /* If no scope filters, trace everywhere */
    if (g_tracing_config.scope_function_count == 0) {
        return true;
    }
    
    return g_tracing_config.in_traced_scope;
}

/* Push function call onto stack */
void tracing_push_call(const char *function_name) {
    if (!g_tracing_config.call_stack) return;
    
    /* Resize if needed */
    if (g_tracing_config.call_stack_size >= g_tracing_config.call_stack_capacity) {
        g_tracing_config.call_stack_capacity *= 2;
        g_tracing_config.call_stack = realloc(g_tracing_config.call_stack,
            sizeof(char*) * g_tracing_config.call_stack_capacity);
    }
    
    g_tracing_config.call_stack[g_tracing_config.call_stack_size++] = 
        function_name ? strdup(function_name) : strdup("(unknown)");
    g_tracing_config.call_stack_depth++;
    
    /* Check if we're entering a traced scope */
    if (g_tracing_config.scope_function_count > 0 && !g_tracing_config.in_traced_scope) {
        if (in_filter_list(function_name, g_tracing_config.scope_functions,
                           g_tracing_config.scope_function_count)) {
            g_tracing_config.in_traced_scope = true;
        }
    }
}

/* Pop function call from stack */
void tracing_pop_call(void) {
    if (!g_tracing_config.call_stack || g_tracing_config.call_stack_size == 0) {
        return;
    }
    
    free(g_tracing_config.call_stack[--g_tracing_config.call_stack_size]);
    g_tracing_config.call_stack_depth--;
    
    /* Check if we're leaving a traced scope */
    if (g_tracing_config.scope_function_count > 0 && g_tracing_config.in_traced_scope) {
        if (g_tracing_config.call_stack_size == 0) {
            g_tracing_config.in_traced_scope = false;
        } else {
            /* Check if current function is still in scope list */
            const char *current = g_tracing_config.call_stack[g_tracing_config.call_stack_size - 1];
            if (!in_filter_list(current, g_tracing_config.scope_functions,
                                g_tracing_config.scope_function_count)) {
                g_tracing_config.in_traced_scope = false;
            }
        }
    }
}

/* Get current timestamp */
void get_timestamp(char *buffer, size_t size) {
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    strftime(buffer, size, "%Y-%m-%dT%H:%M:%S", tm_info);
}

/* Format value for tracing */
const char *format_value_for_trace(Value val) {
    static char buffer[256];
    switch (val.type) {
        case VAL_INT:
            snprintf(buffer, sizeof(buffer), "%lld", val.as.int_val);
            break;
        case VAL_FLOAT:
            snprintf(buffer, sizeof(buffer), "%g", val.as.float_val);
            break;
        case VAL_BOOL:
            snprintf(buffer, sizeof(buffer), "%s", val.as.bool_val ? "true" : "false");
            break;
        case VAL_STRING:
            snprintf(buffer, sizeof(buffer), "\"%s\"", val.as.string_val ? val.as.string_val : "(null)");
            break;
        case VAL_VOID:
            snprintf(buffer, sizeof(buffer), "void");
            break;
        default:
            snprintf(buffer, sizeof(buffer), "(unknown)");
            break;
    }
    return buffer;
}

/* Format type for tracing */
static const char *format_type_for_trace(Type type) {
    switch (type) {
        case TYPE_INT: return "int";
        case TYPE_FLOAT: return "float";
        case TYPE_BOOL: return "bool";
        case TYPE_STRING: return "string";
        case TYPE_VOID: return "void";
        case TYPE_ARRAY: return "array";
        case TYPE_STRUCT: return "struct";
        case TYPE_ENUM: return "enum";
        case TYPE_LIST_INT: return "list_int";
        case TYPE_LIST_STRING: return "list_string";
        default: return "unknown";
    }
}

/* Format call stack for output */
static void format_call_stack(char *buffer, size_t size) {
    buffer[0] = '[';
    size_t pos = 1;
    
    for (int i = 0; i < g_tracing_config.call_stack_size && pos < size - 1; i++) {
        if (i > 0) {
            buffer[pos++] = ',';
            buffer[pos++] = ' ';
        }
        const char *func = g_tracing_config.call_stack[i];
        size_t len = strlen(func);
        if (pos + len + 2 < size) {
            buffer[pos++] = '"';
            memcpy(buffer + pos, func, len);
            pos += len;
            buffer[pos++] = '"';
        }
    }
    
    if (pos < size - 1) {
        buffer[pos++] = ']';
        buffer[pos] = '\0';
    } else {
        buffer[size - 1] = '\0';
    }
}

/* Trace function call */
void trace_function_call(const char *function_name, Value *args, int arg_count,
                         const char *param_names[], int line, int column) {
    if (!should_trace_function(function_name)) return;
    
    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    
    char call_stack_buf[512];
    format_call_stack(call_stack_buf, sizeof(call_stack_buf));
    
    printf("[TRACE] %s FUNCTION_CALL %s:%d:%d\n", timestamp, function_name, line, column);
    printf("  function: \"%s\"\n", function_name);
    printf("  call_stack: %s\n", call_stack_buf);
    printf("  arguments:\n");
    
    for (int i = 0; i < arg_count; i++) {
        const char *param_name = (param_names && i < arg_count) ? param_names[i] : "arg";
        printf("    - name: \"%s\", type: \"%s\", value: %s\n",
               param_name, format_type_for_trace(args[i].type), format_value_for_trace(args[i]));
    }
    printf("\n");
}

/* Trace function definition */
void trace_function_def(const char *function_name, Parameter *params, int param_count,
                        Type return_type, int line, int column) {
    if (!g_tracing_config.enabled || !g_tracing_config.trace_function_definitions) {
        return;
    }
    
    /* Check filters */
    if (g_tracing_config.function_filter_count > 0 || g_tracing_config.regex_pattern_count > 0) {
        if (!should_trace_function(function_name)) {
            return;
        }
    }
    
    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    
    printf("[TRACE] %s FUNCTION_DEF %s:%d:%d\n", timestamp, function_name, line, column);
    printf("  function: \"%s\"\n", function_name);
    printf("  return_type: \"%s\"\n", format_type_for_trace(return_type));
    printf("  parameters:\n");
    
    for (int i = 0; i < param_count; i++) {
        printf("    - name: \"%s\", type: \"%s\"\n",
               params[i].name ? params[i].name : "unnamed",
               format_type_for_trace(params[i].type));
    }
    printf("\n");
}

/* Trace variable declaration */
void trace_var_decl(const char *var_name, Type type, Value initial_value,
                    bool is_mut, int line, int column, const char *scope) {
    if (!should_trace_variable(var_name)) return;
    
    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    
    printf("[TRACE] %s VAR_DECL %s:%d:%d\n", timestamp, var_name, line, column);
    printf("  variable: \"%s\"\n", var_name);
    printf("  type: \"%s\"\n", format_type_for_trace(type));
    printf("  mutable: %s\n", is_mut ? "true" : "false");
    printf("  initial_value: %s\n", format_value_for_trace(initial_value));
    printf("  scope: \"%s\"\n", scope ? scope : "(global)");
    printf("\n");
}

/* Trace variable assignment */
void trace_var_set(const char *var_name, Value old_value, Value new_value,
                   int line, int column, const char *scope) {
    if (!should_trace_variable(var_name)) return;
    
    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    
    printf("[TRACE] %s VAR_SET %s:%d:%d\n", timestamp, var_name, line, column);
    printf("  variable: \"%s\"\n", var_name);
    printf("  old_value: %s\n", format_value_for_trace(old_value));
    printf("  new_value: %s\n", format_value_for_trace(new_value));
    printf("  scope: \"%s\"\n", scope ? scope : "(global)");
    printf("\n");
}

/* Trace variable read */
void trace_var_read(const char *var_name, Value value, int line, int column, const char *scope) {
    if (!should_trace_variable(var_name)) return;
    
    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    
    printf("[TRACE] %s VAR_READ %s:%d:%d\n", timestamp, var_name, line, column);
    printf("  variable: \"%s\"\n", var_name);
    printf("  value: %s\n", format_value_for_trace(value));
    printf("  scope: \"%s\"\n", scope ? scope : "(global)");
    printf("\n");
}

/* Configure tracing from command-line arguments */
void tracing_configure(int argc, char **argv) {
    /* This will be called from main to parse --trace flags */
    /* For now, simple implementation - can be extended */
    g_tracing_config.enabled = false;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--trace") == 0 || strcmp(argv[i], "--trace-all") == 0) {
            g_tracing_config.enabled = true;
            g_tracing_config.trace_functions = true;
            g_tracing_config.trace_variables = true;
            g_tracing_config.trace_function_definitions = true;
        } else if (strncmp(argv[i], "--trace-function=", 17) == 0) {
            const char *func_name = argv[i] + 17;
            g_tracing_config.function_filter_count++;
            g_tracing_config.function_filters = realloc(g_tracing_config.function_filters,
                sizeof(char*) * g_tracing_config.function_filter_count);
            g_tracing_config.function_filters[g_tracing_config.function_filter_count - 1] = strdup(func_name);
            g_tracing_config.enabled = true;
            g_tracing_config.trace_functions = true;
        } else if (strncmp(argv[i], "--trace-var=", 12) == 0) {
            const char *var_name = argv[i] + 12;
            g_tracing_config.variable_filter_count++;
            g_tracing_config.variable_filters = realloc(g_tracing_config.variable_filters,
                sizeof(char*) * g_tracing_config.variable_filter_count);
            g_tracing_config.variable_filters[g_tracing_config.variable_filter_count - 1] = strdup(var_name);
            g_tracing_config.enabled = true;
            g_tracing_config.trace_variables = true;
        } else if (strncmp(argv[i], "--trace-scope=", 14) == 0) {
            const char *func_name = argv[i] + 14;
            g_tracing_config.scope_function_count++;
            g_tracing_config.scope_functions = realloc(g_tracing_config.scope_functions,
                sizeof(char*) * g_tracing_config.scope_function_count);
            g_tracing_config.scope_functions[g_tracing_config.scope_function_count - 1] = strdup(func_name);
            g_tracing_config.enabled = true;
            g_tracing_config.trace_functions = true;
            g_tracing_config.trace_variables = true;
        } else if (strncmp(argv[i], "--trace-regex=", 14) == 0) {
            const char *pattern = argv[i] + 14;
            g_tracing_config.regex_pattern_count++;
            g_tracing_config.regex_patterns = realloc(g_tracing_config.regex_patterns,
                sizeof(char*) * g_tracing_config.regex_pattern_count);
            g_tracing_config.regex_patterns[g_tracing_config.regex_pattern_count - 1] = strdup(pattern);
            g_tracing_config.enabled = true;
            g_tracing_config.trace_functions = true;
            g_tracing_config.trace_variables = true;
        }
    }
}

#else
/* Stub implementation for compiler builds */
TracingConfig g_tracing_config = {0};
#endif /* TRACING_ENABLED */
