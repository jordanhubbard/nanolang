#ifndef TRACING_H
#define TRACING_H

#include "nanolang.h"
#include <stdbool.h>
#include <time.h>

/* Only enable tracing in interpreter builds */
#ifdef NANO_INTERPRETER
#define TRACING_ENABLED
#endif

/* Trace event types */
typedef enum {
    TRACE_FUNCTION_CALL,
    TRACE_FUNCTION_DEF,
    TRACE_VAR_DECL,
    TRACE_VAR_SET,
    TRACE_VAR_READ
} TraceEventType;

/* Tracing configuration */
typedef struct {
    bool enabled;
    bool trace_functions;
    bool trace_variables;
    bool trace_function_definitions;
    
    /* Filters */
    char **function_filters;      /* Function names to trace (NULL-terminated) */
    int function_filter_count;
    char **variable_filters;      /* Variable names to trace (NULL-terminated) */
    int variable_filter_count;
    char **regex_patterns;        /* Regex patterns to match (NULL-terminated) */
    int regex_pattern_count;
    char **scope_functions;       /* Functions to trace inside of (NULL-terminated) */
    int scope_function_count;
    
    /* Scope tracking */
    int call_stack_depth;
    bool in_traced_scope;
    char **call_stack;            /* Current call stack (function names) */
    int call_stack_size;
    int call_stack_capacity;
} TracingConfig;

#ifdef TRACING_ENABLED
/* Global tracing configuration */
extern TracingConfig g_tracing_config;

/* Initialize tracing system */
void tracing_init(void);

/* Cleanup tracing system */
void tracing_cleanup(void);

/* Configure tracing from command-line arguments */
void tracing_configure(int argc, char **argv);

/* Check if we should trace an event */
bool should_trace_function(const char *function_name);
bool should_trace_variable(const char *variable_name);
bool should_trace_in_current_scope(void);

/* Push/pop call stack */
void tracing_push_call(const char *function_name);
void tracing_pop_call(void);

/* Trace events */
void trace_function_call(const char *function_name, Value *args, int arg_count, 
                         const char *param_names[], int line, int column);
void trace_function_def(const char *function_name, Parameter *params, int param_count,
                        Type return_type, int line, int column);
void trace_var_decl(const char *var_name, Type type, Value initial_value, 
                    bool is_mut, int line, int column, const char *scope);
void trace_var_set(const char *var_name, Value old_value, Value new_value,
                   int line, int column, const char *scope);
void trace_var_read(const char *var_name, Value value, int line, int column, const char *scope);

/* Helper: Format value for tracing */
const char *format_value_for_trace(Value val);

/* Helper: Get current timestamp */
void get_timestamp(char *buffer, size_t size);

#else
/* Stub implementations for compiler builds */
extern TracingConfig g_tracing_config;

#define tracing_init() ((void)0)
#define tracing_cleanup() ((void)0)
#define tracing_configure(...) ((void)0)
#define should_trace_function(...) (false)
#define should_trace_variable(...) (false)
#define should_trace_in_current_scope() (false)
#define tracing_push_call(...) ((void)0)
#define tracing_pop_call() ((void)0)
#define trace_function_call(...) ((void)0)
#define trace_function_def(...) ((void)0)
#define trace_var_decl(...) ((void)0)
#define trace_var_set(...) ((void)0)
#define trace_var_read(...) ((void)0)
#define format_value_for_trace(...) ("")
#define get_timestamp(...) ((void)0)
#endif

#endif /* TRACING_H */
