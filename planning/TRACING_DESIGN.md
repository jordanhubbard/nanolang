# Tracing System Design

## Overview
A comprehensive tracing system for nanolang interpreter that provides LLM-readable runtime behavior information.

## Requirements

### What to Trace
1. **Function Calls**: Function name, arguments (with values and types)
2. **Function Definitions**: Function name, signature (parameters and return type)
3. **Variable Declarations**: Variable name, type, initial value
4. **Variable Assignments**: Variable name, old value, new value
5. **Variable Reads**: Variable name, value read

### Filtering/Scoping
- **Global tracing**: Trace everything
- **Function-specific**: Trace only calls/definitions of specific functions
- **Variable-specific**: Trace only specific variables
- **Regex matching**: Trace anything matching a regex pattern
- **Scope-based**: Trace only events inside specific functions (and their callees)

### Output Format
- LLM-readable (structured, clear, complete)
- JSON-like or structured text format
- Include context: line numbers, call stack, timestamps

## Architecture

### Tracing Configuration
```c
typedef struct {
    bool enabled;
    bool trace_functions;
    bool trace_variables;
    bool trace_function_definitions;
    
    // Filters
    char **function_filters;      // Function names to trace
    char **variable_filters;      // Variable names to trace
    char **regex_patterns;        // Regex patterns to match
    char **scope_functions;       // Functions to trace inside of
    
    // Scope tracking
    int call_stack_depth;
    bool in_traced_scope;
} TracingConfig;
```

### Trace Event Types
```c
typedef enum {
    TRACE_FUNCTION_CALL,
    TRACE_FUNCTION_DEF,
    TRACE_VAR_DECL,
    TRACE_VAR_SET,
    TRACE_VAR_READ
} TraceEventType;
```

### Trace Output Format
```
[TRACE] <timestamp> <event_type> <location>
  <details>
```

Example:
```
[TRACE] 2025-11-12T10:30:45.123 FUNCTION_CALL main:151
  function: "calculate_sum"
  arguments:
    - name: "a", type: "int", value: 10
    - name: "b", type: "int", value: 20
  call_stack: ["main"]
  
[TRACE] 2025-11-12T10:30:45.124 VAR_DECL main:152
  variable: "result"
  type: "int"
  initial_value: 0
  scope: "main"
  
[TRACE] 2025-11-12T10:30:45.125 VAR_SET main:153
  variable: "result"
  old_value: 0
  new_value: 30
  scope: "main"
```

## Implementation Plan

1. Create `src/tracing.h` and `src/tracing.c` for tracing infrastructure
2. Add tracing hooks to `src/eval.c` (interpreter)
3. Add tracing hooks to `src/typechecker.c` (for function definitions)
4. Add command-line flags to enable tracing
5. Implement filtering logic
6. Implement scope tracking
7. Format output for LLM readability

