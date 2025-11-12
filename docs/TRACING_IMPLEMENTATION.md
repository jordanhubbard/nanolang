# Tracing System Implementation

## Overview
A comprehensive tracing system has been implemented for the nanolang interpreter that provides LLM-readable runtime behavior information.

## Features Implemented

### 1. Function Tracing
- **Function Calls**: Traces when functions are called with their arguments (values and types)
- **Function Definitions**: Traces when functions are defined with their signatures

### 2. Variable Tracing
- **Variable Declarations**: Traces when variables are declared with type and initial value
- **Variable Assignments**: Traces when variables are set with old and new values
- **Variable Reads**: Traces when variables are read

### 3. Filtering/Scoping
- **Global tracing**: `--trace` or `--trace-all` traces everything
- **Function-specific**: `--trace-function=<name>` traces specific functions
- **Variable-specific**: `--trace-var=<name>` traces specific variables
- **Regex matching**: `--trace-regex=<pattern>` traces anything matching a regex
- **Scope-based**: `--trace-scope=<func>` traces only events inside specific functions

### 4. Output Format
- LLM-readable structured format
- Includes timestamps, line numbers, call stack
- Shows variable values, types, and scope information

## Architecture

### Files Created
- `src/tracing.h` - Tracing system header with conditional compilation
- `src/tracing.c` - Tracing system implementation (interpreter-only)

### Files Modified
- `src/eval.c` - Added tracing hooks for function calls and variable operations
- `src/typechecker.c` - Added tracing hooks for function definitions
- `src/interpreter_main.c` - Added tracing initialization and command-line parsing
- `Makefile` - Added tracing.c to interpreter build, conditional compilation

### Conditional Compilation
- Tracing is only enabled in interpreter builds (`-DNANO_INTERPRETER`)
- Compiler builds use stub macros (no overhead)
- Tracing functions are no-ops when disabled

## Usage Examples

```bash
# Trace everything
./bin/nano program.nano --trace-all

# Trace specific function
./bin/nano program.nano --trace-function=calculate_sum

# Trace specific variable
./bin/nano program.nano --trace-var=result

# Trace everything inside a function
./bin/nano program.nano --trace-scope=main

# Trace using regex
./bin/nano program.nano --trace-regex='^test.*'
```

## Output Format Example

```
[TRACE] 2025-11-12T10:30:45 FUNCTION_DEF add:2:1
  function: "add"
  return_type: "int"
  parameters:
    - name: "a", type: "int"
    - name: "b", type: "int"

[TRACE] 2025-11-12T10:30:45 FUNCTION_CALL add:37:13
  function: "add"
  call_stack: ["main"]
  arguments:
    - name: "a", type: "int", value: 10
    - name: "b", type: "int", value: 20

[TRACE] 2025-11-12T10:30:45 VAR_DECL x:35:5
  variable: "x"
  type: "int"
  mutable: false
  initial_value: 10
  scope: "main"

[TRACE] 2025-11-12T10:30:45 VAR_SET sum:24:9
  variable: "sum"
  old_value: 0
  new_value: 1
  scope: "calculate_sum"
```

## Status

✅ **Completed**: All core tracing functionality implemented
✅ **Completed**: Conditional compilation for interpreter-only builds
✅ **Completed**: Command-line argument parsing
✅ **Completed**: Filtering and scoping support
✅ **Completed**: LLM-readable output format

## Notes

- Tracing is only available in the interpreter (`bin/nano`), not the compiler (`bin/nanoc`)
- No performance overhead in compiled code
- Tracing hooks are integrated throughout the interpreter evaluation engine
- Call stack tracking enables scope-based tracing

