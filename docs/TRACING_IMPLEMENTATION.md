# Tracing System for Debugging

## Overview
The nanolang interpreter includes a comprehensive tracing system designed specifically for LLM-assisted debugging. It provides detailed, structured runtime information about function calls, variable operations, and program flow in an LLM-readable format.

**Purpose**: Enable LLMs to understand program execution behavior, identify bugs, verify correctness, and debug issues without requiring manual debugging tools.

## Quick Start for LLMs

When debugging nanolang code, use tracing to understand runtime behavior:

```bash
# See everything that happens
./bin/nano program.nano --trace-all

# Focus on a specific function
./bin/nano program.nano --trace-function=problematic_function

# Track a variable that's behaving unexpectedly
./bin/nano program.nano --trace-var=suspicious_variable

# See everything inside a function's scope
./bin/nano program.nano --trace-scope=main
```

The output is structured and LLM-readable, showing:
- **What happened**: Function calls, variable changes, reads
- **When it happened**: Timestamps and line numbers
- **Where it happened**: Call stack and scope information
- **What values were involved**: Full value information with types

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

```text
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

## Debugging Workflows for LLMs

### Scenario 1: Function Not Working as Expected

**Problem**: A function returns wrong values or crashes.

**Solution**:
```bash
./bin/nano program.nano --trace-function=problematic_function
```

**What to look for**:

- Are arguments correct? Check `arguments:` section
- Is the function being called? Look for `FUNCTION_CALL` events
- What variables change inside? Look for `VAR_SET` events
- What's the return value? Check final variable values

### Scenario 2: Variable Has Unexpected Value

**Problem**: A variable contains wrong value at some point.

**Solution**:
```bash
./bin/nano program.nano --trace-var=suspicious_variable
```

**What to look for**:

- When was it declared? Check `VAR_DECL` event
- When was it modified? Check `VAR_SET` events (shows old → new)
- When was it read? Check `VAR_READ` events
- What was the sequence? Trace through timestamps

### Scenario 3: Complex Control Flow Issue

**Problem**: Logic error in nested function calls or loops.

**Solution**:
```bash
./bin/nano program.nano --trace-scope=main
```

**What to look for**:

- Call stack shows function nesting (`call_stack:` field)
- Variable changes show state transitions
- Function calls show control flow
- Scope information shows where variables are accessible

### Scenario 4: Finding All Uses of a Pattern

**Problem**: Need to find all functions/variables matching a pattern.

**Solution**:
```bash
./bin/nano program.nano --trace-regex='^test.*'
```

**What to look for**:

- All matching functions will show `FUNCTION_CALL` and `FUNCTION_DEF`
- All matching variables will show `VAR_DECL`, `VAR_SET`, `VAR_READ`

### Scenario 5: Complete Program Analysis

**Problem**: Need full understanding of program execution.

**Solution**:
```bash
./bin/nano program.nano --trace-all > trace_output.txt
```

**What to look for**:

- Complete execution timeline
- All function definitions and calls
- All variable operations
- Full call stack at each point
- Use this to verify program matches specification

## Interpreting Trace Output

### Trace Event Types

1. **FUNCTION_DEF**: Function definition discovered
   - Shows function signature (name, parameters, return type)
   - Appears when function is parsed/type-checked
   - Use to verify function signatures match expectations

2. **FUNCTION_CALL**: Function invocation
   - Shows function name, arguments (with values), call stack
   - Appears when function executes
   - Use to verify correct arguments passed, correct call order

3. **VAR_DECL**: Variable declaration
   - Shows variable name, type, initial value, mutability, scope
   - Appears when `let` statement executes
   - Use to verify variables initialized correctly

4. **VAR_SET**: Variable assignment
   - Shows variable name, old value, new value, scope
   - Appears when `set` statement executes
   - Use to track value changes, find where bugs introduced

5. **VAR_READ**: Variable read
   - Shows variable name, current value, scope
   - Appears when variable used in expression
   - Use to verify correct values read at each point

### Reading Trace Output

Each trace event follows this format:

```text
[TRACE] <timestamp> <EVENT_TYPE> <name>:<line>:<column>
  <field>: <value>
  <field>: <value>
  ...
```

**Key Fields**:

- `timestamp`: When event occurred (ISO 8601 format)
- `line:column`: Source location
- `call_stack`: Function call hierarchy (array of function names)
- `scope`: Current function scope (or "(global)")
- `value`: Actual runtime value (formatted for type)
- `type`: Type information (for variables and parameters)

### Example: Debugging a Bug

**Code**:
```nano
fn calculate_sum(numbers: array<int>) -> int {
    let mut sum: int = 0
    let mut i: int = 0
    while (< i (array_length numbers)) {
        let value: int = (at numbers i)
        set sum (+ sum value)
        set i (+ i 1)
    }
    return sum
}
```

**Trace Output** (with `--trace-scope=calculate_sum`):

```text
[TRACE] 2025-01-15T10:30:45 FUNCTION_CALL calculate_sum:19:13
  function: "calculate_sum"
  call_stack: ["main"]
  arguments:
    - name: "numbers", type: "array<int>", value: [1, 2, 3]

[TRACE] 2025-01-15T10:30:45 VAR_DECL sum:20:5
  variable: "sum"
  type: "int"
  mutable: true
  initial_value: 0
  scope: "calculate_sum"

[TRACE] 2025-01-15T10:30:45 VAR_DECL i:21:5
  variable: "i"
  type: "int"
  mutable: true
  initial_value: 0
  scope: "calculate_sum"

[TRACE] 2025-01-15T10:30:45 VAR_READ i:22:9
  variable: "i"
  value: 0
  scope: "calculate_sum"

[TRACE] 2025-01-15T10:30:45 VAR_SET sum:24:9
  variable: "sum"
  old_value: 0
  new_value: 1
  scope: "calculate_sum"

[TRACE] 2025-01-15T10:30:45 VAR_SET i:25:9
  variable: "i"
  old_value: 0
  new_value: 1
  scope: "calculate_sum"
...
```

**LLM Analysis**:

- Function called with `[1, 2, 3]` ✓
- `sum` initialized to 0 ✓
- `i` initialized to 0 ✓
- Loop reads `i` correctly ✓
- `sum` updated: 0 → 1 ✓
- `i` incremented: 0 → 1 ✓
- Pattern continues correctly → **No bug found in this trace**

## Best Practices for LLM Debugging

1. **Start Narrow**: Use `--trace-function=` or `--trace-var=` first to focus on specific issues
2. **Expand as Needed**: Use `--trace-scope=` to see context around a function
3. **Full Trace Last**: Use `--trace-all` only when you need complete picture
4. **Save Output**: Redirect to file for analysis: `--trace-all > trace.txt`
5. **Compare Traces**: Run with different inputs and compare outputs
6. **Check Call Stack**: Use `call_stack` field to understand function nesting
7. **Track State Changes**: Follow `VAR_SET` events to see how state evolves
8. **Verify Specifications**: Compare trace output to expected behavior from spec

## Technical Notes

- Tracing is only available in the interpreter (`bin/nano`), not the compiler (`bin/nanoc`)
- No performance overhead in compiled code (tracing code is conditionally compiled out)
- Tracing hooks are integrated throughout the interpreter evaluation engine
- Call stack tracking enables scope-based tracing
- Output goes to stdout (stderr for errors/warnings)
- Trace output is deterministic (same program produces same trace order)
