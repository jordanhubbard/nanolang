# NanoLang Standard Library

> **For LLM Agents:** This directory contains reusable NanoLang libraries. Import them with `from "stdlib/MODULE.nano" import ...`

---

## Available Libraries

### 🔍 **stdlib/log.nano** - Structured Logging
Enterprise-grade logging with hierarchical levels and categories.

**When to use:**
- Debugging runtime behavior
- Tracking execution flow
- Production logging
- LLM agents generating self-debugging code

**Quick start:**
```nano
from "stdlib/log.nano" import log_info, log_error, log_warn

fn main() -> int {
    (log_info "app" "Application started")
    (log_error "database" "Connection failed")
    return 0
}
```

**Features:**
- 6 log levels: TRACE, DEBUG, INFO, WARN, ERROR, FATAL
- Category-based organization
- Threshold filtering
- Convenience functions (log without category)

**Full documentation:** `docs/DEBUGGING_GUIDE.md`  
**Examples:** `examples/logging_levels_demo.nano`, `examples/logging_categories_demo.nano`

---

### 📊 **stdlib/coverage.nano** - Runtime Instrumentation
Coverage tracking, performance timing, and execution tracing.

**When to use:**
- Collecting code coverage data
- Performance profiling
- Execution trace analysis
- LLM agents validating generated code

**Quick start:**
```nano
from "stdlib/coverage.nano" import coverage_init, coverage_record, coverage_report

fn my_function(x: int) -> int {
    (coverage_record "my_file.nano" 10 5)
    return (* x 2)
}

fn main() -> int {
    (coverage_init)
    let result: int = (my_function 5)
    (coverage_report)
    return 0
}
```

**Features:**
- Coverage tracking: `coverage_record()`, `coverage_report()`
- Performance timing: `timing_start()`, `timing_end()`
- Execution tracing: `trace_record()`, `trace_report()`

**Data structures:**
- `CoveragePoint`: file, line, column, hit_count
- `TimingPoint`: label, start_time_ms, total_time_ms, call_count
- `TraceEvent`: timestamp_ms, event_type, location, details

**Full documentation:** `docs/SELF_VALIDATING_CODE_GENERATION.md` (see instrumentation section)  
**Examples:** `examples/coverage_demo.nano`

---

### 🔤 **stdlib/regex.nano** - Regular Expressions
Pattern matching and text processing (wrapper for C regex).

**When to use:**
- Text parsing
- Input validation
- Pattern extraction

**Quick start:**
```nano
from "stdlib/regex.nano" import regex_match

fn main() -> int {
    if (regex_match "hello.*world" "hello beautiful world") {
        (println "Match!")
    }
    return 0
}
```

---

### 🧩 **stdlib/ast.nano** - Abstract Syntax Tree
AST manipulation for compiler development.

**When to use:**
- Building compilers
- Code generation
- AST transformations

**Documentation:** See compiler source code

---

### 📝 **stdlib/StringBuilder.nano** - Efficient String Building
Optimized string concatenation for large outputs.

**When to use:**
- Building large strings incrementally
- Avoiding O(n²) concatenation
- Template generation

**Quick start:**
```nano
from "stdlib/StringBuilder.nano" import StringBuilder, sb_new, sb_append, sb_to_string

fn main() -> int {
    let mut sb: StringBuilder = (sb_new)
    set sb (sb_append sb "Hello")
    set sb (sb_append sb " ")
    set sb (sb_append sb "World")
    let result: string = (sb_to_string sb)
    (println result)  # "Hello World"
    return 0
}
```

---

### 🔧 **stdlib/tidy_lexer.nano** - Lexer Utilities
Tokenization helpers for parsers.

**When to use:**
- Building parsers
- Lexical analysis

---

### 🔬 **stdlib/lalr.nano** - LALR Parser Generator
LALR(1) parser construction.

**When to use:**
- Building parsers for complex grammars
- Compiler front-ends

---

## Library Selection Guide for LLM Agents

### When Debugging Generated Code:
1. **First**: Use `stdlib/log.nano` for runtime visibility
2. **Second**: Add shadow tests (mandatory)
3. **Third**: Use `stdlib/coverage.nano` to find untested paths

### When Validating Code Generation:
1. **Coverage**: Use `stdlib/coverage.nano` to track execution
2. **Tracing**: Use `trace_record()` to log function calls
3. **Timing**: Use `timing_start()`/`timing_end()` for performance

### When Building Tools:
- **Text processing**: `stdlib/regex.nano`
- **Large string output**: `stdlib/StringBuilder.nano`
- **Compilers**: `stdlib/ast.nano`, `stdlib/lalr.nano`, `stdlib/tidy_lexer.nano`

---

## Import Syntax

**From stdlib directory:**
```nano
from "stdlib/log.nano" import log_info, log_error
from "stdlib/coverage.nano" import coverage_init, coverage_report
```

**Module-qualified names (after import):**
```nano
from "stdlib/log.nano" import log_info

fn main() -> int {
    (log_info "app" "Started")  # Use directly
    return 0
}
```

---

## Built-in Functions vs. stdlib

**Built-in functions** (always available, no import needed):
- `println`, `print`, `assert`
- Math: `abs`, `min`, `max`, `sqrt`, `pow`, `floor`, `ceil`
- Strings: `str_length`, `str_substring`, `str_equals`
- Arrays: `array_new`, `array_get`, `array_set`, `array_length`
- Conversions: `int_to_string`, `float_to_string`, `parse_int`, `parse_float`

**See full list:** `docs/STDLIB.md`

**stdlib libraries** (require explicit import):
- Logging, coverage, regex, StringBuilder, AST tools

---

## Contributing New Libraries

When adding stdlib libraries:
1. Create `stdlib/YOUR_LIBRARY.nano`
2. Add shadow tests for all functions
3. Update this README with usage guide
4. Add examples to `examples/`
5. Update `AGENTS.md` if relevant for LLM agents

---

## Related Documentation

- **Built-in functions reference:** `docs/STDLIB.md`
- **Debugging guide:** `docs/DEBUGGING_GUIDE.md`
- **Property testing:** `docs/PROPERTY_TESTING_GUIDE.md`
- **Self-validating code generation:** `docs/SELF_VALIDATING_CODE_GENERATION.md`
- **LLM agent training:** `AGENTS.md`

---

## Quick Reference Card

| Need | Use | Import |
|------|-----|--------|
| **Logging** | `stdlib/log.nano` | `log_info`, `log_error`, `log_warn` |
| **Coverage** | `stdlib/coverage.nano` | `coverage_init`, `coverage_report` |
| **Regex** | `stdlib/regex.nano` | `regex_match` |
| **String building** | `stdlib/StringBuilder.nano` | `sb_new`, `sb_append` |
| **AST tools** | `stdlib/ast.nano` | Various AST functions |
| **Parsing** | `stdlib/lalr.nano` | LALR parser functions |

---

**Last updated:** 2026-01-10 (nanolang-feedback epic completion)

### beads.nano
**Programmatic Issue Tracking**

Provides type-safe API for the Beads issue tracker. Create, query, and manage issues directly from NanoLang code.

**Killer Feature**: `assert_with_bead()` - Assertions that automatically create bugs when they fail!

```nano
from "stdlib/beads.nano" import assert_with_bead, bd_stats

# Assertion that creates a P0 bug if it fails
(assert_with_bead
    (!= divisor 0)
    "Division by zero detected"
    0
    "Attempted to divide by zero"
)

# Get project statistics
let stats: BeadStats = (bd_stats)
```

**See**: `stdlib/README_BEADS.md` for complete documentation

**Examples**:
- `examples/advanced/beads_basic_usage.nano` - Query and create issues
- `examples/advanced/beads_assert_with_bead.nano` - Automatic issue creation
- `examples/advanced/beads_workflow_automation.nano` - Workflow automation

**Tests**: `tests/test_beads_module.nano`

---

### process.nano
**Command Execution**

Execute shell commands and capture output from NanoLang.

```nano
from "stdlib/process.nano" import exec_command, CommandResult

let result: CommandResult = (exec_command "ls -la")
(println result.stdout)
```

**Status**: Requires C FFI implementation

---

